"""
Passive DS Impedance Controller — strict paper formulation with energy tank.

Control law (Cartesian torque):
    τ = J^T · D · (v_d − v_ee) + g(q)

DS decomposition (Amanhoud RSS 2019, App. B):
    f_c = 0          (polishing orbit is fully non-conservative)
    f_r = f_motion   (orbit DS component)
    f_n = f_force    (normal force modulation = −(F_des/λ₁)·n)

Energy tank (state s, energy T = s):
    p_r = λ₁·(v_ee·f_r)      power drawn by orbit component
    p_n = λ₁·(v_ee·f_n)      power drawn by force component
    p_d = v_ee^T D v_ee ≥ 0  power dissipated by damping  → charges tank

    β_i   = 0 if (s≤0 and p_i>ε) or (s≥s_max and p_i<−ε), else 1
    β_i'  = 1 if p_i < −ε,  else β_i           (always allow back-injection)

    v_d   = f_c + β_r'·f_r + β_n'·f_n          (gated desired velocity)

    α     = υ⁻(s, s_max−Δs, s_max)             (stop charging when full)
    ṡ     = α·p_d − β_r·p_r − β_n·p_n
    s     = clip(s + dt·ṡ, 0, s_max)

Paper-style damping matrix (built from gated v_d):
    e1 = v_d/‖v_d‖,  E = [e1, e2, e3]
    D  = E · diag(λ₁, λ⊥, λ⊥) · Eᵀ

λ₁  is the damping along v_d.
λ⊥  is the damping in the plane perpendicular to v_d.

Additionally a joint-space null-space stiffness prevents drift to joint limits.
"""

import numpy as np


class PolishingController:
    def __init__(
        self,
        env,
        sphere,
        ds,
        q_ref,
        F_d=15.0,
        lambda_1=None,       # damping along final desired DS direction [N·s/m]
        lambda_perp=None,    # damping perpendicular to DS direction [N·s/m]
        d_n=400.0,           # backward-compat alias for lambda_1
        d_t=600.0,           # backward-compat alias for lambda_perp
        k_null=5.0,          # null-space joint stiffness [N·m/rad]
        b_null=5.0,          # null-space joint damping [N·m·s/rad]
        dls_lambda=0.02,
        F_preload=1,         # approach preload force [N]
        force_tol=3.0,       # measured force threshold for contact detection [N]
        sigma_close=0.95,     # sigma above which force modulation begins
        sigma_contact=0.98,  # sigma above which full polishing force is used (fallback)
        use_force_feedback=False,
        force_ramp_time=2.0, # time [s] to ramp F_preload → F_d after contact
        k_force_fb=0.0,
        # ── Energy tank (Amanhoud RSS 2019 App. B) ────────────────────
        use_energy_tank=True,
        tank_s_max=60.0,     # maximum tank level [J]
        tank_delta_ratio=0.1,# width of the upper saturation ramp as fraction of s_max
        tank_init="full",    # initial tank level: "full", "empty", "half", or float
        power_eps=1e-8,      # threshold for power sign detection [W]
    ):
        self.env = env
        self.sphere = sphere
        self.ds = ds
        self.q_ref = np.asarray(q_ref, dtype=float)
        self.F_d = F_d

        # Paper-style damping eigenvalues (backward-compat: fall back to d_n, d_t).
        if lambda_1 is None:
            lambda_1 = d_t
        if lambda_perp is None:
            lambda_perp = d_n
        self.lambda_1 = lambda_1
        self.lambda_perp = lambda_perp
        # Mirror onto old names so external code reading d_n/d_t still works.
        self.d_n = self.lambda_1
        self.d_t = self.lambda_perp

        self.k_null = k_null
        self.b_null = b_null
        self.dls_lambda = dls_lambda
        self.F_preload = F_preload
        self.force_tol = force_tol
        self.sigma_close = sigma_close
        self.sigma_contact = sigma_contact
        self.use_force_feedback = use_force_feedback
        self.k_force_fb = k_force_fb
        self.force_ramp_time = force_ramp_time
        self.dt = env.model.opt.timestep
        self._n_joints = env.N_JOINTS

        # Ramp state — counts time in contact to linearly scale F_preload→F_d.
        self._contact_ramp_t = 0.0

        # Filter state — raw MuJoCo contact force low-pass.
        self.normal_force_filt = 0.0
        self.force_filter_alpha = 0.1

        # Torque limits.
        self._tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

        # Previous first eigenvector for zero-velocity fallback in damping construction.
        self._last_e1 = None

        # ── Energy tank ───────────────────────────────────────────────
        self.use_energy_tank = use_energy_tank
        self.tank_s_max = float(tank_s_max)
        self.tank_delta_s = tank_delta_ratio * self.tank_s_max
        self.power_eps = power_eps

        if isinstance(tank_init, str):
            _init_map = {"full": self.tank_s_max, "empty": 0.0,
                         "half": self.tank_s_max / 2.0}
            if tank_init not in _init_map:
                raise ValueError(f"tank_init must be 'full', 'empty', 'half', or float; got '{tank_init}'")
            s0 = _init_map[tank_init]
        else:
            s0 = float(tank_init)
        self.tank_s = s0
        self._tank_s_init = s0
        self.tank_s0 = s0           # alias used by visualization code
        self.tank_delta = self.tank_delta_s   # alias used by visualization code
        self.last_tank_info = {}

    # ------------------------------------------------------------------
    def reset(self):
        self.env.data.ctrl[:self._n_joints] = 0.0
        self._contact_ramp_t = 0.0
        self.normal_force_filt = 0.0
        self._last_e1 = None
        self.tank_s = self._tank_s_init
        self.last_tank_info = {}

    # ------------------------------------------------------------------
    def step(self):
        """Compute and apply one torque-control step."""
        ee_pos = self.env.ee_pos()
        q = self.env.qpos()
        qd = self.env.qvel()
        J = self.env.jacobian()          # 3×7

        # ── 1. Nominal motion from DS ──────────────────────────────────
        f_motion, n, sigma = self.ds.compute(ee_pos)

        # ── 2. Filter raw contact force (EMA to reduce MuJoCo noise) ──
        raw_force = self._estimate_normal_force()
        if raw_force is not None:
            self.normal_force_filt = (
                self.force_filter_alpha * raw_force
                + (1.0 - self.force_filter_alpha) * self.normal_force_filt
            )
            normal_force = self.normal_force_filt
        else:
            normal_force = None

        # ── 3. Determine contact state and desired normal force ────────
        if sigma <= self.sigma_close:
            contact_state = "far"
            F_des_normal = 0.0
            self._contact_ramp_t = 0.0
        else:
            if self.use_force_feedback and normal_force is not None:
                contact = normal_force >= self.force_tol
            else:
                contact = sigma >= self.sigma_contact

            if contact:
                contact_state = "contact"
                # Linear ramp from F_preload to F_d over force_ramp_time seconds.
                self._contact_ramp_t = min(self._contact_ramp_t + self.dt, self.force_ramp_time)
                progress = self._contact_ramp_t / self.force_ramp_time
                F_des_normal = self.F_preload + (self.F_d - self.F_preload) * progress
            else:
                contact_state = "preload"
                F_des_normal = self.F_preload
                self._contact_ramp_t = 0.0

        force_active = sigma > self.sigma_close

        # ── 4. DS decomposition (Amanhoud RSS 2019 App. B) ────────────
        # f_c = 0: the polishing orbit DS is fully non-conservative, so
        #          there is no conservative component.
        # f_r = f_motion: orbit motion (tangential, speed-limited).
        # f_n = f_force:  surface-normal force modulation.
        #
        # Force term uses λ₁ (the damping eigenvalue along v_d) per the
        # paper identity D·v_d = λ₁·v_d when e1 ∥ v_d.  The power
        # computation pn = λ₁·(v_ee·f_n) uses the same λ₁ consistently.

        # Speed-limit orbit component before gating.
        spd = np.linalg.norm(f_motion)
        if spd > 0.15:
            f_motion = f_motion * (0.15 / spd)

        f_c = np.zeros(3)
        f_r = f_motion
        f_n = -(F_des_normal / self.lambda_1) * n if force_active else np.zeros(3)

        # ── 5. Current EE Cartesian velocity ───────────────────────────
        v_ee = J @ qd

        # ── 6–11. Energy tank gating ───────────────────────────────────
        if self.use_energy_tank:
            # Power drawn from tank by each DS component
            # (positive p_i = energy flows from tank to robot via that component).
            pr = self.lambda_1 * np.dot(v_ee, f_r)
            pn = self.lambda_1 * np.dot(v_ee, f_n)

            # β_i: gate that prevents the tank from draining below 0 or
            #       filling above s_max.
            def _beta(pi):
                if self.tank_s <= 0.0 and pi > self.power_eps:
                    return 0.0   # tank empty; block energy withdrawal
                if self.tank_s >= self.tank_s_max and pi < -self.power_eps:
                    return 0.0   # tank full; block further injection
                return 1.0

            # β_i': always allow a component that injects energy back into
            #        the tank (braking/returning phase); otherwise use β_i.
            def _beta_prime(pi, bi):
                return 1.0 if pi < -self.power_eps else bi

            beta_r = _beta(pr)
            beta_n = _beta(pn)
            beta_r_prime = _beta_prime(pr, beta_r)
            beta_n_prime = _beta_prime(pn, beta_n)

            # Gated desired velocity (step 8 per spec).
            v_d = f_c + beta_r_prime * f_r + beta_n_prime * f_n

            # Build D from gated v_d (step 9).
            D, _ = self._construct_passive_ds_damping(v_d, n, force_active)

            # Dissipated power (always ≥ 0, charges the tank).
            pd = float(v_ee @ D @ v_ee)

            # α ramps to 0 as s approaches s_max to prevent overfill.
            alpha = self._upsilon_minus(
                self.tank_s,
                self.tank_s_max - self.tank_delta_s,
                self.tank_s_max,
            )

            # Euler update; clip to valid range.
            ds_tank = self.dt * (alpha * pd - beta_r * pr - beta_n * pn)
            self.tank_s = float(np.clip(self.tank_s + ds_tank, 0.0, self.tank_s_max))

            _motion_vec = (beta_r_prime * f_r).copy()
            _force_vec  = (beta_n_prime * f_n).copy()
            self.last_tank_info = {
                # paper-style keys (kept for backward compat)
                "pr":           pr,  "pn":  pn,  "pd":  pd,
                "beta_r":       beta_r,   "beta_r_prime": beta_r_prime,
                "ds_tank":      ds_tank,
                # standard keys matching 6230_tangent convention
                "tank_s":       self.tank_s,
                "tank_s_max":   self.tank_s_max,
                "tank_s_dot":   ds_tank / max(self.dt, 1e-9),
                "alpha":        alpha,
                "beta_t":       beta_r,   "beta_t_prime": beta_r_prime,
                "beta_n":       beta_n,   "beta_n_prime": beta_n_prime,
                "p_t":          pr,  "p_n": pn,  "p_d": pd,
                "motion_vec":   _motion_vec,
                "force_vec":    _force_vec,
                "vd_vec":       v_d.copy(),
                "motion_norm":  float(np.linalg.norm(_motion_vec)),
                "force_norm":   float(np.linalg.norm(_force_vec)),
                "vd_norm":      float(np.linalg.norm(v_d)),
                "F_des_normal": F_des_normal,
                "contact_state": contact_state,
                "normal_force": normal_force,
            }

        else:
            # Tank disabled: pass all components through unmodified.
            v_d = f_c + f_r + f_n
            D, _ = self._construct_passive_ds_damping(v_d, n, force_active)
            self.last_tank_info = {
                "tank_s":       self.tank_s,
                "tank_s_max":   self.tank_s_max,
                "alpha":        1.0,
                "beta_t":       1.0,  "beta_n":       1.0,
                "beta_t_prime": 1.0,  "beta_n_prime": 1.0,
                "p_t":          0.0,  "p_n": 0.0,  "p_d": 0.0,
                "motion_vec":   f_r.copy(),
                "force_vec":    f_n.copy(),
                "vd_vec":       v_d.copy(),
                "motion_norm":  float(np.linalg.norm(f_r)),
                "force_norm":   float(np.linalg.norm(f_n)),
                "vd_norm":      float(np.linalg.norm(v_d)),
                "F_des_normal": F_des_normal,
                "contact_state": contact_state,
                "normal_force": normal_force,
            }

        # ── 12. Cartesian impedance force ──────────────────────────────
        F_cart = D @ (v_d - v_ee)

        # ── 13. Map to joint torques ───────────────────────────────────
        tau = J.T @ F_cart

        # ── 14. Gravity + Coriolis + joint-damping compensation ────────
        # qfrc_bias = gravity + Coriolis; qfrc_passive (joint damping)
        # is cancelled so it does not stall the orbit.
        tau += self.env.data.qfrc_bias[:self._n_joints]
        tau -= self.env.data.qfrc_passive[:self._n_joints]

        # ── 15. Null-space: joint stiffness/damping toward q_ref ───────
        J_pinv = self._dls_pinv(J)
        N = np.eye(self._n_joints) - J_pinv @ J
        tau_null = N @ (self.k_null * (self.q_ref - q) - self.b_null * qd)
        tau += tau_null

        # ── 16. Clip and apply ─────────────────────────────────────────
        tau = np.clip(tau, -self._tau_max, self._tau_max)
        self.env.data.ctrl[:self._n_joints] = tau

        return v_d, n, sigma, contact_state, F_des_normal, normal_force

    # ------------------------------------------------------------------
    def _construct_passive_ds_damping(self, v_d, n, force_active=False):
        """Build paper-style damping matrix with first eigenvector e1 = v_d / ‖v_d‖.

        λ₁  — damping along v_d (the final desired DS direction)
        λ⊥  — damping in the plane perpendicular to v_d
        """
        eps = 1e-9
        norm_vd = np.linalg.norm(v_d)

        if norm_vd < eps:
            if force_active:
                e1 = -n
            elif self._last_e1 is not None:
                e1 = self._last_e1
            else:
                e1 = np.array([1.0, 0.0, 0.0])
        else:
            e1 = v_d / norm_vd

        # Coordinate axis least aligned with e1 → Gram-Schmidt for e2, e3.
        axes = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]
        tmp = min(axes, key=lambda a: abs(np.dot(a, e1)))

        e2 = tmp - np.dot(tmp, e1) * e1
        e2 = e2 / np.linalg.norm(e2)
        e3 = np.cross(e1, e2)

        E = np.column_stack([e1, e2, e3])
        Lambda = np.diag([self.lambda_1, self.lambda_perp, self.lambda_perp])
        D = E @ Lambda @ E.T

        self._last_e1 = e1
        return D, E

    # ------------------------------------------------------------------
    @staticmethod
    def _upsilon_minus(x, x_low, x_high):
        """Linear ramp from 1 (x ≤ x_low) down to 0 (x ≥ x_high).

        Used to taper off the tank-charging coefficient α as the tank
        approaches its maximum level, preventing overfill.
        """
        if x <= x_low:
            return 1.0
        if x >= x_high:
            return 0.0
        return (x_high - x) / (x_high - x_low)

    # ------------------------------------------------------------------
    def _estimate_normal_force(self):
        """Return measured normal contact force [N]."""
        return self.env.contact_normal_force()

    # ------------------------------------------------------------------
    def _dls_pinv(self, J):
        lam2 = self.dls_lambda ** 2
        return J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(3))
