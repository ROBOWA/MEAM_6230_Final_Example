"""
Surface-Frame Passive DS Impedance Controller with Energy Tank + Orientation Tracking.

Control law (Cartesian torque):
    τ = J_lin^T · D_s · (v_d − v_ee) + J_rot^T · d_ori · (ω_d − ω_ee) + g(q)

D_s is a 3×3 surface-frame damping matrix with eigenvalues along
e_track (tangential tracking), e_side (tangential sideways), and e_n (inward normal).

The energy tank gates the potentially-active tangential motion term (motion_term)
and normal force-injection term (force_term).  Dissipative terms are always allowed.

The orientation term aligns the tool z-axis with the inward surface normal,
weighted by sigma and contact state.
"""

import numpy as np


class PolishingController:
    def __init__(
        self,
        env,
        sphere,
        ds,
        q_ref,
        F_d=5.0,
        d_n=400.0,       # normal damping [N·s/m]
        d_t=600.0,       # tangential tracking damping [N·s/m]
        k_null=5.0,      # null-space joint stiffness [N·m/rad]
        b_null=5.0,      # null-space joint damping [N·m·s/rad]
        dls_lambda=0.02,
        F_preload=1.0,
        force_tol=3.0,
        sigma_close=0.9,
        sigma_contact=0.98,
        use_force_feedback=False,
        force_ramp_time=2.0,
        k_force_fb=0.0,
        k_ori=5.0,           # orientation alignment gain [1/s]
        d_ori=10.0,          # orientation damping [N·m·s/rad]
        # Energy tank
        use_energy_tank=False,
        tank_s0=1.0,
        tank_s_max=5.0,
        tank_delta=0.5,
        tank_eps=1e-6,
        d_side=None,         # sideways tangential damping; defaults to d_n
        v_max=0.15,
    ):
        self.env = env
        self.sphere = sphere
        self.ds = ds
        self.q_ref = np.asarray(q_ref, dtype=float)
        self.F_d = F_d
        self.d_n = d_n
        self.d_t = d_t
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
        self.k_ori = k_ori
        self.d_ori = d_ori
        self.v_max = v_max

        # Energy tank
        self.use_energy_tank = use_energy_tank
        self.tank_s0 = float(tank_s0)
        self.tank_s_max = float(tank_s_max)
        self.tank_delta = float(tank_delta)
        self.tank_eps = float(tank_eps)
        self.d_side = float(d_side) if d_side is not None else float(d_n)
        self.tank_s = self.tank_s0
        self.last_tank_info = {}

        self.dt = env.model.opt.timestep
        self._n_joints = env.N_JOINTS

        self._contact_ramp_t = 0.0
        self._last_e_track = None

        self.normal_force_filt = 0.0
        self.force_filter_alpha = 0.1

        self._tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

        # Orientation debug (readable after step())
        self.last_ori_err_norm = 0.0
        self.last_omega_ff = np.zeros(3)
        self.last_omega_d = np.zeros(3)

    # ------------------------------------------------------------------
    def reset(self):
        self.env.data.ctrl[:self._n_joints] = 0.0
        self._contact_ramp_t = 0.0
        self.normal_force_filt = 0.0
        self._last_e_track = None
        self.tank_s = self.tank_s0
        self.last_tank_info = {}
        self.last_ori_err_norm = 0.0
        self.last_omega_ff = np.zeros(3)
        self.last_omega_d = np.zeros(3)

    # ------------------------------------------------------------------
    def _tank_alpha(self):
        """Smooth shutoff of dissipation-to-tank flow as tank approaches s_max."""
        a = self.tank_s_max - self.tank_delta
        b = self.tank_s_max
        s = self.tank_s
        if self.tank_delta <= 0:
            return 1.0 if s < self.tank_s_max else 0.0
        if s < a:
            return 1.0
        if s > b:
            return 0.0
        xi = (s - a) / (b - a)
        return 0.5 * (1.0 + np.cos(np.pi * xi))

    def _tank_beta(self, p):
        """Gate active power injection when tank is empty; block overfill when full."""
        if self.tank_s <= self.tank_eps and p > 0.0:
            return 0.0
        if self.tank_s >= self.tank_s_max - self.tank_eps and p < 0.0:
            return 0.0
        return 1.0

    def _tank_beta_prime(self, p, beta):
        """Dissipative action (p < 0) is always allowed regardless of tank state."""
        if p < 0.0:
            return 1.0
        return beta

    def get_tank_info(self):
        return dict(self.last_tank_info)

    # ------------------------------------------------------------------
    def step(self):
        """Compute and apply one torque-control step.

        Returns:
            v_d           : tank-corrected desired EE velocity (3,)
            n             : outward surface normal (3,)
            sigma         : contact blend weight
            contact_state : 'far' | 'preload' | 'contact'
            F_des_normal  : desired normal force [N]
            normal_force  : filtered measured normal force [N] or None
            tank_info     : dict with tank state, betas, powers, vectors
        """
        ee_pos = self.env.ee_pos()
        q = self.env.qpos()
        qd = self.env.qvel()
        J_lin, J_rot = self.env.jacobian_full()   # 3×7 each

        # ── 1. DS nominal velocity ─────────────────────────────────────
        v_nom, n, sigma = self.ds.compute(ee_pos)

        # ── 2. Contact state & force ramp ─────────────────────────────
        raw_force = self._estimate_normal_force()
        if raw_force is not None:
            self.normal_force_filt = (
                self.force_filter_alpha * raw_force
                + (1.0 - self.force_filter_alpha) * self.normal_force_filt
            )
            normal_force = self.normal_force_filt
        else:
            normal_force = None

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
                self._contact_ramp_t = min(
                    self._contact_ramp_t + self.dt, self.force_ramp_time
                )
                progress = self._contact_ramp_t / self.force_ramp_time
                F_des_normal = self.F_preload + (self.F_d - self.F_preload) * progress
            else:
                contact_state = "preload"
                F_des_normal = self.F_preload
                self._contact_ramp_t = 0.0

        # ── 3. Surface-frame basis ─────────────────────────────────────
        eps = 1e-6
        P_tan = np.eye(3) - np.outer(n, n)
        f_t = P_tan @ v_nom   # tangential projection of nominal DS

        ft_norm = np.linalg.norm(f_t)
        if ft_norm > eps:
            e_track = f_t / ft_norm
        elif self._last_e_track is not None:
            e_track = self._last_e_track
        else:
            t1, _, _ = self.sphere.tangent_frame(ee_pos)
            e_track = t1

        e_n = -n   # inward normal (contact/force direction)

        e_side = np.cross(e_n, e_track)
        e_side_norm = np.linalg.norm(e_side)
        if e_side_norm > eps:
            e_side = e_side / e_side_norm
        else:
            t1, _, _ = self.sphere.tangent_frame(ee_pos)
            e_side = np.cross(e_n, t1)
            side_norm = np.linalg.norm(e_side)
            e_side = e_side / side_norm if side_norm > eps else np.array([0.0, 1.0, 0.0])

        # Re-orthogonalize e_track against e_side and e_n
        e_track = np.cross(e_side, e_n)
        e_track_norm = np.linalg.norm(e_track)
        if e_track_norm > eps:
            e_track = e_track / e_track_norm

        self._last_e_track = e_track.copy()

        # ── 4. Surface-frame damping matrix ───────────────────────────
        E = np.column_stack([e_track, e_side, e_n])
        Lambda = np.diag([self.d_t, self.d_side, self.d_n])
        D = E @ Lambda @ E.T
        D = 0.5 * (D + D.T)   # enforce exact symmetry

        # ── 5. Motion and force terms ──────────────────────────────────
        # Far from surface: use full v_nom for approach.
        # Near/contact: tangential projection + normal force injection.
        if sigma <= self.sigma_close:
            motion_term = v_nom.copy()
            force_term = np.zeros(3)
        else:
            motion_term = f_t.copy()
            v_n_cmd = F_des_normal / self.d_n
            if (
                self.k_force_fb > 0.0
                and normal_force is not None
                and contact_state == "contact"
            ):
                v_n_cmd += self.k_force_fb * (self.F_d - normal_force)
            force_term = v_n_cmd * e_n   # inward = toward surface

        # ── 6. Velocity limit before tank accounting ───────────────────
        v_raw = motion_term + force_term
        raw_spd = np.linalg.norm(v_raw)
        if raw_spd > self.v_max:
            scale = self.v_max / raw_spd
            motion_term = motion_term * scale
            force_term = force_term * scale

        # ── 7. Current EE velocity ─────────────────────────────────────
        v_ee = J_lin @ qd

        # ── 8. Tank power terms ────────────────────────────────────────
        Dm = D @ motion_term
        Df = D @ force_term
        Dv = D @ v_ee
        p_t = float(v_ee @ Dm)
        p_n = float(v_ee @ Df)
        p_d = max(float(v_ee @ Dv), 0.0)   # D is PSD; clamp numerical noise

        # ── 9. Energy tank correction ──────────────────────────────────
        if self.use_energy_tank:
            alpha = self._tank_alpha()
            beta_t = self._tank_beta(p_t)
            beta_n = self._tank_beta(p_n)
            beta_t_prime = self._tank_beta_prime(p_t, beta_t)
            beta_n_prime = self._tank_beta_prime(p_n, beta_n)

            v_d = beta_t_prime * motion_term + beta_n_prime * force_term

            s_dot = alpha * p_d - beta_t * p_t - beta_n * p_n
            self.tank_s = np.clip(
                self.tank_s + self.dt * s_dot, 0.0, self.tank_s_max
            )
        else:
            alpha = 1.0
            beta_t = beta_n = beta_t_prime = beta_n_prime = 1.0
            s_dot = 0.0
            v_d = motion_term + force_term

        # ── 10. Cartesian impedance force ──────────────────────────────
        F_cart = D @ (v_d - v_ee)

        # ── 11. Orientation alignment torque ───────────────────────────
        # Align tool z-axis with inward surface normal, weighted by sigma.
        R_ee = self.env.ee_rot()
        z_tool = R_ee[:, 2]     # current tool z-axis in world frame
        z_des = -n              # desired: point into surface

        ori_err = np.cross(z_tool, z_des)   # axis-angle error, |err| ≈ sin(θ)

        # Feedforward: predict surface-normal rotation due to tangential motion.
        # For a sphere, ṅ ≈ v_tan / R, so ω_ff = n × ṅ = n × v_tan / R.
        v_tan = v_d - np.dot(v_d, n) * n
        R_eff = self.sphere.R + self.ds.r_tool
        omega_ff = np.cross(n, v_tan) / R_eff

        # Reduce feedback gain in contact to avoid disturbing the measured force.
        if contact_state == "contact":
            ori_weight = 0.3 * sigma
        else:
            ori_weight = sigma

        omega_d = self.k_ori * ori_weight * ori_err
        omega_ee = J_rot @ qd
        T_ori = self.d_ori * (omega_d - omega_ee)

        self.last_ori_err_norm = float(np.linalg.norm(ori_err))
        self.last_omega_ff = omega_ff.copy()
        self.last_omega_d = omega_d.copy()

        # ── 12. Map to joint torques ────────────────────────────────────
        tau = J_lin.T @ F_cart + J_rot.T @ T_ori

        # ── 13. Gravity + Coriolis + joint-damping compensation ─────────
        tau += self.env.data.qfrc_bias[:self._n_joints]
        tau -= self.env.data.qfrc_passive[:self._n_joints]

        # ── 14. Null-space: joint stiffness/damping toward q_ref ───────
        J_pinv = self._dls_pinv(J_lin)
        N = np.eye(self._n_joints) - J_pinv @ J_lin
        tau_null = N @ (self.k_null * (self.q_ref - q) - self.b_null * qd)
        tau += tau_null

        # ── 15. Clip and apply ──────────────────────────────────────────
        tau = np.clip(tau, -self._tau_max, self._tau_max)
        self.env.data.ctrl[:self._n_joints] = tau

        # ── 16. Log ────────────────────────────────────────────────────
        self.last_tank_info = {
            "tank_s": self.tank_s,
            "tank_s_dot": s_dot,
            "tank_s_max": self.tank_s_max,
            "alpha": alpha,
            "beta_t": beta_t,
            "beta_n": beta_n,
            "beta_t_prime": beta_t_prime,
            "beta_n_prime": beta_n_prime,
            "p_t": p_t,
            "p_n": p_n,
            "p_d": p_d,
            "motion_norm": float(np.linalg.norm(motion_term)),
            "force_norm": float(np.linalg.norm(force_term)),
            "vd_norm": float(np.linalg.norm(v_d)),
            "contact_state": contact_state,
            "F_des_normal": F_des_normal,
            "normal_force": normal_force,
            "motion_vec": motion_term.copy(),
            "force_vec": force_term.copy(),
            "vd_vec": v_d.copy(),
        }

        return v_d, n, sigma, contact_state, F_des_normal, normal_force, self.last_tank_info

    # ------------------------------------------------------------------
    def _estimate_normal_force(self):
        return self.env.contact_normal_force()

    def _dls_pinv(self, J):
        lam2 = self.dls_lambda ** 2
        return J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(3))
