"""
Passive DS Impedance Controller (no tank energy).

Control law (Cartesian torque):
    τ = J^T · D · (v_d − v_ee) + g(q)

where:
  J     — 3×7 linear Jacobian at the EE site
  D     — 3×3 positive-definite damping matrix, constructed from the nominal DS direction:
              D = d_n I + (d_t - d_n) e1 e1^T
          where e1 = v_nom / ||v_nom||.
          d_t is the high tracking damping along the nominal DS direction.
          d_n is the damping in the perpendicular subspace, which includes the surface
          normal when the nominal DS is tangential to the surface near contact.
  v_d   — desired EE velocity: nominal DS velocity plus a force modulation term
              v_d = v_nom + f_force
          where f_force = -(F_des_normal / d_n) n (near/contact only).
          Because the nominal DS is tangential near contact, n lies in the perpendicular
          eigenspace with eigenvalue d_n, so the steady-state contact force is F_des_normal.
  v_ee  — actual EE linear velocity = J @ q̇
  g(q)  — gravity + Coriolis (from data.qfrc_bias)

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
        F_d=5.0,
        d_n=400.0,       # normal damping [N·s/m]
        d_t=600.0,       # tangential damping [N·s/m]
        k_null=5.0,      # null-space joint stiffness [N·m/rad]
        b_null=5.0,      # null-space joint damping [N·m·s/rad]
        dls_lambda=0.02,
        F_preload=1,       # approach preload force [N]
        force_tol=3.0,       # measured force threshold for contact detection [N]
        sigma_close=0.9,    # sigma above which force modulation begins
        sigma_contact=0.98,  # sigma above which full polishing force is used (fallback)
        use_force_feedback=False,  # use env.contact_normal_force() for contact detection
        force_ramp_time=2.0,    # time [s] to linearly ramp F_preload → F_d after contact
        k_force_fb=0.0,      # force-feedback gain [m/(s·N)]: adjusts v_d_n to correct
                             # measured-force error; compensates d_t coupling artefact
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
        self.dt = env.model.opt.timestep
        self._n_joints = env.N_JOINTS

        # Ramp state — counts time in contact to linearly scale F_preload→F_d
        self._contact_ramp_t = 0.0

        # Persistent eigenvector for continuity of D across steps
        self._last_e1 = None

        # Filter state — raw MuJoCo contact force low-pass
        self.normal_force_filt = 0.0
        self.force_filter_alpha = 0.1   # EMA coefficient (smaller = smoother)

        # Torque limits
        self._tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

    def reset(self):
        self.env.data.ctrl[:self._n_joints] = 0.0
        self._contact_ramp_t = 0.0
        self.normal_force_filt = 0.0
        self._last_e1 = None

    # ------------------------------------------------------------------
    def step(self):
        """Compute and apply one torque-control step."""
        ee_pos = self.env.ee_pos()
        q = self.env.qpos()
        qd = self.env.qvel()
        J = self.env.jacobian()          # 3×7

        # ── 1. DS desired velocity ─────────────────────────────────────
        v_d, n, sigma = self.ds.compute(ee_pos)
        v_nom = v_d.copy()   # nominal DS velocity before force modulation

        # ── 2. Contact surface following ───────────────────────────────
        # Force modulation adds an inward normal velocity component so that
        # D · (v_d - v_ee) generates the desired contact force.
        # Three states: far (DS only), preload (approaching), contact (full F_d).

        # 2a. Filter raw contact force measurement (EMA to reduce MuJoCo noise).
        raw_force = self._estimate_normal_force()
        if raw_force is not None:
            self.normal_force_filt = (
                self.force_filter_alpha * raw_force
                + (1.0 - self.force_filter_alpha) * self.normal_force_filt
            )
            normal_force = self.normal_force_filt
        else:
            normal_force = None

        # 2b. Determine contact state and raw desired normal force.
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

        # 2c. Add force modulation on top of nominal DS (Scheme C).
        # Near/contact: inject an inward normal velocity component so that
        # D generates the desired contact force.  The nominal DS is tangential
        # near contact, so n lies in the perpendicular eigenspace (eigenvalue d_n)
        # and f_force = v_d_n_cmd * (-n) produces F_des_normal in steady state.
        if sigma > self.sigma_close:
            v_d_n_cmd = F_des_normal / self.d_n

            if self.k_force_fb > 0.0 and normal_force is not None and contact_state == "contact":
                f_err = self.F_d - normal_force
                v_d_n_cmd += self.k_force_fb * f_err

            f_force = v_d_n_cmd * (-n)
            v_d = v_nom + f_force

        # Speed limit
        spd = np.linalg.norm(v_d)
        if spd > 0.15:
            v_d = v_d * (0.15 / spd)

        # ── 3. Damping matrix D (nominal-DS frame) ─────────────────────
        # D = d_n I + (d_t - d_n) e1 e1^T
        # e1 tracks the nominal DS direction for continuity across steps.
        eps = 1e-9
        norm_nom = np.linalg.norm(v_nom)

        if norm_nom > eps:
            e1 = v_nom / norm_nom
            if self._last_e1 is not None and np.dot(e1, self._last_e1) < 0.0:
                e1 = -e1
        else:
            if self._last_e1 is not None:
                e1 = self._last_e1
            else:
                e1 = -n

        self._last_e1 = e1

        D = self.d_n * np.eye(3) + (self.d_t - self.d_n) * np.outer(e1, e1)

        # ── 4. Current EE velocity ─────────────────────────────────────
        v_ee = J @ qd

        # ── 5. Cartesian impedance force ───────────────────────────────
        F_cart = D @ (v_d - v_ee)

        # ── 6. Map to joint torques ────────────────────────────────────
        tau = J.T @ F_cart

        # ── 7. Gravity + Coriolis + joint-damping compensation ──────────
        # qfrc_bias = gravity + Coriolis only; qfrc_passive (joint damping)
        # is NOT included and must be cancelled so it doesn't stall the orbit.
        tau += self.env.data.qfrc_bias[:self._n_joints]
        tau -= self.env.data.qfrc_passive[:self._n_joints]

        # ── 8. Null-space: joint stiffness/damping toward q_ref ───────
        J_pinv = self._dls_pinv(J)
        N = np.eye(self._n_joints) - J_pinv @ J
        tau_null = N @ (self.k_null * (self.q_ref - q) - self.b_null * qd)
        tau += tau_null

        # ── 9. Clip and apply ──────────────────────────────────────────
        tau = np.clip(tau, -self._tau_max, self._tau_max)
        self.env.data.ctrl[:self._n_joints] = tau

        return v_d, n, sigma, contact_state, F_des_normal, normal_force

    # ------------------------------------------------------------------
    def _estimate_normal_force(self):
        """Return measured normal contact force [N], always available.

        Returns the held/decayed contact force so the force-feedback path
        (k_force_fb > 0) and the contact-detection path (use_force_feedback)
        both have a non-None reading.  The sigma-based fallback is used for
        contact state detection when use_force_feedback is False.
        """
        return self.env.contact_normal_force()

    # ------------------------------------------------------------------
    def _dls_pinv(self, J):
        lam2 = self.dls_lambda ** 2
        return J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(3))
