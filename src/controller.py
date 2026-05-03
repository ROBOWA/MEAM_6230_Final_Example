"""
Passive DS Impedance Controller (no tank energy).

Control law (Cartesian torque):
    τ = J_lin^T · D · (v_d − v_ee) + J_rot^T · d_ori · (ω_d − ω_ee) + g(q)

where:
  J_lin — 3×7 linear Jacobian at the EE site
  J_rot — 3×7 rotational Jacobian at the EE site
  D     — 3×3 positive-definite damping matrix (anisotropic: d_n in normal, d_t tangential)
  v_d   — desired EE linear velocity from DS + depth spring + force modulation
  v_ee  — actual EE linear velocity = J_lin @ q̇
  ω_d   — desired angular velocity from surface-normal alignment (sigma-weighted)
  ω_ee  — actual EE angular velocity = J_rot @ q̇
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
        k_ori=5.0,           # orientation alignment gain [1/s]
        d_ori=10.0,           # orientation damping [N·m·s/rad]
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
        self.dt = env.model.opt.timestep
        self._n_joints = env.N_JOINTS

        # Ramp state — counts time in contact to linearly scale F_preload→F_d
        self._contact_ramp_t = 0.0

        # Filter state — raw MuJoCo contact force low-pass
        self.normal_force_filt = 0.0
        self.force_filter_alpha = 0.1   # EMA coefficient (smaller = smoother)

        # Torque limits
        self._tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

        # Debug output from last step (read after ctrl.step())
        self.last_ori_err_norm = 0.0
        self.last_omega_ff = np.zeros(3)
        self.last_omega_d = np.zeros(3)

    def reset(self):
        self.env.data.ctrl[:self._n_joints] = 0.0
        self._contact_ramp_t = 0.0
        self.normal_force_filt = 0.0
        self.last_ori_err_norm = 0.0
        self.last_omega_ff = np.zeros(3)
        self.last_omega_d = np.zeros(3)

    # ------------------------------------------------------------------
    def step(self):
        """Compute and apply one torque-control step."""
        ee_pos = self.env.ee_pos()
        q = self.env.qpos()
        qd = self.env.qvel()
        J_lin, J_rot = self.env.jacobian_full()   # 3×7 each

        # ── 1. DS desired velocity ─────────────────────────────────────
        v_d, n, sigma = self.ds.compute(ee_pos)

        # ── 2. Contact surface following ───────────────────────────────
        # Normal velocity = F_des_filt / d_n  (paper-style force modulation).
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
        v_d_tan = v_d - np.dot(v_d, n) * n    # tangential DS component (always cheap)

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

        if sigma > self.sigma_close:
            v_d_n_cmd = F_des_normal / self.d_n
            # Force feedback correction: when k_force_fb > 0, adjust pressing
            # velocity to compensate for the measured force error.  This corrects
            # the extra contact force injected by d_t coupling through J^T on a
            # curved surface, letting d_t stay high (for orbit tracking) without
            # inflating the steady-state contact force.
            if self.k_force_fb > 0.0 and normal_force is not None and contact_state == "contact":
                f_err = self.F_d - normal_force
                v_d_n_cmd += self.k_force_fb * f_err
            v_d = v_d_tan + v_d_n_cmd * (-n)

        # Speed limit
        spd = np.linalg.norm(v_d)
        if spd > 0.15:
            v_d = v_d * (0.15 / spd)

        # ── 3. Damping matrix D (anisotropic) ─────────────────────────
        # Higher damping in the surface-normal direction for force regulation.
        n_dir = n
        D = self.d_t * np.eye(3) + (self.d_n - self.d_t) * np.outer(n_dir, n_dir)

        # ── 4. Current EE velocity ─────────────────────────────────────
        v_ee = J_lin @ qd

        # ── 5. Cartesian impedance force ───────────────────────────────
        F_cart = D @ (v_d - v_ee)

        # ── 6. Orientation alignment torque ────────────────────────────
        # Align tool z-axis with inward surface normal, weighted by sigma.
        R_ee = self.env.ee_rot()
        z_tool = R_ee[:, 2]          # current tool z-axis in world frame
        z_des = -n                   # desired: point into surface

        ori_err = np.cross(z_tool, z_des)   # axis-angle error, |err| ≈ sin(θ)

        # Feedforward: predict how the surface normal rotates as the tool moves.
        # For a sphere, ṅ ≈ v_tan / R, so ω_ff = n × ṅ = n × v_tan / R.
        v_tan = v_d - np.dot(v_d, n) * n
        R_eff = self.sphere.R + self.ds.r_tool   # EE center orbits at R + r_tool
        omega_ff = np.cross(n, v_tan) / R_eff

        # Reduce feedback gain in contact to avoid disturbing the measured force.
        if contact_state == "contact":
            ori_weight = 0.3 * sigma
        else:
            ori_weight = sigma

        omega_d = omega_ff + self.k_ori * ori_weight * ori_err
        # omega_d = self.k_ori * ori_weight * ori_err
        omega_ee = J_rot @ qd
        T_ori = self.d_ori * (omega_d - omega_ee)

        self.last_ori_err_norm = float(np.linalg.norm(ori_err))
        self.last_omega_ff = omega_ff.copy()
        self.last_omega_d = omega_d.copy()

        # ── 7. Map to joint torques ────────────────────────────────────
        tau = J_lin.T @ F_cart + J_rot.T @ T_ori

        # ── 8. Gravity + Coriolis + joint-damping compensation ──────────
        # qfrc_bias = gravity + Coriolis only; qfrc_passive (joint damping)
        # is NOT included and must be cancelled so it doesn't stall the orbit.
        tau += self.env.data.qfrc_bias[:self._n_joints]
        tau -= self.env.data.qfrc_passive[:self._n_joints]

        # ── 9. Null-space: joint stiffness/damping toward q_ref ───────
        J_pinv = self._dls_pinv(J_lin)
        N = np.eye(self._n_joints) - J_pinv @ J_lin
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
