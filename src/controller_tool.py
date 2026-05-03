"""
Passive DS Impedance Controller — cylindrical polishing tool variant.

Identical to controller.py except for force sensing:
  - _estimate_normal_force(n) receives the surface normal and delegates to
    env.contact_normal_force(n), which projects the FT sensor reading onto n.
  - use_force_feedback defaults to True (FT sensor is reliable).

Control law (unchanged):
    τ = J_lin^T · D · (v_d − v_ee) + J_rot^T · d_ori · (ω_d − ω_ee) + g(q)
"""

import numpy as np


class PolishingControllerTool:
    def __init__(
        self,
        env,
        sphere,
        ds,
        q_ref,
        F_d=5.0,
        d_n=400.0,
        d_t=600.0,
        k_null=5.0,
        b_null=5.0,
        dls_lambda=0.02,
        F_preload=1.0,
        force_tol=1.0,           # lower threshold — FT is noise-free
        sigma_close=0.9,
        sigma_contact=0.98,
        use_force_feedback=True,  # FT sensor is reliable → enable by default
        force_ramp_time=2.0,
        k_force_fb=0.0,
        k_ori=2.0,
        d_ori=5.0,
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
        # sigma_close (0.9): below this → "far", no force control applied.
        # ds.sigma_close (0.98): DS gives pure-tangential velocity above this.
        # Intentionally different: controller starts preloading before DS
        # fully hands off the normal direction.
        self.sigma_close = sigma_close
        self.sigma_contact = sigma_contact
        self.use_force_feedback = use_force_feedback
        self.k_force_fb = k_force_fb
        self.force_ramp_time = force_ramp_time
        self.k_ori = k_ori
        self.d_ori = d_ori
        self.dt = env.model.opt.timestep
        self._n_joints = env.N_JOINTS

        self._contact_ramp_t = 0.0

        # Light EMA to smooth FT sensor noise
        self.normal_force_filt = 0.0
        self.force_filter_alpha = 0.2

        self._tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

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
        # Capture FT zero offset at current (non-contact) configuration.
        self.env.set_ft_bias()

    # ------------------------------------------------------------------
    def step(self):
        """Compute and apply one torque-control step."""
        ee_pos = self.env.ee_pos()
        q  = self.env.qpos()
        qd = self.env.qvel()
        J_lin, J_rot = self.env.jacobian_full()

        # ── 1. DS desired velocity ─────────────────────────────────────
        v_d, n, sigma = self.ds.compute(ee_pos)

        # ── 2. Contact surface following ───────────────────────────────
        # 2a. FT-based normal force with light EMA for noise smoothing.
        raw_force = self._estimate_normal_force(n)
        self.normal_force_filt = (
            self.force_filter_alpha * raw_force
            + (1.0 - self.force_filter_alpha) * self.normal_force_filt
        )
        normal_force = self.normal_force_filt

        # 2b. Determine contact state and desired normal force.
        v_d_tan = v_d - np.dot(v_d, n) * n

        if sigma <= self.sigma_close:
            contact_state = "far"
            F_des_normal = 0.0
            self._contact_ramp_t = 0.0
        else:
            if self.use_force_feedback:
                contact = normal_force >= self.force_tol
            else:
                contact = sigma >= self.sigma_contact

            if contact:
                contact_state = "contact"
                self._contact_ramp_t = min(
                    self._contact_ramp_t + self.dt, self.force_ramp_time)
                progress = self._contact_ramp_t / self.force_ramp_time
                F_des_normal = self.F_preload + (self.F_d - self.F_preload) * progress
            else:
                contact_state = "preload"
                F_des_normal = self.F_preload
                self._contact_ramp_t = 0.0

        if sigma > self.sigma_close:
            v_d_n_cmd = F_des_normal / self.d_n
            if self.k_force_fb > 0.0 and contact_state == "contact":
                f_err = self.F_d - normal_force
                v_d_n_cmd += self.k_force_fb * f_err
            v_d = v_d_tan + v_d_n_cmd * (-n)

        spd = np.linalg.norm(v_d)
        if spd > 0.15:
            v_d = v_d * (0.15 / spd)

        # ── 3. Damping matrix D (anisotropic) ─────────────────────────
        D = self.d_t * np.eye(3) + (self.d_n - self.d_t) * np.outer(n, n)

        # ── 4. Current EE velocity ─────────────────────────────────────
        v_ee = J_lin @ qd

        # ── 5. Cartesian impedance force ───────────────────────────────
        F_cart = D @ (v_d - v_ee)

        # ── 6. Orientation alignment torque ────────────────────────────
        # tool_center_site z-axis should point into the surface (-n).
        R_ee   = self.env.ee_rot()
        z_tool = R_ee[:, 2]
        z_des  = -n
        ori_err = np.cross(z_tool, z_des)

        # Feedforward: ω_ff = n × (v_tan / R_eff), scaled in only near contact.
        v_tan    = v_d - np.dot(v_d, n) * n
        R_eff    = getattr(self.ds, 'R_eff', self.sphere.R)
        omega_ff = np.cross(n, v_tan) / self.sphere.R

        # Proportional gain is always 1.0 so orientation tracks -n throughout approach.
        # Feedforward ramps in with sigma to avoid torque spikes far from surface.
        ff_weight = np.clip((sigma - 0.5) / 0.5, 0.0, 1.0)
        omega_d  = ff_weight * omega_ff + self.k_ori * ori_err
        omega_ee = J_rot @ qd
        T_ori    = self.d_ori * (omega_d - omega_ee)

        self.last_ori_err_norm = float(np.linalg.norm(ori_err))
        self.last_omega_ff = omega_ff.copy()
        self.last_omega_d  = omega_d.copy()

        # ── 7. Joint torques ───────────────────────────────────────────
        tau  = J_lin.T @ F_cart + J_rot.T @ T_ori
        tau += self.env.data.qfrc_bias[:self._n_joints]
        tau -= self.env.data.qfrc_passive[:self._n_joints]

        # ── 8. Null-space ──────────────────────────────────────────────
        J_pinv   = self._dls_pinv(J_lin)
        N        = np.eye(self._n_joints) - J_pinv @ J_lin
        tau_null = N @ (self.k_null * (self.q_ref - q) - self.b_null * qd)
        tau     += tau_null

        tau = np.clip(tau, -self._tau_max, self._tau_max)
        self.env.data.ctrl[:self._n_joints] = tau

        return v_d, n, sigma, contact_state, F_des_normal, normal_force

    # ------------------------------------------------------------------
    def _estimate_normal_force(self, n):
        """Normal contact force from FT sensor projection onto surface normal."""
        return self.env.contact_normal_force(n)

    def _dls_pinv(self, J):
        lam2 = self.dls_lambda ** 2
        return J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(3))
