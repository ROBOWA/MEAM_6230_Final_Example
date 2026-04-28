"""
Passive DS Impedance Controller (no tank energy).

Control law (Cartesian torque):
    τ = J^T · D · (v_d − v_ee) + g(q)

where:
  J     — 3×7 linear Jacobian at the EE site
  D     — 3×3 positive-definite damping matrix (anisotropic: d_n in normal, d_t tangential)
  v_d   — desired EE linear velocity from DS + depth spring + force modulation
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
        d_n=200.0,   # normal damping [N·s/m]
        d_t=150.0,   # tangential damping [N·s/m]
        k_null=5.0,  # null-space joint stiffness [N·m/rad]
        b_null=1.0,  # null-space joint damping [N·m·s/rad]
        dls_lambda=0.02,
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
        self.dt = env.model.opt.timestep
        self._n_joints = env.N_JOINTS

        # Torque limits
        self._tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

    def reset(self):
        self.env.data.ctrl[:self._n_joints] = 0.0

    # ------------------------------------------------------------------
    def step(self):
        """Compute and apply one torque-control step."""
        ee_pos = self.env.ee_pos()
        q = self.env.qpos()
        qd = self.env.qvel()
        J = self.env.jacobian()          # 3×7

        # ── 1. DS desired velocity ─────────────────────────────────────
        v_d, n, sigma = self.ds.compute(ee_pos)

        # ── 2. Contact surface following ───────────────────────────────
        # Large d_t causes J^T·D·v_tan to have an outward normal component (coupling).
        # We compensate by using a larger inward normal velocity command.
        # v_d_n_eff = F_d/d_n + coupling_offset; measured F_contact ≈ F_d.
        if sigma > 0.15:
            v_d_tan = v_d - np.dot(v_d, n) * n    # tangential from DS
            # Inward press: nominal 5/200=0.025 m/s + 0.055 to overcome J-coupling
            v_d_n = 0.08 * (-n)                    # 80 mm/s → measured F_n ≈ 5-10 N
            v_d = v_d_tan + v_d_n

        # Speed limit
        spd = np.linalg.norm(v_d)
        if spd > 0.15:
            v_d = v_d * (0.15 / spd)

        # ── 3. Damping matrix D (anisotropic) ─────────────────────────
        # Higher damping in the surface-normal direction for force regulation.
        n_dir = n
        D = self.d_t * np.eye(3) + (self.d_n - self.d_t) * np.outer(n_dir, n_dir)

        # ── 4. Current EE velocity ─────────────────────────────────────
        v_ee = J @ qd

        # ── 5. Cartesian impedance force ───────────────────────────────
        F_cart = D @ (v_d - v_ee)

        # ── 6. Map to joint torques ────────────────────────────────────
        tau = J.T @ F_cart

        # ── 7. Gravity + Coriolis compensation ────────────────────────
        tau += self.env.data.qfrc_bias[:self._n_joints]

        # ── 8. Null-space: joint stiffness/damping toward q_ref ───────
        J_pinv = self._dls_pinv(J)
        N = np.eye(self._n_joints) - J_pinv @ J
        tau_null = N @ (self.k_null * (self.q_ref - q) - self.b_null * qd)
        tau += tau_null

        # ── 9. Clip and apply ──────────────────────────────────────────
        tau = np.clip(tau, -self._tau_max, self._tau_max)
        self.env.data.ctrl[:self._n_joints] = tau

        return v_d, n, sigma

    # ------------------------------------------------------------------
    def _dls_pinv(self, J):
        lam2 = self.dls_lambda ** 2
        return J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(3))
