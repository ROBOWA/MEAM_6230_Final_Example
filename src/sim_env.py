"""
MuJoCo environment wrapper for the polishing demo.

Encapsulates model loading, sensor access, and Jacobian queries
behind a clean API that mirrors the sensor interface of the original
C++/ROS code (robot pose, twist, wrench, surface pose).
"""

import numpy as np
import mujoco


class SimEnv:
    SITE = "attachment_site"
    N_JOINTS = 7

    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Cache IDs
        self._site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.SITE
        )
        self._att_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "attachment"
        )
        self._sensor_force_adr = self._find_sensor_adr("tool_force")

        # Joint limit arrays
        self.q_min = self.model.jnt_range[:self.N_JOINTS, 0]
        self.q_max = self.model.jnt_range[:self.N_JOINTS, 1]

        # Contact force hold state (bridges intermittent zero-contact frames)
        self._last_fn = 0.0
        self._fn_hold_decay = 0.97   # per-step decay when no contact (~30ms half-life at 1kHz)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def reset(self, q0=None):
        mujoco.mj_resetData(self.model, self.data)
        if q0 is not None:
            self.data.qpos[:self.N_JOINTS] = q0
            self.data.ctrl[:self.N_JOINTS] = q0
        mujoco.mj_forward(self.model, self.data)
        self._last_fn = 0.0

    def set_from_keyframe(self, key_name="init"):
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, key_name
        )
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # Sensor access
    # ------------------------------------------------------------------
    def ee_pos(self):
        """EE site position in world frame (3,)."""
        return self.data.site_xpos[self._site_id].copy()

    def ee_vel(self):
        """EE site linear velocity in world frame (3,)."""
        J = self._jacobian_lin()
        return J @ self.data.qvel[:self.N_JOINTS]

    def raw_contact_normal_force(self):
        """Raw MuJoCo contact normal force — may be 0 on intermittent frames."""
        tool_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "tool_sphere"
        )
        sphere_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "polishing_sphere"
        )
        if tool_geom_id < 0 or sphere_geom_id < 0:
            return 0.0
        total_fn = 0.0
        cf = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom[0] == tool_geom_id and c.geom[1] == sphere_geom_id) or \
               (c.geom[0] == sphere_geom_id and c.geom[1] == tool_geom_id):
                mujoco.mj_contactForce(self.model, self.data, i, cf)
                total_fn += max(0.0, float(cf[0]))
        return total_fn

    def contact_normal_force(self):
        """Contact normal force with exponential hold on intermittent zero frames.

        When MuJoCo reports zero contact (tool momentarily not in the contact
        zone), the previous value decays at _fn_hold_decay per step rather than
        jumping to 0.  This prevents the saw-tooth artefact in force plots and
        in the controller's EMA filter without altering control logic.
        """
        raw = self.raw_contact_normal_force()
        if raw > 0.0:
            self._last_fn = raw
        else:
            self._last_fn *= self._fn_hold_decay
        return self._last_fn

    def qpos(self):
        return self.data.qpos[:self.N_JOINTS].copy()

    def qvel(self):
        return self.data.qvel[:self.N_JOINTS].copy()

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------
    def _jacobian_lin(self):
        """Linear Jacobian (3 x N_JOINTS) at the EE site."""
        J_lin = np.zeros((3, self.model.nv))
        J_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_lin, J_rot, self._site_id)
        return J_lin[:, : self.N_JOINTS]

    def jacobian(self):
        return self._jacobian_lin()

    def ee_rot(self):
        """EE site rotation matrix (3×3, columns = x/y/z axes) in world frame."""
        return self.data.site_xmat[self._site_id].reshape(3, 3).copy()

    def jacobian_full(self):
        """Linear and rotational Jacobians (3×N_JOINTS each) at the EE site."""
        J_lin = np.zeros((3, self.model.nv))
        J_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_lin, J_rot, self._site_id)
        return J_lin[:, :self.N_JOINTS], J_rot[:, :self.N_JOINTS]

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def step(self):
        mujoco.mj_step(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[sid])
