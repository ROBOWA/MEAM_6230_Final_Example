"""
MuJoCo environment wrapper — cylindrical polishing tool with wrist FT sensor.

Differences from SimEnv:
  - ee_pos() / ee_rot() / Jacobians: computed at 'tool_center_site'
    (center of the cylindrical polishing face), not 'attachment_site'.
  - Contact force: read from the 'wrist_force' FT sensor projected onto
    the known surface normal.  No MuJoCo contact-pair lookup needed.

FT sign convention
------------------
The MuJoCo 'force' sensor at ft_site measures the constraint force the arm
exerts on the tool-chain subtree (ft_frame + tool), expressed in the ft_site
local frame.  When the tool presses into the surface with normal force F_n
along -n, the arm applies that same force toward the surface, so:

    F_world · n  ≈  -F_n   (negative along outward normal)

Hence  contact_normal_force = max(0, dot(-F_world, n))  gives a positive
scalar proportional to the pressing force.
"""

import numpy as np
import mujoco


class SimEnvTool:
    SITE     = "tool_center_site"   # center of polishing face
    FT_SITE  = "ft_site"            # FT measurement frame
    N_JOINTS = 7

    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self._site_id    = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.SITE)
        self._ft_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.FT_SITE)
        self._force_adr  = self._sensor_adr("wrist_force")

        self.q_min = self.model.jnt_range[:self.N_JOINTS, 0]
        self.q_max = self.model.jnt_range[:self.N_JOINTS, 1]

        self._ft_bias = np.zeros(3)  # world-frame zero offset (set by set_ft_bias)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def reset(self, q0=None):
        mujoco.mj_resetData(self.model, self.data)
        if q0 is not None:
            self.data.qpos[:self.N_JOINTS] = q0
            self.data.ctrl[:self.N_JOINTS] = q0
        mujoco.mj_forward(self.model, self.data)

    def set_from_keyframe(self, key_name="home"):
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    def ee_pos(self):
        """World position of tool_center_site (3,)."""
        return self.data.site_xpos[self._site_id].copy()

    def ee_rot(self):
        """World rotation matrix of tool_center_site (3×3)."""
        return self.data.site_xmat[self._site_id].reshape(3, 3).copy()

    def ee_vel(self):
        """Linear velocity of tool_center_site in world frame (3,)."""
        J_lin, _ = self._jac_raw()
        return J_lin[:, :self.N_JOINTS] @ self.data.qvel[:self.N_JOINTS]

    def qpos(self):
        return self.data.qpos[:self.N_JOINTS].copy()

    def qvel(self):
        return self.data.qvel[:self.N_JOINTS].copy()

    # ------------------------------------------------------------------
    # Jacobians
    # ------------------------------------------------------------------
    def _jac_raw(self):
        J_lin = np.zeros((3, self.model.nv))
        J_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_lin, J_rot, self._site_id)
        return J_lin, J_rot

    def jacobian(self):
        """Linear Jacobian (3×N_JOINTS) at tool_center_site."""
        J_lin, _ = self._jac_raw()
        return J_lin[:, :self.N_JOINTS]

    def jacobian_full(self):
        """Linear and rotational Jacobians (3×N_JOINTS each) at tool_center_site."""
        J_lin, J_rot = self._jac_raw()
        return J_lin[:, :self.N_JOINTS], J_rot[:, :self.N_JOINTS]

    # ------------------------------------------------------------------
    # Force sensing
    # ------------------------------------------------------------------
    def set_ft_bias(self):
        """
        Capture current FT reading as zero offset.

        Call after reset() while the arm is stationary and not in contact.
        Subsequent ft_wrench() calls subtract this bias.
        """
        F_sensor = self.data.sensordata[self._force_adr:self._force_adr + 3].copy()
        R_ft = self.data.site_xmat[self._ft_site_id].reshape(3, 3)
        self._ft_bias = R_ft @ F_sensor

    def ft_wrench(self):
        """
        Bias-corrected wrist force in world frame (3,).

        Sensor output is in ft_site local frame; rotated to world via R_ft.
        Confirmed: dot(-F_world, n) > 0 when tool presses into surface.
        """
        F_sensor = self.data.sensordata[self._force_adr:self._force_adr + 3].copy()
        R_ft = self.data.site_xmat[self._ft_site_id].reshape(3, 3)
        return R_ft @ F_sensor - self._ft_bias

    def contact_normal_force(self, n):
        """
        Estimated contact normal force [N], projected from FT sensor.

        Args:
            n: outward surface normal (unit vector, world frame)

        Returns:
            Positive scalar when tool is pressing into the surface.
        """
        F_world = self.ft_wrench()
        return max(0.0, float(np.dot(-F_world, n)))

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def step(self):
        mujoco.mj_step(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[sid])
