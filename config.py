"""Centralized parameters for the 3/8-sphere polishing demo."""

import numpy as np

# ── Sphere geometry ───────────────────────────────────────────────────
SPHERE_CENTER = np.array([0.5, 0.0, 0.3])  # world frame [m]
SPHERE_RADIUS = 0.3                         # [m]
SPHERE_MAX_POLAR = 3 * np.pi / 4            # 135 deg  (3/8 of full sphere)
TOOL_RADIUS = 0.009                          # polishing tool sphere [m]
TOOL_Z_OFFSET = 0.06                         # tool tip offset along EE z-axis from attachment origin [m]

# ── DS parameters ─────────────────────────────────────────────────────
DS_V_TARGET = 0.05    # reaching speed toward surface [m/s]
DS_OMEGA = np.pi / 3  # circular angular frequency [rad/s]
DS_R_CIRCLE = 0.05    # polishing circle radius [m]
DS_K_LIMIT = 6.0      # radial limit-cycle attraction gain [1/s]  (↑ compensates lower d_t)
DS_D_BLEND = 0.07    # distance scale for reaching↔circular blend [m]

# ── Force control ─────────────────────────────────────────────────────
FORCE_DESIRED = 15.0   # desired normal contact force [N]
FORCE_KF = 0.000     # (unused with torque control — kept for logging)

# ── Passive DS Impedance (torque control) ────────────────────────────
CTRL_D_N = 200.0      # normal damping [N·s/m] — steady-state F_n = F_d (paper-style modulation)
CTRL_D_T = 2000.0     # tangential damping [N·s/m] — high enough for orbit tracking (~78% at omega=π/3)
CTRL_K_NULL = 5.0     # null-space joint stiffness [N·m/rad]
CTRL_B_NULL = 5.0     # null-space joint damping — raised to compensate for cancelled qfrc_passive
CTRL_DLS_LAMBDA = 0.02  # DLS regularization
CTRL_V_MAX = 0.15     # EE speed limit [m/s]
CTRL_FORCE_RAMP_TIME = 2.0   # time [s] to linearly ramp from F_preload to F_desired after contact
CTRL_K_FORCE_FB = 0.0  # force-feedback gain [m/(s·N)]: v_d_n += k*(F_d - F_meas)
CTRL_K_ORI = 5.0     # orientation alignment gain [1/s]: higher value keeps tilt < 2° during approach
CTRL_D_ORI = 15.0     # orientation damping [N·m·s/rad]

# Initial joint configuration – arm positioned above sphere top
# Places EE at approximately (0.5, 0, 0.55)
Q_INIT = np.array([0.0, -0.4, 0.0, -1.9, 0.0, 1, 0.785])

# ── Simulation ────────────────────────────────────────────────────────
SIM_DURATION = 35.0   # total demo duration [s]

# ── Tool setup A: workbench sphere (scene_2.xml + panda_3.xml) ────────
# sphere_patch world position: workbench pos (0.5545,0,0) + local (0,0,0.175)
SPHERE_CENTER_TOOL = np.array([0.5545, 0.0, 0.175])
SPHERE_RADIUS_TOOL = 0.1      # matches sphere_patch size in scene_2.xml
TOOL_RADIUS_TOOL   = 0.0      # legacy; not used by PolishingDSTool
TOOL_CYL_RADIUS    = 0.0155   # cylinder face radius [m] from panda_3.xml size[0]
# Effective sphere radius for DS: R_eff = sqrt(R² - r_cyl²) ≈ 98.8 mm
# Contact offset ≈ 1.2 mm — DS sigma→1 when rim touches, not when center is on sphere.
DS_D_BLEND_TOOL    = 0.03     # blend zone [m] — smaller for R=0.1 sphere
DS_R_CIRCLE_TOOL   = 0.03     # polishing orbit radius [m]
FORCE_DESIRED_TOOL = 5.0      # desired normal contact force [N]
# panda_3.xml "home" keyframe — verify tool_center_site is above sphere top (z=0.275)
Q_INIT_TOOL = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# ── Tool setup B: large sphere demo (scene_demo_tool.xml + panda_4.xml) ──
# Same R=0.3 sphere as the original demo; robot now has wrist FT + cylindrical tool.
# Use SimEnvTool for sensing (tool_center_site + ft_site).
# tool_center_site extends ~0.063 m further than attachment_site — verify Q_INIT_DEMO_TOOL
# by running: env.reset(Q_INIT_DEMO_TOOL); env.forward(); print(env.ee_pos()) — must be > 0.6
SPHERE_CENTER_DEMO_TOOL = np.array([0.5, 0.0, 0.3])   # same as SPHERE_CENTER
SPHERE_RADIUS_DEMO_TOOL = 0.3                          # same as SPHERE_RADIUS
TOOL_RADIUS_DEMO_TOOL   = 0.0                          # tool_center_site = contact point
DS_D_BLEND_DEMO_TOOL    = 0.07                         # same as DS_D_BLEND (R=0.3 sphere)
DS_R_CIRCLE_DEMO_TOOL   = 0.05                         # same as DS_R_CIRCLE
FORCE_DESIRED_DEMO_TOOL = 15.0                         # same as FORCE_DESIRED
# Verified: tool_center_site at [0.493, 0, 0.653] — 53 mm above sphere top (z=0.6)
Q_INIT_DEMO_TOOL = np.array([0.0, -0.6, 0.0, -2.1, 0.0, 2.3, 0.785])

# ── Paths ─────────────────────────────────────────────────────────────
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(
    _HERE, "assets", "franka_panda", "franka_emika_panda", "scene_demo.xml"
)
SCENE_XML_TOOL = os.path.join(
    _HERE, "models", "scene_2.xml"
)
SCENE_XML_DEMO_TOOL = os.path.join(
    _HERE, "assets", "franka_panda", "franka_emika_panda", "scene_demo_tool.xml"
)
