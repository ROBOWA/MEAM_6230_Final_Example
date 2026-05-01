"""Centralized parameters for the 3/8-sphere polishing demo."""

import numpy as np

# ── Sphere geometry ───────────────────────────────────────────────────
SPHERE_CENTER = np.array([0.5, 0.0, 0.3])  # world frame [m]
SPHERE_RADIUS = 0.15                         # [m]
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
                          # compensates d_t coupling artefact; F_err < 1.1 N at d_t=3000

# Initial joint configuration – arm positioned above sphere top
# Places EE at approximately (0.5, 0, 0.55)
Q_INIT = np.array([0.0, -0.4, 0.0, -1.9, 0.0, 1, 0.785])

# ── Simulation ────────────────────────────────────────────────────────
SIM_DURATION = 35.0   # total demo duration [s]

# ── Paths ─────────────────────────────────────────────────────────────
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(
    _HERE, "assets", "franka_panda", "franka_emika_panda", "scene_demo.xml"
)
