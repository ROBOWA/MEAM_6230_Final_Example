"""
Interactive 3D visualisation of the PolishingDS velocity field.

Run:
    python view_ds_3d.py

Drag to rotate, scroll to zoom.  Close the window to exit.

Shows:
  - Transparent 3/8-sphere surface
  - DS velocity arrows ON the sphere surface  (red)
  - DS velocity arrows in the volume ABOVE the sphere (green) — reaching funnel
  - Integrated example trajectories from several start points
  - Attractor point and limit-cycle circle
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config
from sphere_surface import SphereSurface
from ds import PolishingDS

# ── Build DS ───────────────────────────────────────────────────────────────────
sphere = SphereSurface(config.SPHERE_CENTER, config.SPHERE_RADIUS, config.SPHERE_MAX_POLAR)
ds = PolishingDS(
    sphere,
    r_tool=config.TOOL_RADIUS,
    v_target=config.DS_V_TARGET,
    omega=config.DS_OMEGA,
    r_circle=config.DS_R_CIRCLE,
    k_limit=config.DS_K_LIMIT,
    d_blend=config.DS_D_BLEND,
)

C = config.SPHERE_CENTER
R = config.SPHERE_RADIUS
att = C + np.array([0.0, 0.0, R])  # attractor = sphere top

# ── Figure / axes ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# ── 1. Sphere surface ──────────────────────────────────────────────────────────
u_mesh = np.linspace(0, 2 * np.pi, 60)
v_mesh = np.linspace(0, config.SPHERE_MAX_POLAR, 35)
XS = C[0] + R * np.outer(np.cos(u_mesh), np.sin(v_mesh))
YS = C[1] + R * np.outer(np.sin(u_mesh), np.sin(v_mesh))
ZS = C[2] + R * np.outer(np.ones_like(u_mesh), np.cos(v_mesh))
ax.plot_surface(XS, YS, ZS, alpha=0.13, color="steelblue", linewidth=0)

# ── 2. DS arrows ON sphere surface ────────────────────────────────────────────
# Sample a spherical grid; place tool centre just outside the surface.
# v_lc: polar angle of the limit cycle (R*sin(v_lc) = r_circle)
v_lc = np.arcsin(np.clip(config.DS_R_CIRCLE / R, 0, 1))

# Background grid — uniform, coarser
bg_u = np.linspace(0, 2 * np.pi, 14, endpoint=False)
bg_v = np.linspace(0.08, config.SPHERE_MAX_POLAR - 0.08, 7)

# Dense rings near the limit cycle — finer azimuthal spacing, ±3 offsets
dense_u = np.linspace(0, 2 * np.pi, 24, endpoint=False)
dense_v = v_lc + np.linspace(-0.06, 0.06, 5)  # 5 rings spanning ±~3.5 deg

PX, PY, PZ, VX, VY, VZ = [], [], [], [], [], []

def _add_ring(u_arr, vi):
    for ui in u_arr:
        p_surf = C + R * np.array([np.cos(ui)*np.sin(vi),
                                   np.sin(ui)*np.sin(vi),
                                   np.cos(vi)])
        p_tool = p_surf + config.TOOL_RADIUS * sphere.normal(p_surf)
        v_d, _, _ = ds.compute(p_tool)
        PX.append(p_surf[0]); PY.append(p_surf[1]); PZ.append(p_surf[2])
        VX.append(v_d[0]);    VY.append(v_d[1]);    VZ.append(v_d[2])

for vi in bg_v:
    _add_ring(bg_u, vi)
for vi in dense_v:
    if 0 < vi < config.SPHERE_MAX_POLAR:
        _add_ring(dense_u, vi)

PX, PY, PZ = map(np.array, [PX, PY, PZ])
VX, VY, VZ = map(np.array, [VX, VY, VZ])
scale_surf = 0.055 / (np.sqrt(VX**2 + VY**2 + VZ**2).max() + 1e-9)
ax.quiver(PX, PY, PZ, VX * scale_surf, VY * scale_surf, VZ * scale_surf,
          color="tomato", linewidth=0.9, arrow_length_ratio=0.28)

# ── 3. DS arrows: spherical shells outside and inside the sphere ──────────────
def _shell_arrows(r_val, u_arr, v_arr):
    """Evaluate DS at all (u,v) on a spherical shell of radius r_val."""
    px, py, pz, vx, vy, vz = [], [], [], [], [], []
    for vi in np.asarray(v_arr):
        if vi <= 0 or vi >= np.pi:
            continue
        for ui in u_arr:
            p = C + r_val * np.array([np.cos(ui)*np.sin(vi),
                                      np.sin(ui)*np.sin(vi),
                                      np.cos(vi)])
            v_d, _, _ = ds.compute(p)
            px.append(p[0]); py.append(p[1]); pz.append(p[2])
            vx.append(v_d[0]); vy.append(v_d[1]); vz.append(v_d[2])
    return (np.array(px), np.array(py), np.array(pz),
            np.array(vx), np.array(vy), np.array(vz))

# Outside shells — four radii, progressively coarser away from surface.
# The two nearest shells also get extra rings at the limit-cycle polar angle.
out_parts = []
for r_sh, n_u, n_v, add_lc in [
    (R + 0.012, 20, 7, True),   # skin-close: densest
    (R + 0.040, 16, 6, True),   # near
    (R + 0.100, 10, 5, False),  # medium
    (R + 0.180,  8, 4, False),  # far
]:
    u_sh = np.linspace(0, 2*np.pi, n_u, endpoint=False)
    v_bg = list(np.linspace(0.06, config.SPHERE_MAX_POLAR - 0.06, n_v))
    if add_lc:
        v_bg += list(v_lc + np.linspace(-0.055, 0.055, 5))
    out_parts.append(_shell_arrows(r_sh, u_sh, v_bg))

OPX = np.concatenate([d[0] for d in out_parts])
OPY = np.concatenate([d[1] for d in out_parts])
OPZ = np.concatenate([d[2] for d in out_parts])
OVX = np.concatenate([d[3] for d in out_parts])
OVY = np.concatenate([d[4] for d in out_parts])
OVZ = np.concatenate([d[5] for d in out_parts])
scale_out = 0.04 / (np.sqrt(OVX**2 + OVY**2 + OVZ**2).max() + 1e-9)
ax.quiver(OPX, OPY, OPZ, OVX*scale_out, OVY*scale_out, OVZ*scale_out,
          color="mediumseagreen", alpha=0.75, linewidth=0.7, arrow_length_ratio=0.28)

# Inside shells — three radii; extend polar range to show interior field fully.
in_parts = []
for r_sh, n_u, n_v, v_max in [
    (R * 0.92, 14, 6, config.SPHERE_MAX_POLAR),
    (R * 0.65, 10, 5, np.pi * 0.85),
    (R * 0.35,  8, 4, np.pi * 0.70),
]:
    u_sh = np.linspace(0, 2*np.pi, n_u, endpoint=False)
    v_sh = np.linspace(0.06, v_max - 0.06, n_v)
    in_parts.append(_shell_arrows(r_sh, u_sh, v_sh))

IPX = np.concatenate([d[0] for d in in_parts])
IPY = np.concatenate([d[1] for d in in_parts])
IPZ = np.concatenate([d[2] for d in in_parts])
IVX = np.concatenate([d[3] for d in in_parts])
IVY = np.concatenate([d[4] for d in in_parts])
IVZ = np.concatenate([d[5] for d in in_parts])
scale_in = 0.04 / (np.sqrt(IVX**2 + IVY**2 + IVZ**2).max() + 1e-9)
ax.quiver(IPX, IPY, IPZ, IVX*scale_in, IVY*scale_in, IVZ*scale_in,
          color="mediumpurple", alpha=0.65, linewidth=0.7, arrow_length_ratio=0.28)

# ── 4. Integrated example trajectories ───────────────────────────────────────
dt_int = 0.02   # s
T_int  = 30.0   # s
N_int  = int(T_int / dt_int)

# Six start points spread around the sphere top at different heights
starts = []
for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
    starts.append(C + np.array([0.09 * np.cos(ang), 0.09 * np.sin(ang), R + 0.13]))

colors_traj = plt.cm.tab10(np.linspace(0, 0.6, len(starts)))

for i, p0 in enumerate(starts):
    pos = p0.copy()
    traj = [pos.copy()]
    for _ in range(N_int):
        v_d, _, _ = ds.compute(pos)
        spd = np.linalg.norm(v_d)
        if spd > config.CTRL_V_MAX:
            v_d = v_d / spd * config.CTRL_V_MAX
        pos = pos + v_d * dt_int
        traj.append(pos.copy())
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
            lw=1.1, color=colors_traj[i], alpha=0.85)

# ── 5. Attractor & limit circle ───────────────────────────────────────────────
ax.scatter(*att, color="red", s=90, marker="*", zorder=10, label="Attractor")
ax.scatter(*C,   color="k",   s=35, zorder=10, label="Sphere centre")

theta_c = np.linspace(0, 2 * np.pi, 200)
ax.plot(att[0] + config.DS_R_CIRCLE * np.cos(theta_c),
        att[1] + config.DS_R_CIRCLE * np.sin(theta_c),
        att[2] * np.ones(200),
        "r--", lw=1.8, label=f"Limit circle  r={config.DS_R_CIRCLE} m")

# ── Legend ─────────────────────────────────────────────────────────────────────
handles, labels = ax.get_legend_handles_labels()
handles += [
    Line2D([0], [0], color="tomato",         lw=1.5, label="DS field (on surface)"),
    Line2D([0], [0], color="mediumseagreen", lw=1.5, label="DS field (outside, near surface)"),
    Line2D([0], [0], color="mediumpurple",   lw=1.5, label="DS field (inside sphere)"),
    Line2D([0], [0], color="tab:blue",       lw=1.5, label="Example trajectories"),
]
ax.legend(handles=handles, fontsize=7, loc="upper left")

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("PolishingDS — 3D Velocity Field & Trajectories\n"
             "(drag to rotate, scroll to zoom)")
ax.view_init(elev=28, azim=-55)

plt.tight_layout()
plt.show()
