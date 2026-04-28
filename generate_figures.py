"""Generate figures for the report (headless, no viewer)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco

import config
from sphere_surface import SphereSurface
from ds import PolishingDS
from sim_env import SimEnv
from controller import PolishingController

# ── Run simulation ────────────────────────────────────────────────────────────
print("Running simulation (headless)...")
sphere = SphereSurface(config.SPHERE_CENTER, config.SPHERE_RADIUS, config.SPHERE_MAX_POLAR)
env = SimEnv(config.SCENE_XML)
ds  = PolishingDS(sphere, r_tool=config.TOOL_RADIUS, v_target=config.DS_V_TARGET,
                  omega=config.DS_OMEGA, r_circle=config.DS_R_CIRCLE)
ctrl = PolishingController(env, sphere, ds, q_ref=config.Q_INIT,
                           F_d=config.FORCE_DESIRED, d_n=config.CTRL_D_N,
                           d_t=config.CTRL_D_T, k_null=config.CTRL_K_NULL,
                           b_null=config.CTRL_B_NULL, dls_lambda=config.CTRL_DLS_LAMBDA)
env.reset(config.Q_INIT); ctrl.reset()
dt = env.model.opt.timestep

SIM_T = 40.0
N = int(SIM_T / dt)
log_t   = np.zeros(N)
log_pos = np.zeros((N, 3))
log_sigma= np.zeros(N)
log_fn  = np.zeros(N)
log_dist= np.zeros(N)
log_vd  = np.zeros(N)
log_vee = np.zeros(N)

for k in range(N):
    v_d, n, sigma = ctrl.step()
    env.step()
    ee = env.ee_pos()
    log_t[k]    = k * dt
    log_pos[k]  = ee
    log_sigma[k]= sigma
    log_fn[k]   = env.contact_normal_force()
    log_dist[k] = sphere.signed_dist(ee, config.TOOL_RADIUS)
    log_vd[k]   = np.linalg.norm(v_d)
    log_vee[k]  = np.linalg.norm(env.ee_vel())
    if k % 5000 == 0:
        print(f"  {k*dt:.1f}s / {SIM_T:.0f}s")

print("Done. Generating figures...")

OUT = os.path.join(os.path.dirname(__file__), "report_figs")
os.makedirs(OUT, exist_ok=True)
C = config.SPHERE_CENTER
R = config.SPHERE_RADIUS

# ─── Figure 1: 3-D EE trajectory on sphere ───────────────────────────────────
fig = plt.figure(figsize=(6, 5))
ax  = fig.add_subplot(111, projection="3d")
u   = np.linspace(0, 2*np.pi, 60)
v_s = np.linspace(0, 3*np.pi/4, 30)
xs  = C[0] + R*np.outer(np.cos(u), np.sin(v_s))
ys  = C[1] + R*np.outer(np.sin(u), np.sin(v_s))
zs  = C[2] + R*np.outer(np.ones_like(u), np.cos(v_s))
ax.plot_surface(xs, ys, zs, alpha=0.18, color="steelblue", linewidth=0)
# polishing region only (after contact established)
t_contact_start = np.argmax(log_sigma > 0.4)
sc = ax.scatter(log_pos[t_contact_start:, 0], log_pos[t_contact_start:, 1],
                log_pos[t_contact_start:, 2],
                c=log_t[t_contact_start:], cmap="plasma", s=1.5, zorder=5)
plt.colorbar(sc, ax=ax, label="time [s]", shrink=0.6, pad=0.12)
ax.scatter(*C, color="k", s=40, zorder=10, label="Sphere center")
att = np.array(C) + np.array([0, 0, R])
ax.scatter(*att, color="r", s=60, marker="*", zorder=10, label="Attractor")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
ax.set_title("EE Trajectory during Polishing Phase")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_trajectory_3d.pdf", bbox_inches="tight", dpi=200)
fig.savefig(f"{OUT}/fig_trajectory_3d.png", bbox_inches="tight", dpi=200)
plt.close()

# ─── Figure 2: Top-view XY trajectory ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(C[0] + R*np.cos(theta), C[1] + R*np.sin(theta),
        "b--", lw=0.8, alpha=0.4, label="Sphere equator")
ax.plot(C[0] + config.DS_R_CIRCLE*np.cos(theta),
        C[1] + config.DS_R_CIRCLE*np.sin(theta),
        "r--", lw=1.0, label=f"Target circle (r={config.DS_R_CIRCLE} m)")
sc = ax.scatter(log_pos[t_contact_start:, 0], log_pos[t_contact_start:, 1],
                c=log_t[t_contact_start:], cmap="plasma", s=2)
plt.colorbar(sc, ax=ax, label="time [s]")
ax.set_aspect("equal")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("EE Trajectory — Top View (XY Plane)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_trajectory_xy.pdf", bbox_inches="tight", dpi=200)
fig.savefig(f"{OUT}/fig_trajectory_xy.png", bbox_inches="tight", dpi=200)
plt.close()

# ─── Figure 3: Contact force and surface distance ─────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

axes[0].plot(log_t, log_sigma, lw=0.8, color="tab:green")
axes[0].axhline(0.5, color="k", ls="--", lw=0.7, alpha=0.5, label="σ = 0.5")
axes[0].set_ylabel("σ (blend weight)")
axes[0].set_ylim(-0.05, 1.05)
axes[0].set_title("DS Blend Weight  (0 = reaching, 1 = polishing)")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].plot(log_t, log_dist*1e3, lw=0.6, color="tab:blue")
axes[1].axhline(0, color="r", ls="--", lw=0.8, label="Sphere surface")
axes[1].set_ylabel("Signed dist. [mm]")
axes[1].set_title("Signed Surface Distance  (+ above, − in contact)")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

axes[2].plot(log_t, log_fn, lw=0.6, color="tab:orange", label="Measured $F_n$")
axes[2].axhline(config.FORCE_DESIRED, color="r", ls="--", lw=1.0,
                label=f"Target $F_d$ = {config.FORCE_DESIRED} N")
axes[2].set_ylabel("Normal force [N]")
axes[2].set_xlabel("Time [s]")
axes[2].set_title("Contact Normal Force")
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f"{OUT}/fig_force_dist.pdf", bbox_inches="tight", dpi=200)
fig.savefig(f"{OUT}/fig_force_dist.png", bbox_inches="tight", dpi=200)
plt.close()

# ─── Figure 4: DS velocity field (2D slice at z=sphere top) ──────────────────
fig, ax = plt.subplots(figsize=(5, 5))
grid_x = np.linspace(C[0]-0.12, C[0]+0.12, 20)
grid_y = np.linspace(C[1]-0.12, C[1]+0.12, 20)
GX, GY = np.meshgrid(grid_x, grid_y)
VX, VY = np.zeros_like(GX), np.zeros_like(GY)
for i in range(GX.shape[0]):
    for j in range(GX.shape[1]):
        p = np.array([GX[i,j], GY[i,j], C[2]+R])  # at sphere top
        try:
            v, _, _ = ds.compute(p)
            VX[i,j], VY[i,j] = v[0], v[1]
        except:
            pass
spd = np.sqrt(VX**2 + VY**2) + 1e-9
ax.streamplot(grid_x, grid_y, VX.T, VY.T, color=spd.T,
              cmap="viridis", linewidth=0.8, density=1.2, arrowsize=1.2)
circ = plt.Circle((C[0], C[1]), config.DS_R_CIRCLE, fill=False,
                  color="red", lw=1.5, ls="--", label=f"Limit circle r={config.DS_R_CIRCLE} m")
ax.add_patch(circ)
ax.plot(*C[:2], "k+", ms=10, label="Sphere center (projected)")
ax.set_xlim(C[0]-0.12, C[0]+0.12)
ax.set_ylim(C[1]-0.12, C[1]+0.12)
ax.set_aspect("equal")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("DS Velocity Field — Horizontal Slice at Sphere Top")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_ds_field.pdf", bbox_inches="tight", dpi=200)
fig.savefig(f"{OUT}/fig_ds_field.png", bbox_inches="tight", dpi=200)
plt.close()

# ─── Figure 5: EE speed over time ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(log_t, log_vee*1e3, lw=0.7, color="tab:blue", label="$\|v_{ee}\|$")
ax.plot(log_t, log_vd*1e3,  lw=0.7, color="tab:orange", ls="--", label="$\|v_d\|$")
target_tan = config.DS_OMEGA * config.DS_R_CIRCLE * 1e3
ax.axhline(target_tan, color="r", ls=":", lw=1.0, label=f"Target tangential {target_tan:.1f} mm/s")
ax.set_xlabel("Time [s]"); ax.set_ylabel("Speed [mm/s]")
ax.set_title("EE Speed: Desired vs. Actual")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(log_vd.max()*1100, target_tan*1.3))
fig.tight_layout()
fig.savefig(f"{OUT}/fig_speed.pdf", bbox_inches="tight", dpi=200)
fig.savefig(f"{OUT}/fig_speed.png", bbox_inches="tight", dpi=200)
plt.close()

# ─── Stats ────────────────────────────────────────────────────────────────────
contact_mask = log_fn > 0
print(f"\n=== Simulation Statistics ===")
print(f"Total time: {SIM_T:.0f} s  |  dt = {dt*1e3:.1f} ms")
print(f"Contact fraction: {100*contact_mask.mean():.1f}%")
if contact_mask.any():
    print(f"Force when in contact: mean={log_fn[contact_mask].mean():.1f} N  "
          f"std={log_fn[contact_mask].std():.1f} N")
print(f"EE speed (polishing phase): mean={log_vee[t_contact_start:].mean()*1e3:.1f} mm/s")
print(f"\nFigures saved to: {OUT}/")
