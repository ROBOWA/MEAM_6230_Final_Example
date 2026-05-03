"""
DS-Based Surface Polishing Demo — Cylindrical Tool + FT Sensor
==============================================================
Uses scene_2.xml (workbench + sphere_patch) and panda_3.xml (FT sensor + tool).

Contact point: tool_center_site (center of cylindrical polishing face).
Contact force: wrist FT sensor projected onto the local surface normal.
Signed distance: ||tool_center_site - sphere_center|| - R  (no r_tool offset).

Run:
    python run_demo_tool.py
"""

import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config
from src.sphere_surface import SphereSurface
from src.ds_tool import PolishingDSTool
from src.sim_env_tool import SimEnvTool
from src.controller_tool import PolishingControllerTool


# ──────────────────────────────────────────────────────────────────────
def run():
    sphere = SphereSurface(
        config.SPHERE_CENTER_TOOL,
        config.SPHERE_RADIUS_TOOL,
        config.SPHERE_MAX_POLAR,
    )
    env = SimEnvTool(config.SCENE_XML_TOOL)

    ds = PolishingDSTool(
        sphere,
        r_cyl=config.TOOL_CYL_RADIUS,
        v_target=config.DS_V_TARGET,
        omega=config.DS_OMEGA,
        r_circle=config.DS_R_CIRCLE_TOOL,
        k_limit=config.DS_K_LIMIT,
        d_blend=config.DS_D_BLEND_TOOL,
    )

    ctrl = PolishingControllerTool(
        env, sphere, ds,
        q_ref=config.Q_INIT_TOOL,
        F_d=config.FORCE_DESIRED_TOOL,
        d_n=config.CTRL_D_N,
        d_t=config.CTRL_D_T,
        k_null=config.CTRL_K_NULL,
        b_null=config.CTRL_B_NULL,
        dls_lambda=config.CTRL_DLS_LAMBDA,
        force_ramp_time=config.CTRL_FORCE_RAMP_TIME,
        k_force_fb=getattr(config, "CTRL_K_FORCE_FB", 0.0),
        k_ori=getattr(config, "CTRL_K_ORI", 2.0),
        d_ori=getattr(config, "CTRL_D_ORI", 5.0),
    )

    env.reset(config.Q_INIT_TOOL)
    ctrl.reset()
    env.model.opt.timestep = 0.001
    ctrl.dt = env.model.opt.timestep
    dt = env.model.opt.timestep

    max_steps = int(config.SIM_DURATION / dt)
    log_t     = np.zeros(max_steps)
    log_pos   = np.zeros((max_steps, 3))
    log_sigma = np.zeros(max_steps)
    log_force = np.zeros(max_steps)
    log_dist  = np.zeros(max_steps)

    C = config.SPHERE_CENTER_TOOL
    R = config.SPHERE_RADIUS_TOOL

    print(f"Starting tool demo: {config.SIM_DURATION:.0f} s  |  dt={dt*1000:.1f} ms")
    print(f"Sphere: center={C}, R={R} m  (top z={C[2]+R:.3f} m)")
    print(f"Target force: {config.FORCE_DESIRED_TOOL} N  "
          f"|  orbit r={config.DS_R_CIRCLE_TOOL} m")
    print("Close the viewer window to abort early.\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth   = 150
        viewer.cam.elevation = -20
        viewer.cam.distance  = 1.2
        viewer.cam.lookat[:] = [C[0], C[1], C[2] + 0.1]

        step = 0
        t_start = time.time()

        while viewer.is_running() and step < max_steps:
            step_wall = time.time()

            v_d, n, sigma, contact_state, _, fn = ctrl.step()
            env.step()

            # Debug arrows: tool z-axis (red) and outward normal (green)
            viewer.user_scn.ngeom = 0
            if contact_state in ("preload", "contact"):
                ee_now = env.ee_pos()
                z_tool = env.ee_rot()[:, 2]
                _add_arrow(viewer.user_scn, ee_now, -z_tool, 0.06, 0.003,
                           np.array([1.0, 0.2, 0.2, 1.0], np.float32))
                _add_arrow(viewer.user_scn, ee_now,  n,      0.06, 0.003,
                           np.array([0.2, 1.0, 0.2, 1.0], np.float32))

            viewer.sync()

            ee = env.ee_pos()
            log_t[step]     = step * dt
            log_pos[step]   = ee
            log_sigma[step] = sigma
            log_force[step] = fn
            log_dist[step]  = sphere.signed_dist(ee, 0.0)

            elapsed = time.time() - step_wall
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
            step += 1

    steps_done = step
    print(f"\nDone: {steps_done} steps in {time.time()-t_start:.1f} s wall-time.")

    t       = log_t[:steps_done]
    pos     = log_pos[:steps_done]
    sigma_l = log_sigma[:steps_done]
    force_l = log_force[:steps_done]
    dist_l  = log_dist[:steps_done]

    _plot_results(t, pos, sigma_l, force_l, dist_l, C, R,
                  config.FORCE_DESIRED_TOOL)


# ──────────────────────────────────────────────────────────────────────
def _add_arrow(scn, pos, direction, length, radius, rgba):
    """Append a colored arrow to mjvScene.user_scn."""
    if scn.ngeom >= scn.maxgeom:
        return
    d    = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return
    z   = d / norm
    tmp = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x   = np.cross(tmp, z); x /= np.linalg.norm(x)
    y   = np.cross(z, x)
    mat = np.column_stack([x, y, z])
    import mujoco as _mj
    _mj.mjv_initGeom(
        scn.geoms[scn.ngeom],
        _mj.mjtGeom.mjGEOM_ARROW,
        np.array([radius, radius, length]),
        np.asarray(pos, dtype=float),
        mat.flatten(),
        rgba,
    )
    scn.ngeom += 1


# ──────────────────────────────────────────────────────────────────────
def _plot_results(t, pos, sigma, force, dist, C, R, F_d):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("DS Tool Polishing Demo", fontsize=14, fontweight="bold")

    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    _draw_sphere(ax3d, C, R)
    sc = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=t, cmap="plasma", s=1, zorder=5)
    plt.colorbar(sc, ax=ax3d, label="time [s]", shrink=0.6)
    ax3d.set_title("Tool Trajectory on Sphere")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(pos[:, 0], pos[:, 1], c=t, cmap="plasma", s=1)
    theta = np.linspace(0, 2*np.pi, 200)
    ax2.plot(C[0] + R*np.cos(theta), C[1] + R*np.sin(theta),
             "k--", lw=0.8, label="sphere equator")
    ax2.set_aspect("equal"); ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    ax2.set_title("Tool Trajectory (top view)"); ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, dist * 1e3, lw=0.8)
    ax3.axhline(0, color="r", ls="--", lw=0.8, label="surface")
    ax3.set_xlabel("time [s]"); ax3.set_ylabel("distance [mm]")
    ax3.set_title("Signed Surface Distance (+ = above)"); ax3.legend()

    ax4 = fig.add_subplot(2, 3, 4)
    lp  = max(1, int(0.01 / (t[1]-t[0]))) if len(t) > 1 else 1
    flt = uniform_filter1d(force, size=lp)
    ax4.plot(t, force, lw=0.4, alpha=0.3, color="tab:blue", label="Raw FT $F_n$")
    ax4.plot(t, flt,   lw=1.2, color="tab:blue", label="Filtered $F_n$")
    ax4.axhline(F_d, color="r", ls="--", lw=1.2, label=f"F_d = {F_d} N")
    ax4.set_xlabel("time [s]"); ax4.set_ylabel("Force [N]")
    ax4.set_title("Normal Contact Force (FT sensor)"); ax4.legend()

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, sigma, lw=0.8, color="tab:orange")
    ax5.set_ylim(-0.05, 1.05)
    ax5.set_xlabel("time [s]"); ax5.set_ylabel("σ")
    ax5.set_title("Contact Blend Weight")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t, pos[:, 0], lw=0.8, label="x")
    ax6.plot(t, pos[:, 1], lw=0.8, label="y")
    ax6.plot(t, pos[:, 2], lw=0.8, label="z")
    ax6.set_xlabel("time [s]"); ax6.set_ylabel("position [m]")
    ax6.set_title("Tool Contact Site — World Position"); ax6.legend()

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "results_tool.png")
    plt.savefig(out, dpi=150)
    print(f"Plots saved to {out}")
    plt.show()


def _draw_sphere(ax, c, R, alpha=0.15):
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, 3*np.pi/4, 20)
    x = c[0] + R*np.outer(np.cos(u), np.sin(v))
    y = c[1] + R*np.outer(np.sin(u), np.sin(v))
    z = c[2] + R*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, color="steelblue")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
