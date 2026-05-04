"""
DS-based Surface Polishing Demo
================================
Franka Panda polishes the outer surface of a 3/8 sphere using:
  - A DS velocity field (reaching + circular limit-cycle on the tangent plane)
  - Force modulation to maintain a desired contact force
  - Operational-space velocity control via DLS Jacobian pseudo-inverse

Run:
    C:/Users/cindy/anaconda3/envs/ESE5030/python.exe run_demo.py
"""

import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg unavailable
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage import uniform_filter1d

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config
from src.sphere_surface import SphereSurface
from src.ds import PolishingDS
from src.sim_env import SimEnv
from src.controller import PolishingController


# ──────────────────────────────────────────────────────────────────────
def apply_external_disturbance(env, body_id, sim_time):
    """Apply or clear a world-frame force disturbance at body_id's COM.

    Returns True while the disturbance is active.
    """
    active = (
        config.DISTURBANCE_ENABLE
        and config.DISTURBANCE_START_TIME <= sim_time
             < config.DISTURBANCE_START_TIME + config.DISTURBANCE_DURATION
    )
    if active:
        env.data.xfrc_applied[body_id, :3] = config.DISTURBANCE_FORCE
        env.data.xfrc_applied[body_id, 3:] = 0.0
    else:
        env.data.xfrc_applied[body_id, :] = 0.0
    return active


# ──────────────────────────────────────────────────────────────────────
def run():
    # Build objects
    sphere = SphereSurface(config.SPHERE_CENTER, config.SPHERE_RADIUS,
                           config.SPHERE_MAX_POLAR)
    env = SimEnv(config.SCENE_XML)

    ds = PolishingDS(
        sphere,
        r_tool=config.TOOL_RADIUS,
        v_target=config.DS_V_TARGET,
        omega=config.DS_OMEGA,
        r_circle=config.DS_R_CIRCLE,
        k_limit=config.DS_K_LIMIT,
        d_blend=config.DS_D_BLEND,
    )

    ctrl = PolishingController(
        env, sphere, ds,
        q_ref=config.Q_INIT,
        F_d=config.FORCE_DESIRED,
        d_n=config.CTRL_D_N,
        d_t=config.CTRL_D_T,
        k_null=config.CTRL_K_NULL,
        b_null=config.CTRL_B_NULL,
        dls_lambda=config.CTRL_DLS_LAMBDA,
        force_ramp_time=config.CTRL_FORCE_RAMP_TIME,
        k_force_fb=getattr(config, "CTRL_K_FORCE_FB", 0.0),
    )

    # Initialise robot
    env.reset(config.Q_INIT)
    ctrl.reset()
    env.model.opt.timestep = 0.001
    ctrl.dt = env.model.opt.timestep
    dt = env.model.opt.timestep

    # ── Disturbance setup ─────────────────────────────────────────────
    disturbance_body_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, config.DISTURBANCE_BODY_NAME
    )
    if disturbance_body_id < 0:
        raise ValueError(f"Disturbance body '{config.DISTURBANCE_BODY_NAME}' not found in model")

    # ── Logging arrays ────────────────────────────────────────────────
    max_steps = int(config.SIM_DURATION / dt)
    log_t          = np.zeros(max_steps)
    log_pos        = np.zeros((max_steps, 3))  # EE position
    log_sigma      = np.zeros(max_steps)       # contact weight
    log_force      = np.zeros(max_steps)       # normal contact force [N]
    log_dist       = np.zeros(max_steps)       # signed surface distance [m]
    log_disturbance = np.zeros(max_steps, dtype=bool)  # disturbance active flag

    print(f"Starting demo: {config.SIM_DURATION:.0f} s  |  dt={dt*1000:.1f} ms")
    print(f"Sphere: center={config.SPHERE_CENTER}, R={config.SPHERE_RADIUS} m")
    print(f"Target force: {config.FORCE_DESIRED} N  |  circle r={config.DS_R_CIRCLE} m")
    if config.DISTURBANCE_ENABLE:
        print(f"Disturbance: body='{config.DISTURBANCE_BODY_NAME}'  "
              f"F={config.DISTURBANCE_FORCE} N  "
              f"t=[{config.DISTURBANCE_START_TIME}, "
              f"{config.DISTURBANCE_START_TIME + config.DISTURBANCE_DURATION}] s")
    print("Close the viewer window to abort early.\n")

    # ── Trail & force-arrow visualization constants ───────────────────
    _TRAIL_MAXLEN      = 2000   # ~2 s history at 1 kHz
    _TRAIL_STRIDE      = 10     # draw 1-in-10 → ≤200 spheres per frame
    _TRAIL_RADIUS      = 0.003  # trail sphere radius [m]
    _FORCE_ARROW_SCALE = 0.008  # arrow length per N [m/N]
    _trail = deque(maxlen=_TRAIL_MAXLEN)

    # ── Simulation loop ───────────────────────────────────────────────
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Nice camera angle
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.4
        viewer.cam.lookat[:] = [0.4, 0, 0.3]

        step = 0
        t_start = time.time()
        _was_disturbed = False

        while viewer.is_running() and step < max_steps:
            step_wall = time.time()
            sim_time = step * dt

            # Controller step
            v_d, n, sigma, _, _, _ = ctrl.step()

            # Disturbance — applied before mj_step (inside env.step)
            active = apply_external_disturbance(env, disturbance_body_id, sim_time)
            if active and not _was_disturbed:
                print(f"  [t={sim_time:.2f}s] Disturbance ON  "
                      f"body='{config.DISTURBANCE_BODY_NAME}'  "
                      f"F={config.DISTURBANCE_FORCE} N")
            elif not active and _was_disturbed:
                print(f"  [t={sim_time:.2f}s] Disturbance OFF")
            _was_disturbed = active

            # Simulation step
            env.step()

            # ── Custom visuals (trail + force arrow) ──────────────────
            ee = env.ee_pos()
            _trail.append(ee.copy())

            viewer.user_scn.ngeom = 0

            # Tool-sphere trajectory trail (green spheres, fading with age)
            pts = list(_trail)[::_TRAIL_STRIDE]
            n_pts = len(pts)
            for i, p in enumerate(pts):
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 2:
                    break
                age = i / max(n_pts - 1, 1)        # 0 = oldest, 1 = newest
                alpha = float(age ** 1.5)
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([_TRAIL_RADIUS, 0.0, 0.0]),
                    p, np.eye(3).flatten(),
                    np.array([0.2, 1.0, 0.2, alpha], dtype=np.float32),
                )
                viewer.user_scn.ngeom += 1

            # Disturbance force arrow (orange, only while active)
            if active and viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                body_com = env.data.xpos[disturbance_body_id].copy()
                tip = body_com + config.DISTURBANCE_FORCE * _FORCE_ARROW_SCALE
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_connector(
                    g, mujoco.mjtGeom.mjGEOM_ARROW, 0.01, body_com, tip,
                )
                g.rgba[:] = np.array([1.0, 0.35, 0.0, 0.9], dtype=np.float32)
                viewer.user_scn.ngeom += 1

            viewer.sync()

            # Log  (ee computed above in the visual block)
            F_n = env.contact_normal_force()
            d = sphere.signed_dist(ee, config.TOOL_RADIUS)

            log_t[step] = sim_time
            log_pos[step] = ee
            log_sigma[step] = sigma
            log_force[step] = F_n
            log_dist[step] = d
            log_disturbance[step] = active

            # Real-time pacing
            elapsed = time.time() - step_wall
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

            step += 1

    steps_done = step
    wall_time = time.time() - t_start
    print(f"\nDone: {steps_done} steps in {wall_time:.1f} s wall-time.")

    # ── Post-simulation plots ─────────────────────────────────────────
    t = log_t[:steps_done]
    pos = log_pos[:steps_done]
    sigma_log = log_sigma[:steps_done]
    force_log = log_force[:steps_done]
    dist_log = log_dist[:steps_done]
    disturbance_log = log_disturbance[:steps_done]

    _plot_results(t, pos, sigma_log, force_log, dist_log,
                  config.SPHERE_CENTER, config.SPHERE_RADIUS,
                  config.FORCE_DESIRED, disturbance_log)


# ──────────────────────────────────────────────────────────────────────
def _plot_results(t, pos, sigma, force, dist, c, R, F_d, disturbance=None):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("DS-Based Surface Polishing Demo", fontsize=14, fontweight="bold")

    # Disturbance window bounds for shading (derived from log, not config,
    # so it reflects what actually ran even if the window was cut short)
    d_t0 = d_t1 = None
    if disturbance is not None and disturbance.any():
        idx = np.where(disturbance)[0]
        d_t0, d_t1 = t[idx[0]], t[idx[-1]]

    def _shade(ax):
        if d_t0 is not None:
            ax.axvspan(d_t0, d_t1, color="orange", alpha=0.15, label="disturbance")

    # 1. 3-D trajectory on sphere
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    _draw_sphere(ax3d, c, R)
    sc = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=t, cmap="plasma", s=1, zorder=5)
    plt.colorbar(sc, ax=ax3d, label="time [s]", shrink=0.6)
    ax3d.set_title("EE Trajectory on Sphere")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")

    # 2. XY trajectory (top view)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(pos[:, 0], pos[:, 1], c=t, cmap="plasma", s=1)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(c[0] + R * np.cos(theta), c[1] + R * np.sin(theta),
             "k--", lw=0.8, label="sphere equator")
    ax2.set_aspect("equal"); ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    ax2.set_title("EE Trajectory (top view)"); ax2.legend(fontsize=7)

    # 3. Signed surface distance
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, dist * 1e3, lw=0.8)
    ax3.axhline(0, color="r", ls="--", lw=0.8, label="surface")
    _shade(ax3); ax3.legend()
    ax3.set_xlabel("time [s]"); ax3.set_ylabel("distance [mm]")
    ax3.set_title("Signed Surface Distance (+ = above)")

    # 4. Normal contact force
    ax4 = fig.add_subplot(2, 3, 4)
    lp_window = max(1, int(0.01 / (t[1] - t[0]))) if len(t) > 1 else 1  # 10 ms
    force_filt = uniform_filter1d(force, size=lp_window)
    ax4.plot(t, force, lw=0.4, alpha=0.3, color="tab:blue", label="Raw $F_n$")
    ax4.plot(t, force_filt, lw=1.2, color="tab:blue", label="Filtered $F_n$ (LP 5 Hz)")
    ax4.axhline(F_d, color="r", ls="--", lw=1.2, label=f"F_d = {F_d} N")
    _shade(ax4); ax4.legend()
    ax4.set_xlabel("time [s]"); ax4.set_ylabel("Force [N]")
    ax4.set_title("Normal Contact Force")

    # 5. Contact blend weight sigma
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, sigma, lw=0.8, color="tab:orange")
    ax5.set_ylim(-0.05, 1.05)
    _shade(ax5); ax5.legend()
    ax5.set_xlabel("time [s]"); ax5.set_ylabel("σ")
    ax5.set_title("Contact Blend Weight (0=reach, 1=polish)")

    # 6. EE height over time
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t, pos[:, 0], lw=0.8, label="x")
    ax6.plot(t, pos[:, 1], lw=0.8, label="y")
    ax6.plot(t, pos[:, 2], lw=0.8, label="z")
    _shade(ax6); ax6.legend()
    ax6.set_xlabel("time [s]"); ax6.set_ylabel("position [m]")
    ax6.set_title("EE World Position")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "results.png")
    plt.savefig(out_path, dpi=150)
    print(f"Plots saved to {out_path}")
    plt.show()


def _draw_sphere(ax, c, R, alpha=0.15):
    """Draw a transparent sphere mesh in a 3D axes."""
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, 3 * np.pi / 4, 20)   # 3/8 sphere (0..135 deg from top)
    x = c[0] + R * np.outer(np.cos(u), np.sin(v))
    y = c[1] + R * np.outer(np.sin(u), np.sin(v))
    z = c[2] + R * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, color="steelblue")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
