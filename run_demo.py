"""
DS-based Surface Polishing Demo
================================
Franka Panda polishes the outer surface of a 3/8 sphere using:
  - A DS velocity field (reaching + circular limit-cycle on the tangent plane)
  - Force modulation to maintain a desired contact force
  - Operational-space velocity control via DLS Jacobian pseudo-inverse

Run:
    python run_demo.py
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
        use_energy_tank=getattr(config, "CTRL_USE_ENERGY_TANK", True),
        tank_s_max=getattr(config, "CTRL_TANK_S_MAX", 60.0),
        tank_init=getattr(config, "CTRL_TANK_INIT", "full"),
    )

    env.reset(config.Q_INIT)
    ctrl.reset()
    env.model.opt.timestep = 0.001
    ctrl.dt = env.model.opt.timestep
    dt = env.model.opt.timestep

    disturbance_body_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, config.DISTURBANCE_BODY_NAME
    )
    if disturbance_body_id < 0:
        raise ValueError(f"Disturbance body '{config.DISTURBANCE_BODY_NAME}' not found in model")

    max_steps = int(config.SIM_DURATION / dt)

    # ── Logging arrays ────────────────────────────────────────────────
    log_t             = np.zeros(max_steps)
    log_pos           = np.zeros((max_steps, 3))
    log_sigma         = np.zeros(max_steps)
    log_force         = np.zeros(max_steps)
    log_dist          = np.zeros(max_steps)
    log_disturbance   = np.zeros(max_steps, dtype=bool)
    log_tank_s        = np.zeros(max_steps)
    log_tank_alpha    = np.zeros(max_steps)
    log_beta_t        = np.zeros(max_steps)
    log_beta_n        = np.zeros(max_steps)
    log_beta_t_prime  = np.zeros(max_steps)
    log_beta_n_prime  = np.zeros(max_steps)
    log_p_t           = np.zeros(max_steps)
    log_p_n           = np.zeros(max_steps)
    log_p_d           = np.zeros(max_steps)
    log_motion_norm   = np.zeros(max_steps)
    log_force_norm    = np.zeros(max_steps)
    log_vd_norm       = np.zeros(max_steps)
    log_F_des         = np.zeros(max_steps)
    log_contact_num   = np.zeros(max_steps)

    _contact_to_num = {"far": 0, "preload": 1, "contact": 2}

    use_tank = getattr(config, "CTRL_USE_ENERGY_TANK", True)
    print(f"Starting demo: {config.SIM_DURATION:.0f} s  |  dt={dt*1000:.1f} ms")
    print(f"Sphere: center={config.SPHERE_CENTER}, R={config.SPHERE_RADIUS} m")
    print(f"Target force: {config.FORCE_DESIRED} N  |  circle r={config.DS_R_CIRCLE} m")
    print(f"Energy tank: {'ON' if use_tank else 'OFF'}  "
          f"s_max={ctrl.tank_s_max:.1f} J")
    if config.DISTURBANCE_ENABLE:
        print(f"Disturbance: body='{config.DISTURBANCE_BODY_NAME}'  "
              f"F={config.DISTURBANCE_FORCE} N  "
              f"t=[{config.DISTURBANCE_START_TIME}, "
              f"{config.DISTURBANCE_START_TIME + config.DISTURBANCE_DURATION}] s")
    print("Close the viewer window to abort early.\n")
    print(f"{'t':>6}  {'s':>6}  {'β_t′':>6}  {'β_n′':>6}  "
          f"{'p_t':>8}  {'p_n':>8}  {'p_d':>8}  state")
    print("-" * 70)

    # ── Trail & arrow visualization constants ─────────────────────────
    _TRAIL_MAXLEN          = 2000
    _TRAIL_STRIDE          = 10
    _TRAIL_RADIUS          = 0.003
    _FORCE_ARROW_SCALE     = 0.008
    _VEL_ARROW_SCALE       = 0.3
    _ARROW_THICKNESS       = 0.005
    _trail = deque(maxlen=_TRAIL_MAXLEN)

    # ── Energy-tank bar constants ─────────────────────────────────────
    _TANK_BAR_ORIGIN = np.array([0.50, 0.32, 0.10])
    _TANK_BAR_MAX_H  = 0.22
    _TANK_BAR_HALF_W = 0.010

    tank_info = {}
    _print_interval = max(1, int(1.0 / dt))

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
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

            v_d, n, sigma, _, _, _ = ctrl.step()
            tank_info = ctrl.last_tank_info

            active = apply_external_disturbance(env, disturbance_body_id, sim_time)
            if active and not _was_disturbed:
                print(f"  [t={sim_time:.2f}s] Disturbance ON  "
                      f"body='{config.DISTURBANCE_BODY_NAME}'  "
                      f"F={config.DISTURBANCE_FORCE} N")
            elif not active and _was_disturbed:
                print(f"  [t={sim_time:.2f}s] Disturbance OFF")
            _was_disturbed = active

            if step % _print_interval == 0:
                ti = tank_info
                print(
                    f"{sim_time:6.1f}  "
                    f"{ti.get('tank_s', 0.0):6.3f}  "
                    f"{ti.get('beta_t_prime', 1.0):6.3f}  "
                    f"{ti.get('beta_n_prime', 1.0):6.3f}  "
                    f"{ti.get('p_t', 0.0):8.3f}  "
                    f"{ti.get('p_n', 0.0):8.3f}  "
                    f"{ti.get('p_d', 0.0):8.3f}  "
                    f"{ti.get('contact_state', '?')}"
                )

            env.step()

            ee = env.ee_pos()
            _trail.append(ee.copy())

            viewer.user_scn.ngeom = 0
            _max_g = viewer.user_scn.maxgeom

            # Trail (green spheres)
            pts = list(_trail)[::_TRAIL_STRIDE]
            n_pts = len(pts)
            for i, p in enumerate(pts):
                if viewer.user_scn.ngeom >= _max_g - 5:
                    break
                age = i / max(n_pts - 1, 1)
                alpha_c = float(age ** 1.5)
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([_TRAIL_RADIUS, 0.0, 0.0]),
                    p, np.eye(3).flatten(),
                    np.array([0.2, 1.0, 0.2, alpha_c], dtype=np.float32),
                )
                viewer.user_scn.ngeom += 1

            # Disturbance force arrow (orange)
            if active and viewer.user_scn.ngeom < _max_g:
                body_com = env.data.xpos[disturbance_body_id].copy()
                tip = body_com + config.DISTURBANCE_FORCE * _FORCE_ARROW_SCALE
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_connector(
                    g, mujoco.mjtGeom.mjGEOM_ARROW, 0.01, body_com, tip,
                )
                g.rgba[:] = np.array([1.0, 0.35, 0.0, 0.9], dtype=np.float32)
                viewer.user_scn.ngeom += 1

            # Motion term arrow (blue)
            motion_vec = tank_info.get("motion_vec")
            if motion_vec is not None and np.linalg.norm(motion_vec) > 1e-6:
                if viewer.user_scn.ngeom < _max_g:
                    tip = ee + _VEL_ARROW_SCALE * motion_vec
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_connector(
                        g, mujoco.mjtGeom.mjGEOM_ARROW,
                        _ARROW_THICKNESS, ee, tip,
                    )
                    g.rgba[:] = np.array([0.1, 0.4, 1.0, 0.9], dtype=np.float32)
                    viewer.user_scn.ngeom += 1

            # Force term arrow (red)
            force_vec = tank_info.get("force_vec")
            if force_vec is not None and np.linalg.norm(force_vec) > 1e-6:
                if viewer.user_scn.ngeom < _max_g:
                    tip = ee + _VEL_ARROW_SCALE * force_vec
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_connector(
                        g, mujoco.mjtGeom.mjGEOM_ARROW,
                        _ARROW_THICKNESS, ee, tip,
                    )
                    g.rgba[:] = np.array([1.0, 0.1, 0.1, 0.9], dtype=np.float32)
                    viewer.user_scn.ngeom += 1

            # Corrected v_d arrow (green)
            vd_vec = tank_info.get("vd_vec")
            if vd_vec is not None and np.linalg.norm(vd_vec) > 1e-6:
                if viewer.user_scn.ngeom < _max_g:
                    tip = ee + _VEL_ARROW_SCALE * vd_vec
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_connector(
                        g, mujoco.mjtGeom.mjGEOM_ARROW,
                        _ARROW_THICKNESS, ee, tip,
                    )
                    g.rgba[:] = np.array([0.1, 0.9, 0.1, 0.9], dtype=np.float32)
                    viewer.user_scn.ngeom += 1

            # Energy-tank level bar
            _tank_frac = np.clip(
                tank_info.get("tank_s", ctrl.tank_s0) / ctrl.tank_s_max,
                0.0, 1.0,
            )

            if viewer.user_scn.ngeom < _max_g:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                _bg_hh = _TANK_BAR_MAX_H / 2
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_BOX,
                    np.array([_TANK_BAR_HALF_W * 1.5,
                               _TANK_BAR_HALF_W * 1.5,
                               _bg_hh]),
                    _TANK_BAR_ORIGIN + np.array([0.0, 0.0, _bg_hh]),
                    np.eye(3).flatten(),
                    np.array([0.25, 0.25, 0.25, 0.75], dtype=np.float32),
                )
                viewer.user_scn.ngeom += 1

            if viewer.user_scn.ngeom < _max_g:
                _fill_hh = max(_tank_frac * _TANK_BAR_MAX_H / 2, 5e-4)
                _tf = _tank_frac
                _bar_r = 1.0 if _tf < 0.5 else 2.0 * (1.0 - _tf)
                _bar_g = 2.0 * _tf if _tf < 0.5 else 1.0
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_BOX,
                    np.array([_TANK_BAR_HALF_W, _TANK_BAR_HALF_W, _fill_hh]),
                    _TANK_BAR_ORIGIN + np.array([0.0, 0.0, _fill_hh]),
                    np.eye(3).flatten(),
                    np.array([_bar_r, _bar_g, 0.0, 0.95], dtype=np.float32),
                )
                viewer.user_scn.ngeom += 1

            if ctrl.tank_delta > 0 and viewer.user_scn.ngeom < _max_g:
                _soft_frac = (ctrl.tank_s_max - ctrl.tank_delta) / ctrl.tank_s_max
                _band_z = _TANK_BAR_ORIGIN[2] + _soft_frac * _TANK_BAR_MAX_H
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_BOX,
                    np.array([_TANK_BAR_HALF_W * 2.2,
                               _TANK_BAR_HALF_W * 2.2,
                               0.0015]),
                    np.array([_TANK_BAR_ORIGIN[0], _TANK_BAR_ORIGIN[1], _band_z]),
                    np.eye(3).flatten(),
                    np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
                )
                viewer.user_scn.ngeom += 1

            if viewer.user_scn.ngeom < _max_g:
                _cap_z = _TANK_BAR_ORIGIN[2] + _tank_frac * _TANK_BAR_MAX_H
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([_TANK_BAR_HALF_W * 1.8, 0.0, 0.0]),
                    np.array([_TANK_BAR_ORIGIN[0], _TANK_BAR_ORIGIN[1], _cap_z]),
                    np.eye(3).flatten(),
                    np.array([1.0, 1.0, 1.0, 0.95], dtype=np.float32),
                )
                viewer.user_scn.ngeom += 1

            viewer.sync()

            # Log
            F_n = env.contact_normal_force()
            d = sphere.signed_dist(ee, config.TOOL_RADIUS)

            log_t[step]            = sim_time
            log_pos[step]          = ee
            log_sigma[step]        = sigma
            log_force[step]        = F_n
            log_dist[step]         = d
            log_disturbance[step]  = active
            log_tank_s[step]       = tank_info.get("tank_s", 0.0)
            log_tank_alpha[step]   = tank_info.get("alpha", 1.0)
            log_beta_t[step]       = tank_info.get("beta_t", 1.0)
            log_beta_n[step]       = tank_info.get("beta_n", 1.0)
            log_beta_t_prime[step] = tank_info.get("beta_t_prime", 1.0)
            log_beta_n_prime[step] = tank_info.get("beta_n_prime", 1.0)
            log_p_t[step]          = tank_info.get("p_t", 0.0)
            log_p_n[step]          = tank_info.get("p_n", 0.0)
            log_p_d[step]          = tank_info.get("p_d", 0.0)
            log_motion_norm[step]  = tank_info.get("motion_norm", 0.0)
            log_force_norm[step]   = tank_info.get("force_norm", 0.0)
            log_vd_norm[step]      = tank_info.get("vd_norm", 0.0)
            log_F_des[step]        = tank_info.get("F_des_normal", 0.0)
            log_contact_num[step]  = _contact_to_num.get(
                tank_info.get("contact_state", "far"), 0
            )

            elapsed = time.time() - step_wall
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

            step += 1

    steps_done = step
    wall_time = time.time() - t_start
    print(f"\nDone: {steps_done} steps in {wall_time:.1f} s wall-time.")

    # ── Post-simulation plots ─────────────────────────────────────────
    t            = log_t[:steps_done]
    pos          = log_pos[:steps_done]
    sigma_log    = log_sigma[:steps_done]
    force_log    = log_force[:steps_done]
    dist_log     = log_dist[:steps_done]
    dist_log_arr = log_disturbance[:steps_done]

    tank_logs = {
        "tank_s":       log_tank_s[:steps_done],
        "alpha":        log_tank_alpha[:steps_done],
        "beta_t":       log_beta_t[:steps_done],
        "beta_n":       log_beta_n[:steps_done],
        "beta_t_prime": log_beta_t_prime[:steps_done],
        "beta_n_prime": log_beta_n_prime[:steps_done],
        "p_t":          log_p_t[:steps_done],
        "p_n":          log_p_n[:steps_done],
        "p_d":          log_p_d[:steps_done],
        "motion_norm":  log_motion_norm[:steps_done],
        "force_norm":   log_force_norm[:steps_done],
        "vd_norm":      log_vd_norm[:steps_done],
        "F_des":        log_F_des[:steps_done],
        "contact_num":  log_contact_num[:steps_done],
    }

    _plot_results(t, pos, sigma_log, force_log, dist_log,
                  config.SPHERE_CENTER, config.SPHERE_RADIUS,
                  config.FORCE_DESIRED, dist_log_arr)

    _plot_tank_results(t, force_log, tank_logs,
                       config.FORCE_DESIRED,
                       ctrl.tank_s_max,
                       dist_log_arr)


# ──────────────────────────────────────────────────────────────────────
def _shade_disturbance(ax, t, disturbance):
    if disturbance is None or not disturbance.any():
        return
    idx = np.where(disturbance)[0]
    d_t0, d_t1 = t[idx[0]], t[idx[-1]]
    ax.axvspan(d_t0, d_t1, color="orange", alpha=0.15, label="disturbance")


# ──────────────────────────────────────────────────────────────────────
def _plot_results(t, pos, sigma, force, dist, c, R, F_d, disturbance=None):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("DS-Based Surface Polishing Demo", fontsize=14, fontweight="bold")

    def _shade(ax):
        _shade_disturbance(ax, t, disturbance)

    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    _draw_sphere(ax3d, c, R)
    sc = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=t, cmap="plasma", s=1, zorder=5)
    plt.colorbar(sc, ax=ax3d, label="time [s]", shrink=0.6)
    ax3d.set_title("EE Trajectory on Sphere")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(pos[:, 0], pos[:, 1], c=t, cmap="plasma", s=1)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(c[0] + R * np.cos(theta), c[1] + R * np.sin(theta),
             "k--", lw=0.8, label="sphere equator")
    ax2.set_aspect("equal"); ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    ax2.set_title("EE Trajectory (top view)"); ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, dist * 1e3, lw=0.8)
    ax3.axhline(0, color="r", ls="--", lw=0.8, label="surface")
    _shade(ax3); ax3.legend()
    ax3.set_xlabel("time [s]"); ax3.set_ylabel("distance [mm]")
    ax3.set_title("Signed Surface Distance (+ = above)")

    ax4 = fig.add_subplot(2, 3, 4)
    lp_window = max(1, int(0.01 / (t[1] - t[0]))) if len(t) > 1 else 1
    force_filt = uniform_filter1d(force, size=lp_window)
    ax4.plot(t, force, lw=0.4, alpha=0.3, color="tab:blue", label="Raw $F_n$")
    ax4.plot(t, force_filt, lw=1.2, color="tab:blue", label="Filtered $F_n$")
    ax4.axhline(F_d, color="r", ls="--", lw=1.2, label=f"F_d = {F_d} N")
    _shade(ax4); ax4.legend()
    ax4.set_xlabel("time [s]"); ax4.set_ylabel("Force [N]")
    ax4.set_title("Normal Contact Force")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, sigma, lw=0.8, color="tab:orange")
    ax5.set_ylim(-0.05, 1.05)
    _shade(ax5)
    ax5.set_xlabel("time [s]"); ax5.set_ylabel("σ")
    ax5.set_title("Contact Blend Weight (0=reach, 1=polish)")

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


# ──────────────────────────────────────────────────────────────────────
def _plot_tank_results(t, force_log, tl, F_d, tank_s_max, disturbance=None):
    lp_window = max(1, int(0.01 / (t[1] - t[0]))) if len(t) > 1 else 1

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Energy Tank Analysis", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    def _shade(ax):
        _shade_disturbance(ax, t, disturbance)

    ax = axes[0]
    ax.plot(t, tl["tank_s"], lw=1.0, color="tab:purple", label="$s$ (tank energy)")
    ax.axhline(0,          color="k",   ls="--", lw=0.8)
    ax.axhline(tank_s_max, color="tab:red", ls="--", lw=0.8, label="$s_{\\max}$")
    ax.fill_between(t, tl["tank_s"], 0, alpha=0.15, color="tab:purple")
    _shade(ax)
    ax.legend(fontsize=8)
    ax.set_xlabel("time [s]"); ax.set_ylabel("energy [J]")
    ax.set_title("A. Tank Energy")
    ax.set_ylim(bottom=-0.1)

    ax = axes[1]
    ax.plot(t, tl["beta_t_prime"], lw=1.2, color="tab:blue",   label="$\\beta_t'$ (motion)")
    ax.plot(t, tl["beta_n_prime"], lw=1.2, color="tab:red",    label="$\\beta_n'$ (force)")
    ax.plot(t, tl["beta_t"],       lw=0.7, color="tab:blue",   ls="--", label="$\\beta_t$")
    ax.plot(t, tl["beta_n"],       lw=0.7, color="tab:red",    ls="--", label="$\\beta_n$")
    _shade(ax)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8)
    ax.set_xlabel("time [s]"); ax.set_ylabel("scaling")
    ax.set_title("B. Beta Scaling (1=active, 0=blocked)")

    ax = axes[2]
    ax.plot(t, tl["p_t"], lw=0.8, color="tab:blue",   label="$p_t$ (motion power)")
    ax.plot(t, tl["p_n"], lw=0.8, color="tab:red",    label="$p_n$ (force power)")
    ax.plot(t, tl["p_d"], lw=0.8, color="tab:green",  label="$p_d$ (dissipation)")
    ax.axhline(0, color="k", lw=0.5)
    _shade(ax)
    ax.legend(fontsize=8)
    ax.set_xlabel("time [s]"); ax.set_ylabel("power [W]")
    ax.set_title("C. Power Terms")

    ax = axes[3]
    force_filt = uniform_filter1d(force_log, size=lp_window)
    ax.plot(t, force_log,    lw=0.4, alpha=0.3, color="tab:blue", label="$F_n$ raw")
    ax.plot(t, force_filt,   lw=1.2, color="tab:blue",            label="$F_n$ filtered")
    ax.plot(t, tl["F_des"],  lw=1.2, color="tab:orange",          label="$F_{des}$ ramped")
    ax.axhline(F_d,          color="r", ls="--", lw=1.2,          label=f"$F_d$ = {F_d} N")
    _shade(ax)
    ax.legend(fontsize=8)
    ax.set_xlabel("time [s]"); ax.set_ylabel("Force [N]")
    ax.set_title("D. Force Tracking")

    ax = axes[4]
    ax.plot(t, tl["motion_norm"], lw=1.0, color="tab:blue",   label="$||$motion$||$")
    ax.plot(t, tl["force_norm"],  lw=1.0, color="tab:red",    label="$||$force$||$")
    ax.plot(t, tl["vd_norm"],     lw=1.0, color="tab:green",  label="$||v_d||$")
    _shade(ax)
    ax.legend(fontsize=8)
    ax.set_xlabel("time [s]"); ax.set_ylabel("speed [m/s]")
    ax.set_title("E. Velocity Component Norms")

    ax = axes[5]
    ax_r = ax.twinx()
    ax.plot(t, tl["contact_num"], lw=1.0, color="tab:orange",
            label="contact state (0=far,1=pre,2=cont)")
    ax_r.plot(t, np.clip(tl["alpha"], 0, 1), lw=0.8, color="tab:purple",
              ls="--", label="alpha(s)")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["far", "preload", "contact"])
    ax.set_ylim(-0.3, 2.5)
    ax_r.set_ylim(-0.05, 1.15)
    ax_r.set_ylabel("alpha")
    _shade(ax)
    ax.legend(loc="upper left", fontsize=8)
    ax_r.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("time [s]")
    ax.set_title("F. Contact State & Alpha")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "tank_results.png")
    plt.savefig(out_path, dpi=150)
    print(f"Tank plots saved to {out_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────
def _draw_sphere(ax, c, R, alpha=0.15):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, 3 * np.pi / 4, 20)
    x = c[0] + R * np.outer(np.cos(u), np.sin(v))
    y = c[1] + R * np.outer(np.sin(u), np.sin(v))
    z = c[2] + R * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, color="steelblue")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
