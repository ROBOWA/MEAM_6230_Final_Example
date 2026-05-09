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
from collections import deque

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

    print(f"Starting demo: {config.SIM_DURATION:.0f} s  |  dt={dt*1000:.1f} ms")
    print(f"Sphere: center={config.SPHERE_CENTER}, R={config.SPHERE_RADIUS} m")
    print(f"Target force: {config.FORCE_DESIRED} N  |  circle r={config.DS_R_CIRCLE} m")
    if config.DISTURBANCE_ENABLE:
        print(f"Disturbance: body='{config.DISTURBANCE_BODY_NAME}'  "
              f"F={config.DISTURBANCE_FORCE} N  "
              f"t=[{config.DISTURBANCE_START_TIME}, "
              f"{config.DISTURBANCE_START_TIME + config.DISTURBANCE_DURATION}] s")
    print("Close the viewer window to abort early.\n")

    _TRAIL_MAXLEN      = 2000
    _TRAIL_STRIDE      = 10
    _TRAIL_RADIUS      = 0.003
    _FORCE_ARROW_SCALE = 0.008
    _trail = deque(maxlen=_TRAIL_MAXLEN)

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

            active = apply_external_disturbance(env, disturbance_body_id, sim_time)
            if active and not _was_disturbed:
                print(f"  [t={sim_time:.2f}s] Disturbance ON  "
                      f"body='{config.DISTURBANCE_BODY_NAME}'  "
                      f"F={config.DISTURBANCE_FORCE} N")
            elif not active and _was_disturbed:
                print(f"  [t={sim_time:.2f}s] Disturbance OFF")
            _was_disturbed = active

            env.step()

            ee = env.ee_pos()
            _trail.append(ee.copy())

            viewer.user_scn.ngeom = 0

            pts = list(_trail)[::_TRAIL_STRIDE]
            n_pts = len(pts)
            for i, p in enumerate(pts):
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 2:
                    break
                age = i / max(n_pts - 1, 1)
                alpha = float(age ** 1.5)
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([_TRAIL_RADIUS, 0.0, 0.0]),
                    p, np.eye(3).flatten(),
                    np.array([0.2, 1.0, 0.2, alpha], dtype=np.float32),
                )
                viewer.user_scn.ngeom += 1

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

            elapsed = time.time() - step_wall
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

            step += 1

    wall_time = time.time() - t_start
    print(f"\nDone: {step} steps in {wall_time:.1f} s wall-time.")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
