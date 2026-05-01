"""Headless force probe: run 15 s simulation, report contact-force stats."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import config
from sphere_surface import SphereSurface
from ds import PolishingDS
from sim_env import SimEnv
from controller import PolishingController

def probe(d_n=None, d_t=None, F_d=None, duration=15.0, force_ramp_time=None):
    d_n = d_n if d_n is not None else config.CTRL_D_N
    d_t = d_t if d_t is not None else config.CTRL_D_T
    F_d = F_d if F_d is not None else config.FORCE_DESIRED

    sphere = SphereSurface(config.SPHERE_CENTER, config.SPHERE_RADIUS, config.SPHERE_MAX_POLAR)
    env    = SimEnv(config.SCENE_XML)
    ds     = PolishingDS(sphere, r_tool=config.TOOL_RADIUS, v_target=config.DS_V_TARGET,
                         omega=config.DS_OMEGA, r_circle=config.DS_R_CIRCLE,
                         k_limit=config.DS_K_LIMIT, d_blend=config.DS_D_BLEND)
    ctrl   = PolishingController(env, sphere, ds, q_ref=config.Q_INIT,
                                 F_d=F_d, d_n=d_n, d_t=d_t,
                                 k_null=config.CTRL_K_NULL, b_null=config.CTRL_B_NULL,
                                 dls_lambda=config.CTRL_DLS_LAMBDA,
                                 k_force_fb=getattr(config, "CTRL_K_FORCE_FB", 0.0))
    if force_ramp_time is not None:
        ctrl.force_ramp_time = force_ramp_time
    env.reset(config.Q_INIT)
    ctrl.reset()
    env.model.opt.timestep = 0.001
    ctrl.dt = 0.001   # sync controller dt with model

    dt      = env.model.opt.timestep
    n_steps = int(duration / dt)

    forces  = []
    contact_steps = 0

    sigmas  = []
    dists   = []

    for _ in range(n_steps):
        ee = env.ee_pos()
        v_d, n, sigma, state, F_filt, F_meas = ctrl.step()
        env.step()
        fn = env.raw_contact_normal_force()
        d  = sphere.signed_dist(ee, config.TOOL_RADIUS)
        sigmas.append(sigma)
        dists.append(d)
        if fn > 0.1:             # only count steps that are actually in contact
            forces.append(fn)
            contact_steps += 1

    sigmas = np.array(sigmas)
    dists  = np.array(dists)

    if not forces:
        print(f"  d_n={d_n:6.0f}  d_t={d_t:6.0f}  F_d={F_d:.1f}  → NO CONTACT")
        print(f"      sigma: max={sigmas.max():.3f}  dist: min={dists.min()*1e3:.2f}mm")
        return

    forces = np.array(forces)
    mean   = forces.mean()
    std    = forces.std()
    mn, mx = forces.min(), forces.max()
    in_contact = sigmas[sigmas > 0.3]
    ramp_str = f"ramp={ctrl.force_ramp_time:.2f}s"
    print(f"  d_n={d_n:6.0f}  d_t={d_t:5.0f}  {ramp_str}  "
          f"→  mean={mean:5.2f}N  std={std:4.2f}  [{mn:.1f}, {mx:.1f}]")
    return mean, std

if __name__ == "__main__":
    print("=== Final config validation ===")
    probe()   # uses config values: d_n, d_t, F_d, force_ramp_time from config
