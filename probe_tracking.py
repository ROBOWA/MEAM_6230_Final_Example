"""
Probe force quality AND orbit completion for different d_t / k_limit combos.
Key metric: total revolutions completed in `duration` seconds.
Expected at steady state: ~1 rev / 10 s (omega=pi/5, r=5cm).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import config
from sphere_surface import SphereSurface
from ds import PolishingDS
from sim_env import SimEnv
from controller import PolishingController

CIRCLE_CENTER = np.array(config.SPHERE_CENTER) + np.array([0, 0, config.SPHERE_RADIUS])

def probe(d_t, k_limit, d_n=config.CTRL_D_N, duration=40.0):
    sphere = SphereSurface(config.SPHERE_CENTER, config.SPHERE_RADIUS, config.SPHERE_MAX_POLAR)
    env    = SimEnv(config.SCENE_XML)
    ds     = PolishingDS(sphere, r_tool=config.TOOL_RADIUS, v_target=config.DS_V_TARGET,
                         omega=config.DS_OMEGA, r_circle=config.DS_R_CIRCLE,
                         k_limit=k_limit, d_blend=config.DS_D_BLEND)
    ctrl   = PolishingController(env, sphere, ds, q_ref=config.Q_INIT,
                                 F_d=config.FORCE_DESIRED, d_n=d_n, d_t=d_t,
                                 k_null=config.CTRL_K_NULL, b_null=config.CTRL_B_NULL,
                                 dls_lambda=config.CTRL_DLS_LAMBDA,
                                 force_ramp_time=config.CTRL_FORCE_RAMP_TIME)
    env.reset(config.Q_INIT); ctrl.reset()
    env.model.opt.timestep = 0.001
    ctrl.dt = 0.001

    forces, radii, cum_angle = [], [], 0.0
    prev_ang = None

    for _ in range(int(duration / 0.001)):
        ee = env.ee_pos()
        v_d, n, sigma, _, _, _ = ctrl.step()
        env.step()

        if sigma < 0.45:
            continue

        forces.append(env.raw_contact_normal_force())

        e_tan = (ee - CIRCLE_CENTER)
        e_tan = e_tan - np.dot(e_tan, n) * n
        radii.append(np.linalg.norm(e_tan))

        ang = np.arctan2(e_tan[1], e_tan[0])
        if prev_ang is not None:
            d_ang = (ang - prev_ang + np.pi) % (2 * np.pi) - np.pi
            cum_angle += d_ang
        prev_ang = ang

    if not forces:
        print(f"  d_t={d_t:5.0f}  k_lim={k_limit:.1f}  → NO CONTACT")
        return

    forces = np.array(forces)
    revs   = abs(cum_angle) / (2 * np.pi)
    expected_revs = duration / (2 * np.pi / config.DS_OMEGA)   # ideal orbit period

    print(f"  d_t={d_t:5.0f}  k_lim={k_limit:.1f}  "
          f"| F: {forces.mean():.2f}±{forces.std():.2f}N  "
          f"| r_mean={np.mean(radii)*100:.1f}cm  "
          f"| revs: {revs:.2f}/{expected_revs:.1f} ({revs/expected_revs*100:.0f}%)")

if __name__ == "__main__":
    print(f"F_d={config.FORCE_DESIRED}N  d_n={config.CTRL_D_N}  "
          f"(ideal: {2*np.pi/config.DS_OMEGA:.1f}s/rev)\n")

    print("=== d_n sweep: pressing velocity = F_d/d_n (d_t=600, k_lim=6) ===")
    print(f"  F_d={config.FORCE_DESIRED}N  →  v_press = F_d/d_n:")
    for dn in [400, 300, 200, 150, 100, 50]:
        v_press = config.FORCE_DESIRED / dn
        print(f"    d_n={dn:4.0f}  v_press={v_press*1000:.1f}mm/s", end="  ")
        probe(d_t=600, k_limit=6.0, d_n=dn)

    print("\n=== d_t sweep at d_n=150 ===")
    for dt in [600, 1000, 2000, 4000]:
        probe(d_t=dt, k_limit=6.0, d_n=150)
