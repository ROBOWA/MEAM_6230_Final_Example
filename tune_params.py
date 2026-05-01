"""
Force-accuracy + orbit-completion sweep — Phase 3.

Strategy: fix tc=0.10, d_n=400, d_t=1000 (best from prior phases) and sweep
  (k_force_fb, use_force_feedback) to maximise both:
  - Orbit completion for both F_d = 5 N and F_d = 15 N
  - Force accuracy |F_mean - F_d|

Key variables:
  k_force_fb   [m/(s·N)] — proportional correction on v_d_n to cancel
                            d_t coupling-induced normal force error.
  use_ff       bool       — when True, contact state flips to "contact"
                            (applying full F_d) only when measured force ≥
                            force_tol, rather than immediately at sigma=0.45.
                            Prevents the aggressive overshoot-bounce that
                            degrades orbit tracking at high F_d.

After the sweep the script writes the best config to config.py and scene_demo.xml.
"""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import mujoco
import config
from sphere_surface import SphereSurface
from ds import PolishingDS
from sim_env import SimEnv
from controller import PolishingController

# ── Fixed parameters ──────────────────────────────────────────────────────────
SOLREF_TC       = 0.10
SOLREF_DAMPRATIO = 1.5
D_T_VAL         = 1000.0
D_N_VAL         = 400.0

# ── Sweep grid ────────────────────────────────────────────────────────────────
K_FB_VALS   = [0.002, 0.004, 0.008, 0.016, 0.032]   # [m/(s·N)]
USE_FF_VALS = [False, True]                           # use force-based contact trigger
F_D_VALS    = [5.0, 15.0]
DURATION    = 40.0
DT_SIM      = 0.001

CIRCLE_CENTER = np.array(config.SPHERE_CENTER) + np.array([0, 0, config.SPHERE_RADIUS])
ORBIT_PERIOD  = 2 * np.pi / config.DS_OMEGA


def _set_sphere_solref(model, tc, dampratio):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "polishing_sphere")
    if geom_id < 0:
        raise RuntimeError("geom 'polishing_sphere' not found")
    model.geom_solref[geom_id, 0] = tc
    model.geom_solref[geom_id, 1] = dampratio


def run_once(k_force_fb, use_force_feedback, F_d, duration=DURATION):
    sphere = SphereSurface(config.SPHERE_CENTER, config.SPHERE_RADIUS,
                           config.SPHERE_MAX_POLAR)
    env    = SimEnv(config.SCENE_XML)
    _set_sphere_solref(env.model, SOLREF_TC, SOLREF_DAMPRATIO)

    ds   = PolishingDS(sphere, r_tool=config.TOOL_RADIUS,
                       v_target=config.DS_V_TARGET,
                       omega=config.DS_OMEGA, r_circle=config.DS_R_CIRCLE,
                       k_limit=config.DS_K_LIMIT, d_blend=config.DS_D_BLEND)
    ctrl = PolishingController(env, sphere, ds, q_ref=config.Q_INIT,
                               F_d=F_d, d_n=D_N_VAL, d_t=D_T_VAL,
                               k_null=config.CTRL_K_NULL,
                               b_null=config.CTRL_B_NULL,
                               dls_lambda=config.CTRL_DLS_LAMBDA,
                               force_ramp_tau=config.CTRL_FORCE_RAMP_TAU,
                               k_force_fb=k_force_fb,
                               use_force_feedback=use_force_feedback)
    env.reset(config.Q_INIT)
    ctrl.reset()
    env.model.opt.timestep = DT_SIM
    ctrl.dt = DT_SIM

    forces, cum_angle, prev_ang = [], 0.0, None

    for _ in range(int(duration / DT_SIM)):
        ee = env.ee_pos()
        v_d, n, sigma, _, _, _ = ctrl.step()
        env.step()

        if sigma < 0.45:
            continue

        fn = env.raw_contact_normal_force()
        if fn > 0.05:
            forces.append(fn)

        e_tan = ee - CIRCLE_CENTER
        e_tan = e_tan - np.dot(e_tan, n) * n
        ang = np.arctan2(e_tan[1], e_tan[0])
        if prev_ang is not None:
            d_ang = (ang - prev_ang + np.pi) % (2 * np.pi) - np.pi
            cum_angle += d_ang
        prev_ang = ang

    if not forces:
        return None, None, None

    forces  = np.array(forces)
    revs    = abs(cum_angle) / (2 * np.pi)
    exp_rev = duration / ORBIT_PERIOD
    return float(forces.mean()), float(forces.std()), revs / exp_rev


# ── Sweep ─────────────────────────────────────────────────────────────────────
print(f"Phase-3 sweep  tc={SOLREF_TC}  d_n={D_N_VAL}  d_t={D_T_VAL}\n"
      f"omega={config.DS_OMEGA:.4f} rad/s  orbit_period={ORBIT_PERIOD:.1f}s\n")

header = (f"{'use_ff':>6}  {'k_fb':>7}  {'F_d':>5}  "
          f"{'F_mean':>7}  {'F_std':>6}  {'F_err':>6}  {'orbit%':>7}")
print(header)
print("-" * len(header))

results = []   # (k_fb, use_ff, F_d, f_mean, f_std, f_err, orbit_pct)

for use_ff in USE_FF_VALS:
    for k_fb in K_FB_VALS:
        row_pair = []
        for F_d in F_D_VALS:
            f_mean, f_std, orbit_pct = run_once(k_fb, use_ff, F_d)
            if f_mean is None:
                print(f"{str(use_ff):>6}  {k_fb:7.4f}  {F_d:5.1f}  NO CONTACT")
                continue
            f_err = abs(f_mean - F_d)
            print(f"{str(use_ff):>6}  {k_fb:7.4f}  {F_d:5.1f}  "
                  f"{f_mean:7.2f}  {f_std:6.2f}  {f_err:6.2f}  "
                  f"{orbit_pct*100:6.1f}%")
            results.append((k_fb, use_ff, F_d, f_mean, f_std, f_err, orbit_pct))
            row_pair.append(f_err)
        if len(row_pair) == 2:
            print()
    print("-" * len(header))


# ── Find best (k_fb, use_ff) ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Selecting best: min F_err sum, orbit >= 80% (relaxing if needed)")
print("=" * 70)

from collections import defaultdict
groups = defaultdict(dict)
for k_fb, use_ff, F_d, f_mean, f_std, f_err, orbit_pct in results:
    groups[(k_fb, use_ff)][F_d] = (f_mean, f_std, f_err, orbit_pct)

best_key   = None
best_score = float("inf")

for threshold in [0.80, 0.70, 0.60, 0.50]:
    if best_key is not None:
        break
    for (k_fb, use_ff), fd_map in groups.items():
        if 5.0 not in fd_map or 15.0 not in fd_map:
            continue
        _, _, err5,  orb5  = fd_map[5.0]
        _, _, err15, orb15 = fd_map[15.0]
        if orb5 < threshold or orb15 < threshold:
            continue
        score = err5 + err15
        status = ">" if score < best_score else " "
        print(f"  {status} k_fb={k_fb:.4f} use_ff={use_ff}: "
              f"F_err_5={err5:.2f}N  F_err_15={err15:.2f}N  "
              f"orbit {orb5*100:.0f}%/{orb15*100:.0f}%  score={score:.2f}")
        if score < best_score:
            best_score = score
            best_key   = (k_fb, use_ff)
    if best_key is None:
        print(f"\n  (no config at {threshold*100:.0f}% orbit — relaxing...)\n")

if best_key is None:
    print("No qualifying config found. Keeping existing values.")
    sys.exit(0)

best_k_fb, best_use_ff = best_key
best_5_err  = groups[best_key][5.0][2]
best_15_err = groups[best_key][15.0][2]
best_5_orb  = groups[best_key][5.0][3]
best_15_orb = groups[best_key][15.0][3]
print(f"\n>>> Best: d_t={D_T_VAL:.0f}  k_force_fb={best_k_fb}  "
      f"use_force_feedback={best_use_ff}")
print(f"    F_d=5:  F_err={best_5_err:.2f}N  orbit={best_5_orb*100:.0f}%")
print(f"    F_d=15: F_err={best_15_err:.2f}N  orbit={best_15_orb*100:.0f}%")

# ── Write best values back to config.py ──────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), "config.py")
with open(config_path, "r") as f:
    text = f.read()

def replace_param(txt, name, value, fmt=".1f"):
    pattern = rf"^({re.escape(name)}\s*=\s*)[\d.e+-]+(.*)$"
    replacement = rf"\g<1>{value:{fmt}}\2"
    new_txt, n = re.subn(pattern, replacement, txt, flags=re.MULTILINE)
    if n == 0:
        print(f"  WARNING: could not patch {name} in config.py")
    return new_txt

text = replace_param(text, "CTRL_D_T", D_T_VAL)
text = replace_param(text, "CTRL_D_N", D_N_VAL)

for name, value, fmt in [
    ("CTRL_K_FORCE_FB", best_k_fb, ".4f"),
]:
    if name in text:
        text = replace_param(text, name, value, fmt)
    else:
        text += f"\nCTRL_K_FORCE_FB = {value:{fmt}}  # force-feedback gain [m/(s·N)]\n"

with open(config_path, "w") as f:
    f.write(text)
print(f"config.py: CTRL_D_T={D_T_VAL:.0f}  CTRL_D_N={D_N_VAL:.0f}  "
      f"CTRL_K_FORCE_FB={best_k_fb:.4f}")

# ── Write solref_tc back to scene_demo.xml ────────────────────────────────────
xml_path = os.path.join(os.path.dirname(__file__),
                        "assets", "franka_panda", "franka_emika_panda", "scene_demo.xml")
with open(xml_path, "r") as f:
    xml = f.read()

old_solref_pat = r'solref="[^"]*"'
new_solref_str = f'solref="{SOLREF_TC:.3f} {SOLREF_DAMPRATIO:.1f}"'
xml_new, n = re.subn(old_solref_pat, new_solref_str, xml)
if n == 0:
    print("  WARNING: could not patch solref in scene_demo.xml")
else:
    with open(xml_path, "w") as f:
        f.write(xml_new)
    print(f"scene_demo.xml: solref=\"{SOLREF_TC:.3f} {SOLREF_DAMPRATIO:.1f}\"")
