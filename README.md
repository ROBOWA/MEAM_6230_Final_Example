# DS-Based Surface Polishing — MuJoCo Simulation
### MEAM 6230 Course Project · University of Pennsylvania

A Python/MuJoCo reproduction of **dynamical-systems (DS)-based contact tasks**,
originally implemented in C++/ROS by Amanhoud, Khoramshahi & Billard (EPFL LASA, RSS 2019).

**Scenario:** Franka Emika Panda polishes the outer surface of a 3/8-sphere
using a passive DS impedance controller with proportional force feedback.

---

## Quick Start

**Requirements:** Python 3.10+, MuJoCo 3.x, NumPy, Matplotlib

```bash
pip install mujoco numpy matplotlib

# Interactive demo (opens MuJoCo viewer + plots)
python run_demo.py

# Headless figure generation for report
python generate_figures.py

# Parameter sweep / re-tuning
python tune_params.py
```

---

## Project Structure

```
MEAM6230/
├── run_demo.py              # Main entry: launches MuJoCo viewer + post-sim plots
├── generate_figures.py      # Headless simulation → PDF/PNG figures
├── config.py                # All tunable parameters (single source of truth)
├── tune_params.py           # Automated parameter sweep (writes best values to config.py)
├── probe_force.py           # Quick 15 s headless force-accuracy check
├── probe_tracking.py        # Orbit-completion diagnostic
│
├── src/
│   ├── sphere_surface.py    # Analytic 3/8-sphere geometry (normal, tangent frame, signed dist)
│   ├── ds.py                # Polishing DS: reaching + circular limit cycle, smoothstep blend
│   ├── controller.py        # Passive DS Impedance controller with force feedback
│   └── sim_env.py           # MuJoCo wrapper (contact force, Jacobian, hold/decay filter)
│
├── assets/
│   └── franka_panda/franka_emika_panda/
│       ├── scene_demo.xml   # Scene: Franka + 3/8 sphere + contact tuning
│       ├── panda_demo.xml   # Franka model: motor actuators + tool sphere
│       └── assets/          # STL/OBJ meshes (MuJoCo Menagerie)
│
└── report/
    ├── report.tex           # IEEE conference LaTeX source
    └── report.pdf           # Compiled PDF
```

---

## Control Architecture

```
Sphere Surface ──► Polishing DS ──► Passive DS Impedance ──► Franka Panda (MuJoCo)
 (normal n, σ)       (v_d)          τ = J^T D(v_d−v_ee)
                                    + g(q) − τ_passive
                                    + N·null-space
        ◄───────────────────── (p_ee, v_ee, q, q̇, F_contact) ◄──────────────
```

### 1. Polishing Dynamical System (`src/ds.py`)

The DS blends two behaviors using a **smoothstep** weight σ ∈ [0,1]:

- **σ = 0** (d ≥ d_blend = 15 mm above surface): pure reaching — `v_reach = −v_target · n`
- **σ = 1** (d ≤ 0, at/below surface): pure circular limit cycle
- In between: smooth cubic interpolation — `v_d = (1−σ)·v_reach + σ·v_circ`

The circular limit cycle in the tangent plane:
```
v_circ = −k_limit·(‖e_tan‖ − r_circle)·ê_tan  +  ω·r_circle·(n × ê_tan)
          ─────────────────────────────────────     ────────────────────────
          radial attraction toward orbit circle      tangential orbit drive
```

### 2. Passive DS Impedance (`src/controller.py`)

**Anisotropic damping** (stiff normal, high tangential for orbit):
```
D = d_t·I + (d_n − d_t)·n·nᵀ
```

**Control torque:**
```
τ = J^T · D · (v_d − v_ee)    ← impedance (drives orbit + force)
  + qfrc_bias                  ← gravity + Coriolis compensation
  − qfrc_passive               ← cancels joint damping (prevents orbit stall)
  + N·(k_null·(q_ref−q) − b_null·q̇)   ← null-space stabilization
```

**Force modulation** — sets desired normal pressing velocity so steady-state force = F_d:
```
v_d_n = F_des_filtered / d_n   +   k_force_fb · (F_d − F_measured)
        ─────────────────────       ─────────────────────────────────
        paper-style modulation      feedback correction for d_t coupling
```

The `k_force_fb` term compensates a systematic artifact: high `d_t` (needed for orbit tracking) couples tangential impedance force into the normal direction via J^T on the curved sphere, inflating the contact force above F_d. The feedback corrects this without reducing d_t.

---

## Key Parameters (`config.py`)

### Geometry
| Parameter | Value | Description |
|-----------|-------|-------------|
| `SPHERE_CENTER` | `[0.5, 0, 0.3]` m | Sphere center in world frame |
| `SPHERE_RADIUS` | `0.15` m | Sphere radius |
| `SPHERE_MAX_POLAR` | `3π/4` (135°) | Polishing region (top 3/8 of sphere) |
| `TOOL_RADIUS` | `0.009` m | Tool sphere radius |

### DS Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `DS_OMEGA` | `π/3` rad/s | Circular angular frequency (orbit period ≈ 6 s) |
| `DS_R_CIRCLE` | `0.05` m | Polishing orbit radius |
| `DS_K_LIMIT` | `6.0` 1/s | Radial attraction gain toward orbit circle |
| `DS_D_BLEND` | `0.015` m | Blend distance (σ: 0→1 over this range) |
| `DS_V_TARGET` | `0.05` m/s | Reaching speed toward surface |

### Force & Impedance
| Parameter | Value | Description |
|-----------|-------|-------------|
| `FORCE_DESIRED` | `15.0` N | Target normal contact force |
| `CTRL_D_N` | `400` N·s/m | Normal damping |
| `CTRL_D_T` | `3000` N·s/m | Tangential damping — primary orbit driver |
| `CTRL_K_FORCE_FB` | `0.032` m/(s·N) | Force feedback gain |
| `CTRL_K_NULL` | `5.0` N·m/rad | Null-space joint stiffness |
| `CTRL_B_NULL` | `5.0` N·m·s/rad | Null-space joint damping |
| `CTRL_DLS_LAMBDA` | `0.02` | DLS Jacobian regularization |
| `CTRL_FORCE_RAMP_TAU` | `0.04` s | Force command low-pass time constant |

### Contact Model (`scene_demo.xml`)
| Attribute | Value | Reason |
|-----------|-------|--------|
| `solref` | `"0.100 1.5"` | tc=0.10 s → b_contact ≈ 26 N/(m/s) (reduces coupling artifact) |
| `solimp` | `"0.8 0.99 0.001"` | Stiff inside margin zone (no bounce) |
| `margin` | `0.002` m | Contact activates 2 mm before surface (no force dropout) |
| `condim` | `1` | Normal force only (no friction) |

---

## Validated Performance

Measured in 40 s headless simulation (40 000 steps at 1 ms):

| Target Force | Mean Force | Force Error | Orbit Completion | Revolutions |
|-------------|-----------|------------|-----------------|-------------|
| F_d = 5 N  | 6.10 ± 2.13 N | **1.10 N** (22%) | **78.4%** | 5.2 / 6.7 |
| F_d = 15 N | 15.87 ± 2.20 N | **0.87 N** (6%) | **74.8%** | 5.0 / 6.7 |

**To change target force:** edit `FORCE_DESIRED` in `config.py`.

---

## Parameter Tuning

`tune_params.py` runs a multi-phase automated sweep. To re-tune:

```bash
python tune_params.py   # sweeps k_force_fb × use_force_feedback at fixed tc/d_t
                        # writes best (CTRL_D_T, CTRL_K_FORCE_FB) to config.py
                        # writes best solref to scene_demo.xml
```

**Key findings from sweeps:**

1. **`solref tc`** (contact time constant) reduces force coupling but has *no effect* on orbit completion. Setting tc=0.10 cuts b_contact from 264 → 26 N/(m/s).

2. **`d_t`** is the sole driver of orbit completion. Orbit% ≈ 1/(1 + c/√d_t), approaching ~85% asymptotically. Null-space gains (k_null, b_null), k_limit, and force feedback do not affect orbit%.

3. **`k_force_fb`** reduces force error by ~60% at any d_t. Goes unstable above k_fb ≈ 0.05 at d_t = 3000. Optimal: 0.032.

4. **`qfrc_passive` cancellation** (in controller) was essential — Franka's joint damping (1 N·m·s/rad per joint) acts as a constant resistive torque that stalls the orbit.

---

## Reference

> W. Amanhoud, M. Khoramshahi, A. Billard,
> *"A Dynamical System Approach to Motion and Force Generation in Contact Tasks,"*
> Robotics: Science and Systems (RSS), 2019.
> [GitHub](https://github.com/epfl-lasa/ds_based_contact_tasks)

---

## Dependencies

- [MuJoCo 3.x](https://mujoco.org/) — physics simulation
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Franka Panda model
- NumPy, Matplotlib — numerics and plotting
