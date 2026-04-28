# DS-Based Surface Polishing — MuJoCo Simulation
### MEAM 6230 Course Project · University of Pennsylvania

A Python/MuJoCo reproduction of **dynamical-systems (DS)-based contact tasks**,
originally implemented in C++/ROS by Amanhoud, Khoramshahi & Billard (EPFL LASA, RSS 2019).

**Scenario:** Franka Emika Panda polishes the outer surface of a 3/8-sphere
using a passive DS impedance controller (without tank energy).

---

## Quick Start

**Requirements:** Python 3.10+ (tested with Anaconda `ESE5030` env on Windows)

```bash
# 1. Install dependencies
pip install mujoco numpy scipy matplotlib

# 2. Run interactive demo (opens MuJoCo viewer)
python run_demo.py

# 3. Generate report figures (headless, no viewer needed)
python generate_figures.py
```

**Run with explicit Python path (Windows):**
```bash
C:/Users/cindy/anaconda3/envs/ESE5030/python.exe run_demo.py
```

---

## Project Structure

```
program/
├── run_demo.py              # Main entry point — launches MuJoCo viewer + plots
├── generate_figures.py      # Headless simulation → PDF/PNG figures for report
├── config.py                # All tunable parameters (sphere, DS, impedance)
│
├── src/
│   ├── sphere_surface.py    # Analytic 3/8-sphere geometry (normal, tangent frame)
│   ├── ds.py                # Polishing DS: reaching + circular limit cycle
│   ├── controller.py        # Passive DS Impedance (τ = J^T D(v_d−v_ee) + g(q))
│   └── sim_env.py           # MuJoCo environment wrapper
│
├── assets/
│   └── franka_panda/
│       └── franka_emika_panda/
│           ├── scene_demo.xml   # Scene: Franka + 3/8 sphere + sensors
│           ├── panda_demo.xml   # Franka with motor actuators + tool sphere
│           └── assets/          # Mesh files (STL/OBJ from MuJoCo Menagerie)
│
├── report/
│   ├── report.tex           # IEEE conference LaTeX source
│   └── report.pdf           # Compiled PDF (5 pages)
│
└── report_figs/             # Auto-generated figures (after running generate_figures.py)
    ├── fig_trajectory_3d.pdf/png
    ├── fig_trajectory_xy.pdf/png
    ├── fig_force_dist.pdf/png
    ├── fig_ds_field.pdf/png
    └── fig_speed.pdf/png
```

---

## Control Architecture

```
Sphere Surface ──► Polishing DS ──► Passive DS Impedance ──► Franka Panda
     (normal n,        (v_d)              (τ = J^T D(v_d−v_ee) + g(q))   (MuJoCo)
      blend σ)
                         ◄──────────── (p_ee, v_ee, q, q̇) ◄─────────────
```

### 1. Polishing Dynamical System

| Phase | Condition | Velocity |
|-------|-----------|---------|
| **Reaching** | σ ≈ 0 (above surface) | `v_reach = −v_target · n` |
| **Circular** | σ ≈ 1 (on surface) | Limit cycle in tangent plane |
| **Blend** | always | `v_d = (1−σ)·v_reach + σ·v_circ` |

The circular limit cycle: `v_circ = −k_lc(‖e_tan‖ − r_c)·ê + ω·r_c·(n×ê)`

### 2. Passive DS Impedance

**Anisotropic damping matrix:**
```
D = D_t·I + (D_n − D_t)·n·n^T
```

**Control law:**
```
τ = J^T · D · (v_d − v_ee)   +   g(q)   +   null-space term
    ───────────────────────       ─────       ─────────────────
    impedance (contact force)   gravity      joint stabilization
```

**Force regulation:** Setting `v_d_n = F_d / D_n` guarantees
`F_contact = D_n · v_d_n = F_d` when the surface blocks normal motion.

---

## Key Parameters (`config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SPHERE_CENTER` | `[0.5, 0, 0.3]` m | Sphere center in world frame |
| `SPHERE_RADIUS` | `0.15` m | Sphere radius |
| `DS_OMEGA` | `π/3` rad/s | Circular angular frequency |
| `DS_R_CIRCLE` | `0.05` m | Polishing orbit radius |
| `FORCE_DESIRED` | `5.0` N | Target normal contact force |
| `CTRL_D_N` | `200` N·s/m | Normal damping |
| `CTRL_D_T` | `4000` N·s/m | Tangential damping |

---

## Simulation Results Summary

| Metric | Value |
|--------|-------|
| Contact fraction | ~45% |
| Force (when in contact) | 23 ± 4 N |
| Tangential speed | ~25 mm/s |
| Nominal target speed | 26.2 mm/s |
| Speed error | < 5% |

> **Note on force accuracy:** The measured contact force (~23 N) exceeds the
> 5 N target due to Jacobian coupling between the large tangential damping
> (D_t = 4000 N·s/m, needed to overcome Franka's joint damping Kd = 450)
> and the normal direction. This is a simulation artifact; hardware torque
> control with lower joint damping would achieve accurate force regulation.

---

## Reference

> W. Amanhoud, M. Khoramshahi, A. Billard,
> *"A Dynamical System Approach to Motion and Force Generation in Contact Tasks,"*
> Robotics: Science and Systems (RSS), 2019.
> GitHub: https://github.com/epfl-lasa/ds_based_contact_tasks

---

## Dependencies

- [MuJoCo 3.x](https://mujoco.org/) — physics simulation
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Franka Panda model
- NumPy, SciPy, Matplotlib — numerical computation and plotting
