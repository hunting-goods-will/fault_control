# Actuator Fault Control: Rate-Limited, Sensor-Validated Torque Control

A single-joint (1-DOF revolute), torque-controlled actuator model built in NVIDIA Isaac Lab
(v2.3.0, Isaac Sim 5.1.0), demonstrating a rate-limited, sensor-validated control response to
a torque fault, and validated by comparison against an uncontrolled baseline across a batch of
randomized trials.

## Overview

Baseline (uncontrolled) and rate-limited (fixed) controllers are each run across a batch of 64
parallel trials with randomized fault-torque magnitude, paired so both controllers see the
identical sampled fault torques (same seed, same sampling call, before either mode branches).
Results are reported as a distribution across the 64 paired trials, not a single run. See
Results below.

## Motivation

Clamping how fast commanded torque can change, and cross-checking
against a simulated sensor reading before applying it generalizes a control pattern from actual
actuator-fault handling in hardware research. The joint model, masses, and all numeric
parameters in this repo are invented for this project, chosen to be physically plausible and
documented as such.

## Method

**Physical model** — a 1-DOF revolute joint, a single slender arm hanging under gravity from a
fixed base:

| Parameter | Value |
|---|---|
| Arm length | 0.3 m |
| Arm radius | 0.02 m |
| Arm mass | 0.5 kg |
| Joint damping | 0.05 N·m·s/rad |
| Max gravity restoring torque (m·g·L/2) | ≈0.736 N·m |
| Fault torque range | 0.2–0.6 N·m, uniform, seed 42 |
| Trials | n = 64, paired across both controllers |
| Trial duration | 5.0 s |

The fault-torque range is bounded by the max gravity restoring torque, above ≈0.736 N·m no
static equilibrium exists and the arm just spins continuously, so 0.2–0.6 N·m keeps every trial
in the swing-and-settle regime the project is about.

**Controllers**, both implemented in `scripts/run_trials.py`:

- **Baseline** (`--mode baseline`): the fault torque is commanded as an instant step, held
  constant for the full trial. Establishes the overshoot/settling problem.
- **Rate-limited, sensor-validated** (`--mode rate_limited`): commanded torque ramps toward the
  fault-torque target at a bounded rate (`MAX_TORQUE_RATE = 2.0 N·m/s`) instead of stepping
  instantly. Each step, the ramped command is checked against the actuator's own applied-torque
  estimate (`ImplicitActuator`'s PD-law readback, which folds in real joint-velocity feedback);
  if they diverge by more than `SENSOR_VALIDATION_THRESHOLD = 0.1 N·m`, the controller holds the
  last accepted command instead of applying the new one. Both constants are derived from the
  system's own physics (natural-frequency estimate and observed peak joint velocity
  respectively) rather than tuned to the outcome. The full derivation is documented inline in
  `run_trials.py`.

**Validation** (`scripts/validate_controller.py`) checks, per paired trial: peak overshoot lower
under rate limiting, settling time (2%-of-equilibrium tolerance band) shorter or comparable, and
reports the per-trial percent-change distribution plus each metric's relationship to fault-torque
magnitude.

## Results

n = 64 paired trials, fault torque 0.2–0.6 N·m uniform, seed 42.

**Peak overshoot**

| | Baseline | Rate-limited |
|---|---|---|
| Mean | 0.300 rad | 0.213 rad |

Per-trial reduction: **22.5% ± 21.7%** (mean ± std across the 64 trials), lower in 64/64 trials.
The benefit scales with fault severity, meaning it is not a flat percentage. It's a continuous,
nonlinear relationship with fault-torque magnitude (Pearson r = 0.90 against fault torque),
climbing from single digits at low fault torque to >55% reduction near the top of the tested
range, with a sharp knee around fault_torque≈0.42 N·m. This was independently confirmed with a
from-scratch numerical reproduction of the exact system (same equations, same control logic,
same timestep) — see Limitations — so it isn't an leftover of one simulation run.

**Settling time** (2%-of-equilibrium tolerance band)

| | Baseline | Rate-limited |
|---|---|---|
| Mean | 2.367 s | 2.293 s |

Per-trial change: 2.7% ± 8.2%, comparable-or-better in 64/64 trials, but only **23/64 (36%)
trials were actually strictly faster**; the remaining 41/64 were flat or slightly slower and
only counted as "comparable" under a 5% tolerance. **This metric has a confirmed discontinuity
for underdamped systems** (see Limitations) and should be treated as noisier and less trustworthy
than the overshoot result. The apparent large jump in benefit near fault_torque≈0.47–0.6 N·m is
not strong evidence of a larger controller benefit in that range.

**Sensor validation**: held (rejected) 1,166 / 38,400 step-trial pairs (3.0%). Engaging during
the fast part of the swing specifically (consistent with the actual peak joint velocity measured
in the baseline data, 3.93 rad/s), not constant and not dead code.

**Plots**: `results/comparison_plots.png` — paired boxplots for both metrics, plus fault-torque
scatter plots with linear fits. Note: the settling-time scatter's linear fit is misleading given
the discontinuity described in Limitations; the overshoot fit is a reasonable, if imperfect,
description of a nonlinear relationship.

## Repository Structure

```
fault_control/
├── README.md
├── LICENSE
├── assets/
│   └── single_joint_actuator.usda
├── scripts/
│   ├── build_asset.py           # authors the USD asset (one-off)
│   ├── run_trials.py            # baseline + rate-limited trial batches
│   ├── validate_controller.py   # paired-trial validation + fault-torque correlation
│   └── plot_results.py          # comparison plots (results/comparison_plots.png)
├── configs/                      # unused placeholder -- all config lives as documented
│                                # constants in run_trials.py, kept in one place on purpose
├── src/                           # unused placeholder -- project stayed small enough
│                                # not to need a package structure
└── results/
    ├── comparison_plots.png
    └── raw/                       # gitignored -- baseline_trials.npz, rate_limited_trials.npz
```

## Setup & Running

```bash
# environment: Isaac Lab v2.3.0, Isaac Sim 5.1.0, Python 3.11
cd fault_control
source .venv/bin/activate

# author the asset (already generated and committed; only needed if you modify build_asset.py)
./IsaacLab/isaaclab.sh -p scripts/build_asset.py

# run both trial batches (paired via the same --seed)
./IsaacLab/isaaclab.sh -p scripts/run_trials.py --mode baseline --num_envs 64 --seed 42
./IsaacLab/isaaclab.sh -p scripts/run_trials.py --mode rate_limited --num_envs 64 --seed 42

# validate + plot -- numpy/matplotlib only, no Isaac Sim needed for these two
python scripts/validate_controller.py
python scripts/plot_results.py
```

## Limitations

- All physical parameters (arm dimensions, mass, joint damping, fault-torque range) are invented
  for this project -- chosen to be physically plausible, with the reasoning documented inline in
  `run_trials.py`, and not derived from or reused from any real hardware or lab data.
- `MAX_TORQUE_RATE` and `SENSOR_VALIDATION_THRESHOLD` are stated engineering judgment calls,
  derived from the system's own physics (a natural-frequency estimate and the observed peak
  joint velocity) rather than empirically tuned to produce better-looking results -- but they are
  estimates, not measured or optimized constants.
- n = 64, single random seed. The fault-torque range is well-sampled but this is one trial batch,
  not a statistical guarantee across seeds.
- **The settling-time metric has a real, confirmed discontinuity.** "Settling time" here is the
  last time the trajectory exits a 2%-of-equilibrium tolerance band. For a lightly damped
  oscillator (damping ratio ~=0.24-0.32 in this system), that is a discrete function of a
  continuous decay envelope: the number of oscillation cycles needed to settle changes in integer
  steps as fault torque varies smoothly, and the two controllers cross those integer boundaries
  at slightly different fault-torque values. This produces the apparent large settling-time
  "jump" seen near fault_torque~=0.47-0.6 N*m -- confirmed by an independent from-scratch
  numerical reproduction of the exact system (same governing equation, same control logic, same
  timestep), which reproduces the same discontinuity and shows it tracks a discrete change in
  oscillation-cycle count, not a smooth physical improvement. The overshoot metric does not share
  this problem: it's genuinely continuous (confirmed at high-resolution sampling), and its own
  nonlinearity (a sharp knee around fault_torque~=0.42 N*m) appears to be real physics -- most
  plausibly large-angle pendulum nonlinearity (sin(theta) diverging from theta as the equilibrium
  angle grows) -- though the exact mechanism wasn't rigorously isolated beyond ruling out a
  peak-switching artifact as the cause.
- Getting the simulation itself correct took substantial debugging before any control work
  started, worth naming rather than glossing over: the articulation was initially spawning at
  the ground plane, which made a ground-contact interaction look like incorrect pendulum physics
  until it was caught via a viewport screenshot; a missing `DriveAPI` on the hand-authored USD
  joint silently no-opped every torque command; an early Isaac Lab git tag reference turned out
  to be invented (hallucinated from a pip version string) rather than a real release tag. None of
  this is hidden — it's part of how the project actually came together.

## License

MIT -- see [LICENSE](LICENSE).