# Actuator Fault Control: Rate-Limited, Sensor-Validated Torque Control

> Status: in progress. This README is a scaffold — sections below get filled in as the
> project lands (see the `main` branch commit history for how it actually developed).

## Overview

A single-joint (1-DOF revolute) torque-controlled actuator model in [NVIDIA Isaac Lab /
MuJoCo — TBD, see note below], used to demonstrate a rate-limited, sensor-validated
control response to a torque fault.

Baseline (uncontrolled) and fixed (rate-limited) controllers are each run across a batch
of parallel trials with randomized fault magnitude and joint parameters, so results are
reported as a distribution (mean ± std over n trials) rather than a single run.

## Motivation

The type of fix here is clamping how fast commanded torque can change, and cross-checking
against a simulated sensor reading before applying it, generalizes a control pattern from
actuator-fault handling in hardware research. The joint model, masses, and all
numeric parameters in this repo are invented for this project.

## Method

*TBD — filled in once the model and controllers are built (Steps 2–3).*

## Results

*TBD — filled in once the trial batch and comparison are run (Step 4).*

- Peak overshoot: baseline vs. rate-limited (mean ± std, n trials)
- Settling time: baseline vs. rate-limited (mean ± std, n trials)
- Comparison plots: `results/`

## Repository Structure

```
actuator-fault-control-sim/
├── src/            # model + controller implementation
├── configs/        # joint parameters, trial randomization ranges
├── results/         # logs, metrics, plots
├── README.md
└── LICENSE
```

## Setup & Running

*TBD — filled in once the environment is confirmed working (Step 0/1).*

## Limitations

- Synthetic joint model with invented mass/inertia values, not a real system.
- *(Updated as the project develops — e.g. if the Isaac Sim install path didn't pan out
  and this ended up on MuJoCo instead, that'll be stated plainly here, not hidden.)*

## License

MIT — see [LICENSE](LICENSE).
