"""Day 4: comparison plots for the rate-limited vs baseline controller.

Produces one PNG: paired distributions (boxplot) for overshoot and settling
time, plus the two fault-torque-dependence scatter plots that came out of the
Day 3 validation analysis (overshoot benefit and settling benefit both scale
with fault magnitude -- see validate_controller.py output for the numbers).

Reuses settling_time_idx / peak_overshoot / MAX_GRAVITY_TORQUE from
validate_controller.py instead of reimplementing them, so what gets plotted
can't silently drift from what got validated.

Usage:
    python scripts/plot_results.py
(numpy + matplotlib only -- no Isaac Sim needed)
"""

import numpy as np
import matplotlib.pyplot as plt

from validate_controller import MAX_GRAVITY_TORQUE, peak_overshoot, settling_time_idx

OUT_PATH = "results/comparison_plots.png"


def compute_metrics(baseline: dict, rate_limited: dict) -> dict:
    fault_torque = baseline["fault_torque"]
    theta_eq = np.arcsin(fault_torque / MAX_GRAVITY_TORQUE)
    sim_dt = float(baseline["sim_dt"])
    n = len(fault_torque)

    overshoot_base = np.zeros(n)
    overshoot_rl = np.zeros(n)
    settle_base = np.zeros(n)
    settle_rl = np.zeros(n)
    for i in range(n):
        overshoot_base[i] = peak_overshoot(baseline["joint_pos"][i], theta_eq[i])
        overshoot_rl[i] = peak_overshoot(rate_limited["joint_pos"][i], theta_eq[i])
        settle_base[i] = settling_time_idx(baseline["joint_pos"][i], theta_eq[i]) * sim_dt
        settle_rl[i] = settling_time_idx(rate_limited["joint_pos"][i], theta_eq[i]) * sim_dt

    return {
        "fault_torque": fault_torque,
        "overshoot_base": overshoot_base,
        "overshoot_rl": overshoot_rl,
        "settle_base": settle_base,
        "settle_rl": settle_rl,
        "overshoot_pct": 100 * (overshoot_base - overshoot_rl) / overshoot_base,
        "settle_pct": 100 * (settle_base - settle_rl) / settle_base,
    }


def make_plots(m: dict, out_path: str = OUT_PATH) -> None:
    n = len(m["fault_torque"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Panel A: paired overshoot distributions
    ax = axes[0, 0]
    ax.boxplot([m["overshoot_base"], m["overshoot_rl"]], tick_labels=["baseline", "rate-limited"], showmeans=True)
    ax.set_ylabel("Peak overshoot (rad)")
    ax.set_title(f"Peak overshoot, n={n} paired trials")

    # Panel B: paired settling time distributions
    ax = axes[0, 1]
    ax.boxplot([m["settle_base"], m["settle_rl"]], tick_labels=["baseline", "rate-limited"], showmeans=True)
    ax.set_ylabel("Settling time (s, 2% band)")
    ax.set_title(f"Settling time, n={n} paired trials")

    xs = np.linspace(m["fault_torque"].min(), m["fault_torque"].max(), 50)

    # Panel C: overshoot % reduction vs fault torque
    ax = axes[1, 0]
    ax.scatter(m["fault_torque"], m["overshoot_pct"], alpha=0.6)
    slope, intercept = np.polyfit(m["fault_torque"], m["overshoot_pct"], 1)
    ax.plot(xs, slope * xs + intercept, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Fault torque (N*m)")
    ax.set_ylabel("Overshoot reduction (%)")
    ax.set_title("Overshoot benefit vs fault magnitude")

    # Panel D: settling % change vs fault torque
    ax = axes[1, 1]
    ax.scatter(m["fault_torque"], m["settle_pct"], alpha=0.6, color="tab:orange")
    slope, intercept = np.polyfit(m["fault_torque"], m["settle_pct"], 1)
    ax.plot(xs, slope * xs + intercept, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Fault torque (N*m)")
    ax.set_ylabel("Settling time change (%, + = faster)")
    ax.set_title("Settling benefit vs fault magnitude")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[INFO]: Saved comparison plots to {out_path}")


def main():
    baseline = np.load("results/raw/baseline_trials.npz")
    rate_limited = np.load("results/raw/rate_limited_trials.npz")

    if not np.allclose(baseline["fault_torque"], rate_limited["fault_torque"]):
        raise ValueError(
            "fault_torque arrays differ between the two runs -- these are NOT paired trials. "
            "Check that both were run with --seed 42."
        )

    m = compute_metrics(baseline, rate_limited)
    make_plots(m)


if __name__ == "__main__":
    main()