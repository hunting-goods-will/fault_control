"""Paired comparison of baseline vs rate_limited trials.

The required check before trusting the rate-limited controller's results: confirms
peak overshoot is lower under rate limiting, and settling time is shorter or
comparable, across the 64 paired trials. Pure numpy -- no Isaac Sim needed, so
this can run anywhere, including outside the isaaclab environment.

Usage:
    python scripts/validate_controller.py
"""

import numpy as np

MAX_GRAVITY_TORQUE = 0.736  # N*m, from Day 2 (m*g*(L/2)); theta_eq = arcsin(tau / this)
SETTLING_TOLERANCE = 0.02  # 2% relative band around equilibrium -- standard controls convention
COMPARABLE_SLACK = 1.05  # settling time counts as "comparable" if within 5% of baseline


def settling_time_idx(joint_pos: np.ndarray, theta_eq: float, tol: float = SETTLING_TOLERANCE) -> int:
    """Index of the first step after which joint_pos stays within tol*|theta_eq| for good.

    Scans from the end: finds the last index still outside the band, settling
    time is one step after that. Returns 0 if it's within the band the whole time.
    """
    band = tol * abs(theta_eq)
    outside = np.abs(joint_pos - theta_eq) > band
    if not outside.any():
        return 0
    return int(np.where(outside)[0][-1]) + 1


def peak_overshoot(joint_pos: np.ndarray, theta_eq: float) -> float:
    """Max excursion past theta_eq in the direction of approach. 0.0 if it never overshoots."""
    if theta_eq >= 0:
        return max(0.0, float(joint_pos.max()) - theta_eq)
    return max(0.0, theta_eq - float(joint_pos.min()))


def correlate_with_fault_torque(fault_torque: np.ndarray, values: np.ndarray, label: str, n_bins: int = 4) -> None:
    """Prints Pearson r, a linear fit slope, and a quartile-binned mean/std breakdown of
    `values` (e.g. per-trial % change) against fault_torque. numpy-only, no scipy.

    Quartile means matter more than r alone at n=64 -- r can look unremarkable while a
    binned breakdown still shows a clean monotonic trend, or vice versa (r inflated by a
    couple outlier trials). Report both, don't pick whichever looks better.
    """
    r = float(np.corrcoef(fault_torque, values)[0, 1])
    slope, intercept = np.polyfit(fault_torque, values, 1)

    order = np.argsort(fault_torque)
    bin_edges = np.array_split(order, n_bins)
    bin_labels = []
    bin_means = []
    bin_stds = []
    for b in bin_edges:
        lo, hi = fault_torque[b].min(), fault_torque[b].max()
        bin_labels.append(f"{lo:.2f}-{hi:.2f} N*m")
        bin_means.append(values[b].mean())
        bin_stds.append(values[b].std())

    print(f"  {label} vs fault_torque: r={r:.2f}, slope={slope:.1f} %/N*m")
    for lbl, m, s in zip(bin_labels, bin_means, bin_stds):
        print(f"    [{lbl}]  mean {m:+.1f}%  std {s:.1f}%")


def main():
    baseline = np.load("results/raw/baseline_trials.npz")
    rate_limited = np.load("results/raw/rate_limited_trials.npz")

    # paired-trial sanity check -- same seed, same sampled fault torques
    if not np.allclose(baseline["fault_torque"], rate_limited["fault_torque"]):
        raise ValueError(
            "fault_torque arrays differ between the two runs -- these are NOT paired trials. "
            "Check that both were run with --seed 42 and nothing upstream of the fault_torque "
            "sampling changed between runs."
        )

    sim_dt = float(baseline["sim_dt"])
    fault_torque = baseline["fault_torque"]
    theta_eq = np.arcsin(fault_torque / MAX_GRAVITY_TORQUE)
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

    overshoot_better = overshoot_rl < overshoot_base
    settle_ok = settle_rl <= settle_base * COMPARABLE_SLACK
    settle_strictly_faster = settle_rl < settle_base
    settle_comparable_not_faster = settle_ok & ~settle_strictly_faster

    # per-trial percent change -- the distribution Day 4 actually needs, not just the two means
    overshoot_pct_change = 100 * (overshoot_base - overshoot_rl) / overshoot_base
    settle_pct_change = 100 * (settle_base - settle_rl) / settle_base

    print(f"Peak overshoot lower under rate limiting: {overshoot_better.sum()}/{n} trials "
          f"({100 * overshoot_better.mean():.1f}%)")
    print(f"  mean overshoot -- baseline: {overshoot_base.mean():.4f} rad, "
          f"rate_limited: {overshoot_rl.mean():.4f} rad")
    print(f"  per-trial % reduction: mean {overshoot_pct_change.mean():.1f}%, "
          f"std {overshoot_pct_change.std():.1f}%")

    print(f"Settling time shorter or comparable (within {int((COMPARABLE_SLACK - 1) * 100)}%): "
          f"{settle_ok.sum()}/{n} trials ({100 * settle_ok.mean():.1f}%)")
    print(f"  -- of which strictly faster: {settle_strictly_faster.sum()}/{n} "
          f"({100 * settle_strictly_faster.mean():.1f}%)")
    print(f"  -- of which comparable but not faster: {settle_comparable_not_faster.sum()}/{n} "
          f"({100 * settle_comparable_not_faster.mean():.1f}%)")
    print(f"  mean settling time -- baseline: {settle_base.mean():.3f} s, "
          f"rate_limited: {settle_rl.mean():.3f} s")
    print(f"  per-trial % change: mean {settle_pct_change.mean():.1f}%, "
          f"std {settle_pct_change.std():.1f}% (positive = faster)")

    print("\nRelationship to fault magnitude (does the benefit scale with fault_torque?):")
    correlate_with_fault_torque(fault_torque, overshoot_pct_change, "overshoot reduction")
    correlate_with_fault_torque(fault_torque, settle_pct_change, "settling time change")

    if "held" in rate_limited:
        held = rate_limited["held"]
        print(f"Sensor validation held {held.sum()}/{held.size} step-env pairs "
              f"({100 * held.mean():.3f}%)")

    if not (overshoot_better.mean() > 0.5 and settle_ok.mean() > 0.5):
        print(
            "\n[WARNING] The headline claims do not clearly hold across the batch. "
            "Per the Day 3 handoff: don't rationalize this away -- dig into it before committing."
        )


if __name__ == "__main__":
    main()