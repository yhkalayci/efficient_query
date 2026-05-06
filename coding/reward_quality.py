"""
Demonstrate that higher reward implies higher probability of correctness, the
core assumption underlying any reward-guided meta-generation policy.

Produces five diagnostic figures and a printed summary:

  plot_reward_distributions.png:
    Histograms of reward by ground-truth class (correct vs. wrong), pooled
    across all (problem, sample) pairs. Visible separation = signal.

  plot_calibration.png:
    Reward bins on x, empirical correctness rate on y. Monotonically increasing
    = "higher reward => higher correctness" holds in calibrated form.

  plot_topk_correctness.png:
    For each k, plot the empirical probability that the top-k-by-reward (within
    a problem) contains a correct sample, averaged over (problem, permutation).
    This is exactly the quantity the meta-generation algorithm relies on.

  plot_per_problem_auc_hist.png:
    Histogram of per-problem AUCs. Shows whether the reward signal is
    consistently strong across problems, or driven by a few outliers.

  plot_rank_vs_correct.png:
    For each rank position (1=highest reward, N=lowest), the empirical
    correctness rate at that rank, averaged over problems. The leftmost values
    being highest means rank ordering is informative.

Usage:
  python reward_diagnostics.py \\
    --generations run/generations_verified.jsonl \\
    --rewards    run/rewards.jsonl \\
    --reward-key r_score \\
    --out-dir    run/reward_diagnostics
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def auc_safe(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def spearman(x, y):
    """Spearman rank correlation; robust to ties."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--rewards", required=True)
    ap.add_argument("--reward-key", default="r_score",
                    choices=["r_min", "r_mean", "r_last", "r_prod", "r_score"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-bins", type=int, default=15,
                    help="Number of bins for the calibration plot")
    ap.add_argument("--min-correct", type=int, default=1,
                    help="Skip problems with fewer than this many correct samples")
    ap.add_argument("--max-correct-frac", type=float, default=1.0,
                    help="Skip problems with correct fraction above this")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.generations}", flush=True)
    gens = load_jsonl(args.generations)
    print(f"[load] {args.rewards}", flush=True)
    rewards_list = load_jsonl(args.rewards)
    rewards_by_id = {r["id"]: {x["idx"]: x for x in r["rewards"]}
                     for r in rewards_list}

    # Build per-problem (rewards, correct) arrays
    per_problem = {}  # pid -> (rwd_array, cor_array)
    n_skipped_unverified = 0
    n_skipped_filter = 0
    for g in gens:
        pid = g["id"]
        samples = g["samples"]
        n = len(samples)
        if any(s.get("correct") is None for s in samples):
            n_skipped_unverified += 1
            continue
        cor = np.array([int(bool(s["correct"])) for s in samples], dtype=np.int8)
        if cor.sum() < args.min_correct:
            n_skipped_filter += 1
            continue
        if cor.mean() > args.max_correct_frac:
            n_skipped_filter += 1
            continue
        rmap = rewards_by_id.get(pid, {})
        rwd = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            r = rmap.get(i)
            if r is not None and r.get(args.reward_key) is not None:
                rwd[i] = float(r[args.reward_key])
        # Drop samples missing reward
        keep = ~np.isnan(rwd)
        if keep.sum() < 2 or cor[keep].sum() == 0 or cor[keep].sum() == keep.sum():
            n_skipped_filter += 1
            continue
        per_problem[pid] = (rwd[keep], cor[keep])

    print(f"[data] kept {len(per_problem)} problems "
          f"(skipped {n_skipped_unverified} unverified, "
          f"{n_skipped_filter} by filter)", flush=True)
    if not per_problem:
        sys.exit("[fatal] no usable problems")

    # Pooled
    pooled_r = np.concatenate([rwd for rwd, _ in per_problem.values()])
    pooled_c = np.concatenate([cor for _, cor in per_problem.values()])
    print(f"[data] pooled: {len(pooled_r)} samples, "
          f"{int(pooled_c.sum())} correct ({100 * pooled_c.mean():.2f}%)",
          flush=True)

    # ---- Style setup ----
    sns.set_theme(style="whitegrid", context="paper", palette="deep")
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
        "pdf.fonttype": 42,
    })
    COLORS = sns.color_palette("deep").as_hex()

    # ---- Plot 1: reward distributions by class ----
    fig, ax = plt.subplots(figsize=(8, 5))
    r_correct = pooled_r[pooled_c == 1]
    r_wrong = pooled_r[pooled_c == 0]
    lo, hi = float(np.min(pooled_r)), float(np.max(pooled_r))
    bins = np.linspace(lo, hi, 60)
    ax.hist(r_wrong, bins=bins, color=COLORS[3], alpha=0.55,
            label=f"Incorrect (n={len(r_wrong)}, mean={r_wrong.mean():.2f})",
            density=True)
    ax.hist(r_correct, bins=bins, color=COLORS[2], alpha=0.55,
            label=f"Correct (n={len(r_correct)}, mean={r_correct.mean():.2f})",
            density=True)
    ax.axvline(r_wrong.mean(), color=COLORS[3], linestyle="--", alpha=0.7)
    ax.axvline(r_correct.mean(), color=COLORS[2], linestyle="--", alpha=0.7)
    ax.set_xlabel(f"Reward ({args.reward_key})")
    ax.set_ylabel("Density")
    pooled_auc = auc_safe(pooled_c, pooled_r)
    ax.set_title(
        f"Reward distribution by class (pooled across "
        f"{len(per_problem)} problems)\n"
        f"Pooled AUC = {pooled_auc:.3f}, mean separation = "
        f"{r_correct.mean() - r_wrong.mean():.2f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_reward_distributions.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] plot_reward_distributions.pdf", flush=True)

    # ---- Plot 2: calibration (reward bin -> empirical correctness rate) ----
    # Use quantile bins so each bin has roughly equal sample count (cleaner than
    # equal-width bins, which over-weight the dense middle of the reward dist).
    n_bins = args.n_bins
    quantile_edges = np.quantile(pooled_r, np.linspace(0, 1, n_bins + 1))
    # Make edges strictly increasing
    quantile_edges = np.unique(quantile_edges)
    if len(quantile_edges) < 3:
        # Reward is too concentrated for binning; fall back to linear
        quantile_edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(
        np.searchsorted(quantile_edges, pooled_r, side="right") - 1,
        0, len(quantile_edges) - 2,
    )
    bin_centers = 0.5 * (quantile_edges[:-1] + quantile_edges[1:])
    bin_acc = np.zeros(len(bin_centers))
    bin_count = np.zeros(len(bin_centers))
    bin_lo = np.full(len(bin_centers), np.nan)
    bin_hi = np.full(len(bin_centers), np.nan)
    for k in range(len(bin_centers)):
        mask = bin_idx == k
        if mask.sum() == 0:
            continue
        p_hat = pooled_c[mask].mean()
        n_k = int(mask.sum())
        bin_acc[k] = p_hat
        bin_count[k] = n_k
        # Wilson score 95% CI
        z = 1.96
        denom = 1 + z ** 2 / n_k
        center = (p_hat + z ** 2 / (2 * n_k)) / denom
        half = (z * np.sqrt(p_hat * (1 - p_hat) / n_k + z ** 2 / (4 * n_k ** 2)) / denom)
        bin_lo[k] = center - half
        bin_hi[k] = center + half

    fig, ax = plt.subplots(figsize=(8, 5))
    valid = bin_count > 0
    ax.errorbar(bin_centers[valid], bin_acc[valid],
                yerr=[bin_acc[valid] - bin_lo[valid],
                      bin_hi[valid] - bin_acc[valid]],
                fmt="o-", color=COLORS[0], capsize=3,
                label="Empirical correctness rate (95% Wilson CI)")
    ax.axhline(pooled_c.mean(), color="black", linestyle=":", alpha=0.5,
               label=f"Marginal correctness rate = {pooled_c.mean():.3f}")
    ax.set_xlabel(f"Reward bin center ({args.reward_key})")
    ax.set_ylabel("P(correct | reward in bin)")
    spear = spearman(pooled_r, pooled_c)
    ax.set_title(
        f"Calibration: higher reward implies higher correctness probability\n"
        f"Spearman rank correlation = {spear:.3f}, pooled AUC = {pooled_auc:.3f}"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_calibration.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] plot_calibration.pdf", flush=True)

    # ---- Plot 3: top-k correctness ----
    # For each k, P(at least one correct in top-k by reward) averaged over
    # problems. Random baseline is 1 - C(N - n_correct, k) / C(N, k).
    Ns = [len(rwd) for rwd, _ in per_problem.values()]
    N = max(Ns)
    if min(Ns) != max(Ns):
        print(f"[warn] problems have varying N (min={min(Ns)}, max={N}); "
              f"top-k curve is averaged but extrapolated to N={N}", flush=True)

    ks = np.unique(np.round(np.geomspace(1, N, 40)).astype(int))
    topk_succ = np.zeros(len(ks))
    topk_succ_random = np.zeros(len(ks))
    n_problems = len(per_problem)
    for rwd, cor in per_problem.values():
        n = len(rwd)
        # Sort descending by reward
        order = np.argsort(-rwd)
        cor_sorted = cor[order]
        # any-correct in prefix
        cumsum = np.cumsum(cor_sorted)
        any_correct = (cumsum > 0).astype(np.float64)
        # Random baseline: probability a random k-subset contains a correct
        # = 1 - C(n - n_pos, k) / C(n, k)
        n_pos = int(cor.sum())
        for ki, k in enumerate(ks):
            if k > n:
                continue
            topk_succ[ki] += any_correct[k - 1]
            # Compute hypergeometric tail
            # log space for stability
            from math import lgamma
            log_prob_no_correct = (
                lgamma(n - n_pos + 1) - lgamma(n - n_pos - k + 1)
                - (lgamma(n + 1) - lgamma(n - k + 1))
            ) if (n - n_pos - k >= 0) else -np.inf
            topk_succ_random[ki] += 1 - np.exp(log_prob_no_correct)
    topk_succ /= n_problems
    topk_succ_random /= n_problems

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, topk_succ, "o-", color=COLORS[0], linewidth=2,
            label=r"Verify top-$N_{\mathrm{ver}}$ by reward")
    ax.plot(ks, topk_succ_random, "s--", color=COLORS[3], linewidth=1.5,
            alpha=0.7,
            label="Verify random k samples (baseline)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_{\mathrm{ver}}$ (number of samples verified)")
    ax.set_ylabel("P(at least one correct in verified set)")
    ax.set_title(
        f"Top-k vs random-k verification (averaged over {n_problems} problems)\n"
        f"Gap between curves = direct value of reward-based ranking"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_topk_correctness.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] plot_topk_correctness.pdf", flush=True)

    # ---- Plot 4: per-problem AUC histogram ----
    per_problem_aucs = []
    for pid, (rwd, cor) in per_problem.items():
        a = auc_safe(cor, rwd)
        if not np.isnan(a):
            per_problem_aucs.append(a)
    per_problem_aucs = np.asarray(per_problem_aucs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_problem_aucs, bins=20, color=COLORS[4], alpha=0.75,
            edgecolor="black")
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.6,
               label="Chance (AUC = 0.5)")
    ax.axvline(per_problem_aucs.mean(), color=COLORS[3], linestyle="-",
               alpha=0.8, label=f"Mean = {per_problem_aucs.mean():.3f}")
    ax.axvline(np.median(per_problem_aucs), color=COLORS[0], linestyle="-",
               alpha=0.8, label=f"Median = {np.median(per_problem_aucs):.3f}")
    ax.set_xlabel("Per-problem AUC of reward vs. correctness")
    ax.set_ylabel("Number of problems")
    ax.set_title(
        f"Distribution of per-problem AUC across {len(per_problem_aucs)} problems\n"
        f"q25 = {np.percentile(per_problem_aucs, 25):.3f}, "
        f"q75 = {np.percentile(per_problem_aucs, 75):.3f}, "
        f"frac above 0.5 = {(per_problem_aucs > 0.5).mean():.2f}"
    )
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_per_problem_auc_hist.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] plot_per_problem_auc_hist.pdf", flush=True)

    # ---- Plot 5: correctness rate by reward rank ----
    # For each rank position (1 = best reward in problem, N = worst), what
    # fraction of problems have a correct sample at that rank?
    rank_correct = np.zeros(N)
    rank_count = np.zeros(N)
    for rwd, cor in per_problem.values():
        n = len(rwd)
        order = np.argsort(-rwd)
        cor_sorted = cor[order]
        for r in range(n):
            rank_correct[r] += cor_sorted[r]
            rank_count[r] += 1
    rank_acc = np.where(rank_count > 0, rank_correct / np.maximum(rank_count, 1), np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    valid_ranks = rank_count > 0
    ax.plot(np.arange(1, N + 1)[valid_ranks], rank_acc[valid_ranks],
            color=COLORS[0], linewidth=1.8,
            label="P(correct | rank by reward)")
    # Smoothed version with rolling window for visibility
    window = max(5, N // 50)
    smoothed = np.full(N, np.nan)
    for i in range(N):
        lo = max(0, i - window // 2)
        hi = min(N, i + window // 2 + 1)
        if rank_count[lo:hi].sum() > 0:
            smoothed[i] = (
                rank_correct[lo:hi].sum() / rank_count[lo:hi].sum()
            )
    ax.plot(np.arange(1, N + 1), smoothed, color=COLORS[1],
            linewidth=2.5, alpha=0.85,
            label=f"Rolling mean (window={window})")
    ax.axhline(pooled_c.mean(), color="black", linestyle=":", alpha=0.5,
               label=f"Marginal correctness = {pooled_c.mean():.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Rank by reward within a problem (1 = highest reward)")
    ax.set_ylabel("P(sample is correct)")
    ax.set_title(
        f"Correctness rate vs. reward rank, averaged over {n_problems} problems\n"
        f"Monotonically decreasing curve = reward ranks are informative"
    )
    ax.set_ylim(0, 0.6)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_rank_vs_correct.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] plot_rank_vs_correct.pdf", flush=True)

    # ---- Summary text ----
    summary_lines = [
        "=== Reward signal diagnostics ===",
        f"Reward key: {args.reward_key}",
        f"Problems analyzed: {len(per_problem)}",
        f"Pooled samples: {len(pooled_r)} ({int(pooled_c.sum())} correct, "
        f"{100 * pooled_c.mean():.2f}%)",
        "",
        "Pooled signal:",
        f"  AUC                         = {pooled_auc:.4f}",
        f"  Spearman rank correlation   = {spear:.4f}",
        f"  Mean reward (correct)       = {r_correct.mean():.4f}",
        f"  Mean reward (incorrect)     = {r_wrong.mean():.4f}",
        f"  Std reward (correct)        = {r_correct.std():.4f}",
        f"  Std reward (incorrect)      = {r_wrong.std():.4f}",
        f"  Mean separation             = {r_correct.mean() - r_wrong.mean():.4f}",
        "",
        f"Per-problem AUC ({len(per_problem_aucs)} problems with both classes):",
        f"  mean   = {per_problem_aucs.mean():.4f}",
        f"  median = {np.median(per_problem_aucs):.4f}",
        f"  std    = {per_problem_aucs.std():.4f}",
        f"  q25    = {np.percentile(per_problem_aucs, 25):.4f}",
        f"  q75    = {np.percentile(per_problem_aucs, 75):.4f}",
        f"  fraction with AUC > 0.5     = {(per_problem_aucs > 0.5).mean():.4f}",
        f"  fraction with AUC > 0.7     = {(per_problem_aucs > 0.7).mean():.4f}",
        "",
        "Top-k vs random-k advantage at selected k:",
    ]
    for k_target in [1, 5, 10, 50, 100]:
        if k_target > N:
            continue
        ki = int(np.argmin(np.abs(ks - k_target)))
        summary_lines.append(
            f"  k={ks[ki]:3d}: top-k={topk_succ[ki]:.4f}, "
            f"random-k={topk_succ_random[ki]:.4f}, "
            f"advantage={topk_succ[ki] - topk_succ_random[ki]:+.4f}"
        )

    text = "\n".join(summary_lines)
    print("\n" + text, flush=True)
    (out_dir / "summary.txt").write_text(text + "\n")
    print(f"\n[save] summary.txt", flush=True)


if __name__ == "__main__":
    main()