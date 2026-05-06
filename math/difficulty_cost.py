"""
Per-problem analysis: for each problem, draw 100 random permutations of the
sample order, find the cheapest (M, K) non-adaptive strategy that succeeds on
each permutation, and aggregate to get per-problem mean +/- std of (M*, K*).

Then visualize the 2D distribution of these per-problem (M*, K*) means. The
expectation: easy problems cluster at low (M*, K*), hard problems sprawl out
to high M, with intermediate problems forming a spectrum.

Output:
  - mk_per_problem.csv: per-problem M_mean, K_mean, M_std, K_std, cost, etc.
  - plot_mk_scatter.png: log-log scatter, color by difficulty, size by std
  - plot_mk_marginals.png: side-by-side 1D histograms of M_mean and K_mean

Usage:
  python analyze_mk_per_problem.py \\
    --generations run/generations_verified.jsonl \\
    --rewards    run/rewards.jsonl \\
    --reward-key r_score \\
    --c-rew 1.0 --c-ver 10.0 \\
    --n-perm 100 \\
    --out-dir   run/mk_analysis
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def best_mk_for_perm(rewards: np.ndarray, correct: np.ndarray,
                     perm: np.ndarray,
                     c_rew: float, c_ver: float):
    """
    For one permutation, find the (M, K) with smallest cost that succeeds.
    A non-adaptive strategy with parameters (M, K) draws the first M samples
    from `perm`, ranks them by reward (descending), verifies the top-K, and
    succeeds if any of those K is correct.

    We avoid scanning all O(N^2) (M, K) pairs by exploiting structure:
    for each M, the minimum K that succeeds is the rank (1-indexed) of the
    first correct sample among the first M sorted by reward. If there is no
    correct sample in the first M, no K works for this M.

    Returns (M*, K*, cost*) or (None, None, np.inf) if no (M, K) succeeds.
    """
    n = len(perm)
    rewards_filled = np.nan_to_num(rewards, nan=-np.inf)

    # Precompute: for each prefix length M, the indices (within `perm`) sorted
    # by reward descending, restricted to the first M items.
    # Naive O(N^2 log N), but N <= 512 so this is fine for 30 problems x 100 perms.

    best_cost = np.inf
    best_mk = (None, None)
    for M in range(1, n + 1):
        first_M = perm[:M]
        rewards_first_M = rewards_filled[first_M]
        order = np.argsort(-rewards_first_M)  # descending
        ordered_correct = correct[first_M][order]
        # First correct in this ordering = minimum K that succeeds at this M
        any_correct_idx = np.where(ordered_correct == 1)[0]
        if len(any_correct_idx) == 0:
            continue
        K = int(any_correct_idx[0]) + 1
        cost = M * c_rew + K * c_ver
        if cost < best_cost:
            best_cost = cost
            best_mk = (M, K)

    return best_mk[0], best_mk[1], best_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--rewards", required=True)
    ap.add_argument("--reward-key", default="r_score",
                    choices=["r_min", "r_mean", "r_last", "r_prod", "r_score"])
    ap.add_argument("--c-rew", type=float, default=1.0)
    ap.add_argument("--c-ver", type=float, default=10.0)
    ap.add_argument("--n-perm", type=int, default=100,
                    help="Number of random permutations per problem")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-correct", type=int, default=1,
                    help="Skip problems with fewer than this many correct samples")
    ap.add_argument("--max-correct-frac", type=float, default=1.0,
                    help="Skip problems where correct fraction exceeds this")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] generations: {args.generations}", flush=True)
    gens = load_jsonl(args.generations)
    print(f"[load] rewards: {args.rewards}", flush=True)
    rewards_list = load_jsonl(args.rewards)
    rewards_by_id = {r["id"]: {x["idx"]: x for x in r["rewards"]}
                     for r in rewards_list}

    rng = np.random.default_rng(args.seed)

    rows = []  # one row per problem
    n_skipped = 0
    for gi, g in enumerate(gens):
        pid = g["id"]
        samples = g["samples"]
        n = len(samples)

        # Check verified
        if any(s.get("correct") is None for s in samples):
            sys.exit(f"[fatal] {pid} has unverified samples; run verify first")

        cor = np.array([int(bool(s["correct"])) for s in samples], dtype=np.int8)
        if cor.sum() < args.min_correct:
            n_skipped += 1
            continue
        if cor.mean() > args.max_correct_frac:
            n_skipped += 1
            continue

        rmap = rewards_by_id.get(pid, {})
        rwd = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            r = rmap.get(i)
            if r is not None and r.get(args.reward_key) is not None:
                rwd[i] = float(r[args.reward_key])

        # Run n_perm random permutations and collect best (M, K) for each
        Ms, Ks, costs = [], [], []
        n_failed = 0
        for _ in range(args.n_perm):
            perm = rng.permutation(n)
            M, K, c = best_mk_for_perm(rwd, cor, perm,
                                       args.c_rew, args.c_ver)
            if M is None:
                n_failed += 1
                continue
            Ms.append(M)
            Ks.append(K)
            costs.append(c)

        Ms = np.array(Ms)
        Ks = np.array(Ks)
        costs = np.array(costs)
        # Defensive: there must always be at least one (M,K) that succeeds since
        # we filtered to >=1 correct. n_failed should be 0.
        if len(Ms) == 0:
            n_skipped += 1
            continue

        diff = g.get("difficulty", "unknown")
        platform = g.get("platform", "unknown")

        rows.append({
            "id": pid,
            "difficulty": diff,
            "platform": platform,
            "n_samples": n,
            "n_correct": int(cor.sum()),
            "pass_rate": float(cor.mean()),
            "M_mean": float(Ms.mean()),
            "K_mean": float(Ks.mean()),
            "M_std": float(Ms.std()),
            "K_std": float(Ks.std()),
            "M_median": float(np.median(Ms)),
            "K_median": float(np.median(Ks)),
            "cost_mean": float(costs.mean()),
            "cost_median": float(np.median(costs)),
        })

        if (gi + 1) % 10 == 0:
            print(f"[analyze] {gi + 1}/{len(gens)} processed "
                  f"(skipped {n_skipped})", flush=True)

    print(f"[analyze] kept {len(rows)} problems, skipped {n_skipped}",
          flush=True)

    if not rows:
        sys.exit("[fatal] no problems passed the filter")

    # ---- Save CSV ----
    import csv
    csv_path = out_dir / "mk_per_problem.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[save] {csv_path}", flush=True)

    # ---- Style setup ----
    sns.set_theme(style="whitegrid", context="paper", palette="deep")
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "pdf.fonttype": 42,
    })
    COLORS = sns.color_palette("deep").as_hex()

    # ---- 2D scatter: M_mean vs K_mean (cleaner, no labels, no borders) ----
    M_means = np.array([r["M_mean"] for r in rows])
    K_means = np.array([r["K_mean"] for r in rows])
    M_stds = np.array([r["M_std"] for r in rows])
    K_stds = np.array([r["K_std"] for r in rows])
    pass_rates = np.array([r["pass_rate"] for r in rows])
    cost_means = np.array([r["cost_mean"] for r in rows])
    difficulties = [r["difficulty"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 7.5))

    diff_colors = {
        "easy": COLORS[2],
        "medium": COLORS[1],
        "hard": COLORS[3],
    }
    has_difficulty = any(d in diff_colors for d in difficulties)

    if has_difficulty:
        for diff_label, color in diff_colors.items():
            mask = np.array([d == diff_label for d in difficulties])
            if not mask.any():
                continue
            spread = np.sqrt(M_stds[mask] ** 2 + K_stds[mask] ** 2)
            sizes = 30 + 5 * spread
            ax.scatter(
                M_means[mask], K_means[mask],
                s=sizes, c=color, alpha=0.7,
                linewidths=0,
                label=f"{diff_label} (n={mask.sum()})",
            )
        mask_other = np.array([d not in diff_colors for d in difficulties])
        if mask_other.any():
            spread = np.sqrt(M_stds[mask_other] ** 2 + K_stds[mask_other] ** 2)
            sizes = 30 + 5 * spread
            ax.scatter(
                M_means[mask_other], K_means[mask_other],
                s=sizes, c="gray", alpha=0.5, linewidths=0,
                label=f"unknown (n={mask_other.sum()})",
            )
        legend_title = "Difficulty"
    else:
        spread = np.sqrt(M_stds ** 2 + K_stds ** 2)
        sizes = 30 + 5 * spread
        sc = ax.scatter(
            M_means, K_means,
            s=sizes, c=pass_rates, cmap="viridis_r",
            alpha=0.75, linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Pass rate (lower = harder)")
        legend_title = None

    # Reference lines
    n_max = max(M_means.max(), K_means.max()) * 1.5
    ax.plot([1, n_max], [1, n_max], "k--", alpha=0.25, lw=1,
            label="K = M (verify everything)")
    ax.plot([1, n_max], [1, 1], "k:", alpha=0.25, lw=1,
            label="K = 1 (verify only top-1)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M* (avg samples drawn before stopping, log scale)")
    ax.set_ylabel("K* (avg samples verified, log scale)")
    ax.set_title(
        f"Per-problem optimal (M*, K*) averaged over {args.n_perm} permutations\n"
        f"c_rew={args.c_rew}, c_ver={args.c_ver}  |  size = (M, K) std deviation across perms"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title=legend_title, loc="upper left",
              framealpha=0.9)
    ax.set_xlim(0.8, n_max)
    ax.set_ylim(0.8, n_max)
    sns.despine()
    fig.tight_layout()
    out_scatter = out_dir / "plot_mk_scatter.png"
    fig.savefig(out_scatter, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_scatter}", flush=True)

    # ---- Alternative 1: M* and K* vs pass rate (two panels) ----
    # The cleanest way to show "as problems get harder, M* explodes but K* stays
    # small thanks to the reward model". Each panel makes one half of the story.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Pass rate on log scale so easy (>0.5) and hard (<0.01) are both visible
    pr_clipped = np.clip(pass_rates, 1.0 / max(M_means.max(), 512), 1.0)

    for ax_panel, y_data, y_label, color in [
        (axes[0], M_means, "M* (samples drawn)", COLORS[0]),
        (axes[1], K_means, "K* (samples verified)", COLORS[1]),
    ]:
        if has_difficulty:
            for diff_label, dc in diff_colors.items():
                mask = np.array([d == diff_label for d in difficulties])
                if not mask.any():
                    continue
                ax_panel.scatter(
                    pr_clipped[mask], y_data[mask],
                    s=70, c=dc, alpha=0.75, linewidths=0,
                    label=f"{diff_label} (n={mask.sum()})",
                )
            mask_other = np.array([d not in diff_colors for d in difficulties])
            if mask_other.any():
                ax_panel.scatter(
                    pr_clipped[mask_other], y_data[mask_other],
                    s=70, c="gray", alpha=0.5, linewidths=0,
                    label=f"unknown (n={mask_other.sum()})",
                )
        else:
            ax_panel.scatter(
                pr_clipped, y_data,
                s=70, c=color, alpha=0.75, linewidths=0,
            )

        # Theoretical reference: under uniform random ranking, expected M* to
        # see one correct sample is roughly 1 / pass_rate. Plot that as guide.
        pr_ref = np.logspace(np.log10(pr_clipped.min()),
                             np.log10(pr_clipped.max()), 50)
        if y_label.startswith("M*"):
            ax_panel.plot(pr_ref, 1.0 / pr_ref, "k--", alpha=0.3, lw=1,
                          label=r"M $\propto$ 1/pass_rate (no reward signal)")

        ax_panel.set_xscale("log")
        ax_panel.set_yscale("log")
        ax_panel.set_xlabel("Pass rate (correct samples / total)")
        ax_panel.set_ylabel(y_label + ", log scale")
        ax_panel.grid(True, which="both", alpha=0.25)
        ax_panel.legend(loc="upper right", framealpha=0.9)

    axes[0].set_title(f"M* explodes as problems get harder\n"
                      f"(harder = lower pass rate)")
    axes[1].set_title(f"K* stays small thanks to reward model\n"
                      f"(reward ranking concentrates correct samples near top)")
    fig.suptitle(
        f"How optimal (M*, K*) depends on problem difficulty  |  "
        f"c_rew={args.c_rew}, c_ver={args.c_ver}, n_perm={args.n_perm}",
    )
    sns.despine()
    fig.tight_layout()
    out_pass = out_dir / "plot_mk_vs_passrate.png"
    fig.savefig(out_pass, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_pass}", flush=True)

    # ---- Alternative 2: cost-colored scatter with iso-cost lines ----
    # Emphasizes that points further from origin = more expensive problems.
    # Iso-cost lines (M*c_rew + K*c_ver = constant) show what budget would
    # solve which subset of problems if budget were uniform.
    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    sizes = 60 + 4 * np.sqrt(M_stds ** 2 + K_stds ** 2)
    sc = ax.scatter(
        M_means, K_means,
        s=sizes, c=cost_means, cmap="plasma",
        alpha=0.85, linewidths=0,
        norm=__import__("matplotlib").colors.LogNorm(),
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(rf"Avg cost = {args.c_rew:.0f}$\times$M + {args.c_ver:.0f}$\times$K (log)")

    # Iso-cost lines: cost = c_rew * M + c_ver * K = C
    # Solve for K: K = (C - c_rew * M) / c_ver
    n_max = max(M_means.max(), K_means.max()) * 1.5
    cost_levels = np.geomspace(
        max(cost_means.min(), args.c_rew + args.c_ver),
        cost_means.max() * 1.5,
        6,
    )
    M_grid = np.logspace(0, np.log10(n_max), 200)
    for C in cost_levels:
        K_iso = (C - args.c_rew * M_grid) / args.c_ver
        valid = K_iso >= 1
        if not valid.any():
            continue
        ax.plot(M_grid[valid], K_iso[valid],
                "k-", alpha=0.18, lw=1)
        # Label the line near M = M_grid[valid][middle]
        mid = len(M_grid[valid]) // 2
        if mid > 0:
            ax.text(
                M_grid[valid][mid] * 1.05, K_iso[valid][mid] * 0.95,
                f"cost={C:.0f}", fontsize=7, alpha=0.5, color="black",
            )

    ax.plot([1, n_max], [1, n_max], "k--", alpha=0.25, lw=1, label="K = M")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M* (samples drawn, log scale)")
    ax.set_ylabel("K* (samples verified, log scale)")
    ax.set_title(
        "Per-problem (M*, K*) colored by cost  --  iso-cost lines shown\n"
        rf"$c_{{\mathrm{{rew}}}}={args.c_rew}$, $c_{{\mathrm{{ver}}}}={args.c_ver}$"
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="upper left")
    ax.set_xlim(0.8, n_max)
    ax.set_ylim(0.8, n_max)
    sns.despine()
    fig.tight_layout()
    out_iso = out_dir / "plot_mk_iso_cost.png"
    fig.savefig(out_iso, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_iso}", flush=True)

    # ---- Marginal histograms ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Choose log-spaced bins so both ends are visible
    log_max = np.log10(max(M_means.max(), K_means.max()) * 1.2)
    bins = np.logspace(0, log_max, 25)

    axes[0].hist(M_means, bins=bins, color=COLORS[0], alpha=0.7, edgecolor="black")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("M* (mean across permutations)")
    axes[0].set_ylabel("Number of problems")
    axes[0].set_title(f"Distribution of M* across {len(rows)} problems")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].hist(K_means, bins=bins, color=COLORS[1], alpha=0.7, edgecolor="black")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("K* (mean across permutations)")
    axes[1].set_ylabel("Number of problems")
    axes[1].set_title(f"Distribution of K* across {len(rows)} problems")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Per-problem optimal-(M, K) marginals (c_rew={args.c_rew}, c_ver={args.c_ver})"
    )
    sns.despine()
    fig.tight_layout()
    out_marg = out_dir / "plot_mk_marginals.png"
    fig.savefig(out_marg, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_marg}", flush=True)

    # ---- Quick summary ----
    print()
    print(f"=== Summary across {len(rows)} kept problems ===")
    print(f"  M*:  mean={M_means.mean():.1f}  median={np.median(M_means):.1f}  "
          f"min={M_means.min():.1f}  max={M_means.max():.1f}")
    print(f"  K*:  mean={K_means.mean():.1f}  median={np.median(K_means):.1f}  "
          f"min={K_means.min():.1f}  max={K_means.max():.1f}")
    print(f"  M*/K* ratio: mean={(M_means / K_means).mean():.2f}  "
          f"median={np.median(M_means / K_means):.2f}")
    print(f"  cost: mean={np.mean([r['cost_mean'] for r in rows]):.1f}  "
          f"median={np.median([r['cost_median'] for r in rows]):.1f}")


if __name__ == "__main__":
    main()