#!/usr/bin/env python3
"""
compare_with_adaptive.py

For G = 1..G_max, finds the optimal contiguous partition of solvable problems
(sorted by pass rate = n_correct / n_total) into G groups. Each group gets a
single (M*, K*) chosen to maximise average success then minimise cost.
Uses DP over all contiguous partitions — exact same result as exhaustive search.

Produces:
  plot_grouped_g_bars.png        — G on x-axis, cost/success on y-axes
  plot_grouped_cost_vs_success.png — (cost, success) plane: G points + adaptive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare import (
    build_problem_arrays,
    evaluate,
    load_generations,
    load_rewards,
    oracle_min_success_cost,
)


# ── Prefix sums ───────────────────────────────────────────────────────────────

def build_prefix_sums(per_trial_mk: np.ndarray, sorted_indices: np.ndarray) -> np.ndarray:
    """
    prefix[i, M, K] = sum over the first i sorted problems of
        per_trial_mk[pi, :, M, K].mean(axis=0)   (averaged over permutations)

    Shape: (P+1, max_n+1, max_n+1), dtype float64.
    """
    P = len(sorted_indices)
    _, _, n1, n2 = per_trial_mk.shape
    prefix = np.zeros((P + 1, n1, n2), dtype=np.float64)
    for rank, pi in enumerate(sorted_indices):
        prefix[rank + 1] = prefix[rank] + per_trial_mk[pi].mean(axis=0)
    return prefix


# ── Interval optima ───────────────────────────────────────────────────────────

def precompute_interval_optima(
    prefix_succ: np.ndarray,
    n_per_problem: np.ndarray,
    c_rew: float,
    c_ver: float,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    For every contiguous interval [l, r] of sorted problems, find (M*, K*) that
    maximises average success (primary) and minimises cost (secondary).

    Returns:
        interval_success[l, r]  float — achieved success rate
        interval_cost[l, r]     float — cost of (M*, K*)
        interval_mk[l][r]       (M*, K*)
    """
    P = len(n_per_problem)
    max_n = prefix_succ.shape[1] - 1

    M_idx = np.arange(max_n + 1)
    K_idx = np.arange(max_n + 1)
    K_grid, M_grid = np.meshgrid(K_idx, M_idx)          # both shape (max_n+1, max_n+1)
    cost_matrix = M_grid.astype(np.float64) * c_rew + K_grid.astype(np.float64) * c_ver

    interval_success = np.zeros((P, P), dtype=np.float64)
    interval_cost    = np.full((P, P), np.inf, dtype=np.float64)
    interval_mk: list = [[(0, 0)] * P for _ in range(P)]

    n_intervals = P * (P + 1) // 2
    done = 0

    for l in range(P):
        cur_min_n = n_per_problem[l]
        for r in range(l, P):
            cur_min_n = min(cur_min_n, n_per_problem[r])
            n_group   = r - l + 1

            group_avg = (prefix_succ[r + 1] - prefix_succ[l]) / n_group  # (max_n+1, max_n+1)

            valid = (
                (M_grid >= 1) & (K_grid >= 1) &
                (K_grid <= M_grid) & (M_grid <= cur_min_n)
            )
            succ_masked = np.where(valid, group_avg, -1.0)
            max_succ    = float(succ_masked.max())

            at_max       = succ_masked >= max_succ - 1e-9
            cost_at_max  = np.where(at_max, cost_matrix, np.inf)
            best_flat    = int(cost_at_max.argmin())
            best_M, best_K = divmod(best_flat, max_n + 1)

            interval_success[l, r] = max_succ
            interval_cost[l, r]    = float(cost_matrix[best_M, best_K])
            interval_mk[l][r]      = (best_M, best_K)

            done += 1
            if done % 500 == 0 or done == n_intervals:
                print(f'\r[intervals] {done}/{n_intervals}', end='', flush=True)

    print()
    return interval_success, interval_cost, interval_mk


# ── DP ────────────────────────────────────────────────────────────────────────

def dp_best_partition(
    interval_success: np.ndarray,
    interval_cost: np.ndarray,
    P: int,
    G_max: int,
) -> List[Tuple[float, float, list]]:
    """
    For G = 1..G_max, find the contiguous partition of P sorted problems into G groups
    that maximises average success (primary) then minimises average cost (secondary).

    dp_succ[g, i] = max weighted-success sum for problems [0..i-1] in g groups.
    dp_cost[g, i] = min weighted-cost   sum for the same partition.
    dp_from[g, i] = split point j (last group covers [j..i-1]).

    Returns list of (avg_success, avg_cost, groups) for G=1..G_max.
    groups is a list of (l, r) inclusive index pairs in sorted-problem space.
    """
    NEG_INF = -np.inf
    INF     =  np.inf

    dp_succ = np.full((G_max + 1, P + 1), NEG_INF)
    dp_cost = np.full((G_max + 1, P + 1), INF)
    dp_from = np.full((G_max + 1, P + 1), -1, dtype=int)

    # Base: g = 1, single group [0..i-1]
    for i in range(1, P + 1):
        dp_succ[1, i] = i * interval_success[0, i - 1]
        dp_cost[1, i] = i * interval_cost[0, i - 1]
        dp_from[1, i] = 0

    for g in range(2, G_max + 1):
        for i in range(g, P + 1):
            for j in range(g - 1, i):          # last group is [j..i-1]
                ns = dp_succ[g - 1, j] + (i - j) * interval_success[j, i - 1]
                nc = dp_cost[g - 1, j] + (i - j) * interval_cost[j, i - 1]
                if ns > dp_succ[g, i] + 1e-9 or (
                    ns >= dp_succ[g, i] - 1e-9 and nc < dp_cost[g, i] - 1e-9
                ):
                    dp_succ[g, i] = ns
                    dp_cost[g, i] = nc
                    dp_from[g, i] = j

    results = []
    for g in range(1, G_max + 1):
        avg_s = float(dp_succ[g, P]) / P
        avg_c = float(dp_cost[g, P]) / P

        # Traceback partition
        groups = []
        i = P
        for gg in range(g, 0, -1):
            j = int(dp_from[gg, i])
            groups.append((j, i - 1))
            i = j
        groups.reverse()
        results.append((avg_s, avg_c, groups))

    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(
    grouped_results: List[Tuple[float, float, list]],
    ad_avg_cost: float,
    ad_success_rate: float,
    oracle_avg_cost: float,
    pass_rates: np.ndarray,
    sorted_indices: np.ndarray,
    interval_mk: list,
    c_rew: float,
    c_ver: float,
    out_dir: Path,
    task: str = '',
    model_name: str = '',
    reward_model_name: str = '',
):
    import matplotlib.pyplot as plt
    import seaborn as sns
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

    def _make_title(base, task, model_name, reward_model_name):
        if task:
            base = f"{base} -- {task}"
        sub = ""
        if model_name and reward_model_name:
            sub = f"Gen: {model_name}  |  Reward: {reward_model_name}"
        elif model_name:
            sub = f"Gen: {model_name}"
        elif reward_model_name:
            sub = f"Reward: {reward_model_name}"
        return base, sub

    G_max  = len(grouped_results)
    G_vals = list(range(1, G_max + 1))
    g_cost = [r[1] for r in grouped_results]

    marker_kw = dict(marker='o', markersize=6, linewidth=2.0)

    # ---- Plot 1: absolute cost vs G ----
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(G_vals, g_cost, color=COLORS[0], label='DAP_k', **marker_kw)
    ax.axhline(ad_avg_cost,     color=COLORS[3],    linestyle='--', linewidth=2.0,
               label=f'ADAP (cost={ad_avg_cost:.2f})')
    ax.axhline(oracle_avg_cost, color=COLORS[1],    linestyle=':',  linewidth=1.8,
               label=f'SAP (cost={oracle_avg_cost:.2f})')

    ax.set_xlabel(r'Difficulty levels $k$')
    ax.set_ylabel('Average cost per trial')
    ax.set_yscale('log')
    ax.set_xticks(G_vals)
    title1, sub1 = _make_title("DAP_k Cost vs. ADAP", task, model_name, reward_model_name)
    ax.set_title(f"{title1}\n{sub1}" if sub1 else title1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_dir / 'plot_grouped_g_cost.png', dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: cost ratio relative to adaptive ----
    g_ratio     = [c / ad_avg_cost for c in g_cost]
    oracle_ratio = oracle_avg_cost / ad_avg_cost

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(G_vals, g_ratio, color=COLORS[0], label=r'DAP_k / ADAP', **marker_kw)
    ax.axhline(1.0,          color=COLORS[3],    linestyle='--', linewidth=2.0,
               label='ADAP (1.00x)')
    ax.axhline(oracle_ratio, color=COLORS[1],    linestyle=':',  linewidth=1.8,
               label=f'SAP ({oracle_ratio:.2f}x)')

    # annotate each G point with its ratio
    for g, ratio in zip(G_vals, g_ratio):
        ax.annotate(f'{ratio:.2f}x', xy=(g, ratio), xytext=(0, 6),
                    textcoords='offset points', ha='center', fontsize=8)

    ax.set_xlabel(r'Difficulty levels $k$')
    ax.set_ylabel(r'Cost / ADAP cost')
    ax.set_xticks(G_vals)
    title2, sub2 = _make_title("Cost Savings: ADAP over DAP_k", task, model_name, reward_model_name)
    ax.set_title(f"{title2}\n{sub2}" if sub2 else title2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_dir / 'plot_grouped_cost_ratio.png', dpi=160, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generations', required=True)
    ap.add_argument('--rewards',     required=True)
    ap.add_argument('--reward-key',  default='r_last',
                    choices=['r_min', 'r_mean', 'r_last', 'r_prod', 'r_score'])
    ap.add_argument('--c-rew',  type=float, default=1.0)
    ap.add_argument('--c-ver',  type=float, default=10.0)
    ap.add_argument('--c-min',  type=float, default=None)
    ap.add_argument('--n-perm', type=int,   default=10)
    ap.add_argument('--seed',   type=int,   default=0)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--G-max',  type=int,   default=10)
    ap.add_argument('--cost-grid-points', type=int, default=100)
    ap.add_argument('--task',              default='', help='Task label for plot titles (e.g. Math, Coding)')
    ap.add_argument('--model-name',        default='', help='Generation model name for plot titles')
    ap.add_argument('--reward-model-name', default='', help='Reward model name for plot titles')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c_min = args.c_min if args.c_min is not None else min(args.c_rew, args.c_ver)
    print(f'[config] c_rew={args.c_rew}, c_ver={args.c_ver}, c_min={c_min}')

    gens    = load_generations(Path(args.generations))
    rewards = load_rewards(Path(args.rewards))
    print(f'[data] {len(gens)} problems in generations, {len(rewards)} in rewards')

    problems = build_problem_arrays(gens, rewards, args.reward_key)
    max_n    = max(len(v[0]) for v in problems.values())
    cost_grid = np.logspace(
        np.log10(args.c_rew + args.c_ver),
        np.log10(max_n * (args.c_rew + args.c_ver)),
        args.cost_grid_points,
    )

    adaptive_results, per_trial_mk, problem_ids = evaluate(
        problems=problems,
        c_rew=args.c_rew,
        c_ver=args.c_ver,
        c_min=c_min,
        n_perm=args.n_perm,
        seed=args.seed,
        cost_grid=cost_grid,
    )

    ad_costs      = np.array([r['adaptive_cost']   for r in adaptive_results], dtype=np.float64)
    ad_success    = np.array([r['adaptive_success'] for r in adaptive_results], dtype=np.int8)
    ad_avg_cost   = float(ad_costs.mean())
    ad_success_rate = float(ad_success.mean())

    print('[grouped] computing oracle costs...', flush=True)
    oracle_cost_arr, _, _ = oracle_min_success_cost(per_trial_mk, args.c_rew, args.c_ver)
    oracle_flat   = oracle_cost_arr.flatten()
    oracle_avg_cost = float(oracle_flat[np.isfinite(oracle_flat)].mean())

    # Sort solvable problems by pass rate ascending (hardest first)
    pass_rates = np.array([
        float(problems[pid][1].sum()) / len(problems[pid][1])
        for pid in problem_ids
    ])
    sorted_indices = np.argsort(pass_rates)          # indices into problem_ids / per_trial_mk
    n_per_problem  = np.array(
        [len(problems[pid][1]) for pid in problem_ids], dtype=int
    )[sorted_indices]
    P     = len(problem_ids)
    G_max = min(args.G_max, P)

    print('[grouped] building prefix sums...', flush=True)
    prefix_succ = build_prefix_sums(per_trial_mk, sorted_indices)

    print(f'[grouped] precomputing interval optima ({P*(P+1)//2} intervals)...', flush=True)
    interval_success, interval_cost, interval_mk = precompute_interval_optima(
        prefix_succ, n_per_problem, args.c_rew, args.c_ver,
    )

    print(f'[grouped] DP for G=1..{G_max}...', flush=True)
    grouped_results = dp_best_partition(interval_success, interval_cost, P, G_max)

    # ---- Print summary ----
    print('\n=== Grouped Fixed Strategy Results ===')
    print(f'{"":>5}  {"cost":>10}  {"ratio":>8}  {"success":>9}')
    print(f'{"adapt":>5}  {ad_avg_cost:>10.3f}  {"1.00x":>8}  {ad_success_rate:>9.4f}')
    print(f'{"oracle":>5}  {oracle_avg_cost:>10.3f}  {oracle_avg_cost/ad_avg_cost:>7.2f}x  {"1.0000":>9}')
    print()
    for g, (succ, cost, groups) in enumerate(grouped_results, 1):
        ratio = cost / ad_avg_cost if ad_avg_cost > 0 else float('inf')
        print(f'G={g:<2d}   {cost:>10.3f}  {ratio:>7.2f}x  {succ:>9.4f}')
        for gi, (l, r) in enumerate(groups):
            mk = interval_mk[l][r]
            lo_pr = pass_rates[sorted_indices[l]]
            hi_pr = pass_rates[sorted_indices[r]]
            isucc = interval_success[l, r]
            icost = interval_cost[l, r]
            print(f'         group {gi+1}: [{l}..{r}] '
                  f'pass_rate=[{lo_pr:.3f}..{hi_pr:.3f}]  '
                  f'M*={mk[0]} K*={mk[1]}  '
                  f'group_success={isucc:.4f}  group_cost={icost:.2f}')

    # ---- Write summary CSV ----
    import csv
    with open(out_dir / 'grouped_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['G', 'avg_cost', 'cost_ratio_to_adaptive', 'avg_success'])
        for g, (succ, cost, _) in enumerate(grouped_results, 1):
            w.writerow([g, f'{cost:.6f}', f'{cost/ad_avg_cost:.6f}', f'{succ:.6f}'])

    make_plots(
        grouped_results, ad_avg_cost, ad_success_rate, oracle_avg_cost,
        pass_rates, sorted_indices, interval_mk,
        args.c_rew, args.c_ver, out_dir,
        task=args.task,
        model_name=args.model_name,
        reward_model_name=args.reward_model_name,
    )
    print(f'\n[done] outputs in {out_dir}')


if __name__ == '__main__':
    main()
