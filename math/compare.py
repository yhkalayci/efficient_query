#!/usr/bin/env python3
"""
Evaluate the adaptive meta-generation algorithm against non-adaptive (M, K) baselines.

Data / CLI arguments / file formats are intentionally kept unchanged.

Main outputs requested:
  1) Cost-vs-accuracy plot:
       - BEST FIXED non-adaptive under each budget C:
           choose a single (M,K) with M*c_rew + K*c_ver <= C that maximizes
           average success across ALL trials.
       - Adaptive shown as a point at (average adaptive cost, adaptive success rate).
       - Oracle shown as a point at (average oracle minimum cost, 1.0), where oracle
           is the per-trial hindsight-best non-adaptive strategy.

  2) Sorted-per-instance plot:
       - Sort trials by oracle minimum cost.
       - Plot oracle minimum cost and adaptive realized cost for the same trial.
       - Also pick the best FIXED (M,K) under the adaptive average-cost budget and
         mark whether it succeeds on each sorted trial.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# -------------------------- Data loading --------------------------
def load_generations(path: Path) -> Dict[str, List[dict]]:
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = rec["samples"]
    return out


def load_rewards(path: Path) -> Dict[str, Dict[int, dict]]:
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = {r["idx"]: r for r in rec["rewards"]}
    return out


def build_problem_arrays(
    gens: Dict[str, List[dict]],
    rewards: Dict[str, Dict[int, dict]],
    reward_key: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Return {problem_id: (rewards_array, correct_array)} aligned by sample index.
    Missing rewards are NaN and treated as -inf for ranking.
    """
    out = {}
    for pid, samples in gens.items():
        rmap = rewards.get(pid, {})
        n = len(samples)
        rwd = np.full(n, np.nan, dtype=np.float64)
        cor = np.zeros(n, dtype=np.int8)
        for i, s in enumerate(samples):
            cor[i] = int(bool(s.get("correct")))
            rr = rmap.get(i)
            if rr is not None and rr.get(reward_key) is not None:
                rwd[i] = float(rr[reward_key])
        out[pid] = (rwd, cor)
    return out


# -------------------------- Adaptive schedule --------------------------
def build_schedule(c_rew: float, c_ver: float, c_min: float, max_budget: float) -> List[Tuple[int, int, int]]:
    """
    Return list of (s, m_s, k_s) for non-empty shells S_s.

    IMPORTANT: m_s is interpreted as the NUMBER OF NEW SAMPLES drawn at iteration s,
    consistent with the user's textual algorithm ("get the next m_s many rewards").
    """
    schedule = []
    cum_samples = 0

    for s in range(0, 40):
        lo = (2 ** s) * c_min
        hi = (2 ** (s + 1)) * c_min

        if c_rew > 0:
            b_max = int(math.floor(math.log2(max(hi / c_rew, 1e-12)))) + 2
        else:
            b_max = 40

        S = []
        for b in range(0, max(0, b_max) + 1):
            for a in range(0, b + 1):
                cost = c_rew * (2 ** b) + c_ver * (2 ** (b - a))
                if lo <= cost < hi:
                    S.append((a, b))

        if not S:
            continue

        b_star = max(b for (_, b) in S)
        j_star = max(b - a for (a, b) in S)
        m_s = int(math.ceil(2 ** (b_star + 1)))
        k_s = int(math.ceil(6 * (2 ** j_star)))
        schedule.append((s, m_s, k_s))

        cum_samples += m_s
        if cum_samples * c_rew > 4 * max_budget:
            break

    return schedule


def run_adaptive(
    rwd: np.ndarray,
    cor: np.ndarray,
    perm: np.ndarray,
    schedule: List[Tuple[int, int, int]],
    c_rew: float,
    c_ver: float,
) -> Tuple[bool, float, int, int]:
    """
    Run the adaptive algorithm on one permutation.
    Returns (succeeded, total_cost, n_drawn, n_verified).
    """
    n = len(perm)
    rwd_perm = rwd[perm]
    cor_perm = cor[perm]

    drawn = 0
    verified_mask = np.zeros(n, dtype=bool)
    n_verified = 0

    for (_s, m_s, k_s) in schedule:
        drawn = min(drawn + m_s, n)

        candidate_idx = np.where(~verified_mask[:drawn])[0]
        if len(candidate_idx) == 0:
            if drawn >= n:
                cost = drawn * c_rew + n_verified * c_ver
                return False, cost, drawn, n_verified
            continue

        cand_rewards = np.nan_to_num(rwd_perm[candidate_idx], nan=-np.inf)
        if len(candidate_idx) <= k_s:
            to_verify = candidate_idx
        else:
            top_local = np.argpartition(-cand_rewards, k_s - 1)[:k_s]
            to_verify = candidate_idx[top_local]

        verified_mask[to_verify] = True
        n_verified += len(to_verify)

        if cor_perm[to_verify].any():
            cost = drawn * c_rew + n_verified * c_ver
            return True, cost, drawn, n_verified

        if drawn >= n and verified_mask[:drawn].all():
            cost = drawn * c_rew + n_verified * c_ver
            return False, cost, drawn, n_verified

    cost = drawn * c_rew + n_verified * c_ver
    return False, cost, drawn, n_verified


# -------------------------- Non-adaptive --------------------------
def nonadaptive_success_matrix(rwd: np.ndarray, cor: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    S[M, K] = success of non-adaptive strategy:
      draw first M samples of perm, verify top-K by reward.

    Shape (n+1, n+1), entries meaningful only for 1 <= K <= M <= n.
    """
    n = len(perm)
    succ = np.zeros((n + 1, n + 1), dtype=np.int8)
    for M in range(1, n + 1):
        first_M = perm[:M]
        rewards_M = np.nan_to_num(rwd[first_M], nan=-np.inf)
        order = np.argsort(-rewards_M)
        correct_ordered = cor[first_M][order]
        any_correct_prefix = np.cumsum(correct_ordered) > 0
        for K in range(1, M + 1):
            succ[M, K] = int(any_correct_prefix[K - 1])
    return succ


def oracle_min_success_cost(per_trial_mk: np.ndarray, c_rew: float, c_ver: float):
    """
    For each trial, minimum cost(M,K) among successful non-adaptive strategies.
    Returns (cost, M, K) arrays of shape (n_problems, n_perm).
    M and K are 0 for unsolvable trials.
    """
    n_problems, n_perm, n1, _ = per_trial_mk.shape
    n = n1 - 1
    mk_costs = []
    for M in range(1, n + 1):
        for K in range(1, M + 1):
            mk_costs.append((M * c_rew + K * c_ver, M, K))
    mk_costs.sort()

    out_cost = np.full((n_problems, n_perm), np.inf, dtype=np.float64)
    out_M = np.zeros((n_problems, n_perm), dtype=np.int32)
    out_K = np.zeros((n_problems, n_perm), dtype=np.int32)
    for pi in range(n_problems):
        for p in range(n_perm):
            for cost, M, K in mk_costs:
                if per_trial_mk[pi, p, M, K]:
                    out_cost[pi, p] = cost
                    out_M[pi, p] = M
                    out_K[pi, p] = K
                    break
    return out_cost, out_M, out_K


def best_fixed_curve_under_budget(
    per_trial_mk: np.ndarray,
    c_rew: float,
    c_ver: float,
    cost_grid: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """
    For each budget C in cost_grid:
      choose ONE fixed (M,K) with cost <= C maximizing average success over all trials.

    Returns:
      curve_succ[t]      : best average success at budget cost_grid[t]
      chosen_mk[t]       : argmax (M,K)
      chosen_cost[t]     : actual cost of chosen (M,K)
    """
    avg_success = per_trial_mk.mean(axis=(0, 1))
    n = avg_success.shape[0] - 1

    all_points = []
    for M in range(1, n + 1):
        for K in range(1, M + 1):
            cost = M * c_rew + K * c_ver
            succ = float(avg_success[M, K])
            all_points.append((cost, succ, M, K))
    all_points.sort(key=lambda x: x[0])

    curve = np.zeros(len(cost_grid), dtype=np.float64)
    chosen_mk: List[Tuple[int, int]] = []
    chosen_cost = np.zeros(len(cost_grid), dtype=np.float64)

    best_succ_so_far = -1.0
    best_mk_so_far = (0, 0)
    best_cost_so_far = 0.0
    idx = 0
    for t, budget in enumerate(cost_grid):
        while idx < len(all_points) and all_points[idx][0] <= budget:
            cost, succ, M, K = all_points[idx]
            if succ > best_succ_so_far + 1e-12:
                best_succ_so_far = succ
                best_mk_so_far = (M, K)
                best_cost_so_far = cost
            idx += 1
        curve[t] = max(best_succ_so_far, 0.0)
        chosen_mk.append(best_mk_so_far)
        chosen_cost[t] = best_cost_so_far

    return curve, chosen_mk, chosen_cost


def best_fixed_mk_at_budget(
    per_trial_mk: np.ndarray,
    c_rew: float,
    c_ver: float,
    target_cost: float,
) -> Tuple[Tuple[int, int], float, float]:
    """
    Choose best fixed (M,K) under budget <= target_cost.
    Returns (M,K), average_success, actual_cost.
    """
    avg_success = per_trial_mk.mean(axis=(0, 1))
    n = avg_success.shape[0] - 1
    best_mk = (0, 0)
    best_succ = -1.0
    best_cost = 0.0
    for M in range(1, n + 1):
        for K in range(1, M + 1):
            cost = M * c_rew + K * c_ver
            if cost > target_cost:
                continue
            succ = float(avg_success[M, K])
            if succ > best_succ + 1e-12:
                best_succ = succ
                best_mk = (M, K)
                best_cost = cost
    return best_mk, max(best_succ, 0.0), best_cost



def best_fixed_min_cost_for_target_success(
    per_trial_mk: np.ndarray,
    c_rew: float,
    c_ver: float,
    target_success: float,
) -> Tuple[Tuple[int, int], float, float]:
    """
    Find the CHEAPEST fixed (M,K) whose average success is at least target_success.
    Returns (M,K), actual_cost, achieved_success.
    If no such strategy exists, returns ((0,0), inf, best_possible_success).
    """
    avg_success = per_trial_mk.mean(axis=(0, 1))
    n = avg_success.shape[0] - 1

    best_mk = (0, 0)
    best_cost = float("inf")
    best_succ = -1.0
    best_possible = -1.0

    for M in range(1, n + 1):
        for K in range(1, M + 1):
            succ = float(avg_success[M, K])
            cost = M * c_rew + K * c_ver
            if succ > best_possible:
                best_possible = succ
            if succ + 1e-12 < target_success:
                continue
            if (cost < best_cost - 1e-12) or (
                abs(cost - best_cost) <= 1e-12 and succ > best_succ + 1e-12
            ):
                best_cost = cost
                best_succ = succ
                best_mk = (M, K)

    if not np.isfinite(best_cost):
        return (0, 0), float("inf"), max(best_possible, 0.0)
    return best_mk, float(best_cost), float(best_succ)


# -------------------------- Experiment loop --------------------------
def evaluate(
    problems: Dict[str, Tuple[np.ndarray, np.ndarray]],
    c_rew: float,
    c_ver: float,
    c_min: float,
    n_perm: int,
    seed: int,
    cost_grid: np.ndarray,
):
    rng = np.random.default_rng(seed)

    solvable = {pid: (r, c) for pid, (r, c) in problems.items() if int(c.sum()) > 0}
    if not solvable:
        sys.exit("[fatal] no problems with any correct sample")

    schedule = build_schedule(c_rew, c_ver, c_min, float(cost_grid.max()) * 2.0)
    print(f"[schedule] non-empty shells = {len(schedule)}")
    print(f"           {'s':>3s}  {'m_s':>8s}  {'k_s':>8s}  {'iter_cost':>12s}")
    for s, m_s, k_s in schedule[:12]:
        iter_cost = m_s * c_rew + k_s * c_ver
        print(f"           {s:>3d}  {m_s:>8d}  {k_s:>8d}  {iter_cost:>12.2f}")
    if len(schedule) > 12:
        print(f"           ... ({len(schedule) - 12} more)")

    adaptive_results = []
    max_n = max(len(v[0]) for v in solvable.values())
    per_trial_mk = np.zeros((len(solvable), n_perm, max_n + 1, max_n + 1), dtype=np.int8)

    problem_ids = list(sorted(solvable.keys()))
    for pi, pid in enumerate(problem_ids):
        rwd, cor = solvable[pid]
        n = len(rwd)
        n_correct = int(cor.sum())

        for p_idx in range(n_perm):
            perm = rng.permutation(n)

            succ, cost, n_drawn, n_ver = run_adaptive(rwd, cor, perm, schedule, c_rew, c_ver)
            succ_mat = nonadaptive_success_matrix(rwd, cor, perm)
            per_trial_mk[pi, p_idx, : n + 1, : n + 1] = succ_mat

            adaptive_results.append({
                "problem_id": pid,
                "perm_idx": p_idx,
                "n_correct": n_correct,
                "adaptive_success": int(succ),
                "adaptive_cost": float(cost),
                "adaptive_n_drawn": int(n_drawn),
                "adaptive_n_verified": int(n_ver),
            })

        if (pi + 1) % 10 == 0 or (pi + 1) == len(problem_ids):
            print(f"[eval] {pi + 1}/{len(problem_ids)} problems processed", flush=True)

    return adaptive_results, per_trial_mk, problem_ids


# -------------------------- Plotting --------------------------
def make_plots(
    adaptive_results,
    per_trial_mk,
    oracle_min_cost,
    oracle_M_arr,
    oracle_K_arr,
    best_fixed_budget_curve,
    best_fixed_budget_mk,
    best_fixed_budget_actual_cost,
    cost_grid,
    problem_ids,
    c_rew,
    c_ver,
    out_dir: Path,
    task: str = '',
    model_name: str = '',
    reward_model_name: str = '',
):
    import matplotlib.pyplot as plt
    import textwrap
    import seaborn as sns
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

    def _wrapped_title(*lines, width=68):
        wrapped = []
        for line in lines:
            if line is None:
                continue
            wrapped.append(textwrap.fill(str(line), width=width, break_long_words=False, break_on_hyphens=False))
        return "\n".join(wrapped)

    def _make_title(base, task, model_name, reward_model_name):
        if task:
            base = f"{base} ({task})"
        gen_sub = f"Generator model: {model_name}" if model_name else ""
        rew_sub = f"Reward model: {reward_model_name}" if reward_model_name else ""
        return base, gen_sub, rew_sub

    n_trials = len(adaptive_results)
    ad_costs = np.array([r["adaptive_cost"] for r in adaptive_results], dtype=np.float64)
    ad_success = np.array([r["adaptive_success"] for r in adaptive_results], dtype=np.int8)

    oracle_costs_flat = oracle_min_cost.flatten()
    oracle_costs_finite = oracle_costs_flat[np.isfinite(oracle_costs_flat)]

    ad_avg_cost = float(ad_costs.mean())
    ad_success_rate = float(ad_success.mean())
    oracle_avg_cost = float(oracle_costs_finite.mean())
    oracle_success_rate = float(np.isfinite(oracle_costs_flat).mean())  # should be 1.0 here

    # Best fixed under adaptive average-cost budget
    best_mk_at_ad, best_succ_at_ad, best_cost_at_ad = best_fixed_mk_at_budget(
        per_trial_mk, c_rew, c_ver, ad_avg_cost
    )

    # Cheapest best-fixed strategy that matches or exceeds adaptive's average quality.
    match_mk, match_cost, match_succ = best_fixed_min_cost_for_target_success(
        per_trial_mk, c_rew, c_ver, ad_success_rate
    )
    if np.isfinite(match_cost):
        match_gap = float(match_cost - ad_avg_cost)
        match_ratio = float(match_cost / max(ad_avg_cost, 1e-9))
    else:
        match_gap = float("inf")
        match_ratio = float("inf")

    # Success of that single fixed strategy on each trial
    pid_to_pi = {pid: i for i, pid in enumerate(problem_ids)}
    fixed_at_ad_success = np.zeros(n_trials, dtype=np.int8)
    oracle_beats_adaptive = np.zeros(n_trials, dtype=np.int8)

    for idx, r in enumerate(adaptive_results):
        pi = pid_to_pi[r["problem_id"]]
        p_idx = r["perm_idx"]
        M, K = best_mk_at_ad
        fixed_at_ad_success[idx] = int(per_trial_mk[pi, p_idx, M, K]) if M > 0 and K > 0 else 0
        oracle_beats_adaptive[idx] = int(oracle_costs_flat[idx] <= r["adaptive_cost"])

        r["bestfixed_at_adavg_M"] = int(M)
        r["bestfixed_at_adavg_K"] = int(K)
        r["bestfixed_at_adavg_success"] = int(fixed_at_ad_success[idx])
        r["oracle_min_cost"] = float(oracle_costs_flat[idx])

    # ------------------------------------------------------------------
    # Plot 1: requested cost-vs-accuracy plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 6.4))

    ax.step(
        cost_grid,
        best_fixed_budget_curve,
        where="post",
        linewidth=2.3,
        color=COLORS[0],
        label="Uniform",
    )

    # mark adaptive and oracle averages
    ax.scatter(
        [ad_avg_cost],
        [ad_success_rate],
        color=COLORS[3],
        s=220,
        marker="*",
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
        label=f"ADAP (cost={ad_avg_cost:.1f}, success rate={ad_success_rate:.3f})",
    )
    ax.scatter(
        [oracle_avg_cost],
        [oracle_success_rate],
        color=COLORS[1],
        s=180,
        marker="D",
        edgecolors="black",
        linewidths=0.8,
        zorder=6,
        label=f"SAP (cost={oracle_avg_cost:.1f}, success rate={oracle_success_rate:.3f})",
    )

    if np.isfinite(match_cost):
        ax.scatter(
            [match_cost],
            [match_succ],
            color=COLORS[2],
            s=150,
            marker="o",
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            label=(
                f"Uniform matched to ADAP (cost={match_cost:.1f}, "
                f"success rate={match_succ:.3f})"
            ),
        )

    # reference read-offs from the best-fixed curve
    idx_ad = int(np.searchsorted(cost_grid, ad_avg_cost, side="right") - 1)
    idx_ad = max(0, min(idx_ad, len(cost_grid) - 1))
    bf_curve_at_ad = float(best_fixed_budget_curve[idx_ad])
    bf_mk_at_ad = best_fixed_budget_mk[idx_ad]
    bf_cost_at_ad = float(best_fixed_budget_actual_cost[idx_ad])

    idx_or = int(np.searchsorted(cost_grid, oracle_avg_cost, side="right") - 1)
    idx_or = max(0, min(idx_or, len(cost_grid) - 1))
    bf_curve_at_oracle = float(best_fixed_budget_curve[idx_or])

    ax.axvline(ad_avg_cost, color=COLORS[3], linestyle=":", alpha=0.35)
    ax.axvline(oracle_avg_cost, color=COLORS[1], linestyle=":", alpha=0.35)
    ax.axhline(ad_success_rate, color=COLORS[3], linestyle="--", alpha=0.18)
    if np.isfinite(match_cost):
        ax.axvline(match_cost, color=COLORS[2], linestyle=":", alpha=0.45)

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Cost budget")
    ax.set_ylabel("Accuracy / solved fraction")
    title1, gen_sub, rew_sub = _make_title("Cost vs. Accuracy", task, model_name, reward_model_name)
    n_subs1 = sum(bool(s) for s in [gen_sub, rew_sub])
    fig.suptitle(title1, fontsize=22, y=0.98)
    y1 = 0.928
    if gen_sub:
        fig.text(0.5, y1, gen_sub, ha='center', va="top", fontsize=15)
        y1 -= 0.040
    if rew_sub:
        fig.text(0.5, y1, rew_sub, ha='center', va="top", fontsize=15)
    ax.grid(True, alpha=0.3)
    if np.isfinite(match_cost):
        ax.text(
            0.02,
            0.98,
            f"Uniform cost to match ADAP:\n{match_cost:.1f} = {match_ratio:.2f}x ADAP\nGap = {match_gap:+.1f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="0.7"),
        )
    ax.legend(loc="lower right")
    sns.despine()
    fig.tight_layout()
    top1 = 0.84 if n_subs1 == 2 else (0.89 if n_subs1 == 1 else 0.94)
    fig.subplots_adjust(top=top1)
    fig.savefig(out_dir / "plot_requested_cost_vs_accuracy.pdf", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: per-prompt comparison (one permutation per problem)
    # ------------------------------------------------------------------
    perm0_mask = np.array([r["perm_idx"] == 0 for r in adaptive_results])
    oracle_costs_p0 = oracle_min_cost[:, 0]
    ad_costs_p0 = ad_costs[perm0_mask]
    fixed_success_p0 = fixed_at_ad_success[perm0_mask]

    order = np.argsort(oracle_costs_p0)
    sorted_oracle_cost = oracle_costs_p0[order]
    sorted_ad_cost = ad_costs_p0[order]
    sorted_fixed_success = fixed_success_p0[order]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10.0, 7.0))

    # Color the background by whether the best fixed non-adaptive strategy
    # succeeds on each trial. Merge consecutive trials with the same outcome into
    # translucent intervals to avoid a cluttered success/fail panel.
    if len(sorted_fixed_success) > 0:
        run_start = 0
        current_val = int(sorted_fixed_success[0])
        for i in range(1, len(sorted_fixed_success) + 1):
            if i == len(sorted_fixed_success) or int(sorted_fixed_success[i]) != current_val:
                color = COLORS[2] if current_val == 1 else COLORS[3]
                alpha = 0.22 if current_val == 1 else 0.28
                ax.axvspan(run_start - 0.5, i - 0.5, color=color, alpha=alpha, linewidth=0, zorder=0)
                if run_start > 0:
                    ax.axvline(run_start - 0.5, color='white', linewidth=0.8, alpha=0.55, zorder=1)
                if i < len(sorted_fixed_success):
                    run_start = i
                    current_val = int(sorted_fixed_success[i])

    ax.plot(x, sorted_oracle_cost, color=COLORS[1], linewidth=2.5, label='SAP min cost', zorder=3)
    ax.plot(x, sorted_ad_cost, color=COLORS[0], linewidth=2.1, alpha=0.98, label='ADAP cost', zorder=4)

    # Highlight trials where adaptive is much more expensive than oracle.
    ratio = sorted_ad_cost / np.maximum(sorted_oracle_cost, 1e-9)
    high_ratio = ratio >= 8.0
    if np.any(high_ratio):
        ax.scatter(
            x[high_ratio],
            sorted_ad_cost[high_ratio],
            s=22,
            color='black',
            alpha=0.55,
            label=r'ADAP / SAP $\geq 8\times$',
            zorder=5,
        )

    from matplotlib.patches import Patch
    bg_handles = [
        Patch(facecolor=COLORS[2], alpha=0.22, edgecolor='none', label='Uniform succeeds'),
        Patch(facecolor=COLORS[3], alpha=0.28, edgecolor='none', label='Uniform fails'),
    ]

    ax.set_yscale('log')
    ax.set_ylabel('Cost', fontsize=20)
    ax.set_xlabel('Prompts sorted in increasing order of SAP cost', fontsize=20)
    title2, gen_sub2, rew_sub2 = _make_title("Per-prompt Comparison", task, model_name, reward_model_name)
    n_subs2 = sum(bool(s) for s in [gen_sub2, rew_sub2])
    fig.suptitle(title2, fontsize=24, y=0.98)
    y2 = 0.928
    if gen_sub2:
        fig.text(0.5, y2, gen_sub2, ha='center', va="top", fontsize=15)
        y2 -= 0.040
    if rew_sub2:
        fig.text(0.5, y2, rew_sub2, ha='center', va="top", fontsize=15)
    ax.grid(True, alpha=0.3)
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=bg_handles + line_handles,
        labels=[h.get_label() for h in bg_handles] + line_labels,
        loc='upper left',
        frameon=True,
    )

    sns.despine()
    fig.tight_layout()
    top2 = 0.84 if n_subs2 == 2 else (0.89 if n_subs2 == 1 else 0.94)
    fig.subplots_adjust(top=top2)
    fig.savefig(out_dir / 'plot_requested_sorted_by_oracle_cost.pdf', dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 3: cost ratio distribution (useful diagnostic)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ratio_all = ad_costs / np.maximum(oracle_costs_flat, 1e-9)
    bins = np.logspace(np.log10(max(ratio_all.min(), 1e-3)), np.log10(max(ratio_all.max(), 1.0) + 1e-9), 40)
    ax.hist(ratio_all, bins=bins, alpha=0.75, color=COLORS[4])
    ax.axvline(float(np.mean(ratio_all)), color="black", linestyle="--", label=f"mean ratio={np.mean(ratio_all):.2f}")
    ax.axvline(float(np.median(ratio_all)), color=COLORS[2], linestyle=":", label=f"median ratio={np.median(ratio_all):.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("ADAP cost / SAP cost")
    ax.set_ylabel("Count")
    hist_main, hist_gen, hist_rew = _make_title("ADAP / SAP Cost Ratio", task, model_name, reward_model_name)
    hist_sub = "  |  ".join(filter(None, [hist_gen, hist_rew]))
    ax.set_title(f"{hist_main}\n{hist_sub}" if hist_sub else hist_main)
    ax.grid(True, alpha=0.3)
    ax.legend()
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_adaptive_vs_oracle_ratio_hist.pdf", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    adaptive_beats_fixed = int(((ad_success == 1) & (fixed_at_ad_success == 0)).sum())
    fixed_beats_adaptive = int(((ad_success == 0) & (fixed_at_ad_success == 1)).sum())
    both_succeed = int(((ad_success == 1) & (fixed_at_ad_success == 1)).sum())
    both_fail = int(((ad_success == 0) & (fixed_at_ad_success == 0)).sum())

    # Average M and K for each strategy
    ad_avg_M = float(np.mean([r["adaptive_n_drawn"] for r in adaptive_results]))
    ad_avg_K = float(np.mean([r["adaptive_n_verified"] for r in adaptive_results]))

    solvable_mask = np.isfinite(oracle_costs_flat)
    oracle_M_flat = oracle_M_arr.flatten()
    oracle_K_flat = oracle_K_arr.flatten()
    oracle_avg_M = float(oracle_M_flat[solvable_mask].mean()) if solvable_mask.any() else 0.0
    oracle_avg_K = float(oracle_K_flat[solvable_mask].mean()) if solvable_mask.any() else 0.0

    # best fixed under adaptive cost: fixed (M,K) applied to all trials
    bf_ad_M, bf_ad_K = int(best_mk_at_ad[0]), int(best_mk_at_ad[1])
    # best fixed matching full adaptive success: fixed (M,K) applied to all trials
    match_M, match_K = int(match_mk[0]), int(match_mk[1])

    summary_lines = [
        "=== Summary ===",
        f"c_rew = {c_rew}",
        f"c_ver = {c_ver}",
        f"n_problems (>=1 correct sample) = {len(problem_ids)}",
        f"n_trials = {n_trials}",
        "",
        "Average M (samples drawn) and K (samples verified) per strategy:",
        f"  ADAP           avg M = {ad_avg_M:.2f}   avg K = {ad_avg_K:.2f}",
        f"  SAP            avg M = {oracle_avg_M:.2f}   avg K = {oracle_avg_K:.2f}   (per-trial hindsight-best fixed)",
        f"  BF under ADAP cost   M = {bf_ad_M}   K = {bf_ad_K}   (same for all trials)",
        (f"  BF full success      M = {match_M}   K = {match_K}   (same for all trials)"
         if np.isfinite(match_cost) else
         f"  BF full success      M = n/a   K = n/a   (no BF strategy reaches ADAP acc={ad_success_rate:.3f})"),
        "",
        "Requested headline quantities:",
        f"  SAP average cost                        = {oracle_avg_cost:.6f}",
        f"  SAP avg M                               = {oracle_avg_M:.2f}",
        f"  SAP avg K                               = {oracle_avg_K:.2f}",
        f"  ADAP average cost                       = {ad_avg_cost:.6f}",
        f"  ADAP avg M (drawn)                      = {ad_avg_M:.2f}",
        f"  ADAP avg K (verified)                   = {ad_avg_K:.2f}",
        f"  ADAP success rate                       = {ad_success_rate:.6f}",
        f"  Cheapest BF cost matching ADAP          = {match_cost:.6f}" if np.isfinite(match_cost) else "  Cheapest BF cost matching ADAP          = inf",
        f"  Matching BF strategy (M,K)              = {match_mk}",
        f"  Cost gap (BF - ADAP)                    = {match_gap:.6f}" if np.isfinite(match_gap) else "  Cost gap (BF - ADAP)                    = inf",
        f"  Cost ratio (BF / ADAP)                  = {match_ratio:.6f}" if np.isfinite(match_ratio) else "  Cost ratio (BF / ADAP)                  = inf",
        "",
        "BF under various budget regimes:",
        "  Curve file                              = plot_requested_cost_vs_accuracy.pdf",
        "  At budget = ADAP average cost:",
        f"    chosen (M,K)                          = {best_mk_at_ad}",
        f"    M                                     = {bf_ad_M}",
        f"    K                                     = {bf_ad_K}",
        f"    actual chosen cost                    = {best_cost_at_ad:.6f}",
        f"    average success                       = {best_succ_at_ad:.6f}",
        "  Cheapest BF to match ADAP success:",
        f"    (M,K)                                 = {match_mk}",
        f"    M                                     = {match_M}",
        f"    K                                     = {match_K}",
        "",
        "Paired comparison on the SAME trials using that one BF strategy:",
        f"  ADAP succeeds, BF fails                = {adaptive_beats_fixed}",
        f"  BF succeeds, ADAP fails                = {fixed_beats_adaptive}",
        f"  both succeed                            = {both_succeed}",
        f"  both fail                               = {both_fail}",
        "",
        "Cost comparison against SAP:",
        f"  mean(ADAP / SAP)                        = {float(np.mean(ad_costs / np.maximum(oracle_costs_flat, 1e-9))):.6f}",
        f"  median(ADAP / SAP)                      = {float(np.median(ad_costs / np.maximum(oracle_costs_flat, 1e-9))):.6f}",
        "",
        "Requested sorted-instance plot:",
        "  File                                    = plot_requested_sorted_by_oracle_cost.pdf",
        "  Trials are sorted by SAP minimum cost; top panel overlays SAP/ADAP cost,",
        "  bottom panel marks whether the single BF strategy (chosen at ADAP avg-cost budget) succeeds.",
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    (out_dir / "summary.txt").write_text(summary_text + "\n")


# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--rewards", required=True)
    ap.add_argument("--reward-key", default="r_last", choices=["r_min", "r_mean", "r_last", "r_prod"])
    ap.add_argument("--c-rew", type=float, default=1.0, help="Cost per sample generation + reward scoring")
    ap.add_argument("--c-ver", type=float, default=10.0, help="Cost per verification")
    ap.add_argument("--c-min", type=float, default=None, help="c_min in the schedule. Default: min(c_rew, c_ver).")
    ap.add_argument("--n-perm", type=int, default=10, help="Random permutations per problem")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cost-grid-points", type=int, default=100, help="Number of points on the cost grid")
    ap.add_argument('--task',              default='', help='Task label for plot titles (e.g. Math, Coding)')
    ap.add_argument('--model-name',        default='', help='Generation model name for plot titles')
    ap.add_argument('--reward-model-name', default='', help='Reward model name for plot titles')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c_min = args.c_min if args.c_min is not None else min(args.c_rew, args.c_ver)
    print(f"[config] c_rew={args.c_rew}, c_ver={args.c_ver}, c_min={c_min}")

    gens = load_generations(Path(args.generations))
    rewards = load_rewards(Path(args.rewards))
    print(f"[data] generations problems={len(gens)}, rewards problems={len(rewards)}")

    problems = build_problem_arrays(gens, rewards, args.reward_key)
    max_n = max(len(v[0]) for v in problems.values())
    min_cost = args.c_rew + args.c_ver
    max_cost = max_n * (args.c_rew + args.c_ver)
    cost_grid = np.logspace(np.log10(min_cost), np.log10(max_cost), args.cost_grid_points)

    adaptive_results, per_trial_mk, problem_ids = evaluate(
        problems=problems,
        c_rew=args.c_rew,
        c_ver=args.c_ver,
        c_min=c_min,
        n_perm=args.n_perm,
        seed=args.seed,
        cost_grid=cost_grid,
    )

    print("[eval] computing oracle minimum cost per trial...", flush=True)
    oracle_min_cost, oracle_M_arr, oracle_K_arr = oracle_min_success_cost(per_trial_mk, args.c_rew, args.c_ver)

    print("[eval] computing best fixed curve under budget...", flush=True)
    best_fixed_budget_curve, best_fixed_budget_mk, best_fixed_budget_actual_cost = best_fixed_curve_under_budget(
        per_trial_mk, args.c_rew, args.c_ver, cost_grid
    )

    # Save raw outputs
    with open(out_dir / "adaptive_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(adaptive_results[0].keys()))
        writer.writeheader()
        for row in adaptive_results:
            writer.writerow(row)

    np.save(out_dir / "per_trial_mk.npy", per_trial_mk)
    np.save(out_dir / "oracle_min_cost.npy", oracle_min_cost)
    np.save(out_dir / "best_fixed_budget_curve.npy", best_fixed_budget_curve)
    np.save(out_dir / "best_fixed_budget_actual_cost.npy", best_fixed_budget_actual_cost)
    np.save(out_dir / "cost_grid.npy", cost_grid)

    with open(out_dir / "best_fixed_budget_curve.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["budget", "best_success", "chosen_M", "chosen_K", "chosen_actual_cost"])
        for budget, succ, mk, actual_cost in zip(cost_grid, best_fixed_budget_curve, best_fixed_budget_mk, best_fixed_budget_actual_cost):
            writer.writerow([float(budget), float(succ), int(mk[0]), int(mk[1]), float(actual_cost)])

    with open(out_dir / "problem_ids.txt", "w") as f:
        for pid in problem_ids:
            f.write(pid + "\n")

    make_plots(
        adaptive_results=adaptive_results,
        per_trial_mk=per_trial_mk,
        oracle_min_cost=oracle_min_cost,
        oracle_M_arr=oracle_M_arr,
        oracle_K_arr=oracle_K_arr,
        best_fixed_budget_curve=best_fixed_budget_curve,
        best_fixed_budget_mk=best_fixed_budget_mk,
        best_fixed_budget_actual_cost=best_fixed_budget_actual_cost,
        cost_grid=cost_grid,
        problem_ids=problem_ids,
        c_rew=args.c_rew,
        c_ver=args.c_ver,
        out_dir=out_dir,
        task=args.task,
        model_name=args.model_name,
        reward_model_name=args.reward_model_name,
    )

    # Re-save enriched adaptive_results with oracle/fixed annotations
    with open(out_dir / "adaptive_results_enriched.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(adaptive_results[0].keys()))
        writer.writeheader()
        for row in adaptive_results:
            writer.writerow(row)

    print(f"\n[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
