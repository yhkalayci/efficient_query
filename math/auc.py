#!/usr/bin/env python3
"""
Compute AUC of each reward aggregation (r_min, r_mean, r_last, r_prod) versus
ground-truth correctness.

Inputs:
  --generations: path to generations.jsonl (has `samples[i].correct` as ground truth)
  --rewards:     one or more paths to rewards.jsonl files (each has per-sample rewards)
                 Pass multiple with --rewards file1.jsonl --rewards file2.jsonl for
                 head-to-head comparison (e.g. Qwen-PRM-7B vs Qwen-PRM-72B).

Reports:
  - Aggregate AUC across all solvable problems (pooled samples)
  - Per-problem AUC (averaged across problems with at least one correct + one incorrect)
  - Bootstrap 95% CI on aggregate AUC
  - Score distribution stats for correct vs incorrect samples

Usage:
  python compute_auc.py \
    --generations run/generations.jsonl \
    --rewards run/rewards_prm7b.jsonl \
    --rewards run/rewards_prm72b.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    sys.exit("[fatal] scikit-learn required. pip install scikit-learn")


AGGREGATIONS = ["r_min", "r_mean", "r_last", "r_prod"]


def load_rewards(path: Path) -> Dict[str, Dict[int, dict]]:
    """Return {problem_id: {sample_idx: reward_record}}."""
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = {r["idx"]: r for r in rec["rewards"]}
    return out


def load_generations(path: Path) -> Dict[str, List[dict]]:
    """Return {problem_id: [sample_records]}."""
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = rec["samples"]
    return out


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                     n_bootstrap: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """Stratified bootstrap 95% CI for AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_t = y_true[idx]
        y_s = y_score[idx]
        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            continue
        aucs.append(roc_auc_score(y_t, y_s))
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)))


def evaluate_one(rewards: Dict[str, Dict[int, dict]],
                 gens: Dict[str, List[dict]],
                 label: str):
    """Compute all AUCs for one reward-model output."""
    # Flatten: (y_true, y_score_min, y_score_mean, y_score_last, y_score_prod, problem_id)
    rows = []
    n_missing = 0
    for pid, samples in gens.items():
        rmap = rewards.get(pid, {})
        for i, s in enumerate(samples):
            r = rmap.get(i)
            if r is None or r.get("r_min") is None:
                n_missing += 1
                continue
            rows.append((
                int(bool(s.get("correct"))),
                r["r_min"], r["r_mean"], r["r_last"], r["r_prod"],
                pid,
            ))

    if not rows:
        print(f"\n=== {label} ===  NO DATA")
        return

    y_true = np.array([r[0] for r in rows])
    scores = {
        "r_min":  np.array([r[1] for r in rows]),
        "r_mean": np.array([r[2] for r in rows]),
        "r_last": np.array([r[3] for r in rows]),
        "r_prod": np.array([r[4] for r in rows]),
    }
    problem_ids = [r[5] for r in rows]

    n_total = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_total - n_pos

    print(f"\n=== {label} ===")
    print(f"  samples scored:    {n_total:,}  ({n_pos} correct / {n_neg} incorrect)")
    if n_missing:
        print(f"  samples skipped:   {n_missing} (no reward or null r_min)")
    if n_pos == 0 or n_neg == 0:
        print("  [AUC undefined] need both correct and incorrect samples")
        return

    # Aggregate (pooled across all problems)
    print(f"\n  Aggregate AUC (pooled, n={n_total:,}):")
    print(f"    {'agg':<10s} {'AUC':>7s}  {'95% CI':>16s}  {'mean|+':>7s} {'mean|-':>7s}")
    for agg in AGGREGATIONS:
        y_score = scores[agg]
        auc = roc_auc_score(y_true, y_score)
        lo, hi = bootstrap_auc_ci(y_true, y_score, n_bootstrap=500)
        mean_pos = float(y_score[y_true == 1].mean())
        mean_neg = float(y_score[y_true == 0].mean())
        print(f"    {agg:<10s} {auc:.3f}   [{lo:.3f}, {hi:.3f}]    "
              f"{mean_pos:.3f}  {mean_neg:.3f}")

    # Per-problem AUC (only for problems with both classes)
    print(f"\n  Per-problem AUC (averaged across solvable problems):")
    by_problem: Dict[str, List[int]] = {}
    for i, pid in enumerate(problem_ids):
        by_problem.setdefault(pid, []).append(i)
    per_prob_aucs = {agg: [] for agg in AGGREGATIONS}
    n_usable = 0
    for pid, idxs in by_problem.items():
        yt = y_true[idxs]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        n_usable += 1
        for agg in AGGREGATIONS:
            ys = scores[agg][idxs]
            per_prob_aucs[agg].append(roc_auc_score(yt, ys))
    print(f"    n problems with both classes: {n_usable}")
    print(f"    {'agg':<10s} {'mean AUC':>9s} {'median':>7s} {'std':>7s}")
    for agg in AGGREGATIONS:
        arr = np.array(per_prob_aucs[agg])
        if len(arr) == 0:
            print(f"    {agg:<10s}  no data")
        else:
            print(f"    {agg:<10s} {arr.mean():.3f}     {np.median(arr):.3f}   {arr.std():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True,
                    help="generations.jsonl with ground-truth correctness")
    ap.add_argument("--rewards", action="append", required=True,
                    help="rewards.jsonl file; pass multiple times for comparison")
    ap.add_argument("--labels", default=None,
                    help="comma-separated display labels, one per --rewards "
                         "(default: use file names)")
    args = ap.parse_args()

    gens = load_generations(Path(args.generations))
    print(f"[data] {len(gens)} problems in generations file")

    reward_files = [Path(p) for p in args.rewards]
    if args.labels:
        labels = args.labels.split(",")
        if len(labels) != len(reward_files):
            sys.exit("[fatal] --labels count must match --rewards count")
    else:
        labels = [p.stem for p in reward_files]

    for lbl, path in zip(labels, reward_files):
        if not path.exists():
            print(f"[warn] reward file not found: {path}", file=sys.stderr)
            continue
        rewards = load_rewards(path)
        print(f"[data] {lbl}: {len(rewards)} problems in reward file")
        evaluate_one(rewards, gens, label=lbl)


if __name__ == "__main__":
    main()
