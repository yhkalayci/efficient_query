"""
Compute AUC of CodeScaler reward vs ground-truth execution correctness on
HumanEval+ generations. Mirrors the math compute_auc.py but for scalar rewards.

Reports:
  - Pooled AUC across all (problem, sample) pairs, with bootstrap 95% CI
  - Per-problem AUC averaged across solvable problems (>=1 correct, >=1 wrong)
  - Score distribution by class

Usage:
  python compute_auc.py \\
    --generations out/generations.jsonl \\
    --rewards out/rewards.jsonl \\
    --bootstrap 1000
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def auc_safe(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def bootstrap_auc(y_true, y_score, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc_safe(y_true[idx], y_score[idx])
        if not np.isnan(a):
            aucs.append(a)
    if not aucs:
        return (float("nan"), float("nan"), float("nan"))
    aucs = np.asarray(aucs)
    return (float(np.mean(aucs)), float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--rewards", required=True,
                    help="Single rewards.jsonl. Pass multiple with commas to compare.")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    gens = load_jsonl(args.generations)
    gens_by_id = {g["id"]: g for g in gens}

    reward_files = args.rewards.split(",")

    for rf in reward_files:
        print(f"\n========== Rewards file: {rf} ==========", flush=True)
        rewards_list = load_jsonl(rf)
        rewards_by_id = {r["id"]: r for r in rewards_list}

        # Pool across problems
        pooled_y, pooled_s = [], []
        per_problem_aucs = []
        per_problem_stats = []
        n_with_signal = 0
        score_by_class = {0: [], 1: []}

        for pid, g in gens_by_id.items():
            if pid not in rewards_by_id:
                continue
            scores = {r["idx"]: r["r_score"] for r in rewards_by_id[pid]["rewards"]}
            y, s = [], []
            for samp in g["samples"]:
                if samp["idx"] not in scores:
                    continue
                y.append(int(samp["correct"]))
                s.append(scores[samp["idx"]])
            if not y:
                continue
            y = np.asarray(y)
            s = np.asarray(s)
            pooled_y.extend(y.tolist())
            pooled_s.extend(s.tolist())
            score_by_class[0].extend(s[y == 0].tolist())
            score_by_class[1].extend(s[y == 1].tolist())

            if y.sum() > 0 and y.sum() < len(y):
                a = auc_safe(y, s)
                per_problem_aucs.append(a)
                per_problem_stats.append((pid, a, int(y.sum()), len(y)))
                n_with_signal += 1

        pooled_y = np.asarray(pooled_y)
        pooled_s = np.asarray(pooled_s)

        print(f"Total samples pooled: {len(pooled_y)}")
        print(f"Pos rate (any test pass): {pooled_y.mean():.4f}")

        a, lo, hi = (float("nan"), float("nan"), float("nan"))
        if pooled_y.sum() > 0 and pooled_y.sum() < len(pooled_y):
            a_pt = auc_safe(pooled_y, pooled_s)
            a, lo, hi = bootstrap_auc(pooled_y, pooled_s, n_boot=args.bootstrap)
            print(f"Pooled AUC (point):  {a_pt:.4f}")
            print(f"Pooled AUC (boot):   {a:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
        else:
            print("Pooled AUC undefined (degenerate label distribution).")

        if per_problem_aucs:
            arr = np.asarray(per_problem_aucs)
            print(f"\nPer-problem AUC over {n_with_signal} problems with both classes:")
            print(f"  mean = {arr.mean():.4f}")
            print(f"  std  = {arr.std():.4f}")
            print(f"  median = {np.median(arr):.4f}")
            print(f"  q25  = {np.percentile(arr, 25):.4f}")
            print(f"  q75  = {np.percentile(arr, 75):.4f}")

        print(f"\nScore distribution by class:")
        if score_by_class[1]:
            print(f"  correct (n={len(score_by_class[1])}): "
                  f"mean={np.mean(score_by_class[1]):.4f}  "
                  f"std={np.std(score_by_class[1]):.4f}")
        if score_by_class[0]:
            print(f"  wrong   (n={len(score_by_class[0])}): "
                  f"mean={np.mean(score_by_class[0]):.4f}  "
                  f"std={np.std(score_by_class[0]):.4f}")

        # Worst and best per-problem AUCs for sanity
        if per_problem_stats:
            sorted_stats = sorted(per_problem_stats, key=lambda x: x[1])
            print(f"\n5 worst per-problem AUCs:")
            for pid, a, npos, n in sorted_stats[:5]:
                print(f"  {pid}  AUC={a:.3f}  pos={npos}/{n}")
            print(f"5 best per-problem AUCs:")
            for pid, a, npos, n in sorted_stats[-5:]:
                print(f"  {pid}  AUC={a:.3f}  pos={npos}/{n}")


if __name__ == "__main__":
    main()