#!/usr/bin/env python3
"""
Filter a generations.jsonl to keep only problems with at least one correct sample.

These are the only problems where reward-model evaluation is meaningful: if every
sample is wrong, there is no positive class and AUC is undefined. If every sample
is right (rare on HMMT), there is no negative class and AUC is still undefined.
We keep problems where 0 < n_correct < n_samples.

Usage:
  python filter_solvable.py --input run/generations.jsonl --output run/generations_solvable.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to generations.jsonl")
    ap.add_argument("--output", required=True,
                    help="path to write filtered generations.jsonl")
    ap.add_argument("--min-correct", type=int, default=1,
                    help="keep problems with at least this many correct samples")
    ap.add_argument("--max-correct-frac", type=float, default=1.0,
                    help="drop problems where more than this fraction of samples are correct "
                         "(1.0 disables; use e.g. 0.99 to exclude trivially solved problems)")
    ap.add_argument("--write-ids", default=None,
                    help="optional path to write a plain text file of kept problem IDs")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[fatal] input not found: {in_path}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = 0
    kept_ids = []
    all_stats = []
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            total += 1
            samples = rec["samples"]
            n = len(samples)
            n_correct = sum(1 for s in samples if s.get("correct"))
            frac = n_correct / n if n > 0 else 0.0
            all_stats.append((rec["id"], n, n_correct, frac))
            if n_correct < args.min_correct:
                continue
            if frac > args.max_correct_frac:
                continue
            fout.write(json.dumps(rec) + "\n")
            kept_ids.append(rec["id"])
            kept += 1

    if args.write_ids:
        with open(args.write_ids, "w") as f:
            for pid in kept_ids:
                f.write(pid + "\n")

    print(f"[filter] total problems:          {total}")
    print(f"[filter] kept (solvable & not trivially-solved): {kept}")
    print(f"[filter] dropped (n_correct < {args.min_correct}): "
          f"{sum(1 for _, _, nc, _ in all_stats if nc < args.min_correct)}")
    if args.max_correct_frac < 1.0:
        print(f"[filter] dropped (frac > {args.max_correct_frac}): "
              f"{sum(1 for _, _, _, f in all_stats if f > args.max_correct_frac)}")

    # Distribution summary
    print(f"\n[filter] per-problem n_correct distribution (kept only):")
    bins = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 10000)]
    for lo, hi in bins:
        cnt = sum(1 for pid, _, nc, _ in all_stats if lo <= nc < hi and pid in set(kept_ids))
        print(f"           {lo:3d} <= n_correct < {hi:5d}: {cnt}")

    print(f"\n[done] wrote {kept} problems -> {out_path}")


if __name__ == "__main__":
    main()