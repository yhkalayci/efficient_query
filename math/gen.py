#!/usr/bin/env python3
"""
Generate N samples per HMMT Feb 2024 + 2025 problem with Qwen2.5-Math-7B (base)
and compute per-sample correctness and per-problem pass@k curves.

Usage (auto-detects all visible GPUs, uses bf16, max batch):
    python run_hmmt_sampling.py

With overrides:
    python run_hmmt_sampling.py --n-samples 512 --output-dir ./results

Install:
    pip install "vllm>=0.6.0" datasets transformers

Output (in --output-dir):
    generations.jsonl   one line per problem, every sample's pred + correctness
    per_problem.json    compact per-problem stats (n_correct / n_samples / p-hat)
    summary.json        aggregate pass@{1,2,4,...,N}, coverage, p-hat histogram
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams


# -------------------------- Few-shot CoT prompt --------------------------
# Two MATH-style worked examples (not from AIME/HMMT) to teach the \boxed{}
# output format. The explicit directive "Please reason step by step..." is the
# standard math-benchmark prompt used in Qwen2.5 / DeepSeek / AIME eval protocols
# (see Qwen's own "benchmarking guidelines" doc and deepseek-ai/DeepSeek-Math
# evaluation scripts). It encourages longer CoT without suppressing sampling
# diversity, so the pass@k tail is preserved.
FEWSHOT_PROMPT = r"""Please reason step by step, and put your final answer within \boxed{}.

Problem: Find the last three digits of $9^{105}$.
Solution: We compute $9^{105} \pmod{1000}$ via repeated squaring.
$9^2 = 81$
$9^4 = 81^2 = 6561 \equiv 561 \pmod{1000}$
$9^8 \equiv 561^2 = 314721 \equiv 721 \pmod{1000}$
$9^{16} \equiv 721^2 = 519841 \equiv 841 \pmod{1000}$
$9^{32} \equiv 841^2 = 707281 \equiv 281 \pmod{1000}$
$9^{64} \equiv 281^2 = 78961 \equiv 961 \pmod{1000}$
Since $105 = 64 + 32 + 8 + 1$, we have
$9^{105} \equiv 961 \cdot 281 \cdot 721 \cdot 9 \pmod{1000}$.
$961 \cdot 281 \equiv 41$, then $41 \cdot 721 \equiv 561$, then $561 \cdot 9 \equiv 49 \pmod{1000}$.
The answer is \boxed{49}.

Problem: How many positive integers $n$ less than $1000$ satisfy $\lfloor \log_2 n \rfloor \in \{2,4,6,8\}$?
Solution: We count $n$ with $\lfloor \log_2 n \rfloor = k$ for each even $k$ with $2^k < 1000$.
$k=2$: $4 \le n \le 7$, giving 4 values.
$k=4$: $16 \le n \le 31$, giving 16 values.
$k=6$: $64 \le n \le 127$, giving 64 values.
$k=8$: $256 \le n \le 511$, giving 256 values.
Total: $4 + 16 + 64 + 256 = 340$.
The answer is \boxed{340}.

Problem: {problem}
Please reason step by step, and put your final answer within \boxed{}.
Solution:"""


# -------------------------- Answer extraction --------------------------
def extract_boxed_answer(text: str):
    """Return content of the LAST \\boxed{...} in text (handles nested braces), or None."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    end = start
    while end < len(text):
        c = text[end]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:end].strip()
        end += 1
    return None  # unmatched


def normalize_answer(ans):
    """Normalize answer string for comparison."""
    if ans is None:
        return None
    s = ans.strip()
    # Strip simple LaTeX wrappers
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = s.replace(",", "").replace(" ", "")
    # Strip outer $...$
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    # Integer
    try:
        return str(int(s))
    except ValueError:
        pass
    # Float
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return f"{f:.10g}"
    except ValueError:
        pass
    # Reduced fraction a/b
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b != 0:
            g = math.gcd(abs(a), abs(b))
            a //= g
            b //= g
            if b < 0:
                a, b = -a, -b
            return f"{a}/{b}"
    return s


def answers_match(pred, gold):
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    if np_ is None or ng is None:
        return False
    return np_ == ng


# -------------------------- Dataset loading --------------------------
# HMMT Feb 2024 + Feb 2025 = 60 olympiad problems. Some have non-integer answers
# (fractions, closed forms); the parser normalizes simple LaTeX wrappers but
# problems with complicated closed-form answers may be effectively un-scorable.
HMMT_CANDIDATES = [
    # (repo_id, year, split)
    ("MathArena/hmmt_feb_2025", 2025, "train"),
    ("MathArena/hmmt_feb_2024", 2024, "train"),
]


def load_hmmt_problems(extra_repos=None):
    problems = []
    tried = list(HMMT_CANDIDATES)
    if extra_repos:
        tried = [(r, None, "train") for r in extra_repos] + tried

    for repo, year, split in tried:
        try:
            ds = load_dataset(repo, split=split)
        except Exception as e:
            print(f"[warn] could not load {repo}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        loaded_before = len(problems)
        for i, row in enumerate(ds):
            prob = row.get("problem") or row.get("question") or row.get("prompt")
            ans = row.get("answer") or row.get("gold_answer") or row.get("solution_answer")
            if prob is None or ans is None:
                continue
            problems.append({
                "id": f"{repo.split('/')[-1]}_{i:02d}",
                "year": year if year is not None else row.get("year"),
                "problem": str(prob).strip(),
                "answer": str(ans).strip(),
            })
        print(f"[data] {repo}: loaded {len(problems) - loaded_before} problems")
    return problems


# -------------------------- pass@k estimator --------------------------
def pass_at_k(n_total: int, n_correct: int, k: int) -> float:
    """Unbiased pass@k (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if n_total - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n_total - n_correct, k) / math.comb(n_total, k)


# -------------------------- Main --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--output-dir", default="./hmmt_results")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=0,
                        help="0 = use all visible GPUs")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=512,
                        help="Max concurrent sequences in vLLM (throughput knob)")
    parser.add_argument("--dtype", default="auto",
                        choices=["bfloat16", "float16", "auto"],
                        help="auto = bf16 on Ampere+, fp16 on V100/T4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", action="append", default=None,
                        help="Additional HF dataset repo to try (can pass multiple times)")
    parser.add_argument("--no-save-text", action="store_true",
                        help="Skip saving full generated text (default: text is saved)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # GPU config
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        sys.exit("[fatal] no CUDA GPUs detected. This script requires at least 1 GPU.")
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size > 0 else num_gpus
    print(f"[config] {num_gpus} GPU(s) visible, tensor_parallel_size = {tp_size}")
    for i in range(num_gpus):
        p = torch.cuda.get_device_properties(i)
        print(f"         GPU {i}: {p.name} ({p.total_memory / 2**30:.1f} GB)")

    # Load problems
    print("[data] loading HMMT Feb 2024 + 2025...")
    problems = load_hmmt_problems(extra_repos=args.dataset)
    if not problems:
        sys.exit(
            "[fatal] no problems loaded. Either the dataset repos require auth "
            "(run `huggingface-cli login`) or they have moved. Try passing "
            "--dataset <repo_id> with an HMMT repo that has `problem` and `answer` columns."
        )
    year_counts = Counter(p["year"] for p in problems)
    print(f"[data] total problems = {len(problems)}  ({dict(year_counts)})")

    # Build prompts. Use .replace() not .format() because FEWSHOT_PROMPT contains
    # literal { } characters from LaTeX (\boxed{...}) that format() would try to parse.
    prompts = [FEWSHOT_PROMPT.replace("{problem}", p["problem"]) for p in problems]

    # Initialize vLLM
    print(f"[model] loading {args.model}...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        seed=args.seed,
    )
    print(f"[model] loaded in {time.time() - t0:.1f}s")

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop=["\nProblem:", "Problem:"],
    )

    total_completions = args.n_samples * len(problems)
    print(f"[gen] generating {args.n_samples} x {len(problems)} = {total_completions:,} completions "
          f"(one problem at a time, live progress)", flush=True)

    results_path = Path(args.output_dir) / "generations.jsonl"
    per_problem_path = Path(args.output_dir) / "per_problem.json"
    summary_path = Path(args.output_dir) / "summary.json"

    per_problem = []
    tok_total = 0
    t0 = time.time()
    fout = open(results_path, "w", buffering=1)  # line-buffered so each problem flushes immediately

    for pi, prob in enumerate(problems):
        prob_t0 = time.time()
        out = llm.generate([prompts[pi]], sampling_params, use_tqdm=False)[0]

        gold = prob["answer"]
        samples = []
        n_correct = 0
        n_parsed = 0
        tok_counts = []
        for i, comp in enumerate(out.outputs):
            text = comp.text
            pred = extract_boxed_answer(text)
            correct = answers_match(pred, gold) if pred is not None else False
            if pred is not None:
                n_parsed += 1
            if correct:
                n_correct += 1
            tok_counts.append(len(comp.token_ids))
            sample = {
                "idx": i,
                "pred": pred,
                "correct": bool(correct),
                "num_tokens": len(comp.token_ids),
                "finish_reason": comp.finish_reason,
            }
            if not args.no_save_text:
                sample["text"] = text
            samples.append(sample)
        record = {
            "id": prob["id"],
            "year": prob["year"],
            "problem": prob["problem"],
            "answer": gold,
            "n_samples": len(out.outputs),
            "n_parsed": n_parsed,
            "n_correct": n_correct,
            "empirical_pass_rate": n_correct / len(out.outputs),
            "avg_tokens": sum(tok_counts) / len(tok_counts),
            "samples": samples,
        }
        fout.write(json.dumps(record) + "\n")
        per_problem.append({k: record[k] for k in
                            ["id", "year", "answer", "n_samples", "n_parsed",
                             "n_correct", "empirical_pass_rate", "avg_tokens"]})

        tok_total += sum(tok_counts)
        prob_elapsed = time.time() - prob_t0
        wall = time.time() - t0
        eta_sec = wall / (pi + 1) * (len(problems) - pi - 1)
        print(f"  [{pi + 1:>2d}/{len(problems)}] {prob['id']:<30s}  "
              f"{n_correct:>3d}/{len(out.outputs)} correct  "
              f"p̂={record['empirical_pass_rate']:.3f}  "
              f"parsed={n_parsed}/{len(out.outputs)}  "
              f"avg_tok={record['avg_tokens']:>5.0f}  "
              f"({prob_elapsed:.0f}s, ETA {eta_sec / 60:.0f}min)",
              flush=True)

    fout.close()
    elapsed = time.time() - t0
    print(f"[gen] done in {elapsed / 60:.1f} min  "
          f"({tok_total:,} tokens, {tok_total / max(elapsed, 1):.0f} tok/s)", flush=True)

    with open(per_problem_path, "w") as f:
        json.dump(per_problem, f, indent=2)

    # Aggregate pass@k for k in {1, 2, 4, 8, ..., N}
    ks = []
    k = 1
    while k <= args.n_samples:
        ks.append(k)
        k *= 2
    if ks[-1] != args.n_samples:
        ks.append(args.n_samples)

    aggregate = {}
    for k in ks:
        vals = [pass_at_k(r["n_samples"], r["n_correct"], k) for r in per_problem]
        aggregate[f"pass@{k}"] = sum(vals) / len(vals)

    coverage = sum(1 for r in per_problem if r["n_correct"] > 0) / len(per_problem)

    def bucket(p):
        if p == 0: return "0 (unsolved)"
        if p < 0.01: return "(0, 0.01)"
        if p < 0.05: return "[0.01, 0.05)"
        if p < 0.10: return "[0.05, 0.10)"
        if p < 0.25: return "[0.10, 0.25)"
        if p < 0.50: return "[0.25, 0.50)"
        if p < 0.75: return "[0.50, 0.75)"
        return "[0.75, 1.00]"

    hist = Counter(bucket(r["empirical_pass_rate"]) for r in per_problem)

    summary = {
        "model": args.model,
        "n_problems": len(problems),
        "n_samples_per_problem": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "generation_time_sec": elapsed,
        "total_tokens": tok_total,
        "aggregate_pass_at_k": aggregate,
        "coverage": coverage,
        "p_hat_histogram": dict(hist),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Aggregate pass@k ===")
    for k, v in aggregate.items():
        print(f"  {k:<12s} = {v:.4f}")
    print(f"  coverage     = {coverage:.4f}  "
          f"(fraction of problems with >=1 correct sample)")
    print("\n=== Per-problem p̂ histogram ===")
    for bk in ["0 (unsolved)", "(0, 0.01)", "[0.01, 0.05)", "[0.05, 0.10)",
               "[0.10, 0.25)", "[0.25, 0.50)", "[0.50, 0.75)", "[0.75, 1.00]"]:
        bar = "#" * hist.get(bk, 0)
        print(f"  {bk:<16s} {hist.get(bk, 0):>3d}  {bar}")

    print(f"\n[done] wrote:")
    print(f"  {results_path}")
    print(f"  {per_problem_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()