#!/usr/bin/env python3
"""
Calibrate a math subset so a BASE math model has low pass@30 but high pass@200,
then run a final large-sample evaluation on that filtered subset.

Design goals:
- Prefer BASE math models over instruction-tuned chat models for large-k pass@k.
- Prefer MATH-style datasets over raw olympiad sets for 7B-class models.
- Use pilot -> filter -> final to explicitly target the steep pass@k regime.
- Support both ordinary Hugging Face datasets and multi-config datasets such as
  EleutherAI/hendrycks_math.
- Be robust to schema variation across math datasets.
- Be debuggable: print why rows are being filtered out.

Example:
    python run_math_passk_calibrated_v4.py \
        --model Qwen/Qwen2.5-Math-7B \
        --dataset hendrycks_math \
        --split test \
        --subjects algebra geometry number_theory \
        --candidate-limit 256 \
        --pilot-samples 64 \
        --final-samples 256

Install:
    pip install "vllm>=0.6.0" datasets transformers torch

Outputs in --output-dir:
    pilot_generations.jsonl
    pilot_summary.json
    selected_problems.json
    final_generations.jsonl
    final_per_problem.json
    final_summary.json
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from vllm import LLM, SamplingParams


COMPLETION_TEMPLATE = r"""Problem:
{problem}

Solution:
"""

MINIMAL_COT_TEMPLATE = r"""Problem:
{problem}

Please solve the problem carefully. Show your reasoning, and put the final answer in \boxed{{}}.

Solution:
"""

FEWSHOT_TEMPLATE = r"""Problem:
Find the last three digits of $9^{105}$.

Solution:
We compute powers modulo $1000$ by repeated squaring.
$9^2 = 81$.
$9^4 \equiv 81^2 = 6561 \equiv 561 \pmod{{1000}}$.
$9^8 \equiv 561^2 = 314721 \equiv 721 \pmod{{1000}}$.
$9^{{16}} \equiv 721^2 = 519841 \equiv 841 \pmod{{1000}}$.
$9^{{32}} \equiv 841^2 = 707281 \equiv 281 \pmod{{1000}}$.
$9^{{64}} \equiv 281^2 = 78961 \equiv 961 \pmod{{1000}}$.
Since $105 = 64 + 32 + 8 + 1$,
$9^{{105}} \equiv 961 \cdot 281 \cdot 721 \cdot 9 \equiv 49 \pmod{{1000}}$.
Therefore the answer is \boxed{{49}}.

Problem:
How many positive integers $n < 1000$ satisfy $\lfloor \log_2 n \rfloor \in \{{2,4,6,8\}}$?

Solution:
For $\lfloor \log_2 n \rfloor = k$, the valid integers are $2^k \le n \le 2^{{k+1}} - 1$, so there are $2^k$ such integers.
Thus the count is
$2^2 + 2^4 + 2^6 + 2^8 = 4 + 16 + 64 + 256 = 340$.
Therefore the answer is \boxed{{340}}.

Problem:
{problem}

Please solve the problem carefully. Show your reasoning, and put the final answer in \boxed{{}}.

Solution:
"""


def make_prompt(problem: str, prompt_style: str) -> str:
    if prompt_style == "completion":
        return COMPLETION_TEMPLATE.format(problem=problem)
    if prompt_style == "minimal_cot":
        return MINIMAL_COT_TEMPLATE.format(problem=problem)
    if prompt_style == "fewshot":
        return FEWSHOT_TEMPLATE.format(problem=problem)
    raise ValueError(f"Unknown prompt_style: {prompt_style}")


def extract_last_boxed(text: str):
    """Return content of the last \\boxed{...}, handling nested braces."""
    if text is None:
        return None
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def strip_latex_wrappers(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\operatorname\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    return s


def normalize_answer(ans):
    """Normalize common numeric/string answer formats for exact-match style scoring."""
    if ans is None:
        return None

    s = strip_latex_wrappers(str(ans)).strip()

    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()

    s = s.replace(",", "")
    s = s.replace(" ", "")
    s = s.rstrip(".;,")

    try:
        return str(int(s))
    except ValueError:
        pass

    try:
        value = float(s)
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    except ValueError:
        pass

    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        if b != 0:
            g = math.gcd(abs(a), abs(b))
            a //= g
            b //= g
            if b < 0:
                a = -a
                b = -b
            return f"{a}/{b}"

    return s


def answers_match(pred, gold):
    npred = normalize_answer(pred)
    ngold = normalize_answer(gold)
    if npred is None or ngold is None:
        return False
    return npred == ngold


def parse_level_value(level):
    """
    Normalize many possible level formats to a simple digit string when possible.
    Examples:
        5 -> "5"
        "5" -> "5"
        "Level 5" -> "5"
        "level 4" -> "4"
    Falls back to the stripped original string if no digit is found.
    """
    if level is None:
        return None
    s = str(level).strip()
    m = re.search(r"(\d+)", s)
    if m:
        return m.group(1)
    return s




def normalize_subject_value(subject):
    """
    Normalize subject/config names so dataset values like 'Algebra' match CLI
    values like 'algebra' and config names like 'number_theory'.
    """
    if subject is None:
        return None
    s = str(subject).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s


def pass_at_k(n_total: int, n_correct: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k)/C(n, k)."""
    if k <= 0 or n_total <= 0 or n_correct <= 0:
        return 0.0
    if k > n_total:
        k = n_total
    if n_total - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n_total - n_correct, k) / math.comb(n_total, k)


def bernoulli_curve(p: float, k: int) -> float:
    """Approximate pass@k from per-sample success probability p: 1 - (1-p)^k."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return 1.0 - (1.0 - p) ** k


def maybe_get(row, *keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def dataset_repo_from_name(name: str) -> str:
    presets = {
        "hendrycks_math": "EleutherAI/hendrycks_math",
        "math500": "HuggingFaceH4/MATH-500",
    }
    return presets.get(name, name)


def config_names_safe(repo: str):
    try:
        return get_dataset_config_names(repo)
    except Exception:
        return []


def extract_final_answer_from_text(text):
    """
    Try several common patterns for final answers in math datasets.
    Returns a string or None.
    """
    if text is None:
        return None

    text = str(text).strip()

    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed

    patterns = [
        r"(?:final answer is|answer is)\s*\$?([^$\n\.]+)\$?",
        r"(?:therefore|thus|hence)[^.\n]*?\$([^$]+)\$",
        r"(?:therefore|thus|hence)[^.\n]*?([A-Za-z0-9/\-+]+)\s*$",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            return str(matches[-1]).strip()

    tail_patterns = [
        r"\$([^$]+)\$\s*$",
        r"([\-]?\d+(?:/\d+)?)\s*$",
    ]
    for pat in tail_patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()

    return None


def derive_gold_answer(row):
    """
    Robust gold-answer extraction:
    1. Prefer explicit answer-like fields.
    2. Fall back to extracting from solution text.
    """
    direct = maybe_get(row, "answer", "final_answer", "gold_answer", "solution_answer")
    if direct is not None:
        direct = str(direct).strip()
        extracted = extract_final_answer_from_text(direct)
        if extracted is not None:
            return extracted
        return direct

    solution = maybe_get(row, "solution", "full_solution")
    extracted = extract_final_answer_from_text(solution)
    if extracted is not None:
        return extracted

    return None


def load_math_problems(dataset_name: str, split: str, levels=None, subjects=None, limit=None, seed=42):
    repo = dataset_repo_from_name(dataset_name)

    level_filter = set(str(x) for x in levels) if levels is not None else None
    subject_filter = set(normalize_subject_value(x) for x in subjects) if subjects is not None else None

    cfgs = config_names_safe(repo)
    datasets_to_merge = []

    if cfgs:
        chosen_cfgs = cfgs
        if subject_filter is not None:
            chosen_cfgs = [cfg for cfg in cfgs if cfg in subject_filter]

        if not chosen_cfgs:
            raise ValueError(
                f"No dataset configs matched subjects={subjects}. Available configs: {cfgs}"
            )

        for cfg in chosen_cfgs:
            ds_cfg = load_dataset(repo, cfg, split=split)
            ds_cfg = ds_cfg.add_column("__config_name__", [cfg] * len(ds_cfg))
            datasets_to_merge.append(ds_cfg)

        ds = concatenate_datasets(datasets_to_merge)
    else:
        ds = load_dataset(repo, split=split)

    print(f"[debug] dataset rows before filtering: {len(ds)}")
    if len(ds) > 0:
        first = ds[0]
        print(f"[debug] columns: {list(first.keys())}")
        print(f"[debug] first row level raw: {maybe_get(first, 'level', 'difficulty')}")
        first_subject_raw = maybe_get(first, 'subject', 'type', 'category', '__config_name__')
        print(f"[debug] first row subject raw: {first_subject_raw}")
        print(f"[debug] first row subject normalized: {normalize_subject_value(first_subject_raw)}")
        print(f"[debug] first row has direct answer field: {maybe_get(first, 'answer', 'final_answer', 'gold_answer', 'solution_answer') is not None}")
        print(f"[debug] first row has solution field: {maybe_get(first, 'solution', 'full_solution') is not None}")

    problems = []
    skipped_no_problem = 0
    skipped_no_answer = 0
    skipped_level = 0
    skipped_subject = 0

    for idx, row in enumerate(ds):
        problem = maybe_get(row, "problem", "question", "prompt")
        answer = derive_gold_answer(row)
        solution = maybe_get(row, "solution", "full_solution")
        level_raw = maybe_get(row, "level", "difficulty")
        subject = maybe_get(row, "subject", "type", "category", "__config_name__")

        if problem is None:
            skipped_no_problem += 1
            continue

        if answer is None:
            skipped_no_answer += 1
            continue

        level_str = parse_level_value(level_raw)
        subject_str = normalize_subject_value(subject)

        if level_filter is not None:
            if level_str is None or level_str not in level_filter:
                skipped_level += 1
                continue

        if subject_filter is not None:
            if subject_str is None or subject_str not in subject_filter:
                skipped_subject += 1
                continue

        problems.append({
            "id": f"{dataset_name}_{split}_{idx:05d}",
            "dataset": dataset_name,
            "split": split,
            "problem": str(problem).strip(),
            "answer": str(answer).strip(),
            "solution": None if solution is None else str(solution).strip(),
            "level": level_str,
            "subject": subject_str,
        })

    print(f"[debug] kept problems: {len(problems)}")
    print(f"[debug] skipped_no_problem: {skipped_no_problem}")
    print(f"[debug] skipped_no_answer: {skipped_no_answer}")
    print(f"[debug] skipped_level: {skipped_level}")
    print(f"[debug] skipped_subject: {skipped_subject}")

    rng = random.Random(seed)
    if limit is not None and len(problems) > limit:
        rng.shuffle(problems)
        problems = problems[:limit]

    return problems


def build_llm(args):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        sys.exit("[fatal] no CUDA GPUs detected. This script requires at least 1 GPU.")

    tp_size = args.tensor_parallel_size if args.tensor_parallel_size > 0 else num_gpus

    print(f"[config] {num_gpus} GPU(s) visible, tensor_parallel_size = {tp_size}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"         GPU {i}: {props.name} ({props.total_memory / 2**30:.1f} GB)")

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
    return llm


def generate_one_problem(llm, prompt, n_samples, temperature, top_p, max_tokens, seed):
    params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        stop=["\nProblem:", "\n\nProblem:"],
    )
    result = llm.generate([prompt], params, use_tqdm=False)[0]
    return result.outputs


def summarize_outputs(outputs, gold_answer, save_text):
    samples = []
    n_correct = 0
    n_parsed = 0
    total_tokens = 0

    for idx, out in enumerate(outputs):
        text = out.text
        pred = extract_last_boxed(text)
        correct = answers_match(pred, gold_answer) if pred is not None else False

        if pred is not None:
            n_parsed += 1
        if correct:
            n_correct += 1

        tok_count = len(out.token_ids)
        total_tokens += tok_count

        row = {
            "idx": idx,
            "pred": pred,
            "correct": bool(correct),
            "num_tokens": tok_count,
            "finish_reason": out.finish_reason,
        }
        if save_text:
            row["text"] = text
        samples.append(row)

    avg_tokens = total_tokens / len(outputs) if outputs else 0.0
    return {
        "samples": samples,
        "n_correct": n_correct,
        "n_parsed": n_parsed,
        "avg_tokens": avg_tokens,
        "total_tokens": total_tokens,
    }


def run_stage(
    llm,
    problems,
    stage_name,
    output_dir,
    n_samples,
    temperature,
    top_p,
    max_tokens,
    prompt_style,
    base_seed,
    save_text,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generations_path = output_dir / f"{stage_name}_generations.jsonl"
    summary_path = output_dir / f"{stage_name}_summary.json"

    records = []
    total_tokens = 0
    start_time = time.time()

    with open(generations_path, "w", buffering=1) as fout:
        for i, prob in enumerate(problems):
            prob_start = time.time()

            prompt = make_prompt(prob["problem"], prompt_style)
            outputs = generate_one_problem(
                llm=llm,
                prompt=prompt,
                n_samples=n_samples,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=base_seed + i,
            )

            stats = summarize_outputs(outputs, prob["answer"], save_text=save_text)

            record = {
                "id": prob["id"],
                "dataset": prob["dataset"],
                "split": prob["split"],
                "level": prob["level"],
                "subject": prob["subject"],
                "problem": prob["problem"],
                "answer": prob["answer"],
                "n_samples": n_samples,
                "n_parsed": stats["n_parsed"],
                "n_correct": stats["n_correct"],
                "empirical_p_hat": stats["n_correct"] / n_samples if n_samples else 0.0,
                "avg_tokens": stats["avg_tokens"],
                "samples": stats["samples"],
            }
            fout.write(json.dumps(record) + "\n")
            records.append(record)
            total_tokens += stats["total_tokens"]

            wall_elapsed = time.time() - start_time
            done = i + 1
            remaining = len(problems) - done
            eta_sec = (wall_elapsed / done) * remaining if done > 0 else 0.0
            prob_elapsed = time.time() - prob_start

            print(
                f"  [{stage_name} {done:>3d}/{len(problems)}] {prob['id']:<28s} "
                f"{record['n_correct']:>3d}/{n_samples} correct  "
                f"p̂={record['empirical_p_hat']:.3f}  "
                f"parsed={record['n_parsed']}/{n_samples}  "
                f"avg_tok={record['avg_tokens']:.0f}  "
                f"({prob_elapsed:.0f}s, ETA {eta_sec / 60:.0f}min)",
                flush=True,
            )

    elapsed = time.time() - start_time

    ks = []
    k = 1
    while k <= n_samples:
        ks.append(k)
        k *= 2
    if not ks or ks[-1] != n_samples:
        ks.append(n_samples)

    aggregate = {}
    for k in ks:
        vals = [pass_at_k(r["n_samples"], r["n_correct"], k) for r in records]
        aggregate[f"pass@{k}"] = sum(vals) / len(vals) if vals else 0.0

    summary = {
        "stage": stage_name,
        "n_problems": len(records),
        "n_samples_per_problem": n_samples,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "prompt_style": prompt_style,
        "generation_time_sec": elapsed,
        "total_tokens": total_tokens,
        "aggregate_pass_at_k": aggregate,
        "coverage": (
            sum(1 for r in records if r["n_correct"] > 0) / len(records)
            if records else 0.0
        ),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return records, summary


def select_problems_from_pilot(pilot_records, args):
    selected = []
    diagnostics = []

    target_mid = 0.5 * (args.target_pass_low_min + args.target_pass_low_max)

    for rec in pilot_records:
        p_hat = rec["empirical_p_hat"]
        est_low = bernoulli_curve(p_hat, args.target_k_low)
        est_high = bernoulli_curve(p_hat, args.target_k_high)

        keep = (
            args.target_pass_low_min <= est_low <= args.target_pass_low_max
            and args.target_pass_high_min <= est_high <= args.target_pass_high_max
        )

        row = {
            "id": rec["id"],
            "dataset": rec["dataset"],
            "split": rec["split"],
            "level": rec["level"],
            "subject": rec["subject"],
            "problem": rec["problem"],
            "answer": rec["answer"],
            "pilot_n_samples": rec["n_samples"],
            "pilot_n_correct": rec["n_correct"],
            "pilot_p_hat": p_hat,
            f"estimated_pass@{args.target_k_low}": est_low,
            f"estimated_pass@{args.target_k_high}": est_high,
            "selected": keep,
        }
        diagnostics.append(row)
        if keep:
            selected.append(row)

    selected.sort(
        key=lambda x: abs(x[f"estimated_pass@{args.target_k_low}"] - target_mid)
    )
    return selected, diagnostics


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument(
        "--dataset",
        default="hendrycks_math",
        help="Preset name (hendrycks_math, math500) or a custom HF repo id.",
    )
    parser.add_argument("--split", default="test")

    parser.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help="Optional level filter. Example: --levels 4 5. Leave unset to disable.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject/config filter. For hendrycks_math this can be algebra geometry number_theory etc.",
    )
    parser.add_argument("--candidate-limit", type=int, default=256)
    parser.add_argument("--selected-limit", type=int, default=64)
    parser.add_argument("--output-dir", default="./math_passk_calibrated")

    parser.add_argument("--pilot-samples", type=int, default=64)
    parser.add_argument("--pilot-temperature", type=float, default=0.9)
    parser.add_argument("--pilot-top-p", type=float, default=0.95)

    parser.add_argument("--final-samples", type=int, default=256)
    parser.add_argument("--final-temperature", type=float, default=0.95)
    parser.add_argument("--final-top-p", type=float, default=0.95)

    parser.add_argument("--target-k-low", type=int, default=30)
    parser.add_argument("--target-k-high", type=int, default=200)
    parser.add_argument("--target-pass-low-min", type=float, default=0.05)
    parser.add_argument("--target-pass-low-max", type=float, default=0.20)
    parser.add_argument("--target-pass-high-min", type=float, default=0.40)
    parser.add_argument("--target-pass-high-max", type=float, default=0.75)

    parser.add_argument(
        "--prompt-style",
        choices=["completion", "minimal_cot", "fewshot"],
        default="completion",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save-text", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[data] loading candidate pool...")
    problems = load_math_problems(
        dataset_name=args.dataset,
        split=args.split,
        levels=args.levels,
        subjects=args.subjects,
        limit=args.candidate_limit,
        seed=args.seed,
    )
    if not problems:
        sys.exit(
            "[fatal] no problems loaded after filtering. "
            "Try removing --levels, or specify --subjects for hendrycks_math. "
            "Check the [debug] lines above."
        )

    print(f"[data] loaded {len(problems)} candidate problems from {args.dataset}/{args.split}")
    print(f"[data] level histogram: {dict(Counter(p['level'] for p in problems))}")
    print(f"[data] subject histogram: {dict(Counter(p['subject'] for p in problems))}")

    llm = build_llm(args)

    print("[pilot] running pilot stage...")
    pilot_records, _pilot_summary = run_stage(
        llm=llm,
        problems=problems,
        stage_name="pilot",
        output_dir=args.output_dir,
        n_samples=args.pilot_samples,
        temperature=args.pilot_temperature,
        top_p=args.pilot_top_p,
        max_tokens=args.max_tokens,
        prompt_style=args.prompt_style,
        base_seed=args.seed,
        save_text=not args.no_save_text,
    )

    selected, diagnostics = select_problems_from_pilot(pilot_records, args)
    selected_path = Path(args.output_dir) / "selected_problems.json"
    write_json(selected_path, diagnostics)

    if not selected:
        print("[select] no problems matched the requested pass@k window.")
        print("[select] Try increasing --candidate-limit, widening target windows,")
        print("[select] changing subjects, or changing temperatures.")
        return

    if len(selected) > args.selected_limit:
        selected = selected[:args.selected_limit]

    selected_ids = {row["id"] for row in selected}
    final_problems = [p for p in problems if p["id"] in selected_ids]

    print(f"[select] selected {len(final_problems)} problems for the final stage")

    print("[final] running final stage...")
    final_records, final_summary = run_stage(
        llm=llm,
        problems=final_problems,
        stage_name="final",
        output_dir=args.output_dir,
        n_samples=args.final_samples,
        temperature=args.final_temperature,
        top_p=args.final_top_p,
        max_tokens=args.max_tokens,
        prompt_style=args.prompt_style,
        base_seed=args.seed + 100000,
        save_text=not args.no_save_text,
    )

    final_per_problem = []
    for rec in final_records:
        final_per_problem.append({
            "id": rec["id"],
            "dataset": rec["dataset"],
            "split": rec["split"],
            "level": rec["level"],
            "subject": rec["subject"],
            "answer": rec["answer"],
            "n_samples": rec["n_samples"],
            "n_parsed": rec["n_parsed"],
            "n_correct": rec["n_correct"],
            "empirical_p_hat": rec["empirical_p_hat"],
            "avg_tokens": rec["avg_tokens"],
            f"pass@{args.target_k_low}": pass_at_k(rec["n_samples"], rec["n_correct"], args.target_k_low),
            f"pass@{args.target_k_high}": pass_at_k(rec["n_samples"], rec["n_correct"], args.target_k_high),
        })

    final_per_problem_path = Path(args.output_dir) / "final_per_problem.json"
    write_json(final_per_problem_path, final_per_problem)

    print("\n=== Final aggregate pass@k ===")
    for key, value in final_summary["aggregate_pass_at_k"].items():
        print(f"  {key:<12s} = {value:.4f}")
    print(f"  coverage     = {final_summary['coverage']:.4f}")

    print("\n[done] wrote:")
    print(f"  {Path(args.output_dir) / 'pilot_generations.jsonl'}")
    print(f"  {Path(args.output_dir) / 'pilot_summary.json'}")
    print(f"  {selected_path}")
    print(f"  {Path(args.output_dir) / 'final_generations.jsonl'}")
    print(f"  {final_per_problem_path}")
    print(f"  {Path(args.output_dir) / 'final_summary.json'}")


if __name__ == "__main__":
    main()
