#!/usr/bin/env python3
"""
Generate N samples per LiveCodeBench problem with a code model via vLLM.

Loads problems from bzantium/livecodebench (HuggingFace, release_v2, split=test).
Writes generations.jsonl — one line per problem — carrying test cases through
so verify_lcb.py can run without re-downloading.

Output schema per line:
  {
    "id": "<task_id>",
    "problem": "<full prompt>",
    "answer": null,
    "difficulty": "easy"|"medium"|"hard",
    "platform": "leetcode"|"atcoder"|"codeforces",
    "test_type": "stdin"|"functional",
    "tests": [...],
    "fn_name": str | null,
    "starter_code": str,
    "n_samples": N,
    "n_correct": null,
    "empirical_pass_rate": null,
    "avg_tokens": float,
    "samples": [
      {"idx": 0, "text": "<raw>", "code": "<extracted>", "syntactic": bool,
       "n_tests_passed": null, "n_tests_total": null,
       "correct": null, "runtime_error": null, "exec_time_s": null,
       "num_tokens": int, "finish_reason": str},
      ...
    ]
  }

Usage:
    python coding/gen.py --model Qwen/Qwen2.5-Coder-3B --n-samples 512 --output-dir ./results/coding
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import pickle
import re
import sys
import time
import zlib
from pathlib import Path


# ----------------------------- LiveCodeBench loading -----------------------------

def _maybe_decode_private_tests(s):
    """Decode private test cases — may be JSON, or base64+zlib+pickle."""
    if not isinstance(s, str) or not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))
    except Exception:
        return []


def load_livecodebench(version_tag="release_v2", difficulty=None,
                       platforms=None, max_problems=-1):
    """Load LCB problems from bzantium/livecodebench HF mirror."""
    from datasets import load_dataset
    print(f"[data] loading bzantium/livecodebench {version_tag}", flush=True)
    ds = load_dataset("bzantium/livecodebench", version_tag, split="test")

    ONE_SHOT_FUNC = (
        "### Example\n"
        "Problem: Return the sum of two integers a and b.\n"
        "Starter code:\n"
        "```python\n"
        "class Solution:\n"
        "    def add(self, a: int, b: int) -> int:\n"
        "```\n"
        "Solution:\n"
        "```python\n"
        "class Solution:\n"
        "    def add(self, a: int, b: int) -> int:\n"
        "        return a + b\n"
        "```\n\n"
    )

    ONE_SHOT_STDIN = (
        "### Example\n"
        "Problem: Read two integers and print their sum.\n"
        "Solution:\n"
        "```python\n"
        "a, b = map(int, input().split())\n"
        "print(a + b)\n"
        "```\n\n"
    )

    out = []
    n_skipped = 0
    for row in ds:
        diff = (row.get("difficulty") or "unknown").lower()
        plat = (row.get("platform") or "unknown").lower()

        if difficulty is not None and diff not in difficulty:
            continue
        if platforms is not None and plat not in platforms:
            continue

        public_tc = json.loads(row.get("public_test_cases") or "[]")
        private_tc = _maybe_decode_private_tests(row.get("private_test_cases") or "")
        tests = list(public_tc) + list(private_tc)
        if not tests:
            n_skipped += 1
            continue

        fn_name = None
        try:
            meta = json.loads(row.get("metadata") or "{}")
            fn_name = meta.get("func_name") or None
        except Exception:
            pass

        starter = row.get("starter_code") or ""
        question = row["question_content"]

        test_type = (
            "functional"
            if any(t.get("testtype") == "functional" for t in tests)
            else "stdin"
        )

        if test_type == "functional" or starter.strip():
            prompt = (
                ONE_SHOT_FUNC
                + "### Problem\n"
                + question.strip()
                + ("\n\nStarter code:\n```python\n" + starter.strip() + "\n```"
                   if starter.strip() else "")
                + "\n\nSolution:\n```python\n"
            )
        else:
            prompt = (
                ONE_SHOT_STDIN
                + "### Problem\n"
                + question.strip()
                + "\n\nSolution:\n```python\n"
            )

        out.append({
            "id": row.get("question_id") or row.get("task_id") or f"lcb_{len(out):04d}",
            "problem": prompt,
            "answer": None,
            "difficulty": diff,
            "platform": plat,
            "test_type": test_type,
            "tests": tests,
            "fn_name": fn_name,
            "starter_code": starter,
        })

        if max_problems > 0 and len(out) >= max_problems:
            break

    print(f"[data] loaded {len(out)} problems, skipped {n_skipped} (no tests)", flush=True)
    return out


# ----------------------------- Code extraction -----------------------------

def extract_python_code(text: str) -> str:
    """Extract the first ```python ... ``` block, or the whole text if none."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def is_syntactically_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ----------------------------- vLLM engine -----------------------------

def load_vllm_engine(model_name, dtype="float16", tp=1,
                     max_model_len=4096, gpu_memory_utilization=0.9):
    from vllm import LLM
    print(f"[load] loading {model_name} dtype={dtype} tp={tp}", flush=True)
    return LLM(
        model=model_name,
        dtype=dtype,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        seed=42,
    )


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate code samples for LiveCodeBench problems."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument("--output-dir", default="./results/coding")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=0,
                        help="0 = use all visible GPUs")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "auto"])
    parser.add_argument("--difficulty", nargs="+", default=["medium", "hard"],
                        help="Filter by difficulty: easy medium hard")
    parser.add_argument("--platforms", nargs="+", default=None,
                        help="Filter by platform: leetcode atcoder codeforces")
    parser.add_argument("--max-problems", type=int, default=-1,
                        help="Limit number of problems (smoke test)")
    parser.add_argument("--version-tag", default="release_v2",
                        help="LCB HF dataset version tag")
    args = parser.parse_args()

    import torch
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        sys.exit("[fatal] no CUDA GPUs detected.")
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size > 0 else num_gpus
    print(f"[config] {num_gpus} GPU(s), tensor_parallel_size={tp_size}", flush=True)

    problems = load_livecodebench(
        version_tag=args.version_tag,
        difficulty=args.difficulty,
        platforms=args.platforms,
        max_problems=args.max_problems,
    )
    if not problems:
        sys.exit("[fatal] no problems loaded.")

    from vllm import SamplingParams
    llm = load_vllm_engine(
        args.model,
        dtype=args.dtype,
        tp=tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["```", "\n### "],
    )

    out_path = out_dir / "generations.jsonl"
    t_start = time.time()
    tok_total = 0

    with open(out_path, "w", buffering=1) as fout:
        for pi, prob in enumerate(problems):
            pt0 = time.time()
            outputs = llm.generate([prob["problem"]], sampling_params, use_tqdm=False)[0]

            samples = []
            tok_counts = []
            for i, comp in enumerate(outputs.outputs):
                text = comp.text
                code = extract_python_code(text)
                syntactic = is_syntactically_valid(code)
                tok_counts.append(len(comp.token_ids))
                samples.append({
                    "idx": i,
                    "text": text,
                    "code": code,
                    "syntactic": syntactic,
                    "n_tests_passed": None,
                    "n_tests_total": None,
                    "correct": None,
                    "runtime_error": None,
                    "exec_time_s": None,
                    "num_tokens": len(comp.token_ids),
                    "finish_reason": comp.finish_reason,
                })

            avg_tok = sum(tok_counts) / len(tok_counts)
            tok_total += sum(tok_counts)

            record = {
                "id": prob["id"],
                "problem": prob["problem"],
                "answer": None,
                "difficulty": prob["difficulty"],
                "platform": prob["platform"],
                "test_type": prob["test_type"],
                "tests": prob["tests"],
                "fn_name": prob["fn_name"],
                "starter_code": prob["starter_code"],
                "n_samples": len(samples),
                "n_correct": None,
                "empirical_pass_rate": None,
                "avg_tokens": avg_tok,
                "samples": samples,
            }
            fout.write(json.dumps(record) + "\n")

            elapsed = time.time() - t_start
            eta_sec = elapsed / (pi + 1) * (len(problems) - pi - 1)
            n_syn = sum(s["syntactic"] for s in samples)
            print(
                f"  [{pi + 1:>4d}/{len(problems)}] {prob['id']:<35s}"
                f"  syntactic={n_syn}/{len(samples)}"
                f"  avg_tok={avg_tok:>5.0f}"
                f"  ({time.time() - pt0:.0f}s, ETA {eta_sec / 60:.0f}min)",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(
        f"[gen] done in {elapsed / 60:.1f} min  "
        f"({tok_total:,} tokens, {tok_total / max(elapsed, 1):.0f} tok/s)",
        flush=True,
    )
    print(f"[done] wrote {len(problems)} problems to {out_path}")


if __name__ == "__main__":
    main()
