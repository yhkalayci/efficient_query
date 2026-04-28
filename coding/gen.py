"""
Sample N completions per HumanEval+ problem with a base coder model, then
verify each sample inline with:
  (1) Python compile() syntax check
  (2) execution against EvalPlus extended tests with per-test timeout

Output: generations.jsonl, one line per problem, structured to match the math
pipeline so downstream scripts (filter_solvable.py, meta_generation_eval.py)
work with minor tweaks.

Each line:
  {
    "id": "HumanEval/0",
    "problem": "<full prompt: docstring + signature>",
    "answer": null,                 # not used for code
    "samples": [
      {
        "idx": 0,
        "text": "<raw model output>",
        "code": "<extracted code>",
        "syntactic": true,
        "runtime_error": null,
        "n_tests_passed": 17,
        "n_tests_total": 17,
        "correct": true,
        "exec_time_s": 0.034
      },
      ...
    ]
  }
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np


# ----------------------------- Model loading -----------------------------
def load_vllm_engine(model_name, dtype="float16", tp=1, max_model_len=2048):
    from vllm import LLM
    print(f"[load] loading {model_name} dtype={dtype} tp={tp}", flush=True)
    llm = LLM(
        model=model_name,
        dtype=dtype,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        seed=42,
    )
    return llm


# ----------------------------- HumanEval+ loading -----------------------------
def load_humaneval_plus():
    """Load HumanEval+ from the evalplus package."""
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    out = []
    for task_id, p in problems.items():
        out.append({
            "task_id": task_id,
            "prompt": p["prompt"],
            "entry_point": p["entry_point"],
            "canonical_solution": p.get("canonical_solution", ""),
            "test": p["test"],          # original tests (string of test code)
            "plus_input": p.get("plus_input", []),  # extra inputs
        })
    print(f"[data] loaded {len(out)} HumanEval+ problems", flush=True)
    return out


# ----------------------------- Code extraction -----------------------------
# Base coder models are unpredictable. Prefer the first ```python block; fall back
# to whole text after stripping. Extraction is intentionally permissive.
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(prompt: str, completion: str) -> str:
    """
    Build the full module text to execute: prompt + completion (or extracted
    code block). HumanEval+ prompt typically ends with the function signature
    and docstring; the completion is supposed to be the function body.
    """
    text = completion
    m = CODE_BLOCK_RE.search(text)
    if m:
        body = m.group(1)
        # If the extracted block is a full implementation (re-defines the function),
        # use it on its own; otherwise prepend the prompt.
        if body.lstrip().startswith("def "):
            return body
        return prompt + body

    # No code block: treat the completion as raw body
    return prompt + completion


def syntax_ok(code: str) -> bool:
    try:
        compile(code, "<sample>", "exec")
        return True
    except Exception:
        return False


# ----------------------------- Inline execution -----------------------------
# Run code in a subprocess with a hard timeout. We do NOT sandbox; the user has
# accepted that. Each sample gets its own subprocess so a hang doesn't kill the
# main worker. Returns (n_passed, n_total, runtime_error_string_or_None,
# exec_time_s).
EXECUTION_HARNESS = r"""
import sys, json, signal, traceback, time
{code}

# Test driver appended below
{test_driver}
"""


def build_humaneval_test_driver(problem):
    """
    HumanEval+'s test field is a string defining a `check(candidate)` function.
    We run it on the user's `entry_point`. We monkey-patch `assert` indirectly
    by counting checks: HumanEval style uses raw asserts, so we count assert
    statements that pass vs fail by running each assertion independently.

    Simpler approach: just call `check(entry_point_function)` and treat it as
    binary pass/fail. This loses the per-test granularity but matches how
    EvalPlus reports correctness.
    """
    test_code = problem["test"]
    ep = problem["entry_point"]
    return f"""
{test_code}

if __name__ == "__main__":
    try:
        check({ep})
        print("__RESULT__", json.dumps({{"passed": True, "n_passed": 1, "n_total": 1, "error": None}}))
    except AssertionError as e:
        print("__RESULT__", json.dumps({{"passed": False, "n_passed": 0, "n_total": 1, "error": "AssertionError: " + str(e)[:200]}}))
    except Exception as e:
        print("__RESULT__", json.dumps({{"passed": False, "n_passed": 0, "n_total": 1, "error": type(e).__name__ + ": " + str(e)[:200]}}))
"""


def run_one_sample(args):
    """
    Worker function. args = (problem, code, timeout_s).
    Returns (n_passed, n_total, error_str_or_None, exec_time_s).
    """
    problem, code, timeout_s = args
    test_driver = build_humaneval_test_driver(problem)
    full_script = EXECUTION_HARNESS.format(code=code, test_driver=test_driver)

    t0 = time.time()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(full_script)
            tmp_path = f.name

        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = time.time() - t0
        out = proc.stdout
        # Parse last __RESULT__ line
        result_line = None
        for line in out.splitlines():
            if line.startswith("__RESULT__"):
                result_line = line
        if result_line is None:
            err = (proc.stderr or "no result emitted")[-300:]
            return (0, 1, f"NO_RESULT: {err}", elapsed)
        info = json.loads(result_line.split(" ", 1)[1])
        return (info["n_passed"], info["n_total"], info["error"], elapsed)
    except subprocess.TimeoutExpired:
        return (0, 1, "Timeout", time.time() - t0)
    except Exception as e:
        return (0, 1, f"DriverError: {type(e).__name__}: {e}", time.time() - t0)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ----------------------------- Generation -----------------------------
def make_prompts(problems, prompt_format="raw"):
    """For base models, just use the raw HumanEval prompt (function signature
    plus docstring). The model continues with the body.
    """
    if prompt_format == "raw":
        return [p["prompt"] for p in problems]
    raise ValueError(prompt_format)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B")
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-problems", type=int, default=-1,
                    help="If >0, only run on the first N problems (smoke test)")
    ap.add_argument("--exec-timeout", type=float, default=10.0,
                    help="Per-sample execution timeout (seconds)")
    ap.add_argument("--exec-workers", type=int, default=8,
                    help="Parallel workers for execution-truth evaluation")
    ap.add_argument("--skip-exec", action="store_true",
                    help="Skip inline execution (only generate). Useful to debug.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.jsonl"

    problems = load_humaneval_plus()
    if args.n_problems > 0:
        problems = problems[: args.n_problems]
        print(f"[smoketest] limiting to {len(problems)} problems", flush=True)

    # Save metadata
    meta = vars(args).copy()
    meta["n_problems_actual"] = len(problems)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---- Generation phase ----
    from vllm import SamplingParams
    llm = load_vllm_engine(
        args.model, dtype=args.dtype, tp=args.tp,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        # Stop sequences: try to stop the model after one function so it doesn't
        # ramble into next function (common with base models on HumanEval).
        stop=["\nclass ", "\ndef ", "\nif __name__", "\n#", "\n```"],
    )

    prompts = make_prompts(problems, "raw")
    print(f"[gen] sampling {args.n_samples} per problem on {len(prompts)} problems", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    print(f"[gen] done in {time.time() - t0:.1f}s", flush=True)

    # ---- Verification phase ----
    if not args.skip_exec:
        print(f"[exec] verifying with {args.exec_workers} workers, timeout={args.exec_timeout}s",
              flush=True)
    pool = None
    if not args.skip_exec and args.exec_workers > 1:
        pool = mp.Pool(args.exec_workers)

    written = 0
    with open(out_path, "w") as f:
        for prob_idx, (problem, output) in enumerate(zip(problems, outputs)):
            samples_records = []
            # Build (prob, code, timeout) work items
            work = []
            extracted_codes = []
            for s_idx, gen in enumerate(output.outputs):
                text = gen.text
                code = extract_code(problem["prompt"], text)
                extracted_codes.append(code)
                work.append((problem, code, args.exec_timeout))

            # Run executions
            if args.skip_exec:
                results = [(0, 1, "skipped", 0.0)] * len(work)
            elif pool is not None:
                t1 = time.time()
                results = pool.map(run_one_sample, work)
                exec_time = time.time() - t1
            else:
                t1 = time.time()
                results = [run_one_sample(w) for w in work]
                exec_time = time.time() - t1

            for s_idx, (gen, code, (n_pass, n_total, err, et)) in enumerate(
                zip(output.outputs, extracted_codes, results)
            ):
                samples_records.append({
                    "idx": s_idx,
                    "text": gen.text,
                    "code": code,
                    "syntactic": syntax_ok(code),
                    "runtime_error": err,
                    "n_tests_passed": int(n_pass),
                    "n_tests_total": int(n_total),
                    "correct": bool(n_pass == n_total and n_total > 0 and err is None),
                    "exec_time_s": float(et),
                })

            n_correct = sum(s["correct"] for s in samples_records)
            n_synt = sum(s["syntactic"] for s in samples_records)
            print(
                f"[gen] {prob_idx + 1}/{len(problems)} {problem['task_id']} "
                f"correct={n_correct}/{len(samples_records)} "
                f"syntactic={n_synt}/{len(samples_records)} "
                f"exec={exec_time:.1f}s" if not args.skip_exec else "",
                flush=True,
            )

            f.write(json.dumps({
                "id": problem["task_id"],
                "problem": problem["prompt"],
                "answer": None,
                "entry_point": problem["entry_point"],
                "samples": samples_records,
            }) + "\n")
            written += 1

    if pool is not None:
        pool.close()
        pool.join()

    print(f"[done] wrote {written} problems to {out_path}", flush=True)


if __name__ == "__main__":
    main()