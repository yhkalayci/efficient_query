"""
Stage 2: verify generations from stage 1 by executing each sample. CPU-only.
Reads a generations.jsonl produced by gen_lcb.py and writes a verified version
with n_tests_passed, n_tests_total, correct, runtime_error, exec_time_s
populated on each sample.

Two test types are handled:
  - "stdin":      run the script with input piped via sys.stdin, compare stdout
  - "functional": exec code, find the entry function (by fn_name or class
                  Solution heuristic), call with each test's args, compare
                  return value

Parallelism: a multiprocessing pool of N workers, each running one subprocess
per (problem, sample). Default N=64.

Resume: by default skips problems already present in the output file. Use
--overwrite to rerun everything.
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List


# ----------------------------- Subprocess harnesses -----------------------------
EXEC_HARNESS_STDIN = r"""
import sys, json, io
USER_CODE = {code_repr}
TESTS = {tests_repr}

n_passed = 0
n_total = len(TESTS)
errors = []

orig_stdin = sys.stdin
orig_stdout = sys.stdout

for i, t in enumerate(TESTS):
    test_in = t["input"]
    test_out = t["output"].rstrip()

    g = {{"__name__": "__main__"}}
    fake_stdin = io.StringIO(test_in)
    fake_stdout = io.StringIO()
    sys.stdin = fake_stdin
    sys.stdout = fake_stdout
    try:
        exec(USER_CODE, g)
        sys.stdout = orig_stdout
        sys.stdin = orig_stdin
        actual = fake_stdout.getvalue().rstrip()
        if actual == test_out or actual.split() == test_out.split():
            n_passed += 1
        else:
            if len(errors) < 3:
                errors.append(f"test {{i}}: got {{actual[:80]!r}} expected {{test_out[:80]!r}}")
    except SystemExit:
        sys.stdout = orig_stdout
        sys.stdin = orig_stdin
        actual = fake_stdout.getvalue().rstrip()
        if actual == test_out or actual.split() == test_out.split():
            n_passed += 1
        else:
            if len(errors) < 3:
                errors.append(f"test {{i}}: sys.exit then wrong output")
    except Exception as e:
        sys.stdout = orig_stdout
        sys.stdin = orig_stdin
        if len(errors) < 3:
            errors.append(f"test {{i}}: {{type(e).__name__}}: {{str(e)[:120]}}")
    finally:
        sys.stdout = orig_stdout
        sys.stdin = orig_stdin

err_str = "; ".join(errors) if errors else None
print("__RESULT__", json.dumps({{
    "n_passed": n_passed, "n_total": n_total, "error": err_str,
}}))
"""


EXEC_HARNESS_FUNCTIONAL = r"""
import sys, json
USER_CODE = {code_repr}
TESTS = {tests_repr}
FN_NAME = {fn_name_repr}

n_passed = 0
n_total = len(TESTS)
errors = []

g = {{"__name__": "__main__"}}
try:
    exec(USER_CODE, g)
except Exception as e:
    print("__RESULT__", json.dumps({{
        "n_passed": 0, "n_total": n_total,
        "error": "exec_failed: " + type(e).__name__ + ": " + str(e)[:160],
    }}))
    sys.exit(0)

# Find the function: declared name first, then Solution class method, then last def
fn = None
if FN_NAME and FN_NAME in g:
    fn = g[FN_NAME]
else:
    if "Solution" in g:
        try:
            sol = g["Solution"]()
            for name in dir(sol):
                if not name.startswith("_") and callable(getattr(sol, name)):
                    fn = getattr(sol, name)
                    break
        except Exception:
            pass
    if fn is None:
        for name, val in reversed(list(g.items())):
            if callable(val) and not name.startswith("_"):
                fn = val
                break

if fn is None:
    print("__RESULT__", json.dumps({{
        "n_passed": 0, "n_total": n_total, "error": "no_function_found",
    }}))
    sys.exit(0)

for i, t in enumerate(TESTS):
    raw_in = t["input"]
    expected = t["output"]
    try:
        if isinstance(raw_in, list):
            args = raw_in
        else:
            try:
                parsed = json.loads(raw_in)
                args = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                args = []
                for line in raw_in.strip().split("\n"):
                    try:
                        args.append(json.loads(line))
                    except Exception:
                        args.append(line)
        result = fn(*args)
        try:
            expected_val = json.loads(expected) if isinstance(expected, str) else expected
        except Exception:
            expected_val = expected
        if result == expected_val:
            n_passed += 1
        elif str(result).strip() == str(expected_val).strip():
            n_passed += 1
        else:
            if len(errors) < 3:
                errors.append(f"test {{i}}: got {{repr(result)[:60]}} expected {{repr(expected_val)[:60]}}")
    except Exception as e:
        if len(errors) < 3:
            errors.append(f"test {{i}}: {{type(e).__name__}}: {{str(e)[:120]}}")

err_str = "; ".join(errors) if errors else None
print("__RESULT__", json.dumps({{
    "n_passed": n_passed, "n_total": n_total, "error": err_str,
}}))
"""


def run_one_sample(args):
    """Worker function. args is a tuple (job_key, payload).
      job_key: (problem_idx, sample_idx) used to route results back
      payload: (test_type, code, tests, fn_name, timeout_s)
    Returns (job_key, n_passed, n_total, error, exec_time_s).
    """
    job_key, (test_type, code, tests, fn_name, timeout_s) = args
    if test_type == "stdin":
        script = EXEC_HARNESS_STDIN.format(
            code_repr=repr(code), tests_repr=repr(tests),
        )
    else:
        script = EXEC_HARNESS_FUNCTIONAL.format(
            code_repr=repr(code), tests_repr=repr(tests),
            fn_name_repr=repr(fn_name),
        )

    t0 = time.time()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script)
            tmp_path = f.name

        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout_s,
        )
        elapsed = time.time() - t0
        result_line = None
        for line in proc.stdout.splitlines():
            if line.startswith("__RESULT__"):
                result_line = line
        if result_line is None:
            err = (proc.stderr or "no result emitted")[-300:]
            return (job_key, 0, len(tests), f"NO_RESULT: {err}", elapsed)
        info = json.loads(result_line.split(" ", 1)[1])
        return (job_key, info["n_passed"], info["n_total"],
                info["error"], elapsed)
    except subprocess.TimeoutExpired:
        return (job_key, 0, len(tests), "Timeout", time.time() - t0)
    except Exception as e:
        return (job_key, 0, len(tests),
                f"DriverError: {type(e).__name__}: {e}", time.time() - t0)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="generations.jsonl from stage 1")
    ap.add_argument("--output", required=True,
                    help="path to write verified generations.jsonl")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--overwrite", action="store_true",
                    help="rerun even if a verified output exists; default skips problems")
    ap.add_argument("--limit", type=int, default=-1,
                    help="verify only the first N problems (smoke test)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all problems
    problems = []
    with open(in_path) as f:
        for line in f:
            problems.append(json.loads(line))
    if args.limit > 0:
        problems = problems[: args.limit]

    # Resume support: collect already-verified problem IDs
    done_ids = set()
    if out_path.exists() and not args.overwrite:
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        if done_ids:
            print(f"[resume] {len(done_ids)} problems already in {out_path}, "
                  f"skipping them. Use --overwrite to rerun.", flush=True)

    todo = [p for p in problems if p["id"] not in done_ids]
    print(f"[verify] {len(todo)} problems to verify, "
          f"{sum(len(p['samples']) for p in todo)} samples total, "
          f"workers={args.workers}, timeout={args.timeout}s",
          flush=True)

    # Open output in append mode (so resume keeps prior lines)
    out_mode = "w" if (args.overwrite or not out_path.exists()) else "a"
    fout = open(out_path, out_mode)

    pool = mp.Pool(args.workers)

    t_start = time.time()
    total_samples_done = 0

    try:
        for prob_idx, problem in enumerate(todo):
            # Build job list for this problem
            jobs = []
            for samp in problem["samples"]:
                key = (prob_idx, samp["idx"])
                payload = (
                    problem["test_type"],
                    samp["code"],
                    problem["tests"],
                    problem.get("fn_name"),
                    args.timeout,
                )
                jobs.append((key, payload))

            t1 = time.time()
            results = pool.map(run_one_sample, jobs)
            exec_time = time.time() - t1

            # Map results back to samples by sample idx. Each result is a tuple
            # (job_key, n_passed, n_total, err, exec_time_s) where
            # job_key = (problem_idx, sample_idx).
            results_by_sidx = {}
            for r in results:
                job_key, n_pass, n_total, err, et = r
                _, sidx = job_key
                results_by_sidx[sidx] = (n_pass, n_total, err, et)

            for samp in problem["samples"]:
                n_pass, n_total, err, et = results_by_sidx[samp["idx"]]
                samp["n_tests_passed"] = int(n_pass)
                samp["n_tests_total"] = int(n_total)
                samp["correct"] = bool(
                    n_pass == n_total and n_total > 0 and err is None
                )
                samp["runtime_error"] = err
                samp["exec_time_s"] = float(et)

            n_correct = sum(s["correct"] for s in problem["samples"])
            n_synt = sum(s["syntactic"] for s in problem["samples"])
            total_samples_done += len(problem["samples"])
            elapsed_total = time.time() - t_start
            rate = total_samples_done / max(elapsed_total, 1e-6)
            print(
                f"[verify] {prob_idx + 1}/{len(todo)} "
                f"{problem['id']} ({problem['difficulty']}) "
                f"correct={n_correct}/{len(problem['samples'])} "
                f"syntactic={n_synt}/{len(problem['samples'])} "
                f"exec={exec_time:.1f}s "
                f"({rate:.1f} samp/s overall)",
                flush=True,
            )

            fout.write(json.dumps(problem) + "\n")
            fout.flush()

    finally:
        pool.close()
        pool.join()
        fout.close()

    print(f"[done] wrote verified output to {out_path} "
          f"({time.time() - t_start:.1f}s, {total_samples_done} samples)",
          flush=True)


if __name__ == "__main__":
    main()