"""
Stage 1: generate samples for LiveCodeBench problems with vLLM. No execution.
Saves a generations.jsonl with raw model outputs and extracted code, plus the
test cases needed for stage 2.

Each line:
  {
    "id": "<task_id>",
    "problem": "<full prompt>",
    "answer": null,
    "difficulty": "easy"|"medium"|"hard",
    "platform": "leetcode"|"atcoder"|"codeforces",
    "test_type": "stdin"|"functional",
    "tests": [...],            # tests carried through for stage 2
    "fn_name": str | None,
    "starter_code": str,
    "samples": [
      {"idx": 0, "text": "<raw>", "code": "<extracted>", "syntactic": true,
       "n_tests_passed": null, "n_tests_total": null,
       "correct": null, "runtime_error": null, "exec_time_s": null},
      ...
    ]
  }
"""

import argparse
import base64
import json
import pickle
import sys
import time
import zlib
from pathlib import Path


# ----------------------------- Model loading -----------------------------
def load_vllm_engine(model_name, dtype="float16", tp=1, max_model_len=4096):
    from vllm import LLM
    print(f"[load] loading {model_name} dtype={dtype} tp={tp}", flush=True)
    return LLM(
        model=model_name,
        dtype=dtype,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        seed=42,
    )


# ----------------------------- LiveCodeBench loading -----------------------------
def _maybe_decode_private_tests(s):
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
    """Load LCB from bzantium mirror (no broken loading script)."""
    from datasets import load_dataset
    print(f"[data] loading bzantium/livecodebench {version_tag}", flush=True)
    ds = load_dataset("bzantium/livecodebench", version_tag, split="test")

    out = []
    for row in ds:
        diff = row.get("difficulty", "").lower()
        plat = row.get("platform", "").lower()
        if difficulty is not None and diff not in difficulty:
            continue
        if platforms is not None and plat not in platforms:
            continue

        public_tc = json.loads(row.get("public_test_cases") or "[]")
        private_tc = _maybe_decode_private_tests(row.get("private_test_cases") or "")
        tests = list(public_tc) + list(private_tc)
        if not tests:
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
            "functional" if any(t.get("testtype") == "functional" for t in tests)
            else "stdin"
        )

        if test_type == "functional" or starter.strip():
            ONE_SHOT = (
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
            prompt = (
                ONE_SHOT
                + "### Problem\n"
                + f"Problem: {question}\n"
                + (f"Starter code:\n```python\n{starter}\n```\n" if starter.strip() else "")
                + "Solution:\n```python\n"
            )
        else:
            ONE_SHOT = (
                "### Example\n"
                "Problem: Read an integer t (number of test cases). For each test case, "
                "read two integers a and b and print their sum.\n"
                "Solution:\n"
                "```python\n"
                "t = int(input())\n"
                "for _ in range(t):\n"
                "    a, b = map(int, input().split())\n"
                "    print(a + b)\n"
                "```\n\n"
            )
            prompt = (
                ONE_SHOT
                + "### Problem\n"
                + f"Problem: {question}\n"
                + "Solution:\n```python\n"
            )

        out.append({
            "task_id": row.get("question_id") or row.get("question_title", f"lcb_{len(out)}"),
            "prompt": prompt,
            "tests": tests,
            "starter_code": starter,
            "fn_name": fn_name,
            "difficulty": diff,
            "platform": plat,
            "test_type": test_type,
        })
        if max_problems > 0 and len(out) >= max_problems:
            break

    print(f"[data] kept {len(out)} problems "
          f"(difficulty={difficulty}, platforms={platforms})", flush=True)
    return out


# ----------------------------- Code extraction -----------------------------
def extract_code(prompt: str, completion: str, starter: str) -> str:
    """Pull out Python code from completion. Prompt ends with "```python\n" so
    the completion starts inside a code block. Stop sequences should cut at the
    closing ```; if any fence remains in the middle, cut at it.
    """
    text = completion
    if "```" in text:
        text = text.split("```", 1)[0]
    return text.rstrip()


def syntax_ok(code: str) -> bool:
    try:
        compile(code, "<sample>", "exec")
        return True
    except Exception:
        return False


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B")
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lcb-version", default="release_v2")
    ap.add_argument("--difficulty", default="medium,hard")
    ap.add_argument("--platforms", default="leetcode,atcoder,codeforces")
    ap.add_argument("--n-problems", type=int, default=-1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.jsonl"

    difficulty = set(d.strip() for d in args.difficulty.split(","))
    platforms = set(p.strip() for p in args.platforms.split(","))

    problems = load_livecodebench(
        version_tag=args.lcb_version,
        difficulty=difficulty,
        platforms=platforms,
        max_problems=args.n_problems,
    )
    if not problems:
        sys.exit("[fatal] no problems after filtering")

    meta = vars(args).copy()
    meta["n_problems_actual"] = len(problems)
    meta["stage"] = "1_generation"
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

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
        stop=["```", "\n```"],
    )

    prompts = [p["prompt"] for p in problems]
    print(f"[gen] sampling {args.n_samples} per problem on {len(prompts)} problems",
          flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    print(f"[gen] done in {time.time() - t0:.1f}s", flush=True)

    written = 0
    with open(out_path, "w") as f:
        for prob_idx, (problem, output) in enumerate(zip(problems, outputs)):
            samples_records = []
            for s_idx, gen in enumerate(output.outputs):
                code = extract_code(problem["prompt"], gen.text,
                                    problem["starter_code"])
                samples_records.append({
                    "idx": s_idx,
                    "text": gen.text,
                    "code": code,
                    "syntactic": syntax_ok(code),
                    # Verification fields filled in by stage 2
                    "runtime_error": None,
                    "n_tests_passed": None,
                    "n_tests_total": None,
                    "correct": None,
                    "exec_time_s": None,
                })

            n_synt = sum(s["syntactic"] for s in samples_records)
            print(
                f"[gen] {prob_idx + 1}/{len(problems)} "
                f"{problem['task_id']} ({problem['difficulty']}) "
                f"syntactic={n_synt}/{len(samples_records)}",
                flush=True,
            )

            f.write(json.dumps({
                "id": problem["task_id"],
                "problem": problem["prompt"],
                "answer": None,
                "difficulty": problem["difficulty"],
                "platform": problem["platform"],
                "test_type": problem["test_type"],
                "tests": problem["tests"],
                "fn_name": problem["fn_name"],
                "starter_code": problem["starter_code"],
                "samples": samples_records,
            }) + "\n")
            written += 1

    print(f"[done] wrote {written} problems to {out_path}", flush=True)
    print(f"[next] run: python verify_lcb.py --input {out_path} "
          f"--output {out_dir / 'generations_verified.jsonl'} --workers 64",
          flush=True)


if __name__ == "__main__":
    main()