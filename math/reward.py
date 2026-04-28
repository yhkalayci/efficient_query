#!/usr/bin/env python3
"""
Score each generation in a generations.jsonl with a Qwen2.5-Math-PRM-* reward
model using vLLM's native reward-model support.

vLLM has first-class support for Qwen2 reward models via task="reward" and a
STEP-type pooler that returns per-token hidden state at each <extra_0> separator.

This script works for:
  - Qwen/Qwen2.5-Math-PRM-7B  (bf16, any Ampere+ GPU)
  - Qwen/Qwen2.5-Math-PRM-72B (fp8 on L40S/Ada/Hopper for native W8A8, or bf16
    on 2+ A100-80GB, or weight-only Marlin fp8 on A40/A100-40GB)

Usage:
  python score_with_prm_vllm.py \
      --input generations.jsonl \
      --output rewards.jsonl \
      --model Qwen/Qwen2.5-Math-PRM-72B \
      --quantization fp8 \
      --tensor-parallel-size 2

Output schema matches score_with_prm.py exactly so compute_auc.py works on both.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import torch


SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


# -------------------------- Step splitting --------------------------
def split_into_steps(text: str) -> List[str]:
    """Split assistant response into reasoning steps on blank lines, with fallbacks."""
    if not text:
        return []
    steps = [s.strip() for s in text.split("\n\n")]
    steps = [s for s in steps if s]
    if len(steps) >= 2:
        return steps
    steps = [s.strip() for s in text.split("\n")]
    steps = [s for s in steps if s]
    if len(steps) >= 2:
        return steps
    t = text.strip()
    return [t] if t else []


# -------------------------- Reward aggregation --------------------------
def aggregate_rewards(step_scores: List[float]) -> dict:
    if not step_scores:
        return {"r_min": None, "r_mean": None, "r_last": None, "r_prod": None}
    r_min = min(step_scores)
    r_mean = sum(step_scores) / len(step_scores)
    r_last = step_scores[-1]
    r_prod = 1.0
    for s in step_scores:
        r_prod *= s
    return {
        "r_min": round(r_min, 6),
        "r_mean": round(r_mean, 6),
        "r_last": round(r_last, 6),
        "r_prod": round(r_prod, 6),
    }


# -------------------------- Prompt building --------------------------
def build_conversation(tokenizer, problem: str, steps: List[str]) -> str:
    """Build the full conversation string with <extra_0> between steps."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {
            "role": "assistant",
            "content": "<extra_0>".join(steps) + "<extra_0>",
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


# -------------------------- Main --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to generations.jsonl")
    parser.add_argument("--output", required=True, help="Path to rewards.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--tensor-parallel-size", type=int, default=0,
                        help="0 = use all visible GPUs")
    parser.add_argument("--quantization", default=None,
                        help="None for bf16/fp16. 'fp8' for W8A8 on L40S/H100 or "
                             "weight-only fp8-Marlin on Ampere. 'awq' / 'gptq' also accepted.")
    parser.add_argument("--dtype", default="auto",
                        choices=["bfloat16", "float16", "auto"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--limit-problems", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        sys.exit(f"[fatal] input not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Hardware sanity check
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        sys.exit("[fatal] no CUDA GPUs visible")
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size > 0 else num_gpus
    print(f"[config] {num_gpus} GPU(s) visible, tensor_parallel_size={tp_size}")
    for i in range(num_gpus):
        p = torch.cuda.get_device_properties(i)
        cc = f"{p.major}.{p.minor}"
        print(f"         GPU {i}: {p.name} ({p.total_memory / 2**30:.1f} GB, sm_{p.major}{p.minor})")
        if args.quantization == "fp8" and (p.major, p.minor) < (8, 9):
            print(f"         [warn] fp8 requested but sm_{p.major}{p.minor} < sm_89; "
                  f"will fall back to fp8-Marlin weight-only (still memory-efficient, compute in bf16)")

    # Resume support
    already_done = set()
    mode = "w"
    if args.skip_existing and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["id"])
                except Exception:
                    pass
        mode = "a"
        print(f"[resume] {len(already_done)} problems already scored; appending")

    # Import vllm late so the script can print help / early errors without waiting
    print(f"[model] loading {args.model}...", flush=True)
    t0 = time.time()
    from vllm import LLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    step_sep_id = tokenizer.encode("<extra_0>")[0]

    # The PRM uses a 2-class (positive/negative) head per step. We tell vLLM's
    # STEP pooler which token marks a step boundary and which class ids to return.
    # returned_token_ids comes from the PRM's classification head: [neg_id, pos_id].
    # For Qwen2.5-Math-PRM these are indices [0, 1] in the RM head's output dim;
    # we pass them through override_pooler_config so vLLM returns both logits per
    # step and we can softmax ourselves to get the positive-class probability.
    pooler_config = {
        "pooling_type": "STEP",
        "step_tag_id": step_sep_id,
        "returned_token_ids": [0, 1],
    }

    llm_kwargs = dict(
        model=args.model,
        task="reward",
        tensor_parallel_size=tp_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        override_pooler_config=pooler_config,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization

    llm = LLM(**llm_kwargs)
    print(f"[model] loaded in {time.time() - t0:.1f}s", flush=True)

    # Read problems
    problems = []
    with open(in_path) as f:
        for line in f:
            problems.append(json.loads(line))
    if args.limit_problems:
        problems = problems[: args.limit_problems]
    total_samples = sum(len(p["samples"]) for p in problems)
    print(f"[data] {len(problems)} problems, {total_samples:,} total samples", flush=True)

    fout = open(out_path, mode, buffering=1)
    t_start = time.time()
    n_scored_problems = 0

    for pi, prob in enumerate(problems):
        if prob["id"] in already_done:
            continue

        pt0 = time.time()
        question = prob["problem"]

        # Build prompts for every sample that has text. Track which samples we skip.
        prompts = []
        sample_steps = []
        sample_indices = []
        null_results = []  # [(idx, placeholder_record), ...]
        for i, s in enumerate(prob["samples"]):
            text = s.get("text")
            if not text:
                null_results.append((i, {"idx": i, "n_steps": 0, "step_scores": [],
                                         "r_min": None, "r_mean": None,
                                         "r_last": None, "r_prod": None}))
                continue
            steps = split_into_steps(text)
            if not steps:
                null_results.append((i, {"idx": i, "n_steps": 0, "step_scores": [],
                                         "r_min": None, "r_mean": None,
                                         "r_last": None, "r_prod": None}))
                continue
            prompts.append(build_conversation(tokenizer, question, steps))
            sample_steps.append(steps)
            sample_indices.append(i)

        # Score. vLLM handles continuous batching internally; we just submit all
        # prompts for this problem in one call.
        rewards_out = []
        if prompts:
            outputs = llm.encode(prompts)
            assert len(outputs) == len(prompts), "vllm output count mismatch"
            for out, idx, steps in zip(outputs, sample_indices, sample_steps):
                # With STEP pooler, out.outputs.data has shape (n_steps, 2).
                # Column 0 = neg-class logit, Column 1 = pos-class logit per step.
                data = out.outputs.data
                if data is None or len(data) == 0:
                    rewards_out.append({"idx": idx, "n_steps": 0, "step_scores": [],
                                        "r_min": None, "r_mean": None,
                                        "r_last": None, "r_prod": None})
                    continue
                # Softmax the 2 logits per step, take positive-class prob
                import torch.nn.functional as F
                logits = torch.tensor(data) if not isinstance(data, torch.Tensor) else data
                probs = F.softmax(logits, dim=-1)
                step_scores = probs[:, 1].cpu().tolist()
                agg = aggregate_rewards(step_scores)
                rewards_out.append({
                    "idx": idx,
                    "n_steps": len(steps),
                    "step_scores": [round(s, 6) for s in step_scores],
                    **agg,
                })

        # Merge in null records and sort by idx
        rewards_out.extend([r for _, r in null_results])
        rewards_out.sort(key=lambda r: r["idx"])

        record = {
            "id": prob["id"],
            "n_samples": len(prob["samples"]),
            "rewards": rewards_out,
        }
        fout.write(json.dumps(record) + "\n")
        n_scored_problems += 1

        # Log summary
        valid = [r for r in rewards_out if r["r_min"] is not None]
        if valid:
            mean_rmin = sum(r["r_min"] for r in valid) / len(valid)
            mean_rlast = sum(r["r_last"] for r in valid) / len(valid)
            summary = f"<r_min>={mean_rmin:.3f}  <r_last>={mean_rlast:.3f}  n_valid={len(valid)}/{len(prob['samples'])}"
        else:
            summary = "no valid samples"
        elapsed_p = time.time() - pt0
        elapsed_total = time.time() - t_start
        remaining_problems = len(problems) - pi - 1
        eta_min = (elapsed_total / max(n_scored_problems, 1)) * remaining_problems / 60
        print(f"  [{pi + 1:>3d}/{len(problems)}] {prob['id']:<30s}  {summary}  "
              f"({elapsed_p:.0f}s, ETA {eta_min:.0f}min)", flush=True)

    fout.close()
    print(f"\n[done] scored {n_scored_problems} problems in "
          f"{(time.time() - t_start) / 60:.1f} min")
    print(f"[done] output: {out_path}")


if __name__ == "__main__":
    main()