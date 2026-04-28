"""
Score code samples with CodeScaler (or any compatible reward model) via vLLM,
which uses continuous batching across all (prompt, code) pairs.

Significantly faster than the transformers-loop version (reward.py) because:
  - PagedAttention + continuous batching: GPU never idle waiting for the
    longest sequence in a static batch
  - Fused kernels for prefill, no manual padding overhead

Output schema is identical to reward.py: rewards.jsonl, one line per problem,
each with {"id", "n_samples", "rewards": [{"idx", "r_score"}, ...]}.

Usage:
  python reward_vllm.py \\
    --generations runs/lcb_3b_256/generations.jsonl \\
    --model LARK-Lab/CodeScaler-8B \\
    --out    runs/lcb_3b_256/rewards.jsonl \\
    --tp 1 --dtype bfloat16 --max-model-len 8192
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_generations(path):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--model", default="LARK-Lab/CodeScaler-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="Max sequence length (prompt + code). LCB problems can"
                         " be long; bump this if you see truncation warnings.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--n-problems", type=int, default=-1,
                    help="Limit to first N problems (smoke)")
    ap.add_argument("--task", default="reward",
                    choices=["reward", "classify"],
                    help="vLLM pooling task. Use 'reward' for regression-head"
                         " (num_labels=1) checkpoints; 'classify' for "
                         " sequence-classification (num_labels=2) checkpoints.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model} task={args.task} dtype={args.dtype} tp={args.tp}",
          flush=True)
    from vllm import LLM
    from transformers import AutoTokenizer

    llm = LLM(
        model=args.model,
        task=args.task,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    gens = load_generations(args.generations)
    if args.n_problems > 0:
        gens = gens[: args.n_problems]
    print(f"[data] {len(gens)} problems", flush=True)

    def build_input(prompt_text: str, code_text: str) -> str:
        """Apply chat template: user=problem, assistant=candidate code.
        CodeScaler is trained on Qwen3 chat format."""
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": code_text},
        ]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

    # Flatten all (prob_idx, sample_idx, text) into one big batch. vLLM handles
    # scheduling across them via continuous batching.
    flat_texts = []
    flat_keys = []  # (prob_idx, sample_idx) so we can route results back
    for prob_idx, prob in enumerate(gens):
        for s in prob["samples"]:
            flat_texts.append(build_input(prob["problem"], s["code"]))
            flat_keys.append((prob_idx, s["idx"]))

    print(f"[score] running vLLM on {len(flat_texts)} (problem, sample) pairs",
          flush=True)
    t0 = time.time()
    if args.task == "reward":
        # encode() returns hidden states; for reward heads that's the scalar
        # score on the last token after the reward head projection.
        outputs = llm.encode(flat_texts)
        flat_scores = []
        for o in outputs:
            data = o.outputs.data  # tensor; shape varies
            arr = data.cpu().float().numpy() if hasattr(data, "cpu") else np.asarray(data)
            # Reward head produces (seq_len, 1) or (1,) or (seq_len,)
            if arr.ndim == 0:
                flat_scores.append(float(arr))
            elif arr.ndim == 1:
                flat_scores.append(float(arr[-1]))
            else:
                # Take last token, last dim
                flat_scores.append(float(arr[-1].squeeze()))
    else:  # classify
        outputs = llm.classify(flat_texts)
        flat_scores = []
        for o in outputs:
            probs = np.asarray(o.outputs.probs)
            # Convention: positive class is the last logit
            flat_scores.append(float(probs[-1]))

    print(f"[score] vLLM finished in {time.time() - t0:.1f}s "
          f"({len(flat_texts) / max(time.time() - t0, 1e-6):.1f} samples/s)",
          flush=True)

    # Group scores back by problem
    scores_by_prob = {}
    for (prob_idx, sample_idx), score in zip(flat_keys, flat_scores):
        scores_by_prob.setdefault(prob_idx, {})[sample_idx] = score

    with open(out_path, "w") as fout:
        for prob_idx, prob in enumerate(gens):
            sd = scores_by_prob.get(prob_idx, {})
            rewards_records = [
                {"idx": s["idx"], "r_score": sd.get(s["idx"])}
                for s in prob["samples"]
            ]
            fout.write(json.dumps({
                "id": prob["id"],
                "n_samples": len(prob["samples"]),
                "rewards": rewards_records,
            }) + "\n")

    print(f"[done] wrote rewards to {out_path}", flush=True)


if __name__ == "__main__":
    main()