"""
Score code samples with CodeScaler-8B (or 4B). For each (prompt, code) pair we
get a single scalar reward.

Output rewards.jsonl, one line per problem:
  {
    "id": "HumanEval/0",
    "n_samples": 256,
    "rewards": [
      {"idx": 0, "r_score": 0.732},
      ...
    ]
  }

This is simpler than the math PRM scoring because CodeScaler is a single-step
sequence classifier (Skywork-Reward-V2 architecture), not a step-level PRM.

Usage:
  python score_with_codescaler.py \\
    --generations out/generations.jsonl \\
    --model LARK-Lab/CodeScaler-8B \\
    --out out/rewards.jsonl \\
    --batch-size 8

Tested on 2x A40 with bf16. For 1x GPU use --tp 1 and either dtype bf16 or
float16.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn.functional import softmax


def load_generations(path):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--model", default="LARK-Lab/CodeScaler-8B",
                    help="HF model id. Use LARK-Lab/CodeScaler-4B for less memory.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=4096,
                    help="Max sequence length (prompt + code)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-problems", type=int, default=-1,
                    help="Limit to first N problems (smoke)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model} dtype={args.dtype}", flush=True)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        device_map=args.device,
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gens = load_generations(args.generations)
    if args.n_problems > 0:
        gens = gens[: args.n_problems]
    print(f"[data] {len(gens)} problems", flush=True)

    # CodeScaler expects chat-style messages. Per their HF README, they use the
    # underlying Qwen3 chat template with role=user containing the programming
    # problem and role=assistant containing the candidate solution.
    # We follow that pattern.
    def build_input(prompt_text: str, code_text: str):
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": code_text},
        ]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

    t_total = time.time()
    with open(out_path, "w") as fout:
        for prob_idx, prob in enumerate(gens):
            prompt = prob["problem"]
            samples = prob["samples"]
            n = len(samples)

            # Tokenize all (prompt, code) pairs for this problem
            texts = [build_input(prompt, s["code"]) for s in samples]
            scores = [None] * n

            for start in range(0, n, args.batch_size):
                batch = texts[start:start + args.batch_size]
                enc = tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to(args.device)
                with torch.no_grad():
                    logits = model(**enc).logits  # shape (B, num_labels)
                # CodeScaler-8B is a scalar reward model: num_labels=1, logits
                # are the reward score directly. If the checkpoint is shaped
                # (B, 2), use softmax-positive instead.
                if logits.shape[-1] == 1:
                    s = logits.squeeze(-1).float().cpu().tolist()
                else:
                    probs = softmax(logits.float(), dim=-1)
                    s = probs[:, -1].cpu().tolist()
                for i, score in enumerate(s):
                    scores[start + i] = float(score)

            rewards_records = [
                {"idx": s["idx"], "r_score": scores[i]}
                for i, s in enumerate(samples)
            ]
            fout.write(json.dumps({
                "id": prob["id"],
                "n_samples": n,
                "rewards": rewards_records,
            }) + "\n")

            if (prob_idx + 1) % 5 == 0 or prob_idx == len(gens) - 1:
                rate = (prob_idx + 1) / max(time.time() - t_total, 1e-6)
                print(
                    f"[score] {prob_idx + 1}/{len(gens)} "
                    f"({rate:.2f} prob/s)",
                    flush=True,
                )

    print(f"[done] wrote rewards to {out_path}, total {time.time() - t_total:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()