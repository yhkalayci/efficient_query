"""
Score code samples with CodeScaler-8B (or 4B). For each (prompt, code) pair we
get a single scalar reward.

Output rewards.jsonl, one line per problem:
  {
    "id": "...",
    "n_samples": 256,
    "rewards": [
      {"idx": 0, "r_score": 0.732},
      ...
    ]
  }

Usage:
  python coding/reward.py \
    --generations results/coding/verified.jsonl \
    --out         results/coding/rewards.jsonl
"""

import argparse
import concurrent.futures
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


def _forward(model, enc):
    with torch.no_grad():
        return model(**enc).logits


def score_batch(model, enc, timeout_s):
    """Run forward pass with a thread-level timeout. Returns logits or None on timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_forward, model, enc)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--model", default="LARK-Lab/CodeScaler-8B",
                    help="HF model id. Use LARK-Lab/CodeScaler-4B for less memory.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Samples per forward pass. Increase if GPU memory allows.")
    ap.add_argument("--max-length", type=int, default=4096,
                    help="Max sequence length (prompt + code); longer sequences are truncated.")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--batch-timeout", type=float, default=120.0,
                    help="Seconds before a single batch forward pass is considered hung.")
    ap.add_argument("--n-problems", type=int, default=-1,
                    help="Limit to first N problems (smoke test)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    print(f"[config] {num_gpus} GPU(s) visible, device_map=auto, dtype={args.dtype}", flush=True)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        device_map="auto",  # spreads across all visible GPUs
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gens = load_generations(args.generations)
    if args.n_problems > 0:
        gens = gens[: args.n_problems]
    print(f"[data] {len(gens)} problems", flush=True)

    def build_input(prompt_text: str, code_text: str):
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": code_text},
        ]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

    # device for moving tensors — with device_map=auto the model's first layer
    # determines where inputs should land
    input_device = next(model.parameters()).device

    t_total = time.time()
    n_timeouts = 0

    with open(out_path, "w") as fout:
        for prob_idx, prob in enumerate(gens):
            prompt = prob["problem"]
            samples = prob["samples"]
            n = len(samples)

            texts = [build_input(prompt, s["code"] or "") for s in samples]
            scores = [None] * n

            for start in range(0, n, args.batch_size):
                batch_texts = texts[start:start + args.batch_size]
                enc = tok(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to(input_device)

                logits = score_batch(model, enc, args.batch_timeout)

                if logits is None:
                    n_timeouts += 1
                    print(
                        f"[warn] timeout on prob {prob_idx} batch {start}–"
                        f"{start + len(batch_texts) - 1}, setting scores to null",
                        flush=True,
                    )
                    # scores remain None for this slice
                    continue

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

            if (prob_idx + 1) % 10 == 0 or prob_idx == len(gens) - 1:
                elapsed = time.time() - t_total
                rate = (prob_idx + 1) / max(elapsed, 1e-6)
                eta = (len(gens) - prob_idx - 1) / max(rate, 1e-6)
                print(
                    f"[score] {prob_idx + 1}/{len(gens)} "
                    f"({rate:.2f} prob/s, ETA {eta / 60:.1f}min, "
                    f"timeouts={n_timeouts})",
                    flush=True,
                )

    print(f"[done] wrote {out_path} in {time.time() - t_total:.1f}s "
          f"(timeouts={n_timeouts})", flush=True)


if __name__ == "__main__":
    main()
