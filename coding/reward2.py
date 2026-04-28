"""
Score code samples with CodeScaler-8B (or 4B). For each (prompt, code) pair we
get a single scalar reward.

Output rewards.jsonl, one line per problem:
  {
    "id": "<task_id>",
    "n_samples": 256,
    "rewards": [
      {"idx": 0, "r_score": 0.732},
      ...
    ]
  }

Usage:
  python reward.py \\
    --generations out/generations.jsonl \\
    --model LARK-Lab/CodeScaler-8B \\
    --out out/rewards.jsonl \\
    --batch-size 8

Progress is shown as a single tqdm bar over all (problem, sample) pairs.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn.functional import softmax
from tqdm import tqdm


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
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-length", type=int, default=4096,
                    help="Max sequence length (prompt + code)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-problems", type=int, default=-1,
                    help="Limit to first N problems (smoke)")
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "flash_attention_2", "eager"],
                    help="attn_implementation. flash_attention_2 is fastest if "
                         "installed.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model} dtype={args.dtype} attn={args.attn}", flush=True)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=args.device,
    )
    try:
        load_kwargs["attn_implementation"] = args.attn
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, **load_kwargs,
        )
    except (TypeError, ImportError, ValueError) as e:
        print(f"[warn] attn={args.attn} unavailable ({e}); falling back to default",
              flush=True)
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, **load_kwargs,
        )
    model.eval()

    gens = load_generations(args.generations)
    if args.n_problems > 0:
        gens = gens[: args.n_problems]
    total_samples = sum(len(p["samples"]) for p in gens)
    print(f"[data] {len(gens)} problems, {total_samples} samples total", flush=True)

    # CodeScaler expects chat-style messages: role=user (problem),
    # role=assistant (candidate solution).
    def build_input(prompt_text: str, code_text: str) -> str:
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": code_text},
        ]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

    t_total = time.time()
    pbar = tqdm(
        total=total_samples,
        desc="scoring",
        unit="samp",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    n_truncated = 0

    with open(out_path, "w") as fout:
        for prob_idx, prob in enumerate(gens):
            prompt = prob["problem"]
            samples = prob["samples"]
            n = len(samples)
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
                # Track if any sequence got truncated (for diagnostics)
                if (enc["input_ids"].shape[1] >= args.max_length):
                    n_truncated += enc["input_ids"].shape[0]
                with torch.no_grad():
                    logits = model(**enc).logits  # (B, num_labels)
                if logits.shape[-1] == 1:
                    s = logits.squeeze(-1).float().cpu().tolist()
                else:
                    probs = softmax(logits.float(), dim=-1)
                    s = probs[:, -1].cpu().tolist()
                for i, score in enumerate(s):
                    scores[start + i] = float(score)
                pbar.update(len(batch))
                pbar.set_postfix(
                    prob=f"{prob_idx + 1}/{len(gens)}",
                    id=prob["id"][:20],
                )

            rewards_records = [
                {"idx": samples[i]["idx"], "r_score": scores[i]}
                for i in range(n)
            ]
            fout.write(json.dumps({
                "id": prob["id"],
                "n_samples": n,
                "rewards": rewards_records,
            }) + "\n")
            fout.flush()

    pbar.close()
    elapsed = time.time() - t_total
    print(f"[done] wrote rewards to {out_path}", flush=True)
    print(f"[done] {total_samples} samples in {elapsed:.1f}s "
          f"({total_samples / max(elapsed, 1e-6):.1f} samp/s)", flush=True)
    if n_truncated > 0:
        print(f"[warn] {n_truncated} sequences hit max_length={args.max_length} "
              f"and were truncated. Consider --max-length larger if reward "
              f"quality matters.", flush=True)


if __name__ == "__main__":
    main()