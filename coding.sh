#!/usr/bin/env bash
set -euo pipefail

# Coding pipeline: gen → verify (8 CPU workers) → reward → compare (×4) → reward_quality → difficulty_cost
# Resume: skips any stage whose primary output already exists.
#
# Environment variables:
#   CODING_OUTDIR    Output directory       (default: ./results/coding)
#   N_SAMPLES        Samples per problem    (default: 512)
#   CODING_MODEL     HF model ID           (default: Qwen/Qwen2.5-Coder-3B)
#   VERIFY_WORKERS   CPU workers for verify (default: 8)

CODING_OUTDIR="${CODING_OUTDIR:-./results/coding}"
N_SAMPLES="${N_SAMPLES:-512}"
CODING_MODEL="${CODING_MODEL:-Qwen/Qwen2.5-Coder-3B}"
VERIFY_WORKERS="${VERIFY_WORKERS:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$CODING_OUTDIR"

echo "[coding] output dir     : $CODING_OUTDIR"
echo "[coding] n_samples      : $N_SAMPLES"
echo "[coding] model          : $CODING_MODEL"
echo "[coding] verify workers : $VERIFY_WORKERS"

# ── Generation ──
if [ -f "$CODING_OUTDIR/generations.jsonl" ]; then
    echo "[coding] skip gen (exists): $CODING_OUTDIR/generations.jsonl"
else
    echo "[coding] gen: $CODING_MODEL → $CODING_OUTDIR"
    python "$SCRIPT_DIR/coding/gen.py" \
        --model      "$CODING_MODEL" \
        --n-samples  "$N_SAMPLES" \
        --output-dir "$CODING_OUTDIR"
fi

# ── Verification (CPU, parallel) ──
if [ -f "$CODING_OUTDIR/verified.jsonl" ]; then
    echo "[coding] skip verify (exists): $CODING_OUTDIR/verified.jsonl"
else
    echo ""
    echo "[coding] NOTE: CPU-intensive verification with $VERIFY_WORKERS workers — may take significant time"
    echo ""
    python "$SCRIPT_DIR/coding/verify.py" \
        --input   "$CODING_OUTDIR/generations.jsonl" \
        --output  "$CODING_OUTDIR/verified.jsonl" \
        --workers "$VERIFY_WORKERS"
fi

# ── Reward scoring ──
if [ -f "$CODING_OUTDIR/rewards.jsonl" ]; then
    echo "[coding] skip reward (exists): $CODING_OUTDIR/rewards.jsonl"
else
    echo "[coding] reward: $CODING_OUTDIR"
    python "$SCRIPT_DIR/coding/reward.py" \
        --generations "$CODING_OUTDIR/verified.jsonl" \
        --out         "$CODING_OUTDIR/rewards.jsonl"
fi

# ── Compare (4 c_ver values) ──
for C_VER in 1 10 20 30; do
    CDIR="$CODING_OUTDIR/compare_c_ver_${C_VER}"
    if [ -d "$CDIR" ]; then
        echo "[coding] skip compare c_ver=$C_VER (exists): $CDIR"
    else
        echo "[coding] compare c_ver=$C_VER"
        python "$SCRIPT_DIR/coding/compare.py" \
            --generations "$CODING_OUTDIR/verified.jsonl" \
            --rewards     "$CODING_OUTDIR/rewards.jsonl" \
            --reward-key  r_score \
            --c-rew 1 \
            --c-ver "$C_VER" \
            --out-dir "$CDIR" \
            --task Coding \
            --model-name "$CODING_MODEL" \
            --reward-model-name "CodeScaler-8B"
    fi
done

# ── Reward quality (alignment diagnostics) ──
if [ -d "$CODING_OUTDIR/reward_quality" ]; then
    echo "[coding] skip reward_quality (exists): $CODING_OUTDIR/reward_quality"
else
    echo "[coding] reward_quality"
    python "$SCRIPT_DIR/coding/reward_quality.py" \
        --generations "$CODING_OUTDIR/verified.jsonl" \
        --rewards     "$CODING_OUTDIR/rewards.jsonl" \
        --reward-key  r_score \
        --out-dir     "$CODING_OUTDIR/reward_quality"
fi

# ── Difficulty-cost analysis ──
if [ -d "$CODING_OUTDIR/difficulty_cost" ]; then
    echo "[coding] skip difficulty_cost (exists): $CODING_OUTDIR/difficulty_cost"
else
    echo "[coding] difficulty_cost"
    python "$SCRIPT_DIR/coding/difficulty_cost.py" \
        --generations "$CODING_OUTDIR/verified.jsonl" \
        --rewards     "$CODING_OUTDIR/rewards.jsonl" \
        --reward-key  r_score \
        --out-dir     "$CODING_OUTDIR/difficulty_cost"
fi

echo ""
echo "[coding] done. Results in $CODING_OUTDIR"
