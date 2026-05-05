#!/usr/bin/env bash
set -euo pipefail

# Run compare_with_adaptive.py for all math models (×4 c_ver).
# Assumes gen + reward already done. Skips any output dir that already exists.
#
# Environment variables:
#   MATH_OUTDIR   Root output directory (default: ./results/math)
#   MATH_MODELS   Space-separated HF model IDs (default: all 3)
#   G_MAX         Max number of groups (default: 10)

MATH_OUTDIR="${MATH_OUTDIR:-./results/math}"
MATH_MODELS="${MATH_MODELS:-Qwen/Qwen2.5-Math-7B Qwen/Qwen2.5-14B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
G_MAX="${G_MAX:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[math-adaptive] output root : $MATH_OUTDIR"
echo "[math-adaptive] models      : $MATH_MODELS"
echo "[math-adaptive] G_max       : $G_MAX"

for MODEL in $MATH_MODELS; do
    SLUG=$(echo "$MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
    DIR="$MATH_OUTDIR/$SLUG"

    echo ""
    echo "[math-adaptive] ── model: $MODEL ──────────────────────────────────"

    for C_VER in 1 10 20 30; do
        CDIR="$DIR/compare_with_adaptive_c_ver_${C_VER}"
        if [ -d "$CDIR" ]; then
            echo "[math-adaptive] skip c_ver=$C_VER (exists): $CDIR"
        else
            echo "[math-adaptive] c_ver=$C_VER → $CDIR"
            python "$SCRIPT_DIR/math/compare_with_adaptive.py" \
                --generations "$DIR/generations.jsonl" \
                --rewards     "$DIR/rewards.jsonl" \
                --reward-key  r_last \
                --c-rew 1 \
                --c-ver "$C_VER" \
                --G-max "$G_MAX" \
                --out-dir "$CDIR" \
                --task Math \
                --model-name "$MODEL" \
                --reward-model-name "Qwen2.5-Math-PRM-7B"
        fi
    done
done

echo ""
echo "[math-adaptive] done. Results in $MATH_OUTDIR"
