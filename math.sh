#!/usr/bin/env bash
set -euo pipefail

# Math pipeline: generate → reward → compare (×4 c_ver) → reward_quality → difficulty_cost
# Runs all three models independently; each gets its own output subdirectory.
# Resume: skips any stage whose primary output already exists.
#
# Environment variables:
#   MATH_OUTDIR   Root output directory (default: ./results/math)
#   N_SAMPLES     Samples per problem     (default: 512)
#   MATH_MODELS   Space-separated HF model IDs (default: all 3)

MATH_OUTDIR="${MATH_OUTDIR:-./results/math}"
N_SAMPLES="${N_SAMPLES:-512}"
MATH_MODELS="${MATH_MODELS:-Qwen/Qwen2.5-Math-7B Qwen/Qwen2.5-14B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[math] output root : $MATH_OUTDIR"
echo "[math] n_samples   : $N_SAMPLES"
echo "[math] models      : $MATH_MODELS"

for MODEL in $MATH_MODELS; do
    SLUG=$(echo "$MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
    DIR="$MATH_OUTDIR/$SLUG"
    mkdir -p "$DIR"

    echo ""
    echo "[math] ── model: $MODEL ──────────────────────────────────"

    # ── Generation ──
    if [ -f "$DIR/generations.jsonl" ]; then
        echo "[math] skip gen  (exists): $DIR/generations.jsonl"
    else
        echo "[math] gen: $MODEL → $DIR"
        python "$SCRIPT_DIR/math/gen.py" \
            --model "$MODEL" \
            --n-samples "$N_SAMPLES" \
            --output-dir "$DIR"
    fi

    # ── Reward scoring ──
    if [ -f "$DIR/rewards.jsonl" ]; then
        echo "[math] skip reward (exists): $DIR/rewards.jsonl"
    else
        echo "[math] reward: $DIR"
        python "$SCRIPT_DIR/math/reward.py" \
            --input "$DIR/generations.jsonl" \
            --output "$DIR/rewards.jsonl"
    fi

    # ── Compare (4 c_ver values) ──
    for C_VER in 1 10 20 30; do
        CDIR="$DIR/compare_c_ver_${C_VER}"
        if [ -d "$CDIR" ]; then
            echo "[math] skip compare c_ver=$C_VER (exists): $CDIR"
        else
            echo "[math] compare c_ver=$C_VER: $DIR"
            python "$SCRIPT_DIR/math/compare.py" \
                --generations "$DIR/generations.jsonl" \
                --rewards     "$DIR/rewards.jsonl" \
                --reward-key  r_last \
                --c-rew 1 \
                --c-ver "$C_VER" \
                --out-dir "$CDIR"
        fi
    done

    # ── Reward quality (alignment diagnostics) ──
    if [ -d "$DIR/reward_quality" ]; then
        echo "[math] skip reward_quality (exists): $DIR/reward_quality"
    else
        echo "[math] reward_quality: $DIR"
        python "$SCRIPT_DIR/math/reward_quality.py" \
            --generations "$DIR/generations.jsonl" \
            --rewards     "$DIR/rewards.jsonl" \
            --reward-key  r_last \
            --out-dir     "$DIR/reward_quality"
    fi

    # ── Difficulty-cost analysis ──
    if [ -d "$DIR/difficulty_cost" ]; then
        echo "[math] skip difficulty_cost (exists): $DIR/difficulty_cost"
    else
        echo "[math] difficulty_cost: $DIR"
        python "$SCRIPT_DIR/math/difficulty_cost.py" \
            --generations "$DIR/generations.jsonl" \
            --rewards     "$DIR/rewards.jsonl" \
            --reward-key  r_last \
            --out-dir     "$DIR/difficulty_cost"
    fi

done

echo ""
echo "[math] done. Results in $MATH_OUTDIR"
