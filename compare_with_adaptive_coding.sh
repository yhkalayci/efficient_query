#!/usr/bin/env bash
set -euo pipefail

# Run compare_with_adaptive.py for the coding model (×4 c_ver).
# Assumes gen + verify + reward already done. Skips any output dir that already exists.
#
# Environment variables:
#   CODING_OUTDIR   Output directory (default: ./results/coding)
#   G_MAX           Max number of groups (default: 10)

CODING_OUTDIR="${CODING_OUTDIR:-./results/coding}"
G_MAX="${G_MAX:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[coding-adaptive] output dir : $CODING_OUTDIR"
echo "[coding-adaptive] G_max      : $G_MAX"

for C_VER in 1 10 20 30; do
    CDIR="$CODING_OUTDIR/compare_with_adaptive_c_ver_${C_VER}"
    if [ -d "$CDIR" ]; then
        echo "[coding-adaptive] skip c_ver=$C_VER (exists): $CDIR"
    else
        echo "[coding-adaptive] c_ver=$C_VER → $CDIR"
        python "$SCRIPT_DIR/coding/compare_with_adaptive.py" \
            --generations "$CODING_OUTDIR/verified.jsonl" \
            --rewards     "$CODING_OUTDIR/rewards.jsonl" \
            --reward-key  r_score \
            --c-rew 1 \
            --c-ver "$C_VER" \
            --G-max "$G_MAX" \
            --out-dir "$CDIR" \
            --task Coding \
            --model-name "$CODING_MODEL" \
            --reward-model-name "CodeScaler-8B"
    fi
done

echo ""
echo "[coding-adaptive] done. Results in $CODING_OUTDIR"
