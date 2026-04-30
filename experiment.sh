#!/usr/bin/env bash
set -euo pipefail

# Run the full math + coding experiment pipeline sequentially.
# Halts immediately if either pipeline fails.
# See math.sh and coding.sh for environment variable configuration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[experiment] starting math pipeline..."
bash "$SCRIPT_DIR/math.sh"

echo ""
echo "[experiment] starting coding pipeline..."
bash "$SCRIPT_DIR/coding.sh"

echo ""
echo "[experiment] all done."
