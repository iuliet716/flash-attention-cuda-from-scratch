#!/bin/bash
# Run every per-step measurement script.
# Figures -> docs/assets/*.png, raw numbers -> benchmarks/results/*.json
# Usage: source ../env.sh (or activate the flash_attn env), then ./run_all.sh
set -e
cd "$(dirname "$0")"
for s in 00 01 02 03 04; do
    echo "===== bench_step${s}.py ====="
    python "bench_step${s}.py"
done
