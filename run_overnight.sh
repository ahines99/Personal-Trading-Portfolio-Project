#!/bin/bash
# Overnight runner: wait for yfinance baseline to finish, then launch 16-test retest
# Usage: bash run_overnight.sh

cd "$(dirname "$0")"

BASELINE_RESULT="results/_yfinance_test/tearsheet.csv"

echo "=================================================================="
echo "OVERNIGHT RUNNER STARTED at $(date)"
echo "=================================================================="
echo "Waiting for yfinance baseline to complete..."

# Wait for baseline result file (poll every 60 seconds)
while [ ! -f "$BASELINE_RESULT" ]; do
    sleep 60
done

echo ""
echo "yfinance baseline COMPLETE at $(date)"
echo ""
cat "$BASELINE_RESULT" | head -10
echo ""
echo "=================================================================="
echo "Launching 16-test retest at $(date)"
echo "=================================================================="

python run_isolated_retest.py \
    --start-from CZ_all_signals \
    --baseline-from "$BASELINE_RESULT"

echo ""
echo "=================================================================="
echo "OVERNIGHT RUNNER COMPLETE at $(date)"
echo "=================================================================="
