#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Run all project test scripts in order.
# Usage: bash scripts/run_tests.sh
# -----------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -d "venv" ]]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
fi

echo "==> test_preprocess"
python scripts/test_preprocess.py

echo "==> test_predictor"
python scripts/test_predictor.py --employee-number 1

echo "==> test_prediction_integration"
python scripts/test_prediction_integration.py

echo "==> test_history_dashboard"
python scripts/test_history_dashboard.py

echo ""
echo "All tests passed."
