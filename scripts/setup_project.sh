#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-time project setup: virtual environment, dependencies, database, model.
#
# Usage (from anywhere):
#   bash scripts/setup_project.sh
#
# Or from project root:
#   chmod +x scripts/setup_project.sh
#   ./scripts/setup_project.sh
# -----------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> Project root: $PROJECT_ROOT"

# Create virtual environment if missing
if [[ ! -d "venv" ]]; then
  echo "==> Creating virtual environment (venv/)"
  python3 -m venv venv
fi

# Activate venv
# shellcheck source=/dev/null
source venv/bin/activate

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing dependencies"
pip install -r requirements.txt

echo "==> Initializing SQLite database"
python scripts/init_database.py

echo "==> Training model artifacts"
python scripts/train_model.py

echo ""
echo "Setup complete."
echo "Start the app with:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
