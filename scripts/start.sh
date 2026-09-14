#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -x .venv/bin/python ]] || { echo '請先執行 bash scripts/setup.sh'; exit 1; }
.venv/bin/python -m scripts.check
exec .venv/bin/python app.py "$@"
