#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR"
if [[ ! -d .venv-repair ]]; then
  uv venv --python .venv/bin/python .venv-repair
fi
uv pip install --python .venv-repair/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-repair/bin/python -r requirements-repair.txt
uv pip check --python .venv-repair/bin/python
.venv-repair/bin/python -m scripts.inpaint --download
