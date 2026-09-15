#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo '需要先安裝 uv：https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
command -v git >/dev/null || { echo '需要先安裝 git'; exit 1; }
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR"
if [[ ! -d .venv ]]; then
  uv venv --python 3.12 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "本專案要求 Python 3.12；請檢查 .venv"'
device="${1:-cuda}"
case "$device" in
  cuda) torch_index=https://download.pytorch.org/whl/cu128; torch_build=cu128 ;;
  cpu) torch_index=https://download.pytorch.org/whl/cpu; torch_build=cpu ;;
  *) echo '用法：bash scripts/setup.sh [cuda|cpu]'; exit 1 ;;
esac
uv pip install --python .venv/bin/python "torch==2.8.0+$torch_build" "torchvision==0.23.0+$torch_build" --index-url "$torch_index"
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
.venv/bin/python -m scripts.check --hardware-only
.venv/bin/python -m scripts.download_models
uv pip freeze --python .venv/bin/python > .cache/installed-requirements.txt
.venv/bin/python -m scripts.check
echo '安裝與權重檢查完成。接著執行 bash scripts/start.sh，在介面選擇 GPU 或 CPU。'
