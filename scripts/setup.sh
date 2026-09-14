#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo '需要先安裝 uv：https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
command -v git >/dev/null || { echo '需要先安裝 git'; exit 1; }
command -v nvidia-smi >/dev/null || { echo '找不到 nvidia-smi，請先安裝 NVIDIA 驅動。'; exit 1; }
nvidia-smi
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
uv pip install --python .venv/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
.venv/bin/python -m scripts.check --hardware-only
.venv/bin/python -m scripts.download_models
uv pip freeze --python .venv/bin/python > .cache/installed-requirements.txt
.venv/bin/python -m scripts.check
echo '安裝與權重檢查完成。接著執行 bash scripts/start.sh；GPU 生成仍需 smoke_gpu 驗收。'
