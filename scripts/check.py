"""Preflight; exits nonzero when required conditions are missing."""
import argparse
import importlib.metadata
import json
import os
import sys

from tryon.config import ROOT, WEIGHTS, configure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware-only', action='store_true')
    args = parser.parse_args()
    configure()
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    import onnxruntime as ort
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
    print(json.dumps({'python': sys.version, 'torch': torch.__version__,
                      'cuda_build': torch.version.cuda,
                      'cuda_available': torch.cuda.is_available(),
                      'onnx_providers': ort.get_available_providers(),
                      'gradio': importlib.metadata.version('gradio')}, indent=2))
    if not torch.cuda.is_available():
        raise RuntimeError('blocked: CUDA unavailable')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('blocked: BF16 unavailable')
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('blocked: ONNX CUDA provider unavailable')
    print('GPU:', torch.cuda.get_device_name(), 'VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
    if args.hardware_only:
        return
    from scripts.download_models import sha256
    data = json.loads((WEIGHTS / 'manifest.json').read_text())
    required = {'weights/model.safetensors', 'weights/dwpose/yolox_l.onnx', 'weights/dwpose/dw-ll_ucoco_384.onnx'}
    if not required <= data['sha256'].keys():
        raise RuntimeError('blocked: incomplete weights manifest')
    for section in ('sha256', 'parser_cache_sha256'):
        for name, digest in data[section].items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path) != digest:
                raise RuntimeError(f'blocked: missing/corrupt weight {name}')
    # Loading is performed during the first request / smoke test, including parser cache verification.
    print('Preflight passed. GPU model load and generation remain unverified until smoke_gpu runs.')


if __name__ == '__main__':
    main()
