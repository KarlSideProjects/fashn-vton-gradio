"""Explicit SDXL download or one offline inference; run inside .venv-repair."""
import argparse
import json
import os
from pathlib import Path
import time

from tryon.config import configure
from tryon.repair import MODEL_DIR, MODEL_ID, MODEL_REVISION


def download():
    os.environ['HF_HUB_OFFLINE'] = '0'
    from huggingface_hub import snapshot_download
    from scripts.download_models import sha256
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_DIR,
                      allow_patterns=['*.json', '*.txt', '*.fp16.safetensors'])
    paths = sorted(p for p in MODEL_DIR.rglob('*') if p.is_file() and '.cache' not in p.parts
                   and p.name != 'manifest.json')
    data = {'model': MODEL_ID, 'revision': MODEL_REVISION,
            'sha256': {str(p.relative_to(MODEL_DIR)): sha256(p) for p in paths}}
    (MODEL_DIR / 'manifest.json').write_text(json.dumps(data, indent=2))
    print('SDXL inpainting downloaded and checksummed:', MODEL_DIR)


def infer(work):
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from PIL import Image
    from diffusers import AutoPipelineForInpainting
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for local inpainting')
    manifest = json.loads((MODEL_DIR / 'manifest.json').read_text())
    if manifest['revision'] != MODEL_REVISION:
        raise RuntimeError('Repair model revision mismatch; rerun setup_repair.sh')
    job = json.loads((work / 'job.json').read_text())
    box = tuple(job['crop_box'])
    with Image.open(work / 'before.png') as original, Image.open(work / 'mask.png') as mask:
        crop, region = original.convert('RGB').crop(box), mask.convert('L').crop(box)
    size = tuple(max(64, round(d / max(crop.size) * 1024 / 8) * 8) for d in crop.size)
    start = time.perf_counter()
    pipeline = AutoPipelineForInpainting.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float16, variant='fp16',
        use_safetensors=True, local_files_only=True,
    )
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_tiling()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    image = pipeline(
        prompt=job['prompt'], negative_prompt=job['negative_prompt'],
        image=crop.resize(size, Image.Resampling.LANCZOS),
        mask_image=region.resize(size, Image.Resampling.NEAREST),
        width=size[0], height=size[1], strength=job['strength'],
        num_inference_steps=job['steps'], guidance_scale=job['guidance_scale'],
        generator=torch.Generator('cpu').manual_seed(job['seed']),
    ).images[0]
    torch.cuda.synchronize()
    image.resize(crop.size, Image.Resampling.LANCZOS).save(work / 'patch.png')
    metrics = {'load_seconds': round(load_seconds, 3),
               'inference_seconds': round(time.perf_counter()-start, 3),
               'inference_size': size, 'gpu': torch.cuda.get_device_name(),
               'torch_peak_allocated_gib': round(torch.cuda.max_memory_allocated()/2**30, 3),
               'memory_note': 'PyTorch only; excludes other processes.',
               'quality': 'not_automatically_verified'}
    (work / 'metrics.json').write_text(json.dumps(metrics))


if __name__ == '__main__':
    configure()
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path, nargs='?')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        download()
    elif args.work:
        infer(args.work)
    else:
        parser.error('provide a work directory or --download')
