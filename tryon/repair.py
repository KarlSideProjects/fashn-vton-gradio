"""Masked local repair: validate in the UI process, infer in an isolated worker."""
import json
import math
import subprocess
import tempfile
import time
from uuid import uuid4

import numpy as np
from PIL import Image

from .config import ROOT, OUTPUTS, WEIGHTS
from .core import image_digest, prepare_image, save_result

MODEL_ID = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'
MODEL_REVISION = '115134f363124c53c7d878647567d04daf26e41e'
MODEL_DIR = WEIGHTS / 'sdxl-inpaint'
REPAIR_PYTHON = ROOT / '.venv-repair/bin/python'


def prepare_repair(editor, prompt, strength, seed, negative_prompt='', method='sdxl'):
    if not isinstance(editor, dict) or not isinstance(editor.get('background'), Image.Image):
        raise ValueError('請先載入要修補的圖片，並用白色筆刷塗選區域。')
    base = prepare_image(editor['background'])
    mask = np.zeros((base.height, base.width), dtype=bool)
    layers = editor.get('layers')
    if not isinstance(layers, list) or len(layers) > 32:
        raise ValueError('修補圖層格式不正確。')
    for layer in layers:
        if not isinstance(layer, Image.Image) or layer.mode != 'RGBA' or layer.size != base.size:
            raise ValueError('遮罩必須是與原圖同尺寸的透明 RGBA 圖層。')
        mask |= np.asarray(layer.getchannel('A')) > 0
    if not mask.any():
        raise ValueError('尚未塗選修補區域。')
    if mask.all():
        raise ValueError('請保留部分原圖作上下文，不支援整張重畫。')
    if method not in ('sdxl', 'texture'):
        raise ValueError('不支援的修補方式。')
    if not isinstance(prompt, str) or (method == 'sdxl' and not prompt.strip()) or len(prompt) > 1000:
        raise ValueError('請輸入 1–1000 字的修補要求，建議英文描述想要的結果。')
    if not isinstance(negative_prompt, str) or len(negative_prompt) > 1000:
        raise ValueError('避免內容不得超過 1000 字。')
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 0.1 <= strength <= 0.99):
        raise ValueError('重繪強度必須介於 0.1–0.99。')
    if (isinstance(seed, bool) or not isinstance(seed, (int, float))
            or not math.isfinite(seed) or int(seed) != seed or not 0 <= seed < 2**32):
        raise ValueError('Seed 必須是 0–4294967295 的整數。')
    return base, Image.fromarray(mask.astype('uint8') * 255), {
        'prompt': prompt.strip(), 'strength': float(strength), 'seed': int(seed),
        'negative_prompt': negative_prompt.strip(),
        'method': method,
        'steps': 30, 'guidance_scale': 7.5,
    }


def crop_box(mask, padding=64):
    box = mask.getbbox()
    if box is None:
        raise ValueError('遮罩不可為空。')
    x0, y0, x1, y1 = box
    return (max(0, x0-padding), max(0, y0-padding),
            min(mask.width, x1+padding), min(mask.height, y1+padding))


def composite_patch(base, mask, patch, box):
    if patch.size != (box[2]-box[0], box[3]-box[1]):
        raise ValueError('修補結果尺寸不符，未儲存。')
    # Hard mask is intentional: never blur beyond the user's authorized region.
    candidate = base.copy()
    candidate.paste(patch.convert('RGB'), box[:2], mask.crop(box))
    outside = np.asarray(mask) == 0
    if not np.array_equal(np.asarray(base)[outside], np.asarray(candidate)[outside]):
        raise RuntimeError('選區外像素被修改，停止儲存。')
    return candidate


def run_repair(base, mask, options):
    if options['method'] == 'sdxl' and (not REPAIR_PYTHON.is_file() or not (MODEL_DIR / 'manifest.json').is_file()):
        raise RuntimeError('局部修補模型尚未安裝，請執行 bash scripts/setup_repair.sh。')
    start = time.perf_counter()
    directory = OUTPUTS / 'repairs'
    directory.mkdir(parents=True, exist_ok=True)
    # ponytail: one worker per request frees VRAM reliably; cache only if load time becomes a bottleneck.
    with tempfile.TemporaryDirectory(prefix='repair-', dir=ROOT / '.cache/tmp') as temporary:
        from pathlib import Path
        work = Path(temporary)
        base.save(work / 'before.png')
        mask.save(work / 'mask.png')
        box = crop_box(mask)
        (work / 'job.json').write_text(json.dumps({**options, 'crop_box': box}), encoding='utf-8')
        if options['method'] == 'texture':
            import cv2
            crop = np.asarray(base.crop(box))
            region = np.asarray(mask.crop(box))
            Image.fromarray(cv2.inpaint(crop, region, 3, cv2.INPAINT_TELEA)).save(work / 'patch.png')
            (work / 'metrics.json').write_text(json.dumps({
                'backend': 'OpenCV Telea', 'opencv_version': cv2.__version__,
                'quality': 'not_automatically_verified', 'prompt_used': False,
            }))
        else:
            try:
                result = subprocess.run(
                    [str(REPAIR_PYTHON), '-m', 'scripts.inpaint', str(work)], cwd=ROOT,
                    capture_output=True, text=True, timeout=300,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError('局部修補超過 5 分鐘已停止；原圖保持不變。') from exc
            if result.returncode:
                import logging
                logging.getLogger(__name__).error('Local repair worker failed: %s', result.stderr[-4000:])
                raise RuntimeError('局部修補失敗；原圖保持不變。請查看終端紀錄與 GPU 可用顯存。')
        with Image.open(work / 'patch.png') as patch:
            repaired = composite_patch(base, mask, patch, box)
        metrics = json.loads((work / 'metrics.json').read_text())
        metadata = {
            'operation': 'local_inpainting', 'status': 'needs_review',
            'model': MODEL_ID if options['method'] == 'sdxl' else None,
            'revision': MODEL_REVISION if options['method'] == 'sdxl' else None,
            **(options if options['method'] == 'sdxl' else {'method': 'texture', 'inpaint_radius': 3}),
            'requested_options': options,
            'crop_box': box, 'outside_mask_max_difference': 0,
            'base_sha256': image_digest(base), 'mask_sha256': image_digest(mask),
            'before': 'before.png', 'mask': 'mask.png',
            'total_seconds': round(time.perf_counter()-start, 3), **metrics,
        }
        png, record = save_result(work, repaired, metadata)
        # Publish the complete reproducible bundle only after inference and compositing succeed.
        destination = directory / uuid4().hex
        work.rename(destination)
    return str(destination / png.name), str(destination / record.name), metadata
