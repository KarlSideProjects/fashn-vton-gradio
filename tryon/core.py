"""Image validation and persistence; independent of GPU dependencies."""
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps


def prepare_image(image):
    if not isinstance(image, Image.Image):
        raise ValueError('請上傳人物圖片與衣服圖片。')
    if min(image.size) < 64 or image.width * image.height > 20_000_000:
        raise ValueError('圖片短邊至少 64 像素，總像素不得超過 2000 萬。')
    image = ImageOps.exif_transpose(image)
    rgba = image.convert('RGBA')
    background = Image.new('RGBA', rgba.size, 'white')
    background.alpha_composite(rgba)
    return background.convert('RGB')


def validate_options(category, photo_type, steps, seed):
    if category not in ('tops', 'bottoms', 'one-pieces'):
        raise ValueError('不支援的衣物類別。')
    if photo_type not in ('model', 'flat-lay'):
        raise ValueError('請選擇商品照片類型。')
    for name, value, low, high in [('步數', steps, 20, 50), ('Seed', seed, 0, 2**32 - 1)]:
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or int(value) != value or not low <= value <= high):
            raise ValueError(f'{name} 必須是 {low}–{high} 的整數。')
    return dict(category=category, garment_photo_type=photo_type,
                num_timesteps=int(steps), seed=int(seed), num_samples=1,
                guidance_scale=1.5, segmentation_free=True)


def image_digest(image):
    header = f'{image.mode}:{image.size}:'.encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def save_result(directory: Path, image: Image.Image, metadata: dict):
    directory.mkdir(parents=True, exist_ok=True)
    key = uuid4().hex
    png, record = directory / f'{key}.png', directory / f'{key}.json'
    try:
        image.save(png, format='PNG')
        record.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        png.unlink(missing_ok=True)
        record.unlink(missing_ok=True)
        raise
    return png, record
