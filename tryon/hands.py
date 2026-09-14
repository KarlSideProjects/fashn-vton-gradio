"""Conservative hand compositing, not an anatomical correctness classifier."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

class HandRepairError(RuntimeError):
    pass


def _hand_regions(mask):
    remaining = Image.fromarray(mask.astype('uint8')).copy()
    regions = []
    while remaining.getbbox():
        y, x = np.argwhere(np.asarray(remaining) == 1)[0]
        ImageDraw.floodfill(remaining, (int(x), int(y)), 2)
        region = np.asarray(remaining) == 2
        ImageDraw.floodfill(remaining, (int(x), int(y)), 0)
        # ponytail: ignore <32-pixel parser specks; use refined masks for tiny hands.
        if region.sum() >= 32:
            regions.append(region)
            if len(regions) > 2:
                raise HandRepairError('手部分割破碎或超過兩個區域，無法可靠保留。')
    return regions


def _check_geometry(a, b, generated_labels, radius):
    na, nb = int(a.sum()), int(b.sum())
    if min(na, nb) < 32:
        raise HandRepairError('未能可靠辨識原圖或結果的手部，無法自動保留。')
    intersection = int((a & b).sum())
    iou = intersection / int((a | b).sum())
    coverage = intersection / na
    # Fixed thresholds in normalized image coordinates; require GPU-image calibration.
    centroid_shift = float(np.linalg.norm(np.argwhere(a).mean(0) - np.argwhere(b).mean(0)))
    mask = Image.fromarray(a.astype('uint8') * 255)
    dilated = np.asarray(mask.filter(ImageFilter.MaxFilter(2*radius+1))) > 0
    outside = int((b & ~dilated).sum()) / nb
    garments = np.isin(generated_labels, [3, 4, 5, 6])
    occlusion = int((a & garments).sum()) / na
    if iou < 0.6 or coverage < 0.85 or outside > 0.03 or centroid_shift > min(a.shape)*0.015 or occlusion > 0.05:
        raise HandRepairError(
            f'手部輪廓或袖口關係不確定，停止合成 (IoU={iou:.2f}, coverage={coverage:.2f}, '
            f'outside={outside:.2f}, shift={centroid_shift:.1f}px, occlusion={occlusion:.2f})。')
    return {
        'mask_iou': round(iou, 4), 'source_coverage': round(coverage, 4),
        'extra_hand_fraction': round(outside, 4), 'occlusion_fraction': round(occlusion, 4),
        'centroid_shift_pixels': round(centroid_shift, 3),
    }


def restore_hands(source, generated, source_labels, generated_labels):
    size = generated.size
    expected = (size[1], size[0])
    if source_labels.shape != expected or generated_labels.shape != expected:
        raise ValueError('手部分割尺寸不一致，停止合成。')
    original = source.resize(size, Image.Resampling.LANCZOS).convert('RGB')
    a, b = source_labels == 13, generated_labels == 13
    radius = max(1, round(min(size) / 300))
    report = _check_geometry(a, b, generated_labels, radius)
    source_regions, output_regions = _hand_regions(a), _hand_regions(b)
    if not source_regions or len(source_regions) != len(output_regions):
        raise HandRepairError('原圖與結果的手部區域數量不符，停止合成。')
    regions = []
    for region in source_regions:
        index = max(range(len(output_regions)), key=lambda i: int((region & output_regions[i]).sum()))
        match = output_regions.pop(index)
        regions.append(_check_geometry(region, match, generated_labels, radius))
    a = np.logical_or.reduce(source_regions)
    mask = Image.fromarray(a.astype('uint8') * 255)
    # Copy fingers completely. Feather only where the source arm meets the wrist.
    arms = Image.fromarray((source_labels == 12).astype('uint8') * 255)
    wrist = np.asarray(arms.filter(ImageFilter.MaxFilter(2*radius+1))) > 0
    feather = np.asarray(mask.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)/255
    alpha = np.where(a, np.where(wrist, feather, 1), 0)
    output = np.asarray(generated.convert('RGB'), dtype=np.float32)
    pixels = np.asarray(original, dtype=np.float32)
    blended = np.rint(pixels*alpha[..., None] + output*(1-alpha[..., None])).astype('uint8')
    return Image.fromarray(blended), {
        'status': 'restored', 'method': 'source_hand_composite_v2',
        **report, 'regions': regions,
        'feather': 'wrist_only',
        'anatomical_correctness': 'not_verified',
        'note': 'Heuristic geometry gate; segmentation errors and boundary artifacts remain possible.'
    }
