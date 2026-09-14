"""Conservative hand compositing, not an anatomical correctness classifier."""
import numpy as np
from PIL import Image, ImageFilter

class HandRepairError(RuntimeError):
    pass


def restore_hands(source, generated, source_labels, generated_labels):
    size = generated.size
    expected = (size[1], size[0])
    if source_labels.shape != expected or generated_labels.shape != expected:
        raise ValueError('手部分割尺寸不一致，停止合成。')
    original = source.resize(size, Image.Resampling.LANCZOS).convert('RGB')
    a, b = source_labels == 13, generated_labels == 13
    na, nb = int(a.sum()), int(b.sum())
    if min(na, nb) < 32:
        raise HandRepairError('未能可靠辨識原圖或結果的手部，無法自動保留。')
    intersection = int((a & b).sum())
    iou = intersection / int((a | b).sum())
    coverage = intersection / na
    # Fixed thresholds in normalized image coordinates; require GPU-image calibration.
    centroid_shift = float(np.linalg.norm(np.argwhere(a).mean(0) - np.argwhere(b).mean(0)))
    radius = max(1, round(min(size) / 300))
    mask = Image.fromarray(a.astype('uint8') * 255)
    dilated = np.asarray(mask.filter(ImageFilter.MaxFilter(2*radius+1))) > 0
    outside = int((b & ~dilated).sum()) / nb
    garments = np.isin(generated_labels, [3, 4, 5, 6])
    occlusion = int((a & garments).sum()) / na
    if iou < 0.6 or coverage < 0.85 or outside > 0.03 or centroid_shift > min(size)*0.015 or occlusion > 0.05:
        raise HandRepairError(f'手部輪廓或袖口關係不確定，停止合成 (IoU={iou:.2f}, coverage={coverage:.2f}, outside={outside:.2f})。')
    # Feather inward only: never paste old clothing outside the original hand mask.
    alpha = np.asarray(mask.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)/255
    alpha = np.where(a, np.minimum(alpha*1.5, 1), 0)
    output = np.asarray(generated.convert('RGB'), dtype=np.float32)
    pixels = np.asarray(original, dtype=np.float32)
    blended = np.rint(pixels*alpha[..., None] + output*(1-alpha[..., None])).astype('uint8')
    return Image.fromarray(blended), {
        'status': 'restored', 'method': 'source_hand_composite_v1',
        'mask_iou': round(iou, 4), 'source_coverage': round(coverage, 4),
        'extra_hand_fraction': round(outside, 4), 'occlusion_fraction': round(occlusion, 4),
        'centroid_shift_pixels': round(centroid_shift, 3),
        'anatomical_correctness': 'not_verified',
        'note': 'Heuristic geometry gate; segmentation errors and boundary artifacts remain possible.'
    }
