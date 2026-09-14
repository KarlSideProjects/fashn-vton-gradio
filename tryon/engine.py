"""Serialized, CUDA-only access to the real FASHN pipeline."""
import json
import threading
import time
from datetime import datetime, timezone

from .config import OUTPUTS, UPSTREAM_COMMIT, WEIGHTS
from .core import image_digest, prepare_image, save_result, validate_options
from .hands import HandRepairError, restore_hands


class Engine:
    def __init__(self):
        self._pipeline = None
        self._lock = threading.Lock()

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        missing = [str(WEIGHTS / name) for name in (
            'model.safetensors', 'dwpose/yolox_l.onnx', 'dwpose/dw-ll_ucoco_384.onnx'
        ) if not (WEIGHTS / name).is_file()]
        if missing:
            raise RuntimeError('缺少模型權重，請先執行 bash scripts/setup.sh。缺少：' + ', '.join(missing))
        import torch
        import onnxruntime as ort
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用；本程式不會改用 CPU。請檢查 NVIDIA 驅動與 PyTorch。')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('此 GPU 不支援本專案要求的 BF16，停止載入。')
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('ONNX Runtime 缺少 CUDAExecutionProvider。')
        # Load NVIDIA libraries before ONNX creates its sessions.
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        from fashn_vton import TryOnPipeline
        pipeline = TryOnPipeline(weights_dir=str(WEIGHTS), device='cuda')
        pose = pipeline.pose_model.pose_estimation
        for session in (pose.session_det, pose.session_pose):
            if session.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError('DWPose 未成功使用 CUDA，停止生成。請檢查 CUDA/cuDNN 動態函式庫。')
            session.disable_fallback()
        self._pipeline = pipeline
        return pipeline

    def generate(self, person, garment, category, photo_type, steps, seed):
        person, garment = prepare_image(person), prepare_image(garment)
        options = validate_options(category, photo_type, steps, seed)
        with self._lock:
            start = time.perf_counter()
            cold_start = self._pipeline is None
            pipeline = self._load()
            import torch
            torch.cuda.synchronize()
            load_seconds = time.perf_counter() - start
            torch.cuda.reset_peak_memory_stats()
            inference_start = time.perf_counter()
            try:
                from PIL import Image
                import numpy as np
                from fashn_human_parser import LABELS_TO_IDS
                if LABELS_TO_IDS.get('hands') != 13:
                    raise RuntimeError('手部分割標籤與預期不同，停止處理。')
                source_labels = pipeline.hp_model.predict(person)
                attempts = []
                for attempt in range(2):
                    actual_options = {**options, 'seed': (options['seed'] + attempt) % 2**32}
                    result = pipeline(person_image=person, garment_image=garment, **actual_options)
                    if len(result.images) != 1:
                        raise RuntimeError('模型未回傳預期的一張圖片，結果未儲存。')
                    candidate = result.images[0]
                    output_labels = pipeline.hp_model.predict(candidate)
                    aligned_labels = np.asarray(Image.fromarray(source_labels.astype('uint8')).resize(
                        candidate.size, Image.Resampling.NEAREST))
                    try:
                        repaired, hand_report = restore_hands(person, candidate, aligned_labels, output_labels)
                    except HandRepairError as exc:
                        attempts.append({'seed': actual_options['seed'], 'status': 'rejected', 'reason': str(exc)})
                        if attempt == 1:
                            raise HandRepairError('兩次自動嘗試均無法可靠保留手部，未輸出圖片。' + str(exc)) from exc
                    else:
                        attempts.append({'seed': actual_options['seed'], 'status': 'restored'})
                        break
                torch.cuda.synchronize()
                inference_seconds = time.perf_counter() - inference_start
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError('GPU 顯存不足，已停止；請關閉其他 GPU 程式後重試。未降低畫質或改用 CPU。') from exc
            if len(result.images) != 1:
                raise RuntimeError('模型未回傳預期的一張圖片，結果未儲存。')
            metadata = {
                'created_at': datetime.now(timezone.utc).isoformat(),
                'upstream_commit': UPSTREAM_COMMIT,
                'options': actual_options,
                'requested_options': options,
                'hand_repair': hand_report,
                'attempts': attempts,
                'timing_scope': 'source parsing + candidate generation(s) + hand parsing and compositing',
                'person_pixel_sha256': image_digest(person),
                'garment_pixel_sha256': image_digest(garment),
                'gpu': torch.cuda.get_device_name(),
                'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'cold_start': cold_start,
                'load_seconds': round(load_seconds, 3),
                'inference_seconds': round(inference_seconds, 3),
                'torch_peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3),
                'torch_peak_reserved_gib': round(torch.cuda.max_memory_reserved() / 2**30, 3),
                'memory_note': 'PyTorch allocator only; excludes ONNX allocations and other processes.',
                'result_size': list(repaired.size),
            }
            manifest = WEIGHTS / 'manifest.json'
            if manifest.exists():
                metadata['weights_manifest'] = json.loads(manifest.read_text(encoding='utf-8'))
            png, record = save_result(OUTPUTS, repaired, metadata)
            return str(png), str(record), metadata
