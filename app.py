"""Adapted from hemil124/virtual-tryon, Apache-2.0.
Source commit: 29f4ad42a29af63c71a7bc9e53ea7cd5342694c1 (see NOTICE).
Local changes: offline cache/setup, input limits, local examples and launch options.
CPU retains the Space's sampling; GPU uses CPU-initialized noise (see NOTICE).
"""
import os
import argparse
import gc
import math
import threading
import time
from typing import Optional

from tryon.config import configure

configure()
os.environ["HF_HUB_OFFLINE"] = "1"

import gradio as gr
import numpy as np
from PIL import Image

# ─────────────────────────── CONFIG ──────────────────────────── #

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
EXAMPLES_DIR = os.path.join(SCRIPT_DIR, "examples")

CATEGORIES = ["tops", "bottoms", "one-pieces"]
GARMENT_PHOTO_TYPES = ["model", "flat-lay"]

# ──────────────────────── PIPELINE LOADER ────────────────────── #

_pipeline_lock = threading.RLock()
_pipeline: Optional[object] = None
_pipeline_device = None


def get_pipeline(device='cpu'):
    """Keep one pipeline; serialize switching with inference via the same lock."""
    global _pipeline, _pipeline_device
    if device not in ('cpu', 'cuda'):
        raise ValueError('請選擇 CPU 或 GPU。')
    with _pipeline_lock:
        if device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('GPU 不可用；請選 CPU，或執行 bash scripts/setup.sh cuda 安裝 CUDA 版。')
        if _pipeline is not None and _pipeline_device != device:
            previous_device = _pipeline_device
            _pipeline = None
            _pipeline_device = None
            gc.collect()
            if previous_device == 'cuda':
                import torch
                torch.cuda.empty_cache()
        if _pipeline is None:
            required = ["model.safetensors", "dwpose/yolox_l.onnx", "dwpose/dw-ll_ucoco_384.onnx"]
            if any(not os.path.isfile(os.path.join(WEIGHTS_DIR, name)) for name in required):
                raise RuntimeError("Missing weights. Run bash scripts/setup.sh first.")
            from fashn_vton import TryOnPipeline
            if device == 'cuda':
                import onnxruntime as ort
                if hasattr(ort, 'preload_dlls'):
                    ort.preload_dlls()
                from tryon.sampling import CpuNoisePipeline
                factory = CpuNoisePipeline
            else:
                factory = TryOnPipeline
            print(f"Loading pipeline on {device}...")
            _pipeline = factory(weights_dir=WEIGHTS_DIR, device=device)
            _pipeline_device = device
            print("Pipeline ready!")
    return _pipeline


# ─────────────────────────── INFERENCE ───────────────────────── #


def try_on(
    person_image,
    garment_image,
    category: str,
    garment_photo_type: str,
    num_timesteps: int,
    guidance_scale: float,
    seed: int,
    segmentation_free: bool,
    execution_device: str = 'cpu',
):
    """Run virtual try-on inference."""
    if person_image is None:
        raise gr.Error("Please upload a person image.")
    if garment_image is None:
        raise gr.Error("Please upload a garment image.")
    if execution_device not in ('cpu', 'cuda'):
        raise gr.Error('請選擇 CPU 或 GPU。')

    # Normalise seed
    if seed is None or (isinstance(seed, (int, float)) and seed < 0):
        seed = 42
    if not isinstance(seed, (int, float)) or isinstance(seed, bool) or not math.isfinite(seed) or not 0 <= seed <= 2**32 - 1:
        raise gr.Error("Seed must be a finite number between 0 and 4294967295.")
    seed = int(seed)
    if category not in CATEGORIES or garment_photo_type not in GARMENT_PHOTO_TYPES:
        raise gr.Error("Invalid category or photo type.")
    if not isinstance(num_timesteps, (int, float)) or isinstance(num_timesteps, bool) or not 10 <= num_timesteps <= 50 or int(num_timesteps) != num_timesteps:
        raise gr.Error("Sampling Steps must be an integer between 10 and 50.")
    if not isinstance(guidance_scale, (int, float)) or isinstance(guidance_scale, bool) or not 1 <= guidance_scale <= 3:
        raise gr.Error("Guidance Scale must be between 1 and 3.")
    if not isinstance(segmentation_free, bool):
        raise gr.Error("Segmentation-Free must be a boolean.")

    # Ensure PIL RGB
    def to_pil(x):
        if isinstance(x, np.ndarray):
            x = Image.fromarray(x)
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        return Image.open(x).convert("RGB")

    person_img = to_pil(person_image)
    garment_img = to_pil(garment_image)

    if any(min(im.size) < 64 or im.width * im.height > 20_000_000 for im in (person_img, garment_img)):
        raise gr.Error("Images must be at least 64 pixels per side and at most 20 megapixels.")

    try:
        with _pipeline_lock:
            started = time.perf_counter()
            result = get_pipeline(execution_device)(
                person_image=person_img,
                garment_image=garment_img,
                category=category,
                garment_photo_type=garment_photo_type,
                num_samples=1,
                num_timesteps=int(num_timesteps),
                guidance_scale=guidance_scale,
                seed=seed,
                segmentation_free=segmentation_free,
            )
            elapsed = time.perf_counter() - started
        mode = 'GPU / CUDA' if execution_device == 'cuda' else 'CPU / FP32'
        return result.images[0], f"✅ Done! {mode}｜{elapsed:.1f} 秒（含載入）｜CPU FP32 初始噪聲｜無修補"
    except Exception as e:
        return None, f"❌ Error: {e}"


# ─────────────────────────── GRADIO UI ───────────────────────── #

CUSTOM_CSS = """
body { font-family: 'Inter', sans-serif; }

.contain img {
    object-fit: contain !important;
    max-height: 520px !important;
}

#run-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.75rem !important;
    border-radius: 12px !important;
    transition: opacity 0.2s;
}
#run-btn:hover { opacity: 0.85; }

.status-box textarea {
    font-size: 0.9rem !important;
    color: #a3e635 !important;
    background: #1e1e2e !important;
    border-radius: 8px !important;
}

.gr-accordion { border-radius: 10px !important; }
"""

BANNER_MD = """
# 本機 AI 試衣間
上傳人物與衣服照片，選擇 **GPU 或 CPU**，直接生成試穿結果。
GPU 較快；CPU 可能需要二十多分鐘。兩者都使用 CPU 初始噪聲，但不保證結果完全相同。
"""

TIPS_HTML = """
<div style="display: flex; justify-content: center; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 20px; font-size: 0.95rem; color: #a1a1aa;">
    <div style="font-weight: 600; color: #e4e4e7;">💡 Tips for best results:</div>
    <div>👤 Single person, clearly visible</div>
    <div style="color: #52525b;">|</div>
    <div>👕 Match category to garment type</div>
    <div style="color: #52525b;">|</div>
    <div>📸 Use "flat-lay" for product shots</div>
    <div style="color: #52525b;">|</div>
    <div>📐 2:3 aspect ratio optimal</div>
</div>
"""

person_example = os.path.join(EXAMPLES_DIR, "model.webp")
garment_example = os.path.join(EXAMPLES_DIR, "garment.webp")

with gr.Blocks(title="FASHN VTON — Virtual Try-On",
               analytics_enabled=False, delete_cache=(3600, 86400)) as demo:

    gr.Markdown(BANNER_MD)
    gr.HTML(TIPS_HTML)

    with gr.Row(equal_height=False):

        # ── Column 1 : Person ──────────────────────────────────
        with gr.Column(scale=1):
            person_in = gr.Image(
                label="Person Image",
                type="pil",
                sources=["upload", "clipboard"],
                elem_classes=["contain"],
            )
            if os.path.exists(person_example):
                gr.Examples(
                    examples=[[person_example]],
                    inputs=[person_in],
                    label="Person Example",
                )

        # ── Column 2 : Garment ─────────────────────────────────
        with gr.Column(scale=1):
            garment_in = gr.Image(
                label="Garment Image",
                type="pil",
                sources=["upload", "clipboard"],
                elem_classes=["contain"],
            )
            with gr.Row():
                category = gr.Dropdown(
                    choices=CATEGORIES,
                    value="tops",
                    label="Category",
                )
                garment_photo_type = gr.Dropdown(
                    choices=GARMENT_PHOTO_TYPES,
                    value="model",
                    label="Photo Type",
                )
            if os.path.exists(garment_example):
                gr.Examples(
                    examples=[[garment_example]],
                    inputs=[garment_in],
                    label="Garment Example",
                )

        # ── Column 3 : Result ──────────────────────────────────
        with gr.Column(scale=1):
            result_img = gr.Image(
                label="Try-On Result",
                type="pil",
                interactive=False,
                elem_classes=["contain"],
            )
            status = gr.Textbox(
                value="Ready",
                label="Status",
                interactive=False,
                elem_classes=["status-box"],
            )
            execution_device = gr.Radio(
                choices=[('GPU（NVIDIA CUDA）', 'cuda'), ('CPU（原 Space 模式，較慢）', 'cpu')],
                value='cuda', label='推論裝置',
                info='切換後於下次生成載入；只保留一個模型，不自動回退或重試。',
            )
            run_btn = gr.Button("👗 Try On", variant="primary", elem_id="run-btn")

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                num_timesteps = gr.Slider(
                    minimum=10, maximum=50, value=30, step=5,
                    label="Sampling Steps",
                    info="More steps are slower, not a guarantee of better quality. Default: 30.",
                )
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=3.0, value=1.5, step=0.1,
                    label="Guidance Scale",
                    info="How closely to follow the garment details. 1.5 recommended.",
                )
                seed = gr.Number(
                    value=42, label="Seed", precision=0,
                    info="Change seed to get a different variation of the result.",
                )
                segmentation_free = gr.Checkbox(
                    value=True,
                    label="Segmentation-Free (Recommended)",
                    info="Preserves body features and allows unconstrained garment volume.",
                )

    # ── Event ──────────────────────────────────────────────────
    run_btn.click(
        fn=try_on,
        inputs=[
            person_in, garment_in,
            category, garment_photo_type,
            num_timesteps, guidance_scale,
            seed, segmentation_free, execution_device,
        ],
        outputs=[result_img, status],
    )

demo.queue(default_concurrency_limit=1, max_size=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.launch(server_name=args.host, server_port=args.port, share=False, max_file_size="20mb",
                theme=gr.themes.Soft(primary_hue='teal'), css=CUSTOM_CSS)
