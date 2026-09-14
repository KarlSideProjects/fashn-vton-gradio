"""Traditional Chinese Gradio interface for local virtual try-on."""
import argparse
import logging
import os

from tryon.config import ROOT, configure

configure()
os.environ['HF_HUB_OFFLINE'] = '1'  # Download explicitly via setup, never during a request.

import gradio as gr

from tryon.engine import Engine

engine = Engine()
LOG = logging.getLogger(__name__)


def generate(person, garment, category, photo_type, steps, seed):
    # Clear the previous result before running, including when this request fails.
    yield None, None, None, '試穿與手部細節處理中；必要時會自動重試一次，請稍候。'
    try:
        png, record, metadata = engine.generate(person, garment, category, photo_type, steps, seed)
        message = (
            f"完成｜試穿與細節處理 {metadata['inference_seconds']:.1f} 秒｜"
            f"模型載入/準備 {metadata['load_seconds']:.1f} 秒\n"
            f"PyTorch 顯存峰值 {metadata['torch_peak_allocated_gib']:.2f} GiB"
            '（不含 ONNX 與其他程式）'
            f"\n自動處理 {len(metadata['attempts'])} 次；已合成原圖手部，仍請留意邊緣與遮擋效果。"
        )
        yield png, png, record, message
    except Exception as exc:
        LOG.exception('Try-on request failed')
        yield None, None, None, f'未生成：{type(exc).__name__}: {exc}'


def build_app():
    with gr.Blocks(title='本地 AI 試衣間', analytics_enabled=False,
                   theme=gr.themes.Soft(primary_hue='teal'),
                   delete_cache=(3600, 86400)) as demo:
        gr.Markdown('# 本地 AI 試衣間\n上傳人物與衣服照片，看看換裝後的樣子。')
        gr.Markdown('照片在本機處理。生成結果用於穿搭預覽，不能判定實際尺寸與合身程度。')
        with gr.Row():
            person = gr.Image(label='① 人物照片', type='pil', image_mode='RGBA', sources=['upload'], height=400)
            garment = gr.Image(label='② 衣服商品照片', type='pil', image_mode='RGBA', sources=['upload'], height=400)
            result = gr.Image(label='③ 試穿結果', interactive=False, height=400)
        with gr.Row():
            category = gr.Dropdown([('上衣', 'tops'), ('下身', 'bottoms'), ('連身衣', 'one-pieces')],
                                   value='tops', label='要更換的衣物')
            photo_type = gr.Radio([('模特兒穿著', 'model'), ('平拍商品', 'flat-lay')],
                                  value='model', label='商品照片類型')
        with gr.Accordion('生成設定', open=False):
            steps = gr.Slider(20, 50, value=30, step=1, label='生成步數（越多越慢）')
            seed = gr.Number(value=42, precision=0, label='Seed（固定值方便比較）',
                             minimum=0, maximum=2**32 - 1)
        button = gr.Button('開始試穿', variant='primary')
        status = gr.Textbox(label='狀態', value='等待上傳；第一次生成會載入模型。', interactive=False)
        with gr.Row():
            download = gr.File(label='下載試穿圖片')
            report = gr.File(label='下載本次生成紀錄')
        gr.Examples(
            examples=[[str(ROOT / 'examples/model.webp'), str(ROOT / 'examples/garment.webp'), 'tops', 'model']],
            inputs=[person, garment, category, photo_type], cache_examples=False,
            label='先試試官方範例（人物＋模特兒商品照）',
        )
        gr.Markdown('一次處理一張，其餘請求排隊。結果存於 outputs；上傳暫存最長保留約 24 小時後定期清理。')
        button.click(generate, [person, garment, category, photo_type, steps, seed],
                     [result, download, report, status], concurrency_limit=1,
                     concurrency_id='gpu', api_name='try_on')
    return demo.queue(max_size=8, default_concurrency_limit=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    build_app().launch(server_name='127.0.0.1', server_port=args.port, share=False,
                       inbrowser=False, max_file_size='20mb')
