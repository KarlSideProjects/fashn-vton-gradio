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


def generate(person, garment, category, photo_type, steps, seed, preserve_hands=True):
    # Clear the previous result before running, including when this request fails.
    yield None, None, None, ('試穿與手部細節處理中；必要時會自動重試一次，請稍候。' if preserve_hands
                             else '待修補候選生成中；此次略過原手保留，完成後需人工檢查。')
    try:
        png, record, metadata = engine.generate(person, garment, category, photo_type, steps, seed, preserve_hands)
        message = (
            f"完成｜試穿與細節處理 {metadata['inference_seconds']:.1f} 秒｜"
            f"模型載入/準備 {metadata['load_seconds']:.1f} 秒\n"
            f"PyTorch 顯存峰值 {metadata['torch_peak_allocated_gib']:.2f} GiB"
            '（不含 ONNX 與其他程式）'
            f"\n自動處理 {len(metadata['attempts'])} 次；"
            + ('已合成原圖手部，仍請留意邊緣與遮擋效果。' if preserve_hands
               else '待人工檢查／局部修補：未做手部保留，不代表手部正確。')
        )
        yield png, png, record, message
    except Exception as exc:
        LOG.exception('Try-on request failed')
        yield None, None, None, f'未生成：{type(exc).__name__}: {exc}'


def load_repair_image(value):
    if value is None:
        raise gr.Error('尚無圖片可載入。')
    return {'background': value, 'layers': [], 'composite': value}


def repair(editor, prompt, strength, seed, negative_prompt='', method='sdxl'):
    yield None, None, None, '局部修補中；原圖不變，完成後請比較並決定是否使用。'
    try:
        png, record, metadata = engine.repair(editor, prompt, strength, seed, negative_prompt, method)
        yield png, png, record, (
            f"待人工驗收｜共 {metadata['total_seconds']:.1f} 秒｜選區外像素差：0。"
            '未自動判定修補品質；不滿意可調整遮罩重試，原圖仍在左側。')
    except Exception as exc:
        LOG.exception('Local repair failed')
        yield None, None, None, f'未修補：{exc}'


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
            preserve_hands = gr.Checkbox(value=True,
                label='自動保留原手（若被拒絕，可取消勾選生成待修補候選）')
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
        button.click(generate, [person, garment, category, photo_type, steps, seed, preserve_hands],
                     [result, download, report, status], concurrency_limit=1,
                     concurrency_id='gpu', api_name='try_on')
        gr.Markdown('## 局部修補（實驗性）\n生成成功後會自動載入試穿結果，也可自行上傳圖片，用白色筆刷塗選要重畫的區域。'
                    '僅修改塗選範圍，可修手、袖口或其他局部；不保證解剖或文字正確。')
        load = gr.Button('將目前試穿結果載入修補區')
        with gr.Row():
            editor = gr.ImageEditor(label='修補前／白色筆刷選區', type='pil', image_mode='RGBA',
                sources=['upload'], transforms=(), format='png', height=500,
                brush=gr.Brush(colors=['#ffffff'], default_color='#ffffff', color_mode='fixed'))
            repaired = gr.Image(label='修補後（待驗收）', type='pil', format='png', interactive=False, height=500)
        repair_method = gr.Radio([('SDXL 局部重繪（產生新內容）', 'sdxl'),
                                 ('紋理補洞（小污點；不重建手指）', 'texture')],
                                value='sdxl', label='修補方式')
        gr.Markdown('紋理補洞只延伸周圍像素，不使用提示詞、強度或 Seed；不適合大片選區或重建結構。')
        prompt = gr.Textbox(label='希望選區變成什麼樣子（建議英文）',
                            placeholder='例如：a natural fabric cuff, matching the blue sleeve', max_lines=4)
        negative_prompt = gr.Textbox(label='避免出現的內容（選填，建議英文）',
                                      placeholder='例如：text, logo, pink stain', max_lines=3)
        with gr.Row():
            strength = gr.Slider(0.1, 0.99, value=0.99, step=0.01, label='重繪強度（高值改動較多）')
            repair_seed = gr.Number(value=42, precision=0, minimum=0, maximum=2**32-1, label='修補 Seed')
        fix = gr.Button('只重繪塗選區域', variant='primary')
        repair_status = gr.Textbox(label='修補狀態', interactive=False)
        with gr.Row():
            repair_download = gr.File(label='下載修補圖片（人工確認後使用）')
            repair_report = gr.File(label='下載修補紀錄')
        reuse = gr.Button('採用修補後圖片，繼續選區修補')
        gr.Markdown('不滿意可直接重試或重新載入試穿結果。修前圖、遮罩與紀錄存於 outputs/repairs，'
                    '直到自行刪除；不會上傳雲端。修補會釋放試穿模型，下次試穿需重新載入。')
        load.click(load_repair_image, result, editor, api_name=False)
        result.change(lambda value: load_repair_image(value) if value is not None else gr.skip(),
                      result, editor, queue=False, api_name=False)
        reuse.click(load_repair_image, repaired, editor, api_name=False)
        fix.click(repair, [editor, prompt, strength, repair_seed, negative_prompt, repair_method],
                  [repaired, repair_download, repair_report, repair_status],
                  concurrency_limit=1, concurrency_id='gpu', api_name='local_repair')
    return demo.queue(max_size=8, default_concurrency_limit=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    build_app().launch(server_name='127.0.0.1', server_port=args.port, share=False,
                       inbrowser=False, max_file_size='20mb')
