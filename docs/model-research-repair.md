# 局部修補模型研究

研究／取回日期：2026-09-14。目標硬體：RTX 4060 Ti 16 GB、32 GB RAM；目前修補環境：`torch==2.8.0`、`diffusers==0.35.1`、`accelerate==1.10.1`、`transformers==4.57.1`。

審查後註記：本稿的完整矩陣與 holdout 是品質推廣建議，不是開始寫實驗工具的阻擋條件。
本輪採用範圍及延期項目以 [主方案](model-research-composite.md) 與
[獨立審查](model-improvement-adversarial-review.md) 為準；不會安裝本文列出的所有模型。

本文件只做研究與實驗設計，沒有下載新的模型、安裝套件、執行 GPU 推論、上傳私人圖片或修改程式。網頁檢索先使用 Firecrawl MCP；需要固定版本、檔案大小或 gated 頁面的內容時，再讀取官方 Hugging Face API／raw 檔案與官方 GitHub 原始檔。所有外部來源均列出精確 URL 與本次取回日期；「官方文件」的能力描述不等於本專案的品質測量。

## 先給結論

對 16 GB 本機，第一個值得實作的不同基線是 **Stable Diffusion 1.5 inpainting + `lllyasviel/control_v11p_sd15_inpaint`**，而不是再調高目前 SDXL 的強度。兩個 checkpoint 都是公開、未 gated 的 fp16 safetensors，且目前 `.venv-repair` 的 Diffusers 0.35.1 已有對應 pipeline 類別；它們的檔案量屬同一個可管理級距，但尚未在本機下載或實測畫質／顯存。

這個基線只值得驗證「局部遮罩是否較能保持上下文、衣料紋理與接縫」，**不能宣稱能修好手指**。ControlNet 的官方 inpaint hint 會把遮罩內像素設成 `-1`，所以異常手部本身不會直接作為控制圖內容；但主 inpainting 輸入仍含同一張壞圖，遮罩漏掉的錯誤也仍會被保留。它不是手部 mesh、21 關節或解剖判定器。

手部若是可辨識、原尺寸至少約 `60×60` 像素，第二條專門實驗路徑是 **HandRefiner**；它以 MeshGraphormer/MANO 產生手部 mesh 深度，再用 SD1.5 inpainting + ControlNet 重繪。這條路徑較符合「手指數量／形狀」問題，但需要舊依賴、MANO 授權及額外權重，且作者明確列出不可辨識、尺寸過大、小手與未測試 SDXL 移植等限制。

**LaMa** 僅排在背景／平滑衣料小污點分支：它不理解提示詞、手部結構或商品 Logo。**SDXL + ControlNet/IP-Adapter** 可作第二輪對照，但官方文件仍警告許多 SDXL ControlNet checkpoint 是實驗性的；現有 SDXL 已有文字／符號／錯誤紋理失敗紀錄，不應把加條件等同於解決手部。**FLUX.1 Fill dev** 與 **Qwen-Image-Edit-2511** 留作資源與授權允許時的後續研究，不是 16 GB 的第一方案：前者 gated 且 dev 權重為非商用／非生產授權，後者雖 Apache-2.0，但官方模型倉庫約 53.76 GiB，且目前 Diffusers 0.35.1 沒有其 `QwenImageEditPlusPipeline`。

## 本機基線與證據分級

| 事項 | 已知事實 | 證據狀態 |
|---|---|---|
| 現有整合介面 | `tryon/repair.py` 將 `before.png`、`mask.png`、`job.json` 交給修補 worker；worker 產生 `patch.png`／`metrics.json`，最後由硬遮罩貼回，遮罩外像素必須完全相等 | 已讀取本機程式；見 [`tryon/repair.py`](../tryon/repair.py)、[`scripts/inpaint.py`](../scripts/inpaint.py) |
| 現有 SDXL | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` revision `115134f363124c53c7d878647567d04daf26e41e`；設定為 crop 最長邊 1024、30 steps、CFG 7.5、CPU offload | 本機文件／程式及實測紀錄；見 [`docs/verification.md`](verification.md) |
| 現有 SDXL 實測 | 真實 GPU smoke 約 22–24 秒／次、PyTorch peak allocated 約 5.365 GiB（不含其他程序）；衣料色塊案例仍可能產生文字、符號或粉紅錯誤紋理 | 本機測量；不是跨照片品質 benchmark |
| Diffusers 版本 | `.venv-repair/bin/python` 讀到 `diffusers==0.35.1`；可 import `StableDiffusionControlNetInpaintPipeline`、`StableDiffusionXLControlNetInpaintPipeline`、`FluxFillPipeline`；不能 import `QwenImageEditPlusPipeline` | 本機 API smoke；未載入新權重、未推論 |
| 本研究結論 | 以公開 checkpoint、官方 invocation、檔案／API metadata 排名；所有 VRAM、延遲及品質預測均另標為「估計」或「未知」 | 研究判斷，需實驗驗證 |

本地先前研究已覆蓋通用遮罩流程、SAM/DWPose、HandCraft 與 HandRefiner；本文件補上固定 snapshot、實際 API/預設值、Qwen 2511 這類較新編輯模型，以及對「壞圖是否被條件再次保留」的反向檢驗。先前文件仍是背景，不把其單張示例當成泛化證據：[`docs/hand-correction-research.md`](hand-correction-research.md)。

## 排名：16 GB 本機實用性，不是畫質排行榜

| 排名 | 候選 | 16 GB 判斷 | 存取／授權 | 適合與主要風險 |
|---:|---|---|---|---|
| 1 | SD1.5 inpainting | **最適合作為第一基線**；fp16 主要模型檔約 2.0 GiB（含 safety 約 2.55 GiB，磁碟檔案加總，不是 VRAM） | 公開、未 gated；CreativeML OpenRAIL-M | 512 crop、速度／記憶體較友善；解析度與人物、文字能力較弱，不能保證手指 |
| 2 | SD1.5 inpainting + `control_v11p_sd15_inpaint` | **仍屬可行實驗級距**；ControlNet fp16 約 0.673 GiB，加上 base | 兩者公開、未 gated；OpenRAIL／CreativeML OpenRAIL-M 條款需一起審 | 控制圖只保留 mask 外 context；mask 漏選、低強度 latent 泄漏、錯誤上下文仍可能留下缺陷 |
| 3（手部） | HandRefiner（SD1.5 + mesh depth ControlNet） | **可能可行但未測量**；模型檔不大，前處理與舊依賴是主要成本 | MIT 程式；finetuned 權重 Google Drive；MANO 另有授權；公開 HR16 ControlNet 為 Apache-2.0 | 專門處理可辨識 malformed hand；需 `>=60×60` 手、MeshGraphormer、MANO、MediaPipe，且不修手掌尺寸 |
| 4（背景） | LaMa `big-lama` | **最容易在 CPU／GPU 隔離測**；HF archive 約 0.355 GiB | 原始程式 Apache-2.0；HF `smartywu/big-lama` 是公開鏡像 | 適合物件移除／平滑紋理；沒有文字條件或解剖結構，不能當補手模型 |
| 5 | SDXL inpaint + SDXL ControlNet + IP-Adapter | **硬體可能可行但不先做**；現有 SDXL 已實測失敗，額外條件增加整合面 | SDXL OpenRAIL++、depth ControlNet OpenRAIL++、IP-Adapter Apache-2.0 | 深度／參考圖可限制結構或外觀；不代表手指正確，也可能把錯誤衣料／身份帶入 |
| 6 | Qwen-Image-Edit-2511 | **原生 16 GB 不實際**；官方倉庫約 53.756 GiB，沒有本地 16 GB 證據 | 公開、未 gated、Apache-2.0 | 最新開放編輯模型之一，官方宣稱一致性改善；當前 Diffusers 不相容且官方範例沒有 mask 參數，非局部貼片 drop-in |
| 7 | FLUX.1 Fill dev | **不作第一方案**；完整 HF 倉庫約 54.069 GiB，transformer 約 22.17 GiB | gated `auto`；FLUX.1 [dev] Non-Commercial License，非商用／非生產 | 官方 Fill pipeline 具備 in/outpaint；量化／offload 的 16 GB 品質與耗時未測，且需確認非商用條款 |
| — | HandCraft | 不是生成器；只作 hand detection／template control preprocessor | Space metadata 只有 `cc`，沒有具體變體；`yolo.pt` 是公開 pickle 權重 | 可輸出 malformed/standard bounding boxes、opened-palm/fist-back 控制圖；官方 UI 也要求另接 SD/ControlNet，不是完整修補器 |

「16 GB 可行」只表示值得放入隔離環境做峰值測量；模型檔案大小不能換算成推論 VRAM，且沒有任何候選在本研究中取得跨人物／商品的成功率。

## 1. SD1.5 inpainting：首輪基線

### 固定權重與官方證據

官方 Diffusers 文件把 `stable-diffusion-v1-5/stable-diffusion-inpainting` 列為 512×512 inpainting checkpoint，示例使用 `AutoPipelineForInpainting`、fp16、`enable_model_cpu_offload()`，並以 PIL `image`／`mask_image` 呼叫。官方 model card 說明它是在 512×512 做 inpainting 微調，UNet 增加 5 個 mask／masked-image 輸入通道。

- [Diffusers Inpainting guide（SD1.5、mask、padding crop、overlay）](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint#stable-diffusion-inpainting)
- [官方 model card，固定 revision `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/tree/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb)
- [fp16 UNet safetensors（固定 revision）](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/blob/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb/unet/diffusion_pytorch_model.fp16.safetensors)
- [model metadata／檔案 tree API](https://huggingface.co/api/models/stable-diffusion-v1-5%2Fstable-diffusion-inpainting/tree/main?recursive=true&expand=1)

2026-09-14 讀取的 API 顯示 `gated=false`、`private=false`。選定的 fp16 檔案大小約為：UNet 1.601 GiB、text encoder 0.229 GiB、VAE 0.156 GiB、safety checker 0.566 GiB。這是 Hugging Face 檔案的磁碟大小加總，不是 VRAM 預測；未下載檔案，也未測試載入峰值。

授權是 model card 所列的 [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)。model card 也明列不能產生清晰可讀文字、人物／臉部可能不正確及資料集／安全限制；因此它不能被描述成能解決目前 SDXL 的文字、手部或商品 Logo 失敗。

### Diffusers 0.35.1 的實際呼叫

以下是後續隔離實驗的最小 invocation；它是根據官方文件及本機已存在的 0.35.1 API 整理，**本次沒有執行**：

```python
import torch
from diffusers import AutoPipelineForInpainting

pipe = AutoPipelineForInpainting.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    revision="8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.enable_model_cpu_offload()

result = pipe(
    prompt="a natural blue fabric cuff matching the surrounding sleeve",
    negative_prompt="text, logo, watermark, symbol, extra fingers, deformed hand",
    image=crop_512,
    mask_image=mask_512,
    strength=0.65,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
```

本機 0.35.1 的 `StableDiffusionInpaintPipeline.__call__` 預設值為 `strength=1.0`、`num_inference_steps=50`、`guidance_scale=7.5`，並支援 `padding_mask_crop`。官方文件的 `padding_mask_crop=32` 範例可作對照；本專案目前已在上游 `crop_box` 提供 64 像素 context，首輪應避免無意間做兩次 crop。

官方文件對參數的描述是：較高 `strength` 增加噪聲與差異，較低值較像原圖但品質可能下降；這不是「越高越能修手」的證據。建議首輪固定 crop 與 prompt，只掃 `strength={0.45,0.65,0.85}`、`steps={20,30}`、`seed={42,43}`，並以 512 的倍數輸入，隔離解析度因素。

## 2. SD1.5 + `control_v11p_sd15_inpaint`：便宜但可反駁的結構基線

### 精確 snapshot、可取得性與 licence

官方 checkpoint 是 `lllyasviel/control_v11p_sd15_inpaint`，固定 revision `c96e03a807e64135568ba8aecb66b3a306ec73bd`。2026-09-14 API 顯示 `gated=false`、`private=false`；固定的 `diffusion_pytorch_model.fp16.safetensors` HTTP HEAD 為 200，大小 722,598,642 bytes（約 0.673 GiB）。model card 的 config 是 SD1.5 ControlNet（`cross_attention_dim=768`），README 說明它是 inpaint condition 的 Diffusers conversion。

- [官方 ControlNet inpaint model card／固定 revision](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/tree/c96e03a807e64135568ba8aecb66b3a306ec73bd)
- [fp16 ControlNet safetensors（固定 revision）](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/blob/c96e03a807e64135568ba8aecb66b3a306ec73bd/diffusion_pytorch_model.fp16.safetensors)
- [官方 README 的 inpaint invocation](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/blob/c96e03a807e64135568ba8aecb66b3a306ec73bd/README.md)
- [model metadata／檔案 tree API](https://huggingface.co/api/models/lllyasviel%2Fcontrol_v11p_sd15_inpaint/tree/main?recursive=true&expand=1)

model card metadata 標示 `openrail`，正文連到 [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)。實際部署／再散布仍須同時審 base model 與 ControlNet 條款，不把「未 gated」當成「沒有使用限制」。

### 官方條件圖的關鍵語義

Diffusers 官方 inpaint guide 的 `make_inpaint_condition` 不是把壞圖原樣複製給 ControlNet，而是：

1. RGB 轉成 `[0,1]`；
2. 對 `mask > 0.5` 的像素設定為 `-1.0`；
3. 轉成 `NCHW` tensor，作為 `control_image`；
4. 同一輪仍把原始 `init_image` 與 `mask_image` 傳給 `StableDiffusionControlNetInpaintPipeline`。

官方範例明確使用 `lllyasviel/control_v11p_sd15_inpaint`，並以 `control_image=make_inpaint_condition(init_image, mask_image)` 呼叫。其意義是讓控制分支看到未遮罩上下文，而不是提供手指的真值。

- [官方 Diffusers ControlNet inpaint 例程](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint#controlnet)
- [官方 Diffusers source（相同例程，便於固定內容）](https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/inpaint.md#controlnet)

### Diffusers 0.35.1 實際呼叫與預設值

```python
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    assert image.shape[:2] == image_mask.shape[:2]
    image[image_mask > 0.5] = -1.0
    return torch.from_numpy(np.expand_dims(image, 0).transpose(0, 3, 1, 2))

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint",
    revision="c96e03a807e64135568ba8aecb66b3a306ec73bd",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    revision="8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.enable_model_cpu_offload()

result = pipe(
    prompt="a natural blue fabric cuff matching the surrounding sleeve",
    negative_prompt="text, logo, watermark, symbol, extra fingers, deformed hand",
    image=crop_512,
    mask_image=mask_512,
    control_image=make_inpaint_condition(crop_512, mask_512),
    strength=0.65,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.5,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
```

本機 0.35.1 的 `StableDiffusionControlNetInpaintPipeline.__call__` 預設為 `strength=1.0`、`num_inference_steps=50`、`guidance_scale=7.5`、`controlnet_conditioning_scale=0.5`。官方 model card 範例使用 20 steps、`eta=1.0`、CPU generator；本專案首輪先固定 `eta=0` 的既有 Diffusers 預設，再另測 `eta=1.0`，不要把不同 scheduler／eta 的結果混在一起。

### 反向問題：同一張壞圖是否會把缺陷保留下來？

答案不能以模型載入推定，必須做 A/B 實驗；但從官方資料可先得到可反駁的預測：

- **遮罩完整且強度足夠時**：ControlNet hint 的 mask 內像素是 `-1`，所以控制分支沒有直接看到壞手指／色塊；它較可能保留 mask 外的輪廓與背景。
- **主 inpainting 分支仍收到壞 `image`**：低 `strength` 仍可能讓原 latent 影響重繪，不能宣稱缺陷必定消失。
- **遮罩漏掉異常長指、衣料殘影或手腕接縫時**：hint 與最終硬合成都會把漏選區當作不可改區；缺陷會直接存活，或產生不自然邊界。
- **壞圖在 mask 外本來就錯**：ControlNet 的工作是遵循 visual condition，不是辨認該 condition 是否錯；它可能更穩定地保留錯誤。
- **ControlNet inpaint 不是 HandRefiner**：它沒有手部 mesh、五指拓撲或姿勢／遮擋語義；`bad anatomy` 只是在文字條件中，不是硬約束。

因此這條基線的實驗問題應寫成「完整遮罩下是否提高局部修補的上下文／材質一致性」，而不是「是否修好手」。每張結果仍必須先經 `composite_patch` 的遮罩外逐像素檢查，再由人工判斷遮罩內是否有新錯誤。

## 3. SDXL conditioning：ControlNet、IP-Adapter 與現有失敗的對照

### SDXL inpaint 本身不是未知的解法

目前使用的 `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` 固定 revision 是 `115134f363124c53c7d878647567d04daf26e41e`，公開未 gated，model card 授權為 [CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)。官方 card 說它在 1024×1024 訓練，示例為 `guidance_scale=8.0`、20 steps、`strength=0.99`，並明確警告 `strength=1` 會降低品質、人物／臉部可能不正確、不能產生可讀文字。這些限制和本機在衣料案例觀察到的文字／符號／錯誤紋理一致，但不能由此推論 SD1.5 或 ControlNet 一定較好。

- [SDXL inpainting model card／固定 revision](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tree/115134f363124c53c7d878647567d04daf26e41e)
- [SDXL fp16 UNet safetensors](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/blob/115134f363124c53c7d878647567d04daf26e41e/unet/diffusion_pytorch_model.fp16.safetensors)

### SDXL ControlNet

Diffusers 有正式的 `StableDiffusionXLControlNetInpaintPipeline` API；目前可讀取的 SDXL depth ControlNet 是 `diffusers/controlnet-depth-sdxl-1.0-small`，revision `daf3835d036574dff7c158882e8e77b75b024ee5`，公開未 gated，model card metadata 為 OpenRAIL++。其 fp16 ControlNet 檔案約 0.298 GiB。官方 SDXL ControlNet API 頁面同時寫明「許多 SDXL ControlNet checkpoint 是 experimental」，因此它是可測的條件分支，不是可靠度保證。

- [SDXL ControlNet API（含 Inpaint pipeline）](https://huggingface.co/docs/diffusers/en/api/pipelines/controlnet_sdxl#stablediffusionxlcontrolnetinpaintpipeline)
- [SDXL depth ControlNet model card／固定 revision](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-small/tree/daf3835d036574dff7c158882e8e77b75b024ee5)
- [官方 Diffusers ControlNet guide](https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet)

可測 invocation（不代表已驗證成功）：

```python
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0-small",
    revision="daf3835d036574dff7c158882e8e77b75b024ee5",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    revision="115134f363124c53c7d878647567d04daf26e41e",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.enable_model_cpu_offload()
result = pipe(
    prompt=prompt,
    image=crop_1024,
    mask_image=mask_1024,
    control_image=depth_1024,
    strength=0.85,
    num_inference_steps=30,
    guidance_scale=5.0,
    controlnet_conditioning_scale=0.5,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
```

本機 0.35.1 的 SDXL ControlNet inpaint 呼叫預設為 `strength=0.9999`、50 steps、CFG 5.0、ControlNet scale 1.0；這和普通 SDXL inpaint 的 7.5／0.9999 不同，實驗記錄時不可只記 model name。Depth 是由壞圖估出的幾何條件時，異常本身可能進入 depth；HandRefiner 的 mesh depth 不應與普通 depth ControlNet 混稱。

### IP-Adapter／reference guidance

官方 Diffusers 文件把 IP-Adapter 定義為加在既有 diffusion model cross-attention 的影像條件 adapter；它不是像素貼回器。文件說 adapter 檔通常約 100 MB、需要先載入 base model，再用 `load_ip_adapter`；`set_ip_adapter_scale(0.5)` 通常是文字／影像條件的平衡，`1.0` 偏向影像條件。實際倉庫有不同 encoder／adapter，不能混算：本輪選 Plus ViT-H safetensors 與 `models/image_encoder/model.safetensors`，後者為 2,528,373,448 bytes（約 2.35 GiB），不是 `sdxl_models/image_encoder`。這是磁碟大小，非 VRAM 實測；詳見獨立審查 P1-1 的固定來源。

- [官方 IP-Adapter Diffusers guide](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter)
- [IP-Adapter model repository／固定 revision `018e402774aeeddd60609b4ecdb7e298259dc729`](https://huggingface.co/h94/IP-Adapter/tree/018e402774aeeddd60609b4ecdb7e298259dc729)
- [IP-Adapter Apache-2.0 LICENSE](https://github.com/tencent-ailab/IP-Adapter/blob/main/LICENSE)

最小 SDXL reference 呼叫：

```python
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    revision="018e402774aeeddd60609b4ecdb7e298259dc729",
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
    image_encoder_folder="models/image_encoder",
)
pipe.set_ip_adapter_scale(0.5)
result = pipe(
    prompt=prompt,
    image=crop_1024,
    mask_image=mask_1024,
    ip_adapter_image=reference_image,
    strength=0.65,
    num_inference_steps=30,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
```

若要只讓 reference 影響局部，官方提供 `IPAdapterMaskProcessor` 與 `cross_attention_kwargs={"ip_adapter_masks": masks}`；其文件範例主要放在 image-to-image／多 reference 流程，並不是本專案已驗證的 SDXL inpaint 貼片流程。應先在單獨原型確認版本、mask 對齊及影像 encoder offload 順序；官方提醒 IP-Adapter image encoder 必須先載入，之後再 `enable_model_cpu_offload()`。

IP-Adapter 的反向風險是 reference 本身若含錯手、錯衣料或不相干背景，adapter 可能把錯誤的外觀條件帶入；它不能保證人物身份、Logo 或手指拓撲。它值得用來測「商品 crop 是否改善材質／顏色」，不值得當成泛用 anatomy fix。

## 4. FLUX.1 Fill dev：能力強但不適合先塞進 16 GB

Black Forest Labs 官方 repo 把 FLUX.1 Fill [dev] 定義為 in/out-painting，Diffusers 文件的 `FluxFillPipeline` 不要求普通 inpainting 的 `strength`；官方 sample 用 `height=1632`、`width=1232`、`max_sequence_length=512`、seed 0，並以 BF16 直接 `.to("cuda")`。BFL 官方 Fill CLI 也明確需要輸入圖片和黑白 mask：

```bash
python -m flux fill \
  --img_cond_path <path_to_input_image> \
  --img_mask_path <path_to_input_mask>
```

```python
import torch
from diffusers import FluxFillPipeline

pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    revision="358293da0354175698b67ec8299acf928313a78a",
    torch_dtype=torch.bfloat16,
).to("cuda")
result = pipe(
    prompt="a white paper cup",
    image=image,
    mask_image=mask,
    height=1632,
    width=1232,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
```

- [BFL 官方 FLUX repo](https://github.com/black-forest-labs/flux)
- [BFL 官方 Fill usage（含 SHA-256、CLI）](https://github.com/black-forest-labs/flux/blob/main/docs/fill.md)
- [Diffusers Flux pipeline（Fill invocation、量化／資源警告）](https://huggingface.co/docs/diffusers/en/api/pipelines/flux#fill-inpaintingoutpainting)
- [FLUX.1 Fill model card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
- [HF model metadata／固定 revision](https://huggingface.co/api/models/black-forest-labs%2FFLUX.1-Fill-dev/tree/main?recursive=true&expand=1)

2026-09-14 HF API 顯示此 model `gated="auto"`、`license_name=flux-1-dev-non-commercial-license`、revision `358293da0354175698b67ec8299acf928313a78a`。倉庫檔案加總約 54.069 GiB，其中 transformer 約 22.170 GiB、text encoders 約 9.100 GiB、AE 約 0.625 GiB。BFL 官方 dev license 允許的模型使用是 non-commercial／non-production；商用需另向 BFL 取得 license，並有 content filter／review、輸出 AI disclosure 及 biometric／surveillance 等限制，不能只看「open weights」。

- [BFL dev non-commercial license（含 Fill、Kontext 在適用模型範圍）](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev)
- [BFL 官方 open-weight 表格與各模型授權](https://github.com/black-forest-labs/flux/blob/main/README.md#open-weight-models)

Diffusers 官方只說 Flux 在 consumer hardware 上昂貴，可用 CPU offload／量化降低記憶體但會犧牲延遲；沒有給 Fill 在 4060 Ti 16 GB 的峰值或品質保證。`FluxFillPipeline` 類別雖然存在於本機 0.35.1，但本次沒有取得 gated 權限、下載或執行；因此不把任何 16 GB「可能量化成功」當成已驗證方案。

## 5. Qwen-Image-Edit-2511：最新開放編輯候選，但不是局部貼片 drop-in

Qwen 官方在 2026-09-14 的主 README 已把 `Qwen-Image-Edit-2511` 列為最新 image-edit 權重，model card 宣稱比 2509 改善 image drift、人物一致性、工業設計與幾何推理。官方 quick start 使用 `QwenImageEditPlusPipeline`、BF16、`num_inference_steps=40`、`true_cfg_scale=4.0`、`guidance_scale=1.0`、空白 negative prompt，並可輸入 1–3 張圖片。

- [Qwen 官方 model card／2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Qwen 官方 GitHub README（當前版本與 quick start）](https://github.com/QwenLM/Qwen-Image/blob/main/README.md)
- [Qwen 2511 固定 revision `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/tree/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9)
- [HF model metadata／檔案 tree API](https://huggingface.co/api/models/Qwen%2FQwen-Image-Edit-2511/tree/main?recursive=true&expand=1)

API 顯示 `gated=false`、`private=false`、Apache-2.0。取回的倉庫檔案約 53.756 GiB：transformer 約 38.055 GiB、text encoder 約 15.445 GiB、VAE 約 0.236 GiB。官方範例要求安裝最新 Git Diffusers 並將 pipeline 直接放到 CUDA；沒有官方 16 GB offload／量化配方。當前本專案 Diffusers 0.35.1 實際沒有 `QwenImageEditPlusPipeline`，所以即使有磁碟與 VRAM，也要另建相容環境。

更重要的是，官方 2511 quick start 的輸入是整張圖片／多張圖片與文字 prompt，沒有本專案要求的 `mask_image` 局部 inpaint 參數。若硬接，還需另寫局部編輯／硬合成邏輯，並驗證它是否在遮罩外改動人物／衣服；這不是只替換 `MODEL_ID` 就能完成的修補器。Qwen 的一致性展示是官方 showcase，不是 FASHN VTON 失敗照片的交叉測試，故列為後續候選而非首輪。

## 6. LaMa：便宜、確定邊界，但只做紋理／移除

LaMa 官方 repo（WACV 2022）描述它是在 256×256 訓練、對更高解析度（約 2k）及週期性結構有泛化能力的 Fourier-convolution inpainting。官方最小推論是 `bin/predict.py`；`refine=True` 是額外 refinement，而非語義理解：

```bash
git clone https://github.com/advimman/lama.git
cd lama
export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)

# 公開 HF 鏡像；這一步只列為後續實驗，現在不下載
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip

python3 bin/predict.py \
  model.path=$(pwd)/big-lama \
  indir=$(pwd)/LaMa_test_images \
  outdir=$(pwd)/output

# 可選 refinement
python3 bin/predict.py refine=True \
  model.path=$(pwd)/big-lama \
  indir=$(pwd)/LaMa_test_images \
  outdir=$(pwd)/output
```

- [LaMa 官方 README／下載、mask 格式、predict command](https://github.com/advimman/lama/blob/main/README.md)
- [LaMa 官方 Apache-2.0 license](https://github.com/advimman/lama/blob/main/LICENSE)
- [公開 `smartywu/big-lama` 鏡像／固定 revision `05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9`](https://huggingface.co/smartywu/big-lama/tree/05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9)
- [鏡像檔案 tree API（`big-lama.zip` 約 0.355 GiB）](https://huggingface.co/api/models/smartywu%2Fbig-lama/tree/main?recursive=true&expand=1)

原始 repo 的 requirements 仍包含 Torch 1.8、pytorch-lightning 1.2.9 等舊版本；不能把它直接安裝進目前 Torch 2.8 的 repair env。可在獨立環境、CPU 或隔離 worker 驗證，輸入／輸出保持本機。LaMa 對背景殘影、平滑布料小污點有合理工程位置，但對「重建手指」「保持原商品 Logo」「依 prompt 生成袖口」沒有條件或保證；把它接在手部生成器後只會讓失敗更難診斷。

## 7. HandRefiner：真正的手部專家候選，但依賴與前提不可省略

HandRefiner 官方方法是：手部偵測 → MeshGraphormer/MANO mesh → 深度與 mask → SD1.5 inpainting + ControlNet。作者的實作與 FAQ 給出以下可操作限制：

- 可辨識的 malformed hand 才適合；若肉眼已無法辨識，mesh fitting 可能失敗。
- 方法不修正原本過大的手掌尺寸。
- 建議目前 SD1.5 權重的手至少約 `60×60` 像素；小手應先另行放大，且放大不是新增真實資訊。
- finetuned ControlNet 建議 control strength 約 `0.4–0.8`，作者 evaluation 使用 `0.55`；原始 depth ControlNet 可能失敗率更高。
- SDXL 圖片在論文實驗先縮到 512，因為 base model 是 SD1.5；作者明確說 SDXL 移植未測試。

- [HandRefiner 官方 README／FAQ／single-image command](https://github.com/wenquanlu/HandRefiner/blob/main/README.md)
- [HandRefiner 官方安裝／權重來源](https://github.com/wenquanlu/HandRefiner/blob/main/docs/installation.md)
- [HandRefiner 官方手冊／參數與 defaults](https://github.com/wenquanlu/HandRefiner/blob/main/docs/manual.md)
- [HandRefiner 官方程式 MIT license](https://github.com/wenquanlu/HandRefiner/blob/main/LICENSE)
- [HandRefiner 官方 code 固定 HEAD（本次取回）](https://github.com/wenquanlu/HandRefiner/commit/f07e196298bafe871064b2952587548e28ccd467)
- [公開 pruned fp16 ControlNet `hr16/ControlNet-HandRefiner-pruned`](https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/tree/f0917f0595ecb7f6435f49e4b2b28f8dd68ab0cb)
- [HR16 model card／Apache-2.0 metadata](https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/blob/f0917f0595ecb7f6435f49e4b2b28f8dd68ab0cb/README.md)
- [MANO 官方 license](https://mano.is.tue.mpg.de/license.html)

### 官方 command 與資源

```bash
python handrefiner.py \
  --input_img test/1.jpg \
  --out_dir output \
  --strength 0.55 \
  --weights models/inpaint_depth_control.ckpt \
  --prompt "a man facing the camera, making a hand gesture, indoor" \
  --seed 1
```

程式 defaults 包含 `strength=1.0`、`finetuned=True`、`num_samples=1`、`n_iter=1`、`adaptive_control=False`、`padding_bbox=30`、`seed=-1`；實驗不應把 default strength 1.0 與作者建議 0.4–0.8 混為一談。finetuned `inpaint_depth_control.ckpt` 在官方安裝文件連到 Google Drive，並不是一個已核實可固定 HF snapshot 的官方權重；非 finetuned 路徑另需 SD1.5 inpainting 與 `control_v11f1p_sd15_depth`。

HR16 公開 pruned ControlNet 的 fp16 safetensors 約 0.673 GiB；其 model card 標示 Apache-2.0，但整條方法還有 SD1.5 base、Graphormer／HRNet、MediaPipe task、MANO_RIGHT.pkl 與各自授權。官方 requirements 固定 Torch 2.0.0、NumPy 1.23.5、Transformers 4.27.4 等舊組合；不可把頂層 MIT 當成所有模型／資料的授權，也不可未審查就覆蓋目前 repair env。

資源判斷是**估計**：SD1.5 fp16 base 加 ControlNet、Graphormer 與 HRNet 的檔案量比 SDXL／Flux 小，16 GB 很值得做一次隔離峰值測量；但 MeshGraphormer 的前處理、CUDA context、batch 與 pipeline 同時共存可能增加顯存，沒有本機測量前不能承諾 fit。

## 8. HandCraft：可用的控制圖／候選區工具，不是修補模型

HandCraft（WACV 2025）的官方 Space 程式可核對出：

1. `component/bbox.py` 用 `model/yolo.pt` 偵測 standard／malformed hand；
2. `component/skeleton.py` 產生 body／hand keypoints；
3. `component/control.py` 以 `opened-palm` 或 `fist-back` template 做 affine 對齊；
4. UI 提供 `include_undetected`，但最後輸出是 control image、control mask、union mask，並提示使用者另接 Stable Diffusion／ControlNet。

- [HandCraft 官方研究入口](https://github.com/kfzyqin/handcraft)
- [WACV 2025 官方論文頁](https://openaccess.thecvf.com/content/WACV2025/html/Qin_HandCraft_Anatomically_Correct_Restoration_of_Malformed_Hands_in_Diffusion_Generated_WACV_2025_paper.html)
- [官方 Hugging Face Space](https://huggingface.co/spaces/zhenyueqin/handcraft)
- [Space app.py（流程與「另接 SD/ControlNet」說明）](https://huggingface.co/spaces/zhenyueqin/handcraft/blob/main/app.py)
- [Space bbox.py](https://huggingface.co/spaces/zhenyueqin/handcraft/blob/main/component/bbox.py)
- [Space control.py](https://huggingface.co/spaces/zhenyueqin/handcraft/blob/main/component/control.py)
- [Space model tree（`yolo.pt` 136,682,110 bytes）](https://huggingface.co/spaces/zhenyueqin/handcraft/tree/main/model)
- [Space requirements](https://huggingface.co/spaces/zhenyueqin/handcraft/blob/main/requirements.txt)

Space metadata 的 `license: cc` 沒有給具體 CC 變體，不可推定商用或再散布許可；`yolo.pt` 是 pickle 檔，Hugging Face 的安全 metadata 也標示需要注意 pickle imports。若未來採用，只應在隔離環境先檢查來源／雜湊，再把它當人工遮罩建議器，不當自動通過判定或完整生成器。

## 精確整合實驗：先比較 SD1.5，不改 production path

以下是下一輪實驗的可重現邊界，不是本次已執行的實作。

### A. 固定資料與介面

1. 沿用有使用權的 60 組人物／商品配對：40 組開發、20 組 holdout；人物／商品不得跨 split 洩漏。至少覆蓋衣料小污點、Logo／文字、袖口／手腕、可辨識手部（含握拳、持物、長短袖）、複雜背景及原本正確的 clean cases。
2. 先固定現有 FASHN 候選，不在每個 repair model 重新生成試穿圖；同一張 `before.png`、同一張人工完整 mask、同一個 crop box 做 A/B。遮罩要包含異常像素與必要接縫，不可只圈「看起來像手指」的中心。
3. 對每候選均經同一 `composite_patch`／硬遮罩貼回；`outside_mask_max_difference` 必須是 0。保留原圖、候選 patch、mask、options、checkpoint revision、版本與 failure reason。
4. 只在本機讀檔；不把人物照送到 Hugging Face、Space、Google Drive 或任何第三方推論服務。模型先以固定 revision 下載至隔離環境；本次不執行下載。

### B. 候選與參數矩陣

| 分支 | crop／條件 | 首輪參數（工程起點，非官方最佳值） | 停止／注意 |
|---|---|---|---|
| 現有 SDXL baseline | 現有 64 px context，最長邊 1024 | steps 30、CFG 7.5、strength `{0.75,0.90,0.99}`、seed `{42,43}` | 已知可能產生文字／符號／錯紋理；不把同一 model 的多 seed 當新模型證據 |
| SD1.5 inpaint | 同一 crop，縮至最長邊 512、寬高取 8 倍數 | steps `{20,30}`、CFG 7.5、strength `{0.45,0.65,0.85}`、seed `{42,43}` | 記錄 512 解析度造成的布料細節損失；不與 SDXL 跨解析度直接比單張美觀 |
| SD1.5 + inpaint ControlNet | 同一 512 crop；`make_inpaint_condition` 依完整 mask 置 `-1` | 上列 SD1.5 矩陣 + ControlNet scale `{0.5,0.8}`；另做「mask 漏掉殘影」對照 | 直接回答壞圖是否由 hint／主 latent 留下；不得只測 mask 完整的成功例 |
| HandRefiner | 手部 crop／全圖座標一致；方法本身 512 | `--strength {0.4,0.55,0.8}`、50 DDIM steps、CFG 9、padding_bbox 30、seed `{42,43}` | 只在原手 `>=60×60` 且 mesh 可辨識時跑；記錄 mesh／mask，不把失敗視為拒絕品質 |
| LaMa | 同一 mask，先限定背景／平滑布料子集 | `refine=False`；必要時另跑 `refine=True` | 不跑「補手」結論；以 CPU 延遲與紋理接縫做對照 |
| SDXL ControlNet/IP-Adapter | 第二輪才做，固定 SDXL 1024 | depth ControlNet scale `{0.3,0.5,0.8}`；IP scale `{0.3,0.5,0.8}`，reference 僅用可靠商品／人物 crop | 先驗證版本與 mask 對齊；不將 reference consistency showcase 當 FASHN 證據 |
| Flux Fill／Qwen 2511 | 非首輪；只有 access、磁碟、VRAM 與授權均明確才做 | 依官方 sample：Fill 50 steps／max seq 512；Qwen 40 steps／true CFG 4 | 超過 15 GiB 可用顯存或 5 分鐘即記錄 resource failure；不能以 CPU fallback 改寫本機要求 |

### C. 每張圖的量測與接受規則

- 量測 cold load、warm inference、總 wall time、`torch.cuda.max_memory_allocated()`、`max_memory_reserved()`，再以 `nvidia-smi` 記錄整卡峰值；模型檔案大小與 PyTorch allocator 不得互相冒充。
- 兩位評閱者盲看修前／修後局部與全圖，分別標註：目標缺陷真的消失、是否新增缺陷、手部結構／遮擋、人物身份、商品材質／Logo、背景、接縫。握拳不套「一定有五根可見手指」規則。
- 先報每個 defect class 的分子／分母：缺陷修復率、clean case 誤修率、重大新缺陷率、需人工處理率；不只報已被程式接受的圖片。
- 候選只有在「目標缺陷修好、沒有重大新缺陷、遮罩外差異為 0、兩人均可接受」時算成功。單張漂亮、模型載入成功、negative prompt 沒報錯都不算成功。
- SD1.5 + ControlNet 只有在 holdout 相對現有 SDXL baseline 改善，且 clean cases／Logo／身份沒有明顯回退時，才值得接入 UI；否則保留為研究分支。

## 獨立逆向審查（實作前閘門）

這一節刻意先攻擊最容易過度解讀的結論；它不是品質通過證明。

1. **「ControlNet 不看 mask 內，所以一定不會保留錯誤」是錯的。** 主 inpaint image 仍是壞圖；低 strength、VAE 壓縮及 mask 邊界都可能把錯誤帶入。必須測同一壞圖的完整 mask、少一圈 mask、只圈手中心三種版本。
2. **「512 較省顯存，所以一定比較好」是錯的。** SD1.5 能力／解析度較小，衣料高頻紋理與細手指可能反而損失；比較必須分開報 512 與 1024 的 artefact。
3. **「ControlNet depth／IP-Adapter 是結構保證」是錯的。** depth 可能從 malformed hand 估出 malformed geometry；IP-Adapter 是外觀 cross-attention，不是像素複製或 identity verifier。參考圖錯了會把錯誤條件帶進來。
4. **「HandRefiner 是手部萬靈丹」是錯的。** 作者的 mesh 前提排除了不可辨識手，且不修手掌尺寸；長指超出 mask、袖口遮擋、握拳與持物必須個別驗收。其 SDXL 移植 FAQ 是未測試聲明，不可拿來支撐 SDXL 修正。
5. **「LaMa 很小所以可補手」是錯的。** LaMa 沒有 prompt／hand mesh／pose condition；它只應和背景／週期布料任務比較。
6. **「FLUX／Qwen 較新所以應直接替換」是錯的。** FLUX dev 有 gating 與非商用／非生產條款；Qwen 雖 Apache，當前 API／模型規模與 16 GB 不合。先解決 access、license、版本、VRAM，再談 FASHN 品質。
7. **「遮罩外像素相同所以結果安全」是錯的。** 這只保證局部性，不保證遮罩內的手、Logo、衣料或接縫正確；硬邊界還可能可見。UI 必須維持 `needs_review` 與可撤回。
8. **「模型卡的 FID／showcase 等同 VTON repair benchmark」是錯的。** 本研究沒有找到跨人物／商品、同一 FASHN 候選、相同 mask 的公開交叉證據；所有品質結論都必須由本機 holdout 人工驗收取得。
9. **「Apache 專案所以整條 pipeline 可商用」是錯的。** Qwen、IP-Adapter、LaMa、HR16 各有 Apache metadata，但 SD1.5／SDXL／ControlNet 是 OpenRAIL 系列，HandRefiner 還有 MANO／MeshGraphormer，HandCraft Space 的 `cc` 不具體；逐個模型／資產記錄 license 與 gate。
10. **「固定 model ID 足夠重現」是錯的。** 首輪必須寫入 revision、套件版本、dtype、scheduler、steps、CFG、strength、ControlNet/IP scale、seed、crop box 與 mask digest；禁止以 mutable `main` 取代上述固定 snapshot。

## 建議決策

1. **先做 SD1.5 inpaint 與 SD1.5 + `control_v11p_sd15_inpaint` 的離線 A/B。** 這是最小且真正不同於現有 SDXL 的實驗；基線共用相同 FASHN 候選、mask、seed 與硬合成。
2. **手部品質仍差時，才做 HandRefiner 專家分支。** 先在 `>=60×60`、可辨識手的小子集測 mesh／mask／control strength；不要把舊 requirements 或 MANO 資產混入現有 repair env。
3. **背景／平滑衣料另走 LaMa／OpenCV 分支。** 它們的低成本優勢是真實的，但不要用來回答解剖重建。
4. **SDXL ControlNet/IP-Adapter 放第二輪。** 若 SD1.5 解析度造成材質退化，才以固定 SDXL ControlNet 與可靠 reference image 比較；先驗證 official API 版本與 offload 順序。
5. **暫不採用 FLUX Fill／Qwen 2511 作 16 GB 預設。** FLUX 的 gate／dev license 與權重規模、Qwen 的權重規模／Diffusers API 都是實際阻礙，不應以「未來可能量化」取代當下可執行基線。

這個推薦回答的是「下一個可量測、可回退、可保留遮罩外身份的本機實驗」，不是「哪個模型一定修好手」。若 holdout 沒有改善，正確結果是保留人工修補／現有候選，而不是放寬 geometry gate 或無限換 seed。

## 來源索引（均於 2026-09-14 取回）

### 本機與 Diffusers

- [本專案 verification](verification.md)
- [既有 hand-correction research](hand-correction-research.md)
- [本機 repair requirements](../requirements-repair.txt)
- [Diffusers Inpainting guide](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint)
- [Diffusers ControlNet guide](https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet)
- [Diffusers SDXL ControlNet API](https://huggingface.co/docs/diffusers/en/api/pipelines/controlnet_sdxl)
- [Diffusers IP-Adapter guide](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter)
- [Diffusers Flux pipeline API](https://huggingface.co/docs/diffusers/en/api/pipelines/flux)

### Checkpoint／模型官方來源

- [SD1.5 inpainting model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)
- [SD1.5 inpainting metadata API](https://huggingface.co/api/models/stable-diffusion-v1-5%2Fstable-diffusion-inpainting)
- [SD15 ControlNet inpaint model card](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint)
- [SDXL inpainting model card](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
- [SDXL depth ControlNet model card](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-small)
- [IP-Adapter model card](https://huggingface.co/h94/IP-Adapter)
- [FLUX.1 Fill model card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
- [Qwen-Image-Edit-2511 model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [LaMa official repo](https://github.com/advimman/lama)
- [big-lama public HF mirror](https://huggingface.co/smartywu/big-lama)
- [HandRefiner official repo](https://github.com/wenquanlu/HandRefiner)
- [HandRefiner HR16 ControlNet](https://huggingface.co/hr16/ControlNet-HandRefiner-pruned)
- [HandCraft official Space](https://huggingface.co/spaces/zhenyueqin/handcraft)

### 授權／使用限制

- [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
- [CreativeML Open RAIL++ base license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
- [BFL FLUX.1 dev non-commercial license](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev)
- [LaMa Apache-2.0](https://github.com/advimman/lama/blob/main/LICENSE)
- [HandRefiner MIT](https://github.com/wenquanlu/HandRefiner/blob/main/LICENSE)
- [MANO official license](https://mano.is.tue.mpg.de/license.html)
