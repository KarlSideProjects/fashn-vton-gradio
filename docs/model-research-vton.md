# Base virtual try-on 模型研究與可行改進

> 研究日期：2026-09-14（Asia/Taipei）
> 範圍：只研究 base virtual try-on（VTON）生成與有用的輸入轉換；通用 inpainting、手部修復與評測由其他工作處理。
> 研究限制：本次只讀取程式碼與公開一手資料，沒有安裝套件、下載模型、啟動 GPU、上傳圖片、提交或推送。Firecrawl MCP 先行查詢但在本次研究中只回傳空資料，因此改用可直接驗證的官方 GitHub、Hugging Face 與 arXiv 頁面；以下網頁引用均於 2026-09-14 取得。

## 先給決策

第一個可落地的改進不是換模型，而是保留目前 FASHN，做一個可回滾的參數／輸入 A/B：

1. 保留現有預設 `segmentation_free=True`、30 steps、`guidance_scale=1.5`，先把 `segmentation_free` 做成進階選項，對固定圖片配對比較 `True/False`。
2. 同時只加入 upstream 已支援的 steps（30/50）、CFG（至少 1.0/1.5/2.0 的探索值）、正確的 `garment_photo_type`（`model`/`flat-lay`）與固定 seed；不要把 generic SDXL inpainting 當 base VTON 替代品。
3. 輸入端只先驗證現有的 EXIF/RGB/長寬比 fit-and-pad，再以一個「不裁切、不拉伸、必要時補邊」的固定模型比例版本（先以 FASHN 576×864 的 2:3 為目標）做小型 A/B。這個版本是**假說**，不是已證明的品質提升。
4. 只有上述實驗在本機 16 GB 顯存及固定圖片配對上證明改善，才值得做模型替換；若使用情境是非商用，CatVTON 是最便宜的第二候選，IDM-VTON 是較有語意條件控制但整合成本更高的第三候選。

目前不能聲稱任何候選在本專案照片上更好：現有紀錄有 FASHN 的時間／記憶體資料，但沒有跨照片的 base-VTON 品質對照；替代模型也沒有在 RTX 4060 Ti 16 GB 上測過。

## 證據分級

- **本機已量測**：`docs/verification.md` 的啟動、生成、顯存紀錄，以及目前程式碼的實際呼叫參數。
- **官方一手資料**：FASHN 官方程式庫／model card、各候選模型的官方程式庫／model card、官方論文頁。這些數字是 upstream 宣稱，不等同於本機結果。
- **研究推論**：以「推論」標記，不能當成品質成功證據。模型載入成功、mask 合法或輸出能保存，都不代表穿衣品質變好。

## 1. 目前專案的真實 baseline

### 1.1 呼叫路徑與輸入契約

目前路徑是 `app.generate` → `tryon.engine.Engine.generate` → FASHN pipeline；之後可選擇進入手部 preserve 流程。原始碼：

- [`app.py`](../app.py)
- [`tryon/engine.py`](../tryon/engine.py)
- [`tryon/core.py`](../tryon/core.py)

本機 `prepare_image` 會做 EXIF transpose、RGBA 白底合成、轉 RGB，拒絕短邊小於 64 或超過 20M pixels 的圖片。`validate_options` 目前只接受 `tops`、`bottoms`、`one-pieces`，服裝照片類型只接受 `model` 或 `flat-lay`，steps 限定 20–50、seed 限定 32-bit unsigned，並固定回傳：

```text
num_samples=1
guidance_scale=1.5
segmentation_free=True
```

UI 預設是 30 steps、seed 42；目前沒有 UI 選項可以改 CFG、`segmentation_free` 或 `skip_cfg_last_n_steps`。這是值得先做的小型整合點，因為 upstream 已有同名參數，無需引入新模型或新依賴。

### 1.2 本機量測與已知限制

以下是本專案既有驗證紀錄，不是本次重新跑的結果：

- RTX 4060 Ti 16 GB、32 GB RAM；Torch 2.8.0。
- FASHN cold startup 約 8.147 秒，warm load 約 0。
- 目前含 preserve-hands 的端到端 generate/hand processing 約 60.757 / 60.248 秒；這不是純 base model latency，因為流程最多可嘗試兩個 seed 並做後處理。
- Torch peak allocated 約 2.956 GiB、reserved 約 5.084 GiB；不包含 ONNX 或其他程序的完整顯存上限。
- base 生成畫布為 576×864（移除 padding 後輸出尺寸可不同）；既有 SDXL repair 約 22–24 秒、peak allocated 約 5.365 GiB，但過去測試曾在衣服上產生文字／符號／粉紅圖案。
- 現有資料沒有跨照片的 base-VTON 品質證據，手部流程也明確將 `anatomical_correctness` 記為未驗證。

詳見 [`docs/verification.md`](verification.md) 與 [`docs/hand-correction-research.md`](hand-correction-research.md)。因此本報告不把 repair 測試結果外推成 base VTON 品質，也不把 FASHN 的顯存餘裕外推成其他模型的顯存需求。

## 2. FASHN upstream：目前真正可調的品質槓桿

### 2.1 版本、更新與可及性

專案鎖定 FASHN upstream commit [`7c0f10af3f91ad4048fe9729c470a13ef905d25a`](https://github.com/fashn-AI/fashn-vton-1.5/commit/7c0f10af3f91ad4048fe9729c470a13ef905d25a)。該 commit 的訊息是修正 `onnxruntime-gpu` 依賴與 README 安裝說明；它不是模型權重或推論品質更新。以研究日直接比較 pinned commit 與 `main` 的 pipeline、preprocessing、範例、README、`pyproject.toml`，未發現影響本次推論路徑的內容差異；這表示升到目前 `main` 沒有已知的品質修補可直接收穫，且會失去可重現的 pin。

官方 repo 目前只有少量公開活動：

- [官方 repo](https://github.com/fashn-AI/fashn-vton-1.5) 有程式、範例與下載腳本。
- [Issues #4：Where is the paper?](https://github.com/fashn-AI/fashn-vton-1.5/issues/4) 與 [Issues #5：accessories](https://github.com/fashn-AI/fashn-vton-1.5/issues/5) 在研究日仍未提供可用的品質／配件修正答案。
- 唯一的開放 PR 是 [#3 Docker image](https://github.com/fashn-AI/fashn-vton-1.5/pull/3)，重點是 CPU-compatible Docker/CI，不是生成品質。
- [Releases](https://github.com/fashn-AI/fashn-vton-1.5/releases) 沒有正式 release；因此不應把「最新版本」當成已驗證的品質升級。

權重實際可取得：官方 [Hugging Face model card](https://huggingface.co/fashn-ai/fashn-vton-1.5) 提供模型下載，repo 的 [`scripts/download_weights.py`](https://raw.githubusercontent.com/fashn-AI/fashn-vton-1.5/main/scripts/download_weights.py) 下載 model 與 DWPose，human parser 首次使用時另外快取。model card 宣稱模型約 972M parameters、輸出 576×864、Ampere 用 BF16、約 8 GB VRAM、H100 約 5 秒；這些是 upstream 宣稱，不是本 RTX 4060 的新量測。

### 2.2 pipeline 與參數事實

官方目前的 [pipeline.py](https://raw.githubusercontent.com/fashn-AI/fashn-vton-1.5/main/src/fashn_vton/pipeline.py) 與 pinned 版本的 [pipeline.py](https://raw.githubusercontent.com/fashn-AI/fashn-vton-1.5/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/pipeline.py) 顯示：

- 人像 pose 由 DWPose 自動推論；`flat-lay` 服裝用 dummy keypoints，穿在人身上的 `model` 服裝使用 DWPose。
- FASHN human parser 預測人體／服裝區域，category 控制服裝 label；`segmentation_free=True` 會跳過人像 masking，並非一個空泛的 UI 名稱。
- 服裝 mask 在 `flat-lay` 路徑會關閉；服裝若已穿在人身上，則可使用相應處理。
- 圖片做 aspect-preserving resize、padding，模型以 BF16 推論；採 Euler flow schedule。
- 預設 `guidance_scale=1.5`，最後 1 個 sampling step 跳過 CFG，以減少色彩過飽和。
- `num_timesteps` 官方範例建議 20（快）、30（平衡）、50（品質）；`num_samples` 可 1–4；seed 可固定重現。

官方 [basic inference example](https://github.com/fashn-AI/fashn-vton-1.5/blob/main/examples/basic_inference.py) 也公開了 `--no-segmentation-free`、`--guidance-scale`、`--num-timesteps`、`--garment-photo-type` 等選項。這直接支持先做參數 A/B，不支持「一定要換模型」的結論。

### 2.3 preprocessing 可用的實際槓桿

官方 [transforms.py](https://raw.githubusercontent.com/fashn-AI/fashn-vton-1.5/main/src/fashn_vton/preprocessing/transforms.py) 的 `AspectPreserveResize` 和 `ResizePad` 會以 fit-and-pad 保持比例；pose preprocessing 可禁止 upsampling。官方 [agnostic.py](https://raw.githubusercontent.com/fashn-AI/fashn-vton-1.5/main/src/fashn_vton/preprocessing/agnostic.py) 的 mask 邏輯使用 parser labels、body coverage、identity labels，並保留 hands/feet 等身份區域，還有依高度縮放的 dilation/buffer。

這裡有一個重要但尚未量測的取捨：目前專案固定 `segmentation_free=True`，因此不能直接假設 upstream 的 agnostic-mask 優勢已被使用；改成 `False` 可能改善服裝覆蓋，也可能誤傷人體／背景或降低 identity 保留。正確做法是固定同一批輸入做 A/B，而不是依程式名稱推斷品質。

輸入端現有 `prepare_image` 的 EXIF/RGB/白底處理應保留。額外的 crop、背景移除、銳化、放大或服裝邊緣清理都應是獨立變因；一次改多項會無法知道改善來自哪裡。尤其 upstream 已經做 aspect-preserving resize，額外 resize 只能在固定比例、不裁切且有明確目的時測試。

### 2.4 license 與未解的 finetune/LoRA 問題

FASHN VTON model card 標示 Apache-2.0，但整個本機 pipeline 不應簡化成「所有元件 Apache-2.0」：

- FASHN VTON 權重：model card 的 [License](https://huggingface.co/fashn-ai/fashn-vton-1.5) 為 Apache-2.0。
- DWPose/YOLOX：官方 repo 以 Apache-2.0 元件使用。
- FASHN human parser：官方 [parser repo](https://github.com/fashn-AI/fashn-human-parser) 與 [LICENSE](https://github.com/fashn-AI/fashn-human-parser/blob/main/LICENSE) 採 NVIDIA SegFormer source license；其 [Hugging Face card](https://huggingface.co/fashn-ai/fashn-human-parser) 也標為 `nvidia-segformer`。商用部署必須分別核對 parser 條款。

截至研究日，FASHN 官方 repo 沒有 `train.py`、LoRA adapter、finetune recipe 或額外 reference-person conditioning 的公開、可直接執行入口；model card 的 training 說明是從頭訓練，包含 18M masked pairs 與 4M synthetic triplets，而非給本專案直接微調的 checkpoint。這是 bounded absence（針對目前官方 repo/model card），不是宣稱永遠沒有私有訓練工具。

因此目前不建議為了「提升某一類衣服」自行猜測 MMDiT layer mapping 做 LoRA。若未來真的要做，至少要先準備同一人物／服裝條件的高品質配對、pose/parser target、驗證 split、顯存與 checkpoint 合併方案；成本與風險遠高於先測 upstream 已支援的開關。

## 3. 替代模型：能不能在本機真正跑

下表把官方宣稱、可取得性與本機未知分開。除非另寫「本機量測」，表中的 VRAM/速度都是 upstream 宣稱或推論。

| 候選 | 公開可取得的實作 | 條件／輸入能力 | 資源與相容性 | License／門檻 | 判斷 |
|---|---|---|---|---|---|
| **FASHN VTON 1.5** | 官方 repo、HF 權重、下載腳本均可取得 | 人像 + garment；DWPose；`model`/`flat-lay`；無 reference-person 輸入 | 本機已跑通；現有 Torch 2.8/ONNX CUDA；上游宣稱約 8 GB | VTON Apache-2.0，但 parser 有 NVIDIA SegFormer 條款 | 首選，先做參數與輸入 A/B |
| **CatVTON** | [官方 repo](https://github.com/Zheng-Chong/CatVTON)、[HF 權重](https://huggingface.co/zhengchong/CatVTON)、mask-free 版本與 Gradio app 公開 | 人像 + garment + mask/automask；SD1.5 inpainting；沒有本專案所需的額外 reference-person 分支 | README 宣稱 1024×768 小於 8 GB；本機未測。官方 requirements pin Torch 2.1.2、Diffusers 0.29.2、xformers，與現環境不一致，應隔離 venv | code、checkpoint、demo 為 CC BY-NC-SA 4.0；商用受限 | 非商用時的第二候選；最容易先做 smoke test |
| **IDM-VTON** | [官方 repo](https://github.com/yisol/IDM-VTON)、[HF checkpoint](https://huggingface.co/yisol/IDM-VTON)、inference/train script 公開 | 以 IP-Adapter garment visual features + parallel UNet 保留低層 garment features；需 human parser、DensePose、OpenPose | 官方 inference 768×1024、30 steps、guidance 2；環境 pin Torch 2.0.1/CUDA 11.8、Diffusers 0.25，與現環境衝突；無本機 4060 量測 | code/checkpoint 為 CC BY-NC-SA 4.0；商用受限 | 若 FASHN 對 garment semantics 不足才做隔離實驗；整合成本高於 CatVTON |
| **FastFit** | [官方 repo](https://github.com/Zheng-Chong/FastFit) 列出 code、inference/eval 與 FastFit-MR/SR 權重 | 多參考服裝（top/bottom/dress/shoes/bags）、KV cache；不是目前單 garment 需求的必要條件 | 論文宣稱約 3.5× speed-up，但沒有本機 VRAM/4060 證據；依賴 SD1.5 inpainting/AutoMasker | FastFit non-commercial license；商用需聯絡權利人；DressCode-MR dataset 有學術存取條件 | 只有多件服裝／多 reference 是需求時才值得 |
| **RefVTON / RefTon** | [官方 repo](https://github.com/360CVGroup/RefTon) 與 [HF LoRA](https://huggingface.co/qihoo360/RefVTON) 公開；可選 reference person | FLUX-Kontext；可用 agnostic+cloth、person+cloth、可選 reference-person 來保存 garment identity/fabric/drape | 官方訓練命令是 8 GPUs；無 RTX 4060 單卡量測；FLUX-Kontext-dev 是更重且受限的 backbone | HF adapter card 標 Apache-2.0，但 repo 在研究日未見清楚 LICENSE；[FLUX-Kontext-dev license](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/blob/main/LICENSE.md) 的 backbone 條款仍會限制整體使用 | 有 reference-person 需求才考慮；不是本專案第一個替換 |
| **CatV2TON** | [官方 repo](https://github.com/Zheng-Chong/CatV2TON) 有 256/512 image/video 更新；HF [CatV2TON](https://huggingface.co/zhengchong/CatV2TON) 有頁面 | DiT image/video 路線，非現有最小替換；公開頁面的可重現單卡結果不足 | 沒有本機結果；HF card 的 license 為 CC-BY-NC-ND-4.0，另有 V2 頁面但資訊不完整 | ND 限制與實作／權重版本要再核對 | 研究追蹤，不作目前落地候選 |
| **Oxygen-TryOn** | [官方 GitHub](https://github.com/OxygenVision/Oxygen-TryOn) 在研究日把 technical report、online demo、inference code、model weights 列為尚未完成項；同名 HF 頁曾回 401，且頁面所指 GitHub URL 不一致 | Qwen3-VL/FLUX-like 新架構的論文／宣稱，不等於可取得的 runnable checkpoint | 無可驗證的單卡安裝與 4060 資源數據 | 實作與權重可及性、license 未能核實 | 排除；不能把論文／行銷 benchmark 當成可部署方案 |

### 3.1 CatVTON：最可行的隔離 smoke test

官方 [README](https://raw.githubusercontent.com/Zheng-Chong/CatVTON/main/README.md) 宣稱 CatVTON 約 899.06M total parameters、49.57M trainable parameters，並在 1024×768 小於 8 GB VRAM；另有 mask-free 版本。官方 [app.py](https://raw.githubusercontent.com/Zheng-Chong/CatVTON/main/app.py) 會對 person 做 resize/crop、對 cloth 做 resize/padding，可使用 automask 或 user mask；[requirements.txt](https://raw.githubusercontent.com/Zheng-Chong/CatVTON/main/requirements.txt) 則 pin 到舊版 Torch/Diffusers/xformers。

這使它在 16 GB 卡上「可能」可跑，但「小於 8 GB」只是作者宣稱；相容性、速度與衣服品質都要在獨立環境量測。其 CC BY-NC-SA 4.0 也意味著不能直接把結果當成商用替代方案。

### 3.2 IDM-VTON：reference/garment semantics 更有根據，但更重

官方 [paper](https://arxiv.org/abs/2403.05139) 描述 IDM-VTON 用 garment image encoder 的高層語意透過 cross-attention 融合，並用 parallel UNet 的低層 garment features 透過 self-attention 融合；官方 [inference.py](https://raw.githubusercontent.com/yisol/IDM-VTON/main/inference.py) 是 SDXL ControlNet inpaint 路線，依賴 parser、DensePose、OpenPose 與 IP-Adapter/CLIP 資源。

這比目前 FASHN 的單一 garment conditioning 更適合驗證「衣服 logo／材質／結構是否被保留」的假說，但不能把論文機制直接轉成專案品質提升。官方 [environment.yaml](https://raw.githubusercontent.com/yisol/IDM-VTON/main/environment.yaml) pin 了 Python 3.10、Torch 2.0.1/CUDA 11.8、Diffusers 0.25 等舊組合，必須隔離，且本報告沒有對其在 RTX 4060 Ti 的顯存或速度做測量。

### 3.3 其餘模型的狹窄用途

- FastFit 的價值在多 reference 與 KV caching；若目前只有一件 garment，它增加的資料／license／整合面積沒有明顯理由。
- RefVTON 是目前候選中唯一明確把「可選穿著 reference person」放進輸入條件的方向，但 FLUX-Kontext backbone 與 8-GPU 訓練設定使它不適合作為 4060 Ti 的第一個實驗。
- CatV2TON 與 Oxygen-TryOn 的公開頁面／license／runnable 狀態不足，先列為監測項，不納入本輪 implementation。

## 4. finetune、LoRA、reference 與 pose conditioning

### 4.1 目前最重要的差異

| 方向 | FASHN 1.5 | CatVTON | IDM-VTON | FastFit / RefVTON |
|---|---|---|---|---|
| garment conditioning | garment photo + category；內部 parser/pose | garment + person + mask/automask | garment visual encoder + parallel UNet/IP-Adapter | FastFit 多 reference；RefVTON 可選 reference person |
| pose conditioning | DWPose 自動；flat-lay 使用 dummy keypoints | SCHP + DensePose automask | DensePose/OpenPose/ControlNet 等多元條件 | 依各自 SD/FLUX pipeline |
| 公開 LoRA/finetune | 官方沒有可直接用的 recipe/checkpoint | 官方 repo 有訓練／模型材料，但 license 非商用 | 官方 repo 有 `train_xl.py`，需要資料與 SDXL/IP-Adapter 資源 | FastFit 有專用權重；RefVTON 有 LoRA，但 backbone/條款更重 |
| 額外 reference person | 沒有現成參數 | 沒有本專案所需的現成 branch | 不是預設的任意 reference-person API | 是 RefVTON 的主要賣點，FastFit 是多 garment reference |

對本專案而言，「做 FASHN LoRA」不是一個小改動：需要自行決定 MMDiT 哪些 layer 注入 adapter、整理人物／服裝配對、重新產生 parser/pose target、處理 576×864 訓練形狀與驗證 split，還要證明 adapter 沒有把身份、背景或衣服類別過擬合。FASHN upstream 沒有提供這條路的公開可重現配方，因此先跳過。

若真正需求是「同一件衣服的細節要更像」，先用 IDM-VTON 的公開 conditioning 機制做隔離對照，或等有合規 reference-person 需求時再評估 RefVTON；這兩者都不是在 FASHN 上加一個 `lora_path` 就能完成。

### 4.2 輸入轉換的最小原則

1. 保留 EXIF orientation、RGBA→白底 RGB、尺寸上限等現有安全邊界。
2. 讓使用者明確選 `model`（服裝已穿在模特身上）或 `flat-lay`（平拍服裝）；不要用自動猜測取代可追蹤的 metadata。
3. 對人像與服裝都使用比例保持的 fit-and-pad，避免未記錄的拉伸與裁切；額外 3:4/模型比例補邊只作獨立實驗。
4. 不先做 aggressive background removal、銳化、超解析或人工 mask；這些變因若需要，應各自產生版本與 metadata。
5. 固定 seed、category、photo type、輸出尺寸，否則無法判斷參數或 transform 的影響。

## 5. 建議的精確整合實驗（先不換模型）

這是給 implementation/evaluation agent 的最小、可重現計畫；本研究沒有執行它。

### 5.1 固定資料與輸出

- 建立 12 組**本地**固定輸入：每種 `tops`、`bottoms`、`one-pieces` 各 4 組；其中包含已穿著的 `model` garment 與平拍的 `flat-lay` garment，避免只在一種來源上調參。
- 每組記錄輸入檔案 hash、原始尺寸、EXIF 處理前後、category、photo type；不可上傳私人圖片。
- 每次 `num_samples=1`，seed 42；第二次只在要檢查 seed 敏感度時用 43。先將 `preserve_hands=False` 隔離 base VTON，手部由另一條工作流處理。
- 每張輸出都寫入：FASHN commit、Torch/ONNX 版本、steps、CFG、`segmentation_free`、skip-CFG steps、transform variant、wall time、peak allocated/reserved、是否 OOM。

### 5.2 第一階段 6 組 screening 矩陣

挑 6 組代表輸入，固定所有其他條件，依序跑：

| ID | steps | CFG | segmentation-free | transform | 目的 |
|---|---:|---:|---|---|---|
| A baseline | 30 | 1.5 | true | 現有 `prepare_image` | 專案目前行為 |
| B mask-aware | 30 | 1.5 | false | 現有 `prepare_image` | 驗證 upstream agnostic mask 是否值得暴露 |
| C more steps | 50 | 1.5 | true | 現有 `prepare_image` | 速度／品質 trade-off |
| D lower CFG | 30 | 1.0 | true | 現有 `prepare_image` | 避免衣服色彩／細節過度引導；探索值 |
| E higher CFG | 30 | 2.0 | true | 現有 `prepare_image` | 檢查條件遵循是否改善或產生過飽和；探索值 |
| F ratio fit-pad | 30 | 1.5 | true | EXIF/RGB 後不裁切 fit-and-pad 到固定模型比例 | 輸入轉換假說；不可與 B/C 混為一談 |

30/50 steps、CFG 預設 1.5、`segmentation_free` 選項與最後跳過 CFG 是 upstream 有明確程式支援的；1.0/2.0 是**測試值，不是官方品質建議**。若 F 的補邊在實作上改變背景語意，保留現有輸入作為唯一 baseline，並在 metadata 中明記。

### 5.3 第二階段與採用門檻

1. 由 evaluation agent 盲測第一階段所有輸出；先只選最多兩個候選條件，不增加新的自由度。
2. 在全部 12 組上重跑 baseline 與候選條件，分類報告 tops/bottoms/one-pieces、model/flat-lay、服裝遮蓋、衣服細節、人體 identity、背景污染與失敗率。
3. 只在候選條件跨多組輸入都改善，且 16 GB 顯存無 OOM、延遲可接受、沒有明顯背景／身份回歸時，才把它做成 UI 進階選項；否則維持現有預設。
4. 不把單張漂亮結果、模型載入成功、mask 面積合理或輸出可保存當成採用證據。

最小 implementation 變更應該是把 upstream 已有的 `segmentation_free`、`guidance_scale`（必要時 `skip_cfg_last_n_steps`）從固定值變成可記錄的內部選項，預設值不變；不要先抽象出新的模型介面、下載器或 LoRA 框架。

### 5.4 第二模型的條件式 smoke test

若第一階段仍顯示 FASHN 對 garment identity/細節有系統性問題，再依序做：

1. 在獨立 venv 測 CatVTON 官方 checkpoint、官方推薦輸入流程，先只跑 3 組公開／非私人測試圖，記錄 VRAM、時間、依賴與 license；不把它接進主環境。
2. 若 CatVTON mask/SD1.5 路線對細節不夠，再以同樣 3 組輸入測 IDM-VTON；不要因為論文分數高就跳過環境與顯存驗證。
3. 只有產品需求明確需要多 garment reference 或穿著 reference，才開 FastFit/RefVTON 的專題；這些不是目前單 garment base quality 的最小解。

## 6. 風險、未知與結論

- **品質未知**：沒有本機跨照片的 FASHN `segmentation_free`/CFG/steps 對照；不要提前宣稱 B、C、D、E 或 F 會提升品質。
- **資源未知**：FASHN 已在本機跑通；CatVTON 的 `<8 GB` 是作者宣稱；IDM-VTON、FastFit、RefVTON 沒有本機 4060 Ti 16 GB 數據。
- **相容性**：CatVTON/IDM-VTON 官方 environment pin 與現有 Torch 2.8／Diffusers 0.35.1 不同；隔離環境是必要條件，不能為了測試覆蓋現有安裝。
- **法律／商用**：FASHN VTON Apache-2.0 不會自動覆蓋 human parser 的 NVIDIA 條款；CatVTON、IDM-VTON 與 FastFit 是非商用條款；RefVTON adapter 與 FLUX backbone 要分開核對。
- **可及性**：Oxygen-TryOn 的官方可執行碼／權重在研究日仍不可驗證；CatV2TON 的公開頁面不足以證明本專案可重現。
- **訓練／LoRA**：目前沒有足夠的配對資料、官方 recipe 或驗證證據支持直接做 FASHN LoRA；先不做。

**總結建議**：先把 FASHN 的 upstream 參數與輸入變因做成可記錄、可回滾的 A/B，預設仍維持 `segmentation_free=True`、30 steps、CFG 1.5。若要做替代模型，非商用前提下先隔離 smoke test CatVTON，再考慮 IDM-VTON；只有 reference/multi-item 需求才研究 RefVTON/FastFit，Oxygen-TryOn 暫不列入可部署清單。

## 一手來源索引

### FASHN

- [FASHN VTON 官方 repo](https://github.com/fashn-AI/fashn-vton-1.5)
- [Pinned commit 7c0f10a](https://github.com/fashn-AI/fashn-vton-1.5/commit/7c0f10af3f91ad4048fe9729c470a13ef905d25a)
- [FASHN model card](https://huggingface.co/fashn-ai/fashn-vton-1.5)
- [pipeline.py](https://raw.githubusercontent.com/fashn-AI/fashn-vton-1.5/main/src/fashn_vton/pipeline.py)
- [transforms.py](https://raw.githubusercontent.com/fashn-AI/fashn-vton-1.5/main/src/fashn_vton/preprocessing/transforms.py)
- [agnostic.py](https://raw.githubusercontent.com/fashn-AI/fashn-vton-1.5/main/src/fashn_vton/preprocessing/agnostic.py)
- [basic inference example](https://github.com/fashn-AI/fashn-vton-1.5/blob/main/examples/basic_inference.py)
- [FASHN human parser repo](https://github.com/fashn-AI/fashn-human-parser)
- [FASHN human parser license](https://github.com/fashn-AI/fashn-human-parser/blob/main/LICENSE)
- [FASHN human parser model card](https://huggingface.co/fashn-ai/fashn-human-parser)

### 替代模型

- [CatVTON repo](https://github.com/Zheng-Chong/CatVTON) · [README raw](https://raw.githubusercontent.com/Zheng-Chong/CatVTON/main/README.md) · [requirements raw](https://raw.githubusercontent.com/Zheng-Chong/CatVTON/main/requirements.txt) · [app raw](https://raw.githubusercontent.com/Zheng-Chong/CatVTON/main/app.py) · [HF model](https://huggingface.co/zhengchong/CatVTON)
- [IDM-VTON repo](https://github.com/yisol/IDM-VTON) · [README raw](https://raw.githubusercontent.com/yisol/IDM-VTON/main/README.md) · [environment raw](https://raw.githubusercontent.com/yisol/IDM-VTON/main/environment.yaml) · [inference raw](https://raw.githubusercontent.com/yisol/IDM-VTON/main/inference.py) · [paper](https://arxiv.org/abs/2403.05139) · [HF model](https://huggingface.co/yisol/IDM-VTON)
- [FastFit repo](https://github.com/Zheng-Chong/FastFit) · [paper](https://arxiv.org/abs/2508.20586)
- [RefVTON/RefTon repo](https://github.com/360CVGroup/RefTon) · [HF adapter](https://huggingface.co/qihoo360/RefVTON) · [paper](https://arxiv.org/abs/2511.00956) · [FLUX-Kontext-dev license](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/blob/main/LICENSE.md)
- [CatV2TON repo](https://github.com/Zheng-Chong/CatV2TON) · [HF model](https://huggingface.co/zhengchong/CatV2TON)
- [Oxygen-TryOn official repo](https://github.com/OxygenVision/Oxygen-TryOn) · [project page](https://oxygenvision.github.io/Oxygen-TryOn/) · [paper](https://arxiv.org/abs/2607.21694) · [unverified HF page](https://huggingface.co/LYAWWH/Oxygen-TryOn)
