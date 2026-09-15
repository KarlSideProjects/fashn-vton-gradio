# FASHN VTON 1.5：理論來源查核筆記

查核日期：2026-09-15。以 Firecrawl 讀取作者論文全文、FASHN 官方專案頁、模型卡與固定版本原始碼。這份筆記供 README 引用；理論教學公式不等於已驗證的 FASHN 完整訓練配方。

## 1. Flow Matching：學習「目前該往哪裡改」

Lipman 等人的 **Flow Matching for Generative Modeling** 提出以向量場回歸訓練連續流模型。向量場可理解為：給定目前的資料數值與時間，告訴我們每個數值應以多快的速度改變。訓練可採用條件路徑建立監督訊號，而不必每次都先跑完整段 ODE。生成時才從簡單分布（例如高斯噪聲）出發，用數值方法逐步解 ODE。[原論文 §2–3](https://arxiv.org/html/2210.02747v2#S2)

- 適合初學者：「模型學會每一步怎麼修改畫布；取樣器照著建議一步步更新。」
- 「畫布」指整張圖的數值陣列；速度指像素值或表示的變化率，不是布料在照片上的物理移動速度。
- Flow Matching 是訓練方法的廣泛框架，可搭配不同機率路徑，包含部分 diffusion 路徑；不能直接定義為「永遠走直線」。[原論文 §4](https://arxiv.org/html/2210.02747v2#S4)

## 2. Rectified Flow：用直線插值提供容易取得的訓練目標

Liu、Gong、Liu 的 **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**，以兩端樣本之間的直線插值提供速度監督。令 `x₀` 是噪聲、`x₁` 是訓練目標圖像，可用以下教學寫法：[原論文 §2.1、式 (1)](https://arxiv.org/html/2209.03003v1#S2.E1)

```text
xₜ = (1 − t) x₀ + t x₁
目標速度 = x₁ − x₀
訓練：讓 vθ(xₜ, t) 接近目標速度
生成：x下一步 ≈ x目前 + Δt × vθ(x目前, t)
```

最後一行是 Euler 更新的直覺。在影像條件生成中，可另外把人物、衣服等條件 `c` 交給網路，寫成 `vθ(xₜ, t, c)`；這是用來解釋條件生成的延伸記法，不是 FASHN 公開訓練程式的逐行翻譯。

- 訓練有目標圖片可計算速度；推論時沒有未來的完成圖，必須由學過的網路預測方向。
- **直的是建構訓練樣本的插值路徑，不保證學得的每條生成路徑完全筆直。** 原論文圖 2 特別區分直線插值與學得的 flow；reflow 是另外重建配對、再訓練以拉直路徑的程序。[原論文 §2.1、圖 2](https://arxiv.org/html/2209.03003v1#S2.F2)
- 因此不能由「Rectified Flow」推論 FASHN 用了一次以上 reflow、保證一步生成，或套用了論文全部訓練細節。
- 時間方向是記號慣例。Liu 可用 `0 → 1` 表示噪聲到資料；SD3 論文式 (13) 採 `t=0` 為資料、`t=1` 為噪聲，生成方向相反。README 應與實際取樣程式保持一致。[SD3 原論文 §3、式 (13)](https://arxiv.org/html/2403.03206v1#S3.E13)

## 3. DiT 與 MMDiT：負責預測的網路架構

**DiT**（Diffusion Transformer）是 Peebles、Xie 在 **Scalable Diffusion Models with Transformers** 探討的架構：把空間輸入切成 patches、轉成向量 tokens，再由 Transformer blocks 處理。原始論文的主要模型處理 VAE latent；「DiT」這個架構名稱本身不能證明另一模型也使用 VAE。[原論文 §3.2](https://arxiv.org/html/2212.09748v2#S3.SS2)

**MMDiT / MM-DiT** 出自 Esser 等人的 **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis**（SD3）。原模型使用文字與影像兩套權重，並在 attention 運算時合併兩種 token 序列，讓兩邊交換資訊。它描述網路怎麼安排與交換特徵；Flow Matching / Rectified Flow 描述生成流與學習方式，屬於不同層次。[原論文 §4、圖 2](https://arxiv.org/html/2403.03206v1#S4)

適合初學者的比喻：把圖片分成很多小卡片，每張卡片變成一串數字；attention 讓卡片參考其他相關卡片。卡片是特徵表示，並非把衣服真的切塊、搬移、黏貼。FASHN 的人物與衣服都是影像輸入；不能照搬 SD3 的文字輸入、文字編碼器或 VAE 流程。

## 4. FASHN 官方已公開的事實

| 問題 | 查核結論與來源 |
| --- | --- |
| 是否直接在像素空間生成？ | 是。官方明言直接處理 RGB，不經 VAE 編碼；使用 12×12 patch embedding。切 patch 與 VAE 壓縮不是同一件事。[專案頁 Architecture](https://fashn.ai/research/vton-1-5) |
| MMDiT 怎麼用於換裝？ | 官方描述人物與服裝共同處理：8 個 double-stream blocks、16 個 single-stream blocks，前置 4 個 patch-mixer blocks；姿勢以單通道灰階圖加入，服裝類別 embedding 加入時間條件。[專案頁 Architecture](https://fashn.ai/research/vton-1-5) |
| 是否需要文字 prompt？ | 公開模型輸入是人物 RGB、衣服 RGB、類別與姿勢。固定版本 `TryOnPipeline.__call__` 沒有文字 prompt 參數；類別字串是選項標籤。[模型卡 Inputs](https://huggingface.co/fashn-ai/fashn-vton-1.5#inputs)、[固定版本 pipeline.py](https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/pipeline.py) |
| Maskless 是否表示完全不做人像分割？ | 不能這樣寫。固定版本仍對人物與衣服呼叫 human parser；`segmentation_free` 控制人物遮罩，flat-lay 控制服裝遮罩。可寫「預設不先遮掉人物的原衣服區域」。[固定版本 pipeline.py](https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/pipeline.py) |
| 訓練資料有公開說明嗎？ | 有。作者報告從零訓練：第一階段 18M 遮罩試穿配對；第二階段以 50/50 混合遮罩配對與第一階段模型生成的 4M 合成三元組；訓練最多丟棄 75% tokens。[專案頁 Training](https://fashn.ai/research/vton-1-5)、[模型卡 Training](https://huggingface.co/fashn-ai/fashn-vton-1.5#training) |
| 為什麼合成三元組有幫助？ | 官方說明：模型先生成穿上其他衣服的人物，再以原本圖片為 ground truth 建立三元組，讓後續模型學習從未遮罩人物換裝並保留身形。這是作者對訓練機制的說明。[專案頁 Training](https://fashn.ai/research/vton-1-5) |
| 能否宣稱完全保留身形、logo？ | 不能。免 VAE 是減少某一來源的資訊損失，不代表生成無誤；作者列出身形仍可能變動、原衣殘留，以及 576×864 解析度限制。[專案頁 Limitations](https://fashn.ai/research/vton-1-5) |

## 5. README 建議用語與證據界線

可寫：「FASHN 用人物和服裝圖片作為條件，由 Transformer 反覆預測目前圖像應如何更新，取樣器從噪聲逐步產生換裝結果。直接在 RGB 像素空間處理，代表不用先透過 VAE 把圖壓進另一個潛在空間；模型仍會把圖片切成 patches 並轉為特徵向量。」架構與像素空間部分見[官方專案頁](https://fashn.ai/research/vton-1-5)；具體速度與 Euler 更新須搭配 README 作者查核的取樣原始碼引用。

訓練流程可以引用上述作者公布的兩階段資料規模，但此輪查核的官方頁面仍以「Paper coming soon」標示論文，且未建立完整損失、時間取樣分布、最佳化器、資料處理與訓練超參數的可重現配方。不要把 Lipman、Liu 或 SD3 的示範公式／設定直接寫成 FASHN 已確認的訓練細節。[官方專案頁 Citation](https://fashn.ai/research/vton-1-5)、[固定版本 README](https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/README.md)
