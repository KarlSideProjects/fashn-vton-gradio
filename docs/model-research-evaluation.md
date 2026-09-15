# 多模型品質路由與評估研究

**研究日期：** 2026-09-14（Asia/Taipei）
**研究用途：** 實作前的獨立品質與失敗語義審查；本文件只提出評估與路由決策，不修改程式、不下載模型、不啟動 GPU、不上傳影像。
**研究方法：** 先以 Firecrawl MCP 查閱官方模型卡、官方程式庫、官方文件與論文 HTML；文末列出可直接核對的原始 URL 與取用日期。外部論文中的數字是論文作者的報告，不是本機量測。

## 結論先行

審查後更正：以下的 SD1.5／HandRefiner 合併分類僅保留為原研究脈絡，兩者不是同一 backend。
本輪一般 SD1.5 與 inpaint ControlNet 不使用 mesh，也不套用 HandRefiner 的 60×60 限制，
不能因手小或有 Logo 就禁止人工實驗。完整 pilot／holdout 是品質宣稱與自動路由推廣門檻，
不是人工比較工具開始實作的先決條件；定案見 [主方案](model-research-composite.md)。
正常使用可選 no-op；但離線 clean control 必須刻意執行候選且保留原圖，才能量測誤修。
像素改變與新增視覺缺陷分開記錄，不把所有無害變化算品質失敗。參見
[獨立審查 P1-7](model-improvement-adversarial-review.md)。

目前最可靠的方案不是讓一個品質分數自動挑出「最好」的生成結果，而是把路由限制在可證明的適用條件，再讓使用者在最多兩個、同一輸入與同一遮罩 lineage 下的候選中選擇。自動化可以負責 no-op、明確的 OpenCV 小修補、遮罩與輸出安全性驗證、資源與逾時處理；不能在未校準的 IQA 分數或單次模型成功回傳的前提下自動接受候選。

建議保留以下 bounded policy：

1. 永遠先保留原始 FASHN 結果作為 no-op 基線；already-correct 案例是必要的負對照。
2. 只在人工確認的遮罩、可用的 person/garment reference、固定的模型 revision 與固定參數下，產生最多兩個候選。候選不可在沒有使用者明確採用前，成為下一階段的輸入。
3. OpenCV Telea 只處理小、平滑、沒有結構或紋理語義的污點；手、袖口、拳頭、logo、文字、明顯遮擋一律不把 Telea 的成功當成品質成功。
4. SDXL 是目前已有本地量測的生成候選；SD1.5/HandRefiner 與 IP-Adapter 分支先作受限實驗，不應被宣稱為已證實的提升。FLUX Fill 因為 12B 規模、權重存取門檻與非商用授權，不列入預設路由。
5. 任何候選即使通過外部像素不變、尺寸、格式和 hash 檢查，仍只標成 needs_review；只有使用者採用才是 accepted。

本結論補足既有的 [docs/verification.md](verification.md) 與 [docs/hand-correction-research.md](hand-correction-research.md)：前兩份文件已驗證工程流程與單模型 smoke test，本報告專門處理「模型差異和 seed reroll 如何分離」、「候選如何公平比較」、「何時不得自動路由或接受」。

## 證據邊界與現有本機基線

### 已知的工程現況

父工作區在 31a2551，現有 Engine.generate 先執行 FASHN，再可選擇 source_hand_composite_v2；嘗試數量上限為兩個 seed，並有幾何拒絕條件。Engine.repair 會釋放 FASHN pipeline，再呼叫 tryon.repair.run_repair；後者目前只有 SDXL 或 OpenCV Telea，並以硬式 patch composite 保證遮罩外像素不變，結果狀態仍是 needs_review。restore_hands 是來源手像素的受限合成，幾何 gate 通過不代表解剖品質已驗證。相關流程可由 [tryon/engine.py](../tryon/engine.py)、[tryon/repair.py](../tryon/repair.py)、[tryon/hands.py](../tryon/hands.py) 和 [scripts/inpaint.py](../scripts/inpaint.py) 核對。

現有文件所記錄的實測，不可與模型卡的理想硬體數字混用：

| 項目 | 已量測或已驗證的事實 | 對本研究的意義 |
| --- | --- | --- |
| FASHN | verification 記錄冷啟動約 8.147 秒、暖機生成約 60.248–60.757 秒；PyTorch peak 約 2.956 GiB allocated、5.084 GiB reserved | 只能說明此工作區此時的流程成本，不代表其他 candidate 的畫質或別的 GPU 成本 |
| SDXL inpaint | verification 的 local smoke 約 22–24 秒/次；PyTorch peak allocated 約 5.365 GiB | 可作為現有本地候選的成本基線，仍不能把 checkpoint 載入或 patch 成功當作視覺通過 |
| OpenCV Telea | verification 的小 blemish 約 0.17 秒 | 適合當低成本 control；不適合手、袖口、logo 或文字 |
| 遮罩外像素 | run_repair 的硬式 composite 在已測流程中 outside_mask_max_difference=0 | 是安全不變量，不是 inside-mask 的品質指標 |
| SDXL 視覺 smoke | 顏色區塊布料在 strength 0.75 尚能留住色塊；其他 strength、prompt 或 seed 曾 hallucinate 文字、符號或粉紅色，strength 1.0 也有品質退化 | 需要 logo/texture adversarial cases；不能用固定 seed 的單一成功案例宣稱模型改善 |
| 未驗證範圍 | 跨照片 identity、拳頭、長袖/袖口、tiny hands、複雜遮擋、logo/細紋理、品質自動評分均尚無 cross-photo evidence | 這些正是本次小型 holdout 必須覆蓋的 strata |

FASHN 官方模型卡報告其模型是 972M pixel-space MMDiT、輸出 576x864，主要涵蓋 tops、bottoms、one-pieces，並提出約 5 秒 H100、約 8 GB VRAM 的估計；模型卡同時承認 body shape、長短版衣物轉換、較低解析度等限制。這是供應者的模型卡聲明，不是 4060 Ti 16 GB 的量測；官方 repository 固定 revision 也應寫入每筆 audit，不能只記名稱。[FASHN model card](https://huggingface.co/fashn-ai/fashn-vton-1.5)、[FASHN pinned README](https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/README.md)

### 證據分級

- **A：** 官方程式碼、官方文件、官方模型卡或原始論文，能支持 API、權重、限制、授權或 benchmark 設計。
- **B：** 本 repo 的既有 verification 或 source inspection，能支持目前流程和本機成本。
- **C：** 建議的實驗設計或工程推論，尚未在本機驗證；下文會明確標示為「估算」、「待量測」或「建議」。

## 原始來源的可用性與限制

### 模型與控制方法

| 來源 | 原始證據支持什麼 | 存取、授權、可行性判斷 |
| --- | --- | --- |
| [FASHN model card](https://huggingface.co/fashn-ai/fashn-vton-1.5) | FASHN 的輸入類別、576x864、maskless 預設、DWPose/segmentation 依賴、模型卡限制與約 2 GB 模型加 parser 的重量級概況 | Apache-2.0；是目前實際基線，但模型卡的 H100/VRAM 數字不能當本機承諾 |
| [Diffusers SDXL inpaint model card](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | SDXL inpainting 是以 mask 填補、strength 小於 1 較合適；模型卡警告 legible text、faces/people、完美 photorealism 與 strength 1.0 的限制 | CreativeML Open RAIL++-M，且模型卡標明研究用途；現有 .venv-repair 已有 revision 與實測，不應把新 checkpoint 混進比較 |
| [Diffusers inpaint guide v0.35.1](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/inpaint) | 白色 mask 為填補、黑色為保留；padding mask crop 讓模型在遮罩區域及周圍工作；ControlNet 可提供結構條件；overlay 可強制遮罩外保留；生成仍受隨機性與 prompt 影響 | 與現有 diffusers 0.35.1 對齊；官方文件描述的是機制和取捨，不是本專案的品質保證 |
| [Diffusers IP-Adapter guide v0.35.1](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/ip_adapter) | image encoder feature 會注入 cross-attention；scale 會改變文字與影像條件的平衡；可預計算 embedding、以 binary mask 指定區域，也可和 ControlNet 組合 | 這是 reference conditioning，不是 pixel copy 或 logo/identity 保證；adapter、base、image encoder 的相容性及成本須一併鎖定 |
| [HandRefiner official repository](https://github.com/wenquanlu/HandRefiner) | mesh/ControlNet 條件的手部 inpainting；官方建議 control 0.4–0.8；現有 SD1.5 權重希望手至少約 60x60；SDXL 圖片 resize 到 512，SDXL port 由作者明示未測試；seed 改變可能有幫助 | repository MIT，但 MANO、MeshGraphormer、SD1.5、ControlNet 和模型資產須分開核對；只作 eligible hand subset 的研究候選，不作 SDXL 預設替代 |
| [FLUX.1-Fill-dev model card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) | 12B masked fill pipeline、官方 bfloat16/50 steps 範例，以及邊界顏色、複雜紋理與 prompt 失敗限制 | gated checkpoint，採 FluxDev Non-Commercial License 與 AUP；不符合目前預設的 ungated/16 GB bounded route，暫不下載、不納入預設實驗 |

### 品質與 VTON benchmark

| 來源 | 原始證據支持什麼 | 對本機決策的限制 |
| --- | --- | --- |
| [VTBench paper](https://arxiv.org/html/2505.19571v1) | 以 general image quality、garment preservation、background、hand-occlusion 等層次評估 unpaired VTON；指出 FID/KID 是分布指標，不能可靠判斷單張圖的 texture/size | 是很好的 rubric 設計參考，不等於有可直接裝入本 repo 的 evaluator；[官方 repo](https://github.com/HUuxiaobin/VTBench) 目前以 README/圖表為主，不能宣稱已有可重現本機 harness |
| [VTON-IQA paper](https://arxiv.org/html/2603.13057v1) | reference-free image-level IQA 使用 person、garment、try-on 三者；VTON-QBench 報告 62,688 張圖、431,800 annotations、13,838 annotators；也承認現有資料偏 studio/controlled，scalar 缺少可解釋 attribute feedback | 可作離線研究 scorer/資料設計參考，不可直接把 scalar 當 auto-accept；paper 所述 dataset、權重與本機 Torch2.8 相容性尚未驗證 |
| [VTON-IQA official repository](https://github.com/litelightlite/VTON-IQA) | 官方提供 scorer、batch evaluator、checkpoint/download 說明；repo 更新中標示 code/model/dataset 為 CC BY-NC-SA 4.0，並列出 Python 3.10.14、Torch 2.6.0 | 需要下載額外模型/資料且有 non-commercial share-alike 條件；不在本次無安裝、無下載的本機預設路徑，只能在合法隔離環境作 exploratory ranking |
| [OpenVTON-Bench paper](https://arxiv.org/html/2601.22725v1) | 提出 background、identity、texture、shape、realism 五維度，論文報告 VLM+SAM3/morphology 組合與 Kendall tau 0.833 對 SSIM 0.611，並描述大規模人評 | 這些是論文 claim，不是本機測量；論文的資料/實驗規模涉及 A800 80 GB，且資料條款偏 research-only/takedown，不能當可直接 ship 的自動 gate |
| [VITON-HD official repository](https://github.com/shadow2496/VITON-HD) | 官方 benchmark 有 11,647 train、2,032 test 的 frontal top pairs，1024x768，提供研究程式碼與資料入口 | CC BY-NC 4.0、研究用途；可作公開 benchmark 背景，不應將其資料混入私有或商業評估，也沒有必要為本次小 pilot 下載 |
| [DressCode official repository](https://github.com/aimagelab/dress-code) | 50K+ high-resolution pair、tops/bottoms/dresses、keypoints/skeleton/label/dense pose 等結構資產 | repository 明示資料不向 private companies 發放並有接受條款；不適合作為本工作區的預設資料來源 |

### 可重現與記憶體

| 來源 | 原始證據支持什麼 | 實作含義 |
| --- | --- | --- |
| [PyTorch reproducibility notes 2.8](https://docs.pytorch.org/docs/2.8/notes/randomness.html) | 同一 seed 不保證跨 release、commit、平台、CPU/GPU 完全重現；可設定 Python/NumPy/PyTorch seeds、deterministic algorithms、cuDNN benchmark 與 CUDA workspace；deterministic 通常較慢 | audit 必須鎖環境與裝置並記錄設定；seed 是控制變數，不是品質證明 |
| [Diffusers memory optimization v0.35.1](https://huggingface.co/docs/diffusers/v0.35.1/en/optimization/memory) | model CPU offload 通常比 sequential offload 快但省 RAM 較少；sequential 更省但可能極慢；VAE tiling 可降記憶體但可能產生 tile tone variation；效果依 pipeline 而異 | 既有 enable_model_cpu_offload/VAE tiling 是可行起點，需量測延遲、CPU RAM、VRAM 和畫質，不可只看 PyTorch allocator peak |

## Reference、candidate 與 identity 的資料契約

多模型公平比較的最小輸入應固定為四件事：

| 名稱 | 內容 | 可接受用途 |
| --- | --- | --- |
| person reference | 原始人物照片、身份/姿勢/皮膚/背景的來源 | 只可作人物 identity 和場景參照；不因 candidate 變好而自動替換 |
| garment reference | 原始服裝照片或 FASHN 使用的 garment input | 服裝類別、版型、顏色、紋理、logo 的參照；不可默認使用上一候選的裁切 |
| base image | 同一個 FASHN 輸出，或明確標註的已批准前一階段輸出 | 所有 backend 的同一起點；no-op 必須保留 |
| repair mask | 同一份人工確認 mask，另記 crop/padding/feather 參數 | 只授權指定區域；候選不可自行擴張成另一份 mask 而仍算公平比較 |

每一筆 job 需把 person bytes、garment bytes、base bytes、mask bytes 在正規化後 hash，並記錄 mode、尺寸、EXIF 處理、mask bbox、mask pixel count。candidate 只能引用這些 hash；若採 IP-Adapter，另記 reference crop 與 image embedding 的來源 hash。生成 candidate 不得回灌作下一 candidate 的 reference，除非使用者明確按下 adopt，且 lineage 由新 job 開始。

「保留遮罩外像素」和「保留人物身份/衣服參照」是不同性質的 invariant：前者可用 byte/pixel diff 驗證，後者只能用成對人工 rubric 和適用範圍內的輔助指標驗證。IP-Adapter 的 image feature conditioning 可能穩定方向或 appearance，但官方文件沒有承諾逐像素身份、文字或 logo 保真；錯誤的 reference 反而可能把錯誤鎖進結果。

## 建議的路由矩陣

這裡的「自動」只代表根據可觀察的適用條件選擇一條候選流程，不代表自動接受畫質。

| 輸入症狀/資格 | 建議路由 | 自動可做的範圍 | 不得做的事與失敗語義 |
| --- | --- | --- | --- |
| 使用者判定已經正確、無明確 defect | no-op/base | 記錄 zero-edit baseline，跳過生成 | 不可為了增加 candidate 而 reroll；若 scorer 認為更高也不能覆蓋 base |
| 小、平滑、無紋理語義的污點 | OpenCV Telea | 只在 mask 面積、形狀與人工類別 gate 通過時執行，做硬 composite | 不可用在手指、袖口、布料紋理、文字、logo；成功回傳仍 needs_review |
| 原始手可可靠保留，且 source/target 手框幾何 gate 通過 | source_hand_composite_v2 / restore-hands branch | 可做低風險像素合成與 exact outside-mask check | 不可把幾何匹配當成手指解剖正確；anatomical_correctness 必須仍是未驗證 |
| 手、拳頭、袖口或局部遮擋，人工 mask 明確 | 現有 SDXL inpaint candidate；合資格手部可另測 SD1.5/HandRefiner | 同一 base/mask 產生最多兩個候選，固定 crop、strength、prompt、seed protocol | HandRefiner 僅在手部足夠大且依賴可用時試驗；SDXL port 未經 HandRefiner 作者測試；不能自動選中或鏈式再生成 |
| tiny hand、低解析手、嚴重互相遮擋、mask 不確定 | manual review / request better mask | 只做 mask/尺寸/identity 的安全檢查 | 不以「多 seed」補救不適用的 mask；任一 backend 不合資格都應明示 manual_required |
| texture、logo、文字、細密圖案 | reference-aware candidate（SDXL + IP-Adapter 若合法且可用）或人工修補 | 只可產生並比較候選；reference hash 和 adapter/base compatibility 必須完整記錄 | 不得因 VTON-IQA 或相似度 scalar 高而 auto-accept；SDXL 既有 smoke 已有文字/符號 hallucination |
| 大範圍身形、衣物類別或 garment fit 錯誤 | 重新檢視 FASHN input/output 或人工處理 | 可回到明確的新 FASHN job，建立新 lineage | 不應用小 crop inpaint 假裝修正全身 garment geometry |
| 需要 FLUX Fill 或 gated/缺 checkpoint backend | explicit unavailable/manual | preflight 可以告知不支援並保留 base | 不可靜默改用另一 model、遠端 API 或上傳私有影像；未完成授權/資源審核前不下載 |

實際實作仍應遵守父 composite 草案的上限：每個 repair session 最多兩個 inference candidates，SD1.5、SDXL、IP-Adapter 不是各自再乘兩個 seed 的無界分支。若其中一個候選因 OOM、逾時、缺權重、非法 reference 或 validation 失敗，另一個候選可以保留並標註 partial；「一個成功」不等於「自動接受」。

## 最小、可在本機執行的評估 protocol

### 資料設計

先做 30-case pilot，再決定是否擴展既有研究文件建議的 60 pairs（40 development、20 holdout）。pilot 使用獲得同意且只保留本機的輸入，不下載 VITON-HD/DressCode，也不把私有影像上傳任何服務。每個 case 要有原始 person、garment、固定 base、人工 gold mask、缺陷標籤，以及「若可取得」的原始正確參照。

| strata | pilot 數量 | 必須覆蓋的反例 |
| --- | ---: | --- |
| fist/手部遮擋 | 6 | 握拳、手掌重疊衣物、手腕被袖口遮住；不要只用張開五指 |
| long-sleeve/cuff | 6 | 長袖與手腕邊界、袖口材質、手袖交界 |
| tiny hands | 6 | 影像中手小於約 60x60、遠距離、模糊或部分出框；60x60 是 HandRefiner 官方現有 SD1.5 權重建議，不是品質門檻 |
| texture/logo/text | 6 | 平面色塊、細紋、重複圖案、真實 logo/文字與相似非文字紋理 |
| already-correct/no-op | 6 | 無需修補、疑似缺陷但其實是合理姿勢、低信心 mask；這一組測 false repair |

以 person identity 與 garment identity 分割 development/holdout，不能同一人物或同一服裝只換 seed 就同時進兩組。最小 pilot 可用 20 dev、10 locked holdout；seed、prompt、crop 與候選順序由 dev 固定後，holdout 的人工 mask、結果和評論順序鎖定。若 30 cases 的置信區間過寬，只能報告探索性結果，不得把它升格為通用模型排名。

### 候選矩陣

每一 case 先產生一次固定的 base，之後至少有 no-op control。適用 subset 才執行 backend，避免把一個不可能解的 hand model 和 unrelated garment case 混成總分：

| 候選 | 目的 | 施加條件 |
| --- | --- | --- |
| C0 no-op | 檢驗原圖與 false repair | 不生成，保留 base |
| C1 Telea | 低成本 negative/control | 僅 smooth blemish subset |
| C2 現有 SDXL | 現有可用的 neural baseline | 同一 base、mask、crop、prompt family 與固定 checkpoint revision |
| C3 SD1.5/HandRefiner | 手部結構控制的受限對照 | 只在足夠大的 hand subset，依官方尺寸/control 建議；依賴和授權先通過 |
| C4 SDXL + IP-Adapter | reference-conditioning ablation | 只在 reference crop 清楚、adapter/base/encoder 相容且資源可用時；與 C2 成對比較 |

C2–C4 每 case 先用兩個預先宣布的 seed（例如 42、43），每個 candidate 的 prompt、negative prompt、steps、guidance、strength、scheduler、crop padding、precision、offload 模式全部固定。若現有 UI 的候選上限是兩個，則把 seed 變體也算入上限，不得「每模型兩 seed」後又自稱只有兩候選。

### 把 model effect 和 seed effect 分開

同一整數 seed 只代表「同 backend、同版本、同裝置設定下的控制變量」；SDXL 和 SD1.5 的 latent、解析度、scheduler 和前處理不同，不能把 seed 42 跨模型當作相同隨機樣本。

每個 subset 至少記錄下面四個對照：

1. **重現性：** 同一 backend、revision、input hash、mask hash、參數、seed，在兩個新鮮 process 各跑一次。output hash 若不同，標記 nonrepeatable，不拿來作自動 gate。
2. **模型效果：** 同一 base/mask/prompt/seed，配對比較 C2 與合資格的 C3 或 C4。兩者差異才可暫稱 backend effect，且只適用該 strata。
3. **seed 敏感度：** 同一 backend 和所有設定，只改 seed 42/43；記錄候選內 variance、最差結果與最好結果，不把 best-of-2 宣稱為模型本身改善。
4. **成本控制：** 可另報告 fixed-budget best-of-2，但要同時報告單次和兩次時間/VRAM/失敗率；best-of-2 是額外計算換來的選擇機會，不是單模型品質證據。

dev 階段若要估 seed variance，可擴到四個 seed，但不得在 locked holdout 中臨時尋找使某模型勝出的 seed。任何候選先與 C0 比較，再由盲評者選擇；不可把第一個好看的 seed 當成代表，也不可把多次 reroll 的最高分當成平均品質。

## 遮罩、結構與品質量測

### Hard safety metrics

每個候選都先做不可協商的 machine checks；這些 check 失敗時自動拒絕該候選是安全決策，但通過不能自動接受：

- 輸出可解碼為預期 PNG、mode/尺寸正確，且沒有 NaN、空檔或 stale output。
- 以正規化後 base 和 mask 計算遮罩外最大 channel diff、changed pixel count；目前流程目標是 max diff = 0。
- output 的 base、person、garment、mask、model revision、job lineage hash 完整對得上本次 job。
- mask 非空，bbox、面積比例、連通元件數、crop padding 在 strata 的預設範圍；超出範圍轉 manual_required。
- candidate 不得改寫原始 base；另存 candidate artifact 和 metadata，失敗清理暫存檔但留下可稽核的錯誤事件。

遮罩外 exactness 是目前 pipeline 的強項，卻會隱藏 seam 或遮罩內 hallucination；故必須分開報告 outside_mask_exact、inside_mask_changed 和視覺評分。

### Mask/pose 的結構檢查

人工 gold mask 是 pilot 的標準，不以 SAM2 自動 mask 取代。SAM2 官方 repository 支持 promptable segmentation 和 automatic mask generation，因此將來可用於提出 mask 候選；它只能說「哪裡可能是目標」，不能說「修補後手或 logo 正確」，也不能作 auto-accept。

若有正確原始參照，可計算 mask IoU、Dice、boundary F-score、手/袖口區域 coverage；若沒有 ground truth，則只報告 mask 面積、bbox、人工信心與 disagreement，不能偽造 IoU。pose/hand control 的輸入和輸出另記 keypoint/mesh 可用性、遮擋比例、手像素尺寸；ControlNet 或 mesh 通過不代表其構造與 person identity 相符。

### Human rubric

每個 case 將 person、garment、base、candidate 以隨機化且盲化的 candidate label 並排展示，至少兩位 reviewer 各自填寫 0/1 failure 和 1–5 ordinal 分數。評分欄位必須拆開，而非合併成一個「品質」：

| 欄位 | 問題 | 立即 severe failure 的例子 |
| --- | --- | --- |
| target repair | 指定 defect 是否合理改善 | defect 未改善，或把自然皺褶/合理拳頭當缺陷改掉 |
| person identity | 臉、膚色、身體比例、姿勢是否維持 | 人物變臉、膚色跳變、肢體或衣物邊界不可能 |
| hand/pose/occlusion | 手、腕、袖口和遮擋是否符合原姿勢 | 多手指、融手、漂浮手、袖口穿透；握拳不要求呈現五根手指 |
| garment fidelity | 類別、輪廓、長短、版型是否維持 | 上衣變裙、長袖變短袖、袖口消失或新增不合理結構 |
| texture/logo | 顏色、紋理、文字、logo 是否保留且不 hallucinate | 新增不可讀符號、偽 logo、重複圖樣、邊界粉色/色帶 |
| edge/lighting/background | 局部接縫是否合乎光照且未污染背景 | halo、硬接縫、背景改變、遮罩外雖未變但邊界明顯不自然 |
| accept decision | 是否值得取代 C0 | 只有 reviewer 明確 accept 才能進 accepted state |

若任一 reviewer 標記 severe failure，候選不得 auto-accept；兩位 reviewer 不一致就進 manual/arbitration。already-correct 案例中任何不必要變化都是 false repair，即使其他 scalar 或 aesthetic 分數升高也不能覆蓋原圖。

SSIM/LPIPS 只有在有對應正確參照時才作 paired diagnostic；FID/KID 是資料分布指標，VTBench 也特別指出它們不適合判斷單張 texture、size 或 hand-occlusion。VTON-IQA 可在合法、隔離、版本相容的環境試用，但 scalar 只作候選排序提示，不能取代上述 rubric。結果應按 strata 報告 repair_win_vs_noop、false_repair、severe_identity_failure、severe_garment_failure、manual_required、runtime_failure，並提供每一比例的 exact binomial 或 Wilson interval；n=30 時不要作寬泛排名。

## 資源、offloading 與可審計的成本

硬體是 RTX 4060 Ti 16 GB、32 GB RAM；目前文件已有 FASHN、SDXL、Telea 的實測，但沒有 SD1.5/HandRefiner、IP-Adapter 或 FLUX 的本機量測。下表刻意分離 fact 和 guess：

| 項目 | 狀態 | 評估規則 |
| --- | --- | --- |
| FASHN/SDXL/Telea 時間與 PyTorch allocator | B：來自本 repo verification | 作 baseline；每次仍記冷/暖狀態，不能跨不同 process 隨意比較 |
| SDXL offload | B：現有 scripts 使用 model CPU offload、VAE tiling；Diffusers 官方說 model offload 較快但省 RAM 少、sequential offload 更省但很慢 | 先確保 FASHN pipeline 釋放，再逐候選執行；記錄 offload 模式，不能同時併跑兩大模型 |
| FLUX Fill 權重 | C：12B × bfloat16 約 24 GB 的算術下界，尚未下載/量測，還未計 activation/overhead | 不是「16 GB 一定不能跑」的測量結論，但足以在未做資源審核前排除；不以 swap/offload 猜測 production 可行 |
| 外部 VRAM/RAM | C：PyTorch peak 不涵蓋所有 ONNX、driver、桌面、另一 process 或 CPU staging | audit 另取 nvidia-smi before/peak/after、RSS/CPU RAM、model load、inference、postprocess 時間 |
| batch/candidate 成本 | C：每 case 最多兩次 inference | 報告 single candidate 與 pair total；timeout/OOM 不能被平均掉 |

實驗執行採單 request、單 GPU、不同 backend 間完整 unload；不得為了測 throughput 而讓候選互相爭搶 16 GB。每個 candidate 建立 before/after/mask/job metadata bundle，並把 load time、inference time、postprocess time、GPU allocator peak、系統 GPU peak、CPU RAM peak 分欄。VAE tiling 降低記憶體可能帶來 tile tone variation，因此該選項也要作 paired visual check；offload 只改善資源路徑，不能被當成畫質改善。

### Deterministic audit manifest

每次 job 至少記錄：

- input 的正規化規則、mode、尺寸、person/garment/base/mask SHA-256；
- repo commit、model repository、完整 revision/commit hash、adapter/encoder revision、license acknowledgement；
- Python、Torch、CUDA、cuDNN、Diffusers、driver、GPU 名稱、精度、attention backend、scheduler；
- prompt、negative prompt、steps、guidance、strength、crop/padding/resize、ControlNet/IP-Adapter scale；
- seed、Python random/NumPy/Torch seed 狀態、deterministic algorithm 設定、cuDNN benchmark、CUDA workspace 變數；
- offload/tiling 設定、起止時間、分段 latency、VRAM/RAM peak、輸出 SHA-256；
- outside_mask_max_difference、changed pixel count、mask validation、status、reviewer decision、reject reason。

同一裝置和同一環境用同一輸入跑兩個新鮮 process；輸出 hash 相同才標為 repeatable。若 hash 不同，即使看起來一樣也保留兩份結果並標記 nonrepeatable；若只想比較視覺，可另記 image metric，但不能抹掉 reproducibility failure。依 PyTorch 官方說明，deterministic algorithms 也可能變慢，且跨版本/裝置無法保證 bitwise 一致；現有 scripts 的 CPU generator manual seed 是必要但不足的 audit。

## 明確的失敗語義

狀態機應把「候選成功生成」和「候選被採用」分開：

| 階段 | 失敗條件 | 結果 |
| --- | --- | --- |
| preflight | model revision、license、reference、mask、尺寸、VRAM 預估或參數不合格 | 不執行 inference；保留原始 base；manual_required/backend_unavailable，帶可行動 reason |
| inference | exception、OOM、timeout、process 被殺、空輸出 | 該 candidate 無效且不進比較；若另一候選完成，標 partial_candidates；不可靜默改 remote 或換 model |
| postprocess | decode/尺寸/hash 問題、遮罩外 max diff 非 0、stale lineage | 丟棄該 candidate；保留原始 base；記錄 validation failure；不把失敗候選送 reviewer 當正常結果 |
| review | candidate 有 hallucination、identity/garment/pose severe failure 或 reviewer disagreement | candidate rejected 或 manual_arbitration；原始 base 不變 |
| adopt | 使用者明確選 candidate | 建立下一個 lineage；重新 hash 和保存 user decision；不把同 session 的未選 candidate 當新 reference |
| all fail | 沒有合格 candidate，或使用者拒絕全部 | 完整 job 狀態為 needs_manual_review，原始 base 可下載/重試；不得回報 repair success |

逾時與 OOM 是可診斷的 backend failure，不是「改 seed 再試」的理由。對已知不合資格的 tiny hand、logo 或模糊 mask，應在 preflight 轉人工；對未知失敗，最多依明確的全局 candidate budget 重試一次，並在報告中分開 backend_retry 和 seed_reroll。任何自動 early exit 只能是硬安全拒絕，不能是品質接受。

## 實作前的獨立 adversarial review

以下是針對「同一輸入比較 SD1.5、SDXL、可選 IP-Adapter，讓人選下一步」提案的反例審查：

1. **模型提升其實可能只是 seed 提升。** 如果只展示新模型最漂亮的一張，無法區分 architecture、scheduler、prompt、crop 或 reroll。要求同一 base/mask/設定的 paired seeds、fresh-process reproducibility 和 C0 no-op；best-of-2 必須單獨標為付費的選擇策略。
2. **同一 seed 跨模型不等價。** seed 42 在 SD1.5 與 SDXL 不共享 latent/解析度/denoising trajectory；跨模型只可作固定控制值，不能解釋為同一隨機樣本。
3. **遮罩外 exactness 可能遮蔽內部災難。** hard composite 可以保證背景不變，但仍會留下 halo、手指融合、假文字、衣服邊界或不符合光照的 seam。故 exact diff 是 hard gate，human inside-mask rubric 才是品質 gate。
4. **結構 control 可能與服裝 fidelity 衝突。** VTON-HandFit 論文的消融與 control-strength 分析顯示，手部深度控制增強形狀/姿勢時可能犧牲皺褶和紋理；控制更強不等於整體更好。[VTON-HandFit paper](https://arxiv.org/html/2408.12340v2)
5. **HandRefiner 的資格容易被誤用。** 官方現有權重是 SD1.5、希望手約 60x60 以上，作者明示 SDXL port 未測試；tiny hand、長袖交界、出框拳頭不應自動送入此路由，也不能把它的 MIT repository 等同於所有外部資產的授權。
6. **reference branch 可能鎖定錯誤。** IP-Adapter 只注入 image feature；若 garment crop 有背景、錯衣服或低清 logo，候選可能更堅定地重現錯誤。reference hash、crop 與 reviewer identity/garment 分項不可省略。
7. **IQA 分數有 domain shift 和 benchmark leakage。** VTON-IQA 的作者自己限定 standard studio/controlled setting，且 scalar 沒有可解釋 attribute；VTBench/OpenVTON-Bench 的 paper 結果不代表私有照片或目前 FASHN 輸出的分布。任何未在本地 holdout 校準的 threshold 都不能 auto-accept。
8. **already-correct 是最容易被忽略的負對照。** 生成器幾乎總能改變一些像素或 aesthetic score；若沒有 no-op 組，route 可能獎勵無必要的修補。
9. **公開 benchmark 的授權不是資料可用性。** VITON-HD 是 CC BY-NC 4.0，DressCode 有不向 private companies 發放的條款，VTON-IQA code/model/dataset 是 CC BY-NC-SA 4.0；FLUX Fill 是 gated 且非商用。研究結果要和可 shipping 的 checkpoint/data/license 分欄保存。
10. **VRAM 估算不是運行證據。** allocator peak 可能漏掉 ONNX、driver、CPU staging 和桌面；16 GB 卡上 offload 可能跑得動但速度過慢，或 VAE tiling 導致色調差異。沒有 local measurement 就只能標 unknown，不能放進自動路由。
11. **偽成功容易被流程吞掉。** process return code 0、PNG 存檔與 outside_mask_max_difference=0 都只能代表 runtime/safety 通過。candidate 必須維持 needs_review，所有失敗、使用者拒絕和 partial candidate 必須能重建。

## Exact integration experiment

這是一個可以在實作前核准、且不需要擴張到整個 benchmark 的最小試驗：

### Phase 0：固定資料與 provenance

建立 30 個只存本機的 permissioned cases，按五個 strata 各 6 個，分成 20 dev 和 10 locked holdout；以 person/garment identity 分割。對 person、garment、base、manual gold mask 做正規化與 hash，記錄 case ID、缺陷類型、mask 信心、手尺寸和遮擋標籤；禁止任何 cloud/API upload。

### Phase 1：凍結 no-op 與基線

對每個 case 固定同一 FASHN base，保存 C0。記錄現有 FASHN 與 repair 的 revision、參數、seed、load/inference/VRAM/RAM；在已知 small smooth subset 執行 C1 Telea。若 base 已正確，直接將 C0 作 expected winner，不因測試需求生成修補。

### Phase 2：候選產生

在同一 base/mask/crop 上執行 C2 SDXL；手部 eligible subset 才執行 C3 SD1.5/HandRefiner；清楚 garment reference 且資源/授權通過才執行 C4 SDXL + IP-Adapter。每個 session 最多兩個 inference candidates，使用固定 seed 42、43；先完成 preflight，再逐一執行，不做候選 chaining。每個 candidate 立即通過 output/lineage/outside-mask checks，任何 hard failure 不進人工比較。

### Phase 3：blind paired review

將 C0 和成功候選隨機命名並排，至少兩 reviewer 依本報告七欄 rubric 評分。先記 severe failure，再記 repair、identity、garment、hand/pose、texture/logo、edge/lighting；reviewer 不知道 model、seed、candidate 先後。只要 reviewer disagreement，就進 arbitration，沒有 arbitration 就保持 manual_required。

### Phase 4：分解結果

對每個 backend/stratum/seed 輸出：

- C0 對照的 repair win、false repair、severe identity/garment/pose/texture failure；
- reviewer accept/reject/manual、agreement rate、runtime failure、timeout/OOM；
- outside-mask exact rate、mask validation rate、repeatability rate；
- median/p95 load、inference、total latency、VRAM/RAM peak；
- seed 42/43 的差異與 candidate count，明確標註 single、two-seed、best-of-2；
- exact binomial/Wilson interval 和未覆蓋 strata。

不要把不同 eligibility 的 backend 合成一個總分；先作 within-stratum paired comparison，再決定是否值得將 pilot 擴到既有 60-pair protocol。若 C2/C3/C4 沒有在 locked holdout 相對 C0 顯示人工可辨識的淨改善，保留 no-op/manual 路由，不為了「多模型」而加 fallback。

### Phase 5：實作 gate

只有以下條件都具備，才可實作 bounded router：

1. 所有 accepted candidate 都有完整 provenance、output hash 和 outside-mask max diff = 0。
2. backend 缺失、逾時、OOM、reference 不合法、mask 不合資格與 reviewer reject 都各有可重現的 failure reason。
3. holdout 以 paired human review 顯示某 route 在其適用 strata 比 C0 有淨改善，且結果同時揭露 seed variance、manual rate 和 severe regression；n=30 的探索性 interval 不得包裝成普遍品質保證。
4. 已確認實際 checkpoint 的 license/gating、下載來源、版本、16 GB GPU 的時間/RAM/VRAM 成本；任何未知項保持 manual-only。
5. 自動決策只限 eligibility、安全拒絕和 no-op/低風險 Telea；品質接受永遠要求明確使用者 adopt。

## 尚未知道、因此不可在文件外宣稱的事項

- FASHN 跨 photo identity、fist、long-sleeve/cuff、tiny hands、複雜 occlusion 和 logo 的實際成功率尚未量測。
- SDXL smoke 中已出現文字/符號與顏色 hallucination；尚不知道在固定手部/袖口 masks 和 garment references 下，C2 相對 C0 的淨人評結果。
- SD1.5/HandRefiner、IP-Adapter 分支尚未在此 4060 Ti 16 GB 環境量測；其 checkpoint、外部資產 license、Torch2.8 相容性和 latency 都是 unknown。
- VTON-IQA、VTBench、OpenVTON-Bench 的資料與 evaluator 沒有在本次下載或執行；論文 benchmark 數字不能代替 local calibration。
- SAM2 可作 segmentation proposal，但本機 hand/garment mask IoU 和人工修正成本尚未有數據。
- deterministic flag 能否讓每一個 pipeline 在目前 CUDA/driver/attention backend bitwise 重現尚未量測；按照 PyTorch 官方文件，跨版本/平台仍不可保證。
- 任何未經使用者明確同意的私有影像上傳、遠端 inference、gated checkpoint 使用，都不屬於本實驗。

## 原始來源清單（Firecrawl 取用：2026-09-14）

以下均為本次研究直接核對的 primary source；paper/repository 的作者數字仍應以本機重跑或合法取得資料後再升級證據等級。

1. [FASHN VTON 1.5 model card](https://huggingface.co/fashn-ai/fashn-vton-1.5)
2. [FASHN VTON 1.5 pinned README](https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/README.md)
3. [Stable Diffusion XL inpainting model card](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
4. [Diffusers inpainting v0.35.1](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/inpaint)
5. [Diffusers IP-Adapter v0.35.1](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/ip_adapter)
6. [HandRefiner official repository](https://github.com/wenquanlu/HandRefiner)
7. [FLUX.1-Fill-dev model card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
8. [VTBench paper](https://arxiv.org/html/2505.19571v1) and [VTBench repository](https://github.com/HUuxiaobin/VTBench)
9. [VTON-IQA paper](https://arxiv.org/html/2603.13057v1) and [VTON-IQA repository](https://github.com/litelightlite/VTON-IQA)
10. [OpenVTON-Bench paper](https://arxiv.org/html/2601.22725v1)
11. [VTON-HandFit paper](https://arxiv.org/html/2408.12340v2)
12. [VITON-HD official repository](https://github.com/shadow2496/VITON-HD)
13. [DressCode official repository](https://github.com/aimagelab/dress-code)
14. [PyTorch reproducibility notes v2.8](https://docs.pytorch.org/docs/2.8/notes/randomness.html)
15. [Diffusers memory optimization v0.35.1](https://huggingface.co/docs/diffusers/v0.35.1/en/optimization/memory)
16. [SAM2 official repository](https://github.com/facebookresearch/sam2)

**研究結論：** 在本機證據補齊以前，推薦「C0 no-op + 適用條件路由 + 最多兩個同 lineage candidates + 人工 adopt + 完整 audit」，不推薦「多模型輸出後由未校準 IQA 自動挑選」。這個 bounded protocol 可以直接檢驗 model effect、seed effect、mask/structure trade-off 和實際 16 GB 成本，同時保留原始輸入與可回復的失敗語義。
