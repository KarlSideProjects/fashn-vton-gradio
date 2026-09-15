# 多模型複合流程：主協調者研究與待審方案

日期：2026-09-14。這是設計研究，不是已完成的品質改善聲明。
Orca Run：`run_bca805876cf5`；三位研究 worker 使用已確認生效的 `gpt-5.6-luna / max`。
基準版本：`31a2551`。須先彙整各研究與獨立 adversarial review，再實作。

| Orca 工作 | Task／Dispatch | 已核對的結果 |
|---|---|---|
| VTON 模型研究，Luna max | `task_04456b2cbc95`／`ctx_5e3d5ac5bc97` | 已完成，見 [VTON 研究](model-research-vton.md)，worker 已釋放 |
| 修補模型研究，Luna max | `task_f2e935db58f8`／`ctx_a7706fc5f5b5` | 已完成，見 [修補研究](model-research-repair.md)，worker 已釋放 |
| 評估研究，Luna max | `task_43cc53ba073f`／`ctx_4304b97c0f0f` | 已完成，見 [評估研究](model-research-evaluation.md)，worker 已釋放 |
| 獨立實作前審查 | `task_1bc282267880`／`ctx_520764230d09` | 有條件 GO，7 項 P1；報告核對後才派實作，worker 已釋放 |
| 模型後端實作 | `task_c60ff97246c9`／`ctx_e40f256fce8a` | 實作中，尚未驗收 |
| 比較與採用流程 | `task_9973e00e2927`／`ctx_5014ed2db69b` | 實作中，尚未驗收 |

## 目前缺口

目前只有 FASHN → 原手幾何合成，以及手動 SDXL／OpenCV 修補；原手檢查失敗不代表
生成圖解剖一定錯誤，SDXL 成功退出也不代表缺陷已修好。前次測試已觀察到 SDXL 衣料補洞
新增文字與符號，故不能把增加重試次數視作改善模型能力。
程式依據：[Engine](../tryon/engine.py)、[修補](../tryon/repair.py)、[驗證紀錄](verification.md)。

## 查證的組合能力

1. Diffusers 的 inpainting 可接專用修補 checkpoint、ControlNet 與局部裁切；
   裁切提高局部輸入尺度，但無法還原原本不存在的資訊。最終遮罩外保護仍由本專案
   的 RGB 合成負責，不能假設擴散模型自己不動外部。
   [Diffusers 0.35.1 inpainting](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/inpaint)。
2. IP-Adapter 加入圖像條件，Plus 使用 patch embeddings，能與同底模的微調模型及
   ControlNet 組合。這是外觀參考，不是手部姿勢或文字精確複製。
   [官方模型卡](https://huggingface.co/h94/IP-Adapter)、
   [官方程式](https://github.com/tencent-ailab/IP-Adapter)。
3. 優先考慮 SDXL Plus ViT-H 而非 bigG：作者說明 H 版本較省記憶體；
   模型卡列 H encoder 約 632M、bigG 約 1845M 參數。不能只算 adapter 檔案大小而漏掉 encoder。
   官方 API 本日回應 `gated=false`、`license=apache-2.0`，revision
   `018e402774aeeddd60609b4ecdb7e298259dc729`；含
   `sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors` 與
   `models/image_encoder/model.safetensors`。
   [模型 API](https://huggingface.co/api/models/h94/IP-Adapter)。
4. 非正方形參考圖可能被 CLIP 中央裁切而丟失邊緣；必須明確預處理並保存實際參考圖，
   不應默默把完整商品照的中央當作使用者要參考的部位。
   [作者的非正方形說明](https://github.com/tencent-ailab/IP-Adapter#ip-adapter-for-non-square-images)。
5. `enable_model_cpu_offload` 是讓各模型元件依序上 GPU，不是所有模型同時共駐；
   adapter 要在設定 offload 前載入。可重用現有隔離修補程序，退出後釋放 CUDA context。
   [記憶體指南](https://huggingface.co/docs/diffusers/v0.35.1/en/optimization/memory#model-offloading)、
   [IP-Adapter 指南](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/ip_adapter)。

本地交叉檢查：已安裝 Diffusers 0.35.1 的 `StableDiffusionXLInpaintPipeline` 與
`StableDiffusionInpaintPipeline` 實際簽名都含 `mask_image`、`ip_adapter_image`、
`padding_mask_crop`。官方 IP-Adapter 文件的 inpainting 分頁範例卻使用
`AutoPipelineForImage2Image`，不能逐字照抄而忽略遮罩；實作應使用真正的 inpainting pipeline。
H encoder safetensors API 大小為 2,528,373,448 bytes（約 2.35 GiB），不是只有 100MB adapter。
這些研究證據本身只證明介面及資產存在；後續已完成 SDXL inpaint＋adapter 真實推論，
結果與資源限制另見 [驗證報告](model-improvement-verification.md)。

工程判斷：不同模型處理同一候選，讓人比較後再採用，比無條件串接多個生成器更容易
定位改善來源。結構控制若取自錯誤生成圖，可能鎖住錯誤；取原人物圖也可能與新袖口不相容。
因此這輪不把任一 ControlNet、分割信心或影像相似度當成自動品質裁判。

## 研究時提出的實作方向

下列為研究時的候選範圍；後續審查處置與定案見下節，實測結果見
[多模型驗證報告](model-improvement-verification.md)。不代表所有研究備選都已採用：

- 保留目前 FASHN 預設，加入上游確實支援的遮罩模式／CFG 調整，以固定 Seed 比較。
  不因有新模型就偷偷改預設底模，亦不把 CFG 更大描述為更正確。
- 局部修補可明確切換 SDXL、另一個專用 inpainting 底模，以及 SDXL＋參考圖 adapter。
  專用新底模待修補研究確認；依需求選擇而非用名稱推定品質。
- 提供有上限的同輸入多模型比較（最多兩次神經推論）。每張都以同一修前圖和遮罩生成，
  不自動將第一張的錯誤傳進第二張；人工採用後才進下一輪修補。
- 若提供參考條件，必須由使用者明確上傳合適部位；不默認用已扭曲的生成手當參考。
  參考缺失、權重未裝或版本不符，均明確報錯，不能換成沒有參考的 SDXL 卻標成已使用參考。
- 記錄實際 backend、checkpoint revision、參數、參考圖 hash、候選來源與每次嘗試結果。
  不回傳未通過遮罩外保護的候選；不覆寫原輸入。失敗不能讓上一輪圖被誤認成本輪成果。
- 新權重只在明確安裝步驟下載，使用 safetensors 與固定 revision；不引入雲端圖片服務，
  不把私人照片、權重或驗證產物提交 Git。

## 驗收底線

### 獨立審查處置：接受有條件 GO

已讀 [獨立反向審查](model-improvement-adversarial-review.md)。以下條件已納入實作，
工程測試、真實 GPU 案例、失敗與證據限制逐項記於 [驗證報告](model-improvement-verification.md)；
只有實驗性功能放行，不代表品質改善／預設推廣放行：

- P1-1：固定 Plus ViT-H 與正確 encoder；測 ControlNet 負值 hint、真實 inpaint 路徑。
- P1-2：完整必需檔案白名單、固定 revision、離線全檔 hash 核對；保留 SD1.5 checker 並拒絕被過濾結果。
- P1-3：所有神經方法卸載 FASHN、同鎖串行、每 worker 300 秒；分記模型／整卡／CPU 成本，不干擾外部程序。
- P1-4：等比例 resize＋padding、明確逆變換；reference 預覽、原圖及 processor 輸入 hash／用途。
- P1-5：同底圖同遮罩、最多兩次；partial／全失敗也留紀錄，不自動選優，不把 seed 當跨模型相同噪聲。
- P1-6：server session、job 世代、來源與輸出 hash 綁定；原子採用，拒絕跨 session、過期與重複選擇。
- P1-7：研究中的 SD1.5／HandRefiner 與推廣門檻矛盾以下列定案為準；真實負對照必須執行但不自動採用。

工程驗收與小樣本 AI 視覺檢視分開報告；不冒充雙人盲評。至少做手部、布料與正確區域，
並分別比較 SD1.5 ± ControlNet、SDXL ± reference；完整與漏選遮罩另列為不同試驗。
P2 的新模型訓練、HandRefiner／MANO、SAM、IQA、自動路由、FLUX／Qwen 與完整盲評延期，
不是本輪宣稱已實作的能力。實作由兩名專責 worker 分工，協調者負責整合與實機驗證。

### 首輪實作範圍（已完成獨立審查與實作）

首輪選擇 `sd15`、`sd15_control`（inpaint ControlNet）、`sdxl_reference`（Plus ViT-H），
保留現有 `sdxl`／`texture` 與預設值。這不是把 SD1.5 等同 HandRefiner：
不安裝 MANO／mesh，亦不對一般 SD1.5 套用 HandRefiner 的 60×60 資格限制。
SD1.5 的內容安全檢查保持啟用；被過濾的候選須明確失敗，不提供全黑圖假裝成功。

採人工選模型及同輸入兩模型比較，不實作自動品質評分或自動修補路由。
30／60 案例與雙人盲評是日後品質提升／預設推廣的門檻，不是開始開發實驗性比較工具
的先決條件；本輪不聲稱通用改善。先完成既有固定案例與負對照的實際 GPU 探索。
FASHN 僅增加上游 CFG／segmentation-free 設定，不更改預設，也不微調權重。

每個新增權重資產須固定 revision、離線載入、執行前核對 manifest 檔案雜湊。
參考圖明確上傳並以保持長寬比的方形 padding 處理，保存實際 encoder 輸入及 hash。
ControlNet 的遮罩內 hint 設為 -1，不能聲稱因此消除主 inpainting latent 的錯誤。
所有比較均保留原圖、不預選候選；採用必須核對目前底圖／遮罩與候選來源相符，
並防止跨 session 或舊結果被採用。

### 候選介面與失敗語意（已定案）

保留 `sdxl`／`texture` 方法名稱，新神經方法使用明確名稱，例如 `sd15`／`sdxl_reference`；
只允許程式白名單中的 checkpoint，不接受任意遠端 URL、任意 Python 程式或使用者指定模型路徑。
單次修補不自動選模型；另設「比較兩個模型」動作，以同一份正規化 RGB 圖與遮罩依序執行。
相同 Seed 只是固定各模型各自的隨機性，不代表不同架構／解析度抽到相同噪聲。

比較前驗證兩個選項、所需參考圖與本地權重。設定錯誤時不啟動推論；執行期一個候選失敗時，
保留另一個成功候選並明列失敗方法／原因，不能拿舊候選補位或稱完整比較成功。
每個神經子程序最多 300 秒、每次比較最多兩個；總成本上限與進度需在介面說明。
比較結果不預先選優勝者；使用者選擇後可下載，再明確「採用」才替換編輯底圖。

候選依 session 保存，不以不受信任的瀏覽器路徑讀取伺服器任意檔案；
每個結果保存 base/mask/reference hash、實際參數與模型識別，比較紀錄連結各候選及失敗。
採用後的下一輪 base hash 能追溯到前一輪輸出；不宣稱 hash 等同人工品質標註。

功能：單模型與多模型使用同一驗證／合成路徑；不相容 backend、參考缺失、模型未裝、
部分失敗、逾時、重複請求與重設選區有測試。既有自動載入與兩次嚴格原手拒絕行為不能意外失效。

執行：至少真實載入／執行新增神經路徑，逐像素檢查遮罩外不變；記錄記憶體與耗時。
本機目前 16GB GPU 已有其他程序使用約 6.2GB，不得終止非本任務程序來製造測試通過。

品質：以同一修前圖、遮罩、提示與 Seed 比較不同模型；保留未修補基準。
包含簡單衣料缺陷、手部／袖口，以及本來正確的區域，人工記錄「更好／不變／更差」。
使用者的握拳長袖圖可作案例，但不得據此調全域門檻或宣稱跨照片泛化。
如果新路徑沒有實測改善，明確標實驗，不以程式測試通過代替品質結論。

完整跨人物／商品的盲評標準沿用 [既有研究](hand-correction-research.md)，
本輪小樣本只能支持局部工程可行性，不能支持通用改善率。

## 本輪未修改程式前的基準

- 35 項 unittest 通過，日誌 `.cache/model-improvement-baseline-tests.log`。
- 使用者原人物／長袖商品，30 steps、Seed 42、`preserve_hands=False` 真實執行完成，
  485×864、推論約 30.434 秒；結果 `outputs/088d600a4cd1437086a489f5b26c649c.png`，
  同名 JSON 記錄實際參數。日誌 `.cache/model-improvement-user-baseline.log`。
- 人工檢視：長袖已生成，但胸前手部與原握拳形狀不同；小手細節不足以由全圖判定正確。
  新衣袖遮住原手錶／部分手腕，不能把貼回整段原前臂當成正確修復。
  此圖只作固定案例，不用來調整全域手部拒絕門檻。照片與產物均保留本機、Git 忽略。
- 現有 SDXL 在相同案例胸前手部人工遮罩、strength 0.85、Seed 42 的修補約 21.476 秒，
  檔案組 `outputs/repairs/1d74a2b449e94987a241bc7a5ed9cb48/` 保存修前、遮罩與結果。
  提示要求自然男性握拳、帽繩與藍袖口。人工檢視拳頭輪廓較完整，但袖口出現深色邊緣，
  尚不能算完整修復。後續模型應使用這份完全相同的修前圖／遮罩作對照，不另挑較容易案例。
- 同一 SDXL 輸入／參數於新程序重跑約 20.092 秒，輸出 pixel hash 均為
  `73d2c6a5b00e725ffe96a25e5cd91412e7df8378e07a70ba1a7035edb1619843`。
  重跑組 `outputs/repairs/3b6ebc789c274fcab3e1a6f4ab46d323/`；僅此配對證實可重現，
  不外推不同模型、裝置或版本，也不表示內容正確。

## 本機固定探索案例

產物根目錄 `outputs/experiments/cases-c421e39393c44388856c2d903f153e38/`（Git 忽略）。
每案都有 `before.png`、二值 `mask.png`、固定 `job.json`、`reference.png` 與來源 `case.json`。

- `hand`：沿用上述使用者握拳案例，strength 0.85、Seed 42，不重新生成底圖。
- `hand_incomplete_mask`：同底圖，遮罩侵蝕 2 px；故意漏選的反例，不與完整遮罩混算。
- `fabric`：沿用官方範例的小污點底圖與遮罩，strength 0.99、Seed 42。
- `clean`：相同官方範例位置，使用加入污點前的試穿結果；刻意修補以觀察不必要改動。

手部參考為原人物 `(480,760,568,838)` crop，仍有少量舊袖與手錶邊緣；
這是需要檢視是否被帶回的風險，不是正確姿勢控制。布料參考為官方商品
`(205,535,365,655)` crop，只含黑色布料。來源照片及測試產物不上傳。
已完成本協調者 AI 影像檢視，不是人類盲評；逐案結果、失敗與資源成本見
[多模型驗證報告](model-improvement-verification.md)。
