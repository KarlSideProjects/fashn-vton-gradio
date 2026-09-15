# 多模型複合修補：獨立實作前反向審查

審查／外部來源取回日期：2026-09-14。這是實驗方案審查，不是生產品質認證。
範圍：FASHN CFG／segmentation-free 選項、`sd15`、`sd15_control`、`sdxl_reference`，保留 `sdxl`／`texture`，同輸入兩候選及人工採用。
受審主稿：[composite](model-research-composite.md)，SHA-256 `babace460e119babc0de3b2079ac285bd73bdd8d1fcfa612e1bc54159bfaec42`；並讀取 [VTON](model-research-vton.md)、[repair](model-research-repair.md)、[evaluation](model-research-evaluation.md)、[既有驗證](verification.md)、[前次研究](hand-correction-research.md)。
程式基線為 `31a2551`；協調者回報後續 `c7c7db6` 僅改 README。本審查沒有安裝、模型下載、GPU 工作、私人圖上傳或程式修改。

## 判定：有條件 GO，僅限人工操作的實驗功能

方案有真正不同的底模與參考條件，不只是改名稱或重複 seed；共用既有裁切／硬合成／隔離 worker 即可開始實作。**以下 P1 必須納入實作與驗收，才能將新路徑標為可用；不要求先完成 30／60 案例才能寫程式。** 若省略條件、靜默降級或把執行成功宣稱為修好，該版本為 NO-GO。

未發現主稿中已確立且無法隔離的 P0 阻擋；不要為了審查形式製造 P0。以下是具體漏項、文件矛盾及必要驗收，不表示尚未實作的功能已存在漏洞。
保留原預設、人工 no-op／採用、最多兩次依序推論、沒有自動品質裁判，均合理。廣泛改善／預設推廣仍須跨人物與商品的保留集證據。

## P1-1：固定相容組合；不可照抄研究中的 adapter 範例

**事實：** repair 研究的 IP 範例載入 `ip-adapter_sdxl.bin`，不是主稿定案的 Plus ViT-H safetensors；其「ViT-H encoder 約 3.436 GiB」也不是選定檔案大小。[固定 H encoder 檔案 API](https://huggingface.co/api/models/h94/IP-Adapter/tree/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder) 列 `model.safetensors` 為 2,528,373,448 bytes，約 2.35 GiB，這是磁碟大小。
**必改：** `sdxl_reference` 固定 `sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors`，encoder 明確指定同 snapshot 的 `models/image_encoder`，或明確建構該 encoder；不能只沿用 `image_encoder_folder="image_encoder"`。
[Diffusers 0.35.1 loader](https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/src/diffusers/loaders/ip_adapter.py) 會將沒有 `/` 的 encoder 目錄接在 adapter subfolder 下，這會走到 `sdxl_models/image_encoder`；[H config](https://huggingface.co/h94/IP-Adapter/raw/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder/config.json) hidden size 是 1280，而[另一 encoder config](https://huggingface.co/h94/IP-Adapter/raw/018e402774aeeddd60609b4ecdb7e298259dc729/sdxl_models/image_encoder/config.json) 是 1664。
用真正的 SDXL inpainting pipeline 接 `mask_image`；adapter／encoder 在 CPU offload hooks 設定前載入。驗收須實際跑過 Plus 分支，記錄 encoder 路徑／config 與 adapter 名稱；缺失或錯配須失敗，不能回退無參考 SDXL。

**避免錯誤反駁：** 選定 [SD1.5 inpaint UNet](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/raw/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb/unet/config.json) 是 9 通道、[SD1.5 ControlNet](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/raw/c96e03a807e64135568ba8aecb66b3a306ec73bd/config.json) 是 4 通道，但兩者 cross-attention 都是 768。
[0.35.1 官方 pipeline](https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint.py) 先讓 ControlNet 處理 latent，再於 UNet 前串接 mask／masked-image latent，故此通道差不是拒絕理由。這仍不是 SDXL／FASHN 可直接載 SD1.5 ControlNet 的證據，也不是本機品質證據。
`sd15_control` 的 hint 必須是同一 transformed RGB／mask 產生的 NCHW tensor，遮罩內為 -1、外部為 [0,1]；不可轉成會截斷負值的 PIL／uint8。驗收小 tensor 的形狀、值域及遮罩座標。[官方 hint 範例](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/blob/c96e03a807e64135568ba8aecb66b3a306ec73bd/README.md)

## P1-2：離線資產契約須涵蓋完整 pipeline，而非只有 adapter

**事實：** 本次 Firecrawl 核對以下固定 revision API，皆為公開、`gated=false`，並列有所需 safetensors；只核對 metadata／設定，沒有下載或驗證權重 bytes。

| 資產 | 固定 revision／存取證據 | 已核對的授權標示 |
|---|---|---|
| SD1.5 inpaint，含 safety checker、feature extractor | [`8a4288a76071f7280aedbdb3253bdb9e9d5d84bb`](https://huggingface.co/api/models/stable-diffusion-v1-5/stable-diffusion-inpainting/revision/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb) | CreativeML OpenRAIL-M |
| SD1.5 inpaint ControlNet | [`c96e03a807e64135568ba8aecb66b3a306ec73bd`](https://huggingface.co/api/models/lllyasviel/control_v11p_sd15_inpaint/revision/c96e03a807e64135568ba8aecb66b3a306ec73bd) | metadata openrail；正文連 OpenRAIL-M |
| IP-Adapter Plus／H encoder | [`018e402774aeeddd60609b4ecdb7e298259dc729`](https://huggingface.co/api/models/h94/IP-Adapter/revision/018e402774aeeddd60609b4ecdb7e298259dc729) | repository metadata Apache-2.0；不取代 SDXL 底模條款 |

**現有缺口：** [scripts/inpaint.py](../scripts/inpaint.py) 下載時寫 SHA-256，推論只比 revision 字串；僅沿用此流程並未做到主稿要求的執行前核驗。
**必改：** 安裝白名單列出每條路徑真正會載入的權重與 configs／tokenizers／scheduler／processor；檔案完整後才發布 manifest。推論先檢查 model ID、固定 revision、必需檔案及內容 hash；禁止空 manifest 被當作成功，也不可只核對 manifest 剛好列出的少數檔案。
所有 loader 使用已核驗本機目錄、`local_files_only=True`／離線模式、safetensors；現有 `*.fp16.safetensors` 下載模式不會涵蓋無 fp16 後綴的 H encoder／Plus 權重，必須按資產指定。離線 smoke 不能偷偷從其他 cache 補齊漏檔。
安裝時自算 hash 能偵測後續損壞，不能單獨證明來源真偽；記錄固定來源，對有官方 LFS SHA-256 的檔案可比對它，不能以 mutable `main` 充當 pin。權重可取用與授權允許是不同問題；這張表不構成完整商用認證。
**驗收：** 缺一個 encoder／checker／config、竄改一個檔案、revision 不符，都在 CUDA 推論前停止；SD1.5 checker 保持啟用，讀取 `nsfw_content_detected`，被過濾候選不能以黑圖成功發布。不得以黑像素比例猜測是否觸發 checker。

## P1-3：所有神經方法都要卸載 FASHN，逾時與峰值須量對

**現有缺口：** [Engine.repair](../tryon/engine.py) 只在 `method == 'sdxl'` 卸載 FASHN；新增三種名稱若不改此條件，會把 FASHN／ONNX 留在 GPU。全程仍須共用 Engine lock 與 UI GPU queue，而非每個新按鈕各有一把鎖。
**必改：** 每個神經候選前釋放本任務的 FASHN 管線；依序執行獨立 worker，確認前一個退出及逾時清理後才開始下一個，不終止外部 GPU 程序。維持每 worker 300 秒及每次比較最多兩個；UI 的成本說明要分開排隊、preflight、最多 600 秒的 worker 預算與合成時間，不稱整個請求必定 600 秒內結束。
**證據界線：** 舊 SDXL 約 22–24 秒／5.365 GiB allocated 是既有紀錄；SD1.5／ControlNet／Plus 的 4060 Ti 16GB、32GB RAM 峰值與延遲目前未知。模型檔加總不是 VRAM 下界或保證，offload 不會移除 host RAM 成本。
現有 infer 在載入後才 reset peak，不能用它量整體峰值；新驗證須覆蓋載入、encoder、denoise、decode，分報 PyTorch allocated／reserved、整卡採樣峰值、CPU RSS、冷載入與總耗時，並註明採樣可能漏掉瞬時峰值。
**驗收：** 載入 FASHN 後依序執行三條新方法；OOM／timeout 後下一個合法請求可執行且無本任務殘留 worker。測試失敗可保持該分支 unavailable，不可藉「可能省顯存」標成可用。

## P1-4：共用座標契約；reference 外觀不能當成姿勢真值

**現況：** [prepare_repair／crop_box／composite_patch](../tryon/repair.py) 已有 64 px context 與硬遮罩貼回；[infer](../scripts/inpaint.py) 各邊取 8 倍數並設最小 64，極窄 crop 會改變長寬比。
**必改：** 同一 job 只計算一次 RGB 正規化、授權 mask、原座標 crop；SD1.5／SDXL 可有 512／1024 工作尺寸，但須記錄比例、padding、插值及逆變換。不能上游 crop 後再開 pipeline 自動 crop，或讓 control／mask 各自中央裁切。
對極窄區域優先等比例 resize＋padding；若保留現行扭曲，必須明記限制並排除姿勢保真聲稱。mask 用 nearest；工作 padding 不能擴張原圖授權區域；貼回後以原始正規化 base 做 exact outside-mask 比對。
參考圖由使用者明確提供並預覽方形補邊結果，記錄原 reference hash、用途（皮膚／手部外觀或商品材質）、padding／resize。若稱「實際 encoder 輸入」，還須記錄 processor 設定及處理後 tensor／pixels 的尺寸與 hash，不能拿上傳檔 hash 冒充。
[IP-Adapter 文件](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/ip_adapter) 支持外觀條件，不支持逐像素身份／文字保真。原人物手與新袖口可能遮擋不相容；增加 reference scale 可能帶回舊袖、手錶、背景或別種拳頭姿態，這是待測假說。`sd15_control` 更只是上下文 inpaint hint，不是手部 pose／mesh 約束。
**驗收：** 非正方形／邊界 crop、EXIF 旋轉輸入、細長 mask、細筆畫及斷開選區均對齊；比較原尺寸全圖與放大局部。手小於 60×60 不能因此禁止一般 SD1.5；HandRefiner 的資格不適用本輪。

## P1-5：比較紀錄要能容納失敗；seed 不能消除混雜變因

**現有缺口：** run_repair 只在成功時發布 bundle；timeout／例外的 temporary directory 會刪除。比較層必須另留一次 job 的完整嘗試紀錄，否則「記錄每次失敗」只是規格文字。
**必改：** 兩個不同白名單選項先完成 preflight；開始時清空舊候選。固定 base／mask／prompt 與每 backend 的明確參數，不把 A 的結果送 B，不以 seed reroll 代替第二模型。第一個失敗後可繼續第二個，但每個 slot 至多一次推論，不額外加隱藏 fallback。
紀錄 base／mask／reference、crop、實際 scheduler／steps／CFG／strength／control 或 IP scale、版本、seed、狀態與耗時；成功 output hash 和失敗 reason 均掛同 job。全失敗也留小型 metadata，不發布偽結果；被安全過濾的原始生成圖不應保存為可下載候選。
**驗收：** A 失敗 B 成功、A 成功 B 失敗、全失敗、preflight 錯誤、尺寸錯誤各有結果；partial 明列缺哪個方法，不以舊圖補位。metadata 不得繼續把非 `sdxl` 全寫成 texture／model=null。
同一 seed 跨架構不是相同噪聲；SD1.5 對 SDXL 是整套 pipeline 比較，含解析度／scheduler 等差異。要聲稱 ControlNet 或 reference 有貢獻，分別比較 SD1.5 ± ControlNet、SDXL ± Plus，固定該對其餘設定，不能只拿 SD1.5-ControlNet 對 SDXL-reference 推論單一條件效果。

## P1-6：採用須綁定 server session 與請求世代

**現有缺口：** [app.py](../app.py) 的 reuse 把 repaired 圖直接交給 `load_repair_image`，沒有 candidate provenance；新多候選不能只延長這條連線。`api_name=False`／隱藏按鈕也不是候選所有權檢查。
**必改：** server-side session 保存不可由瀏覽器任意指定的 candidate ID → 已驗證 artifact 映射；採用時核對所屬 session、最新 job 世代、base／mask／reference hash、output hash 與 candidate 狀態。hash 是內容識別，不能替代 session 所有權。
編輯底圖、重設／重畫選區、換 reference、新請求都使舊選擇失效；即使 reset 後 bytes 恰好相同，也不能由較晚完成的舊請求蓋回新候選。採用時原子檢查當前世代並清空舊 mask／候選；重複點擊要明確拒絕或無副作用。
下載／採用只解析該 session 已記錄的輸出，不接受任意伺服器路徑、另一 session 的 UUID 或竄改圖片作候選。這是本功能資料邊界要求，不要求擴建登入／權限平台。
**驗收：** 兩 session 交叉 candidate ID、舊 request 延遲完成、編輯期間按採用、重複點擊、路徑穿越／外部路徑，均不讀取或採用不屬於目前 job 的檔案。

## P1-7：修正研究矛盾，避免沒有負對照的「品質驗收」

evaluation 的 C3／Phase 2 混寫「SD1.5/HandRefiner」，又把 tiny hand／logo 當 preflight 不合資格；這不應套在人工選模型的本輪。repair／VTON 文件要求 holdout 改善後才加入 UI，也與主稿「先實作實驗工具」不同。**以主稿明確範圍為準，實作說明須標出哪些研究建議延期，不能同時宣稱都已遵循。**
evaluation 一面要求 clean cases 測 false repair，一面在 Phase 1 要求 clean case 不生成；後者無法測生成器誤修率。一般使用仍推薦 no-op；在離線評估中，需對已知正確區域刻意跑候選、保留原圖而不自動採用，才能觀察回退。
「像素有變」不等於「品質變差」；clean case 分開記 changed pixels 與人工判定新增缺陷，不能把無害變化全部算重大失敗。未知品質不是 preflight 拒絕的客觀依據。
本輪至少真實看同一固定手部／袖口案例、簡單布料缺陷及一個原本正確區域：修前／修後全圖、原尺寸局部、人物／商品參考一起看，逐一記目標改善、姿態／遮擋、衣料／假字、接縫與新增缺陷。不能用單元測試、圖片存檔或 exit code 替代。
初次探索允許一位具名評閱者，但須說明人數；AI 影像檢視不得冒充兩位人類盲評。一般可用標示需要真實 GPU 功能證據；品質改善與預設推廣另需保留集及獨立人評。

## 最小整合實驗與放行規則

1. **開始實作前：** 將上述條件列入實作清單；以白名單映射擴充既有 worker，不新建通用插件／訓練框架。FASHN 只暴露 upstream 已支援且驗證有限數值／布林型別的 CFG／segmentation-free，預設與原手兩次拒絕語義不變。
2. **CPU／無權重檢查：** 測 hint 值域、crop 逆變換、manifest 漏檔／損壞、checker 拒絕、兩模型 partial／全失敗、逾時清理、跨 session／stale adoption；mock 外部模型只證明流程。
3. **同資料 GPU 探索：** 沿用主稿 `outputs/repairs/1d74a2b449e94987a241bc7a5ed9cb48/` 的修前圖／mask，不另挑容易案例；另設布料缺陷與 clean control。每次 pair 最多兩個候選，一次固定 seed 42；seed 43 是另外宣告的成本，不是暗中 best-of-many。
4. **可歸因的配對：** 手／布料各比較 SD1.5 對 SD1.5-ControlNet，以及 SDXL 對 SDXL-Plus；每對固定 crop、prompt、strength、steps／CFG／scheduler，reference 只用於 Plus。SDXL 對 SD1.5 可另報整套系統取捨。完整 mask 與漏一圈 mask 另作反例，不混成同輸入結果。
5. **成本與視覺：** 所有新路徑至少真正離線載入與執行一次，記整體峰值／時間、候選 hash、outside-mask=0、實際參考輸入，以及更好／不變／更差和具體缺陷；圖僅存本機。未執行分支保持 unavailable，不能以 UI 選項存在宣稱完成。
6. **放行：** 工程不變量通過而畫質尚無改善，可放行明確標示的實驗功能；不得稱「已修好」或提升預設。兩種真實底模與 reference 分支未跑通，就只能報部分完成，不能將模式切換／新 seed 當成果替代。高階自動路由與通用品質承諾維持 NO-GO。

## P2：可延期，不能拖成首輪必要工程

HandRefiner／MANO、SDXL depth ControlNet、SAM、自動 IQA、FASHN LoRA、FLUX／Qwen、更多 VTON 底模、參考 embedding cache、完整 30／60 案例及雙人盲評平台均可延期。先量測有限本地案例，不為研究備選模型的 gating／依賴建立通用下載器；若之後納入，再獨立核對其實際權重與授權。
完整 benchmark 可用試算表與本地圖片，不需先寫評分服務；但擴大品質宣稱前不能略過它。repair 研究的 SDXL fp16 UNet URL 少了 revision 中的 `d04`；完整值應為 `115134f363124c53c7d878647567d04daf26e41e`（目前程式值正確）。這是引用筆誤，不是權重已損壞的證據；文案清理不能算模型能力提升。

## 審查證據與限制

已使用 codebase-memory Tier 2，確認 project `home-karl-Workspace-KarlSpace-fashn-vton-gradio`，generation `2026-09-14T15:04:41Z`；search 10 筆無後頁、run_repair 雙向 depth=2 trace 無截斷，並讀 exact snippet 及 app／engine／repair／core／inpaint 原始碼。
上述程式與六份研究／驗證文件 coverage 皆 metadata_match、無記錄缺口；這不是完整性證明。requirements-repair.txt 為 not_tracked，已全讀；探測到不存在的 scripts/download_repair.py 未作證據，實際下載函式在 scripts/inpaint.py。未做全 repo 安全稽核。
外部核對使用 Firecrawl MCP，限官方 Diffusers 0.35.1 原始碼／文件與選定 HF snapshot 的 API／config；所有連結均於 2026-09-14 取回。權重可及性僅到 API／檔案條目層，不代表本次成功下載；模型品質與新分支本機資源數據均仍未知。
