# 通用局部校正研究：VTON 與手部修復

研究日期：2026-09-14。範圍已依使用者補充，從手部擴大為袖口、衣物細節、背景與其他局部生成缺陷。
本次只研究與寫文件：沒有安裝新推論模型、修改生成程式、調整門檻或上傳使用者照片。
以下「建議」是待驗證的工程方案，不是已完成整合或改善率保證。

## 結論：先做通用局部重繪，不再以手部門檻充當品質判定

建議以「人工可修正的遮罩 + 局部裁切重繪 + 遮罩外像素鎖定 + 比較驗收」作共同核心；
手部姿勢／深度控制只是可選條件。先驗證在正確人工遮罩下能否修好，再加入自動選區，
才能分清失敗來自定位、生成還是合成，不會把所有問題都當成手部偵測問題。

通用方法已存在：Diffusers 提供遮罩式 inpainting；Inpaint Anything 展示了
SAM 選區後接 LaMa 移除／補洞，或接 Stable Diffusion 文字引導填補／替換的流程。
這證明可共用局部編輯流程，但不是證明能自動辨識所有生成缺陷。
[Diffusers](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint)、
[Inpaint Anything 官方實作](https://github.com/geekyutao/Inpaint-Anything)。

## 通用工具比較與本機優先順序

| 方案 | 能處理什麼 | 主要限制與取捨 | 建議 |
|---|---|---|---|
| SDXL Inpainting + 裁切選區 | 以文字和遮罩重畫袖口、皮膚、衣物局部等 | 仍可能生成人體錯誤或錯字；需新權重與整合測試 | 第一個通用實驗基準 |
| LaMa | 移除背景雜物、補洞與周邊紋理延伸 | 沒有以文字／手部結構指定正確解剖的能力，不能當補手保證 | 只有去除／背景修補需求才加入 |
| FLUX.1 Fill dev | 文字引導的遮罩填補 | 12B、存取需接受條款；非填補區也可能色偏、邊界可能出線 | SDXL 實測不足後再比較 |
| Inpaint Anything | 既有選區→移除／填補／替換流程範例 | 舊 main 與 main_2026 的依賴不同；新版標示 beta | 借用流程與獨立試驗，不整包塞入現有環境 |

SDXL 官方 checkpoint 以 1024×1024 訓練，model card 明列人物／文字限制與
`strength=1` 的品質問題，不能假設重繪越強越好；model card 標示 CreativeML Open RAIL++-M。
[SDXL Inpainting model card](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)。

LaMa 官方提供影像與遮罩推論，安裝範例仍含 Torch 1.8；
本 repo 固定 Torch 2.8，因此原始安裝指令不宜直接套進現有 venv。
[LaMa](https://github.com/advimman/lama)、[本機依賴](../requirements.txt)。

FLUX Fill model card 標示 12B 與 FLUX dev Non-Commercial License；
模型使用條款與輸出用途要分開確認，不能只看「open weights」。
僅以 12B×2 bytes 粗估 BF16 權重就約 24 GB（十進位，還不含其他模組及運算記憶體），
故不可假設原生 BF16 全 GPU 常駐能放入 16GB；量化／offload 的品質與耗時仍需實測。
[FLUX Fill 官方 model card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)。

Inpaint Anything 的 main_2026 分支標示 beta，列出 SAM3、SDXL 與可選 FLUX Fill；
這不是本專案已驗證的相容組合。[官方分支說明](https://github.com/geekyutao/Inpaint-Anything#-new-main_2026-branch--modernized-stack--robotics-support-beta)。

## 通用校正流程（建議，尚未實作）

```text
原始試穿候選
  → 問題選區：人工筆刷／點選，之後才加自動建議
  → 決定保留內容、允許變更區域及參考來源
  → 帶上下文裁切 → 局部重繪（必要時加參考／姿勢／深度）
  → 僅貼回允許區域 → 比較品質與遮罩外像素
  → 接受 / 撤回 / 有限重試 / 交人工處理
```

1. **選區不等於判錯。** SAM 可協助描邊，但不能告訴我們袖口或手指是否合理。
   先提供可增減的遮罩、原結果預覽及撤回；偵測失敗不應阻止人手選區修補。
2. **兩種範圍分開。** 上下文裁切可含周圍袖口／身體／背景；
   真正允許改動的遮罩另存。修一根多餘手指時，也要覆蓋它留下的影像殘跡，
   但不能把整張圖當修補區。邊界融合帶包含在明確的允許範圍內。
3. **小區域先裁切放大。** Diffusers 的 `padding_mask_crop` 提供同區域裁切、
   放大修補與貼回流程；具體 pipeline／版本是否支援要在原型時核對。
   放大能分配更多推論像素，不會憑空恢復真實細節。
   [官方裁切說明](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint#padding-mask-crop)。
4. **參考內容依問題選擇。** 修手以人物原圖中可靠的可見部分為參考；
   修新衣服以商品圖為參考，不能把舊衣服紋理貼回去。IP-Adapter 可提供影像條件，
   ControlNet 可提供額外結構條件；權重須與所選底模相容，不能混用 SD1.5／SDXL／FASHN。
   這些條件不是像素複製或正確性的保證。
   [IP-Adapter 官方說明](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter)、
   [ControlNet inpainting](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint#controlnet)。
5. **合成層保證局部性。** 官方文件指出 inpaint 模型仍可能改動未遮罩處，
   可用 `apply_overlay` 保留那些區域，但有接縫取捨。
   本方案要求最終 PNG 在允許區域外與「修復前候選」逐像素相等；
   不是與人物原圖相等，否則會撤銷換衣。全圖重新縮放或 JPEG 重壓縮不能混入此檢查。
   [官方保留區域說明](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint)。
6. **修復可失敗。** 設定候選數／時間上限，不無限換 seed；
   不接受時保留原候選與原因作診斷，不能把未通過的版本標成「已校正」。
   預覽／人工接受是待新增產品行為，不是現有程式已提供。

### 不同缺陷的處理分流

| 現象 | 優先處理 | 不應做的事 |
|---|---|---|
| 遮罩邊界錯誤 | 人工修遮罩；必要時增加點選分割 | 一律提高全域容錯率 |
| 袖口／手腕銜接錯 | 連同必要接縫小範圍重繪，檢查前後遮擋 | 強貼全部原手，蓋住合理的新袖子 |
| 手指畸形 | 可靠原手保留，否則局部重繪；必要時結構約束 | 以偵測到 21 關節宣稱五指正確 |
| 背景殘影／多餘物件 | 遮罩移除與補洞 | 重新生成整張衣服和人物 |
| Logo／文字／衣服紋理失真 | 優先有對應關係的商品參考／局部材質對齊，人工驗收 | 承諾生成模型能精確還原商標文字 |
| 手臂大幅移位／全身比例錯 | 回到試穿候選或換生成方案；大區域變更另行確認 | 用小遮罩硬修幾何上不相容的肢體 |
| 本來不可見的手 | 保留遮擋；不可靠時標記待確認 | 自動補出兩隻手，或把偵測不到當成不存在 |

以上分流是工程判斷；精確袖口遮擋、人物身分及商品忠實度仍需資料驗證。

---

## 可選手部專家：研究證據

## 工具各自能回答什麼

| 工作 | 可用工具 | 能提供的證據與邊界 |
| --- | --- | --- |
| 找手、估計關節 | MediaPipe Hand Landmarker | 輸出左右手、影像／世界座標、每手 21 個關節；偵測與 presence confidence 是定位訊號，並非解剖正確率。[官方文件](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) |
| 身體與手腕上下文 | DWPose | 全身姿態估計，可協助建立手部候選框與人物對應；有 ONNX 分支，可免裝 mmcv。它是姿態模型，不會生成修好的像素。[官方實作](https://github.com/IDEA-Research/DWPose) |
| 將選定區域轉成像素遮罩 | SAM 2／2.1 | 支援影像提示式分割與自動遮罩；可用點／框細化選區。分割到畸形手仍然只是分割，並未判斷應該長出哪些手指。[官方實作](https://github.com/facebookresearch/sam2) |
| 修復手的形狀與外觀 | HandRefiner | 以手部 mesh 的深度條件，引導 SD1.5 inpainting／ControlNet 重新生成局部像素。[官方實作](https://github.com/wenquanlu/HandRefiner) |

工程推論：固定輸出 21 點的估計器可能在畸形圖上擬合出合理骨架，因此「有 21 點」不能當成通過驗收。同樣地，「未偵測到手」必須保留為未知，不能直接推論人物沒有手；畫面外、袖子遮住、握拳、模糊、小手與偵測失敗需要分開記錄。正常握拳不應套用「必須看見五根手指」的驗收規則。

## HandRefiner：有公開實作的專用後處理候選

流程是偵測手 → MeshGraphormer 擬合手部 mesh → 產生深度與遮罩 → 用 ControlNet 條件式 inpainting。作者提供一般 SD1.5 inpaint＋depth ControlNet 和手部微調權重兩條路徑，並指出一般權重失敗率可能較高；微調權重的建議 control strength 為 0.4–0.8。這是候選初值範圍，不是本專案最佳參數。[方法與限制](https://github.com/wenquanlu/HandRefiner)、[官方安裝說明](https://github.com/wenquanlu/HandRefiner/blob/main/docs/installation.md)

作者建議目前權重使用至少約 60×60 像素的手部；嚴重到無法辨識的手可能擬合失敗，方法也不會修正原先過大的手掌尺寸。失敗時應先檢查深度圖，再檢查遮罩是否完整包含異常長指，必要時增加 padding 或提供手繪遮罩。論文中的 SDXL 來源圖仍先縮到 512×512，由 SD1.5 修復；作者明確說未測試其建議的 SDXL 移植方式。[官方 FAQ](https://github.com/wenquanlu/HandRefiner#important-qa)

工程建議：在帶上下文的局部 crop 上實驗，記錄縮放後手部尺寸與原始尺寸；放大只能改善模型輸入尺度，不能保證恢復不存在的資訊。握拳、手握物品與袖口遮擋應獨立評估 mesh 是否保留原姿勢及前後關係，不能因生成較漂亮就接受姿態改變。

整合判斷：將此候選接在 FASHN 產出的 RGB 圖片之後，使用獨立重繪模型，是合理的像素層介面設計；把 SD1.5 ControlNet 權重直接載入 FASHN 的 MMDiT 不屬於已驗證做法。這裡只提出後處理實驗，沒有宣稱已完成相容性測試。

## 較新候選與實際公開程度

**HandCraft，WACV 2025。** 論文以參數化手模型建立遮罩與深度條件，並提出 MalHand 資料集；其主張是不必重新訓練擴散編輯器即可做手部恢復。[正式論文頁](https://openaccess.thecvf.com/content/WACV2025/html/Qin_HandCraft_Anatomically_Correct_Restoration_of_Malformed_Hands_in_Diffusion_Generated_WACV_2025_paper.html)

作者 GitHub 只是展示頁，但有指向 Hugging Face 的官方原始碼。已核對 Space 包含 `model/yolo.pt`；`bbox.py` 有正常／異常類別處理，`app.py` 提供 `opened-palm`、`fist-back` 兩個模板，以及補入未偵測手的選項。公開介面到控制圖／遮罩即結束，並指示使用者另接 Stable Diffusion／ControlNet，因此不能把它描述為現成的完整修手服務；模板也不能保證保留任意原始手势。這個異常偵測器可以列入校準實驗，不能直接作為 FASHN 影像的通過／拒絕標準。[官方來源入口](https://github.com/kfzyqin/handcraft)、[Space 程式](https://huggingface.co/spaces/zhenyueqin/handcraft/blob/main/app.py)、[偵測器程式](https://huggingface.co/spaces/zhenyueqin/handcraft/blob/main/component/bbox.py)、[權重目錄](https://huggingface.co/spaces/zhenyueqin/handcraft/tree/main/model)

**VTON-HandFit，CVPR 2025。** 它從換衣模型內部處理手部遮擋，加入手部結構／外觀特徵、Hand-Pose Aggregation Net 與訓練損失；HaMeR、DINOv2、DWPose、DensePose 都參與其方法。這代表替換／訓練換衣架構的研究方向，非通用圖片上的即插即用局部修補器。[官方方法頁](https://vton-handfit.github.io/)、[CVPR 論文](https://openaccess.thecvf.com/content/CVPR2025/papers/Liang_VTON-HandFit_Virtual_Try-on_for_Arbitrary_Hand_Pose_Guided_by_Hand_CVPR_2025_paper.pdf)

已確認官方程式公開、安裝依賴與 code/checkpoints 的 CC BY-NC-SA 4.0 聲明；但本次核對的 README 沒有完整權重下載指引，`checkpoints/ootd` 只列 feature extractor、scheduler 與 model index。這些證據不足以確認完整推論權重已可取得，亦不等於證明其他路徑一定沒有權重。因此先列研究參考，不能把它排成隨時可執行的替換方案。[官方 repository](https://github.com/VTON-HandFit/VTON-HandFit)、[核對的 checkpoint 目錄](https://github.com/VTON-HandFit/VTON-HandFit/tree/main/checkpoints/ootd)

## 依賴、授權與資源限制

- HandRefiner 程式標示 MIT，但完整方法另需 MeshGraphormer、MANO、SD1.5 inpainting 和 ControlNet／微調權重。官方安裝要求使用者遵守 MeshGraphormer 授權並自行取得 `MANO_RIGHT.pkl`；MANO 官方標準授權限非商業用途，商業授權另有入口。不能將頂層 MIT 當成所有模型資產的商用許可；微調 checkpoint 與底模條款仍須逐一核對。[HandRefiner 安裝](https://github.com/wenquanlu/HandRefiner/blob/main/docs/installation.md)、[MANO 依賴](https://github.com/wenquanlu/HandRefiner/blob/main/docs/meshgraphormer.md)、[MANO 官方授權](https://mano.is.tue.mpg.de/license.html)
- HandRefiner 的 requirements 固定 torch 2.0.0、numpy 1.23.5、pytorch_lightning 1.4.2，安裝文件指定 MediaPipe 0.10.0；SAM 2 現行官方程式要求 Python ≥3.10、torch ≥2.5.1。實驗宜隔離環境、用圖片與遮罩交換，避免直接把這些版本限制合併進現有應用。[HandRefiner requirements](https://github.com/wenquanlu/HandRefiner/blob/main/requirements.txt)、[SAM 2 安裝](https://github.com/facebookresearch/sam2#installation)
- SAM 2 官方明確將模型 checkpoint、訓練程式與 demo 程式列為 Apache 2.0；DWPose repository 標示 Apache 2.0，但選定模型及其他預處理資產仍要按實際來源記錄。HandCraft Space 的 metadata 僅寫 `license: cc`，沒有指出具體 CC 變體；不應由這個欄位推定商用或再散布條件，Ultralytics 等依賴也須另外確認。[SAM 2 授權](https://github.com/facebookresearch/sam2#license)、[DWPose](https://github.com/IDEA-Research/DWPose)、[HandCraft metadata](https://huggingface.co/spaces/zhenyueqin/handcraft/blob/main/README.md)、[HandCraft requirements](https://huggingface.co/spaces/zhenyueqin/handcraft/blob/main/requirements.txt)
- 本次來源沒有給出「FASHN＋這些修正器」在 RTX 4060 Ti 的可用 VRAM 與延遲保證。建議先釋放／卸載換衣模型，再逐階段測量峰值 VRAM、載入時間與每張修正延遲；也需區分 8GB／16GB 型號。SAM 2 README 的速度表是在 A100 上測量，不能直接換算到本機。[SAM 2 benchmark 條件](https://github.com/facebookresearch/sam2#model-description)

## 建議與信心

先建立通用的局部選區與重繪介面，手繪遮罩是必要 fallback；定位／分割模型只能建議區域。原圖手部健康且位置、遮擋相容時，可保留原手；需要新增／重建像素時再比較一般 inpainting 和 HandRefiner 分支。錯誤 mesh、遮罩不完整、姿勢不確定時，應保留原結果及診斷供人工修正，不應無限換 seed 或自動張開原本握拳的手。

高信心：上述工具的工作分工、HandRefiner 的 SD1.5 與小手限制、HandCraft 公開控制圖程式、VTON-HandFit 屬模型內部方法。中信心：以獨立 crop 重繪作為本專案實驗介面。未知：真實 VTON 圖的改善率、握拳／遮擋泛化、膚色與光照一致性、衣物細節保留、完整授權適用性與本機 VRAM／延遲；均需要本地資料集驗證，不能由論文示例或單張成功推定。

## 現有程式為何不夠泛用

以下是本次讀取本機程式所確認的事實，不是推測模型內部原因：

- `restore_hands` 與 `_check_geometry` 以 hands=13 的硬標籤計算重疊、位置、衣物覆蓋；
  沒有手指解剖分類器。13% occlusion 是標籤重疊，不是手指錯誤率。
  單手／整體都套固定 5% 衣物覆蓋門檻，沒有區分合理袖口與錯誤遮擋。
  [合成與門檻程式](../tryon/hands.py)
- 缺少可靠手部標籤就拒絕，連原本看不到手的照片也可能無法輸出；
  「確定不可見／偵測不確定／可見且可信」需要分開，而不是只看遮罩是否為空。
  [同一判斷路徑](../tryon/hands.py)
- `Engine.generate` 每次只改 seed 重試一次，仍使用相同模型與處理方式；兩次拒絕時
  在儲存結果前拋錯。失敗候選、遮罩與全部嘗試沒有寫成診斷檔，難以分清生成錯誤與檢查誤判。
  [生成流程](../tryon/engine.py)
- 25 項測試驗證程式行為，官方範例的冷／暖 GPU 測試驗證能執行，均不是跨照片品質基準。
  使用者現在已提供失敗案例的兩張輸入，但本輪沒有重跑，也沒有以它調門檻。
  [既有驗證紀錄](verification.md)

因此，先前 v2 是保守的原手合成補強，不能作為通用校正系統。
原圖保留仍可當低成本分支，但不能要求所有問題都通過它。

## 如何驗證不是只修好這一張

研究依據：VTBench 將手部遮擋、背景、衣物紋理與尺寸適應等分開評估，並使用人類偏好，
指出整體指標不足以反映實際觀感；VTON-IQA／VTON-QBench 研究則針對常無真實換衣對照圖的情況，
建立人類回饋式品質評估。這支持多維度驗收，但不代表任何自動評分器可保證每張圖正確。
[VTBench 論文](https://arxiv.org/abs/2505.19571)、
[VTON-IQA 論文](https://arxiv.org/abs/2603.13057)。
本次查閱的 VTBench 官方 repository 主要呈現 README 與圖示，未核實完整可執行測試集；
這裡只借用評估原則，不宣稱已安裝或跑過 benchmark。
[VTBench repository](https://github.com/HUuxiaobin/VTBench)

### 建議的第一輪實驗（數量是工程起點，不是統計認證）

1. 收集有使用權的 60 組人物／商品配對，40 組開發校準、20 組保留驗收；
   人物身分、近重複照片與商品不能跨兩組洩漏。每組固定 2 個 seed。
   覆蓋全身／半身、小手、握拳、交叉手臂、持物、手藏起來、長短袖雙向切換、
   複雜背景、深淺衣色、文字 Logo，以及原本已正確的結果；不同體型、膚色與真實肢體差異也要保留。
   動漫先列獨立測試域，未通過前不宣稱支援。
2. 每個固定試穿候選比較：不修復、現有 v2、人工正確遮罩的通用 inpainting；
   只有手部子集再比較結構約束分支。使用相同候選可隔離修補效果，
   另測含原有重試的端到端時間，避免把換 seed 的差異算成修復成功。
3. 先用人工遮罩驗證修補能力，再用自動遮罩跑同一批：前者失敗表示修補器／條件不足，
   後者額外失敗才是定位問題。不能只報已被系統接受的圖片。
4. 每張顯示原人物、商品、修前／修後全圖與局部，至少兩位評閱者盲看版本；
   分別記錄目標缺陷是否修好、是否產生新缺陷、姿態／身分保持、衣物忠實度與接縫。
   有分歧保留紀錄，握拳與自然肢體差異不套固定可見手指數。

### 必須一起報告的指標

- 缺陷修復率：人工確認有缺陷的案例中，真正修好且未造成其他重大損害的比例。
- 誤修率：原本正確的案例被改壞的比例；另報選區造成的不必要改動。
- 錯誤接受／錯誤拒絕率，以及需人工處理率；以人工標註為參考，逐情境列分子／分母。
- 最終合成在允許修改區域外的最大像素差必須為 0；這只能保證局部性，不能保證區域內品質。
- 手部結構、遮擋、商品紋理／Logo、人物／背景分別評分，不以單一 IoU、CLIP 或偵測信心取代。
- 冷／暖載入、每個階段延遲、總嘗試次數與 GPU 總顯存峰值；不能只算 PyTorch allocator。

升級條件：保留驗收組的缺陷修復率有改善、正常案例不新增重大破壞、
遮罩外像素檢查全數通過，且各情境拒絕率與成本可接受。逐類附例圖與樣本數；
小樣本仍不足以宣稱「泛用已解決」。保留組看過後若繼續調參，下一版須另留新驗收組。

## 建議執行順序

1. **先補可重現診斷與局部編輯實驗。** 以隔離環境測一個 SDXL inpainting checkpoint，
   人工遮罩、裁切放大、原候選保留、遮罩外像素鎖定與前後比較即可；先不引入全部模型。
   診斷影像儲存應可選、僅本機、Git 忽略、有限保留；不得把私人照片提交到 repo 或默認上傳雲端。
2. **人工選區有效後才自動化。** 重用現有 Human Parser／DWPose 提出候選區域，
   需要精細描邊才加 SAM。規則門檻在開發集校準；不能讓使用者這一張決定全域參數。
3. **針對剩餘缺陷比較專家。** 手部才試 HandRefiner／結構條件；
   通用修補品質不足才比較 FLUX Fill。模型分階段載入，先測 4060 Ti 16GB 而不承諾共駐可行。
4. **若局部修補仍無法保住衣服與姿勢，才評估底模替換／訓練。**
   VTON-HandFit 是這一層的研究參考，不是當前可直接開關的修補功能。

這個順序解決的是一類流程缺口：即使未來出現新的局部缺陷，仍有可選區、可約束、
可撤回、可驗收的處理路徑；不承諾任何模型能自動修好所有未知錯誤。
