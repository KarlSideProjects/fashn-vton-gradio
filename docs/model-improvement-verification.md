# 多模型局部修補實測（2026-09-14）

結論：三條新增神經路徑已在本機實際執行，可供**人工操作的實驗比較**，不是通用修正器。
固定案例未證明增加 ControlNet 或外觀參考能穩定改善；手部消失、假字與硬遮罩接縫都有出現。
保留原 SDXL／FASHN 預設，不自動選優、採用、微調權重或將結果稱為「已修好」。

## 研究、審查與實作順序

Orca run：`run_bca805876cf5`。三名 Codex Luna/max 研究 worker 先分頭研究官方資料，
另一名獨立 Codex worker 完成反向審查，再派兩名 Codex worker 分工實作；協調者做整合及本機 QA。
以下六個 dispatch 均成功結案、終端 released；沒有以非 Orca 子代理替代。

| 工作 | Task／Dispatch | 可查閱證據 |
|---|---|---|
| VTON 改善調查 | `task_04456b2cbc95`／`ctx_5e3d5ac5bc97` | [VTON 研究](model-research-vton.md) |
| 局部修補模型調查 | `task_f2e935db58f8`／`ctx_a7706fc5f5b5` | [修補研究](model-research-repair.md) |
| 評估與複合流程調查 | `task_43cc53ba073f`／`ctx_4304b97c0f0f` | [評估研究](model-research-evaluation.md) |
| 獨立反向審查 | `task_1bc282267880`／`ctx_520764230d09` | [有條件 GO 與七項 P1](model-improvement-adversarial-review.md) |
| 後端實作 | `task_c60ff97246c9`／`ctx_e40f256fce8a` | `repair.py`、`repair_models.py`、`inpaint.py` 及測試 |
| 介面／比較流程實作 | `task_9973e00e2927`／`ctx_5014ed2db69b` | `app.py`、`engine.py`、`comparison.py` 及測試 |

研究方案與明確延期項目見 [複合流程](model-research-composite.md)。研究的模型規格／授權來源
已由 worker 使用官方來源查核，獨立審查另作交叉核對；本文件記錄本機證據，不增加商用授權承諾。

## 固定資料 GPU 配對

RTX 4060 Ti 16GB、32GB RAM；Torch 2.8.0+cu128、Diffusers 0.35.1、Transformers 4.57.1。
固定案例位於 `outputs/experiments/cases-c421e39393c44388856c2d903f153e38/`，來源／遮罩詳見複合流程文件。
每對使用同底圖、遮罩、提示、seed 42、30 steps、CFG 7.5；手 strength .85、布料 .99，
control／reference scale .5。不同底模有不同工作尺寸與 scheduler，並非跨架構的相同噪聲實驗。
每條路徑一次，沒有隱藏 reroll 或自動採用；共八對、十六次嘗試，十四個候選、兩個明確拒絕。

| 案例 | 方法對 | 本機 experiment JSON（`outputs/experiments/`） | 結果／比較總秒數 |
|---|---|---|---|
| 手部，完整 mask | SD1.5 ± ControlNet | `0f6399e86057439eba381069df125a55.json` | 2 候選／28.694 |
| 手部，完整 mask | SDXL ± Plus | `11d122e73bc64c828618bfd1eadaa662.json` | 2 候選／69.875 |
| 布料污點 | SD1.5 ± ControlNet | `b31bd474838e4c91b00b4f91f1030c27.json` | 2 候選／28.848 |
| 布料污點 | SDXL ± Plus | `1776c20f6c9740ac8142511556e8df09.json` | 2 候選／75.559 |
| 原本乾淨區域 | SD1.5 ± ControlNet | `0e5f9d37939e404fb773c6eae07dafc5.json` | 2 候選／29.933 |
| 原本乾淨區域 | SDXL ± Plus | `508346ed5493481b9677e10962d319f1.json` | 2 候選／72.502 |
| 手部，mask 侵蝕 2 px | SD1.5 ± ControlNet | `c8b7b7566119419c92e6b1b44c94d2c9.json` | 兩者 safety checker 拒絕／26.482 |
| 手部，mask 侵蝕 2 px | SDXL ± Plus | `9ab8a86b93644e92949076a5cc5fed11.json` | 2 候選／67.492 |

所有成功候選獨立核對原尺寸，以及原始正規化底圖的遮罩外像素完全一致。
第一、二對先載入 FASHN 再修補，確認同 Engine 的 FASHN 已卸載；未終止原有外部 GPU 程序。
每個 Plus 候選都保存實際 encoder 的 `[1,3,224,224]` fp16 tensor、hash、processor 設定、
固定 adapter／encoder 路徑，以及等比例補邊參考；並非僅記上傳圖 hash。

漏選反例的兩個 SD1.5 worker 確實被安全檢查拒絕，未發布 patch／黑圖，也沒有關閉 checker 重試。
原始 failure 證據是 `outputs/repairs/303f8cb82f644198b08a9dc424fbb0b9/failure.json` 與
`outputs/repairs/795d71ef7e334815920adfea16f5c588/failure.json`；全失敗比較為
`outputs/comparisons/ed33141121dc4fa7a2c2769dd97658fc/comparison.json`。
當時介面錯誤過度籠統，整合後已修正為傳遞 worker 原因，並把失敗檔案位置附入比較紀錄。
這是安全分類器的判斷，不據此認定原照片不當，也不把缺少可見輸出算成修復成功。

## 單次成本範例（完整手部案例）

| 方法 | 單候選總秒 | 模型載入秒 | 推論秒 | PyTorch allocated／reserved GiB | worker CPU peak RSS GiB | 整卡採樣峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|
| SD1.5 | 11.928 | .399 | 4.940 | 2.478／2.652 | 6.321 | 9232 |
| SD1.5 + ControlNet | 13.330 | .361 | 5.593 | 3.200／3.363 | 6.321 | 9960 |
| SDXL | 27.327 | .473 | 17.111 | 5.365／5.816 | 8.142 | 12472 |
| SDXL + Plus | 33.163 | 1.097 | 18.181 | 6.171／6.662 | 10.549 | 13338 |

這些是單次、已安裝且可能命中 OS 檔案快取的獨立 worker，**不是冷磁碟延遲或平均值**。
單候選總秒還含 preflight／程序啟動／合成；pair 總秒另含雙方預檢等開銷。
PyTorch 峰值涵蓋載入、encoder、denoise、decode；不含外部程序與 ONNX。
CPU 是 worker lifetime RSS；另有父子程序 RSS 採樣，可能重複計算共享頁。
整卡含原有約 6.2GB 其他程序；0.5 秒加命令耗時的採樣會漏掉瞬時峰值，不是 VRAM 保證。

## 視覺檢視（單一 AI 評閱者，非人類盲評）

評閱者：協調者 Codex。看過原人物／商品、完整結果與局部放大；本機聯絡表為各 experiment
同名 `-sheet.png`，未上傳圖片。以下是具體觀察，不是解剖學標註或跨照片改善率。

| 案例 | SD1.5 | + ControlNet | SDXL | + Plus |
|---|---|---|---|---|
| 完整手部 | 更差：手變模糊小塊 | 更差：袖子蓋住手 | 握拳輪廓較完整，但新增黑腕邊，不能算完全修好 | 握拳保留，但粗黑腕邊且細節較平，未見穩定優於 SDXL |
| 布料污點 | 亮污點消失，但留下硬邊深色方塊；未完全修好 | 同樣有深色方塊；無明顯額外改善 | 新增「28」樣的白色數字，不能採用 | 新增「26」樣的數字，參考未避免假字 |
| 原本乾淨布料 | 更差：新增深色方塊 | 更差：新增深色方塊 | 新增較淡方塊接縫 | 新增較淡方塊接縫；都應保留原圖 |
| 漏選手部 | 無可見候選；安全拒絕 | 無可見候選；安全拒絕 | 手勢接近原候選但輪廓仍不自然，未明確改善 | 手前端出現暗綠色團塊，更差 |

完整手部約 1299–1301 個像素改變，布料／乾淨區域各 1520，漏選 SDXL 各 991；
changed pixels 只表示修改範圍，表中缺陷判定來自影像檢視，沒有把像素差當品質分數。
未見遮罩外的新缺陷與 exact outside-mask 檢查一致，但不是全身原本就正確的證明。
小污點仍可考慮既有 Telea；它不能重建手部，不能由單一示範推廣成通用方案。

## 工程驗證與證據界線

- 最終 `.venv/bin/python -m unittest discover -s tests -q`：74 tests，全部通過。
  `compileall`、安裝／啟動腳本 `bash -n`、`git diff --check` 通過。
- CPU 回歸含輸入／EXIF、逐手拒絕、hint 負值／座標、資產漏檔與損壞、離線 loader、
  安全拒絕、部分／全部失敗、逾時、跨 session／舊世代、竄改輸出與原子採用。
- 真實 Gradio HTTP／佇列 smoke 通過；缺少輸入回傳明確錯誤，沒有假 PNG。
- 真實 Chrome CPU 流程：上傳、筆刷、Telea、無預選、明確選擇／下載、採用後清除 mask，再修補因空 mask 被拒絕。
- 真實 Chrome GPU 流程：官方人物／商品、30 steps、seed 42、CFG 1.5、segmentation-free 關閉，
  生成成功自動載入 editor，再上傳衣料參考、看補邊預覽、完成 SD1.5／ControlNet 雙候選，
  無預選、明確下載、no-op 後舊候選失效；最後一輪沒有瀏覽器 JS error。
  截圖 `.cache/generated-autoimport-browser.png`、`.cache/generated-pair-browser.png`；
  這是功能驗證，不把額外的 UI 測試結果納入上述固定八對品質比較。
- 三條新增神經方法已成功離線執行；缺模型預檢曾實際留下 `preflight_failed`、0 候選，沒有啟動 CUDA 推論。
- 逾時恢復另以真實子程序、注入 .2 秒 timeout 驗證：確認 PID 已被回收，失敗檔案組
  `outputs/repairs/86d61611da724c8091f6947c38cce70d/` 沒有發布 patch；同一 Engine 隨後完成真正 SD1.5 修補，
  紀錄 `outputs/repairs/c2577400f107483ea0f28418cd82ba01/f57612704b1141d4b75d141e0aa565ec.json`。
  這不是實際等待 300 秒；沒有故意耗盡 GPU，不能當作 OOM 壓力認證。

尚未做 30／60 案例保留集、雙人盲評、商用品質認證、自動 IQA／SAM／HandRefiner／LoRA，
亦未訓練或改善底模權重。依獨立審查的放行規則，工程可行但品質尚不可靠時只提供實驗工具。

### 整合時發現的 Gradio 相容問題

自動載入後第一次筆刷可重現 `this.history.push is not a function`。
已排除全頁截圖的影響，瀏覽器探針確認 `CommandManager.replay` 收到失去 prototype 的
普通 `{command,next,previous}` 物件。上游 [CommandManager 原始碼](https://github.com/gradio-app/gradio/blob/main/js/imageeditor/shared/core/commands.ts)
預期的是含方法的節點；本機 5.50.0 的來源與 bundle 也已直接讀取。
未找到可直接引用為此症狀官方修復的 issue，因此不聲稱已有官方解法。

`scripts/patch_gradio.py` 在介面啟動時只修改固定 5.50.0、SHA-256 已核對的單一 frontend asset：
遇到不能回放的普通物件，保留新 editor 已載入的底圖／layers，不採用無效舊撤銷歷史。
版本／內容不符拒絕修改；重複執行不再改檔。未升級套件或修改模型。
短測試以已存 GPU 輸出替代生成 callback，確認修補前失敗、修補後自動載入／筆刷／採用無 JS error；
最後另重跑真實 GPU 端到端，沒有沿用測試替身。首次更新須強制重新整理瀏覽器快取。
新畫的筆刷可撤銷／重做；瀏覽器截圖回放邊緣有最大 3/255 的抗鋸齒差，
不是後端硬遮罩「選區外完全不變」契約的放寬。

### 七項審查條件的驗收對照

| 條件 | 證據與結論 |
|---|---|
| P1-1 相容模型／hint | `test_repair_models` 值域及 loader 測試；四個 Plus 真實 tensor 核對；SD1.5／ControlNet 真實配對 |
| P1-2 完整資產／離線／安全 | 固定資產安裝成功；漏檔／損壞／越界測試；兩次真正 safety 拒絕未發布黑圖 |
| P1-3 串行／卸載／成本 | warm FASHN 配對通過、峰值及耗時表、實際短逾時回收後同 Engine GPU 修補；未聲稱 OOM 壓力認證 |
| P1-4 座標／reference | 邊界、細線、非方形、EXIF 測試；所有候選 exact outside-mask；實際補邊與 encoder hash |
| P1-5 失敗可追溯／有界比較 | partial／全失敗測試及真實全失敗紀錄；固定八對最多各兩個方法、無自動採用 |
| P1-6 session／世代 | 交叉 session、延遲回傳、竄改、重複採用與原子失效測試；真實瀏覽器選擇／下載／no-op |
| P1-7 負對照／誠實品質 | 原本乾淨區域真的執行，漏選反例另列；單一 AI 評閱明列失敗，不冒充盲評 |

圖譜 Tier 2 已核對實作與測試路徑；`.venv` 的前端原始碼屬排除範圍，改用直接讀取，
不把圖譜無缺口當成全部程式正確的證明。

## 2026-09-15：正常流程改為背景自動處理

使用者修正需求：正常生成不能要求畫遮罩、選模型或挑候選。以上手動多模型研究為歷史
驗證，工具仍保留於預設摺疊的進階區，不再是預設工作流程。

- 重用既有 Human Parser 與原圖合成；自動定位手部，以及幾何高度一致的臉／頭髮內部。
  衣物與配件鄰近區域不做臉／頭髮貼回。兩手可匹配時獨立驗收，不放寬逐手幾何門檻。
- 最多兩次 FASHN（Seed 42／43）；取第一個通過所有已檢出部位幾何限制的結果，否則回到
  第一次結果，保留其中通過檢查的局部。分割出錯／第二次生成失敗不丟掉已有結果。
- 沒有加裝新模型、沒有自動接受 SDXL 等神經重繪；原有負對照不能支持這種品質承諾。
- 84 項單元測試、compileall、diff check、真實 HTTP／佇列檢查通過。涵蓋逐手拒絕、
  臉部錯位、衣物與配件保護、乾淨像素不變、分割失敗、重試失敗及摺疊進階工具。
- Chrome 真實 GPU 端到端：兩组都只上傳人物／商品並按一次「開始試穿」，其餘保持預設，
  無手動畫筆、提示詞、模型或候選選擇，兩組均輸出 PNG／JSON 且無 JavaScript error。

最終實作實測（均自動嘗試兩次、選回 Seed 42）：

| 案例 | 結果紀錄（outputs/） | 背景結果 | 推論與處理時間 |
|---|---|---|---|
| 官方人物＋商品 | `994c367579b142cd88748f368915daac.json` | partial；一隻手保留，3866 個像素改變；其餘區域拒絕 | 60.921 秒 |
| 樹林人物＋長袖商品 | `00bdb18fd9d544fdb632a5d37599fda6.json` | kept_original；修前／修後像素雜湊相同，沒有宣稱修好 | 60.369 秒 |

已直接查看官方案例修前／修後全圖，修改範圍在影像左側手部；像素差範圍
`[55, 427, 143, 521]`。這是原手保留的執行證據，不是解剖／視覺品質盲評。
臉與頭髮路徑通過合成測試，但這兩組真實案例均未被接受，尚無真實品質改善證據。
本次完成的是無互動執行、保守局部保留與回退；通用瑕疵偵測和可靠神經重建仍未完成。
修前圖、逐區決策與雜湊可由結果 JSON 的 `automatic_repair` 追溯；私人圖片未提交 Git。
