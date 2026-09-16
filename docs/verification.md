# 驗證紀錄 — 2026-09-14

> 歷史紀錄：本文描述已被替換的早期流程，修補、輸出格式與測試數不代表目前版本；請以[專案首頁](../README.md)、[操作手冊](usage.md)及現行程式為準。

## 最新：通用局部修補

已新增可手動塗選的 Gradio ImageEditor、修前／修後比較、採用後繼續修補、PNG／JSON 下載，
以及可明確選擇略過原手保留的待修補試穿候選。沒有部署自動判錯、SAM、HandRefiner 或雲端服務。

環境：原試穿依賴不變；獨立 `.venv-repair` 使用 Torch 2.8.0、Diffusers 0.35.1、
Accelerate 1.10.1；SDXL revision `115134f363124c53c7d878647567d04daf26e41e`，
權重約 6.5 GiB，模型檔已下載並產生雜湊 manifest。紋理補洞重用原環境 OpenCV 4.12.0。

| 檢查 | 結果 |
|---|---|
| `.venv/bin/python -m unittest discover -s tests -q` | 35 tests，通過；包含自動載入事件／清除舊選區、遮罩驗證、精確局部合成、逾時無輸出、待修補候選及真實 OpenCV 小污點測試 |
| `.venv/bin/python -m scripts.smoke_ui --port 7862` | HTTP 200，試穿／局部修補的缺圖請求均明確失敗且清空輸出 |
| `.venv/bin/python -m compileall -q app.py tryon scripts tests` | 通過 |
| `bash -n scripts/setup_repair.sh` | 通過 |
| SDXL 真實 GPU smoke | 多次執行完成；約 22–24 秒／次，PyTorch peak allocated 約 5.365 GiB，不含其他程序 |
| Chrome 真實操作 | 上傳、筆刷選區、SDXL GPU 重繪／CPU 紋理補洞、下載連結、採用後繼續修補已操作；驗證採用後舊遮罩清空，再修補會要求重新塗選 |
| 選區外像素 | 所有完成的修補由合成層檢查與修前完全相等 |

GPU smoke 指令（人造色塊是可重現測試瑕疵，不是使用者失敗照片）：

```bash
.venv/bin/python -m scripts.smoke_repair \
  --image outputs/901040b72aeb4d87bfe52eb5ed15cfa6.png \
  --prompt 'a close-up photograph of smooth plain black fabric'
```

**功能通過不等於視覺校正成功。** 人工檢視發現 SDXL 在這個衣服色塊案例中，
0.75 強度會保留色塊；0.99 與不同提示／Seed 也可能生成文字、符號或粉紅圖案。
另行測試強度 1.0 得到不匹配紋理，因此沒有把 1.0 開放為正式 UI 選項。
所有這些失敗都不能算修復成功，UI／JSON 保持「待人工驗收」。

相同小色塊以 OpenCV 紋理補洞約 0.17 秒完成，人工檢視色塊移除、背景與人物未改；
僅支持這類簡單瑕疵的可行性，不代表能修手。對照檔案組：
`outputs/repairs/537f9a6d56804aa8b0b713e206e741be/`。
SDXL 失敗案例之一：`outputs/repairs/4d34401edc5e4ce2bfb6e6a260c4cc8c/`。
瀏覽器 QA 截圖／日誌位於 `.cache/repair-browser-success.png`、`.cache/repair-browser-final.log`。
這些本機測試產物被 Git 忽略，不是套件隨附資料。

尚未完成：研究報告中的跨人物／商品保留驗收集、使用者握拳與長袖案例的修復品質、
複雜遮擋、Logo 精確還原、自動錯誤偵測與完整手機介面驗收。
不宣稱通用生成缺陷已解決；此版完成的是可選區、可回退、可重現的局部修補入口。

## 前次：手部保留 v2

環境：Python 3.12.14、Torch 2.8.0+cu128、RTX 4060 Ti 16GB。
本地虛擬環境搬移後的 Python 路徑已修復，原有套件、權重與快取保留。

| 檢查 | 結果 |
|---|---|
| `.venv/bin/python -m unittest discover -s tests -v` | 25 tests，OK |
| `.venv/bin/python -m scripts.check` | CUDA、BF16、ONNX 與權重雜湊檢查通過 |
| `.venv/bin/python -m scripts.smoke_ui` | 真實 HTTP 200；缺圖佇列請求明確失敗且不輸出檔案 |
| `.venv/bin/python -m compileall -q app.py tryon scripts tests` | 通過 |
| `git diff --check` | 通過 |
| `.venv/bin/python -m scripts.smoke_gpu` | 真實冷、暖生成完成，exit code 0 |

新增 6 項合成回歸測試：細手指完整保留、較小手部消失、多出小手、僅手腕羽化、
雙手分別記錄、較小手部袖口遮擋。新增測試在修改前有 4 項失敗；修改後全部通過。
舊版羽化會把生成像素混入細手指；整體遮罩分數也可能掩蓋較小手部的缺失或遮擋。
v2 保留原圖手指像素，僅手腕接縫羽化，並增加逐手幾何檢查。

官方範例：`examples/model.webp` + `examples/garment.webp`，tops/model、30 steps、請求 seed 42。
冷、暖兩次皆因單手袖口遮擋約 7% 拒絕 seed 42，再以 seed 43 成功合成。

| 執行 | 載入秒數 | 生成與手部處理秒數 | 輸出（PNG；同名 JSON 記錄參數） |
|---|---:|---:|---|
| 冷啟動 | 8.147 | 60.757 | `outputs/901040b72aeb4d87bfe52eb5ed15cfa6.png` |
| 暖啟動 | 0 | 60.248 | `outputs/c23330b996664051885ce28da1566b7f.png` |

兩次均記錄 `source_hand_composite_v2`、兩個手部區域及 `wrist_only`；
PyTorch 峰值 allocated 2.956 GiB、reserved 5.084 GiB，不含 ONNX 與其他程序顯存。
對照舊版官方範例約 30 秒，較嚴格的檢查觸發第二次推論，這次約需 60 秒。
新舊結果使用不同的最終 seed，不能把整張圖片差異直接當成修復品質提升的證明。

人工檢視官方範例輸出，手腕色差／接縫仍可能可見。沒有使用者的失敗照片，
尚未驗收動漫、複雜遮擋、極小手部與不可見手部；也未完成全面門檻校準或瀏覽器互動驗收。
這是幾何檢查與原手保留，不是手指數量或解剖正確性判定。
未新增推論依賴、SAM 權重或雲端服務；repo 內的 segmentation skill 僅供後續工作參考。
生成檔與執行日誌位於 Git 忽略的 `outputs/`、`.cache/hand-repair-gpu.log`。

## 歷史紀錄：初始 CPU 介面檢查

以下保留先前環境的驗證紀錄；其中 GPU 阻擋已由上方最新實測解除，並非目前狀態。

環境：Linux x86_64、Python 3.12.14、Gradio 5.50.0、Pillow 11.3.0、
Hugging Face Hub 0.36.2、NumPy 2.2.6。這是 CPU 介面檢查環境，未安裝推論依賴。

| 檢查 | 結果 |
|---|---|
| `python -m unittest discover -s tests -v` | 9 tests，OK |
| `python -m scripts.smoke_ui` | 真實 Gradio HTTP 200；佇列/API 缺少圖片請求回傳明確錯誤，三個輸出均為 None |
| `ruff check --select F app.py tryon scripts tests` | 通過 |
| `python -m compileall -q app.py tryon scripts` | 通過 |
| `bash -n scripts/setup.sh scripts/start.sh` | 通過；僅語法檢查 |
| 官方兩張圖片 | PIL 解碼成功；Git blob SHA 與上游一致 |
| PyPI 核心 GPU 依賴 dry-run | 指定 Torch/Torchvision/ONNX/parser/Transformers/NumPy/OpenCV 組合可解析；未安裝或執行 |

9 項測試內容：缺圖拒絕、RGBA 白底、EXIF 校正、參數範圍、單張參數、唯一 PNG/JSON、
真實 Gradio 組態、錯誤時清空圖片/下載、真實 Engine 缺少權重明確失敗。
檔案測試使用臨時資料夾；沒有替代 GPU 管線或生成假試穿圖。

測試看到 Gradio 6 遷移相關 DeprecationWarning，以及 Gradio 建立時的 asyncio ResourceWarning；
目前固定 5.50.0，HTTP/佇列測試通過。這不構成長時間負載穩定性驗證。

## 未驗證與阻擋

- **GPU 推論：blocked。** 此環境沒有 `nvidia-smi`、PyTorch 或模型權重。
  實際執行 `python -m scripts.smoke_gpu` 以 exit code 1 停止，訊息：
  `RuntimeError: 缺少模型權重，請先執行 bash scripts/setup.sh`。
  沒有產出試穿圖片，沒有 4060 Ti 耗時/顯存的實測結果。
- **完整安裝：not verified。** shell 腳本僅檢查語法與核心依賴解析；
  上游 Git 下載與 Hugging Face 權重下載未在這裡完成。
- **瀏覽器視覺 QA：blocked。** 本地 Playwright 無 Chromium，官方瀏覽器下載逾時。
  HTTP/API 通過不能替代圖片上傳、範例點選、RWD 與結果下載的瀏覽器端驗收。
- **遠端 GitHub：blocked。** 已連上 jhihwei 並取得 repositories 清單；
  連線可操作現有 repository，但沒有建立 repository 的功能，未提供本專案目的 repository。
  未修改其他 repository，未建立 PR，未執行 GitHub Actions。

## 你的 GPU 上接續驗收

```bash
bash scripts/setup.sh
.venv/bin/python -m scripts.smoke_gpu
bash scripts/start.sh
```

檢查 `outputs` 中兩張真實輸出和 JSON（冷/暖啟動），再以瀏覽器測試官方範例、
自行上傳、平拍商品模式、失敗後重試、PNG/JSON 下載。
衣服圖案、Logo、身分與體型保留要人工檢查，不能只看 exit code。

## 自動手部保留更新

新增 7 項 CPU 遮罩/合成測試與 3 項流程測試：成功後儲存修復圖、第二次成功記錄實際 seed、
兩次拒絕不寫出結果。流程測試明確替代 GPU 外部邊界，不代表真實模型執行成功。
新增功能尚未做 GPU、真實照片品質、分割精準度與門檻校準。
不具備自動解剖判斷、生成式補手或複雜遮擋推理能力。
