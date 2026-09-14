# 本地 AI 試衣間

本機執行的虛擬試穿與局部修補工具，使用 FASHN VTON v1.5 和繁體中文 Gradio 介面。
上傳人物照與商品照即可產生試穿候選，再用筆刷選取手部、袖口、衣料或背景瑕疵進行修補。

基本流程：上傳兩張照片 → 生成試穿圖 → 自動載入修補區 → 塗選與修補 → 人工確認後下載。
自動載入不等於自動修補；仍須自行選區並按下修補按鈕。

這是實驗工具，不保證手指、人物身分、商品圖案或遮擋關係正確，
也不能判定實際尺寸與合身程度。功能測試通過不等於視覺品質合格；
已完成的測試與限制見 [驗證紀錄](docs/verification.md)。

## 目前提供什麼

- 試穿：支援上衣、下身、連身衣，以及模特兒商品照和平拍商品照。
- 原手保留：通過幾何檢查才合成原圖手部；不安全時拒絕輸出，也可自行關閉。
- 局部修補：手動筆刷選區，可選 SDXL 生成式重繪或 OpenCV 紋理補洞。
- 結果管理：修前／修後比較、採用後繼續修補，以及 PNG／JSON 下載。

目前沒有自動瑕疵偵測、自動挑選最佳結果或多模型自動切換。
局部修補並不限於手部，但每次結果都需要人工驗收。

## 安裝與啟動

目標環境為 Ubuntu、NVIDIA RTX 4060 Ti 16GB、32GB 系統 RAM。
基本試穿請預留 30–50GB 磁碟空間；SDXL 修補另需權重、環境與快取空間。
先安裝相容的 NVIDIA 驅動，確認 `nvidia-smi` 可執行，並準備 `git` 與
[uv](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
git clone https://github.com/KarlSideProjects/fashn-vton-gradio.git
cd fashn-vton-gradio
bash scripts/setup.sh
bash scripts/start.sh
```

開啟 <http://127.0.0.1:7860>。若要改連接埠：

```bash
bash scripts/start.sh --port 7861
```

基本安裝建立專案內 Python 3.12 `.venv`，使用 PyTorch 2.8.0 CUDA 12.8 wheel、
Gradio 5.50.0 與固定 commit 的 FASHN；不修改系統 Python。
首次安裝需連線下載套件及權重，生成請求期間使用 Hugging Face 離線模式。
缺少權重或 CUDA 時明確失敗，不改用其他模型或 CPU 試穿。

### 選用：SDXL 局部重繪

完成基本安裝後，再執行：

```bash
bash scripts/setup_repair.sh
```

SDXL 使用獨立 `.venv-repair`，不修改試穿環境；fp16 權重約 6.5 GiB，
儲存在 `weights/sdxl-inpaint`，固定 revision 並記錄 SHA-256。
「紋理補洞」重用基本環境的 OpenCV，不需要安裝 SDXL。

更新程式後，請先停止舊服務，再執行 `bash scripts/start.sh` 並重新整理網頁。
已有權重與快取可保留；請勿同時啟動多個程序競用同一張 GPU。

## 試穿操作

1. 上傳人物與衣服商品照片，或點選官方範例。
2. 選擇上衣、下身或連身衣；商品由模特兒穿著時選「模特兒穿著」，否則選「平拍商品」。
3. 初次可使用 30 步、Seed 42；固定 Seed 方便比較。
4. 按「開始試穿」。預設啟用「自動保留原手」，不適合合成時會以 Seed + 1 重試一次。
5. 成功後下載 PNG 與 JSON 紀錄；同一張結果也會自動載入下方修補區，清除舊選區。

試穿與修補共用單一工作鎖，一次處理一張，佇列最多 8 個等待任務。
上游生成畫布為 576×864，去除 padding 後尺寸可能不同；不做生成式放大。
試穿本身可能改動背景、臉部、體型、文字與 Logo，需人工比較原人物與商品照片。

### 遇到 `HandRepairError` 怎麼辦

此功能是原圖手部合成，不是手指解剖判定。程式重用 Human Parser 的手部標籤，
檢查整體及逐手位置、重疊與衣物遮擋。通過後完整複製原手遮罩內的手指像素，
僅在手臂接到手腕處羽化，不貼回遮罩外的舊衣物。
紀錄方法為 `source_hand_composite_v2`，包含各區域指標與 `wrist_only` 羽化方式。

看不到手、手太小、分割錯誤、手部位移或長短袖切換，都可能使檢查拒絕。
兩次嘗試均未通過時，會出現 `HandRepairError` 且不輸出圖片。
IoU、coverage、occlusion 等數值是遮罩幾何指標，不是手指正確率。

若看到「兩次自動嘗試均無法可靠保留手部，未輸出圖片」，可改走人工修補流程：

1. 展開「生成設定」，取消勾選「自動保留原手」。
2. 再按「開始試穿」，等待產生待人工檢查的候選。
3. 生成成功後，候選會自動載入修補區，再塗選需要重畫的部位。

這不會放寬幾何門檻，也不表示手部已修好。被拒絕的那次請求沒有輸出可供自動載入；
如果修補區已有舊圖，請勿誤認為它是本次失敗請求的新結果。

## 局部修補

1. 直接使用自動載入的試穿結果，或在修補區上傳現有圖片；不必先執行試穿才能修補。
2. 用白色筆刷塗選完整瑕疵與必要接縫；可用橡皮擦縮小選區，不能整張塗滿。
3. 選擇修補方式，按「只重繪塗選區域」。
4. 比較修前／修後；不滿意可調整選區或參數重試。
5. 確認後下載圖片與紀錄，或按「採用修補後圖片，繼續選區修補」。採用後須重新塗選。

| 方式 | 適合用途 | 設定與限制 |
|---|---|---|
| SDXL 局部重繪 | 嘗試重建手部、袖口或其他局部內容 | 需額外安裝；使用提示詞、避免內容、強度與 Seed；仍可能改錯結構或新增圖案 |
| 紋理補洞 | 衣料或平滑背景上的小污點 | OpenCV Telea 延伸周圍像素；不使用提示詞、強度或 Seed，不能重建手指 |

SDXL 提示詞建議用英文描述希望得到的結果，例如
`a natural fabric cuff, matching the blue sleeve`；避免內容可填 `text, logo, pink stain`。
重繪強度可在 0.1–0.99 間調整，預設 0.99；較強不代表較正確，提示詞也不是硬性約束。

### 圖片載入與回退

| 操作 | 修補區的行為 |
|---|---|
| 新的試穿生成成功 | 自動換成新生成圖，清除舊筆刷選區 |
| 試穿生成中或失敗 | 保留現有修補底圖與選區，不載入空圖 |
| 按「將目前試穿結果載入修補區」 | 重新載入目前試穿結果，清除選區 |
| 按「採用修補後圖片，繼續選區修補」 | 換成修後圖，清除選區，下一輪須重新塗選 |

修補不覆寫原始檔案。不滿意時可保留原底圖重試，或重新載入試穿結果；
開始新的試穿前，請先下載要保留的修補成果。

### 修補能保證與不能保證的事

程式以硬遮罩貼回結果，逐像素確認選區外與正規化後的修補前 RGB 圖完全一致。
這不代表與最初的人物照一致，也不保證選區內內容正確；硬邊界仍可能有接縫。
結果一律標示 `needs_review`，尚無自動解剖、商品忠實度或通用瑕疵判定。

實測 SDXL 在示範衣料污點案例中仍可能產生文字、符號或錯誤紋理；
OpenCV 移除過簡單小污點，但不能據此推定能修好手部或所有照片。
目前未整合 SAM 自動選區或 HandRefiner；repo 內的 segmentation skill 是開發參考，
不等於已安裝 SAM 模型。後續方向見 [局部校正研究](docs/hand-correction-research.md)。

## 常見問題

- **生成圖沒有自動出現在修補區**：確認本次試穿確實成功。更新程式後需重啟服務並重新整理網頁；
  也可按「將目前試穿結果載入修補區」。若本次被手部檢查拒絕，請依上方 `HandRepairError` 流程重新生成。
- **局部修補模型尚未安裝**：執行 `bash scripts/setup_repair.sh`，或改選不需模型的紋理補洞。
- **尚未塗選修補區域**：先用筆刷選區；載入新圖或採用修後圖都會清除上一輪選區。
- **顯存不足或修補逾時**：確認沒有其他程序佔用 GPU。SDXL 會先卸載同一應用的試穿模型，
  修補子程序結束後釋放其顯存；下一次試穿需重新載入。子程序超過 5 分鐘會停止。
- **生成或修補很慢**：首次需載入模型，自動原手保留可能執行兩次試穿。
  本機示範 SDXL 約 22–24 秒／次，並非所有圖片與設備的時間保證。
- **修完仍不正確**：比較全圖與局部，調整選區、提示或 Seed；不要僅因請求成功就接受結果。

## 檔案與隱私

預設僅綁定 `127.0.0.1`，不建立 Gradio 公開分享網址，圖片在本機處理。
Gradio 會暫存上傳圖片，執行期間每小時清理超過 24 小時的暫存；停止期間不排程清理。

| 位置 | 內容 |
|---|---|
| `app.py`、`tryon/` | 介面、試穿、手部合成與局部修補 |
| `scripts/`、`tests/` | 安裝、啟動、檢查及回歸測試 |
| `examples/` | 官方範例與來源說明 |
| `weights/` | 生成、姿勢與選用 SDXL 權重及 manifest；Git 忽略 |
| `.cache/` | Python、uv、Hugging Face、Parser 與 Gradio 快取；Git 忽略 |
| `outputs/` | 試穿 PNG 與 JSON 紀錄；Git 忽略，保留至自行刪除 |
| `outputs/repairs/<id>/` | 修前圖、遮罩、工作參數、候選局部、修後圖與紀錄；Git 忽略，保留至自行刪除 |

修補檔案組包含照片，不能將整個工作目錄當成可公開的原始碼上傳。
失敗修補的暫存組會清理；成功產物不自動刪除。
基本權重下載會記錄 revisions 與 SHA-256；Parser 沿用上游快取機制並記錄 snapshot／雜湊。
主要相依版本固定，完整環境版本清單保存在 `.cache/installed-requirements.txt`，尚非完整 lockfile。

## 開發與驗證

在完整環境執行測試及 HTTP／佇列檢查（使用未佔用連接埠）：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.smoke_ui --port 7862
.venv/bin/python -m compileall -q app.py tryon scripts tests
bash -n scripts/setup.sh scripts/start.sh scripts/setup_repair.sh
```

僅驗證 CPU 介面、不下載大型模型：

```bash
uv venv --python 3.12 .venv-ui
uv pip install --python .venv-ui/bin/python -r requirements-ui.txt
.venv-ui/bin/python -m unittest discover -s tests -v
```

UI-only 環境未安裝 OpenCV 時會略過紋理補洞實測；其餘測試包含輸入驗證、手部幾何、
選區合成、失敗路徑及自動載入事件。這些不是跨照片視覺品質測試。

真實 GPU 檢查另外執行：

```bash
.venv/bin/python -m scripts.check
.venv/bin/python -m scripts.smoke_gpu
# 需先安裝 SDXL；在官方範例副本加入測試色塊，再執行局部修補。
.venv/bin/python -m scripts.smoke_repair
```

`check` 檢查 CUDA、BF16、ONNX provider 與基本權重雜湊，不等於推論通過。
`smoke_gpu` 執行官方範例冷／暖啟動；`smoke_repair` 驗證真實重繪及選區外像素保護，
不自動判定修補品質。ONNX session 必須優先使用 CUDA，但個別不支援的算子可能在 CPU 執行。
JSON 的 PyTorch 顯存數字不含 ONNX 或其他程序；整卡使用量請另看 `nvidia-smi`。

## 授權與來源

本地 wrapper 採 Apache-2.0，見 [LICENSE](LICENSE) 與 [NOTICE](NOTICE)。
第三方模型、套件、技能與素材依各自授權；專案授權不等於人物代言、商標或模型使用許可。

- [FASHN VTON v1.5](https://github.com/fashn-AI/fashn-vton-1.5)：固定 commit `7c0f10af3f91ad4048fe9729c470a13ef905d25a`，另見 [模型卡](https://huggingface.co/fashn-ai/fashn-vton-1.5)。
- [SDXL Inpainting 模型卡與條款](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)。
- [官方示範素材來源](examples/SOURCES.md)。
- [GitHub repository](https://github.com/KarlSideProjects/fashn-vton-gradio)。
