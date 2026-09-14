# 本地 AI 試衣間 — FASHN VTON v1.5 + Gradio

繁體中文的本地虛擬試穿工具：人物照片 + 商品照片 → 試穿 PNG。
目標設備：Ubuntu、NVIDIA RTX 4060 Ti 16GB、32GB 系統 RAM、30–50GB 可用磁碟。

**目前狀態：CPU/Gradio 檢查已執行；尚未在 GPU 上生成圖片。**
本專案不是已驗收的生產服務。詳見 [驗證紀錄](docs/verification.md)。

## 快速啟動

先在你的 Ubuntu 安裝好 NVIDIA 驅動，確認 `nvidia-smi` 成功。
另需 `git` 及 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
在專案根目錄執行：

```bash
bash scripts/setup.sh
bash scripts/start.sh
```

開啟 http://127.0.0.1:7860。點選官方範例，保留「上衣 / 模特兒穿著 / 30 步 / Seed 42」，按「開始試穿」。

安裝腳本建立專案內 Python 3.12 `.venv`，安裝 PyTorch 2.8.0 CUDA 12.8 wheel、
Gradio 5.50.0 及固定 commit 的 FASHN。CUDA wheel 仍需要相容的系統 NVIDIA 驅動。
不依賴 Ubuntu 系統 Python 的版本，也不修改系統 Python。
第一次需要連線下載套件與模型；任何一步失敗即停止，不換模型或改用 CPU。

`weights` 保留生成與姿勢權重；Human Parser 權重存在 `.cache/huggingface`。
程式啟動後設定 Hugging Face 離線模式，請先完成下載。
第一次下載會解析 Hugging Face revisions 並記錄 SHA256，重跑下載沿用已記錄的生成/姿勢 revisions。
Parser 使用上游套件的快取行為，快取 snapshot 與檔案雜湊也記錄在 manifest；不要單獨更新它。
成功安裝後，實際相依版本保存在 `.cache/installed-requirements.txt`。
主要版本已固定，尚未完成完整 GPU 環境 lock 與相容性實測。

## 使用方式

1. 人物照上傳到左側；商品照上傳到中間。
2. 選擇「上衣 / 下身 / 連身衣」。
3. 商品穿在別的模特兒身上選「模特兒穿著」，平拍單件商品選「平拍商品」。
4. 建議從 30 步開始，固定 seed 做品質比較。
5. 完成後下載 PNG 與 JSON 紀錄，也可直接在 `outputs/` 找到。

單一 GPU 管線常駐、每次一張、最多 8 個等待任務。不要用多個程序同時啟動同一張 GPU。
輸出不做生成式放大，避免再改動衣服細節。上游生成畫布是 576×864，去除 padding 後尺寸可能不同。
背景、臉部、體型、Logo 和文字並非保證完全不變；真人與動漫/Q版的效果需分開驗收。
此功能是穿搭外觀生成，不是實體尺寸或布料物理模擬。

## GPU 驗收

```bash
.venv/bin/python -m scripts.check
.venv/bin/python -m scripts.smoke_gpu
```

`check` 檢查 CUDA、BF16、ONNX provider 與權重雜湊；不把它視為推論通過。
`smoke_gpu` 真正執行兩次官方範例，分別記錄冷啟動、暖啟動的耗時與結果。
缺少 GPU / 權重會以非零狀態結束，沒有假圖或跳過後顯示通過的模式。
ONNX session 實際載入時必須優先使用 CUDA；明確禁止 session 執行失敗後切換 provider。
部分不支援 CUDA 的 ONNX 算子仍可能由 Runtime 配置至 CPU，本專案不宣稱每個算子都在 GPU 上。

JSON 中 PyTorch 顯存數字不含 ONNX 與其他程序。需量測整張卡時，另開終端：

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,power.draw --format=csv -l 1
```

驗收時檢查：臉部、體型、原衣殘留、手指/手臂遮擋、衣服圖案、Logo、袖口與下襬。
先測官方範例，再測自己 30 組人物/商品配對。程式完成生成不代表視覺品質合格。

## 本地資料

| 位置 | 內容 |
|---|---|
| `app.py` | 繁中 Gradio 介面 |
| `tryon/` | 圖片處理、CUDA 管線、輸出紀錄 |
| `scripts/` | 安裝、啟動、下載、環境檢查、GPU smoke test |
| `examples/` | 官方示範人物/商品照片及來源 |
| `weights/` | 模型與 SHA256 manifest；不進 Git |
| `.cache/` | Python/uv/Hugging Face/Gradio 暫存；不進 Git |
| `outputs/` | PNG 與 JSON；不進 Git |

預設僅綁定 `127.0.0.1`，不產生 Gradio 公開分享網址，不提供任意遠端 URL 下載。
Gradio 上傳暫存會寫入本機，執行期間每小時清理超過 24 小時的暫存；停止期間不會排程清理。
`outputs` 持續保留，需自行刪除。不應將整個資料夾連同快取公開上傳。
程式不另存原始照片永久副本，但 Gradio 會暫存圖片；JSON 保存正規化後的像素雜湊。

## CPU 檢查

CPU CI 只安裝介面相依套件，不下載大型模型：

```bash
uv venv --python 3.12 .venv-ui
uv pip install --python .venv-ui/bin/python -r requirements-ui.txt
.venv-ui/bin/python -m unittest discover -s tests -v
```

這些檢查涵蓋圖片驗證、EXIF/透明度處理、參數限制、檔案儲存、真實 Gradio 組態及失敗路徑。
測試採 unittest；測試圖片是程式產生的小型 fixture，沒有用來冒充模型產物。

## GitHub

公開 repository：[KarlSideProjects/fashn-vton-gradio](https://github.com/KarlSideProjects/fashn-vton-gradio)。

## 授權與來源

本地 wrapper 採 Apache-2.0；第三方模型、套件、素材依各自授權。
保留 [LICENSE](LICENSE) 與 [NOTICE](NOTICE)。官方範例詳見 [來源](examples/SOURCES.md)。

- FASHN: https://github.com/fashn-AI/fashn-vton-1.5
- 固定 commit: `7c0f10af3f91ad4048fe9729c470a13ef905d25a`
- 模型卡: https://huggingface.co/fashn-ai/fashn-vton-1.5

示範人物與商標僅用於技術測試；專案授權不等於人物代言或商標使用授權。

## 自動手部保留版（實驗性更新）

操作不變：兩張圖片 → 開始試穿。重用現有 Human Parser（hands=13），
比較原圖與生成圖的手部分割、位置及衣物重疊，通過保守門檻後將原手合成回去。
羽化僅限原手遮罩內側，避免貼回舊衣服。若不適合合成，自動以 seed+1 重試一次。
兩次都不適合則明確失敗，不輸出未修復候選；JSON 記錄實際 seed、嘗試與門檻數值。

這一版沒有生成式局部補手。新露出手、手藏在口袋、袖口關係不明、手部未偵測到等情況
會失敗（包括本來沒有可見手部的照片）。分割模型若誤判，仍可能產生不正確的合成。
IoU 等門檻只是幾何啟發式，不是手指数量或解剖正確性的品質判定。
目前未在 GPU / 真實失敗照片上校準或驗收，請不要當成已修復手指缺陷的正式版本。
計時現在包含人物解析、最多兩次生成、結果解析與合成；不再僅是一次模型推論時間。
無額外模型下載；保留原有 weights 與 .cache，更新程式即可。
