# FASHN VTON — 操作、環境與驗證

回到 [初學者理論指南](../README.md)。這份文件保留完整操作與實測紀錄。

直接採用 [hemil124/virtual-tryon 的 app.py](https://huggingface.co/spaces/hemil124/virtual-tryon/blob/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/app.py)，
固定來源 commit `29f4ad42a29af63c71a7bc9e53ea7cd5342694c1`，沿用它的單次生成流程：
人物＋衣服 → FASHN VTON 1.5 → 直接顯示模型輸出。介面已升級至 **Gradio 6.13.0**，
可選擇 **GPU / CUDA** 或 **CPU**，不需要啟動兩個服務。

**不再包含我方手部／臉部合成、幾何攔截、自動重試、局部重繪或候選比較。**
沒有隱藏的修補步驟，也沒有額外修補模型需要安裝。

## 安裝與啟動

Linux、Python 3.12、Git 與 [uv](https://docs.astral.sh/uv/)；首次安裝需網路下載套件與權重。
GPU 模式需要 NVIDIA 驅動與 CUDA PyTorch；CPU 模式不需要 GPU。
模型及中間 tensors 會佔用數 GiB 記憶體，請留足可用 RAM／VRAM。

```bash
bash scripts/setup.sh cuda  # 預設；安裝後可切換 GPU 與 CPU
# 僅需 CPU 的電腦：bash scripts/setup.sh cpu
bash scripts/start.sh
# 自訂連接埠：bash scripts/start.sh --port 7861
```

開啟 http://127.0.0.1:7860 。已有 CUDA 環境及權重時，升級介面即可，不必重下載模型：

```bash
uv pip install --python .venv/bin/python -r requirements-ui.txt
```

舊程序不會因修改檔案而自動更新；先在原終端停止它，再執行 start.sh。
CPU 版會明顯慢於先前 GPU 版；首次生成另需載入模型，不保證只需幾分鐘。
本機 Ryzen 9 7945HX 實測同一組照片、30 步 CPU 生成含模型載入約 **23 分 32 秒**；
這是一次實測，不是所有硬體或照片的時間保證。

安裝保留已固定的 FASHN pipeline commit `7c0f10af3f91ad4048fe9729c470a13ef905d25a`。
其 20 個 Python 檔案已與該 Space 內附版本核對一致。Space 的安裝來源未鎖版本，
因此這不等於重製了遠端完整環境。Gradio 6.13.0 支援目前固定的 Hugging Face Hub 0.36.2；
未直接升到 6.27.0，因其要求 Hub >=1.16，與 Transformers 4.57.1 的 Hub <1 限制衝突。
版本依賴見 [6.13.0 metadata](https://pypi.org/pypi/gradio/6.13.0/json) 與
[6.27.0 metadata](https://pypi.org/pypi/gradio/6.27.0/json)。新版 CSS／theme 設定依
[Gradio 6 遷移指引](https://www.gradio.app/guides/gradio-6-migration-guide) 移至 launch。

## 使用

1. 上傳人物照片與衣服照片。
2. 選 `tops`（上衣）、`bottoms`（下身）或 `one-pieces`（連身）。
3. 商品照片由模特兒穿著時選 `model`；平拍商品選 `flat-lay`。
4. 選擇「推論裝置」（預設 GPU），按 **Try On**，完成後從結果圖下載。

沿用 Space 預設：30 steps、CFG 1.5、Seed 42、segmentation-free 開啟、每次一張。
Seed 留空或負數會改為 42。一次處理一個請求，最多 10 個等待任務。
不會自動換 Seed、挑選候選或修補生成圖。步數更高不保證品質更好。

### GPU／CPU 差異

| 模式 | 推論 | 初始噪聲 | 同照片實測（30 步、含載入） |
| --- | --- | --- | --- |
| GPU | CUDA；支援時使用 BF16 | CPU FP32 產生後轉 GPU dtype | 約 47 秒（RTX 4060 Ti） |
| CPU | 原 Space 的 CPU／FP32 | 原生 CPU FP32 | 約 23 分 32 秒（Ryzen 9 7945HX） |

**不是 GPU 本身較差。** 已驗證這組照片主要受初始噪聲差異影響；GPU 模式因此採用
CPU 初始噪聲，沒有額外修圖。精度、前處理運算及硬體仍有差異，所以相同 Seed
不保證逐像素相同，也不保證每張照片都改善。

切換選項只影響下一次生成。模型載入與生成共用一把鎖；切換時先卸載原模型，
只快取目前裝置的一份 pipeline。GPU 不可用或生成失敗時明確報錯，不暗中回退 CPU。
狀態欄顯示實際裝置、含載入耗時及 CPU 初始噪聲標記。

結果是穿搭預覽，不是實際合身程度或解剖正確性的保證；模型仍可能改動手、臉、
背景、未指定更換的衣物、文字與 Logo。確認有權使用上傳照片，並自行檢查成品。

## 與來源程式的必要差異

- 權重下載移到 setup；生成期間離線執行，不連接外站生成服務。
- 綁定 `127.0.0.1`、不開 share；保留 `--host`／`--port` 與 20 MB 上傳限制。
- 保留參數及圖片尺寸上限，避免 API 請求繞過介面耗盡資源。
- 快取留在 repo 的 `.cache/`，停用遙測；Gradio 運行時定期清理過期圖片快取。
- 範例沿用 repo 既有官方 WEBP；不是對方介面使用的 JPEG。

CPU 模式直接使用上游 pipeline。GPU 使用 `tryon/sampling.py` 的小型 sampler override：
沿用固定版本的 Euler／CFG，只把初始噪聲放到 CPU FP32 產生再轉回 GPU。
沒有修改模型權重或 FASHN 安裝套件檔案，也沒有全域攔截 `torch.randn`。

## 失敗經驗：先重現基準，再考慮修補

最初看到手指或衣料不自然，我方先加入原手／臉部貼回、袖口幾何檢查、自動重試，
後來又加入 SDXL／SD1.5 局部重繪、ControlNet、外觀參考及候選操作。
這增加了操作與失敗點，卻沒有先確認為什麼相同照片在對方流程中更自然。
幾何門檻還曾造成兩次生成後仍拒絕輸出；遮罩重疊率不等於手指或衣料正確率。

2026-09-15 的同照片控制實驗發現：

- 完全不做後處理，也能逐像素重現我方問題圖；該案例不是修補造成的，但修補也沒解決它。
- 保持條件圖、模型及 GPU／BF16 運算，只改 CPU 產生初始噪聲，花裙便保留，上衣也更接近對方。
- 保留原本噪聲，只改 FP32 運算仍失敗；關閉 segmentation-free 也沒有改善。
- 同 Seed 不代表 CPU／GPU 產生相同噪聲；不能只看介面參數相同就認定推論條件一致。

因此採用對方的簡單主流程，移除自製修補。這不是證明 CPU 噪聲對所有照片更好，
也不是宣稱原模型不再有瑕疵；只是不再用未經驗證的修補掩蓋尚未釐清的差異。
**先固定照片、版本與推論條件重現問題；有跨案例證據後，才值得增加複雜度。**

六組實測、數值及限制見 [控制變因報告](space-controlled-diagnosis.md)，
對方原始碼查核見 [Space 研究](hemil124-space-research.md)。
其他 `docs/` 修補研究保留為歷史紀錄，不代表目前功能；舊診斷腳本依賴已移除的 wrapper，
若需重跑舊流程應從備份還原到獨立目錄，不要混入新版。

## 驗證

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.smoke_ui --port 7862
.venv/bin/python -m scripts.check
.venv/bin/python -m scripts.smoke_inference
.venv/bin/python -m scripts.smoke_inference --device cuda
```

前兩項不需模型，檢查直接回傳、只推論一次、CPU 設定、輸入限制及真實 HTTP／佇列。
check 核對環境與權重；smoke_inference 才會真正執行 30 步生成（預設 CPU），結果另外存到
`outputs/space-smoke/<id>/`，須檢視圖片，不能把請求成功當作品質認證。
可加 `--person /path/person.jpg --garment /path/garment.jpg` 測試自己的同一組照片。

2026-09-15 替換後驗證：6 項單元測試、HTTP／佇列檢查、權重檢查、Python 編譯及
Shell 語法檢查通過。新版同照片真實 CPU 生成完成，未套用任何後處理；視覺檢查確認
花裙保留，上衣接近對方成品。與對方全圖 RGB MAE 為 2.266（0–255），僅代表此圖
相似度，不是通用品質認證；例如原人物手持物仍消失，不能宣稱模型已無瑕疵。
本機證據：`outputs/space-smoke/9364358b086c415fbc7696436bff17c2/` 的 PNG 與 JSON。

加入切換後的 GPU 真實生成約 46.975 秒，結果保留花裙並接近對方；本機證據：
`outputs/space-smoke/e614f01d115f4e348b3201f6f0cf1b85/`。11 項測試驗證裝置切換、
請求互斥、無 GPU 不回退、CPU 採樣公式一致及多 Seed 初始噪聲轉換。
僅安裝 UI 相依的 CI 會明確跳過兩項需要 Torch／FASHN 的數值測試。
另已實際驗證 CPU → GPU → CPU → GPU 的模型載入與 provider 切換；切回 CPU 後，
本測試程序的 PyTorch GPU 配置量回到 0。新版瀏覽器兩個裝置選項可切換，390px
視窗未見水平溢出；介面截圖保留於 `outputs/gradio6-device-switch.png`。

## 檔案與隱私

一般介面結果由 Gradio 快取提供下載，不再另存我方的生成 JSON 或修補檔案組。
`outputs/` 舊結果、`weights/` 舊修補權重及 `.venv-repair/` **均未刪除**，不會被新版載入。
程式替換前的未提交實作已備份於本機 `.cache/pre-space-wr6Kzp/source.tar.gz`，可恢復。
快取、備份、權重與私人圖片皆由 Git 忽略，不應公開上傳。

## 授權

本機入口改編自 hemil124 的 Space；FASHN VTON 1.5 來自 FASHN AI。
保留 Apache-2.0 [LICENSE](../LICENSE) 與來源／修改說明 [NOTICE](../NOTICE)。
此專案與 hemil124、FASHN AI 沒有隸屬或背書關係；各模型與相依套件沿用各自授權。
