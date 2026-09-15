# hemil124/virtual-tryon：模型與範例查核

查核日期：2026-09-15。範圍為公開原始碼、Hugging Face API 與 Gradio 設定；未上傳照片、未呼叫生成。原始碼固定於 `29f4ad42a29af63c71a7bc9e53ea7cd5342694c1`；查核當下 API 回報相同部署版本、`RUNNING`、`cpu-basic`。[Space API](https://huggingface.co/api/spaces/hemil124/virtual-tryon)

## 模型與預設參數

`app.py` 從 `fashn-ai/fashn-vton-1.5` 下載 `model.safetensors`，姿態模型來自 `fashn-ai/DWPose`，並建立 `TryOnPipeline(weights_dir=..., device="cpu")`。它確實配置模型推論，沒有在應用入口呼叫另一個商用生成 API。[app.py](https://huggingface.co/spaces/hemil124/virtual-tryon/blob/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/app.py)

| 項目 | Space 預設 |
| --- | --- |
| 類別 / 服裝照片類型 | `tops` / `model` |
| Sampling Steps | 30；介面範圍 10–50 |
| Guidance Scale | 1.5；介面範圍 1.0–3.0 |
| Seed | 42；空值或負值也改為 42 |
| Segmentation-Free | `True` |
| 每次生成 | 1 張；直接回傳 `result.images[0]` |

介面預設同時經目前運行中的 `/config` 核對；每次樣本數與 seed 正規化則由入口程式確認。[Gradio 設定](https://hemil124-virtual-tryon.hf.space/config)、[app.py](https://huggingface.co/spaces/hemil124/virtual-tryon/blob/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/app.py)

**版本限制：** `requirements.txt` 安裝的是未鎖 commit 的 `git+https://github.com/fashn-AI/fashn-vton-1.5.git`，沒有以 `-e .` 安裝內附套件；權重下載也沒有指定 revision。因此公開 Space commit 能固定入口程式，不能單憑它確定部署時安裝的 pipeline commit、套件版本或快取權重雜湊。內附 `src/` 的分析不能直接當作執行環境鑑識結果。[requirements.txt](https://huggingface.co/spaces/hemil124/virtual-tryon/blob/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/requirements.txt)、[app.py](https://huggingface.co/spaces/hemil124/virtual-tryon/blob/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/app.py)

## 範例是什麼

- 操作介面的 Person Example / Garment Example 分別載入 `examples/data/model.jpeg` 與 `garment.jpeg`，只連接輸入欄位。即時設定也顯示兩個 `load_example` 事件只寫入輸入照片；`try_on` 按鈕才連接結果圖與狀態欄。這不是預先生成結果的範例快取流程。[app.py](https://huggingface.co/spaces/hemil124/virtual-tryon/blob/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/app.py)、[Gradio 設定](https://hemil124-virtual-tryon.hf.space/config)
- README 頂端展示圖另外來自 `static.fashn.ai/repositories/fashn-vton-v15/results/hero_collage.webp`，屬於靜態展示素材；README 沒提供這張拼圖每個案例的完整生成參數與執行紀錄，不能拿它證明本 Space 的某次即時輸出品質。這不代表使用者看到的就是該拼圖。[README](https://huggingface.co/spaces/hemil124/virtual-tryon/blob/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/README.md)

## 修補與後處理

入口程式將影像轉成 RGB，呼叫 pipeline 後直接回傳第一張圖片；沒有額外的人臉修復、手部修復、超解析或挑選最佳 seed 步驟。[app.py](https://huggingface.co/spaces/hemil124/virtual-tryon/blob/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/app.py)

內附 pipeline 的完整生成路徑為：等比例預縮圖 → DWPose 與人體解析 → 人物／服裝條件圖 → 縮放補邊 → Euler 採樣 → 轉 PIL → 去補邊。其模型輸入為寬 576 × 高 864；預設最後一步跳過 CFG、`time_shift_mu=1.5`。生成後沒有另外的手臉重繪或原圖貼回處理；人物分割關閉選項是前處理控制，不是生成後修復。這些描述限定於內附原始碼，受上述安裝版本限制。[pipeline.py](https://huggingface.co/spaces/hemil124/virtual-tryon/blob/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/src/fashn_vton/pipeline.py)、[TryOnModel](https://huggingface.co/spaces/hemil124/virtual-tryon/blob/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/src/fashn_vton/tryon_mmdit.py)

## 可以下的結論

公開證據支持「這是 FASHN VTON 1.5 的 CPU 介面，提供一般生成參數與輸入範例」。沒有找到入口另外使用更強模型或專用手臉修復的證據。未執行同一組照片、參數與已確認版本的對照生成，因此不能判定它比本機模型更好，也不能將使用者觀察到的差異歸因於範例快取。

## 與本機的核對結果

以下由主代理完成檔案、權重紀錄與圖片核對，納入本次研究；未更動應用程式原始碼。

- 主代理將此 Space commit 的 `src/fashn_vton/` 下 20 個 Python 檔案與本機 `.venv/lib/python3.12/site-packages/fashn_vton/` 逐一於記憶體比對，全部文字一致。這表示公開內附 pipeline 與本機安裝版一致；仍不能消除 Space 未鎖定安裝版本的不確定性。[Space 檔案樹](https://huggingface.co/api/spaces/hemil124/virtual-tryon/tree/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1?recursive=true&expand=false)
- 本機 manifest 記錄的模型 SHA-256 為 `d6cd38286885bc29fa487ea9383f80ffeb95862e7747c630d42c5d3c05bdd35a`，與查核時官方模型 API 的 `model.safetensors` LFS oid 相同。這是本機紀錄與官方檔案識別碼的核對，Space 快取權重內容未驗證。[官方模型檔案 API](https://huggingface.co/api/models/fashn-ai/fashn-vton-1.5/tree/main)
- Space 實際介面使用的兩張 JPEG 都是 768 × 1024：人物為淺色背景、灰色長袖上衣與酒紅色褲子的女子；服裝為紅色 Adidas 長袖商品照。人物／服裝 SHA-256 分別為 `98bbd2986b0d50f934145235da693fcb092537520c1175396922b7ade3b4a06e`、`e4001842c2b9c87e6bfab5c5b9e99d6b29b136a1147ed4c2d314d03cd65`。[人物 JPEG](https://huggingface.co/spaces/hemil124/virtual-tryon/resolve/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/examples/data/model.jpeg)、[服裝 JPEG](https://huggingface.co/spaces/hemil124/virtual-tryon/resolve/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/examples/data/garment.jpeg)
- 本機 WEBP 範例則是戴帽坐姿男子（832 × 1248）及黑色 T 恤（576 × 864）；它們與 Space 倉庫內未被該介面引用的同名 WEBP 位元組雜湊完全相同。比較雙方「預設範例」時，要先確認是 JPEG 還是 WEBP。[人物 WEBP](https://huggingface.co/spaces/hemil124/virtual-tryon/resolve/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/examples/data/model.webp)、[服裝 WEBP](https://huggingface.co/spaces/hemil124/virtual-tryon/resolve/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/examples/data/garment.webp)
- 主代理檢視使用者本機照片：960 × 1708、樹景複雜背景、短袖換長袖、拳頭接近軀幹。Space JPEG 是長袖換長袖與單純背景。輸入難度不同是合理待驗假說，尚未透過控制變因證明它造成品質差異；使用者照片未上傳遠端。
- 本機與公開 pipeline 的模型輸入尺寸、30 steps、CFG 1.5、seed 42 相同，但 `_sample` 在指定 `device` 與 `dtype` 上直接建立 `torch.randn`。CPU FP32 與 GPU BF16 並非同一組執行條件；相同 seed 本身不足以證明初始噪聲或最終像素相同。[pipeline.py](https://huggingface.co/spaces/hemil124/virtual-tryon/blob/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1/src/fashn_vton/pipeline.py)
