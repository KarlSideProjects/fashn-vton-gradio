# 驗證紀錄 — 2026-09-14

## 已驗證

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
