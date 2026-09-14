# FASHN Gradio Implementation Plan

**Goal:** 在 16GB NVIDIA GPU 上提供本地繁中雙圖試穿介面。
**Architecture:** Gradio → validated request → serialized GPU pipeline → PNG + JSON。
**Tech Stack:** Python 3.12、Gradio 5.50.0、PyTorch、FASHN VTON 1.5。
**Spec:** docs/design.md

- [x] 先寫圖片/參數驗證、結果儲存的 unittest，執行確認缺少實作。
- [x] 實作 tryon/core.py，保留 RGB 轉換、EXIF 校正、參數範圍及唯一輸出路徑。
- [x] 實作 tryon/engine.py：延遲載入 CUDA-only 管線、lock、真實計時和記憶體資料。
- [x] 實作 app.py：繁中表單、範例、佇列、清除舊結果與顯式錯誤。
- [x] 實作 setup/start、model download、GPU smoke test 與 CI CPU 檢查。
- [x] 執行 unittest、Gradio HTTP/事件驗證與語法檢查；將未具備的 GPU 驗收列入報告。
- [x] 打包可交付專案。
- [ ] GitHub 目標可用時推送；目前 blocked，不修改不相關 repository。

驗證及交付限制詳見 verification.md。GitHub 推送因缺少目的 repository 尚未執行。
