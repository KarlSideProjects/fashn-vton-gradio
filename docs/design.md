# 本地虛擬試穿設計

需求已由使用者確認：FASHN VTON v1.5 + Gradio，Ubuntu / RTX 4060 Ti 16GB，網路示範素材。

採獨立 Python 應用，不修改上游模型。以固定上游 commit 安裝推論套件。
Gradio 提供繁中人物/商品上傳、衣物類別、商品照片類型、步數與 seed。
預設 30 步、seed 42、segmentation_free=True、num_samples=1。
GPU 管線延遲載入且常駐；thread lock 與 Gradio concurrency_limit=1 序列化。
沒有 CUDA、BF16 或權重時明確失敗，不切換 CPU 或假圖。
PNG 和 JSON 以 UUID 儲存，記錄輸入像素雜湊、參數、時間、GPU allocated/reserved 峰值。
預設 localhost、share=False；資產、快取、暫存與模型保留在專案資料夾。
非多人公開服務；照片會暫存在本機，outputs 保留至使用者刪除。

驗收：CPU 層檢查圖片/參數與結果儲存；真實 Gradio 啟動與失敗路徑；GPU 測試獨立執行。
不以 CPU 測試宣稱 GPU 生成通過。以官方 examples/data 的真人與衣服作第一組驗收。
