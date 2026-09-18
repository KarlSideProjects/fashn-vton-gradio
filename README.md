# fashn-vton-gradio｜本機 AI 試衣間

> **僅限非商業用途。** 歡迎用於研究、教學與交流；完整條款與適用範圍見 [LICENSE](LICENSE) 及 [授權規範](LICENSING.md)。

有一張人物照與一張衣服照，想先看穿搭風格可能長什麼樣子嗎？這個專案讓你在自己的 Linux 電腦上開啟網頁介面，產生一張 AI 換裝預覽。它適合試用虛擬試穿、比較生成條件，或作為理解生成式影像限制的起點；它不是量身工具，也不保證真實衣料、尺寸或合身程度。

本地程式把 **FASHN VTON 1.5** 包成可安裝的 Gradio 介面。按下 **Try On** 時，模型會參考人物照、衣服照和服裝類別，生成一張新的圖片；不是把衣服直接貼到原照上。

![人物輸入範例，來自 FASHN 官方 repository](examples/model.webp)

這張人物照與[商品輸入範例](examples/garment.webp)是官方未修改的輸入素材，不是本專案產生的結果；出處、使用界線與檔案雜湊見 [範例來源](examples/SOURCES.md)。

## 今天可以做什麼

- 上傳人物照與商品照，選擇上衣、下身或連身衣，以及商品是模特兒穿著或平拍；每次產生一張可下載的預覽圖。
- 在同一介面選 CPU 或 NVIDIA GPU。GPU 不可用時，程式會明確說明，並由你決定是否改選 CPU。
- 用同一組輸入比較不同生成結果，或先閱讀不需要程式與微積分基礎的 [生成模型圖解與小實驗](docs/model-learning-guide.md)。

本專案的本地貢獻是介面、安裝流程、裝置切換、取樣條件與驗證文件；模型與推論程式仍來自 FASHN AI，沒有重新訓練或修改模型權重。

## 目前結果與限制

目前可執行的流程是「兩張照片 → 一張原始換裝圖」，沒有候選挑選、自動重試或局部修補。程式會檢查圖片、基本設定與模型權重；現有測試和 smoke 檢查確認這些行為，**不代表真實照片的生成品質已被普遍驗證**。完整檢查範圍與歷史結果見 [操作與驗證紀錄](docs/usage.md)。

一份固定人物／商品照與設定的控制變因紀錄顯示：在該案例中，改變生成時的隨機起點後，花裙得以保留；只改運算精度則沒有解決問題。這是可追查的單案例診斷，不是任何做法都會讓所有服裝改善的證明。方法、數值與邊界見 [控制變因報告](docs/space-controlled-diagnosis.md)。

請只上傳自己有權使用的照片，並人工檢查成品。手指、臉、背景、Logo、未指定更換的衣物與遮擋關係都可能改變；不要把結果用來判斷尺寸、合身度或商品真實外觀。照片、權重、快取與輸出會留在本機專案資料夾，請勿加入 Git 或公開上傳。

## 開始試穿

需要 Linux、Python 3.12、Git 與 [uv](https://docs.astral.sh/uv/)。第一次安裝會下載套件與模型，需要網路和數 GiB 的可用 RAM；若用 GPU，還需要可用的 NVIDIA 驅動與足夠的 VRAM。CPU 能執行，但可能需要很久；完整環境需求與疑難排解在 [操作手冊](docs/usage.md)。

```bash
git clone https://github.com/KarlSideProjects/fashn-vton-gradio.git
cd fashn-vton-gradio
bash scripts/setup.sh cuda
bash scripts/start.sh
```

只有 CPU 時，把安裝指令改成 `bash scripts/setup.sh cpu`。接著開啟 `http://127.0.0.1:7860`，先把「推論裝置」改為 **CPU**，選擇兩張範例、`tops` 與 `model`，再按 **Try On**。第一次使用會先載入模型；預設只監聽自己的電腦，不開公開分享。可用 `bash scripts/start.sh --port 7861` 改用其他連接埠。

第一次請先保留介面的進階預設設定，把它當作可比較的起點，而不是品質保證。文件記錄的一次同組照片執行，在 RTX 4060 Ti 約 47 秒、Ryzen 9 7945HX CPU 約 23 分 32 秒（均含載入）；這不是跨硬體基準或速度承諾。

## 想深入一點

- [操作手冊](docs/usage.md)：完整安裝、參數、測試與疑難排解。
- [控制變因報告](docs/space-controlled-diagnosis.md)：為何同一 Seed 在不同條件下可能得到不同成品。
- [初學者理論指南](docs/model-learning-guide.md)：從影像數值、噪聲、Seed 到小實驗的白話說明。
- [理論來源筆記](docs/fashn-theory-sources.md)：區分公開模型資料與教學用簡化說明。

## 探索提案（不是既有承諾）

給想判斷初始噪聲做法是否能跨案例成立的研究者或本機整合者：以取得公開或明確同意使用的多組人物／服裝照片及多個 Seed，沿用目前的控制變因方法，產出版本化的條件、結果表與人工檢核紀錄。完成的可觀察標準是每個比較都有相同基準條件、模型／權重版本與明確的品質檢核欄位；代價是準備可合法使用的資料並逐筆檢視。在此之前，不把單案例結果延伸為通用品質宣稱。

## 來源、歸屬與授權

本地入口改編自固定版本的 [hemil124/virtual-tryon Space](https://huggingface.co/spaces/hemil124/virtual-tryon/tree/29f4ad42a29af63c71a7bc9e53ea7cd5342694c1)，模型與推論程式來自固定版本的 [FASHN AI FASHN VTON 1.5](https://github.com/fashn-AI/fashn-vton-1.5/tree/7c0f10af3f91ad4048fe9729c470a13ef905d25a)。本地新增／修改內容採[非商用研究授權](LICENSE)；上游 Apache-2.0、[NOTICE](NOTICE) 與[範例來源](examples/SOURCES.md)仍有效，模型、第三方套件、照片、肖像與商標各依其原有條件使用。本專案與 hemil124、FASHN AI 沒有隸屬或背書關係。
