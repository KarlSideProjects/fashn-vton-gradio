# 同照片：Space 與本機的控制變因對照

日期：2026-09-15。使用者提供對方 WEBP 與我方 PNG，並確認使用同一組人物／商品照片。
本輪只執行本機診斷，不修改 app 預設、不上傳照片、不重新訓練或安裝模型。

## 固定輸入與失敗重現

從本機 Gradio 快取找回人物與商品原檔，以正規化 RGB 像素雜湊核對既有生成紀錄：

- 人物：`dee83cce8516b1caf37e721876c6b10054079d238a644679f31d6de9c053a2ad`
- 商品：`e394432f7aba5e01e568c9ebbc5b0216d1847b30128bbc620dfe25e68b9d2a68`
- 我方輸出：`99a325bb9725893800fd95e575ad785ea814639aefb317f4cdd14fe2052c19e7`

固定 tops／model、30 步、CFG 1.5、Seed 42、576×864 模型畫布，去補邊輸出 485×864。
直接呼叫既有 pipeline，移除 UI、手部／臉部貼回及重試，重新生成後仍與使用者提供的我方
PNG 逐像素相同。原有紀錄 `outputs/48ff6216e73e4be8bb5991def2ef222d.json` 也記錄修補
改動 0 像素。這排除了「後加的細節修補造成這張圖不自然」。

診斷入口：`PYTHONPATH=. .venv/bin/python .cache/debug_space_compare.py baseline`。
已實際執行，輸出 `matches_reported_bad_result=true`，並由花裙檢查預期失敗：
`AssertionError: REPRODUCED: flower skirt was replaced by dark trousers`。

花裙檢查只用這個固定案例的裙身內部 `[x=90:200,y=550:700]` 紅色花紋比例：原圖 25.93%、
對方 25.73%、我方 1.49%。它能抓出這次裙子變褲子的症狀，但不是通用服裝／解剖品質判定；
任何保留花紋的結果仍需檢查衣料、形狀、遮擋和背景。

## 實驗隔離

重用原 pipeline 的前處理及採樣。只在獨立診斷子程序的 `_sample` 範圍攔截一次初始
`torch.randn`，確認只呼叫一次；未修改安裝套件或線上服務。
每個變體儲存實際條件 tensors、初始 noise、輸出、雜湊、耗時和顯存。

1. baseline：原生 CUDA／BF16 初始噪聲與運算。
2. cpu_noise：CPU／FP32 產生 Seed 42 噪聲後轉 GPU／BF16；其餘不變。
3. fp32_cpu_noise：重用第 2 組噪聲與條件圖的相同數值，只改 GPU／FP32 運算。
4. masked：重用原生 CUDA／BF16 噪聲，只關閉 segmentation-free。
5. fp32_gpu_noise：重用 baseline 的相同噪聲與條件圖數值，只改 GPU／FP32 運算。
6. fp32_cpu_native：CPU／FP32 噪聲不先量化為 BF16，使用原生 FP32 條件圖與 GPU 運算；
   這是接近 Space 數值設定的組合對照，不是單變因測試，也不是遠端 CPU 執行的完整複製。

本機 `model.safetensors` 的 366 個 tensors 全為 BF16。將模型切成 FP32 是同一批已儲存
權重的精確升型，沒有憑空恢復不存在的高精度權重。FP32 組仍使用本機 GPU 的姿態／分割，
且 Space 的實際套件版本與使用者此次參數未取得，不能宣稱位元級重現遠端環境。

本次檔案均在 `outputs/experiments/space-controlled-20260915/`，包含私人照片，不提交 Git。
`inputs/` 保存固定人物／商品、對方成品與原本我方成品；各變體的 `report.json` 可追溯實測。

## 實測結果

| 變體 | 花裙保留檢查 | 與對方全圖 MAE ↓ | 上衣區 MAE ↓ | 單次秒數 |
| --- | --- | ---: | ---: | ---: |
| baseline | 失敗 | 12.725 | 16.513 | 30.614 |
| cpu_noise | 通過 | 2.358 | 1.818 | 30.776 |
| fp32_cpu_noise | 通過 | 2.297 | 1.761 | 103.530 |
| masked | 失敗 | 17.439 | 19.429 | 30.793 |
| fp32_gpu_noise | 失敗 | 12.749 | 16.642 | 104.138 |
| fp32_cpu_native | 通過 | 2.297 | 1.758 | 104.158 |

MAE 為 0–255 RGB 像素的平均絕對差，上衣區固定 `[x=140:370,y=280:550]`。
這只衡量與使用者提供之對方成品的相似度，不是通用品質分數；對方 WEBP 也包含壓縮誤差。
耗時為同一台 GPU 上各一次 pipeline 呼叫，不含載入模型，不是多次效能基準。

逐張視覺檢查：CPU 噪聲三組均保留花裙，上衣陰影與皺褶明顯更接近對方成品。
原 CUDA 噪聲兩組仍把裙身變成深色褲裝；masked 還出現較大的背景、地板及服裝改動。
CPU 噪聲結果並非完美：例如原人物手持的冰淇淋仍消失，對方成品也有這個問題。

### 結論與邊界

1. **這個固定案例的主要差異是初始噪聲，而不是手部修補或單純 BF16 運算。**
   cpu_noise 與 baseline 的四組條件 tensor 雜湊完全相同，只有初始噪聲改變，花裙便保留。
   反向對照保留 baseline 噪聲、只升 FP32 運算仍失敗，支持此因果判斷。
2. **不需要把整個模型改成 CPU 推論。** CPU 只產生噪聲、GPU 繼續 BF16 推論，
   這次約 31 秒即可取得接近對方的結果。固定相同噪聲再改 FP32 約 104 秒，視覺差異很小。
3. **這不是 CPU 噪聲普遍更美的證明。** 它改變了同一 Seed 對應的隨機樣本；目前只有
   一組照片及一個 Seed，不能當作通用局部修補、手指修正或跨照片品質保證。
4. 本輪沒有修改線上預設。後續最小改善候選是明確固定噪聲生成方式並記錄其版本；
   上線前需用多組人物／服裝與多個 Seed 驗證，並保留舊紀錄的重現方式。
   不應因此直接關閉 segmentation-free 或全面改 FP32。

建議查看 `outputs/experiments/space-controlled-20260915/cpu_noise/result.png`。
完整六組雜湊隔離、基線逐像素重現與固定案例花裙檢查已通過。可在 repo 重跑以下
不需 GPU 的結果驗證（依賴本機已保存之實驗檔，不是 CI 測試）：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/experiments/space-controlled-20260915')
names = ['baseline', 'cpu_noise', 'fp32_cpu_noise', 'masked', 'fp32_gpu_noise', 'fp32_cpu_native']
r = {n: json.loads((p / n / 'report.json').read_text()) for n in names}
assert all('error' not in v for v in r.values())
assert r['baseline']['matches_reported_bad_result']
assert r['baseline']['condition_sha256'] == r['cpu_noise']['condition_sha256']
assert r['baseline']['noise_sha256'] != r['cpu_noise']['noise_sha256']
for a, b in [('baseline', 'fp32_gpu_noise'), ('cpu_noise', 'fp32_cpu_noise')]:
    assert r[a]['condition_sha256'] == r[b]['condition_sha256']
    assert r[a]['noise_sha256'] == r[b]['noise_sha256']
assert r['masked']['noise_sha256'] == r['baseline']['noise_sha256']
assert [k for k in r['baseline']['condition_sha256'] if r['baseline']['condition_sha256'][k] != r['masked']['condition_sha256'][k]] == ['ca_images']
assert [r[n]['skirt_red_preserved'] for n in names] == [False, True, True, False, False, True]
print('PASS: controlled comparison')
PY
```
