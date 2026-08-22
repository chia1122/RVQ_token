目前 phase 1 已足以形成以下正式描述：
Across three matched seeds, the direct SpeechTokenizer RVQ-prefix CTC probe achieved its lowest error rates at depth 1. Increasing depth degraded both control and dysarthric ASR performance. From depth 4 onward, several seeds showed increased blank-frame ratios, deletion rates, empty hypotheses, and early best epochs, indicating seed-sensitive optimization toward under-generating CTC solutions.

同時保留限制：
* 只有三個 seeds。
* 固定 speaker-disjoint split 的 subgroup coverage 有限。
* WER 在深層接近飽和。
* checkpoint 以 validation WER 選擇。
* 尚未做 mixed-effects analysis。
* 不作 clinical interpretation。

因此目前可以保留的結論是：
Under WER-based checkpoint selection, depth 1 performed best, while deeper runs frequently selected early blank/deletion-dominated checkpoints. Validation histories show substantially lower CER at later epochs, indicating strong checkpoint-selection sensitivity.

# Phase 1 — SpeechTokenizer RVQ Depth Trajectory

## 1. 研究目的

本階段評估 SpeechTokenizer 不同 RVQ prefix depth 對 direct-token CTC ASR
performance 的影響。

比較條件為：

- K1：只使用第一層 RVQ codebook
- K2：使用 Q1–Q2
- …
- K8：使用 Q1–Q8

本階段不包含 speaker identity、dysarthria、severity classification probes，
也不進行 mixed-effects analysis 或 clinical interpretation。

WER、CER、S/D/I、empty hypothesis ratio 與 CTC blank-frame ratio均視為
ASR／CTC pipeline 指標，不代表 clinical intelligibility。

---

## 2. 資料與表示

- Corpus：TORGO
- Codec：SpeechTokenizer
- Codec model：`speechtokenizer_hubert_avg`
- Number of codebooks：8
- Codebook size：1024
- RVQ depths：1–8
- Speech condition：
  - control
  - dysarthric
- Dataset split：使用既有 speaker-disjoint train/valid/test split
- Severity labels與 speaker inclusion：沿用既有 metadata，未修改

每個 token sequence 使用 `[T, N]` 格式。Depth K 只使用前 K 個 RVQ
codebooks。Token IDs 作為離散 indices，未直接進行數值平均。

---

## 3. Pipeline smoke tests

### 3.1 Codec reconstruction smoke

使用單一 test utterance 重建 K1–K8：

- utterances：1
- reconstructed WAV files：8
- failures：0
- sample rate：16 kHz

結果確認 SpeechTokenizer token loading 與 K1–K8 reconstruction pipeline
均可執行。

### 3.2 CTC training smoke

使用 depth 1、seed 1337、兩筆 overfit samples、1 epoch：

- forward/backward：成功
- checkpoint saving：成功
- prediction output：成功
- grouped metrics：成功

Smoke test 僅用於 pipeline validation，不作為正式實驗結果。

---

## 4. CTC length audit

最初設定使用 `time_reduction=4`，正式 sweep 遇到：

```text
ValueError: CTC target is longer than time-reduced encoder output

這表示部分 character targets 比 time-reduced encoder output 更長。

完成全 index length audit 後，正式 trajectory 統一改用：

```text
time_reduction = 2
subsampling = conv
```

沒有裁切 transcript、排除 utterance、修改 split 或更動資料標籤。

---

## 5. 正式實驗設定

- Depths：1–8
- Seeds：1337、2026、3407
- Runs：8 depths × 3 seeds = 24 runs
- Epochs：30
- Model dimension：256
- Transformer encoder layers：4
- Attention heads：4
- Feedforward dimension：1024
- Learning rate：3e-4
- Weight decay：1e-2
- Time reduction：2
- Subsampling：Conv1d
- Primary checkpoint selection metric：validation WER
- Device：CUDA
- Physical batch size：`[由 sweep_config.json 補入]`
- Gradient accumulation：`[由 sweep_config.json 補入]`
- Effective batch size：`[由以上兩項計算]`
- PyTorch version：`[待補]`
- CUDA version：`[待補]`
- GPU model：`[待補]`
- Git commit／working-tree identifier：`[待補]`

所有 depths 使用相同 split、model capacity、optimizer、training budget 與
seed protocol。每個 depth/seed 使用獨立 output directory。

---

## 6. Primary trajectory：WER-selected checkpoints

### 6.1 Overall WER/CER

| Depth | CER mean | CER SD | WER mean | WER SD | Valid runs |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5437 | 0.0104 | 0.9366 | 0.0042 | 3 |
| 2 | 0.6416 | 0.0104 | 0.9770 | 0.0060 | 3 |
| 3 | 0.6719 | 0.0052 | 0.9911 | 0.0127 | 3 |
| 4 | 0.7979 | 0.0923 | 0.9912 | 0.0028 | 3 |
| 5 | 0.8003 | 0.0555 | 0.9914 | 0.0012 | 3 |
| 6 | 0.8402 | 0.0247 | 0.9933 | 0.0016 | 3 |
| 7 | 0.8251 | 0.0301 | 0.9951 | 0.0006 | 3 |
| 8 | 0.8111 | 0.0484 | 0.9938 | 0.0007 | 3 |

在 WER-selected protocol 下，depth 1 的 overall WER 與 CER 最低。

WER 從 depth 3 開始接近 1，顯示 word-level metric 已接近飽和。此時很小的
WER 差異可能影響 checkpoint selection，但不一定反映 character-level
recognition quality。

### 6.2 Error composition and CTC diagnostics

| Depth | Substitution rate | Deletion rate | Insertion rate | Empty ratio | Blank ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7067 | 0.1992 | 0.0307 | 0.0013 | 0.8148 |
| 2 | 0.6582 | 0.3001 | 0.0187 | 0.0031 | 0.8601 |
| 3 | 0.6524 | 0.3193 | 0.0194 | 0.0026 | 0.8561 |
| 4 | 0.4602 | 0.5264 | 0.0046 | 0.0545 | 0.9249 |
| 5 | 0.4852 | 0.5034 | 0.0029 | 0.0273 | 0.9230 |
| 6 | 0.4361 | 0.5567 | 0.0004 | 0.0526 | 0.9408 |
| 7 | 0.4552 | 0.5387 | 0.0012 | 0.0199 | 0.9300 |
| 8 | 0.4641 | 0.5281 | 0.0016 | 0.0171 | 0.9159 |

Depth 1–3 的錯誤主要來自 substitution。Depth 4 之後，deletion rate
超過 substitution rate，同時伴隨 blank-frame ratio 和 empty hypothesis
ratio 上升。

這些結果表示 WER-selected 深層 checkpoints 傾向產生過短 hypothesis，
並呈現 blank/deletion-dominated behavior。

Blank-frame ratio 本身不能單獨證明 CTC collapse；但 blank、deletion 與
empty hypothesis 同時增加，支持 partial collapse／under-generation 的描述。

---

## 7. Control and dysarthric subgroup results

### 7.1 CER

| Depth | Control CER | Dysarthric CER |
|---:|---:|---:|
| 1 | 0.4790 | 0.7738 |
| 2 | 0.5875 | 0.8336 |
| 3 | 0.6219 | 0.8497 |
| 4 | 0.7724 | 0.8888 |
| 5 | 0.7764 | 0.8849 |
| 6 | 0.8244 | 0.8965 |
| 7 | 0.8066 | 0.8904 |
| 8 | 0.7850 | 0.9036 |

### 7.2 WER

| Depth | Control WER | Dysarthric WER |
|---:|---:|---:|
| 1 | 0.9113 | 1.0247 |
| 2 | 0.9608 | 1.0336 |
| 3 | 0.9770 | 1.0405 |
| 4 | 0.9883 | 1.0014 |
| 5 | 0.9886 | 1.0014 |
| 6 | 0.9919 | 0.9983 |
| 7 | 0.9940 | 0.9989 |
| 8 | 0.9921 | 0.9997 |

Control 與 dysarthric speech 的 ASR performance 都隨 depth 加深而惡化。

淺層時兩組差距較大；深層時差距縮小，主要原因是 control performance
也接近高錯誤區域。這不能解讀為 dysarthric speech 獲得改善或群組公平性提升。

Dysarthric WER 超過 1 是合法結果，表示 substitutions、deletions 與
insertions 的總數超過 reference word 數量。

這些 subgroup 數值僅描述固定 test speakers 的 ASR error，不能當成
clinical intelligibility、診斷結果或 utterance-level clinical severity。

---

## 8. Seed and checkpoint-selection behavior

Depth 1–3 的 WER-selected best epochs 多落在 18–30。

Depth 4–8 出現較強的 seed sensitivity。例如：

- Depth 4, seed 1337：
  - best epoch 17
  - test CER 0.6934
  - empty ratio 0.0024
- Depth 4, seed 3407：
  - best epoch 3
  - test CER 0.8682
  - empty ratio 0.1029

Depth 8 也出現不同 seed 落入不同狀態：

- seed 2026：
  - best epoch 21
  - CER 0.7567
  - blank ratio 0.8708
- seed 3407：
  - best epoch 6
  - CER 0.8493
  - blank ratio 0.9448

因此深層 mean/SD 可能混合了不同 optimization regimes，不能只報告平均值。

---

## 9. Checkpoint-selection audit

比較 WER-selected epoch 與 history 中 minimum-validation-CER epoch後發現：

- Depth 1–3：兩種 selection 的差距較小。
- Depth 4–8：WER-selected epoch 常落在 3–9，但 minimum-CER epoch 多落在
  27–30。
- 深層 validation WER 接近飽和，使 WER-based selection 對 character-level
  improvement 缺乏解析度。

代表性例子：

| Depth | Seed | WER-selected epoch | Selected valid CER | Min-CER epoch | Min valid CER |
|---:|---:|---:|---:|---:|---:|
| 4 | 3407 | 3 | 0.8669 | 29 | 0.6587 |
| 5 | 2026 | 4 | 0.8477 | 29 | 0.6695 |
| 6 | 1337 | 4 | 0.8462 | 29 | 0.6760 |
| 8 | 3407 | 6 | 0.8350 | 30 | 0.7142 |

這表示 primary trajectory 的深層 degradation 同時包含：

1. deeper-prefix representation／optimization difficulty；
2. WER-saturated checkpoint-selection artifact。

不能將所有 degradation 都歸因於 deeper RVQ layers。

---

## 10. CER-selection pilot

為檢查 checkpoint-selection sensitivity，執行：

- Depths：1、4
- Seed：3407
- Epochs：30
- Primary difference：`selection_metric=cer`
- 其餘設定：應與原始 trajectory 相同
- Output root：獨立於 primary trajectory

### Pilot results

| Depth | Selection | Best epoch | Test WER | Test CER | Empty ratio | Blank ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | WER | 28 | 0.9413 | 0.5394 | 0.0015 | 0.8156 |
| 1 | CER | 27 | 0.9333 | 0.5354 | 0.0015 | 0.8066 |
| 4 | WER | 3 | 0.9931 | 0.8682 | 0.1029 | 0.9592 |
| 4 | CER | 29 | 1.0117 | 0.6728 | 0.0024 | 0.8470 |

Depth 1 幾乎不受 selection metric 影響。

Depth 4 使用 CER selection 後：

- CER 改善 0.1954；
- empty ratio 從 0.1029 降至 0.0024；
- blank ratio 從 0.9592 降至 0.8470；
- best epoch 從 3 移至 29；
- WER 則增加 0.0186。

這證明 WER selection 在 depth 4 選到早期 blank-dominated checkpoint。
晚期 checkpoint 的 character recognition 明顯改善，但尚未轉化為更多完整
正確單字。

CER selection 下，depth 4 CER 仍比 depth 1 高 0.1374，因此 checkpoint
selection 不能完全解釋 depth degradation。

由於 pilot 只有一個 seed，結果只能支持 sensitivity hypothesis，不能代表
完整 CER-selected trajectory。

---

## 11. Current interpretation

目前最保守且可支持的結論為：

> Across three matched seeds, the WER-selected direct SpeechTokenizer
> RVQ-prefix CTC trajectory achieved its lowest error rates at depth 1.
> Deeper runs frequently selected early checkpoints characterized by higher
> blank-frame ratios, deletion rates, and empty hypotheses. Validation
> histories and a CER-selection pilot showed that WER-based selection
> substantially exaggerated the apparent depth-4 degradation. Nevertheless,
> depth 4 remained worse than depth 1 under CER selection.

中文摘要：

> 在三個配對 seeds 與固定訓練設定下，以 validation WER 選擇 checkpoint
> 的 SpeechTokenizer direct-token CTC trajectory 在 depth 1 取得最低錯誤率。
> 深層 runs 經常選到高 blank、高 deletion 及較多 empty hypotheses 的早期
> checkpoint。Validation history 與 CER-selection pilot 顯示，WER selection
> 明顯放大了 depth 4 的表現下降；但即使改用 CER selection，depth 4 仍差於
> depth 1。

---

## 12. Limitations

1. 每個 depth 只有三個 seeds。
2. 使用單一固定 speaker-disjoint split。
3. Test split 的 severity／speaker coverage 有限。
4. 深層 WER 接近飽和，不適合作為唯一 checkpoint-selection signal。
5. CER minimum 多位於 epoch 27–30，可能仍有 training-budget sensitivity。
6. CER-selection pilot 只有 depths 1、4 與單一 seed。
7. 尚未執行完整 CER-selected trajectory。
8. 尚未比較其他 fusion mechanisms。
9. 尚未執行 codec-native embedding probe 或 acoustic baseline。
10. 沒有進行 mixed-effects analysis。
11. WER/CER 不能解讀為 clinical intelligibility。
12. `speaker_metadata.csv` 中尚有 citation TODO，未自行補寫來源。

---

## 13. Recommended next step

若 GPU budget 允許，執行完整 CER-selected sensitivity trajectory：

- depths 1–8；
- seeds 1337、2026、3407；
- epochs 30；
- 保持原始 physical batch、gradient accumulation、model capacity、
  optimizer、split 與 time reduction；
- 只將 checkpoint selection metric 改為 CER；
- 使用獨立 output root。

完成後比較：

1. WER-selected vs CER-selected depth trajectories；
2. blank、empty、S/D/I；
3. best epoch distribution；
4. control/dysarthric subgroup ASR results；
5. seed sensitivity。

若 CER-selected best epochs 仍大量落在 epoch 30，再將 extended training
budget 視為另一個獨立 sensitivity experiment。
```

撰寫前請從兩個 `sweep_config.json` 補上：

- 原始與 pilot 的 physical batch size
- gradient accumulation
- effective batch
- PyTorch/CUDA/GPU 版本
- Git commit 或程式版本識別

如果 pilot 與原始 effective batch 不一致，必須在第 10 節明確標記為
「checkpoint selection + batch sensitivity」，不能稱為純 selection comparison。