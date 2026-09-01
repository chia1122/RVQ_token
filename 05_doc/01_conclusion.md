# Phase 1 and matched follow-up — SpeechTokenizer RVQ Depth Trajectories

> **Record status:** completed fixed-split pilot plus completed seven-fold
> speaker-disjoint cumulative-prefix and individual-layer follow-ups. This
> document preserves the original metrics and checkpoint-selection comparison
> and records both formal 168-run matrices, their paired analysis, and the
> completed 70,065-item frozen-ASR cumulative reconstruction baseline.
> It does not establish the final pathology-aware fusion method, a clinical
> information hierarchy, or the absence of linguistic information in later RVQ
> layers. The current roadmap is
> [Pathology-aware RVQ Layer Fusion for Dysarthric ASR](PATHOLOGY_AWARE_RVQ_FUSION_ROADMAP.md).

## 1. 研究目的

本階段評估 SpeechTokenizer 不同 RVQ prefix depth 對 direct-token CTC ASR
performance 的影響。比較 K1 至 K8，其中 depth K 使用 Q1–QK。

本階段不包含 speaker identity、dysarthria 或 severity classification probes，
也不進行 mixed-effects analysis 或 clinical interpretation。WER、CER、S/D/I、
empty hypothesis ratio 與 CTC blank-frame ratio 均為 ASR／CTC pipeline 指標，
不能解讀為 clinical intelligibility。

---

## 2. 資料與表示

- Corpus：TORGO
- Codec：SpeechTokenizer
- Codec model：`speechtokenizer_hubert_avg`
- Number of codebooks：8
- Codebook size：1024
- RVQ depths：1–8
- Speech conditions：control、dysarthric
- Dataset split：沿用既有 speaker-disjoint train/valid/test split
- Severity labels 與 speaker inclusion：沿用既有 metadata，未修改

每個 token sequence 使用 `[T, N]` 格式。Depth K 只使用前 K 個 RVQ
codebooks。Token IDs 僅作為離散 indices，未直接相加或平均為數值特徵。

### 2.1 Representation audit note

根據完成本實驗時的實際程式：

1. `RVQTokenDataset` 以 `codes[:, :K]` 讀入 Q1–QK；
2. 未指定 `--active-rvq-layers` 時，`train_probe.py` 啟用 Q1–QK 全部 layers；
3. 每個 layer 使用獨立、task-trained `nn.Embedding`；
4. `layer_fusion=sum` 的 forward 計算為 learned embeddings 相加後除以
   `sqrt(K)`。

因此本文件 K1–K8 的正式 trajectory 應精確標示為：

```text
representation_mode = discrete_learned
rvq_mode = cumulative
condition(K) = cumulative_q1_k
fusion = sqrt_normalized_sum
```

這不是 `individual_qk` trajectory，也不是 frozen codec-native cumulative
latent。模型輸出的 equal `normalized_layer_weights` 是描述性權重；實際 forward
scale 是 `1/sqrt(K)`，不是 arithmetic mean `1/K`。

現有程式可透過明確指定 active layer 建立 individual learned-layer condition。
原 Phase 1 sweep 未使用該模式；後續正式 seven-fold matched individual Q1–Q8
trajectory 已完成，結果記錄於第 11 節。
因此原始數值不需要改動，但其意義只限於 cumulative learned-prefix ASR
performance。

---

## 3. Pipeline validation

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

### 3.3 CTC length audit

最初設定 `time_reduction=4` 時，正式 sweep 遇到：

```text
ValueError: CTC target is longer than time-reduced encoder output
```

這表示部分 character targets 比 time-reduced encoder output 更長。完成全 index
length audit 後，正式 trajectory 統一使用：

```text
time_reduction = 2
subsampling = conv
```

沒有裁切 transcript、排除 utterance、修改 split 或更動資料標籤。

---

## 4. 正式實驗設定

- Depths：1–8
- Seeds：1337、2026、3407
- Runs：每個 selection protocol 為 8 depths × 3 seeds = 24 runs
- Epochs：30
- Physical batch size：8
- Gradient accumulation：1
- Effective per-step batch size：8
- Model dimension：256
- Transformer encoder layers：4
- Attention heads：4
- Feedforward dimension：1024
- Learning rate：3e-4
- Weight decay：1e-2
- Time reduction：2
- Subsampling：Conv1d
- Primary checkpoint selection：validation WER
- Sensitivity checkpoint selection：validation CER
- Device：CUDA
- Python：3.9.25
- PyTorch：2.5.1+cu124
- PyTorch CUDA build：12.4
- CUDA Toolkit (`nvcc`)：12.6（V12.6.85）
- cuDNN：9.1.0（PyTorch reported 90100）
- NVIDIA driver：570.211.01
- GPU：2 × NVIDIA GeForce RTX 3090（每張 24,576 MiB）
- Git commit：`16220a602285261d7a152402c32230d0315959f5`
- Git commit summary：`16220a6 2026-08-22 15:43:29 +0800 feat: Update command script and add tmux session management for RVQ depth sweep`
- Recorded Git status：clean（`git status --short` 無輸出）

兩個 sweeps 使用相同 codec、depths、seeds 與其餘 training arguments。設定檔比較
只顯示 `--grad-accum-steps` 的文字差異：WER-selected run 未明確指定，因此採用
預設值 1；CER-selected run 明確指定為 1。兩者的實際 gradient accumulation
相同，因此不是 batch-size sensitivity comparison。

所有 depths 使用相同 split、model capacity、optimizer、training budget 與 seed
protocol。每個 depth/seed 使用獨立 output directory。

---

## 5. Primary trajectory：WER-selected checkpoints

### 5.1 Overall WER/CER

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

在 WER-selected protocol 下，depth 1 的 overall WER 與 CER 最低。WER 從
depth 3 開始接近 1，顯示 word-level metric 已接近飽和；此時很小的 WER 差異
可能改變 checkpoint selection，但不一定反映 character-level recognition
quality。

### 5.2 Error composition and CTC diagnostics

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

Depth 1–3 的錯誤主要來自 substitution。Depth 4 之後，deletion rate 超過
substitution rate，並伴隨 blank-frame ratio 與 empty hypothesis ratio 上升。
這表示 WER-selected 深層 checkpoints 傾向 under-generation。

Blank-frame ratio 本身不能單獨證明 CTC collapse；但 blank、deletion 與 empty
hypothesis 同時增加，支持 partial collapse／blank-deletion-dominated behavior
的描述。

---

## 6. WER-selected control/dysarthric subgroup results

| Depth | Control CER | Dysarthric CER | Control WER | Dysarthric WER |
|---:|---:|---:|---:|---:|
| 1 | 0.4790 | 0.7738 | 0.9113 | 1.0247 |
| 2 | 0.5875 | 0.8336 | 0.9608 | 1.0336 |
| 3 | 0.6219 | 0.8497 | 0.9770 | 1.0405 |
| 4 | 0.7724 | 0.8888 | 0.9883 | 1.0014 |
| 5 | 0.7764 | 0.8849 | 0.9886 | 1.0014 |
| 6 | 0.8244 | 0.8965 | 0.9919 | 0.9983 |
| 7 | 0.8066 | 0.8904 | 0.9940 | 0.9989 |
| 8 | 0.7850 | 0.9036 | 0.9921 | 0.9997 |

Control 與 dysarthric speech 的 ASR performance 均隨 depth 加深而惡化。深層
兩組差距縮小，主要是 control performance 也進入高錯誤區域，不能解讀為
dysarthric speech 改善或群組公平性提升。

Dysarthric WER 超過 1 是合法結果，表示 substitutions、deletions 與
insertions 的總數超過 reference word 數量。這些 subgroup 結果只描述固定
test speakers 的 ASR error，不能當成 clinical intelligibility、診斷結果或
utterance-level clinical severity。

目前研究紀錄中尚未提供 CER-selected trajectory 的 condition/severity/speaker
彙整值，因此本文件不推測或補造這些數值。

---

## 7. Seed and checkpoint-selection audit

WER-selected depth 1–3 的 best epochs 多落在 18–30；depth 4–8 則出現較強
的 seed sensitivity。例如：

- Depth 4, seed 1337：epoch 17、test CER 0.6934、empty ratio 0.0024。
- Depth 4, seed 3407：epoch 3、test CER 0.8682、empty ratio 0.1029。
- Depth 8, seed 2026：epoch 21、CER 0.7567、blank ratio 0.8708。
- Depth 8, seed 3407：epoch 6、CER 0.8493、blank ratio 0.9448。

比較 WER-selected epoch 與 history 中 minimum-validation-CER epoch 後發現，
depth 4–8 的 WER-selected epoch 常落在 3–9，而 minimum-CER epoch 多落在
27–30。代表性例子如下：

| Depth | Seed | WER-selected epoch | Selected valid CER | Min-CER epoch | Min valid CER |
|---:|---:|---:|---:|---:|---:|
| 4 | 3407 | 3 | 0.8669 | 29 | 0.6587 |
| 5 | 2026 | 4 | 0.8477 | 29 | 0.6695 |
| 6 | 1337 | 4 | 0.8462 | 29 | 0.6760 |
| 8 | 3407 | 6 | 0.8350 | 30 | 0.7142 |

深層 validation WER 接近飽和，使 WER-based selection 對 character-level
improvement 缺乏解析度。因此 primary trajectory 的深層 degradation 同時包含
deeper-prefix representation／optimization difficulty 與 checkpoint-selection
artifact，不能全部歸因於 deeper RVQ layers。

---

## 8. Sensitivity trajectory：CER-selected checkpoints

完整 CER-selected sensitivity sweep 已完成，共包含 8 depths × 3 seeds = 24
個 runs。`sweep_runs.csv` audit 顯示 24 個 runs 全部為 `valid`，沒有缺失或
non-valid run。

### 8.1 Overall results

以下為三個 seeds 的 overall mean。

| Depth | WER | CER | Deletion rate | Empty ratio | Blank ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9507 | 0.5440 | 0.1622 | 0.0008 | 0.7997 |
| 2 | 0.9884 | 0.6157 | 0.2126 | 0.0023 | 0.8258 |
| 3 | 1.0136 | 0.6495 | 0.2048 | 0.0018 | 0.8291 |
| 4 | 1.0100 | 0.6718 | 0.2412 | 0.0033 | 0.8511 |
| 5 | 1.0302 | 0.6856 | 0.2048 | 0.0011 | 0.8355 |
| 6 | 1.0393 | 0.6994 | 0.1964 | 0.0016 | 0.8396 |
| 7 | 1.0355 | 0.7177 | 0.2313 | 0.0016 | 0.8483 |
| 8 | 1.0265 | 0.7244 | 0.2434 | 0.0021 | 0.8533 |

CER selection 下，empty hypothesis ratio 全部低於 0.0033，blank ratio 約為
0.80–0.85，deletion rate 約為 0.16–0.24。相較於 WER-selected 深層 runs，
CER selection 大幅降低了 early blank/deletion-dominated checkpoint 的影響。

然而，CER 仍從 depth 1 的 0.5440 隨 depth 整體增加至 depth 8 的 0.7244。
因此 checkpoint selection 可以解釋 WER-selected trajectory 中相當一部分深層
劣化，但不能完全消除 depth-dependent degradation。

### 8.2 Condition and severity results

以下數值為 mean ± sample SD，所有 cell 均有三個有效 seeds。

| Depth | Control CER | Dysarthric CER | Control WER | Dysarthric WER |
|---:|---:|---:|---:|---:|
| 1 | 0.4818 ± 0.0037 | 0.7649 ± 0.0097 | 0.9310 ± 0.0329 | 1.0198 ± 0.0147 |
| 2 | 0.5621 ± 0.0068 | 0.8063 ± 0.0217 | 0.9679 ± 0.0165 | 1.0601 ± 0.0439 |
| 3 | 0.5978 ± 0.0081 | 0.8333 ± 0.0263 | 0.9950 ± 0.0050 | 1.0787 ± 0.0533 |
| 4 | 0.6316 ± 0.0078 | 0.8146 ± 0.0148 | 1.0002 ± 0.0147 | 1.0443 ± 0.0246 |
| 5 | 0.6465 ± 0.0079 | 0.8244 ± 0.0246 | 1.0162 ± 0.0104 | 1.0790 ± 0.0727 |
| 6 | 0.6640 ± 0.0061 | 0.8251 ± 0.0255 | 1.0308 ± 0.0136 | 1.0690 ± 0.0499 |
| 7 | 0.6802 ± 0.0106 | 0.8511 ± 0.0392 | 1.0292 ± 0.0052 | 1.0575 ± 0.0312 |
| 8 | 0.6926 ± 0.0102 | 0.8374 ± 0.0334 | 1.0168 ± 0.0242 | 1.0603 ± 0.0085 |

在 CER-selected trajectory 中，control 與 dysarthric condition 的 CER 都呈現
隨 depth 增加而升高的整體趨勢。Dysarthric condition 在所有 depths 的 CER
均高於 control；這只描述目前固定 test split 的 ASR error，不能視為 clinical
intelligibility 或 clinical group inference。

此 test split 的 severity summary 實際只包含 `control` 與
`moderate-to-severe`。在目前 coverage 中，`control` severity rows 與 control
condition rows 相同，`moderate-to-severe` rows 與 dysarthric condition rows
相同，因此其 WER/CER mean、SD 與上表完全一致。這是 test-speaker composition
造成的相同分組，不能外推至其他 severity levels，也不能推論 utterance-level
severity。

### 8.3 Speaker results

完整 CER-selected summary 只包含三位 test speakers：FC03、M05、MC04。以下為
mean ± sample SD，每個 cell 均有三個有效 seeds。

| Depth | FC03 CER | M05 CER | MC04 CER | FC03 WER | M05 WER | MC04 WER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4634 ± 0.0049 | 0.7649 ± 0.0097 | 0.5113 ± 0.0042 | 0.9335 ± 0.0445 | 1.0198 ± 0.0147 | 0.9270 ± 0.0145 |
| 2 | 0.5329 ± 0.0084 | 0.8063 ± 0.0217 | 0.6090 ± 0.0047 | 0.9716 ± 0.0219 | 1.0601 ± 0.0439 | 0.9620 ± 0.0080 |
| 3 | 0.5778 ± 0.0051 | 0.8333 ± 0.0263 | 0.6299 ± 0.0215 | 1.0064 ± 0.0115 | 1.0787 ± 0.0533 | 0.9767 ± 0.0056 |
| 4 | 0.6055 ± 0.0062 | 0.8146 ± 0.0148 | 0.6736 ± 0.0133 | 1.0115 ± 0.0225 | 1.0443 ± 0.0246 | 0.9821 ± 0.0029 |
| 5 | 0.6338 ± 0.0133 | 0.8244 ± 0.0246 | 0.6670 ± 0.0076 | 1.0362 ± 0.0166 | 1.0790 ± 0.0727 | 0.9842 ± 0.0061 |
| 6 | 0.6457 ± 0.0114 | 0.8251 ± 0.0255 | 0.6933 ± 0.0056 | 1.0517 ± 0.0166 | 1.0690 ± 0.0499 | 0.9974 ± 0.0150 |
| 7 | 0.6682 ± 0.0096 | 0.8511 ± 0.0392 | 0.6995 ± 0.0124 | 1.0467 ± 0.0021 | 1.0575 ± 0.0312 | 1.0013 ± 0.0132 |
| 8 | 0.6792 ± 0.0079 | 0.8374 ± 0.0334 | 0.7143 ± 0.0150 | 1.0292 ± 0.0364 | 1.0603 ± 0.0085 | 0.9970 ± 0.0080 |

這些 speaker rows 是固定 test speakers 的描述性 ASR 結果。Speaker 不是彼此
可替換的獨立 clinical subjects；本階段不以這三位 speakers 進行 population-level
clinical inference。

---

## 9. WER selection 與 CER selection 的實際差異

下表定義：

```text
delta = CER-selected result - WER-selected result
```

負的 `delta_CER` 表示 CER selection 的 character error 較低；正的
`delta_WER` 表示 CER selection 的 word error 較高。Delta 使用原始未四捨五入
數值計算，因此可能與表中顯示到小數點後四位的兩欄直接相減有 0.0001 的差異。

| Depth | WER-selected WER | CER-selected WER | delta WER | WER-selected CER | CER-selected CER | delta CER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9366 | 0.9507 | +0.0142 | 0.5437 | 0.5440 | +0.0002 |
| 2 | 0.9770 | 0.9884 | +0.0114 | 0.6416 | 0.6157 | -0.0258 |
| 3 | 0.9911 | 1.0136 | +0.0225 | 0.6719 | 0.6495 | -0.0224 |
| 4 | 0.9912 | 1.0100 | +0.0188 | 0.7979 | 0.6718 | -0.1261 |
| 5 | 0.9914 | 1.0302 | +0.0387 | 0.8003 | 0.6856 | -0.1147 |
| 6 | 0.9933 | 1.0393 | +0.0460 | 0.8402 | 0.6994 | -0.1409 |
| 7 | 0.9951 | 1.0355 | +0.0404 | 0.8251 | 0.7177 | -0.1073 |
| 8 | 0.9938 | 1.0265 | +0.0327 | 0.8111 | 0.7244 | -0.0866 |

實際差異可歸納為：

1. Depth 1 對 selection metric 幾乎不敏感；CER 差只有約 0.0002。
2. Depth 2–3 使用 CER selection 後，CER 降低約 0.022–0.026。
3. Depth 4–8 的差異最大，CER 降低約 0.087–0.141，說明 WER selection
   明顯放大深層 character-level degradation。
4. CER selection 在所有 depths 的 WER 都較高，增加約 0.011–0.046。
5. 因此這不是 CER selection 對所有 ASR 指標的一致改善，而是 checkpoint
   objective trade-off：較低的 CER、較少的 blank/deletion under-generation，
   但較高的 WER。
6. 即使採用 CER selection，depth 1 仍有最低 CER；depth 8 相較 depth 1
   增加 0.1804。因此 deeper cumulative learned prefixes 在目前
   sqrt-normalized-sum direct-token CTC 設定下，未改善 character-level ASR
   performance。這不證明 individual later layers 不含 linguistic 或互補資訊。

WER-selected trajectory 保留為 primary analysis；CER-selected trajectory 是
事後進行的 checkpoint-selection sensitivity analysis，不取代 primary result。

---

## 10. Seven-fold CER-selected cumulative-prefix follow-up

### 10.1 Protocol and aggregation audit

正式 follow-up 使用 versioned 15-speaker protocol：8 位 dysarthric、7 位
control speakers。七個 predefined cyclic folds 依序作 test，下一 fold 作
validation，其餘 folds 作 train。所有 runs 使用 validation CER 選擇 checkpoint。

完整矩陣為：

```text
7 rotations × 8 cumulative depths × 3 seeds = 168 runs
```

正式工作站 audit 結果：

- 7/7 rotations 完成；
- 168/168 runs 為 `valid`；
- depth 1–8、seeds 1337／2026／3407 完整；
- 15 位 speakers 全部恰好進入 test rotation；
- combined long-format：12,240 rows；
- run-summary：1,840 rows；
- pooled-micro summary：1,656 rows；
- speaker-macro summary：448 rows；
- aggregation status：`valid`。

三種 aggregation 不可互換：

- **run-macro**：每個 depth 對 7 rotations × 3 seeds 的 21 個 test results
  等權平均；
- **pooled-micro**：先在每個 seed 內合併七個 test folds 的 numerator 與
  denominator，再對三個 seeds 摘要；
- **speaker-macro**：先讓每位 speaker 等權，再對三個 seeds 摘要。

`ctc_blank_frame_ratio` 因既有 `results.json` 未保存 pooled aggregation 所需的
`valid_frames` denominator，只保留 run/fold-macro 或 speaker-macro 結果，未偽裝成
pooled-micro metric。

### 10.2 Overall cumulative trajectory

下表的 pooled-micro 與 speaker-macro 數值皆為三個 seed-level aggregates 的平均：

| Depth | Pooled-micro WER | Pooled-micro CER | Speaker-macro WER | Speaker-macro CER |
|---:|---:|---:|---:|---:|
| 1 | 0.8857 | 0.4912 | 0.9007 | 0.5347 |
| 2 | 0.9255 | 0.5252 | 0.9355 | 0.5648 |
| 3 | 0.9490 | 0.5481 | 0.9568 | 0.5866 |
| 4 | 0.9541 | 0.5571 | 0.9591 | 0.5946 |
| 5 | 0.9633 | 0.5685 | 0.9689 | 0.6049 |
| 6 | 0.9791 | 0.5787 | 0.9802 | 0.6129 |
| 7 | 0.9752 | 0.5830 | 0.9767 | 0.6171 |
| 8 | 0.9803 | 0.5866 | 0.9828 | 0.6205 |

Q1 在四個 overall metrics 均為最佳 cumulative depth。K8 相對 K1：

| Metric | K1 | K8 | K8 − K1 |
|---|---:|---:|---:|
| Pooled-micro WER | 0.8857 | 0.9803 | +0.0946 |
| Pooled-micro CER | 0.4912 | 0.5866 | +0.0954 |
| Speaker-macro WER | 0.9007 | 0.9828 | +0.0821 |
| Speaker-macro CER | 0.5347 | 0.6205 | +0.0858 |

Pooled-micro 與 speaker-macro 的方向一致，表示整體結論不是由 utterance 數較多的
speaker 單獨造成。CER 隨 cumulative depth 由 K1 至 K8 持續增加；WER 在個別相鄰
depth 有小幅波動，但 K2–K8 均高於 K1。

### 10.3 Condition speaker-macro CER

下表為每個 seed 先對該 condition 的 speakers 等權平均，再報告三 seeds 的
mean ± SD：

| Depth | Control CER | Dysarthric CER |
|---:|---:|---:|
| 1 | 0.4219 ± 0.0019 | 0.6334 ± 0.0035 |
| 2 | 0.4606 ± 0.0023 | 0.6561 ± 0.0034 |
| 3 | 0.4835 ± 0.0051 | 0.6768 ± 0.0034 |
| 4 | 0.4968 ± 0.0028 | 0.6803 ± 0.0021 |
| 5 | 0.5091 ± 0.0053 | 0.6888 ± 0.0055 |
| 6 | 0.5213 ± 0.0015 | 0.6931 ± 0.0050 |
| 7 | 0.5248 ± 0.0041 | 0.6979 ± 0.0023 |
| 8 | 0.5317 ± 0.0049 | 0.6982 ± 0.0014 |

Q1 同時是 control 與 dysarthric speaker-macro CER 的最佳 cumulative depth。
K8−K1 分別為 control `+0.1098`、dysarthric `+0.0648`。兩 condition 的差距在
較深 depth 縮小，是因 control CER 惡化較多，不代表 dysarthric ASR 改善或任何
clinical fairness improvement。

### 10.4 Speaker and rotation consistency

每個 speaker 的 CER 先跨三 seeds 平均。15/15 speakers 的最佳 cumulative depth
皆為 Q1，且沒有任何 speaker 在 K8 優於 K1：

| Speaker | Test rotation | Best depth | CER K1 | CER K8 | K8 − K1 |
|---|---:|---:|---:|---:|---:|
| F01 | 1 | 1 | 0.6877 | 0.7510 | +0.0633 |
| F03 | 6 | 1 | 0.5172 | 0.6072 | +0.0900 |
| F04 | 2 | 1 | 0.5321 | 0.5856 | +0.0536 |
| FC01 | 7 | 1 | 0.3001 | 0.4743 | +0.1741 |
| FC02 | 4 | 1 | 0.3100 | 0.4485 | +0.1385 |
| FC03 | 3 | 1 | 0.3781 | 0.5243 | +0.1462 |
| M01 | 4 | 1 | 0.7608 | 0.7942 | +0.0334 |
| M02 | 5 | 1 | 0.7455 | 0.7726 | +0.0271 |
| M03 | 7 | 1 | 0.3217 | 0.4707 | +0.1490 |
| M04 | 3 | 1 | 0.7243 | 0.7818 | +0.0574 |
| M05 | 7 | 1 | 0.7781 | 0.8223 | +0.0442 |
| MC01 | 1 | 1 | 0.3691 | 0.5007 | +0.1316 |
| MC02 | 6 | 1 | 0.5665 | 0.5930 | +0.0265 |
| MC03 | 2 | 1 | 0.6248 | 0.6617 | +0.0369 |
| MC04 | 5 | 1 | 0.4046 | 0.5192 | +0.1146 |

Speaker-level K8−K1 CER 增幅範圍為 `+0.0265` 至 `+0.1741`。Speaker 間仍有
明顯差異，且部分 control speakers 的 CER 高於部分 dysarthric speakers，因此
condition aggregate 不能取代 per-speaker reporting，也不能作 clinical severity
或 intelligibility 解讀。

七個 rotations 亦全部以 Q1 為最佳 cumulative depth，且 K8 均劣於 K1：

| Rotation | Best depth | CER K1 | CER K8 | K8 − K1 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.3967 | 0.5224 | +0.1257 |
| 2 | 1 | 0.6024 | 0.6433 | +0.0409 |
| 3 | 1 | 0.4510 | 0.5785 | +0.1275 |
| 4 | 1 | 0.4301 | 0.5406 | +0.1105 |
| 5 | 1 | 0.5319 | 0.6138 | +0.0819 |
| 6 | 1 | 0.5394 | 0.6008 | +0.0614 |
| 7 | 1 | 0.5316 | 0.6350 | +0.1034 |

Rotation-level K8−K1 CER 增幅範圍為 `+0.0409` 至 `+0.1275`。因此目前固定
cumulative fusion 的 degradation 並非由單一 speaker、rotation 或 seed 驅動。

### 10.5 Meaning for the fusion roadmap

正式七-fold結果確認：在 `discrete_learned`、`cumulative_q1_k`、
`sqrt_normalized_sum` 與 validation-CER selection 設定下，加入更多 RVQ layers
會一致降低 ASR performance。這是固定 cumulative fusion 的可靠 negative
baseline，也是後續 complementarity diagnosis 的動機。

然而，這仍無法區分「later layers 沒有可轉移 ASR 資訊」與「later layers 有互補
資訊，但固定相加造成 destructive interference」。後續 matched individual Q1–Q8
已完成並顯示 later residual layers 的獨立 transcription recoverability 有限；這仍
不等同於證明 later layers 不含 acoustic 或 fusion-complementary information。因此
adaptive-fusion decision gate 尚未通過，仍需 reconstruction 與 fixed-fusion
baselines。

---

## 11. Seven-fold matched individual-layer follow-up

### 11.1 Protocol and positive-control audit

正式 matched individual matrix 沿用 cumulative trajectory 的七個 speaker-disjoint
rotations、三個 seeds、model capacity、optimizer、training budget 與 validation-CER
checkpoint selection：

```text
7 rotations × 8 individual layers × 3 seeds = 168 runs
```

工作站 audit 確認 168/168 runs 為 valid，且每個 `individual_qK` 僅啟用 QK 的
task-trained discrete embedding。這不是 frozen codec-native embedding，也不是把
token ID 當作可加總的數值特徵。

Q1 是預先指定的 positive control，因為 `individual_q1` 與 `cumulative_q1` 使用
相同 representation。兩者 run-macro 差異接近零：CER delta `+0.0017`，WER delta
`-0.0001`。Speaker-level Q1 direction 亦大致均衡（CER 9 worse／6 better；WER
7 worse／7 better／1 equal），支持 paired pipeline 沒有明顯偏向任一 condition。

### 11.2 Individual versus cumulative actual values

下表是七 rotations × 三 seeds 的 21 個 paired overall results 之 run-macro mean；
delta 定義為 `individual QK − cumulative Q1:QK`，正值代表 individual error 較高：

| Depth | Individual CER | Cumulative CER | Delta CER | Individual WER | Cumulative WER | Delta WER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4993 | 0.4976 | +0.0017 | 0.8881 | 0.8882 | -0.0001 |
| 2 | 0.9163 | 0.5313 | +0.3851 | 0.9993 | 0.9271 | +0.0722 |
| 3 | 0.8788 | 0.5540 | +0.3248 | 1.0001 | 0.9501 | +0.0500 |
| 4 | 0.8880 | 0.5622 | +0.3257 | 0.9994 | 0.9549 | +0.0444 |
| 5 | 0.8689 | 0.5734 | +0.2955 | 0.9985 | 0.9642 | +0.0343 |
| 6 | 0.8977 | 0.5832 | +0.3145 | 0.9974 | 0.9789 | +0.0185 |
| 7 | 0.8923 | 0.5873 | +0.3050 | 0.9981 | 0.9755 | +0.0226 |
| 8 | 0.8968 | 0.5906 | +0.3062 | 0.9995 | 0.9800 | +0.0196 |

Individual Q2–Q8 WER 約為 `0.997–1.000`，已接近飽和，因此 CER 對 residual-layer
差異較有解析力。Q5 是 individual residual layers 中 CER 最低者，但仍遠劣於 Q1；
這只表示相對 recoverability，不代表 Q5 可獨立保存完整 transcription。

### 11.3 Speaker consistency and magnitude audit

每位 speaker 先跨三 seeds 平均。Q2–Q8 的 CER direction 對 15/15 speakers 均為
individual worse，且每位 speaker 在七個 residual depths 全為正 delta：

| Depth | CER individual worse | CER individual better | WER individual worse | WER individual better |
|---:|---:|---:|---:|---:|
| 2 | 15 | 0 | 11 | 4 |
| 3 | 15 | 0 | 13 | 2 |
| 4 | 15 | 0 | 12 | 3 |
| 5 | 15 | 0 | 12 | 3 |
| 6 | 15 | 0 | 10 | 5 |
| 7 | 15 | 0 | 11 | 4 |
| 8 | 15 | 0 | 11 | 4 |

WER direction 較不一致與其接近 1.0 的 metric saturation 相符，不能推翻一致的 CER
pattern。對 Q2–Q8 CER delta 先跨 depths 平均，再讓 speakers 等權後，control 約為
`+0.3790`、dysarthric 約為 `+0.2205`。最大的差異主要來自 control speakers 與
mild speaker M03，而 severe speakers 約為 `+0.1536`，因此方向與主要 magnitude
並非由少數 severe speakers 拉高。

兩位 mild speakers 均為七個 residual depths 全正 delta，但 magnitude 不同：M03
平均 `+0.4385`，F04 平均 `+0.3144`。這兩位 speakers 不足以定義一般化的 mild
pattern。Severe-speaker delta 較小也不能解讀為 residual layers 對 severe speech
保存較多資訊，因為當 individual 與 cumulative errors 都高時，兩者差值會受到
baseline-error compression。

### 11.4 Decoding failure mechanism

Q2–Q8 individual models 呈現一致的 high-blank、deletion-dominated
under-generation，而不是普遍輸出完全空白 hypothesis：

| Depth | Individual deletion | Cumulative deletion | Individual blank | Cumulative blank | Individual length ratio | Cumulative length ratio | Individual epoch | Cumulative epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1677 | 0.1586 | 0.8098 | 0.8071 | 0.7911 | 0.7983 | 26.24 | 27.38 |
| 2 | 0.5764 | 0.1557 | 0.9451 | 0.8095 | 0.2605 | 0.7779 | 24.05 | 26.81 |
| 3 | 0.5040 | 0.1624 | 0.9285 | 0.8104 | 0.3054 | 0.7705 | 27.48 | 27.00 |
| 4 | 0.5177 | 0.1688 | 0.9338 | 0.8117 | 0.3023 | 0.7734 | 28.29 | 28.24 |
| 5 | 0.4848 | 0.1700 | 0.9259 | 0.8150 | 0.3182 | 0.7567 | 27.86 | 27.71 |
| 6 | 0.5701 | 0.1614 | 0.9335 | 0.8132 | 0.2843 | 0.7575 | 27.43 | 27.90 |
| 7 | 0.5630 | 0.1680 | 0.9346 | 0.8137 | 0.2880 | 0.7569 | 28.33 | 29.14 |
| 8 | 0.5360 | 0.1700 | 0.9327 | 0.8133 | 0.2891 | 0.7588 | 27.38 | 29.05 |

相對 cumulative，individual Q2–Q8 blank-frame ratio 增加約 `0.111–0.136`，
hypothesis/reference character-length ratio 減少約 `0.439–0.517`，deletion rate
增加約 `0.315–0.421`。Empty-hypothesis ratio 只增加約 `0.001–0.017`，表示模型
通常仍有輸出，但 hypothesis 只約 reference length 的 26%–32%。Individual
substitution 與 insertion rates 較低並非改善，而是大量 omitted content 在 edit
alignment 中轉為 deletion。

Individual selected epochs 多在 24–28，且除 Q2 外大致接近 cumulative，因此不能
把 failure 簡化成 premature checkpoint selection。結果支持 residual Q2–Q8 在目前
matched CTC protocol 下的獨立 transcription recoverability 有限，但不證明其中
不含 acoustic、speaker、clinical-related 或可由其他 fusion 利用的資訊。

### 11.5 Completed cumulative reconstruction frozen-ASR baseline

SpeechTokenizer 原生 cumulative-prefix reconstruction baseline 已完成：

```text
Original audio ────────────────> one frozen ASR
Cumulative Q1 reconstruction ─> the same frozen ASR
Cumulative Q1:Q2 ─────────────> the same frozen ASR
...
Cumulative Q1:Q8 ─────────────> the same frozen ASR
```

正式 reconstruction 使用 7,785 個 enrolled utterances，產生 62,280 個 Q1–Q8 WAV，
sample rate 為 16 kHz，0 failures。Frozen-ASR evaluation 對 original 與 k1–k8
共產生 70,065 predictions；每個 condition 恰有 7,785 predictions，沒有 duplicate
utterance/condition pairs、reference mismatches、metadata mismatches 或 failures。
Evaluator 為同一個 `faster-whisper 1.2.1`、`large-v3`、float16、English、beam size
5 configuration。ASR 未針對任何 reconstruction condition fine-tune。

這個 frozen evaluation 使用全部 paired utterances，不需要把相同 audio 重複套入七個
training rotations；它不是一個新的 cross-validation training protocol。Original audio
是相同 evaluator 下的 reference condition。

此實驗遵循 codec 原生 residual decoding：QK 是對 Q1:Q(K−1) 的殘差修正，不是
獨立完整 waveform representation。因此不進行把 Q1:Q(K−1) 任意補零後的
individual Q2–Q8 reconstruction，也不把這種 out-of-distribution latent 稱為正式
codec condition。

這條 reconstruction path 測量固定 ASR protocol 下可辨識 linguistic content 的
保留，與 direct-token learned-embedding probe 互補。它不能單獨證明整體語音資訊、
human intelligibility、clinical intelligibility、severity 或 diagnosis 被保留。

### 11.6 Overall and condition reconstruction trajectories

下表為 corpus-level utterance-micro WER/CER：

| Condition | WER | CER | Delta WER vs original | Delta CER vs original |
|---|---:|---:|---:|---:|
| Original | 0.2580 | 0.1593 | 0.0000 | 0.0000 |
| K1 | 0.8033 | 0.5604 | +0.5454 | +0.4012 |
| K2 | 0.6583 | 0.4477 | +0.4003 | +0.2885 |
| K3 | 0.5568 | 0.3726 | +0.2988 | +0.2134 |
| K4 | 0.4745 | 0.3144 | +0.2165 | +0.1552 |
| K5 | 0.4336 | 0.2793 | +0.1756 | +0.1201 |
| K6 | 0.4219 | 0.2847 | +0.1639 | +0.1254 |
| K7 | 0.3855 | 0.2581 | +0.1275 | +0.0988 |
| K8 | 0.3883 | 0.2502 | +0.1304 | +0.0910 |

K1→K8 的 WER 由 0.8033 降至 0.3883，CER 由 0.5604 降至 0.2502。K8
消除了約 77.3% 的 K1 excess CER，但仍未恢復 original performance。最低 WER 在
K7，最低 CER 在 K8；K5→K6 CER 小幅惡化，K7→K8 WER 小幅惡化，因此 trajectory
是 generally improving 而非所有 metrics 嚴格單調。

Control 與 dysarthric utterance-micro 都由 K1 向 deeper prefixes 改善：

| Group | Original CER | K1 CER | K7 CER | K8 CER | K8 − original |
|---|---:|---:|---:|---:|---:|
| Control | 0.0459 | 0.4055 | 0.1258 | 0.1190 | +0.0731 |
| Dysarthric | 0.3677 | 0.8454 | 0.5012 | 0.4916 | +0.1239 |

K5→K6 overall CER 反轉來自 dysarthric CER 由 0.5135 增至 0.5482；K7→K8
overall WER 反轉亦來自 dysarthric WER 由 0.7076 增至 0.7346。Frozen Whisper
在 original audio 已呈現明顯 condition-level ASR performance gap；這不能解讀為
clinical severity、human intelligibility 或 fairness improvement。

### 11.7 Speaker-macro and per-speaker reconstruction audit

讓 15 位 speakers 等權後，trajectory direction 仍與 utterance-micro 一致：

| Group | Original speaker-macro CER | K1 | K7 | K8 |
|---|---:|---:|---:|---:|
| Overall | 0.2291 | 0.6660 | 0.3376 | 0.3366 |
| Control | 0.0478 | 0.4167 | 0.1290 | 0.1217 |
| Dysarthric | 0.3878 | 0.8841 | 0.5202 | 0.5247 |

K8 CER 優於 K1 對 15/15 speakers 成立，因此 cumulative-depth benefit 不是由少數
speakers 或不等 utterance counts 驅動。最佳 reconstructed depth counts 為 K8：10、
K7：4、K5：1。所有 7 位 control speakers 的最佳 CER 都在 K8；dysarthric
speakers 則為 K8：3、K7：4、K5：1。

K8 相對 K7 為 10 speakers 改善、5 speakers 惡化。Dysarthric speaker-macro 的
K7 CER 0.5202 略優於 K8 的 0.5247，雖然 dysarthric utterance-micro 在 K8 較好，
顯示 aggregate conclusion 對 estimand 有依賴，不能用 utterance-micro 取代
speaker-level reporting。

K8 相對 original 仍對 14/15 speakers 較差；唯一例外 M05 的 original CER 0.5433、
K7 0.4446、K8 0.5012。這可能表示 codec transformation 更符合 frozen ASR 的輸入
偏好，不能稱為新增原始資訊或改善 human intelligibility。

### 11.8 Token–reconstruction trajectory reversal

兩條正式 trajectory 呈現方向反轉：

```text
Direct-token cumulative CTC:
15/15 speakers 的 Q1 最佳；0/15 的 Q1:Q8 優於 Q1。

Codec-native cumulative reconstruction:
15/15 speakers 的 K8 優於 K1；10/15 的最佳 reconstructed depth 是 K8。
```

因此 later residual layers 含有可被 codec-native decoder 使用的資訊。Direct-token
fixed sqrt-normalized probe 的 deeper-prefix degradation 不能再解讀為 later layers
缺乏有用資訊；它更符合 representation/fusion mismatch，包括 task-trained embeddings
未保留 native residual geometry、fixed sum destructive interference，或 probe 未有效
建模 codebook hierarchy 等候選解釋。不同 evaluator 的絕對 CER 不可直接互換，
此處比較的是 within-experiment trajectory direction。

這項 reversal 提供 fixed-fusion research 的動機，但 adaptive-fusion decision gate
仍未通過。下一步需在相同 speaker folds、seeds、capacity、budget 與 CER selection
下比較 Q1、現有 sqrt-normalized sum、concatenation plus projection 與 static learned
weighting，才能判斷 later-layer information 是否能被 direct-token ASR 穩定利用。

---

## 12. Fixed-split pilot interpretation

目前最保守且可支持的英文描述為：

> Across three matched seeds, the primary direct SpeechTokenizer RVQ-prefix
> CTC trajectory, selected using validation WER, achieved its lowest error
> rates at depth 1. Deeper WER-selected runs frequently selected early
> checkpoints characterized by higher blank-frame ratios, deletion rates,
> and empty hypotheses. A full validation-CER-selected sensitivity trajectory
> largely removed this deep-layer blank/deletion-dominated behavior and
> substantially reduced test CER at depths 4–8. Nevertheless, CER still
> increased from 0.544 at depth 1 to 0.724 at depth 8 under CER selection.
> CER selection improved character-level performance at deeper depths while
> slightly worsening WER, demonstrating a checkpoint-objective trade-off.

中文摘要：

> 在三個配對 seeds 與固定訓練設定下，以 validation WER 選擇 checkpoint 的
> SpeechTokenizer direct-token CTC primary trajectory 在 depth 1 取得最低錯誤率。
> 深層 runs 經常選到具有較高 blank-frame ratio、deletion rate 與 empty
> hypotheses 的早期 checkpoint。完整的 validation-CER-selected sensitivity
> trajectory 大幅降低了深層 blank/deletion-dominated behavior，並改善 depth
> 4–8 的 test CER；然而，CER selection 下的 CER 仍由 depth 1 的 0.544
> 增加至 depth 8 的 0.724。CER selection 改善了深層 character-level
> performance，但 WER 略為惡化，顯示 checkpoint selection objective 之間
> 存在取捨。

這些結果只支持目前模型與實驗設定下的 ASR performance 描述，不支持 clinical
intelligibility、臨床診斷或 utterance-level severity 的推論。

### 12.1 Meaning under the revised research direction

在新的研究方向中，Phase 1 的用途是診斷與方法動機：固定 cumulative fusion
隨 depth 加深沒有改善 CER，且 checkpoint objective 會顯著改變深層結果。它支持
後續進行 individual-layer、fixed-fusion 與 complementarity audit，但不支持下列
推論：

- Q1 已包含所有 dysarthric ASR 所需資訊；
- Q2–Q8 沒有 linguistic information；
- Q1–Q8 已形成 pathology information hierarchy；
- 較高 CER 是由 pathological acoustic information 單獨造成；
- adaptive pathology-aware fusion 必然優於固定 fusion。

只有在 speaker-disjoint folds 中確認 later layers 具有互補資訊，或 fixed
multilayer fusion 對多位 dysarthric speakers／phoneme categories 有一致改善後，
才足以通過 adaptive-fusion decision gate。

---

## 13. Limitations

1. 每個 depth 只有三個 seeds。
2. 原 Phase 1 pilot 使用單一 fixed split；正式 follow-up 已改用七個
   speaker-disjoint rotations，但整體仍只有 15 位 speakers。
3. Severity speaker counts 不平衡；moderate 與 moderate-to-severe 各只有一位
   speaker，因此 severity 結果只作描述性 reporting。
4. 深層 WER 接近飽和，不適合作為唯一 checkpoint-selection signal。
5. 多個 minimum-CER epochs 位於 epoch 27–30，仍可能存在 training-budget
   sensitivity。
6. 原 fixed-split CER sensitivity 的 coverage 只有 FC03、M05、MC04；七-fold
   follow-up 改善 speaker coverage，但不消除小 corpus 與 severity imbalance。
7. Matched individual Q1–Q8 已完成，但尚未比較其他 token fusion mechanisms。
8. 正式 cumulative reconstruction frozen-ASR trajectory 已完成，但 frozen ASR
   對 dysarthric speech 的 domain mismatch 與 evaluator bias 仍限制 interpretation；
   codec-native embedding probe 與 acoustic baseline 尚未執行。
9. 沒有進行 mixed-effects analysis 或 statistical significance testing。
10. CUDA runs 未啟用嚴格 deterministic mode；小幅 rerun 差異可能包含 GPU
    nondeterminism。
11. WER/CER 不能解讀為 clinical intelligibility。
12. Severity label source 已由研究者確認為 TORGO 原始論文；本文件不自行補造
    尚未記錄的作者、頁碼或表格細節。

---

## 14. Recommended next step

1. 保存並鎖定 cumulative 與 individual 各 168-run aggregation、paired comparison、
   frozen-ASR reconstruction audit 與 experiment protocol；
2. 完成 Stage 0 representation/provenance table；
3. 保持 individual Q2–Q8 非原生 reconstruction 排除於正式 conditions；
4. 比較 Q1、fixed normalized sum、concatenation 與 static learned weighting；
5. 綜合 token-domain、reconstruction 與 fixed-fusion 證據後，再以 complementarity
   decision gate 決定是否進入 utterance-adaptive fusion。

本 Phase 1 record 不新增mixed-effects model、clinical interpretation或probe
結果。尚未完成的 concatenation、adaptive gating 與 pathology-aware objectives
不得由本文件視為已完成。
