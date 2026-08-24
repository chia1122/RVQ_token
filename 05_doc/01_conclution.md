# Phase 1 — SpeechTokenizer RVQ Depth Trajectory

> **Record status:** completed fixed-split pilot and diagnostic evidence. This
> document preserves the original metrics and checkpoint-selection comparison.
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

現有程式可透過明確指定 active layer 建立 individual learned-layer condition，
但 Phase 1 sweep 未使用該模式，也未完成 matched individual Q1–Q8 trajectory。
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

## 10. Current interpretation

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

### 10.1 Meaning under the revised research direction

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

## 11. Limitations

1. 每個 depth 只有三個 seeds。
2. 使用單一固定 speaker-disjoint split。
3. Test split 的 severity／speaker coverage 有限。
4. 深層 WER 接近飽和，不適合作為唯一 checkpoint-selection signal。
5. 多個 minimum-CER epochs 位於 epoch 27–30，仍可能存在 training-budget
   sensitivity。
6. CER-selected severity coverage 只有 `control` 與 `moderate-to-severe`，且
   speaker coverage 只有 FC03、M05、MC04，不代表其他 severity 或 speakers。
7. 尚未比較其他 token fusion mechanisms。
8. 尚未執行 codec-native embedding probe 或 acoustic baseline。
9. 沒有進行 mixed-effects analysis 或 statistical significance testing。
10. CUDA runs 未啟用嚴格 deterministic mode；小幅 rerun 差異可能包含 GPU
    nondeterminism。
11. WER/CER 不能解讀為 clinical intelligibility。
12. `speaker_metadata.csv` 中仍有 citation TODO，未自行猜測或補寫來源。

---

## 12. Recommended next step

1. 合併並 audit 已完成的七個 speaker rotations（168/168 valid runs）；
2. 分開報告 run-macro、pooled-micro 與 speaker-macro trajectory；
3. 鎖定 folds、seeds、training budget、capacity 與 checkpoint-selection protocol；
4. 完成 Stage 0 representation/provenance table；
5. 建立 matched individual-layer 與 fixed-fusion baselines；
6. 以 complementarity decision gate 決定是否進入 utterance-adaptive fusion。

本 Phase 1 record 不新增mixed-effects model、clinical interpretation或probe
結果。尚未實作的individual sweep、concatenation、adaptive gating與pathology-aware
objectives不得由本文件視為已完成。
