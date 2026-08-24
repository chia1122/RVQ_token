# Dataset Protocol Amendment — Include Mild Speakers and Speaker-Level Folds

## 1. Status and scope

本文件最初記錄經確認的 TORGO dataset protocol proposal；目前該 proposal 已落實為
versioned metadata、七個預先定義的 speaker folds、cyclic rotations 與 audit 工具。
主要變更是將 F04 與 M03 納入後續實驗。

本 proposal 不回溯修改或覆蓋 Phase 1 的 fixed-split trajectory。Phase 1 結果
仍代表原始 protocol；新 protocol 必須使用不同的 config、manifest output、token
output 與 experiment output identifiers。

本文件保留 proposal 的設計依據。實作採用新的 versioned resources，且未修改：

- canonical `speaker_metadata.csv`；
- canonical `speaker_splits.json`；
- Phase 1 manifests、codec tokens 與 experiment outputs。

目前 protocol 的 canonical 研究路線與階段定位見
[Pathology-aware RVQ Layer Fusion roadmap](PATHOLOGY_AWARE_RVQ_FUSION_ROADMAP.md)。

---

## 2. Protocol amendment

F04 與 M03 的 severity 維持既有 `mild`，但 inclusion flag 由原 protocol 的
`false` 改為新 protocol 的 `true`。

| Speaker | Condition | Gender | Severity | Original inclusion | New inclusion |
|---|---|---|---|---|---|
| F04 | dysarthric | female | mild | false | true |
| M03 | dysarthric | male | mild | false | true |

這是 speaker inclusion protocol 的變更，不是 severity label 的重新定義。
F04、M03 的 severity 與 inclusion 依據已由研究者確認來自 TORGO 原始論文。
正式更新 metadata 時，`severity_source` 應填入實際採用的 TORGO 論文引用；本文件
不自行補造作者、年份、頁碼或表格編號。

新 protocol 的 speaker coverage 為：

- dysarthric：8 speakers；
- control：7 speakers；
- total：15 speakers。

---

## 3. Versioning and backward compatibility

不得直接覆蓋 Phase 1 使用的 manifest 與 outputs。建議新增 versioned resources：

```text
04_Code/torgo_manifest/config/speaker_metadata_including_mild_v1.csv
04_Code/torgo_manifest/config/speaker_folds_including_mild_v1.json
04_Code/torgo_manifest/output_including_mild_v1/
```

後續 codec tokens 與 model outputs 也必須使用新目錄，例如：

```text
04_Code/torgo_manifest/speechtokenizer_including_mild_v1_tokens/
04_Code/rvq_asr/trajectories/including_mild_v1/
```

上述 token 與 experiment output directories 不應提交至 Git。

原始檔案保留用途：

- `speaker_metadata.csv`：Phase 1 protocol provenance；
- `speaker_splits.json`：Phase 1 fixed development split；
- `torgo_manifest/output/`：Phase 1 manifest provenance。

若未來決定直接更新 canonical `speaker_metadata.csv`，仍必須使用新的 manifest、
token 與 experiment output directories，並在版本紀錄中標示 protocol boundary。

---

## 4. Proposed seven speaker folds

納入 F04 與 M03 後，共有 8 位 dysarthric speakers 與 7 位 control speakers。
若使用八個 outer folds，至少會有一個 fold 無法同時包含 control 與 dysarthric
speaker。因此採用七個 folds，其中 fold G 包含兩位 dysarthric speakers。

| Fold | Dysarthric speakers | Control speakers | Dysarthric severity | Estimated utterances |
|---|---|---|---|---:|
| A | F01 | MC01 | severe | 1,184 |
| B | F04 | MC03 | mild | ≤1,037 |
| C | M04 | FC03 | severe | 1,228 |
| D | M01 | FC02 | severe | 1,333 |
| E | M02 | MC04 | severe | 1,009 |
| F | F03 | MC02 | moderate | 985 |
| G | M03, M05 | FC01 | mild, moderate-to-severe | ≤1,009 |

F04 的 244 rows 與 M03 的 401 rows 目前都在既有
`excluded_samples.csv` 中被標記為 `speaker_excluded_by_protocol`。這些數量只是
selected-channel candidate rows 的上限。改成 included 後，builder 才會繼續套用
transcript normalization、unintelligible/numeric transcript、audio existence 與 WAV
validation，因此正式 utterance 數必須由新 manifest build audit 決定。

Fold assignment 的設計目標為：

1. 每個 fold 至少包含一位 control 與一位 dysarthric speaker；
2. 每位 included speaker 恰好出現在一個 fold；
3. fold utterance totals 儘量接近；
4. 不以 utterance-level random split 取代 speaker-level split；
5. 不因 fold balancing 修改任何 severity label。

Control speaker assignment 經最小交換調整：Fold B 使用 MC03，Fold D 使用
FC02。這可避免任何 fold 完全由 female speakers 組成，同時保留 condition
coverage、speaker disjointness 與合理的 utterance-count balance。由於 7 folds
只有 6 位 female speakers，不可能讓每個 fold 都同時包含 female 與 male；因此
Fold E 仍為 all-male，這項 gender coverage limitation 必須保留在 audit 與結果
解讀中。

---

## 5. Train/validation/test rotation

採用 cyclic rotation。每次以一個 fold 作 test、下一個 fold 作 validation，其餘
五個 folds 作 train。

| Rotation | Train folds | Validation fold | Test fold |
|---:|---|---|---|
| 1 | C, D, E, F, G | B | A |
| 2 | D, E, F, G, A | C | B |
| 3 | E, F, G, A, B | D | C |
| 4 | F, G, A, B, C | E | D |
| 5 | G, A, B, C, D | F | E |
| 6 | A, B, C, D, E | G | F |
| 7 | B, C, D, E, F | A | G |

此 rotation protocol 必須滿足：

- train、validation、test speakers 完全不重疊；
- 每位 included speaker 恰好進入 test 一次；
- 每位 included speaker 恰好進入 validation 一次；
- 每個 test fold 都有 control 與 dysarthric coverage；
- 每個 depth、seed 使用完全相同的 folds；
- checkpoint selection 只使用該 rotation 的 validation fold；
- test fold 不參與 hyperparameter 或 checkpoint selection。

完整 K1–K8、三 seeds、七 rotations 的 run 數為：

```text
8 depths × 3 seeds × 7 rotations = 168 runs
```

每個 rotation/depth/seed 必須使用獨立 output directory，不能覆蓋其他 runs。

### 5.1 Protocol classification

目前實作是研究者預先定義的七個 speaker folds，加上
`next_fold_cyclic` validation strategy。它保證 rotation 內的 train、validation、
test speaker-disjoint，但不是由 sklearn 動態產生的 `GroupKFold` 或
`StratifiedGroupKFold`，也不是每次只留一位 speaker 的 LOSO。後續報告必須使用
「predefined seven-fold cyclic speaker-disjoint protocol」等準確名稱，不能將三者
視為同一設計。

---

## 6. Severity coverage and interpretation limits

新 protocol 的 dysarthric severity coverage 為：

| Severity | Speakers | Number of speakers |
|---|---|---:|
| mild | F04, M03 | 2 |
| moderate | F03 | 1 |
| moderate-to-severe | M05 | 1 |
| severe | F01, M01, M02, M04 | 4 |

納入 mild speakers 可改善既有 test split 完全沒有 mild coverage 的問題，但不能
使 severity groups 變成平衡樣本。Moderate 與 moderate-to-severe 仍各只有一位
speaker，因此：

- severity results 只作描述性 reporting；
- utterances 不能視為獨立 clinical subjects；
- 不宣稱 utterance-level severity；
- 不進行 clinical diagnosis；
- 不將 WER/CER 解讀為 clinical intelligibility；
- 未來若進行 statistical analysis，必須保留 speaker clustering。

---

## 7. Fold audit requirements

後續 fold audit 工具必須執行以下驗證。

### 7.1 Metadata and inclusion

1. Metadata 與 fold config 中的 speaker IDs 完全對應。
2. 15 位 included speakers 全部存在。
3. F04、M03 為：
   - `speaker_type=dysarthric`；
   - `severity=mild`；
   - `include_in_experiment=true`。
4. 每位 included speaker只出現在一個 fold。
5. 沒有未經記錄的 label 或 inclusion modification。
6. `severity_source` 不得為空；citation 字串由研究者提供，不由程式猜測。

### 7.2 Split integrity

1. 每個 fold 都包含 control 與 dysarthric speakers。
2. 每個 rotation 的 train、validation、test 完全 speaker-disjoint。
3. 每位 included speaker 恰好 test 一次。
4. 每位 included speaker 恰好 validation 一次。
5. 每個 role 至少包含一筆有效 utterance。
6. Generated split configs 可由現有 manifest builder 讀取。

### 7.3 Coverage statistics

每個 rotation/role 至少輸出：

- speaker IDs；
- speaker count；
- condition speaker counts；
- severity speaker counts；
- gender speaker counts；
- utterance count；
- duration hours；
- unique normalized-text count；
- missing or invalid audio count；
- exclusions by reason。

Severity 缺少某個 level 應記錄為 coverage limitation，不應自動把其他 label
合併、重新命名或補值。

---

## 8. Prompt-overlap policy

既有 Phase 1 fixed split 中，99.36% test utterances 的 normalized text 也曾出現在
train。Speaker-disjoint 能防止同一 speaker 同時出現在 train/test，但不能保證
prompt-disjoint。

新 protocol 的 primary estimand 定義為：

> ASR generalization to unseen speakers under a largely shared-prompt TORGO
> protocol.

Prompt overlap 在 primary protocol 中允許，但每個 rotation 必須 audit：

- train/validation/test unique normalized texts；
- train–validation、train–test、validation–test unique-text overlap；
- test utterances whose normalized text appeared in train；
- seen-prompt 與 unseen-prompt utterance counts；
- 若資料量足夠，分別提供 seen/unseen-prompt ASR metrics。

若要研究 novel-prompt generalization，應建立另一個 versioned
speaker-and-prompt-disjoint sensitivity protocol。該 protocol 不得覆蓋或混入本次
speaker-fold primary results，並須先確認資料保留率是否足以支援訓練與評估。

---

## 9. Training balance policy

不同 speakers 的 utterance 數量差異很大。資料 split 與 training sampling 應分開
處理：

- validation/test 保留全部符合條件的 utterances，不進行平衡抽樣；
- training 可評估 speaker-uniform sampling；
- 若使用 speaker-uniform sampling，先等機率選 speaker，再選該 speaker 的
  utterance；
- sampling policy、steps/epoch、optimizer budget 與 seed 必須對所有 depths
  相同；
- sampler 不得根據 validation/test 結果事後調整；
- sampler sensitivity 必須使用獨立 experiment identifier。

在 sampler 尚未實作前，至少必須同時報告 micro 與 speaker-macro metrics，避免
utterance 數較多的 speakers 完全主導結果。

---

## 10. Reporting protocol

每個 depth、seed、rotation 均保留獨立結果。Trajectory aggregation 至少提供：

1. micro overall WER/CER；
2. per-speaker WER/CER；
3. speaker-macro WER/CER；
4. condition speaker-macro WER/CER；
5. descriptive severity results；
6. S/D/I；
7. empty hypothesis ratio；
8. CTC blank-frame ratio；
9. valid fold/run count；
10. prompt-overlap coverage。

Seeds、folds、speakers 與 utterances 不代表相同層級的獨立 observations。正式
統計設計前先保留各層級 identifiers，不將所有 utterances 當成獨立 subjects。

WER/CER 僅作 ASR performance 指標，不作 clinical intelligibility 解讀。

---

## 11. Fold-definition and audit implementation status

Fold-definition and audit stage 已新增：

```text
04_Code/torgo_manifest/config/speaker_metadata_including_mild_v1.csv
04_Code/torgo_manifest/config/speaker_folds_including_mild_v1.json
04_Code/torgo_manifest/audit_speaker_folds.py
04_Code/torgo_manifest/test_audit_speaker_folds.py
```

並更新操作文件：

```text
04_Code/torgo_manifest/README.md
```

Audit CLI 提供：

- config-only validation；
- cyclic train/validation/test rotation generation；
- builder-compatible split configs；
- optional manifest coverage audit；
- optional exclusions-by-reason audit；
- prompt-overlap statistics；
- non-empty output-directory overwrite protection。

本階段未擴大為 training pipeline、sampler 或 codec adapter 重構，也未修改
canonical metadata、fixed split、Phase 1 manifests 或 experiment results。

---

## 12. Tests and smoke-audit status

Synthetic tests 已涵蓋：

1. 七個 folds 正確解析；
2. included speakers 完整且不重複；
3. F04、M03 inclusion/severity assertion；
4. 每個 fold 的 condition coverage；
5. rotation train/validation/test speaker disjointness；
6. 每位 speaker test/validation 各一次；
7. invalid/missing/duplicate speakers 會失敗；
8. severity coverage 會報告但不自動改 label；
9. prompt-overlap 統計正確；
10. generated split configs 與既有 builder schema 相容；
11. synthetic fixtures 不需要 TORGO audio、codec checkpoint 或 GPU；
12. config-only CLI smoke 輸出至暫存目錄。

Manifest-aware synthetic smoke 已完成。其後已在 Linux 工作站以 versioned
metadata 重建正式 manifest，並完成 manifest-aware audit。該 artifact 位於
repository 外的持久實驗儲存空間，不納入 Git。

已記錄的驗證結果分為 repository 測試與工作站實驗 audit：

- fold/audit synthetic tests：11 passed；
- 完整 `torgo_manifest` regression suite：22 passed；
- config-only CLI smoke：passed，產生 7 個 split configs；
- manifest-aware synthetic integration smoke：passed；
- 舊 Phase 1 manifest negative smoke：依預期拒絕，且未留下 partial output；
- 正式 TORGO mild-speaker manifest audit：passed，共 7,785 utterances、15 speakers，
  missing audio 為 0；
- 七個 rotation-specific token indices audit：passed，每個 index 皆有 7,785 rows，
  未發現 missing token files 或 metadata mismatch；
- 七個 CER-selected formal rotations：168/168 runs valid；每個 rotation 均為
  24/24 valid，且皆有 long-format 與 summary CSV；
- 跨 rotation aggregation 工具已實作；正式工作站 aggregation output 仍需另行
  執行與 audit。

後三項為工作站上外部 artifacts 的 audit 紀錄；repository 不保存 manifest JSONL、
codec tokens、model outputs 或完整 audit outputs。它們不是由本次文件更新重新執行
所得。

不執行完整 GPU training，也不提交 TORGO audio、manifests、tokens、checkpoints
或完整 experiment outputs。

---

## 13. Recommended execution order

1. 保留 versioned metadata、fold config 與 repository 外 artifacts 的 provenance。
2. 保留七個 rotations 的 168/168 valid audit，且不可重用任何既有 output directory。
3. 執行並 audit speaker-macro、condition、severity 與 per-speaker 彙整，保留 rotation、seed、
   depth identifiers。
4. 依 canonical roadmap 完成 Stage 0 representation extraction feasibility audit。
5. 在 matched protocol 下建立 individual Q1–Q8 與固定 cumulative fusion baselines。
6. 只有在 baseline 與 complementarity gate 通過後，才啟動 adaptive fusion。

目前不需要重新建立已通過 audit 的 manifest。若重建，必須使用新的 versioned output
directory，並重新執行完整 manifest、token-index 與 split audit。
