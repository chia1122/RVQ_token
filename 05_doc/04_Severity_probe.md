> **Roadmap status: revised and gated.** Severity probing 現屬 Stage 3 diagnostic
> probe，並可在 Stage 9 作 exploratory analysis；仍受 speaker-level label、樣本數與
> citation protocol 限制。本文件保留原始規格，不表示已有結果。Canonical 定義見
> [Pathology-aware RVQ Layer Fusion roadmap](PATHOLOGY_AWARE_RVQ_FUSION_ROADMAP.md)。

請完成第四階段：
新增 speaker-level dysarthria severity probing pipeline。

開始前先進行資料可行性檢查，暫時不要修改程式：
1. 列出所有 dysarthric speakers 及其 severity。
2. 列出每個 severity 的 speaker 數、utterance 數與時數。
3. 檢查 severity_source 是否仍有 citation TODO。
4. 檢查 mild speakers 是否被排除。
5. 判斷四分類是否具有最低可行性。

如果 severity citation 未確認，或 label protocol 存在矛盾，
請先停止並報告，不要自行猜測或修改 clinical labels。

在標註確認後，實作以下內容。

A. 任務定義
支援三種模式：

1. Multiclass：
   mild / moderate / moderate-to-severe / severe

2. Ordinal：
   mild < moderate < moderate-to-severe < severe

3. Optional regression：
   0 / 1 / 2 / 3

Regression 模式必須註明：
這是實驗上的等距 operationalization，不是臨床連續尺度。

B. 資料範圍
- 只使用 dysarthric speakers。
- control 不可作為 severity 等級。
- severity 是 speaker-level label。
- 每位 speaker 的所有 utterances 必須使用相同 label。

C. 切分
- 必須 speaker-disjoint。
- 優先使用 LOSO 或 GroupKFold。
- fold 必須保存並可重現。
- 如果某 fold 無法包含所有 severity，不得靜默忽略。
- 需報告每個 fold 的 class composition。

D. Probe
支援：
- linear multiclass classifier
- ordinal classifier
- optional shallow MLP robustness probe

預設使用 mean+std pooling，不使用大型 Transformer。

E. 指標
Multiclass：
- UAR
- Macro-F1
- balanced accuracy
- confusion matrix
- per-class recall

Ordinal/regression：
- MAE
- Spearman correlation
- quadratic weighted kappa

所有結果同時保留：
- utterance-level prediction
- speaker-level aggregation

F. Baseline
每個 fold 都要計算：
- random/chance baseline
- majority baseline
- ordinal median baseline

G. 結果限制
結果只能描述為：
“representation contains information associated with
speaker-level dysarthria severity.”

不能描述為：
“模型能評估每段 utterance 的臨床嚴重程度。”

H. 測試
至少檢查：
1. control 不會進入 severity dataset；
2. speaker 不跨 split；
3. speaker 的 severity label 一致；
4. ordinal mapping 固定；
5. fold 缺少 class 時有清楚警告或停止；
6. speaker-level aggregation 正確；
7. baseline 計算正確；
8. test set 不參與 model/threshold selection。

完成後提供：
1. severity 資料可行性報告；
2. 修改檔案；
3. task formulation；
4. fold composition；
5. 測試結果；
6. smoke test；
7. 完整 severity trajectory 命令；
8. 仍存在的研究限制。
