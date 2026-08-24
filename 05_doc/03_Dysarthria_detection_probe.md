> **Roadmap status: revised.** 本 probe 現屬 Stage 3 diagnostic probe，並作為
> Stage 6 pathology-aware fusion 的輔助分析；它不是 clinical diagnosis。本文件保留
> 原始規格，尚不代表正式實驗已完成。Canonical 定義見
> [Pathology-aware RVQ Layer Fusion roadmap](PATHOLOGY_AWARE_RVQ_FUSION_ROADMAP.md)。

請完成第三階段：
新增 control vs dysarthric detection probing pipeline。

前置條件：
- manifest 已包含明確 condition：
  control 或 dysarthric。
- 完整 depth sweep 與共用 result schema 已存在。
- 本階段只做二元 dysarthria detection。

研究目的：
分析與 dysarthria condition 相關的資訊主要出現在 early 還是 late RVQ stages，
並比較不同 codec 的 trajectory。

A. 資料切分
- 必須使用 speaker-disjoint evaluation。
- 支援 Stratified GroupKFold，group=speaker_id。
- 額外支援 Leave-One-Speaker-Out。
- 同一 speaker 絕對不能跨 train/test。
- validation 也應以 speaker 為 group。
- fold assignment 必須保存成檔案並可重現。
- 不得使用 speaker identity 任務的 session-disjoint split。

B. 混淆因素
保留並檢查：
- gender
- duration
- prompt type（若 manifest 有）
- speaker utterance count
- session
- microphone

加入可選分析：
- gender-matched subset
- common-prompt subset
- speaker-balanced sampling 或 speaker-level weighting

如果資料無法支援其中一項，請報告，不要虛構欄位。

C. Representation
- 支援 Q1、Q1:2……Q1:N。
- 沿用 discrete learned representation。
- 預設 mean+std pooling。
- 不在本階段加入 codec-native embedding。

D. Probe
主要 probe：
- logistic regression 或單層 linear classifier。

可選 robustness probe：
- 一層 shallow MLP。

不要加入大型 sequence encoder。

E. 指標
每個 fold 輸出：
- UAR
- Macro-F1
- AUROC
- sensitivity
- specificity
- balanced accuracy
- confusion matrix
- per-speaker prediction

閾值必須由 validation set 決定，不能使用 test set 調整。

F. 統計單位
- 保留 utterance-level predictions。
- 同時產生 speaker-level aggregation。
- 不得把數千個 utterances 當成完全獨立的臨床樣本。
- confidence interval 必須以 speaker 為 resampling unit。

G. Sweep
支援：
codec × depth × seed × fold × probe_type。

輸出 long-format：
codec, task, depth, seed, fold, speaker_id,
condition, metric, value。

H. 測試
至少檢查：
1. speaker 不會跨 train/test；
2. condition mapping 正確；
3. fold 可重現；
4. 每個 fold 的 train/test 都有兩個 condition；
5. class weighting 正確；
6. speaker aggregation 正確；
7. AUROC 在單類 fold 時不會產生誤導結果；
8. test set 未參與 threshold selection。

限制：
- 本任務是 condition detection，不是醫療診斷系統。
- 不宣稱模型具有臨床診斷能力。
- 不修改 TORGO severity 定義。
- 不執行完整 GPU sweep。

完成後提供：
1. fold 設計；
2. 修改檔案；
3. 混淆因素處理方式；
4. 測試結果；
5. smoke test；
6. 執行完整 detection trajectory 的命令。
