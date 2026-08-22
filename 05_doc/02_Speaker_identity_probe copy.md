請在目前的 RVQ_token repository 中完成第二階段：
新增 speaker identity probing pipeline。

前置條件：
- 第一階段的完整 RVQ depth trajectory 已完成。
- manifest 和 token index 已包含 speaker_id、session_id、condition、
  severity、codec、num_codebooks 等資訊。
- 本階段只做 speaker identity，不做 dysarthria 或 severity prediction。

研究目的：
測量已知 speaker identity 在 Q1、Q1:2……Q1:N 中的可讀取程度，
以及 speaker information 在哪個 RVQ depth 開始飽和。

A. 資料切分
- Speaker identity 任務允許同一 speaker 出現在 train/valid/test。
- 但 utterance 不得重複。
- 優先使用 session-disjoint split。
- 若部分 speaker 缺少足夠 session，請報告並提供可重現的 fallback。
- 不得直接沿用 clinical probe 的 speaker-disjoint split。
- 新增 split leakage validation。

B. Representation
- 支援 cumulative RVQ depth：
  Q1、Q1:2……Q1:N。
- 沿用目前的 discrete learned representation。
- 不要在本階段實作 codec-native embeddings。
- frame-level representation 預設使用 mean pooling。
- 額外支援 mean+std pooling 作為可選 ablation。
- 不同 depth 必須保持相同輸出維度。

C. Probe
- 實作 linear speaker classifier 作為主要 probe。
- 可額外支援一層 shallow MLP，但不要使用大型 Transformer。
- 所有 depth 使用相同 architecture、training budget 和超參數。
- 支援 class weighting，處理各 speaker utterance 數量不平衡。

D. 指標
輸出：
- Top-1 accuracy
- Macro-F1
- balanced accuracy
- per-speaker precision/recall/F1
- confusion matrix
- 每位 speaker 的 utterance 數量

E. Sweep
新增或擴充 sweep script，支援：
codec × depth × seed × pooling × probe_type。

每個 run 使用獨立 output directory，不能互相覆蓋。

F. 結果格式
建立 long-format 結果，至少包含：
codec, task, depth, seed, pooling, probe_type,
split, speaker_id, metric, value。

產生：
- 每個 depth 的 mean ± SD
- speaker trajectory summary
- per-speaker results
- confusion matrix
- run configuration

G. 測試
至少測試：
1. session leakage 可被偵測；
2. 每個 speaker 都有合法的 train/test samples；
3. pooling 忽略 padded frames；
4. Q1:K 只使用前 K 層；
5. class label mapping 固定且可重現；
6. sweep 不覆蓋其他 runs；
7. Macro-F1 和 balanced accuracy 計算正確。

限制：
- 不做 speaker verification、EER 或 minDCF。
- 不把 speaker identity 結果解釋為 unseen-speaker generalization。
- 不修改 severity label。
- 不執行完整 GPU sweep。

完成後提供：
1. 修改檔案清單；
2. 資料切分設計；
3. probe 架構；
4. 測試結果；
5. smoke-test 結果；
6. 執行完整 speaker trajectory 的命令。
