> **Roadmap status: superseded as the final analysis stage.** 跨 seed、rotation 與
> representation 的統計整合現改列新版 roadmap 的 Stage 7；本文件保留原始規格，
> 其中 mixed-effects analysis 不屬於目前已完成範圍。Canonical 定義見
> [Pathology-aware RVQ Layer Fusion roadmap](PATHOLOGY_AWARE_RVQ_FUSION_ROADMAP.md)。

請完成第五階段：
統一彙整 linguistic、speaker、dysarthria、severity 四種 probe 的
RVQ information trajectories，並加入統計分析。

前置條件：
- 四種 probe 已能獨立執行。
- 各 probe 已輸出一致或可轉換的 long-format results。
- 本階段不重新設計或訓練 probe。

A. 統一結果 schema
將結果整理為 long-format，至少包含：
experiment_id, git_commit, codec, checkpoint, task,
representation_mode, rvq_mode, depth, seed, fold,
group_type, group_value, speaker_id, condition,
severity, metric, value, num_samples。

B. 多 seed 彙整
- 預設支援至少 3 個 seeds。
- 計算 mean、SD、95% confidence interval。
- 保留有效 run 數與失敗 run 清單。
- 不得只選最佳 seed 報告。

C. Trajectory
對每個 codec/task 計算：
- depth score
- marginal gain
- normalized score
- saturation depth
- full-RVQ gap

對「越低越好」的 WER/CER/MAE，
必須先定義一致方向的 score 或使用正確的不等式，
不能直接套用越高越好的 saturation 公式。

D. Saturation
支援 configurable threshold，例如 90%、95%、99%。

定義與實作必須明確處理：
- non-monotonic trajectory
- negative scores
- full-RVQ 不是最佳點
- metric 越低越好的情況

E. Bootstrap
- 使用 speaker-cluster bootstrap。
- speaker 是 resampling unit。
- 不使用純 utterance bootstrap 作主要信賴區間。
- 支援不同 depth 的 paired comparison。
- paired comparison 必須使用同一組 speaker/items。

F. Interaction analysis
輸出適合 mixed-effects analysis 的資料。

如果在 Python 實作 mixed model，可使用：
Score ~ Depth + Condition + Depth:Condition + (1|Speaker)

若使用 R，新增可重現 script。

核心輸出：
- Depth effect
- Condition effect
- Depth × Condition interaction
- confidence interval
- effect size
- multiple-comparison correction

不要因模型無法收斂而靜默改模型。

G. 圖表
產生：
- score vs depth
- marginal gain vs depth
- control vs dysarthric trajectory
- codec comparison
- saturation summary

圖表必須有：
- metric 名稱
- confidence interval
- sample/fold/seed 說明
- 越高或越低越好的方向

H. 測試
至少檢查：
1. 多 seed mean/SD 正確；
2. missing runs 不會被當成零；
3. lower-is-better saturation 正確；
4. non-monotonic trajectory 處理正確；
5. speaker bootstrap 不會拆散同一 speaker；
6. paired comparison 對齊相同項目；
7. multiple-comparison correction 正確；
8. synthetic interaction 可被正確回收。

限制：
- 不重新訓練 probe。
- 不把顯著性等同於臨床重要性。
- 不把 utterance 數當作獨立 speaker 數。

完成後提供：
1. 統一 schema；
2. trajectory/saturation 定義；
3. 統計方法；
4. 測試結果；
5. 圖表輸出位置；
6. 完整彙整命令；
7. 尚未滿足統計前提的分析。
