> **Roadmap status: historical Phase 1 specification.** 本文件保留原始需求，不再作為
> 多階段研究的 canonical 終點。其內容對應新版 roadmap 的 Stage 0（representation
> audit）、Stage 1（ASR baseline）與 Stage 2（cumulative depth trajectory）。目前
> 研究方向見 [Pathology-aware RVQ Layer Fusion roadmap](PATHOLOGY_AWARE_RVQ_FUSION_ROADMAP.md)。

請在目前的 RVQ_token repository 中完成第一階段開發。

開始修改前，請先：

1. 閱讀 README.md、AGENTS.md，以及 04_Code/ 下與 TORGO manifest、
   codec extraction、RVQ ASR、codec reconstruction 有關的程式。
2. 盤點目前實作與本任務的差距。
3. 提出逐步實作計畫，先不要修改檔案。
4. 特別檢查目前工作目錄是否有未提交變更，不要覆蓋既有修改。

第一階段目標：
將目前的 pipeline 擴充成可以可靠執行完整 RVQ depth trajectory，
但這一階段不要新增 speaker、dysarthria、severity probes。

需要完成：

A. 完整 RVQ depth
- 將 SpeechTokenizer、EnCodec、DAC reconstruction 的預設 depth，
  由 1,2,4,6,8 改為 1,2,3,4,5,6,7,8。
- 不要把 condition order 寫死；evaluation 與 result comparison
  必須能動態處理實際存在的 k1...kN。
- 必須驗證 requested depth 不超過實際 num_codebooks。

B. Manifest schema
- 在 TORGO manifest 加入明確的 condition 欄：
  control 或 dysarthric。
- 下游 token index、reconstruction index、predictions 必須保留 condition。
- 評估結果必須分別提供 overall、control、dysarthric、severity、speaker。
- 暫時不要自行更改 severity label 或納入／排除任何 speaker。
- 如果 speaker_metadata.csv 仍有 citation TODO，只報告問題，不要猜引用。

C. CTC trajectory
- 讓現有 rvq_asr training pipeline 支援自動執行 depth 1 到 N。
- 每個 depth 必須使用相同 split、model capacity、optimizer、
  training budget 與 seed。
- 新增一個 sweep script，而不是複製八份 training code。
- 每個 run 使用獨立 output directory。
- 彙整每個 depth 的 WER、CER、S/D/I、empty hypothesis ratio、
  CTC blank-frame ratio，以及 control/dysarthric/severity/speaker 結果。

D. 結果彙整
- 新增 long-format CSV，至少包含：
  codec, depth, seed, group_type, group_value, metric, value。
- 新增 trajectory summary：
  每個 depth 的 mean、SD，以及有效 run 數量。
- 這階段不要做 mixed-effects model 或 clinical interpretation。

E. 測試
- 新增或修改測試，以檢查：
  1. depth 1 到 N 都能被正確解析；
  2. condition 從 manifest 傳遞到 prediction；
  3. 動態 condition order；
  4. depth 不可超過 num_codebooks；
  5. sweep 不會覆蓋其他 depth/seed 的輸出；
  6. 現有功能不被破壞。
- 執行相關測試與一個最小 smoke test。
- 若缺少 TORGO 音訊、codec checkpoint 或 GPU，
  使用 synthetic fixtures／mock 完成單元測試，並清楚列出未執行的實驗。

限制：
- 不要下載或提交 TORGO 音訊、codec checkpoints、token files 或模型輸出。
- 不要修改研究標籤來源。
- 不要宣稱 WER 等於 clinical intelligibility。
- 不要同時進行大規模 codec adapter 重構。
- 不要刪除既有功能或使用破壞性 Git 指令。
- 保持 backward compatibility；若無法保持，先說明理由。
- 不要執行完整 GPU 訓練。

完成後請提供：
1. 修改檔案清單；
2. 每個修改的目的；
3. 測試結果；
4. 尚未完成或被環境阻擋的部分；
5. 下一階段建議；
6. 可直接執行完整 depth sweep 的命令範例。
