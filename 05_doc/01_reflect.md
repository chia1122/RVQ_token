已完成唯讀盤點，目前尚未修改檔案。

工作樹與規範：

- `git status --short` 為空，沒有未提交變更。
- repository 與父層皆找不到 `AGENTS.md`；因此無額外 repository-specific agent 規範可套用。
- 已閱讀 root `README.md`、相關模組 README、manifest builder、三種 codec extractor/reconstructor、ASR evaluator/comparator、RVQ CTC training/data/model/text 與現有測試。
- `speaker_metadata.csv` 仍有多筆 `citation TODO`。只會報告，不會猜測或修改引用、severity、speaker 納排。
- repository 追蹤了一些既有 runs/results；實作與測試不會改動這些產物。

## 目前差距

A. RVQ depth：

- 三種 reconstructor 預設仍是 `1,2,4,6,8`。
- evaluator 預設及比較欄位順序寫死為 `original,k1,k2,k4,k6,k8`。
- DAC 已明確用 `num_codebooks` 驗證；EnCodec/SpeechTokenizer 主要靠 tensor shape 在 decode 時才失敗，尚未統一做明確驗證。
- depth parser 已能解析任意正整數，但測試未涵蓋完整 `1..N`。

B. Manifest schema：

- manifest 有 `speaker_type`，但沒有要求的 `condition`。
- token index、reconstruction index、CTC predictions 沒有完整傳遞 control/dysarthric。
- reconstruction 程式目前已把 `condition` 用作 `k1/k2/...`，與新 manifest `condition=control/dysarthric` 發生命名衝突。
- reconstruction evaluator 只有 overall/severity/speaker，缺 control/dysarthric 分組。

C. CTC trajectory：

- 單一 run 已驗證 depth 不超過 index 第一筆的 `num_codebooks`，dataset 也會逐筆驗證。
- 現有模型會配置全部可用 codebook embeddings，只啟用前 K 層，已有維持 capacity 的基礎。
- 尚無 depth sweep orchestrator、run collision 防護或跨 run 彙整。
- 已有 WER、CER、empty hypothesis ratio、blank-frame ratio；缺少 S/D/I 與 condition 分組。
- `train_probe_with_limits.py` 幾乎是 `train_probe.py` 的複製版，只有 split limits 與 logging 差異。這階段不做大規模重構，但需避免兩者 schema 漂移。

D. 結果彙整：

- 尚無指定格式的 long CSV。
- 尚無跨 seed 的 mean、sample SD、有效 run 數量。
- 現有 report scripts 偏向固定實驗名稱，不適合動態 trajectory。

E. 測試：

- 現有測試只涵蓋少量 helper/model/text 行為。
- 缺 condition propagation、動態 depth order、完整 codebook validation、sweep output isolation、aggregation 與 CPU synthetic smoke test。
- 此輪依「先不要修改檔案」要求，尚未執行測試，避免產生或更新 cache。

## 預計實作計畫

1. Schema 與相容性

   - manifest 新增 canonical `condition`，值由既有 `speaker_type` 驗證映射，只允許 `control` 或 `dysarthric`。
   - 保留 `speaker_type`，不改既有 severity/speaker 設定。
   - 三種 token payload/index 傳遞 `condition`。
   - CTC dataset 對新 index 使用 `condition`；舊 index 缺欄時使用 `unknown`，以維持 LibriSpeech與舊 token index 可讀性。

2. 解決 `condition` 名稱衝突

   - downstream canonical `condition` 保留 control/dysarthric。
   - 將 `original/kN` 明確存為 `rvq_condition`。
   - repository 內的 reader 同時接受：
     - 新格式：`condition` + `rvq_condition`
     - 舊格式：`condition=kN/original`
   - CLI `--conditions` 暫時保留名稱，以免破壞既有命令，但內部表示改用 `rvq_condition`。
   - 這能維持舊資料可讀；新 predictions 的 `condition` 語意會依本任務改為 control/dysarthric。外部若直接把 prediction 的舊 `condition` 當 K 值，需改讀 `rvq_condition`，這是同名欄位衝突下無法完全避免的相容性邊界。

3. 完整 depth reconstruction

   - SpeechTokenizer、EnCodec、DAC 預設改成 `1,2,3,4,5,6,7,8`。
   - 建立共用 depth parser/order helper：
     - 正整數、去重、數值排序；
     - `original` 優先，接著依 K 數值排序；
     - 不假定只有八層或固定清單。
   - 三種 reconstructor 對每個 token payload 的實際 `num_codebooks` 明確驗證，再開始輸出該 utterance。
   - reconstruction index 傳遞 condition。

4. Reconstruction evaluation

   - conditions 可從 reconstruction index 自動發現，亦保留明確 `--conditions` 選項。
   - summary 提供 overall、condition、severity、speaker。
   - comparison CSV 欄位依實際存在的 `original/k1...kN` 動態生成。
   - comparator 改用同一動態排序規則。

5. CTC metrics 與 predictions

   - dataset/collator/prediction/frame output 傳遞 condition。
   - 新增可測試的 Levenshtein alignment，計算 substitutions、deletions、insertions。
   - 保留既有結果 keys，另外新增統一的 grouped metrics：
     - overall
     - condition：control/dysarthric
     - severity
     - speaker
   - 每組輸出 WER、CER、S/D/I counts/rates、empty hypothesis ratio、CTC blank-frame ratio及有效 utterance/reference totals。

6. Depth sweep script

   - 新增單一 `rvq_asr/sweep_depths.py`，呼叫現有 training entry point，不複製八份訓練程式。
   - 預設由 token index 的實際一致 codebook count 產生 `1..N`。
   - 所有 depths 共用同一 token index、model/optimizer/budget arguments 與 seeds。
   - output path 固定隔離為類似：
     `OUTPUT_ROOT/<codec>/depth_k<K>/seed_<SEED>/`
   - 拒絕重複 depth/seed、越界 depth、會碰撞的 output；提供明確 resume 行為，不靜默覆蓋。
   - 使用獨立 subprocess 執行每個 run，讓 GPU/失敗狀態彼此隔離。
   - 支援 dry-run，供單元測試驗證命令與輸出路徑。

7. Trajectory aggregation

   - 產生 `trajectory_long.csv`，至少包含：
     `codec,depth,seed,group_type,group_value,metric,value`
   - 產生 `trajectory_summary.csv`，包含：
     `codec,depth,group_type,group_value,metric,mean,sd,n_valid`
   - SD 使用 sample SD；單一有效 run 時 SD 留空，不偽造為 0。
   - 缺檔、失敗或非有限值不納入 `n_valid`，另留 audit/status。
   - 不加入 mixed-effects model 或 clinical interpretation。

8. 測試與 smoke test

   - synthetic manifest/token fixtures，避免 TORGO/checkpoint/GPU。
   - 覆蓋：
     1. `1..N` depth parsing；
     2. manifest → token → reconstruction/prediction condition；
     3. 動態 `original,k1...kN` order；
     4. depth 超過任一實際 `num_codebooks` 時失敗；
     5. depth/seed output isolation；
     6. legacy index、CLI defaults、既有 model/text 功能。
   - 執行相關 unittest/pytest。
   - 執行一個極小 CPU CTC smoke run或 mock sweep。
   - 不執行完整 GPU training、codec checkpoint 載入或 TORGO 實驗。

9. 文件

   - 更新 root 與模組 README 的完整 depth、schema、輸出和 sweep 命令。
   - 明確說明 WER/CER 是 ASR 指標，不等同 clinical intelligibility。
   - 報告 citation TODO 與未執行的資料/GPU實驗。

如果這份計畫可接受，我下一輪會依此開始修改與驗證。