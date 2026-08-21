請完成第九階段：
擴充 codec reconstruction evaluation，
比較 reconstruction fidelity 與 information-probe trajectory。

A. Reconstruction conditions
支援：
- original
- Q1
- Q1:2
- ...
- Q1:N

條件必須動態讀取，不得寫死 N=8。

B. Metrics
至少加入：
- WER
- CER
- STOI
- SI-SDR
- mel-spectral distance
- speaker embedding cosine similarity

若 PESQ 套件、sample rate 或授權條件不適用，
請不要強行加入，改為記錄限制。

C. Group results
分別輸出：
- overall
- control
- dysarthric
- severity
- speaker
- codec
- depth

D. Pairing
跨 depth 和跨 codec 比較必須使用相同 utterance intersection。
需報告：
- 原始樣本數
- paired samples 數
- 排除樣本與原因

E. Relationship analysis
將 reconstruction metrics 與 probe metrics依 codec/depth 對齊，
分析：
- reconstruction quality vs linguistic accessibility
- reconstruction quality vs speaker accessibility
- reconstruction quality vs dysarthria accessibility
- reconstruction quality vs severity accessibility

輸出 correlation，但不要把 correlation 解釋為 causation。

F. 測試
至少檢查：
1. original/Q1:QN 正確對齊；
2. metric sample rate要求；
3. silent/invalid audio處理；
4. paired intersection正確；
5. group aggregation正確；
6. lower/higher-is-better方向正確；
7. 缺失條件不會被當成零；
8. synthetic identical waveform得到合理metric。

完成後提供：
1. metric 定義；
2. 修改檔案；
3. pairing protocol；
4. 測試結果；
5. 執行命令；
6. 結果表格式；
7. metric 在 dysarthric speech 上的限制。
