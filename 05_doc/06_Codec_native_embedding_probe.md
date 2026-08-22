請完成第六階段：
新增 frozen codec-native embedding representation，
並與目前的 discrete learned embedding probe 分開比較。

研究目的：
區分：
1. 從離散 token ID 經任務訓練後可恢復多少資訊；
2. codec 原生 codebook vectors 中有多少資訊可直接讀取。

A. 先做介面調查
修改前請先調查並報告：
- SpeechTokenizer、EnCodec、DAC 的 codebook/quantizer 實作位置；
- 如何取得每個 codebook 的原生 embedding vectors；
- 每個 codec 的 quantization aggregation 方式；
- 是否能正確重現 decoder 使用的 cumulative latent；
- 哪些 parameters 必須 frozen；
- 各 codec 是否存在不同 scaling、normalization 或 projection。

如果無法忠實取得某 codec 的 native embedding，
不要用自行建立的 nn.Embedding 冒充 native embedding。

B. Representation modes
明確分成：

discrete_learned：
token ID → task-trained embedding

codec_native：
token ID → frozen codec codebook lookup

輸出 metadata 必須記錄 representation_mode。

C. Cumulative representation
codec_native 模式必須依照該 codec 實際 RVQ reconstruction 邏輯建立：
Q1、Q1:2……Q1:N。

不能直接把 token ID 相加。

如果 codec 使用額外 projection、scale 或 normalization，
必須忠實保留並加上測試。

D. Individual representation
同時支援：
- cumulative Q1:K
- individual QK residual

CLI：
--representation-mode codec_native
--rvq-mode cumulative/individual
--depth K

E. Frozen constraint
- codec parameters 完全 frozen。
- 預設 probe 使用 linear 或 shallow architecture。
- 記錄 trainable parameter count。
- 驗證 optimizer 不包含 codec parameters。

F. Validation
加入 reconstruction-consistency test：
- full native cumulative latent 應與原 codec quantized latent一致；
- 或在合理數值誤差內重建相同輸出。

如果無法精確比較，清楚記錄 validation 方法和限制。

G. 測試
至少檢查：
1. codec parameters 不更新；
2. token lookup 使用正確 codebook；
3. Q1:K 只含前 K 層；
4. individual QK 只含指定 residual；
5. padding 不產生額外 embedding；
6. native/full latent consistency；
7. 不同 depth 輸出維度一致；
8. discrete_learned 與 codec_native 不會混用 checkpoint。

實作順序：
1. SpeechTokenizer
2. EnCodec
3. DAC

每完成一個 codec，先測試後再做下一個。
不要在本階段加入 Mimi。

完成後提供：
1. 每個 codec 的 quantizer 調查；
2. 修改檔案；
3. native embedding 定義；
4. consistency validation；
5. 測試結果；
6. 執行比較的命令；
7. 無法完全跨 codec 對齊的差異。
