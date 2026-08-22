## 各階段工作目的

| 階段                              | 工作目的                                                                          |
| ------------------------------- | ----------------------------------------------------------------------------- |
| Stage 1：完整 RVQ trajectory       | 讓現有 pipeline 能完整執行 Q1、Q1:2……Q1:N，並分組彙整 linguistic 結果。                         |
| Stage 2：Speaker probe           | 分析 speaker identity 在各 RVQ depth 的出現與飽和位置。                                    |
| Stage 3：Dysarthria detection    | 分析 control／dysarthric 資訊主要存在於哪些 RVQ stages。                                   |
| Stage 4：Severity probe          | 分析與 speaker-level dysarthria severity 相關的資訊如何隨 depth 改變。                      |
| Stage 5：統計分析                    | 彙整多 seed／fold 結果，計算 marginal gain、saturation 與 condition × depth interaction。 |
| Stage 6：Codec-native embedding  | 區分 codec 原生表徵資訊與下游任務重新學習出的資訊。                                                 |
| Stage 7：Acoustic baselines      | 比較 Log-Mel、MFCC、encoder latent 與 RVQ，定位資訊在哪個階段流失。                             |
| Stage 8：Codec adapter           | 統一 SpeechTokenizer、EnCodec、DAC 的程式介面，降低重複程式碼。                                 |
| Stage 9：Reconstruction fidelity | 比較重建品質與 linguistic／speaker／clinical-information trajectory 是否同步。              |

