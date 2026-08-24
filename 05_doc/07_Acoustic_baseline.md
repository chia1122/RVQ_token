> **Roadmap status: revised.** Acoustic、pre-quantization 與 full-RVQ controls 現列
> 新版 roadmap 的 Stage 1 baseline family。本文件保留原始規格，尚不表示所有
> baselines 已實作或完成。Canonical 定義見
> [Pathology-aware RVQ Layer Fusion roadmap](PATHOLOGY_AWARE_RVQ_FUSION_ROADMAP.md)。

請完成第七階段：
新增 acoustic features、pre-quantization encoder latent 和 full-RVQ baselines。

研究目的：
定位 linguistic／speaker／dysarthria／severity information
是在 codec encoder、quantization early stages 或 late stages 流失。

A. Baselines
新增：
1. Log-Mel
2. MFCC
3. Pre-quantization encoder latent
4. Full RVQ
5. Original/reconstructed audio ASR result reference

B. Acoustic preprocessing
固定並記錄：
- sample rate
- window length
- hop length
- FFT size
- mel bins
- normalization
- padding
- duration handling

跨 codec 比較時，不能因 codec sample rate 不同而靜默改變 baseline 定義。

C. Encoder latent
針對每個 codec：
- 定位 quantizer 前的 encoder representation；
- 完全 frozen；
- 記錄 frame rate 和 feature dimension；
- 不可誤用 decoder latent或 quantized latent；
- 輸出統一 index 和 metadata。

D. Probe fairness
不同 representation 可以先投影到相同 model_dim，
但 probe architecture、training budget、seed、split 必須固定。

同時記錄：
- 原始 input dimension
- projected dimension
- trainable parameter count
- frame rate

E. Comparison table
產生：
Log-Mel
MFCC
Encoder latent
Q1
Q1:2
...
Q1:N

比較每一個 task 和 codec。

F. 測試
至少檢查：
1. acoustic frame length正確；
2. padding不影響 pooling；
3. encoder在quantizer之前抽取；
4. codec完全frozen；
5. metadata完整；
6. 相同split被重用；
7. trainable parameter count被記錄；
8. baseline輸出可被共用probe讀取。

限制：
- acoustic target 只能稱為 baseline，不是臨床 ground truth。
- 不在本階段加入 articulation/prosody/phonation rating。
- 不執行完整大規模訓練。

完成後提供：
1. baseline 定義；
2. 每個 codec 的 encoder hook 位置；
3. 修改檔案；
4. fairness 控制；
5. 測試結果；
6. 執行命令；
7. 跨 representation 比較限制。
