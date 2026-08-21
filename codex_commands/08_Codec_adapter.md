請完成第八階段：
將 SpeechTokenizer、EnCodec、DAC 的重複 extraction、
native representation 和 reconstruction 邏輯重構成統一 codec adapter。

重要：
- 這是內部重構，不改變現有研究結果。
- 所有重構前後的輸出必須接受 regression test。
- 保留舊 CLI 或提供 backward-compatible wrapper。

A. Adapter interface
建立：

class CodecAdapter:
    encode(...)
    get_codes(...)
    get_native_embeddings(...)
    build_cumulative_representation(...)
    build_individual_representation(...)
    decode_prefix(...)
    get_metadata(...)

B. Registry
支援：
--codec speechtokenizer
--codec encodec
--codec dac

checkpoint、sample rate、bandwidth 等由 config 提供，
不要在共用程式中寫死。

C. Unified schemas
統一：
- manifest
- token index
- representation index
- reconstruction index
- experiment metadata

加入 schema version，並提供舊格式 migration/validation。

D. Configuration
新增 YAML configuration：
- dataset
- codec
- representation
- probe
- sweep
- evaluation

CLI arguments 可覆蓋 YAML，但最終 resolved config 必須保存。

E. Backward compatibility
- 舊 extraction script 可變成 wrapper。
- 舊 command 不應無預警失效。
- 若無法相容，提供 migration guide 和清楚錯誤訊息。

F. Regression tests
選定 synthetic/small fixture，驗證重構前後：
- token IDs 相同；
- codebook shape 相同；
- native representation 數值一致；
- reconstructed waveform 在容許誤差內一致；
- metadata 一致；
- downstream probe 可以讀取。

G. Packaging
補齊：
- pyproject.toml
- dependency groups
- installation instructions
- module entry points
- version information

限制：
- 不加入新 codec。
- 不修改資料切分。
- 不改變 probe 架構。
- 不重新定義任何研究 metric。
- 不執行完整訓練。

完成後提供：
1. 新舊架構對照；
2. 修改/移動檔案；
3. backward compatibility 方法；
4. regression test 結果；
5. migration guide；
6. 新 CLI 使用範例；
7. 仍保留的 technical debt。
