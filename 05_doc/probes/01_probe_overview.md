# Individual-RVQ probe protocol

This milestone measures information readily recoverable from each frozen SpeechTokenizer codebook Q1–Q8. It does not use cumulative representations, waveform decoding, codec fine-tuning, or adaptive fusion.

SpeechTokenizer uses 16 kHz audio, hop length 320, eight 1024-entry codebooks, native 1024-dimensional codebook vectors, and a 50 Hz (20 ms) representation rate. All probes share the codec-native frozen codebook loader and the seven speaker-exclusive TORGO rotations. Rotation 01 is the development split; formal reports aggregate all seven rotations as mean ± sample standard deviation while retaining per-rotation results.

Every run writes `config.json`, `results.json`, `training_history.json`, `best.pt`, and where applicable `predictions.jsonl`. A weak score means the target is not readily recoverable by this probe under this setup, not that the representation contains no such information. Dysarthria scores may still reflect speaker-correlated information despite speaker-exclusive evaluation.
