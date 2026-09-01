# P3: dysarthria probe

Research question: how much control-versus-dysarthric information is recoverable from one individual RVQ codebook?

One frozen `[T,1024]` codebook sequence is masked-mean pooled, LayerNorm-normalized, and passed directly to a two-class linear classifier with cross-entropy loss. The primary metric is Macro-F1; outputs also include UAR/balanced accuracy, per-class precision/recall/F1, and a confusion matrix.

Evaluation is speaker-exclusive. Good performance does not establish pathology-specific information independent of speaker identity, recording conditions, text, or other correlated factors. Severity prediction is intentionally deferred.
