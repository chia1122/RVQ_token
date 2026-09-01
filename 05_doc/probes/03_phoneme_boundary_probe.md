# P2: phoneme boundary probe

Research question: how much local phonetic-boundary information is recoverable from one individual RVQ codebook?

The official MFA `english_us_arpa` GMM-HMM acoustic model and dictionary produce phone intervals. The experiment dictionary contains only the first listed pronunciation for deterministic alignment; the audit records all available alternatives and OOV G2P use. Full intervals remain in target artifacts. Training labels are internal junctions between adjacent non-silence phones; utterance edges and gaps/silence transitions are excluded.

Seconds map to the 20 ms codec grid using `floor(time / 0.02 + 0.5)`, clamped to `[0,T-1]`. Collisions yield one positive frame and are counted. The model is LayerNorm → projection → one width-3 convolution → linear logit. BCEWithLogitsLoss uses positive weight computed from training frames only.

Evaluation reports exact-frame and ±1-frame precision, recall, and F1 plus prevalence and TP/FP/FN. Matching is one-to-one and prioritizes smallest frame distance. Formal results cover all seven speaker-exclusive rotations and report each rotation plus mean ± sample standard deviation. Alignment errors and the chosen silence-boundary definition remain limitations.
