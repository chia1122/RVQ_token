# P1: phoneme CTC probe

Research question: how much phoneme identity is recoverable from one individual RVQ codebook?

Input is `[T,1024]` from exactly one frozen native codebook. Targets are ARPAbet phones: the first CMUdict pronunciation is selected, stress digits are removed, silence is excluded, and OOV words use `g2p_en`; preprocessing records source and pronunciation alternatives. The model is LayerNorm → bottleneck → shallow temporal convolutions → phoneme/blank classifier. It uses CTC loss and reports PER, substitutions, deletions, insertions, reference count, blank-frame ratio, and empty-prediction ratio.

Speaker-exclusive rotations are reused unchanged. Limitations include deterministic pronunciation choice, transcript/G2P errors, CTC optimization, and the deliberately limited probe capacity.
