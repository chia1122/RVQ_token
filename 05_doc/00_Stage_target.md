# Revised Stage Overview

> **Status: superseded as an execution roadmap.** This file preserves the
> mapping from the original Stage 1–9 plan to the current research direction.
> The canonical plan is
> [Pathology-aware RVQ Layer Fusion for Dysarthric ASR](PATHOLOGY_AWARE_RVQ_FUSION_ROADMAP.md).

The original plan treated RVQ information probing as the endpoint. The revised
plan treats probing as diagnosis and a decision gate for a pathology-aware
fusion method.

## Current Stage 0–9 roadmap

| Current stage | Purpose | Status |
|---|---|---|
| Stage 0: Representation and protocol audit | Fix individual/cumulative terminology, embedding provenance, fusion equations, checkpoint selection, and split protocol. | partially completed |
| Stage 1: Reliable linguistic baselines | Establish Q1, individual layers, cumulative prefixes, fixed full-RVQ fusion, concatenation, and acoustic baselines. | partially completed |
| Stage 2: Speaker-disjoint evaluation | Use and complete the audited seven-fold cyclic protocol; do not call it GroupKFold/LOSO. | partially completed |
| Stage 3: Complementarity diagnosis | Test whether later layers provide cross-speaker linguistic or dysarthric-ASR utility. | planned |
| Stage 4: Fixed fusion baselines | Compare Q1, fixed sum/mean normalization, concatenation, and static learned weights. | partially implemented; comparison planned |
| Stage 5: Utterance-adaptive fusion | Produce utterance-specific RVQ layer weights without requiring true pathology labels at inference. | planned |
| Stage 6: Pathology-aware objectives | Compare CTC, auxiliary, invariant/alignment, and sparse-selection objectives. | planned |
| Stage 7: Formal evaluation | Run matched ablations with dysarthric CER as the primary ASR metric. | planned |
| Stage 8: Cross-codec generalization | Validate SpeechTokenizer first, then perform one EnCodec replication. | planned |
| Stage 9: Extensions | Keep severity, reconstruction, bitrate, privacy, native embeddings, and additional codecs secondary. | planned/exploratory |

## Old-to-new Stage mapping

| Original stage | Revised location |
|---|---|
| Stage 1: Complete RVQ trajectory | Stage 0 representation audit, Stage 1 pilot baselines, and Stage 2 protocol foundation |
| Stage 2: Speaker probe | Stage 3 complementarity diagnosis; possible Stage 9 speaker-privacy extension |
| Stage 3: Dysarthria detection | Stage 3 diagnosis; possible Stage 6 auxiliary objective |
| Stage 4: Severity probe | Stage 3 diagnostic or Stage 9 exploratory work |
| Stage 5: Statistical analysis | Stage 7 formal evaluation |
| Stage 6: Codec-native embedding | Stage 0 audit, Stage 3 diagnosis, or Stage 9 extension |
| Stage 7: Acoustic baselines | Stage 1 reliable baselines |
| Stage 8: Codec adapter | Minimal Stage 8 support; broad refactor deferred |
| Stage 9: Reconstruction fidelity | Stage 9 exploratory extension |

Historical Stage files remain in this directory and are marked
`revised/superseded`. Their original requirements and results are not deleted.

