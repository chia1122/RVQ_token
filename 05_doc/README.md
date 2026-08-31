# Research Planning Index

This directory records both the current research roadmap and the historical
development prompts that led to it. The current research direction is:

> **Pathology-aware RVQ Layer Fusion for Dysarthric ASR**

The layer-wise trajectory and probing work is diagnostic evidence and method
motivation. It is not the final research contribution.

## Current documents

| Document | Status | Purpose |
|---|---|---|
| [Pathology-aware RVQ fusion roadmap](PATHOLOGY_AWARE_RVQ_FUSION_ROADMAP.md) | canonical | Current research questions, Stage 0–9 roadmap, experiment matrix, decision gates, and success criteria. |
| [Phase 1 conclusion](01_conclution.md) | completed trajectory record | Preserves the fixed-split pilot and formal seven-fold cumulative/individual SpeechTokenizer results. |
| [Dataset protocol amendment](01_dataset_modify.md) | partially completed protocol | Records mild-speaker inclusion, seven cyclic speaker folds, audit status, and remaining evaluation work. |
| [Stage overview](00_Stage_target.md) | revised index | Maps the superseded Stage 1–9 plan to the current Stage 0–9 roadmap. |
| [Manifest/Phase 1 specification](01_Manifest_pre.md) | historical, completed | Original Phase 1 implementation request. |
| [Reflection placeholder](01_reflect.md) | reserved | Intentionally empty; no duplicate roadmap content is maintained here. |

## Superseded stage prompts

The following files are retained for research history. Their old Stage numbers
must not be interpreted as the current execution order.

| Historical document | Current mapping |
|---|---|
| [Speaker identity probe](02_Speaker_identity_probe.md) | Stage 3 diagnostic; possible Stage 9 privacy extension |
| [Dysarthria detection probe](03_Dysarthria_detection_probe.md) | Stage 3 diagnostic; possible Stage 6 auxiliary objective |
| [Severity probe](04_Severity_probe.md) | Stage 3 diagnostic or Stage 9 exploratory analysis |
| [Seed trajectory/statistics](05_seed_trajectory.md) | Stage 7 formal evaluation |
| [Codec-native embedding probe](06_Codec_native_embedding_probe.md) | Stage 0 audit, Stage 3 diagnosis, or Stage 9 extension |
| [Acoustic baseline](07_Acoustic_baseline.md) | Stage 1 reliable baselines |
| [Codec adapter](08_Codec_adapter.md) | Minimal Stage 8 support; broad refactor deferred |
| [Reconstruction fidelity](09_Reconstruction_fidelity.md) | Stage 1 controlled reconstruction-ASR baseline; broader fidelity work remains Stage 9 |

## Current status snapshot

As of 2026-08-31:

- Phase 1 fixed-split SpeechTokenizer cumulative-prefix trajectories are
  complete for WER- and CER-selected checkpoints and remain pilot evidence.
- The current direct-token representation is a task-trained discrete embedding
  representation, not a frozen codec-native representation.
- The versioned 15-speaker, seven-fold cyclic protocol and rotation-specific
  token-index tooling are implemented and audited.
- The cyclic protocol is speaker-disjoint, but it is not a generic GroupKFold,
  StratifiedGroupKFold, or LOSO implementation.
- The CER-selected seven-rotation trajectory was externally audited as
  168/168 valid runs: seven rotations, eight depths, and three seeds.
- Cross-rotation aggregation is complete with audited run-macro, pooled-micro,
  speaker-macro, per-speaker, and per-rotation outputs.
- The matched individual Q1–Q8 matrix completed 168/168 valid runs. Paired
  analysis found higher Q2–Q8 individual CER for all 15 speakers and a
  high-blank, deletion-dominated under-generation pattern.
- Cumulative Q1:Q8 reconstruction and original audio were evaluated by one
  frozen faster-whisper configuration. The formal audit contains 70,065 valid
  predictions with no failures or pairing mismatches.
- Reconstruction K8 improved over K1 for all 15 speakers, whereas the learned
  direct-token cumulative trajectory favored Q1 for all 15 speakers. K8 still
  remained worse than original audio for 14 speakers.
- Non-native individual Q2–Q8 reconstruction remains excluded. Concatenation
  baselines, utterance-adaptive gating, pathology-aware objectives, and
  cross-codec replication remain pending unless explicitly stated otherwise
  in the canonical roadmap.

WER and CER are ASR performance metrics. They are not clinical intelligibility
scores. Dysarthria- or severity-associated probes are not clinical diagnostic
systems, and speaker-level severity labels are not utterance-level clinical
scores.
