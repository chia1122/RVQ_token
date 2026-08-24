# Pathology-aware RVQ Layer Fusion for Dysarthric ASR

## 1. Status and scope

This is the canonical research roadmap for the repository. It supersedes the
old plan in which RVQ information probing was treated as the endpoint. The old
documents remain available as research history and are mapped to this roadmap
in [the planning index](README.md).

The current layer-wise ASR trajectories and future probes serve two purposes:

1. diagnose what information is accessible at different RVQ layers; and
2. determine whether a pathology-aware fusion method is justified.

They are not the final proposed contribution. The main intended contribution
is an utterance-adaptive RVQ layer-fusion method for dysarthric ASR.

This roadmap does not claim that the planned models, probes, objectives, or
cross-codec experiments already exist. Status labels mean:

- **completed**: implemented and verified by the repository or a recorded
  completed experiment;
- **partially completed**: some required components or audits exist, but the
  full stage is not complete;
- **planned**: not yet implemented or not yet validated.

## 2. Research motivation

SpeechTokenizer Q1-only input and fixed fusion of all RVQ layers impose the
same representation policy on every utterance. That policy may not optimally
balance linguistic content and pathological acoustic variation in dysarthric
speech. Later RVQ layers may contain nuisance speaker/acoustic information,
useful complementary cues, or both. A higher CER at a deeper cumulative prefix
cannot by itself identify which explanation is correct.

The research objective is therefore to develop **Pathology-aware RVQ Layer
Fusion**: a model that derives layer weights from the utterance itself and
adaptively combines semantic-oriented and acoustic-oriented RVQ information.
The intended outcome is lower dysarthric-speech CER without a clear loss on
control speech.

Inference must not require a ground-truth dysarthria or severity label. Any
pathology-aware behavior must be inferred from acoustic/representation evidence
available at inference time.

## 3. Research questions

**RQ1.** Do individual RVQ layers and cumulative RVQ prefixes retain different
amounts of linguistic, speaker, and dysarthria-associated information?

**RQ2.** Do later RVQ layers provide complementary information for dysarthric
ASR, rather than Q1 being the only layer with useful linguistic information?

**RQ3.** Does static learned fusion outperform Q1-only, uniform sum, and
concatenation baselines under matched capacity and training conditions?

**RQ4.** Does utterance-adaptive pathology-aware fusion further reduce
dysarthric CER while maintaining control CER?

**RQ5.** Are improvements consistent across speaker-disjoint folds, multiple
speakers, and a second codec?

## 4. Representation audit and terminology

### 4.1 Stored tokens

Codec token payloads use `[T, N]`, where `T` is the frame axis and `N` is the
number of RVQ codebooks. Token IDs are categorical indices. They must never be
added, averaged, or interpreted as continuous numeric features.

### 4.2 Current Phase 1 representation

The current direct-token ASR path is:

```text
token ID at Qi
    -> trainable embedding table for Qi
    -> layer fusion
    -> temporal subsampling
    -> Transformer encoder
    -> CTC
```

For `--num-rvq-layers K`, `RVQTokenDataset` selects `codes[:, :K]`.
When `--active-rvq-layers` is omitted, `train_probe.py` activates every layer
from Q1 through QK. In `layer_fusion=sum`, `model.py` combines the learned
embeddings as:

```text
(E1 + E2 + ... + EK) / sqrt(K)
```

The formal Phase 1 depth trajectory is therefore:

```text
representation_mode = discrete_learned
rvq_mode = cumulative
fusion = sqrt_normalized_sum
```

It is not an individual-QK experiment and is not a codec-native cumulative
latent. The model's reported `normalized_layer_weights` for the fixed-sum case
are descriptive equal weights; the forward scaling is `1 / sqrt(K)`, not the
arithmetic mean `1 / K`.

### 4.3 Required condition names

Future experiment metadata must distinguish these conditions explicitly:

```text
individual_q1
individual_q2
...
individual_q8

cumulative_q1
cumulative_q1_2
...
cumulative_q1_8
```

For a general N-codebook codec, the names must be generated dynamically rather
than assuming N=8.

### 4.4 Current support boundaries

- An explicitly configured `--active-rvq-layers K` selects an individual
  learned-embedding layer when the input includes that layer.
- A reference-config-driven individual-layer sweep is implemented. It derives
  folds, seeds, optimizer, budget, capacity, and CER selection from the
  completed cumulative configs. Formal individual Q1–Q8 runs remain pending.
- `layer_fusion=learned` learns one global softmax weight per active layer. It
  is static learned fusion, not utterance-adaptive gating.
- Concatenation plus projection is not implemented.
- Frozen codec-native cumulative and individual representations are not
  implemented in the current ASR model.

## 5. Protocol status

### 5.1 Fixed-split pilot

The completed Phase 1 results used one fixed speaker-disjoint split. They are
retained as pilot and diagnostic evidence. They do not establish population-
level clinical effects, a pathology-information hierarchy, or absence of
linguistic information in later layers.

### 5.2 Seven-fold cyclic protocol

The versioned TORGO protocol includes 15 speakers: eight dysarthric and seven
control speakers. F04 and M03 retain their speaker-level `mild` labels and are
included in this versioned protocol. Seven predefined folds are rotated so that
the current fold is test, the next fold is validation, and the remaining five
folds are train.

Implemented components include:

- versioned speaker metadata and fold config;
- speaker-disjoint fold and manifest audit;
- prompt-overlap audit;
- rotation-specific token indexes sharing one master token store;
- collision-safe depth/seed output directories.

This is a predefined cyclic speaker-fold protocol. It is not a generic
GroupKFold, StratifiedGroupKFold, or LOSO implementation. Those mechanisms must
remain marked as planned unless separately implemented and audited.

The formal manifest audit reported 7,785 utterances, 15 speakers, no missing
audio, and valid fold coverage. The rotation-specific index audit reported
seven 7,785-row indexes with no missing token files or metadata mismatches.
The CER-selected depth trajectory and aggregation were externally audited as
168/168 valid runs across seven rotations, eight depths, and three seeds. The
aggregation contains 12,240 long-format rows and separate run-macro,
pooled-micro, and speaker-macro summaries. The repository does not contain the
full experiment artifacts.

The formal cumulative-prefix baseline found Q1 to be the best depth for all
15 speakers and all seven test rotations. No speaker or rotation obtained a
lower CER at Q1:Q8 than at Q1. Overall pooled-micro CER increased from 0.4912
at Q1 to 0.5866 at Q1:Q8; speaker-macro CER increased from 0.5347 to 0.6205.
Control and dysarthric speaker-macro CER both favored Q1:

| Depth | Control speaker-macro CER | Dysarthric speaker-macro CER |
|---:|---:|---:|
| Q1 | 0.4219 | 0.6334 |
| Q1:Q8 | 0.5317 | 0.6982 |

This establishes a consistent negative result for fixed cumulative
sqrt-normalized fusion. It does not establish that individual later layers
lack linguistic or complementary information.

The primary protocol estimates ASR generalization to unseen speakers under a
largely shared-prompt TORGO setting. It is not a prompt-disjoint generalization
protocol.

## 6. Revised Stage 0–9 roadmap

### Stage 0 — Representation and protocol audit

**Status: partially completed**

Confirm and record:

- `[T, N]` token axes and valid codebook depth;
- individual versus cumulative representations;
- task-trained embedding tables versus frozen codec-native embeddings;
- the exact fixed and learned fusion equations;
- pretraining and checkpoint provenance;
- validation-CER checkpoint selection;
- fixed-split versus cyclic speaker-fold protocols;
- parameter counts and active-layer definitions.

Completed evidence includes the current discrete-learned cumulative-prefix
definition, depth validation, CER checkpoint selection support, and cyclic fold
audit. Codec-native representation consistency and a complete pretraining
provenance audit remain planned.

**Exit criterion:** every baseline has an unambiguous representation name,
split identifier, checkpoint-selection rule, and trainable-parameter count.

### Stage 1 — Reliable linguistic baselines

**Status: partially completed**

Required matched baselines:

1. Q1 only;
2. every individual QK;
3. every cumulative Q1:QK prefix;
4. full-RVQ uniform/sqrt-normalized sum;
5. concatenation plus projection;
6. necessary acoustic baselines, initially Log-Mel and a frozen
   pre-quantization encoder representation if faithfully available.

The fixed-split pilot and seven-fold Q1/cumulative Q1:QK trajectories are
complete baselines. The model also contains fixed sqrt-normalized sum and
static learned weights, but no full matched fusion comparison has been
completed. Individual-layer sweep infrastructure is implemented, but its
formal 168-run matrix has not been executed. Concatenation and acoustic
baseline matrices remain planned.

**Fairness controls:** same folds, seeds, backbone, optimizer, training budget,
checkpoint-selection metric, and effective batch size. Parameter counts and
inference cost must be reported.

### Stage 2 — Speaker-disjoint evaluation protocol

**Status: completed for the current predefined cyclic protocol**

Use the current versioned 15-speaker distribution and seven cyclic folds. Every
depth and fusion method must reuse the same train/validation/test speakers.
Validation alone selects checkpoints; test data must not tune thresholds,
gating, hyperparameters, or stopping rules.

Completed components are the predefined seven-fold config, cyclic rotation
builder, leakage audit, manifest audit, rotation-specific token indexes, and
the 168-run CER-selected trajectory. Formal run-macro, pooled-micro,
speaker-macro, per-speaker, and per-rotation audits are complete. A generic
GroupKFold/LOSO implementation is not part of the completed protocol.

**Exit criterion:** all rotations pass leakage and coverage audits, and all
baseline methods can consume identical saved split assignments.

### Stage 3 — Complementarity diagnosis

**Status: planned**

Use lightweight diagnostic analyses to determine whether later layers contain
cross-speaker information that could improve dysarthric ASR:

- linguistic accessibility for individual and cumulative conditions;
- condition-associated information under speaker-disjoint evaluation;
- speaker information under a separate, task-appropriate split;
- phoneme- or error-category improvements by layer;
- descriptive severity-associated information where statistically defensible;
- frozen codec-native analysis only when native vectors are faithfully
  recoverable.

Speaker identity and clinical tasks must not automatically share a split.
Clinical probes use speaker-disjoint evaluation; speaker identity requires its
own session/utterance-disjoint protocol for known identities. Clinical
confidence intervals must not treat utterances from one speaker as independent
subjects.

**Decision gate:** proceed to pathology-aware adaptive fusion only if later
layers show cross-speaker complementary information, or if a fixed multilayer
method improves identifiable dysarthric speakers/phonemes under matched
evaluation. If later layers expose only speaker identity without transferable
dysarthria-associated or linguistic utility, re-examine the representation and
research hypothesis before implementing adaptive gating.

**Current gate status: not passed.** The completed cumulative-prefix baseline
shows that fixed sqrt-normalized addition degrades CER consistently across all
speakers and rotations. Individual-layer and alternative fixed-fusion
experiments are still required to distinguish absent complementarity from
destructive fixed fusion.

### Stage 4 — Fixed fusion baselines

**Status: partially implemented, formal comparison planned**

Compare:

1. Q1 only;
2. full-RVQ uniform sum;
3. full-RVQ mean or sqrt-normalized sum, with the exact equation recorded;
4. concatenation plus projection;
5. static learned weighted sum.

Current code supports Q1, cumulative sqrt-normalized sum, active-layer masks,
and static global softmax weights. It does not implement concatenation plus
projection, and existing ad hoc runs do not constitute the required matched
speaker-fold comparison.

**Exit criterion:** determine the strongest fixed baseline without changing
capacity, folds, seeds, or budget after viewing test results.

### Stage 5 — Utterance-adaptive RVQ fusion

**Status: planned**

Implement a small gating network that produces one layer-weight vector per
utterance. The first version must use utterance-level gating rather than
frame-level gating. The gate may consume pooled codec/encoder evidence but may
not require a ground-truth condition or severity label at inference.

Required outputs include per-utterance weights, average active codebooks,
weight entropy or sparsity, trainable parameter count, and inference cost.

**Exit criterion:** adaptive gating outperforms the strongest matched fixed
fusion baseline on dysarthric CER without a clear control-CER penalty.

### Stage 6 — Pathology-aware training objectives

**Status: planned**

Compare, without assuming the winner in advance:

1. CTC only;
2. CTC plus a dysarthria-associated auxiliary objective;
3. CTC plus a pathology-invariant or matched-prompt alignment objective;
4. CTC plus sparse/top-k layer selection.

Auxiliary condition prediction is a representation-learning objective, not a
clinical diagnosis. Speaker-level severity must not be treated as an
utterance-level clinical score. Ablations must separate the effects of the
gate, auxiliary loss, sparsity, and increased parameter count.

### Stage 7 — Formal evaluation and ablation

**Status: planned**

Under identical folds, seeds, backbone, optimizer, training budget, and
checkpoint-selection rules, compare:

- Q1 only;
- uniform/sqrt-normalized sum;
- concatenation plus projection;
- static learned weighting;
- adaptive gating;
- adaptive gating without pathology objective;
- adaptive gating without sparsity constraint;
- the complete proposed model.

**Primary metric:** dysarthric CER.

**Secondary metrics:** overall CER, control CER, WER, substitutions, deletions,
insertions, blank-frame ratio, empty-hypothesis ratio, average active
codebooks, parameter count, and inference cost.

Report utterance-micro and speaker-macro metrics separately. Summaries must
retain fold, seed, speaker, condition, and severity identifiers. WER/CER are
ASR performance measures and must not be described as clinical intelligibility.

### Stage 8 — Cross-codec generalization

**Status: planned**

First establish the method with SpeechTokenizer. Then replicate the frozen
design decisions with EnCodec. Equal RVQ depths across codecs do not imply
equal bitrates, frame rates, dimensions, or information rates; those properties
must be reported.

Only the minimum adapter work required for a faithful EnCodec replication is
in scope. Do not broadly refactor all codec adapters before the fusion method
has passed its SpeechTokenizer decision gate.

### Stage 9 — Extensions

**Status: planned / exploratory**

Secondary work includes:

- severity-associated probing;
- reconstruction fidelity;
- bitrate–CER trade-offs;
- speaker privacy;
- codec-native embedding analysis;
- DAC or Mimi comparisons;
- carefully defined clinical-information preservation analyses.

These extensions must not displace the primary fusion comparison. Mimi is not
part of the first cross-codec replication.

## 7. Experiment matrix

| Experiment family | Representation | Fusion | Split | Current status |
|---|---|---|---|---|
| Phase 1 pilot | cumulative discrete-learned Q1:QK | sqrt-normalized sum | one fixed speaker-disjoint split | completed |
| Seven-rotation trajectory | cumulative discrete-learned Q1:QK | sqrt-normalized sum | seven cyclic speaker folds | completed and aggregated |
| Individual diagnostic | individual discrete-learned QK | one active layer | same saved speaker folds | implemented; formal runs pending |
| Fixed fusion | full discrete-learned RVQ | uniform, mean/sqrt-normalized, concat, static learned | same saved speaker folds | partially implemented; comparison planned |
| Adaptive fusion | full discrete-learned RVQ | utterance-adaptive gating | same saved speaker folds | planned |
| Objective ablation | adaptive representation | CTC and auxiliary/sparse variants | same saved speaker folds | planned |
| Cross-codec replication | codec-specific faithful representation | selected fusion | same protocol where feasible | planned |
| Codec-native analysis | frozen native vectors | codec-defined individual/cumulative | diagnostic protocol | planned/exploratory |

Commands for concatenation, adaptive gating, auxiliary objectives, and
cross-codec fusion are intentionally omitted: those features are not currently
implemented and must not be presented as executable. The individual sweep has
an executable reference-config-driven CLI documented in the repository README.

## 8. Success criteria

The principal method must satisfy all of the following before being presented
as successful:

1. lower dysarthric CER than Q1-only;
2. lower dysarthric CER than the strongest static fusion baseline;
3. no clear degradation of control CER under a predefined tolerance;
4. improvement across multiple speaker-disjoint folds;
5. improvement not driven by one test speaker;
6. improvement not explained solely by additional trainable parameters;
7. analyzable layer weights or active-codebook behavior;
8. at least one successful second-codec replication.

Before formal evaluation, the control-CER non-inferiority tolerance and the
fold-level consistency rule must be specified using validation/development
evidence, not selected after inspecting final test results.

## 9. Stop and rollback conditions

Stop adaptive-fusion development and return to a simpler fixed method, or
revisit the representation hypothesis, if any of the following occurs:

- later layers show no cross-speaker complementary information;
- adaptive fusion does not outperform concatenation or static weighting;
- gains occur only for one speaker or one split;
- gains disappear under parameter-matched controls;
- the gate primarily encodes speaker identity without transferable ASR value;
- the method requires a ground-truth pathology label at inference;
- a second-codec replication fails after codec-specific dimensional and
  bitrate differences are controlled.

Rollback is a valid research result. Fixed fusion should become the primary
method if it is the strongest reliable model.

## 10. Interpretation boundaries

The project must not claim that:

- WER or CER equals clinical intelligibility;
- dysarthria classification is clinical diagnosis;
- a speaker-level severity label is an utterance-level clinical score;
- current Q1–Q8 results prove a pathology-information hierarchy;
- higher CER for a deeper prefix proves that later layers contain no
  linguistic information;
- statistical significance alone establishes clinical importance.

The defensible target is improved ASR performance for dysarthric speech under
speaker-disjoint evaluation, accompanied by transparent control-speech,
speaker-level, capacity, and efficiency analyses.

## 11. Immediate next steps

1. Preserve and freeze the audited folds, seeds, capacity, budget, selection
   protocol, and aggregation outputs.
2. Complete Stage 0 representation/provenance tables.
3. Dry-run, smoke-test, and execute the implemented matched individual-layer
   sweep without changing the saved speaker folds.
4. Aggregate and pair individual QK with cumulative Q1:QK results.
5. Add concatenation plus projection and run the complete fixed-fusion matrix.
6. Apply the Stage 3 decision gate before implementing utterance-adaptive
   pathology-aware fusion.

No new probes, models, formal training commands, or experimental results are
created by this planning document.
