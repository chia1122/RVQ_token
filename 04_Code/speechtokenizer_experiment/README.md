# TORGO SpeechTokenizer workflow

This workflow uses the frozen local `speechtokenizer_hubert_avg` checkpoint.
The canonical TORGO source is the 15-speaker, seven-rotation protocol described
in `../torgo_manifest/README.md`.

## Canonical artifacts

Generated manifests, tokens, token indexes, and runs remain outside Git:

```bash
export RVQ_ARTIFACT_ROOT="${RVQ_ARTIFACT_ROOT:-$HOME/rvq_token_artifacts}"
export MASTER_TOKEN_ROOT="$RVQ_ARTIFACT_ROOT/tokens/speechtokenizer_hubert_avg/torgo_including_mild_v1_master"
export ROTATION_INDEX_ROOT="$RVQ_ARTIFACT_ROOT/token_indices/speechtokenizer_hubert_avg/torgo_including_mild_v1_v1"
```

The validated master store contains 7,785 utterances, eight codebooks of size
1,024, and 50 Hz token sequences in `[T,8]` order. Rotation indexes share the
same `.pt` files and change only split metadata.

## Install and verify SpeechTokenizer

```bash
python -m pip install -e /home/rachel/06_opensource_toolkit/SpeechTokenizer
python -c "from speechtokenizer import SpeechTokenizer; print(ok)"
```

The current checkpoint files are:

```text
/home/rachel/06_opensource_toolkit/SpeechTokenizer/model_hub/speechtokenizer_hubert_avg/config.json
/home/rachel/06_opensource_toolkit/SpeechTokenizer/model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt
```

## Validate existing tokens and rotations

```bash
python -m json.tool "$MASTER_TOKEN_ROOT/extraction_summary.json"
python -m json.tool "$ROTATION_INDEX_ROOT/rotation_index_audit.json"

for index in "$ROTATION_INDEX_ROOT"/rotation_*/tokens.jsonl; do
  printf %st "$(basename "$(dirname "$index")")"
  wc -l < "$index"
done
```

Formal experiments must use all seven rotations. A single rotation may be used
only for development or smoke testing.

## Scope of the current milestone

The active milestone probes each frozen codec-native codebook independently:

```text
Q1, Q2, ..., Q8
```

Do not use cumulative Q1:QK inputs, learned layer fusion, adaptive fusion, or
codec fine-tuning in this milestone. Historical direct-token CTC commands and
results remain in `rvq_asr/`, but their learned token embeddings are not the
frozen codec-native representations required by the new probes.
