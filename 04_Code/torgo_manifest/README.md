# TORGO seven-rotation protocol

This directory builds and validates speaker-disjoint TORGO manifests. The
canonical protocol is `torgo_including_mild_v1`: 15 speakers (eight
dysarthric and seven control) assigned to seven folds. For each rotation, the
current fold is test, the next fold is validation, and the remaining five
folds are train.

The only canonical protocol sources are:

```text
config/speaker_metadata.csv
config/speaker_folds.json
```

The earlier 13-speaker fixed split and its generated manifests were removed.
Historical pilot results may remain in research records, but must not be used
for new probe comparisons.

## Validate and generate all rotations

Run from the repository root:

```bash
export PYTHONDONTWRITEBYTECODE=1
export RVQ_ARTIFACT_ROOT="${RVQ_ARTIFACT_ROOT:-$HOME/rvq_token_artifacts}"
export PROTOCOL_ROOT="$RVQ_ARTIFACT_ROOT/protocols/torgo_including_mild_v1"

test ! -e "$PROTOCOL_ROOT"
python 04_Code/torgo_manifest/audit_speaker_folds.py \
  --speaker-metadata 04_Code/torgo_manifest/config/speaker_metadata.csv \
  --fold-config 04_Code/torgo_manifest/config/speaker_folds.json \
  --output-dir "$PROTOCOL_ROOT"
```

The audit writes `fold_audit.json` and seven
`generated_splits/rotation_*.json` files. It rejects duplicate, missing, or
overlapping speakers. Each generated split is accepted by
`build_torgo_manifest.py`.

## Build a rotation manifest

Use the same metadata and one generated split. Example for rotation 01:

```bash
export ROTATION_ID=rotation_01_test_a
export MANIFEST_ROOT="$RVQ_ARTIFACT_ROOT/manifests/torgo_including_mild_v1/$ROTATION_ID"

python 04_Code/torgo_manifest/build_torgo_manifest.py \
  --index TORGO_Transcript/torgo.csv \
  --audio-root /data/TORGO \
  --speaker-metadata 04_Code/torgo_manifest/config/speaker_metadata.csv \
  --split-config "$PROTOCOL_ROOT/generated_splits/$ROTATION_ID.json" \
  --output-dir "$MANIFEST_ROOT"
```

The builder selects `headMic` by default, rejects unusable transcripts,
checks unique utterance/audio IDs, and independently asserts speaker
disjointness. Outputs include `torgo_all.jsonl`, one JSONL per split,
`excluded_samples.csv`, `dataset_statistics.csv`, and `build_audit.json`.

Run manifest tests with:

```bash
python -m unittest discover -s 04_Code/torgo_manifest -p "test_*.py"
```

## Build rotation-specific token indexes

After extracting one validated master token store, build lightweight indexes
for every generated speaker rotation. The indexes share the same `.pt` token
files: they preserve each master-relative `token_path` and change only the
index row's `split` according to the rotation config. Do not copy the master
token files seven times.

The token payload may retain the split recorded during extraction. For RVQ ASR,
the rotation-specific `tokens.jsonl` is authoritative for split, speaker,
condition, severity, and transcript metadata; the shared payload supplies the
codec codes.

Linux Bash example from `04_Code/`:

```bash
export PYTHONDONTWRITEBYTECODE=1
export RVQ_ARTIFACT_ROOT="$HOME/rvq_token_artifacts"

MasterTokenRoot="$RVQ_ARTIFACT_ROOT/tokens/speechtokenizer_hubert_avg/torgo_including_mild_v1_master"
GeneratedSplitsRoot="$RVQ_ARTIFACT_ROOT/protocols/torgo_including_mild_v1/generated_splits"
RotationIndexRoot="$RVQ_ARTIFACT_ROOT/token_indices/speechtokenizer_hubert_avg/torgo_including_mild_v1"

test -f "$MasterTokenRoot/tokens.jsonl"
test -d "$GeneratedSplitsRoot"
test ! -e "$RotationIndexRoot"

python torgo_manifest/build_rotation_token_indices.py \
  --master-token-index "$MasterTokenRoot/tokens.jsonl" \
  --master-token-root "$MasterTokenRoot" \
  --generated-splits-dir "$GeneratedSplitsRoot" \
  --output-dir "$RotationIndexRoot"
```

The builder dynamically discovers `rotation_*.json`, validates full and
disjoint speaker assignment, verifies every shared token path, and refuses to
overwrite a non-empty output directory. It writes one `tokens.jsonl` per
rotation plus `rotation_index_audit.json`. These generated indexes and audits
are experiment artifacts and must remain outside Git.

Inspect the result without loading model checkpoints or audio:

```bash
python -m json.tool "$RotationIndexRoot/rotation_index_audit.json"

for index in "$RotationIndexRoot"/rotation_*/tokens.jsonl; do
  printf '%s\t' "$(basename "$(dirname "$index")")"
  wc -l < "$index"
done
```

Use a rotation index with the unchanged master token root:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m rvq_asr.sweep_depths \
  --token-index "$RotationIndexRoot/rotation_01_test_a/tokens.jsonl" \
  --token-root "$MasterTokenRoot" \
  --output-root "$RVQ_ARTIFACT_ROOT/trajectories/rotation_01_test_a" \
  --codec speechtokenizer \
  --depths auto \
  --seeds 1337,2026,3407 \
  --dry-run \
  -- \
  --epochs 30 \
  --batch-size 8 \
  --model-dim 256 \
  --encoder-layers 4 \
  --heads 4 \
  --feedforward-dim 1024 \
  --learning-rate 3e-4 \
  --weight-decay 1e-2 \
  --time-reduction 2 \
  --subsampling conv \
  --device cuda
```

Keep a separate output root for every rotation. Complete seven-rotation
training is a later experiment; index generation itself does not run training.

## Extract Meta EnCodec RVQ tokens

Only use the final `output/torgo_all.jsonl`; do not extract from a draft whose
audio entries are marked missing. On the Linux server, activate the environment
that contains matching PyTorch/torchaudio builds and install Meta EnCodec if it
is not already present:

```bash
python -c "import torch, torchaudio, encodec; print(torch.__version__, torch.cuda.is_available())"
```

First perform a one-utterance smoke test:

```bash
head -n 1 "$MANIFEST_ROOT/torgo_all.jsonl" \
  > /tmp/torgo_smoke.jsonl

python 04_Code/torgo_manifest/extract_encodec_tokens.py \
  --manifest /tmp/torgo_smoke.jsonl \
  --audio-root /data/TORGO \
  --output-dir 04_Code/torgo_manifest/encodec_tokens_smoke \
  --model encodec_24khz \
  --bandwidth 6.0 \
  --device cuda
```

Inspect the result:

```bash
cat 04_Code/torgo_manifest/encodec_tokens_smoke/extraction_summary.json

python - <<'PY'
import torch
from pathlib import Path

path = next(Path("04_Code/torgo_manifest/encodec_tokens_smoke").glob("*/*/*.pt"))
item = torch.load(path, map_location="cpu", weights_only=False)
print(path)
print("codes shape [T, N]:", tuple(item["codes"].shape))
print("token range:", int(item["codes"].min()), int(item["codes"].max()))
print("metadata:", item["speaker_id"], item["severity"], item["split"])
PY
```

For the 24 kHz model at 6 kbps, the expected `N` is normally 8. The extractor
records and validates the actual model output rather than assuming this value.

After the smoke test succeeds, run the full extraction:

```bash
python 04_Code/torgo_manifest/extract_encodec_tokens.py \
  --manifest "$MANIFEST_ROOT/torgo_all.jsonl" \
  --audio-root /data/TORGO \
  --output-dir 04_Code/torgo_manifest/encodec_tokens_24khz_6kbps \
  --model encodec_24khz \
  --bandwidth 6.0 \
  --device cuda
```

Existing valid `.pt` files are reused, so the same command resumes an
interrupted run. Use `--overwrite` only when deliberately regenerating all
tokens. By default extraction stops at the first bad item; `--skip-errors`
continues and records failures in `failures.jsonl`.

Each `.pt` file contains integer `codes` in `[T, N]` order and the associated
speaker, severity, split, transcript, sample-rate, model, and bandwidth
metadata. `tokens.jsonl` is the index for the downstream ASR dataset, while
`extraction_summary.json` records the common codebook dimensions.

## Extract DAC RVQ tokens

DAC uses the same final manifest and writes the same `[T, N]` token interface,
but it must use a separate output directory. Confirm the server environment:

```bash
python -c "import torch, torchaudio, dac; print(torch.__version__, torch.cuda.is_available())"
```

The import name is `dac`; the PyPI package name is commonly
`descript-audio-codec`. Run a one-utterance 24 kHz smoke test first:

The extractor converts every source waveform to the selected DAC checkpoint's
sample rate before calling DAC preprocessing. Both the original and codec
sample rates remain recorded in each token file.

```bash
python 04_Code/torgo_manifest/extract_dac_tokens.py \
  --manifest /tmp/torgo_smoke.jsonl \
  --audio-root /data/TORGO \
  --output-dir 04_Code/torgo_manifest/dac_tokens_smoke \
  --model 24khz \
  --device cuda
```

The first run downloads the official DAC checkpoint. Inspect the actual RVQ
dimensions and frame rate reported by that checkpoint:

```bash
cat 04_Code/torgo_manifest/dac_tokens_smoke/extraction_summary.json
```

After the smoke test succeeds, run the full extraction:

```bash
python 04_Code/torgo_manifest/extract_dac_tokens.py \
  --manifest "$MANIFEST_ROOT/torgo_all.jsonl" \
  --audio-root /data/TORGO \
  --output-dir 04_Code/torgo_manifest/dac_tokens_24khz \
  --model 24khz \
  --device cuda
```

Available official model selectors are `16khz`, `24khz`, and `44khz`. Do not
mix their outputs. For cross-codec experiments, record the sample rate, token
frame rate, number of codebooks, codebook size, and effective bitrate from each
checkpoint; equal RVQ layer counts alone do not imply equal information rates.

## Train the shared Transformer CTC probe

The probe implementation is in `04_Code/rvq_asr/`. It reads the common
`tokens.jsonl` format, so switching codecs changes only `--token-index` and
`--token-root`. Run it as a Python module from the repository root by exposing
`04_Code` on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/04_Code${PYTHONPATH:+:$PYTHONPATH}"
```

First verify one EnCodec batch with four RVQ layers:

```bash
python - <<'PY'
from pathlib import Path
from torch.utils.data import DataLoader
from rvq_asr.data import CTCBatchCollator, RVQTokenDataset
from rvq_asr.text import CharacterTokenizer

root = Path("04_Code/torgo_manifest/encodec_tokens_24khz_6kbps")
dataset = RVQTokenDataset(root / "tokens.jsonl", root, "train", 4, CharacterTokenizer())
loader = DataLoader(dataset, batch_size=2, collate_fn=CTCBatchCollator(1024))
batch = next(iter(loader))
print("codes [B,T,K]:", tuple(batch["codes"].shape))
print("input lengths:", batch["input_lengths"].tolist())
print("target lengths:", batch["target_lengths"].tolist())
PY
```

Run a small one-epoch training smoke test before the full experiment:

```bash
python -m rvq_asr.train_probe \
  --token-index 04_Code/torgo_manifest/encodec_tokens_24khz_6kbps/tokens.jsonl \
  --token-root 04_Code/torgo_manifest/encodec_tokens_24khz_6kbps \
  --output-dir 04_Code/rvq_asr/runs/encodec_k4_smoke \
  --num-rvq-layers 4 \
  --epochs 1 \
  --batch-size 8 \
  --model-dim 128 \
  --encoder-layers 2 \
  --feedforward-dim 512 \
  --time-reduction 4 \
  --subsampling conv \
  --device cuda
```

For a full EnCodec `Q1:Q4` run, use the fixed probe capacity:

```bash
python -m rvq_asr.train_probe \
  --token-index 04_Code/torgo_manifest/encodec_tokens_24khz_6kbps/tokens.jsonl \
  --token-root 04_Code/torgo_manifest/encodec_tokens_24khz_6kbps \
  --output-dir 04_Code/rvq_asr/runs/encodec_k4_seed1337 \
  --num-rvq-layers 4 \
  --epochs 30 \
  --batch-size 16 \
  --time-reduction 4 \
  --subsampling conv \
  --seed 1337 \
  --device cuda
```

For DAC, keep every model/training option identical and change only the token
source and output directory:

```bash
python -m rvq_asr.train_probe \
  --token-index 04_Code/torgo_manifest/dac_tokens_24khz/tokens.jsonl \
  --token-root 04_Code/torgo_manifest/dac_tokens_24khz \
  --output-dir 04_Code/rvq_asr/runs/dac_k4_seed1337 \
  --num-rvq-layers 4 \
  --epochs 30 \
  --batch-size 16 \
  --time-reduction 4 \
  --subsampling conv \
  --seed 1337 \
  --device cuda
```

Use `rvq_asr.sweep_depths` to run every depth from `1` through the selected
checkpoint's actual `num_codebooks`. Use the same seeds (at least three),
architecture, optimizer, and decoding method for every ablation. Each run
writes `best.pt`, `results.json`, and `test_predictions.jsonl`; `results.json`
contains overall, per-severity, and per-speaker WER.

Every ablation instantiates embedding tables for all codebooks reported by the
checkpoint but activates only the first `K`. This keeps total model capacity
constant across every K; inactive embedding tables receive no gradients.

Before self-attention, the probe applies learned Conv1d subsampling using the
fixed `--time-reduction` factor (default `4`) and updates CTC lengths. The old
fixed average behavior remains available as `--subsampling average` for the
recorded baseline. Transformer attention memory grows quadratically with frame
length, so keep the same subsampling method and reduction factor in every
codec/layer ablation. Use `--grad-accum-steps` when a smaller physical batch is
needed, so experiments can retain the same effective batch size.

The seven-rotation protocol is mandatory for formal results. A single rotation may be used only for development or smoke testing.
