# TORGO manifest builder

This tool converts the existing TORGO CSV index into normalized, speaker-independent JSONL manifests.

## Before the final build

1. Put the official TORGO audio somewhere accessible. The audio root must directly contain `F01/`, `F03/`, `M01/`, and the other speaker directories.
2. Replace the `citation TODO` text in `config/speaker_metadata.csv` with the paper/table citation used for the severity labels.
3. Review `config/speaker_splits.json`. It is speaker-disjoint, but the small number of speakers prevents every severity from appearing in every split.

The current paper-based protocol labels `F01`, `M01`, `M02`, and `M04` as severe; `M05` as moderate-to-severe; `F03` as moderate; and `F04`/`M03` as mild/ignored. The latter two have `include_in_experiment=false`, so their selected-channel recordings are retained in the exclusion audit but omitted from the ASR manifests.

## Draft build without audio

Run from the repository root (the directory containing both `04_Code/` and
`TORGO_Transcript/`).

Linux Bash:

```bash
test -f 04_Code/torgo_manifest/build_torgo_manifest.py
test -f TORGO_Transcript/torgo.csv

python 04_Code/torgo_manifest/build_torgo_manifest.py \
  --index TORGO_Transcript/torgo.csv \
  --audio-root /data/TORGO \
  --speaker-metadata 04_Code/torgo_manifest/config/speaker_metadata.csv \
  --split-config 04_Code/torgo_manifest/config/speaker_splits.json \
  --output-dir 04_Code/torgo_manifest/output_draft \
  --allow-missing-audio
```

PowerShell:

```powershell
python 04_Code/torgo_manifest/build_torgo_manifest.py `
  --index TORGO_Transcript/torgo.csv `
  --audio-root C:/path/to/TORGO `
  --speaker-metadata 04_Code/torgo_manifest/config/speaker_metadata.csv `
  --split-config 04_Code/torgo_manifest/config/speaker_splits.json `
  --output-dir 04_Code/torgo_manifest/output_draft `
  --allow-missing-audio
```

## Final build

Remove `--allow-missing-audio`. Missing or invalid audio is then recorded in `excluded_samples.csv` and cannot silently enter a manifest.

Linux Bash:

```bash
python 04_Code/torgo_manifest/build_torgo_manifest.py \
  --index TORGO_Transcript/torgo.csv \
  --audio-root /data/TORGO \
  --speaker-metadata 04_Code/torgo_manifest/config/speaker_metadata.csv \
  --split-config 04_Code/torgo_manifest/config/speaker_splits.json \
  --output-dir 04_Code/torgo_manifest/output
```

PowerShell:

```powershell
python 04_Code/torgo_manifest/build_torgo_manifest.py `
  --index TORGO_Transcript/torgo.csv `
  --audio-root D:/datasets/TORGO `
  --speaker-metadata 04_Code/torgo_manifest/config/speaker_metadata.csv `
  --split-config 04_Code/torgo_manifest/config/speaker_splits.json `
  --output-dir 04_Code/torgo_manifest/output
```

The builder selects only `headMic` by default, ignores the old utterance-level `test_data` field, rejects unintelligible transcripts, checks unique utterance/audio IDs, and asserts that speakers do not cross splits. Numeric transcripts are explicitly excluded instead of being silently corrupted; the current CSV contains one such selected-channel row (`FEBRUARY 13TH`). Add a documented number-expansion policy before retaining it.

Outputs include `torgo_all.jsonl`, one JSONL per split, `excluded_samples.csv`, `dataset_statistics.csv`, and `build_audit.json`.

Run tests with Bash or PowerShell:

```bash
python -m unittest discover -s 04_Code/torgo_manifest -p "test_*.py"
```

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
head -n 1 04_Code/torgo_manifest/output/torgo_all.jsonl \
  > 04_Code/torgo_manifest/output/smoke.jsonl

python 04_Code/torgo_manifest/extract_encodec_tokens.py \
  --manifest 04_Code/torgo_manifest/output/smoke.jsonl \
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
  --manifest 04_Code/torgo_manifest/output/torgo_all.jsonl \
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
  --manifest 04_Code/torgo_manifest/output/smoke.jsonl \
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
  --manifest 04_Code/torgo_manifest/output/torgo_all.jsonl \
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

Repeat with `--num-rvq-layers 1`, `2`, `4`, `6`, and `8` when the selected
checkpoint has at least eight codebooks. Use the same seeds (at least three),
architecture, optimizer, and decoding method for every ablation. Each run
writes `best.pt`, `results.json`, and `test_predictions.jsonl`; `results.json`
contains overall, per-severity, and per-speaker WER.

Every ablation instantiates embedding tables for all codebooks reported by the
checkpoint but activates only the first `K`. This keeps total model capacity
constant across `K=1/2/4/6/8`; inactive embedding tables receive no gradients.

Before self-attention, the probe applies learned Conv1d subsampling using the
fixed `--time-reduction` factor (default `4`) and updates CTC lengths. The old
fixed average behavior remains available as `--subsampling average` for the
recorded baseline. Transformer attention memory grows quadratically with frame
length, so keep the same subsampling method and reduction factor in every
codec/layer ablation. Use `--grad-accum-steps` when a smaller physical batch is
needed, so experiments can retain the same effective batch size.

The current fixed split is suitable for pipeline development but not the final
severity claim because its test speakers do not cover every severity. Replace
it with speaker-level cross-validation before reporting final probing results.
