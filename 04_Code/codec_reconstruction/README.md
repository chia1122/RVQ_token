# RVQ-prefix reconstruction and pretrained-ASR evaluation

This experiment decodes every EnCodec prefix (`Q1` through `Q1:Q8`) to WAV
and evaluates every available depth with one fixed faster-whisper
checkpoint and decoding configuration.

The same workflow supports DAC. If DAC is the primary codec, use
`reconstruct_dac_prefixes.py` and the DAC token root below; the pretrained-ASR
evaluator is codec-independent.

## SpeechTokenizer runtime K8 evaluation (no saved tokens)

This mode implements the following path without reading or writing codec token
files:

`TORGO WAV -> SpeechTokenizer encode (memory only) -> Q1:Q8 decode -> WAV -> fixed pretrained ASR`

Run these commands from `04_Code/` on Linux. Point the variables at persistent
storage outside the repository; reconstructed WAVs and ASR outputs must not be
committed.

```bash
ManifestFile="$TORGO_MILD_MANIFEST_ROOT/torgo_all.jsonl"
ReconRoot="$RVQ_ARTIFACT_ROOT/reconstructions/speechtokenizer_runtime_k8_v1"
AsrRoot="$RVQ_ARTIFACT_ROOT/asr/speechtokenizer_runtime_k8_large_v3_v1"

test -f "$ManifestFile"
test -n "$AudioRoot" && test -d "$AudioRoot"
test -f "$STConfig"
test -f "$STCheckpoint"
test ! -e "$ReconRoot"
test ! -e "$AsrRoot"
```

First reconstruct and evaluate one utterance:

```bash
python codec_reconstruction/reconstruct_speechtokenizer_prefixes.py \
  --input-mode audio \
  --manifest "$ManifestFile" \
  --audio-root "$AudioRoot" \
  --output-dir "${ReconRoot}_smoke" \
  --config "$STConfig" \
  --checkpoint "$STCheckpoint" \
  --layers 8 \
  --split all \
  --limit 1 \
  --device cuda

python codec_reconstruction/evaluate_with_faster_whisper.py \
  --manifest "$ManifestFile" \
  --audio-root "$AudioRoot" \
  --reconstruction-index "${ReconRoot}_smoke/reconstruction_index.jsonl" \
  --reconstruction-root "${ReconRoot}_smoke" \
  --output-dir "${AsrRoot}_smoke" \
  --conditions original,k8 \
  --split all \
  --limit-per-condition 1 \
  --model large-v3 \
  --language en \
  --beam-size 5 \
  --device cuda \
  --compute-type float16
```

After verifying that the K8 WAV is non-empty and the smoke ASR produced one
prediction, run the full reconstruction and evaluation with new output paths:

```bash
python codec_reconstruction/reconstruct_speechtokenizer_prefixes.py \
  --input-mode audio \
  --manifest "$ManifestFile" \
  --audio-root "$AudioRoot" \
  --output-dir "$ReconRoot" \
  --config "$STConfig" \
  --checkpoint "$STCheckpoint" \
  --layers 8 \
  --split all \
  --device cuda

python codec_reconstruction/evaluate_with_faster_whisper.py \
  --manifest "$ManifestFile" \
  --audio-root "$AudioRoot" \
  --reconstruction-index "$ReconRoot/reconstruction_index.jsonl" \
  --reconstruction-root "$ReconRoot" \
  --output-dir "$AsrRoot" \
  --conditions original,k8 \
  --split all \
  --model large-v3 \
  --language en \
  --beam-size 5 \
  --device cuda \
  --compute-type float16
```

The evaluator loads `faster-whisper large-v3` once and uses it for both
`original` and `k8`, with the same language, beam size, temperature,
`condition_on_previous_text`, text normalization, and corpus-level scoring.
This controlled rerun is required for a strict comparison. The older
`Whisper_dwer` scripts use the OpenAI `whisper` backend and utterance-mean WER,
so their original-audio result is useful historical context but is not a
strictly interchangeable baseline for this faster-whisper K8 result. If an
original baseline was already produced by this exact evaluator and settings,
K8 alone can instead be run with `--conditions k8` and no `--audio-root`.

Report WER/CER as ASR transcription error, not as clinical intelligibility.
For clinical group comparisons, summarize speakers as subjects rather than
treating utterances as independent subjects.

## DAC-first smoke test

Confirm the completed DAC extraction and actual codebook count:

```bash
cat 04_Code/torgo_manifest/dac_tokens_24khz/extraction_summary.json
```

Then reconstruct one test utterance:

```bash
python 04_Code/codec_reconstruction/reconstruct_dac_prefixes.py \
  --token-index 04_Code/torgo_manifest/dac_tokens_24khz/tokens.jsonl \
  --token-root 04_Code/torgo_manifest/dac_tokens_24khz \
  --manifest $MANIFEST_ROOT/torgo_all.jsonl \
  --audio-root /data/TORGO \
  --output-dir 04_Code/codec_reconstruction/outputs/dac_smoke \
  --layers 1,2,3,4,5,6,7,8 \
  --model 24khz \
  --split test \
  --limit 1 \
  --device cuda
```

Evaluate Original and all DAC prefixes with the same ASR:

```bash
python 04_Code/codec_reconstruction/evaluate_with_faster_whisper.py \
  --manifest $MANIFEST_ROOT/torgo_all.jsonl \
  --audio-root /data/TORGO \
  --reconstruction-index 04_Code/codec_reconstruction/outputs/dac_smoke/reconstruction_index.jsonl \
  --reconstruction-root 04_Code/codec_reconstruction/outputs/dac_smoke \
  --output-dir 04_Code/codec_reconstruction/asr_results/dac_smoke_large_v3 \
  --conditions auto \
  --split test \
  --limit-per-condition 1 \
  --model large-v3 \
  --language en \
  --beam-size 5 \
  --device cuda \
  --compute-type float16
```

After the smoke test, remove `--limit`, change both smoke paths to a unique
full-run directory (for example `outputs/dac_24khz`), and use `--split all`.

## 1. Environment

From the repository root on Linux:

```bash
python -c "import torch, torchaudio, encodec; print(torch.cuda.is_available())"
python -c "import faster_whisper; print(faster_whisper.__file__)"
```

If needed, install `faster-whisper` in the server environment. Its first use
may download the selected ASR checkpoint.

## 2. One-utterance reconstruction smoke test

```bash
python 04_Code/codec_reconstruction/reconstruct_encodec_prefixes.py \
  --token-index 04_Code/torgo_manifest/encodec_tokens_24khz_6kbps/tokens.jsonl \
  --token-root 04_Code/torgo_manifest/encodec_tokens_24khz_6kbps \
  --manifest $MANIFEST_ROOT/torgo_all.jsonl \
  --audio-root /data/TORGO \
  --output-dir 04_Code/codec_reconstruction/outputs/encodec_smoke \
  --layers 1,2,3,4,5,6,7,8 \
  --split test \
  --limit 1 \
  --device cuda
```

The summary should report one utterance, eight WAV files, and zero failures.
Listen to or inspect every generated WAV before the full run. K1 is expected to
sound coarser than K8, but every file must be non-empty and have approximately
the source duration.

## 3. Six-condition ASR smoke test

```bash
python 04_Code/codec_reconstruction/evaluate_with_faster_whisper.py \
  --manifest $MANIFEST_ROOT/torgo_all.jsonl \
  --audio-root /data/TORGO \
  --reconstruction-index 04_Code/codec_reconstruction/outputs/encodec_smoke/reconstruction_index.jsonl \
  --reconstruction-root 04_Code/codec_reconstruction/outputs/encodec_smoke \
  --output-dir 04_Code/codec_reconstruction/asr_results/encodec_smoke_large_v3 \
  --conditions auto \
  --split test \
  --limit-per-condition 1 \
  --model large-v3 \
  --language en \
  --beam-size 5 \
  --device cuda \
  --compute-type float16
```

Keep `model`, `language`, `beam-size`, and text normalization fixed for all
conditions. Use a new output directory if any of these settings change.

## 4. Full reconstruction

Use `--split all` to include every enrolled speaker and severity. This is valid
for a frozen pretrained ASR evaluation because no TORGO samples train the ASR.

```bash
python 04_Code/codec_reconstruction/reconstruct_encodec_prefixes.py \
  --token-index 04_Code/torgo_manifest/encodec_tokens_24khz_6kbps/tokens.jsonl \
  --token-root 04_Code/torgo_manifest/encodec_tokens_24khz_6kbps \
  --manifest $MANIFEST_ROOT/torgo_all.jsonl \
  --audio-root /data/TORGO \
  --output-dir 04_Code/codec_reconstruction/outputs/encodec_24khz_6kbps \
  --layers 1,2,3,4,5,6,7,8 \
  --split all \
  --device cuda
```

Expected output count is `7785 x 8 = 62280` WAV files. Rerunning the command
reuses existing WAV files unless `--overwrite` is supplied.

## 5. Full pretrained-ASR evaluation

```bash
python 04_Code/codec_reconstruction/evaluate_with_faster_whisper.py \
  --manifest $MANIFEST_ROOT/torgo_all.jsonl \
  --audio-root /data/TORGO \
  --reconstruction-index 04_Code/codec_reconstruction/outputs/encodec_24khz_6kbps/reconstruction_index.jsonl \
  --reconstruction-root 04_Code/codec_reconstruction/outputs/encodec_24khz_6kbps \
  --output-dir 04_Code/codec_reconstruction/asr_results/encodec_24khz_6kbps_large_v3 \
  --conditions auto \
  --split all \
  --model large-v3 \
  --language en \
  --beam-size 5 \
  --device cuda \
  --compute-type float16
```

The evaluator checkpoints `predictions.jsonl` every 100 items and resumes when
the same command/output directory is used. `--overwrite` deliberately starts
the ASR predictions again.

Outputs:

- `predictions.jsonl`: utterance-level reference, hypothesis, WER, CER,
  control/dysarthric `condition`, and depth-specific `rvq_condition`.
- `summary.csv`: WER/CER grouped by overall, control/dysarthric condition, speaker, and severity.
- `comparison_by_condition.csv`: horizontal depth comparison for control/dysarthric speech.
- `comparison_by_speaker.csv`: horizontal table with every actually available depth.
- `comparison_by_severity.csv`: horizontal condition comparison by severity.
- `failures.jsonl`: failed items.
- `experiment.json`: fixed ASR and decoding configuration.

The primary comparison is within the same speaker/severity across `original`,
every actually available `kN`. This reconstructed-audio experiment is a
complement to, not a replacement for, direct discrete-token probing.

## Severe-speaker experiment and paired bootstrap

Both reconstruction and ASR evaluation accept a comma-separated speaker
filter. Reconstruct all enrolled Severe speakers:

```bash
python 04_Code/codec_reconstruction/reconstruct_dac_prefixes.py \
  --token-index 04_Code/torgo_manifest/dac_tokens_24khz/tokens.jsonl \
  --token-root 04_Code/torgo_manifest/dac_tokens_24khz \
  --manifest $MANIFEST_ROOT/torgo_all.jsonl \
  --audio-root /data/TORGO \
  --output-dir 04_Code/codec_reconstruction/outputs/dac_severe \
  --layers 1,2,3,4,5,6,7,8 \
  --model 24khz \
  --split all \
  --speakers F01,M01,M02,M04 \
  --device cuda
```

Run the fixed pretrained ASR on Original and every prefix:

```bash
python 04_Code/codec_reconstruction/evaluate_with_faster_whisper.py \
  --manifest $MANIFEST_ROOT/torgo_all.jsonl \
  --audio-root /data/TORGO \
  --reconstruction-index 04_Code/codec_reconstruction/outputs/dac_severe/reconstruction_index.jsonl \
  --reconstruction-root 04_Code/codec_reconstruction/outputs/dac_severe \
  --output-dir 04_Code/codec_reconstruction/asr_results/dac_severe_large_v3 \
  --conditions auto \
  --split all \
  --speakers F01,M01,M02,M04 \
  --model large-v3 \
  --language en \
  --beam-size 5 \
  --device cuda \
  --compute-type float16
```

Finally compare K4 and K8 with 10,000 paired utterance bootstrap samples per
speaker:

```bash
python 04_Code/codec_reconstruction/paired_bootstrap.py \
  --predictions 04_Code/codec_reconstruction/asr_results/dac_severe_large_v3/predictions.jsonl \
  --condition-a k4 \
  --condition-b k8 \
  --samples 10000 \
  --seed 1337 \
  --output 04_Code/codec_reconstruction/asr_results/dac_severe_large_v3/k4_vs_k8_bootstrap.csv
```

The reported delta is `K8 - K4`; a negative value favors K8. A 95% confidence
interval entirely below zero provides evidence that K8 improves that speaker.
An interval entirely above zero favors K4. An interval spanning zero is
inconclusive.
