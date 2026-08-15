# Codec_RVQ

Speech codec and Residual Vector Quantization (RVQ) experiments for dysarthric speech processing and automatic speech recognition (ASR).

This repository contains tools for:

- preparing the TORGO speech dataset;
- extracting discrete speech tokens using neural audio codecs;
- reconstructing speech from different RVQ prefix depths;
- evaluating reconstructed speech with pretrained ASR;
- training ASR probes directly on RVQ tokens;
- pretraining RVQ-based ASR probes on LibriSpeech;
- running SpeechTokenizer experiments;
- benchmarking codec reconstruction on UA-Speech.

---

## Repository Structure

```text
Codec_RVQ/
│
├── codec_reconstruction/
│   ├── README.md
│   ├── reconstruct_dac_prefixes.py
│   ├── reconstruct_encodec_prefixes.py
│   ├── reconstruct_speechtokenizer_prefixes.py
│   ├── evaluate_with_faster_whisper.py
│   ├── compare_codec_results.py
│   ├── paired_bootstrap.py
│   └── test_helpers.py
│
├── librispeech_pretraining/
│   ├── README.md
│   ├── build_librispeech_manifest.py
│   └── test_manifest.py
│
├── rvq_asr/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── text.py
│   ├── train_probe.py
│   ├── test_model.py
│   ├── test_text.py
│   └── reports/
│
├── speechtokenizer_experiment/
│   └── README.md
│
├── torgo_manifest/
│   ├── README.md
│   ├── config/
│   ├── build_torgo_manifest.py
│   ├── extract_dac_tokens.py
│   ├── extract_encodec_tokens.py
│   ├── extract_speechtokenizer_tokens.py
│   ├── test_build_torgo_manifest.py
│   ├── test_extract_dac_tokens.py
│   └── test_extract_encodec_tokens.py
│
└── benchmark_UA_auto.py
```

---

# 1. Project Architecture

The main TORGO workflow is:

```text
TORGO audio + transcript index
            │
            ▼
     torgo_manifest/
            │
            ├── Build normalized JSONL manifests
            │
            ▼
     Speech codec encoder
     ├── DAC
     ├── EnCodec
     └── SpeechTokenizer
            │
            ▼
       RVQ token files
            │
       ┌────┴─────────────┐
       │                  │
       ▼                  ▼
rvq_asr/          codec_reconstruction/
       │                  │
       │             Decode RVQ prefixes
       │             K1 / K2 / K4 /
       │             K6 / K8
       │                  │
       ▼                  ▼
Direct token       Reconstructed WAV
ASR probing               │
                          ▼
                  pretrained ASR
                  (faster-whisper)
                          │
                          ▼
                     WER / CER
```

The repository therefore supports two related approaches:

1. **Direct RVQ-token probing** — train an ASR model directly from discrete codec tokens.
2. **Codec reconstruction evaluation** — reconstruct audio using different numbers of RVQ layers and evaluate the reconstructed speech using a fixed pretrained ASR system.

---

# 2. Main Components

## `torgo_manifest/`

Dataset preparation and codec-token extraction for TORGO.

### Main scripts

**`build_torgo_manifest.py`**

Converts the TORGO transcript/index information into normalized JSONL manifests with speaker metadata and dataset splits.

**`extract_dac_tokens.py`**

Encodes TORGO audio using DAC and stores the resulting RVQ tokens.

**`extract_encodec_tokens.py`**

Encodes TORGO audio using EnCodec and stores the resulting RVQ tokens.

**`extract_speechtokenizer_tokens.py`**

Encodes TORGO audio using SpeechTokenizer.

See `torgo_manifest/README.md` for detailed dataset preparation instructions.

---

## `codec_reconstruction/`

Reconstructs speech from different RVQ prefix depths and evaluates the resulting audio.

Supported RVQ conditions include:

```text
K1  = first RVQ layer
K2  = first 2 RVQ layers
K4  = first 4 RVQ layers
K6  = first 6 RVQ layers
K8  = first 8 RVQ layers
```

### Reconstruction scripts

```text
reconstruct_dac_prefixes.py
reconstruct_encodec_prefixes.py
reconstruct_speechtokenizer_prefixes.py
```

These scripts take previously extracted codec tokens and decode different RVQ prefixes back into WAV files.

### ASR evaluation

`evaluate_with_faster_whisper.py`

Runs the same pretrained faster-whisper model on:

```text
Original
K1
K2
K4
K6
K8
```

The evaluator produces files including:

```text
predictions.jsonl
summary.csv
comparison_by_speaker.csv
comparison_by_severity.csv
failures.jsonl
experiment.json
```

The primary ASR metrics are:

- Word Error Rate (WER)
- Character Error Rate (CER)

### Statistical comparison

`paired_bootstrap.py`

Performs paired bootstrap comparisons between two reconstruction conditions.

### Cross-codec comparison

`compare_codec_results.py`

Used to compare evaluation results produced by different codec conditions.

See `codec_reconstruction/README.md` for the complete reconstruction and evaluation commands.

---

## `rvq_asr/`

Implements ASR probing directly from discrete RVQ tokens.

### Main files

**`data.py`**

Dataset loading and preparation for RVQ token sequences.

**`model.py`**

Neural network components used by the RVQ ASR probe.

**`text.py`**

Text processing utilities used during ASR training and evaluation.

**`train_probe.py`**

Main training entry point.

Example module invocation:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m rvq_asr.train_probe \
    --token-index PATH_TO_TOKENS/tokens.jsonl \
    --token-root PATH_TO_TOKENS \
    --output-dir PATH_TO_OUTPUT \
    --num-rvq-layers 4 \
    --device cuda
```

Additional training parameters are available in `train_probe.py`.

---

## `librispeech_pretraining/`

Pretraining workflow for RVQ-token ASR models using LibriSpeech.

The expected LibriSpeech subsets are:

```text
LibriSpeech/
├── train-clean-100/
├── dev-clean/
└── test-clean/
```

### Workflow

```text
LibriSpeech
     │
     ▼
build_librispeech_manifest.py
     │
     ▼
LibriSpeech JSONL manifest
     │
     ▼
extract_dac_tokens.py
     │
     ▼
DAC RVQ tokens
     │
     ▼
rvq_asr.train_probe
     │
     ▼
Pretrained ASR probe
     │
     ▼
TORGO fine-tuning
```

Build the manifest with:

```bash
python librispeech_pretraining/build_librispeech_manifest.py \
    --audio-root /data/LibriSpeech \
    --output-dir librispeech_pretraining/manifest
```

Then extract DAC tokens using the same DAC model configuration used for TORGO.

See `librispeech_pretraining/README.md` for the complete pretraining and fine-tuning protocol.

---

## `speechtokenizer_experiment/`

Contains the workflow for experiments using the local SpeechTokenizer model.

The current setup uses the `speechtokenizer_hubert_avg` checkpoint.

The general workflow is:

```text
TORGO
  │
  ▼
SpeechTokenizer
  │
  ▼
Q1–Q8 tokens
  │
  ├───────────────┐
  ▼               ▼
RVQ probe     Reconstruction
                  │
                  ▼
             faster-whisper
```

See `speechtokenizer_experiment/README.md` for detailed commands.

---

## `benchmark_UA_auto.py`

Standalone EnCodec reconstruction benchmark for UA-Speech.

The script:

1. loads WAV files from a configured UA-Speech directory;
2. runs EnCodec at multiple target bandwidths;
3. reconstructs the audio;
4. saves reconstructed WAV files;
5. evaluates reconstruction quality using PESQ and STOI.

Unlike the main TORGO pipeline, several paths and experiment settings are currently defined directly inside this script.

Check and modify these variables before running:

```python
folder_num
data_dir
output_base_dir
test_bandwidths
```

---

# 3. Installation

## Clone the repository

```bash
git clone https://github.com/chia1122/Codec_RVQ.git
cd Codec_RVQ
```

## Create a Python environment

A dedicated Python environment is strongly recommended.

For example, with Conda:

```bash
conda create -n codec_rvq python
conda activate codec_rvq
```

> **Important:** The repository currently does not contain a root-level `requirements.txt` or `environment.yml`. Therefore, the exact dependency versions used in the original environment are not yet reproducible from the repository alone.

---

# 4. Main Dependencies

The codebase currently uses packages including:

```text
PyTorch
torchaudio
NumPy
tqdm
EnCodec
DAC
faster-whisper
PESQ
pystoi
SpeechTokenizer
```

Different workflows require different subsets of these dependencies.

Before running the EnCodec workflow, verify:

```bash
python -c "import torch, torchaudio, encodec; print(torch.cuda.is_available())"
```

Before running pretrained ASR evaluation, verify:

```bash
python -c "import faster_whisper; print(faster_whisper.__file__)"
```

For SpeechTokenizer, install the local package from the repository/workspace containing the `SpeechTokenizer` source:

```bash
python -m pip install -e SpeechTokenizer
```

If required:

```bash
python -m pip install beartype tensorboard
```

GPU execution is recommended for codec extraction, reconstruction, ASR evaluation, and model training.

---

# 5. Dataset Setup

## TORGO

The TORGO audio root should directly contain speaker directories such as:

```text
/data/TORGO/
├── F01/
├── F03/
├── F04/
├── M01/
├── M02/
├── M03/
├── M04/
├── M05/
└── ...
```

The manifest builder also expects the TORGO transcript/index CSV used by the project.

Before generating the final manifest, review:

```text
torgo_manifest/config/speaker_metadata.csv
torgo_manifest/config/speaker_splits.json
```

See `torgo_manifest/README.md` for the expected metadata and split configuration.

## LibriSpeech

For pretraining:

```text
/data/LibriSpeech/
├── train-clean-100/
├── dev-clean/
└── test-clean/
```

Official LibriSpeech `*.trans.txt` annotations are used to build the manifest.

## UA-Speech

UA-Speech is used only by the standalone:

```text
benchmark_UA_auto.py
```

The input and output directories are currently configured directly inside that script.

---

# 6. Recommended Execution Order

For a new environment, run the project in the following order.

### Step 1 — Prepare TORGO manifest

```bash
python torgo_manifest/build_torgo_manifest.py \
    --index PATH_TO_TORGO_INDEX/torgo.csv \
    --audio-root /data/TORGO \
    --speaker-metadata torgo_manifest/config/speaker_metadata.csv \
    --split-config torgo_manifest/config/speaker_splits.json \
    --output-dir torgo_manifest/output
```

Check the generated manifest before continuing.

### Step 2 — Extract codec tokens

Choose one codec.

For DAC:

```bash
python torgo_manifest/extract_dac_tokens.py \
    --manifest torgo_manifest/output/torgo_all.jsonl \
    --audio-root /data/TORGO \
    --output-dir torgo_manifest/dac_tokens_24khz \
    --model 24khz \
    --device cuda
```

The extraction output should contain a token index and extraction summary.

### Step 3 — Reconstruct RVQ prefixes

Example using DAC:

```bash
python codec_reconstruction/reconstruct_dac_prefixes.py \
    --token-index torgo_manifest/dac_tokens_24khz/tokens.jsonl \
    --token-root torgo_manifest/dac_tokens_24khz \
    --manifest torgo_manifest/output/torgo_all.jsonl \
    --audio-root /data/TORGO \
    --output-dir codec_reconstruction/outputs/dac_smoke \
    --layers 1,2,4,6,8 \
    --model 24khz \
    --split test \
    --limit 1 \
    --device cuda
```

Always perform a small smoke test before processing the entire dataset.

### Step 4 — Evaluate reconstructed audio

```bash
python codec_reconstruction/evaluate_with_faster_whisper.py \
    --manifest torgo_manifest/output/torgo_all.jsonl \
    --audio-root /data/TORGO \
    --reconstruction-index codec_reconstruction/outputs/dac_smoke/reconstruction_index.jsonl \
    --reconstruction-root codec_reconstruction/outputs/dac_smoke \
    --output-dir codec_reconstruction/asr_results/dac_smoke_large_v3 \
    --conditions original,k1,k2,k4,k6,k8 \
    --split test \
    --limit-per-condition 1 \
    --model large-v3 \
    --language en \
    --beam-size 5 \
    --device cuda \
    --compute-type float16
```

After confirming that the smoke test works, remove the limits and use a separate output directory for the full run.

---

# 7. Running Tests

Several modules contain test files.

Examples:

```bash
python -m pytest torgo_manifest/
python -m pytest rvq_asr/
python -m pytest codec_reconstruction/
python -m pytest librispeech_pretraining/
```

Run the relevant tests after modifying dataset processing, codec extraction, model, or text-processing code.

---

# 8. Important Notes for Future Maintainers

### Paths

Some existing documentation uses paths such as:

```text
04_Code/...
```

because the code was originally run inside a larger project workspace.

This GitHub repository already contains those modules at its root. When running directly from a clone of `Codec_RVQ`, use paths relative to the repository root as shown in this README.

### External data

The speech datasets are not stored in this repository.

You must separately obtain and configure the required:

- TORGO data;
- LibriSpeech data;
- UA-Speech data, when using the UA-Speech benchmark.

### Model checkpoints

Some models or checkpoints may be downloaded automatically by their libraries, while SpeechTokenizer currently expects a local checkpoint/configuration.

Verify checkpoint locations before starting large experiments.

### Smoke tests first

Do not start full reconstruction or ASR evaluation immediately.

First verify:

1. the manifest is correct;
2. token extraction reports no unexpected failures;
3. one utterance can be reconstructed;
4. reconstructed WAV files are valid;
5. pretrained ASR can process all requested conditions.

### Keep experimental configurations fixed

When comparing RVQ conditions, keep ASR settings such as model checkpoint, language, beam size, and text normalization fixed.

Use a new output directory whenever an experimental configuration changes.

---

# 9. Current Maintenance Status

This repository contains working code and module-level experiment instructions, but the environment has not yet been fully packaged for reproducibility.

Recommended maintenance tasks:

- [ ] Add a root-level `requirements.txt` or `environment.yml`
- [ ] Record the Python version used on the experiment server
- [ ] Record the PyTorch and CUDA versions
- [ ] Document the exact DAC, EnCodec, and SpeechTokenizer versions/checkpoints
- [ ] Remove or parameterize hard-coded paths in `benchmark_UA_auto.py`
- [ ] Standardize paths between the original `04_Code/` workspace and this standalone repository
- [ ] Add a unified experiment configuration system if the project continues to grow

---

## Repository

Codec_RVQ is maintained at:

`chia1122/Codec_RVQ`