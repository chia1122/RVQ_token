# Codec_RVQ

Speech codec and Residual Vector Quantization (RVQ) experiments for dysarthric speech processing and automatic speech recognition (ASR).

## Current research direction

The current research direction is **Pathology-aware RVQ Layer Fusion for
Dysarthric ASR**. Layer-wise ASR trajectories and information probes are used
as diagnostic evidence and method motivation; they are not the final research
endpoint. The proposed direction is to test whether utterance-adaptive fusion
of RVQ layers can reduce dysarthric-speech CER without a clear degradation of
control-speech CER.

See the [research planning index](05_doc/README.md) and the canonical
[Pathology-aware RVQ Fusion roadmap](05_doc/PATHOLOGY_AWARE_RVQ_FUSION_ROADMAP.md).

The current direct-token depth trajectory uses task-trained discrete
embeddings for cumulative Q1:QK prefixes. It must not be confused with an
individual-QK trajectory or a frozen codec-native representation. WER and CER
are ASR performance metrics, not clinical intelligibility measures.

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
       │             K1 / K2 / K3 / K4 /
       │             K5 / K6 / K7 / K8
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

The canonical dataset protocol is the 15-speaker, seven-rotation
`torgo_including_mild_v1` protocol. Its versioned source files are:

```text
04_Code/torgo_manifest/config/speaker_metadata.csv
04_Code/torgo_manifest/config/speaker_folds.json
```

Generate and validate all speaker-disjoint rotations before building manifests.
See `04_Code/torgo_manifest/README.md` for exact commands. The removed
13-speaker fixed split must not be used for new experiments.

## LibriSpeech

LibriSpeech remains available for historical ASR pretraining workflows. It is
not part of the individual-codebook probe milestone.

## UA-Speech

UA-Speech remains available to `benchmark_UA_auto.py`; it is not part of the
current TORGO probe protocol.

---

# 6. Existing analysis workflows

The cumulative ASR trajectory below is retained as historical tooling. The
current milestone evaluates frozen codec-native individual Q1--Q8
representations and must not launch cumulative or adaptive-fusion experiments.

### Step 6 — Aggregate seven speaker-fold rotations

After all seven CER-selected rotations pass their individual 24-run audits,
combine them without rerunning training. Use a new output directory; the
aggregator refuses to overwrite a non-empty directory and ignores a
`rotation_00` smoke directory.

```bash
export PYTHONPATH="$PWD/04_Code"
export PROTOCOL_TRAJECTORY_ROOT="$RVQ_ARTIFACT_ROOT/trajectories/torgo_including_mild_v1_cer_v1"
export COMBINED_TRAJECTORY_ROOT="$RVQ_ARTIFACT_ROOT/trajectory_aggregates/torgo_including_mild_v1_cer_v1"

python -m rvq_asr.aggregate_rotations \
  --trajectory-root "$PROTOCOL_TRAJECTORY_ROOT" \
  --output-dir "$COMBINED_TRAJECTORY_ROOT" \
  --protocol-id torgo_including_mild_v1_cer_v1
```

The output separates three estimands:

- `trajectory_run_summary.csv`: fold/run-macro means across the 21
  rotation-seed results per depth;
- `trajectory_pooled_micro_*.csv`: reference-count-weighted metrics pooled
  across the seven test folds separately for each seed, followed by a
  three-seed summary;
- `trajectory_speaker_macro_*.csv`: equal-speaker means overall and within
  condition/severity groups.

`ctc_blank_frame_ratio` remains a fold/run- or speaker-macro metric because
the existing result schema does not retain the valid-frame denominator needed
for an exact pooled-micro value.

### Step 7 — Run matched individual RVQ layers

The individual-layer sweep inherits every training argument, seed, codec, and
rotation-specific token path from the completed cumulative trajectory. It
does not accept replacement training hyperparameters.

```bash
export PYTHONPATH="$PWD/04_Code"
export CUMULATIVE_ROOT="$RVQ_ARTIFACT_ROOT/trajectories/torgo_including_mild_v1_cer_v1"
export INDIVIDUAL_ROOT="$RVQ_ARTIFACT_ROOT/trajectories/torgo_including_mild_v1_individual_cer_v1"

python -m rvq_asr.sweep_individual_layers \
  --reference-trajectory-root "$CUMULATIVE_ROOT" \
  --output-root "$INDIVIDUAL_ROOT" \
  --protocol-id torgo_including_mild_v1_individual_cer_v1 \
  --dry-run > /tmp/torgo_individual_dry_run.json
```

Audit the dry run before removing `--dry-run`. The full SpeechTokenizer plan
contains seven rotations, eight individual layers, and three seeds (168
runs). Each QK run uses `--num-rvq-layers K --active-rvq-layers K` and writes
to an `individual_qK/seed_*` directory. Resume an interrupted formal run with
`--resume`; existing completed `results.json` files are skipped.

After all runs are valid, aggregate them into a new directory:

```bash
export INDIVIDUAL_AGGREGATE_ROOT="$RVQ_ARTIFACT_ROOT/trajectory_aggregates/torgo_including_mild_v1_individual_cer_v1"

python -m rvq_asr.aggregate_rotations \
  --trajectory-root "$INDIVIDUAL_ROOT" \
  --output-dir "$INDIVIDUAL_AGGREGATE_ROOT" \
  --protocol-id torgo_including_mild_v1_individual_cer_v1
```

Pair the individual and cumulative long-format results only after both
aggregation audits pass:

```bash
export CUMULATIVE_AGGREGATE_ROOT="$RVQ_ARTIFACT_ROOT/trajectory_aggregates/torgo_including_mild_v1_cer_v1"
export COMPARISON_ROOT="$RVQ_ARTIFACT_ROOT/trajectory_comparisons/torgo_individual_vs_cumulative_cer_v1"

python -m rvq_asr.compare_representations \
  --cumulative-root "$CUMULATIVE_AGGREGATE_ROOT" \
  --individual-root "$INDIVIDUAL_AGGREGATE_ROOT" \
  --output-dir "$COMPARISON_ROOT"
```

For error metrics, a negative `delta_individual_minus_cumulative` favors the
individual QK condition. The comparison does not by itself demonstrate that
later layers are complementary under fusion.

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
