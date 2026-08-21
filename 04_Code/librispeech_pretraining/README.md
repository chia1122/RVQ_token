# LibriSpeech DAC-token pretraining and TORGO fine-tuning

Use official LibriSpeech `*.trans.txt` annotations. Do not use the existing
workspace transcript CSVs for pretraining: their row counts and example errors
indicate that they are not a clean one-row-per-official-utterance reference.

## 1. Required subsets

The default protocol expects the following directories under one audio root:

```text
/data/LibriSpeech/
├── train-clean-100/
├── dev-clean/
└── test-clean/
```

Each chapter directory must contain FLAC files and its official
`speaker-chapter.trans.txt` file.

## 2. Build the official manifest

```bash
python 04_Code/librispeech_pretraining/build_librispeech_manifest.py \
  --audio-root /data/LibriSpeech \
  --output-dir 04_Code/librispeech_pretraining/manifest
```

Inspect:

```bash
cat 04_Code/librispeech_pretraining/manifest/manifest_summary.json
wc -l 04_Code/librispeech_pretraining/manifest/librispeech_*.jsonl
```

The expected train-clean-100 count is approximately 28.5k utterances. A count
around 135k is not train-clean-100 and must be investigated before continuing.

## 3. Extract tokens with the same DAC checkpoint

Use exactly the same model selector as TORGO (`24khz` here):

```bash
python librispeech_pretraining/extract_speechtokenizer_librispeech_tokens.py \
  --manifest librispeech_pretraining/manifest/librispeech_test.jsonl\
  --audio-root /data/LibriSpeech \
  --output-dir librispeech_pretraining/speechtokenizer_hubert_avg_tokens_test_clean \
  --config /home/rachel/06_opensource_toolkit/SpeechTokenizer/model_hub/speechtokenizer_hubert_avg/config.json \
  --checkpoint /home/rachel/06_opensource_toolkit/SpeechTokenizer/model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt \
  --subsets test-clean \
  --device cuda
```

Verify zero failures and confirm that `num_codebooks`, `codebook_size`, sample
rate, and checkpoint match the TORGO DAC extraction summary.

## 4. Pretrain a K=4 probe

From the repository root:

```bash
export PYTHONPATH="$PWD/04_Code${PYTHONPATH:+:$PYTHONPATH}"

python -m rvq_asr.train_probe \
  --token-index 04_Code/librispeech_pretraining/dac_tokens_24khz/tokens.jsonl \
  --token-root 04_Code/librispeech_pretraining/dac_tokens_24khz \
  --output-dir 04_Code/librispeech_pretraining/runs/dac_k4_pretrain \
  --num-rvq-layers 4 \
  --epochs 30 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --model-dim 256 \
  --encoder-layers 4 \
  --heads 4 \
  --feedforward-dim 1024 \
  --time-reduction 4 \
  --subsampling conv \
  --learning-rate 3e-4 \
  --seed 1337 \
  --device cuda
```

This has physical batch 4 and effective batch 16. If memory allows a larger
physical batch, reduce accumulation so the effective batch stays fixed.

### combine all the tokens
python librispeech_pretraining/merge_speechtokenizer_tokens.py \
  --train-dir librispeech_pretraining/speechtokenizer_hubert_avg_tokens_train_clean_100 \
  --dev-dir librispeech_pretraining/speechtokenizer_hubert_avg_tokens_dev_clean \
  --test-dir librispeech_pretraining/speechtokenizer_hubert_avg_tokens_test_clean \
  --output librispeech_pretraining/tokens_librispeech_all.jsonl

## 5. Fine-tune the same K=4 model on TORGO

All architecture and RVQ settings must match pretraining. Use a lower learning
rate and a new output directory:

```bash
python -m rvq_asr.train_probe \
  --token-index 04_Code/torgo_manifest/dac_tokens_24khz/tokens.jsonl \
  --token-root 04_Code/torgo_manifest/dac_tokens_24khz \
  --init-checkpoint 04_Code/librispeech_pretraining/runs/dac_k4_pretrain/best.pt \
  --output-dir 04_Code/librispeech_pretraining/runs/torgo_dac_k4_finetune \
  --num-rvq-layers 4 \
  --epochs 30 \
  --batch-size 4 \
  --model-dim 256 \
  --encoder-layers 4 \
  --heads 4 \
  --feedforward-dim 1024 \
  --time-reduction 4 \
  --subsampling conv \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --dropout 0.1 \
  --evaluate-train \
  --seed 1337 \
  --device cuda
```

## 6. Layer ablation protocol

First verify that pretrained K=4 materially improves TORGO validation/test CER
over random initialization. Then repeat the complete pretrain/fine-tune process
for every available K from 1 through N and seeds 1337, 2026, and 3407.

Do not initialize a K=8 experiment from a checkpoint pretrained with active
K=4. Although all embedding tables exist, Q5-Q8 were inactive and received no
pretraining gradients. Every K needs the same amount of pretraining under its
own active-layer condition, unless a separately specified RVQ layer-dropout
pretraining protocol is used for all experiments.

The current fixed TORGO split remains a development protocol. Final severity
claims require speaker-level cross-validation, with pretraining weights held
fixed and only TORGO fine-tuning repeated per fold.
