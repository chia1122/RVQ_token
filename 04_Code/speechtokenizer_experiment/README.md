# TORGO SpeechTokenizer layer experiment

This workflow uses the local `speechtokenizer_hubert_avg` checkpoint and writes
the same `[T,N]` token/index and reconstruction interfaces used by DAC.

## 1. Install the local package

From the repository root in the server environment:

```bash
python -m pip install -e SpeechTokenizer
python -c "from speechtokenizer import SpeechTokenizer; print('ok')"
```

The upstream package imports its trainer eagerly but does not list all trainer
dependencies in `install_requires`. If the verification import reports missing
`beartype` or TensorBoard, install them in the same environment:

```bash
python -m pip install beartype tensorboard
```

## 2. Extract all TORGO Q1-Q8 tokens

```bash
python 04_Code/torgo_manifest/extract_speechtokenizer_tokens.py \
  --manifest 04_Code/torgo_manifest/output/torgo_all.jsonl \
  --audio-root /data/TORGO \
  --output-dir 04_Code/torgo_manifest/speechtokenizer_hubert_avg_tokens \
  --config SpeechTokenizer/model_hub/speechtokenizer_hubert_avg/config.json \
  --checkpoint SpeechTokenizer/model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt \
  --device cuda
```

Verify exactly 7,140 saved utterances, zero failures, eight codebooks, codebook
size 1,024, and 50 Hz token frame rate:

```bash
cat 04_Code/torgo_manifest/speechtokenizer_hubert_avg_tokens/extraction_summary.json
wc -l 04_Code/torgo_manifest/speechtokenizer_hubert_avg_tokens/tokens.jsonl
wc -l 04_Code/torgo_manifest/speechtokenizer_hubert_avg_tokens/failures.jsonl
```

## 3. One-utterance reconstruction smoke test

```bash
python 04_Code/codec_reconstruction/reconstruct_speechtokenizer_prefixes.py \
  --token-index 04_Code/torgo_manifest/speechtokenizer_hubert_avg_tokens/tokens.jsonl \
  --token-root 04_Code/torgo_manifest/speechtokenizer_hubert_avg_tokens \
  --manifest 04_Code/torgo_manifest/output/torgo_all.jsonl \
  --audio-root /data/TORGO \
  --output-dir 04_Code/codec_reconstruction/outputs/speechtokenizer_smoke \
  --config SpeechTokenizer/model_hub/speechtokenizer_hubert_avg/config.json \
  --checkpoint SpeechTokenizer/model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt \
  --layers 1,2,3,4,5,6,7,8 \
  --split all \
  --speakers F01 \
  --limit 1 \
  --device cuda
```

Listen to all eight WAV files and verify the summary reports eight files and zero
failures.

## 4. Severe-speaker reconstruction and fixed-ASR evaluation

Remove `--limit` and use a unique full output directory:

```bash
python 04_Code/codec_reconstruction/reconstruct_speechtokenizer_prefixes.py \
  --token-index 04_Code/torgo_manifest/speechtokenizer_hubert_avg_tokens/tokens.jsonl \
  --token-root 04_Code/torgo_manifest/speechtokenizer_hubert_avg_tokens \
  --manifest 04_Code/torgo_manifest/output/torgo_all.jsonl \
  --audio-root /data/TORGO \
  --output-dir 04_Code/codec_reconstruction/outputs/speechtokenizer_severe \
  --config SpeechTokenizer/model_hub/speechtokenizer_hubert_avg/config.json \
  --checkpoint SpeechTokenizer/model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt \
  --layers 1,2,3,4,5,6,7,8 \
  --split all \
  --speakers F01,M01,M02,M04 \
  --device cuda

python 04_Code/codec_reconstruction/evaluate_with_faster_whisper.py \
  --manifest 04_Code/torgo_manifest/output/torgo_all.jsonl \
  --audio-root /data/TORGO \
  --reconstruction-index 04_Code/codec_reconstruction/outputs/speechtokenizer_severe/reconstruction_index.jsonl \
  --reconstruction-root 04_Code/codec_reconstruction/outputs/speechtokenizer_severe \
  --output-dir 04_Code/codec_reconstruction/asr_results/speechtokenizer_severe_large_v3 \
  --conditions auto \
  --split all \
  --speakers F01,M01,M02,M04 \
  --model large-v3 \
  --language en \
  --beam-size 5 \
  --device cuda \
  --compute-type float16
```

## 5. Paired comparison with DAC

Only condition/utterance pairs present in both prediction files are compared:

```bash
python 04_Code/codec_reconstruction/compare_codec_results.py \
  --dac-predictions 04_Code/codec_reconstruction/asr_results/dac_severe_large_v3/predictions.jsonl \
  --speech-predictions 04_Code/codec_reconstruction/asr_results/speechtokenizer_severe_large_v3/predictions.jsonl \
  --output 04_Code/codec_reconstruction/asr_results/dac_vs_speechtokenizer_severe.csv
```

## 6. Use the common direct-token CTC probe

No Dataset/model changes are required because the index and `.pt` schema match
DAC. First repeat the 100-sample K=4 overfit sanity check with a new run path:

```bash
export PYTHONPATH="$PWD/04_Code${PYTHONPATH:+:$PYTHONPATH}"

python -m rvq_asr.train_probe \
  --token-index 04_Code/torgo_manifest/speechtokenizer_hubert_avg_tokens/tokens.jsonl \
  --token-root 04_Code/torgo_manifest/speechtokenizer_hubert_avg_tokens \
  --output-dir 04_Code/rvq_asr/runs/speechtokenizer_k4_overfit100 \
  --num-rvq-layers 4 \
  --overfit-samples 100 \
  --epochs 100 \
  --batch-size 4 \
  --model-dim 128 \
  --encoder-layers 2 \
  --feedforward-dim 512 \
  --time-reduction 4 \
  --subsampling conv \
  --learning-rate 1e-3 \
  --weight-decay 0 \
  --dropout 0 \
  --device cuda
```

After this reaches near-zero WER/CER, run the fixed K=4 baseline and then the
same K=1/2/4/6/8 and seed protocol. A SpeechTokenizer checkpoint pretrained on
LibriSpeech does not mean the separate CTC probe is pretrained; direct-token
ASR still benefits from LibriSpeech token pretraining.
