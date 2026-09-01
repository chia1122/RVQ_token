# Running the probes

Run from the repository root. These paths match the current workspace.

```bash
export PYTHONPATH=04_Code
export CODEC=/home/rachel/06_opensource_toolkit/SpeechTokenizer/model_hub/speechtokenizer_hubert_avg
export TOKENS=/home/rachel/rvq_token_artifacts/tokens/speechtokenizer_hubert_avg/torgo_including_mild_v1_master
export ROTATIONS=/home/rachel/rvq_token_artifacts/token_indices/speechtokenizer_hubert_avg/torgo_including_mild_v1_v1
export INDEX=$ROTATIONS/rotation_01_test_a/tokens.jsonl
export TARGETS=/home/rachel/rvq_token_artifacts/probes/targets
```

1. Prepare phoneme targets:

```bash
NLTK_DATA=/home/rachel/nltk_data:/home/rachel/06_opensource_toolkit/nltk_data /home/rachel/.conda/envs/audio_codec/bin/python -m rvq_probes.prepare_phoneme_targets --token-index $TOKENS/tokens.jsonl --lexicon /home/rachel/06_opensource_toolkit/nltk_data/corpora/cmudict/cmudict --output-dir $TARGETS/torgo_arpabet_nostress_v1 --use-g2p
```

2. Validate all seven speaker rotations, then prepare and validate MFA alignments and boundary labels:

```bash
/home/rachel/.conda/envs/audio_codec/bin/python -m rvq_probes.validate_splits --rotations-root $ROTATIONS
NLTK_DATA=/home/rachel/nltk_data /home/rachel/.conda/envs/audio_codec/bin/python -m rvq_probes.prepare_mfa_corpus --token-index $TOKENS/tokens.jsonl --audio-root /data/TORGO --base-dictionary /home/rachel/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict --output-dir $TARGETS/torgo_mfa_arpa_first_v2
PATH=/home/rachel/.conda/envs/mfa/bin:$PATH /home/rachel/.conda/envs/mfa/bin/mfa validate $TARGETS/torgo_mfa_arpa_first_v2/corpus $TARGETS/torgo_mfa_arpa_first_v2/first_pronunciation.dict --acoustic_model_path english_us_arpa --output_directory $TARGETS/torgo_mfa_arpa_first_v2/validation -t $TARGETS/torgo_mfa_arpa_first_v2/temp_validate --clean
PATH=/home/rachel/.conda/envs/mfa/bin:$PATH /home/rachel/.conda/envs/mfa/bin/mfa align $TARGETS/torgo_mfa_arpa_first_v2/corpus $TARGETS/torgo_mfa_arpa_first_v2/first_pronunciation.dict english_us_arpa $TARGETS/torgo_mfa_arpa_first_v2/aligned --output_format json -t $TARGETS/torgo_mfa_arpa_first_v2/temp_align --clean
/home/rachel/.conda/envs/audio_codec/bin/python -m rvq_probes.prepare_boundary_targets --token-index $TOKENS/tokens.jsonl --alignment-dir $TARGETS/torgo_mfa_arpa_first_v2/aligned --output-dir $TARGETS/torgo_boundary_20ms_v2 --frame-duration 0.02 --allow-missing
```

3–8. One-layer commands use the corresponding trainer; all-layer/all-rotation commands use the sweep runner:

```bash
# P1 Q1 rotation 01
/home/rachel/.conda/envs/audio_codec/bin/python -m rvq_probes.train_phoneme_ctc --token-index $INDEX --token-root $TOKENS --phoneme-targets $TARGETS/torgo_arpabet_nostress_v1/phoneme_targets.jsonl --phoneme-vocabulary $TARGETS/torgo_arpabet_nostress_v1/phoneme_vocabulary.json --codec-config $CODEC/config.json --codec-checkpoint $CODEC/SpeechTokenizer.pt --output-dir runs/probes/phoneme_ctc/rotation_01/q1 --rvq-layer 1 --seed 1337 --device cuda
/home/rachel/.conda/envs/audio_codec/bin/python -m rvq_probes.run_probe_sweep --probe phoneme_ctc --rotations-root $ROTATIONS --token-root $TOKENS --phoneme-targets $TARGETS/torgo_arpabet_nostress_v1/phoneme_targets.jsonl --phoneme-vocabulary $TARGETS/torgo_arpabet_nostress_v1/phoneme_vocabulary.json --codec-config $CODEC/config.json --codec-checkpoint $CODEC/SpeechTokenizer.pt --output-root runs/probes/phoneme_ctc --seed 1337 --device cuda
# P2 Q1 rotation 01
/home/rachel/.conda/envs/audio_codec/bin/python -m rvq_probes.train_phoneme_boundary --token-index $INDEX --token-root $TOKENS --boundary-targets $TARGETS/torgo_boundary_20ms_v2/boundary_targets.jsonl --codec-config $CODEC/config.json --codec-checkpoint $CODEC/SpeechTokenizer.pt --output-dir runs/probes/phoneme_boundary/rotation_01/q1 --rvq-layer 1 --seed 1337 --device cuda
/home/rachel/.conda/envs/audio_codec/bin/python -m rvq_probes.run_probe_sweep --probe phoneme_boundary --rotations-root $ROTATIONS --token-root $TOKENS --boundary-targets $TARGETS/torgo_boundary_20ms_v2/boundary_targets.jsonl --codec-config $CODEC/config.json --codec-checkpoint $CODEC/SpeechTokenizer.pt --output-root runs/probes/phoneme_boundary --seed 1337 --device cuda
# P3 Q1 rotation 01
/home/rachel/.conda/envs/audio_codec/bin/python -m rvq_probes.train_dysarthria --token-index $INDEX --token-root $TOKENS --codec-config $CODEC/config.json --codec-checkpoint $CODEC/SpeechTokenizer.pt --output-dir runs/probes/dysarthria/rotation_01/q1 --rvq-layer 1 --seed 1337 --device cuda
/home/rachel/.conda/envs/audio_codec/bin/python -m rvq_probes.run_probe_sweep --probe dysarthria --rotations-root $ROTATIONS --token-root $TOKENS --codec-config $CODEC/config.json --codec-checkpoint $CODEC/SpeechTokenizer.pt --output-root runs/probes/dysarthria --seed 1337 --device cuda
```

9. Generate the information table and per-rotation table:

```bash
/home/rachel/.conda/envs/audio_codec/bin/python -m rvq_probes.summarize_probes --runs-root runs/probes --output-dir runs/probes
```

10. Run unit tests and repeat the documented smoke pattern with `--device cpu --epochs 1 --limit-per-split 4`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=04_Code /home/rachel/.conda/envs/audio_codec/bin/python -m unittest discover -s 04_Code/rvq_probes -p 'test_*.py'
```
