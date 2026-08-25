#!/usr/bin/env python3
"""Encode or load SpeechTokenizer codes and decode Q1:QK prefixes to WAV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from reconstruct_encodec_prefixes import (
    DEFAULT_LAYERS,
    load_jsonl,
    load_token,
    original_output_samples,
    parse_layers,
    safe_name,
    speech_condition,
    validate_requested_layers,
)


def load_model(config: Path, checkpoint: Path, device: str):
    try:
        from speechtokenizer import SpeechTokenizer
    except ImportError as exc:
        raise SystemExit(
            f"SpeechTokenizer import failed: {exc}. The upstream package may require "
            "optional trainer dependencies such as beartype and tensorboard."
        ) from exc
    model = SpeechTokenizer.load_from_checkpoint(str(config), str(checkpoint))
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def decode_prefix(model, codes_tn, num_layers: int, device: str):
    import torch

    if codes_tn.ndim != 2 or codes_tn.shape[1] < num_layers:
        raise ValueError(f"Cannot decode K={num_layers} from {tuple(codes_tn.shape)}")
    codes_nbt = codes_tn[:, :num_layers].transpose(0, 1).unsqueeze(1).long().to(device)
    with torch.inference_mode():
        waveform = model.decode(codes_nbt)
    return waveform[0].detach().cpu()


def validate_input_mode(
    input_mode: str,
    token_index: Path | None,
    token_root: Path | None,
) -> None:
    """Validate inputs without importing torch or SpeechTokenizer."""
    if input_mode == "tokens" and (token_index is None or token_root is None):
        raise ValueError("--input-mode tokens requires --token-index and --token-root")


def select_rows(
    rows: list[dict],
    split: str,
    speakers: set[str],
    limit: int,
) -> list[dict]:
    selected = [
        row
        for row in rows
        if (split == "all" or row["split"] == split)
        and (not speakers or row["speaker_id"] in speakers)
    ]
    return selected[:limit] if limit else selected


def encode_audio(model, audio_path: Path, device: str):
    """Encode one waveform and return transient SpeechTokenizer codes [T,N]."""
    import torch
    import torchaudio

    waveform, source_sample_rate = torchaudio.load(str(audio_path))
    if waveform.numel() == 0:
        raise ValueError(f"Audio is empty: {audio_path}")
    if waveform.shape[0] > 1:
        waveform = waveform[:1]
    if source_sample_rate != int(model.sample_rate):
        waveform = torchaudio.functional.resample(
            waveform, source_sample_rate, int(model.sample_rate)
        )
    with torch.inference_mode():
        codes_nbt = model.encode(waveform.unsqueeze(0).to(device))
    if codes_nbt.ndim != 3 or codes_nbt.shape[1] != 1:
        raise ValueError(
            "Expected SpeechTokenizer codes [N,1,T], found "
            f"{tuple(codes_nbt.shape)}"
        )
    return codes_nbt[:, 0, :].transpose(0, 1).contiguous().cpu()


def reconstruct(args: argparse.Namespace) -> None:
    import torch
    import torchaudio

    layers = parse_layers(args.layers)
    validate_input_mode(args.input_mode, args.token_index, args.token_root)
    manifest_rows = load_jsonl(args.manifest)
    manifests = {row["utt_id"]: row for row in manifest_rows}
    speakers = {item.strip() for item in args.speakers.split(",") if item.strip()}
    source_rows = (
        load_jsonl(args.token_index) if args.input_mode == "tokens" else manifest_rows
    )
    selected = select_rows(source_rows, args.split, speakers, args.limit)
    if not selected:
        raise ValueError(f"No {args.input_mode} rows matched split/speakers")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.config, args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows, failures = [], []
    for index, source_row in enumerate(selected, start=1):
        utt_id = source_row["utt_id"]
        try:
            manifest = manifests[utt_id]
            source_audio = args.audio_root / manifest["audio_path"]
            if not source_audio.is_file():
                raise FileNotFoundError(f"Audio not found: {source_audio}")
            if args.input_mode == "tokens":
                token_path = args.token_root / source_row["token_path"]
                payload = load_token(token_path)
                if payload.get("codec_model") != "speechtokenizer_hubert_avg":
                    raise ValueError(f"Codec metadata mismatch for {utt_id}")
                validate_requested_layers(payload, layers, token_path)
                codes_tn = payload["codes"]
            else:
                codes_tn = encode_audio(model, source_audio, device)
                payload = {
                    "codes": codes_tn,
                    "num_codebooks": int(codes_tn.shape[1]),
                }
                validate_requested_layers(payload, layers, source_audio)
            target_samples = original_output_samples(source_audio, int(model.sample_rate))
            for num_layers in layers:
                relative = Path(f"k{num_layers}") / manifest["split"] / manifest["speaker_id"] / f"{safe_name(utt_id)}.wav"
                destination = args.output_dir / relative
                if not destination.is_file() or args.overwrite:
                    waveform = decode_prefix(model, codes_tn, num_layers, device)[:, :target_samples]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(f"{destination.stem}.tmp.wav")
                    torchaudio.save(str(temporary), waveform, int(model.sample_rate))
                    os.replace(temporary, destination)
                output_rows.append({
                    "condition": speech_condition(manifest),
                    "rvq_condition": f"k{num_layers}", "num_rvq_layers": num_layers,
                    "utt_id": utt_id, "audio_path": relative.as_posix(),
                    "speaker_id": manifest["speaker_id"], "severity": manifest["severity"],
                    "split": manifest["split"], "text_norm": manifest["text_norm"],
                    "sample_rate": int(model.sample_rate),
                })
        except Exception as exc:
            failures.append({"utt_id": utt_id, "error": f"{type(exc).__name__}: {exc}"})
            if not args.skip_errors:
                raise
        if index % args.log_every == 0 or index == len(selected):
            print(f"Processed {index}/{len(selected)}; wavs={len(output_rows)}; failures={len(failures)}")
    for filename, values in (("reconstruction_index.jsonl", output_rows), ("failures.jsonl", failures)):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="\n") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
    summary = {
        "utterances": len(selected), "layers": layers, "wav_files": len(output_rows),
        "failures": len(failures), "codec_model": "speechtokenizer_hubert_avg",
        "sample_rate": int(model.sample_rate), "split": args.split,
        "input_mode": args.input_mode,
    }
    (args.output_dir / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-mode", choices=("tokens", "audio"), default="tokens",
        help="Load saved codes (default) or encode manifest audio in memory.",
    )
    parser.add_argument("--token-index", type=Path)
    parser.add_argument("--token-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--split", choices=("all", "train", "valid", "test"), default="all")
    parser.add_argument("--speakers", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.limit < 0 or args.log_every < 1:
        parser.error("Invalid --limit/--log-every")
    try:
        parse_layers(args.layers)
        validate_input_mode(args.input_mode, args.token_index, args.token_root)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    reconstruct(parse_args())
