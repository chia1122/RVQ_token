#!/usr/bin/env python3
"""Extract SpeechTokenizer Q1-Q8 tokens from a validated TORGO manifest."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from extract_encodec_tokens import (
    atomic_torch_save,
    load_manifest,
    read_existing_token,
    resolve_audio,
    token_relative_path,
    validate_codes,
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


def load_audio(path: Path, sample_rate: int, device: str):
    try:
        import torchaudio
    except ImportError as exc:
        raise SystemExit("Install compatible torch and torchaudio") from exc
    waveform, source_sample_rate = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise ValueError("Audio is empty")
    if waveform.shape[0] > 1:
        waveform = waveform[:1]
    if source_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sample_rate, sample_rate)
    return waveform.unsqueeze(0).to(device), source_sample_rate


def extract(args: argparse.Namespace) -> None:
    import torch

    rows = load_manifest(args.manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    model = load_model(args.config, args.checkpoint, device)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_layers = int(config["n_q"])
    codebook_size = int(config["codebook_size"])
    hop_length = math.prod(int(stride) for stride in config["strides"])
    token_frame_rate = float(model.sample_rate / hop_length)
    index_rows, failures = [], []
    dimensions = Counter()

    for item_number, row in enumerate(rows, start=1):
        relative_token = token_relative_path(row)
        token_path = output_dir / relative_token
        try:
            existing = None if args.overwrite else read_existing_token(token_path, row["utt_id"])
            if existing is None:
                audio_path = resolve_audio(args.audio_root, row["audio_path"])
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Audio not found: {audio_path}")
                waveform, source_sample_rate = load_audio(audio_path, model.sample_rate, device)
                with torch.inference_mode():
                    codes_nbt = model.encode(waveform)
                if codes_nbt.ndim != 3:
                    raise ValueError(f"Expected SpeechTokenizer [N,B,T], found {tuple(codes_nbt.shape)}")
                codes_bnt = codes_nbt.permute(1, 0, 2).contiguous()
                validate_codes(codes_bnt, codebook_size)
                if codes_bnt.shape[1] != expected_layers:
                    raise ValueError(f"Expected {expected_layers} layers, found {codes_bnt.shape[1]}")
                codes_tn = codes_bnt[0].transpose(0, 1).contiguous().cpu()
                if codes_tn.max().item() <= 32767:
                    codes_tn = codes_tn.to(torch.int16)
                payload = {
                    "utt_id": row["utt_id"], "codes": codes_tn,
                    "shape_order": "T,N", "num_frames": int(codes_tn.shape[0]),
                    "num_codebooks": int(codes_tn.shape[1]), "codebook_size": codebook_size,
                    "speaker_id": row["speaker_id"], "condition": row["condition"],
                    "severity": row["severity"],
                    "split": row["split"], "text_norm": row.get("text_norm", ""),
                    "audio_path": row["audio_path"], "source_sample_rate": source_sample_rate,
                    "codec_sample_rate": int(model.sample_rate),
                    "codec_model": "speechtokenizer_hubert_avg",
                    "codec_checkpoint": str(args.checkpoint.resolve()),
                    "hop_length": hop_length, "token_frame_rate": token_frame_rate,
                }
                atomic_torch_save(payload, token_path)
            else:
                payload = existing
            if payload.get("codec_model") != "speechtokenizer_hubert_avg":
                raise ValueError(f"Existing token file is not SpeechTokenizer: {token_path}")
            dimensions[(payload["num_codebooks"], payload["codebook_size"])] += 1
            index_rows.append({
                "utt_id": row["utt_id"], "token_path": relative_token.as_posix(),
                "num_frames": payload["num_frames"], "num_codebooks": payload["num_codebooks"],
                "codebook_size": payload["codebook_size"], "speaker_id": row["speaker_id"],
                "condition": row["condition"], "severity": row["severity"],
                "split": row["split"],
                "text_norm": row.get("text_norm", ""),
            })
        except Exception as exc:
            failures.append({"utt_id": row["utt_id"], "error": f"{type(exc).__name__}: {exc}"})
            if not args.skip_errors:
                raise
        if item_number % args.log_every == 0 or item_number == len(rows):
            print(f"Processed {item_number}/{len(rows)}; saved={len(index_rows)}; failed={len(failures)}")

    if not index_rows or len(dimensions) != 1:
        raise RuntimeError(f"Invalid extraction result: saved={len(index_rows)}, dimensions={dict(dimensions)}")
    for filename, values in (("tokens.jsonl", index_rows), ("failures.jsonl", failures)):
        with (output_dir / filename).open("w", encoding="utf-8", newline="\n") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
    (num_codebooks, observed_size), _ = dimensions.most_common(1)[0]
    summary = {
        "manifest": str(args.manifest.resolve()), "audio_root": str(args.audio_root.resolve()),
        "codec_model": "speechtokenizer_hubert_avg",
        "codec_checkpoint": str(args.checkpoint.resolve()),
        "codec_sample_rate": int(model.sample_rate), "hop_length": hop_length,
        "token_frame_rate": token_frame_rate,
        "utterances_requested": len(rows), "utterances_saved": len(index_rows),
        "utterances_failed": len(failures), "num_codebooks": num_codebooks,
        "codebook_size": observed_size, "shape_order": "T,N", "device": device,
    }
    (output_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    return args


if __name__ == "__main__":
    extract(parse_args())
