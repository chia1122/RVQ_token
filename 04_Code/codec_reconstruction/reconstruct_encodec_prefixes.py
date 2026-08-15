#!/usr/bin/env python3
"""Decode Q1:QK prefixes from saved EnCodec tokens into WAV files."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def parse_layers(value: str) -> list[int]:
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers or layers[0] < 1:
        raise ValueError("--layers must contain positive comma-separated integers")
    return layers


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_token(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(model_name: str, bandwidth: float, device: str):
    try:
        from encodec import EncodecModel
    except ImportError as exc:
        raise SystemExit("Install the `encodec` package before reconstruction") from exc
    if model_name == "encodec_24khz":
        model = EncodecModel.encodec_model_24khz()
    elif model_name == "encodec_48khz":
        model = EncodecModel.encodec_model_48khz()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.set_target_bandwidth(bandwidth)
    if getattr(model, "normalize", False):
        raise ValueError(
            "This checkpoint uses waveform normalization, but saved token files do not contain "
            "per-frame scales. Re-extract tokens with scales before reconstruction."
        )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def original_output_samples(audio_path: Path, codec_sample_rate: int) -> int:
    import torchaudio

    info = torchaudio.info(str(audio_path))
    return round(info.num_frames * codec_sample_rate / info.sample_rate)


def decode_prefix(model, codes_tn, num_layers: int, device: str):
    import torch

    if codes_tn.ndim != 2 or codes_tn.shape[1] < num_layers:
        raise ValueError(f"Cannot decode K={num_layers} from shape {tuple(codes_tn.shape)}")
    codes_bnt = codes_tn[:, :num_layers].transpose(0, 1).unsqueeze(0).long().to(device)
    with torch.inference_mode():
        quantized = model.quantizer.decode(codes_bnt)
        waveform = model.decoder(quantized)
    return waveform[0].detach().cpu()


def reconstruct(args: argparse.Namespace) -> None:
    import torch
    import torchaudio

    layers = parse_layers(args.layers)
    token_rows = load_jsonl(args.token_index)
    manifest_rows = {row["utt_id"]: row for row in load_jsonl(args.manifest)}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.bandwidth, device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    failures = []

    selected = [row for row in token_rows if args.split == "all" or row["split"] == args.split]
    if args.limit:
        selected = selected[: args.limit]
    for item_number, token_row in enumerate(selected, start=1):
        utt_id = token_row["utt_id"]
        try:
            manifest = manifest_rows[utt_id]
            token_path = (args.token_root / token_row["token_path"]).resolve()
            payload = load_token(token_path)
            if payload.get("codec_model") != args.model or float(payload.get("bandwidth_kbps")) != args.bandwidth:
                raise ValueError(f"Codec metadata mismatch in {token_path}")
            source_audio = (args.audio_root / manifest["audio_path"]).resolve()
            target_samples = original_output_samples(source_audio, model.sample_rate)
            for num_layers in layers:
                relative = Path(f"k{num_layers}") / manifest["split"] / manifest["speaker_id"] / f"{safe_name(utt_id)}.wav"
                destination = output_dir / relative
                if not destination.is_file() or args.overwrite:
                    waveform = decode_prefix(model, payload["codes"], num_layers, device)
                    waveform = waveform[:, :target_samples]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(f"{destination.stem}.tmp.wav")
                    torchaudio.save(str(temporary), waveform, model.sample_rate)
                    os.replace(temporary, destination)
                output_rows.append({
                    "condition": f"k{num_layers}",
                    "num_rvq_layers": num_layers,
                    "utt_id": utt_id,
                    "audio_path": relative.as_posix(),
                    "speaker_id": manifest["speaker_id"],
                    "severity": manifest["severity"],
                    "split": manifest["split"],
                    "text_norm": manifest["text_norm"],
                    "sample_rate": model.sample_rate,
                })
        except Exception as exc:
            failures.append({"utt_id": utt_id, "error": f"{type(exc).__name__}: {exc}"})
            if not args.skip_errors:
                raise
        if item_number % args.log_every == 0 or item_number == len(selected):
            print(f"Processed {item_number}/{len(selected)}; wavs={len(output_rows)}; failures={len(failures)}")

    with (output_dir / "reconstruction_index.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    with (output_dir / "failures.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    summary = {
        "utterances": len(selected), "layers": layers,
        "wav_files": len(output_rows), "failures": len(failures),
        "codec_model": args.model, "bandwidth_kbps": args.bandwidth,
        "sample_rate": model.sample_rate, "split": args.split,
    }
    (output_dir / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-index", type=Path, required=True)
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="1,2,4,6,8")
    parser.add_argument("--split", choices=("all", "train", "valid", "test"), default="all")
    parser.add_argument("--model", choices=("encodec_24khz", "encodec_48khz"), default="encodec_24khz")
    parser.add_argument("--bandwidth", type=float, default=6.0)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="Limit utterances for a smoke test")
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    try:
        parse_layers(args.layers)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    reconstruct(parse_args())
