#!/usr/bin/env python3
"""Decode Q1:QK prefixes from saved DAC tokens into WAV files."""

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


def load_model(model_name: str, device: str):
    try:
        import dac
    except ImportError as exc:
        raise SystemExit("Install `descript-audio-codec` before reconstruction") from exc
    checkpoint = dac.utils.download(model_type=model_name)
    model = dac.DAC.load(checkpoint)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, str(checkpoint)


def prefix_latent_from_codes(model, codes_bnt):
    output = model.quantizer.from_codes(codes_bnt)
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("DAC quantizer.from_codes returned an empty tuple")
        return output[0]
    return output


def decode_prefix(model, codes_tn, num_layers: int, device: str):
    import torch

    if codes_tn.ndim != 2 or codes_tn.shape[1] < num_layers:
        raise ValueError(f"Cannot decode K={num_layers} from shape {tuple(codes_tn.shape)}")
    codes_bnt = codes_tn[:, :num_layers].transpose(0, 1).unsqueeze(0).long().to(device)
    with torch.inference_mode():
        latent = prefix_latent_from_codes(model, codes_bnt)
        waveform = model.decode(latent)
    return waveform[0].detach().cpu()


def reconstruct(args: argparse.Namespace) -> None:
    import torch
    import torchaudio

    layers = parse_layers(args.layers)
    token_rows = load_jsonl(args.token_index)
    manifest_rows = {row["utt_id"]: row for row in load_jsonl(args.manifest)}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.model, device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows, failures = [], []

    requested_speakers = {
        speaker.strip() for speaker in args.speakers.split(",") if speaker.strip()
    }
    selected = [
        row for row in token_rows
        if (args.split == "all" or row["split"] == args.split)
        and (not requested_speakers or row["speaker_id"] in requested_speakers)
    ]
    if not selected:
        raise ValueError("No DAC token rows matched --split/--speakers")
    if args.limit:
        selected = selected[: args.limit]
    for item_number, token_row in enumerate(selected, start=1):
        utt_id = token_row["utt_id"]
        try:
            manifest = manifest_rows[utt_id]
            token_path = (args.token_root / token_row["token_path"]).resolve()
            payload = load_token(token_path)
            if payload.get("codec_model") != f"dac_{args.model}":
                raise ValueError(f"DAC model metadata mismatch in {token_path}")
            validate_requested_layers(payload, layers, token_path)
            source_audio = (args.audio_root / manifest["audio_path"]).resolve()
            target_samples = original_output_samples(source_audio, int(model.sample_rate))
            for num_layers in layers:
                relative = Path(f"k{num_layers}") / manifest["split"] / manifest["speaker_id"] / f"{safe_name(utt_id)}.wav"
                destination = output_dir / relative
                if not destination.is_file() or args.overwrite:
                    waveform = decode_prefix(model, payload["codes"], num_layers, device)
                    waveform = waveform[:, :target_samples]
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
        "codec_model": f"dac_{args.model}", "checkpoint": checkpoint,
        "sample_rate": int(model.sample_rate), "split": args.split,
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
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--split", choices=("all", "train", "valid", "test"), default="all")
    parser.add_argument("--speakers", default="", help="Optional comma-separated speaker IDs")
    parser.add_argument("--model", choices=("16khz", "24khz", "44khz"), default="24khz")
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.log_every < 1 or args.limit < 0:
        parser.error("--log-every must be positive and --limit cannot be negative")
    try:
        parse_layers(args.layers)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    reconstruct(parse_args())
