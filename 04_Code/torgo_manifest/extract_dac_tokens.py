#!/usr/bin/env python3
"""Extract Descript Audio Codec (DAC) RVQ tokens from a TORGO manifest."""

from __future__ import annotations

import argparse
import json
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


def load_model(model_name: str, device: str):
    try:
        import dac
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install the `descript-audio-codec` package"
        ) from exc

    checkpoint = dac.utils.download(model_type=model_name)
    model = dac.DAC.load(checkpoint)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, str(checkpoint)


def infer_codebook_size(model) -> int:
    quantizer = model.quantizer
    value = getattr(quantizer, "codebook_size", None)
    if value is None and getattr(quantizer, "quantizers", None):
        value = getattr(quantizer.quantizers[0], "codebook_size", None)
    if value is None:
        raise ValueError("Could not infer DAC codebook size from the loaded checkpoint")
    return int(value)


def load_audio(path: Path, model, device: str):
    try:
        import torchaudio
        from torchaudio.functional import resample
    except ImportError as exc:
        raise SystemExit("Missing dependency: install compatible `torch` and `torchaudio`") from exc

    waveform, source_sample_rate = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise ValueError("Audio is empty")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_sample_rate != model.sample_rate:
        waveform = resample(waveform, source_sample_rate, model.sample_rate)
    waveform = waveform.unsqueeze(0).to(device)  # [B, C, samples]
    return model.preprocess(waveform, model.sample_rate), source_sample_rate


def encode_waveform(model, waveform):
    import torch

    with torch.inference_mode():
        output = model.encode(waveform)
    return codes_from_encode_output(output)


def codes_from_encode_output(output):
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise ValueError("Unexpected DAC encode output; expected tuple with codes at index 1")
    return output[1]


def extract(args: argparse.Namespace) -> None:
    import torch

    rows = load_manifest(args.manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")

    model, checkpoint = load_model(args.model, device)
    codebook_size = infer_codebook_size(model)
    index_rows = []
    failures = []
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
                waveform, source_sample_rate = load_audio(audio_path, model, device)
                codes_bnt = encode_waveform(model, waveform)
                validate_codes(codes_bnt, codebook_size)
                codes_tn = codes_bnt[0].transpose(0, 1).contiguous().cpu()
                if codes_tn.max().item() <= 32767:
                    codes_tn = codes_tn.to(torch.int16)
                payload = {
                    "utt_id": row["utt_id"],
                    "codes": codes_tn,
                    "shape_order": "T,N",
                    "num_frames": int(codes_tn.shape[0]),
                    "num_codebooks": int(codes_tn.shape[1]),
                    "codebook_size": codebook_size,
                    "speaker_id": row["speaker_id"],
                    "condition": row["condition"],
                    "severity": row["severity"],
                    "split": row["split"],
                    "text_norm": row.get("text_norm", ""),
                    "audio_path": row["audio_path"],
                    "source_sample_rate": source_sample_rate,
                    "codec_sample_rate": int(model.sample_rate),
                    "codec_model": f"dac_{args.model}",
                    "codec_checkpoint": checkpoint,
                    "hop_length": int(model.hop_length),
                    "token_frame_rate": float(model.sample_rate / model.hop_length),
                }
                atomic_torch_save(payload, token_path)
            else:
                payload = existing

            dimensions[(payload["num_codebooks"], payload["codebook_size"])] += 1
            index_rows.append({
                "utt_id": row["utt_id"],
                "token_path": relative_token.as_posix(),
                "num_frames": payload["num_frames"],
                "num_codebooks": payload["num_codebooks"],
                "codebook_size": payload["codebook_size"],
                "speaker_id": row["speaker_id"],
                "condition": row["condition"],
                "severity": row["severity"],
                "split": row["split"],
                "text_norm": row.get("text_norm", ""),
            })
        except Exception as exc:  # Preserve a complete audit when --skip-errors is used.
            failures.append({"utt_id": row["utt_id"], "error": f"{type(exc).__name__}: {exc}"})
            if not args.skip_errors:
                raise

        if item_number % args.log_every == 0 or item_number == len(rows):
            print(f"Processed {item_number}/{len(rows)}; saved={len(index_rows)}; failed={len(failures)}")

    if not index_rows:
        raise RuntimeError("No tokens were extracted")
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent codebook dimensions across outputs: {dict(dimensions)}")

    with (output_dir / "tokens.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    with (output_dir / "failures.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    (num_codebooks, observed_codebook_size), _ = dimensions.most_common(1)[0]
    summary = {
        "manifest": str(args.manifest.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "codec_model": f"dac_{args.model}",
        "codec_checkpoint": checkpoint,
        "codec_sample_rate": int(model.sample_rate),
        "hop_length": int(model.hop_length),
        "token_frame_rate": float(model.sample_rate / model.hop_length),
        "device": device,
        "utterances_requested": len(rows),
        "utterances_saved": len(index_rows),
        "utterances_failed": len(failures),
        "num_codebooks": num_codebooks,
        "codebook_size": observed_codebook_size,
        "shape_order": "T,N",
    }
    (output_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("16khz", "24khz", "44khz"), default="24khz")
    parser.add_argument("--device", help="For example: cuda, cuda:1, or cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    return args


if __name__ == "__main__":
    extract(parse_args())
