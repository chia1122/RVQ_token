#!/usr/bin/env python3
"""Extract Meta EnCodec RVQ tokens from a validated TORGO manifest."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    rows = []
    required = {"utt_id", "audio_path", "speaker_id", "severity", "split"}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise ValueError(f"Manifest line {line_number} is missing {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    utt_ids = [row["utt_id"] for row in rows]
    if len(utt_ids) != len(set(utt_ids)):
        raise ValueError("Manifest contains duplicate utt_id values")
    return rows


def resolve_audio(audio_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"audio_path must be a safe relative path: {relative_path!r}")
    root = audio_root.resolve()
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Audio path escapes audio root: {relative_path!r}") from exc
    return result


def token_relative_path(row: dict) -> Path:
    safe_utt_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["utt_id"])
    return Path(row["split"]) / row["speaker_id"] / f"{safe_utt_id}.pt"


def validate_codes(codes, codebook_size: int):
    import torch

    if codes.ndim != 3 or codes.shape[0] != 1:
        raise ValueError(f"Expected EnCodec codes [1, N, T], found {tuple(codes.shape)}")
    if codes.shape[1] < 1 or codes.shape[2] < 1:
        raise ValueError(f"Empty EnCodec output: {tuple(codes.shape)}")
    if codes.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"Expected integer token IDs, found {codes.dtype}")
    minimum = int(codes.min().item())
    maximum = int(codes.max().item())
    if minimum < 0 or maximum >= codebook_size:
        raise ValueError(
            f"Token IDs outside [0, {codebook_size - 1}]: min={minimum}, max={maximum}"
        )


def load_model(model_name: str, bandwidth: float, device: str):
    try:
        from encodec import EncodecModel
    except ImportError as exc:
        raise SystemExit("Missing dependency: install `encodec` before extracting tokens") from exc

    if model_name == "encodec_24khz":
        model = EncodecModel.encodec_model_24khz()
    elif model_name == "encodec_48khz":
        model = EncodecModel.encodec_model_48khz()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.set_target_bandwidth(bandwidth)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_audio(path: Path, model, device: str):
    try:
        import torchaudio
        from encodec.utils import convert_audio
    except ImportError as exc:
        raise SystemExit("Missing dependency: install compatible `torch` and `torchaudio`") from exc

    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise ValueError("Audio is empty")
    waveform = convert_audio(waveform, sample_rate, model.sample_rate, model.channels)
    return waveform.unsqueeze(0).to(device), sample_rate


def encode_waveform(model, waveform):
    import torch

    with torch.inference_mode():
        encoded_frames = model.encode(waveform)
    if not encoded_frames:
        raise ValueError("EnCodec returned no frames")
    codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
    return codes


def atomic_torch_save(payload: dict, destination: Path) -> None:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def read_existing_token(path: Path, expected_utt_id: str) -> dict | None:
    if not path.exists():
        return None
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"utt_id", "codes", "num_frames", "num_codebooks"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Incomplete existing token file: {path}")
    if payload["utt_id"] != expected_utt_id:
        raise ValueError(f"utt_id mismatch in existing token file: {path}")
    if tuple(payload["codes"].shape) != (payload["num_frames"], payload["num_codebooks"]):
        raise ValueError(f"Invalid [T, N] shape in existing token file: {path}")
    return payload


def extract(args: argparse.Namespace) -> None:
    import torch

    rows = load_manifest(args.manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")

    model = load_model(args.model, args.bandwidth, device)
    codebook_size = int(getattr(model.quantizer, "bins", 1024))
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
                    "severity": row["severity"],
                    "split": row["split"],
                    "text_norm": row.get("text_norm", ""),
                    "audio_path": row["audio_path"],
                    "source_sample_rate": source_sample_rate,
                    "codec_sample_rate": model.sample_rate,
                    "codec_model": args.model,
                    "bandwidth_kbps": args.bandwidth,
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
                "severity": row["severity"],
                "split": row["split"],
                "text_norm": row.get("text_norm", ""),
            })
        except Exception as exc:  # Continue so a long extraction run has a complete failure audit.
            failures.append({"utt_id": row["utt_id"], "error": f"{type(exc).__name__}: {exc}"})
            if not args.skip_errors:
                raise

        if item_number % args.log_every == 0 or item_number == len(rows):
            print(f"Processed {item_number}/{len(rows)}; saved={len(index_rows)}; failed={len(failures)}")

    if not index_rows:
        raise RuntimeError("No tokens were extracted")
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent codebook dimensions across outputs: {dict(dimensions)}")

    index_path = output_dir / "tokens.jsonl"
    with index_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    with (output_dir / "failures.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    (num_codebooks, observed_codebook_size), _ = dimensions.most_common(1)[0]
    summary = {
        "manifest": str(args.manifest.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "codec_model": args.model,
        "bandwidth_kbps": args.bandwidth,
        "codec_sample_rate": model.sample_rate,
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
    parser.add_argument("--model", choices=("encodec_24khz", "encodec_48khz"), default="encodec_24khz")
    parser.add_argument("--bandwidth", type=float, default=6.0, help="Target bandwidth in kbps")
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
