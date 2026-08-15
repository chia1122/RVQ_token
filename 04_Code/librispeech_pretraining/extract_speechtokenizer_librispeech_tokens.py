#!/usr/bin/env python3
"""Extract SpeechTokenizer Q1-Q8 tokens from a LibriSpeech JSONL manifest.

Expected manifest fields (minimum):
    utt_id
    audio_path
    speaker_id
    split
    text_norm

Optional fields such as corpus, subset, severity, speaker_type, chapter_id,
audio_status, and text_raw are preserved in the token/index metadata when present.

Example manifest row:
{
  "audio_path": "dev-clean/1272/135031/1272-135031-0005.flac",
  "audio_status": "available",
  "chapter_id": "135031",
  "corpus": "LibriSpeech",
  "severity": "normal",
  "speaker_id": "1272",
  "speaker_type": "control",
  "split": "valid",
  "subset": "dev-clean",
  "text_norm": "I BEGGED RUGGEDO LONG AGO TO SEND HIM AWAY BUT HE WOULD NOT DO SO",
  "text_raw": "I BEGGED RUGGEDO LONG AGO TO SEND HIM AWAY BUT HE WOULD NOT DO SO",
  "utt_id": "1272-135031-0005"
}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_FIELDS = (
    "utt_id",
    "audio_path",
    "speaker_id",
    "split",
)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate a LibriSpeech JSONL manifest."""
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in row]
            if missing:
                raise ValueError(
                    f"Manifest row {line_number} is missing required fields: {missing}"
                )

            if row.get("audio_status") not in (None, "available"):
                # Keep unavailable rows out of extraction rather than failing later.
                continue

            rows.append(row)

    if not rows:
        raise ValueError(f"No usable manifest rows found in {path}")

    return rows


def resolve_audio(audio_root: Path, audio_path: str) -> Path:
    """Resolve relative manifest audio paths against --audio-root."""
    path = Path(audio_path)
    if path.is_absolute():
        return path
    return audio_root / path


def token_relative_path(row: dict[str, Any]) -> Path:
    """Create a stable token path that mirrors LibriSpeech speaker/chapter layout."""
    utt_id = str(row["utt_id"])
    speaker_id = str(row["speaker_id"])

    chapter_id = row.get("chapter_id")
    if chapter_id is None:
        # LibriSpeech utt_id normally follows speaker-chapter-utterance.
        parts = utt_id.split("-")
        chapter_id = parts[1] if len(parts) >= 2 else "unknown_chapter"

    return Path(str(speaker_id)) / str(chapter_id) / f"{utt_id}.pt"


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Safely save a torch payload without leaving partial files."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_existing_token(path: Path, expected_utt_id: str):
    """Load an existing token file and verify that it belongs to the same utterance."""
    if not path.is_file():
        return None

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("utt_id") != expected_utt_id:
        raise ValueError(
            f"Existing token utt_id mismatch: expected {expected_utt_id}, "
            f"found {payload.get('utt_id')} in {path}"
        )
    return payload


def validate_codes(codes_bnt, codebook_size: int) -> None:
    """Validate SpeechTokenizer codes in [B,N,T] format."""
    if codes_bnt.ndim != 3:
        raise ValueError(f"Expected [B,N,T] codes, found shape {tuple(codes_bnt.shape)}")

    if codes_bnt.numel() == 0:
        raise ValueError("SpeechTokenizer returned empty codes")

    min_code = int(codes_bnt.min().item())
    max_code = int(codes_bnt.max().item())

    if min_code < 0:
        raise ValueError(f"Negative token id found: {min_code}")

    if max_code >= codebook_size:
        raise ValueError(
            f"Token id {max_code} exceeds codebook size {codebook_size}"
        )


def load_model(config: Path, checkpoint: Path, device: str):
    try:
        from speechtokenizer import SpeechTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install the local SpeechTokenizer package before extraction:\n"
            "  python -m pip install -e SpeechTokenizer"
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

    # SpeechTokenizer expects mono input. LibriSpeech is normally mono already.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if source_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            source_sample_rate,
            sample_rate,
        )

    # [channels, time] -> [batch, channels, time]
    return waveform.unsqueeze(0).to(device), source_sample_rate


def extract(args: argparse.Namespace) -> None:
    import torch

    rows = load_manifest(args.manifest)

    if args.subsets:
        allowed_subsets = set(args.subsets.split(","))
        rows = [row for row in rows if row.get("subset") in allowed_subsets]
        if not rows:
            raise SystemExit(
                f"No rows remain after --subsets filter: {sorted(allowed_subsets)}"
            )

    if args.splits:
        allowed_splits = set(args.splits.split(","))
        rows = [row for row in rows if row.get("split") in allowed_splits]
        if not rows:
            raise SystemExit(
                f"No rows remain after --splits filter: {sorted(allowed_splits)}"
            )

    if args.limit is not None:
        rows = rows[: args.limit]

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

    index_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    dimensions = Counter()

    for item_number, row in enumerate(rows, start=1):
        relative_token = token_relative_path(row)
        token_path = output_dir / relative_token

        try:
            existing = (
                None
                if args.overwrite
                else read_existing_token(token_path, row["utt_id"])
            )

            if existing is None:
                audio_path = resolve_audio(args.audio_root, row["audio_path"])

                if not audio_path.is_file():
                    raise FileNotFoundError(f"Audio not found: {audio_path}")

                waveform, source_sample_rate = load_audio(
                    audio_path,
                    model.sample_rate,
                    device,
                )

                with torch.inference_mode():
                    codes_nbt = model.encode(waveform)

                # SpeechTokenizer returns [N,B,T].
                if codes_nbt.ndim != 3:
                    raise ValueError(
                        f"Expected SpeechTokenizer [N,B,T], "
                        f"found {tuple(codes_nbt.shape)}"
                    )

                codes_bnt = codes_nbt.permute(1, 0, 2).contiguous()
                validate_codes(codes_bnt, codebook_size)

                if codes_bnt.shape[1] != expected_layers:
                    raise ValueError(
                        f"Expected {expected_layers} layers, "
                        f"found {codes_bnt.shape[1]}"
                    )

                # Save common project format [T,N].
                codes_tn = codes_bnt[0].transpose(0, 1).contiguous().cpu()

                if int(codes_tn.max().item()) <= 32767:
                    codes_tn = codes_tn.to(torch.int16)

                payload = {
                    "utt_id": row["utt_id"],
                    "codes": codes_tn,
                    "shape_order": "T,N",
                    "num_frames": int(codes_tn.shape[0]),
                    "num_codebooks": int(codes_tn.shape[1]),
                    "codebook_size": codebook_size,
                    "speaker_id": row["speaker_id"],
                    "split": row["split"],
                    "subset": row.get("subset"),
                    "chapter_id": row.get("chapter_id"),
                    "corpus": row.get("corpus", "LibriSpeech"),
                    "speaker_type": row.get("speaker_type"),
                    "severity": row.get("severity", "normal"),
                    "text_norm": row.get("text_norm", ""),
                    "text_raw": row.get("text_raw", ""),
                    "audio_path": row["audio_path"],
                    "source_sample_rate": source_sample_rate,
                    "codec_sample_rate": int(model.sample_rate),
                    "codec_model": "speechtokenizer_hubert_avg",
                    "codec_checkpoint": str(args.checkpoint.resolve()),
                    "hop_length": hop_length,
                    "token_frame_rate": token_frame_rate,
                }

                atomic_torch_save(payload, token_path)
            else:
                payload = existing

            if payload.get("codec_model") != "speechtokenizer_hubert_avg":
                raise ValueError(
                    f"Existing token file is not SpeechTokenizer: {token_path}"
                )

            dimensions[
                (payload["num_codebooks"], payload["codebook_size"])
            ] += 1

            index_rows.append(
                {
                    "utt_id": row["utt_id"],
                    "token_path": relative_token.as_posix(),
                    "num_frames": payload["num_frames"],
                    "num_codebooks": payload["num_codebooks"],
                    "codebook_size": payload["codebook_size"],
                    "speaker_id": row["speaker_id"],
                    "chapter_id": row.get("chapter_id"),
                    "subset": row.get("subset"),
                    "split": row["split"],
                    "severity": row.get("severity", "normal"),
                    "speaker_type": row.get("speaker_type"),
                    "text_norm": row.get("text_norm", ""),
                    "audio_path": row["audio_path"],
                }
            )

        except Exception as exc:
            failures.append(
                {
                    "utt_id": str(row.get("utt_id", "")),
                    "audio_path": str(row.get("audio_path", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

            if not args.skip_errors:
                raise

        if item_number % args.log_every == 0 or item_number == len(rows):
            print(
                f"Processed {item_number}/{len(rows)}; "
                f"saved={len(index_rows)}; failed={len(failures)}"
            )

    if not index_rows:
        raise RuntimeError(
            f"No tokens were successfully extracted; failures={len(failures)}"
        )

    if len(dimensions) != 1:
        raise RuntimeError(
            f"Inconsistent token dimensions: {dict(dimensions)}"
        )

    for filename, values in (
        ("tokens.jsonl", index_rows),
        ("failures.jsonl", failures),
    ):
        with (output_dir / filename).open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            for value in values:
                handle.write(
                    json.dumps(
                        value,
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    + "\n"
                )

    (num_codebooks, observed_size), _ = dimensions.most_common(1)[0]

    summary = {
        "manifest": str(args.manifest.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "corpus": "LibriSpeech",
        "codec_model": "speechtokenizer_hubert_avg",
        "codec_checkpoint": str(args.checkpoint.resolve()),
        "codec_sample_rate": int(model.sample_rate),
        "hop_length": hop_length,
        "token_frame_rate": token_frame_rate,
        "utterances_requested": len(rows),
        "utterances_saved": len(index_rows),
        "utterances_failed": len(failures),
        "num_codebooks": num_codebooks,
        "codebook_size": observed_size,
        "shape_order": "T,N",
        "device": device,
        "subsets_filter": args.subsets,
        "splits_filter": args.splits,
        "limit": args.limit,
    }

    (output_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
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

    # Useful for LibriSpeech smoke tests and subset-specific extraction.
    parser.add_argument(
        "--subsets",
        help="Comma-separated LibriSpeech subsets, e.g. train-clean-100,dev-clean",
    )
    parser.add_argument(
        "--splits",
        help="Comma-separated manifest splits, e.g. train,valid,test",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N selected utterances (smoke testing)",
    )

    args = parser.parse_args()

    if args.log_every < 1:
        parser.error("--log-every must be positive")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    return args


if __name__ == "__main__":
    extract(parse_args())
