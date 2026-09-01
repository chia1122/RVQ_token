#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from rvq_probes.metrics import phoneme_error_metrics
from rvq_probes.phoneme_ctc import PhonemeCollator, PhonemeCTCProbe, PhonemeDataset
from rvq_probes.phonemes import PhonemeTokenizer
from rvq_probes.representation import load_speechtokenizer_codebook
from rvq_probes.splits import load_index, validate_speaker_disjoint


def set_seed(seed):
    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def move(batch, device):
    for key in ("representations", "input_lengths", "targets", "target_lengths"):
        batch[key] = batch[key].to(device)
    return batch


def ctc_loss(criterion, logits, batch, lengths):
    return criterion(
        logits.log_softmax(-1).transpose(0, 1), batch["targets"],
        lengths, batch["target_lengths"],
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = move(batch, device)
        logits, lengths = model(batch["representations"], batch["input_lengths"])
        if torch.any(lengths < batch["target_lengths"]):
            raise ValueError("CTC input length is shorter than target length")
        loss = ctc_loss(criterion, logits, batch, lengths)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.item())
    return total / len(loader)


@torch.inference_mode()
def evaluate(model, loader, criterion, tokenizer, device, predictions_path=None):
    model.eval()
    total_loss = 0.0
    totals = {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_phonemes": 0}
    blank_frames = valid_frames = empty = utterances = 0
    predictions = []
    for batch in loader:
        batch = move(batch, device)
        logits, lengths = model(batch["representations"], batch["input_lengths"])
        total_loss += float(ctc_loss(criterion, logits, batch, lengths).item())
        ids = logits.argmax(-1).cpu()
        for index, length in enumerate(lengths.cpu().tolist()):
            frame_ids = ids[index, :length].tolist()
            hypothesis = tokenizer.decode_ctc(frame_ids)
            reference = batch["phonemes"][index]
            metric = phoneme_error_metrics(reference, hypothesis)
            for key in totals:
                totals[key] += metric[key]
            blank_frames += sum(value == tokenizer.blank_id for value in frame_ids)
            valid_frames += len(frame_ids)
            empty += int(not hypothesis)
            utterances += 1
            row = batch["rows"][index]
            predictions.append({
                "utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
                "condition": row["condition"], "severity": row["severity"],
                "reference_phonemes": reference, "predicted_phonemes": hypothesis,
                **metric,
            })
    if predictions_path:
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    denominator = totals["reference_phonemes"]
    return {
        "loss": total_loss / len(loader),
        **totals,
        "per": (
            totals["substitutions"] + totals["deletions"] + totals["insertions"]
        ) / denominator if denominator else 0.0,
        "ctc_blank_frame_ratio": blank_frames / valid_frames if valid_frames else 0.0,
        "empty_prediction_ratio": empty / utterances if utterances else 0.0,
        "utterances": utterances,
    }


def make_loader(dataset, tokenizer, args, shuffle=False):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, collate_fn=PhonemeCollator(tokenizer),
        generator=generator,
    )


def main(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    rows = load_index(args.token_index)
    split_summary = validate_speaker_disjoint(rows)
    vocabulary = json.loads(args.phoneme_vocabulary.read_text())
    tokenizer = PhonemeTokenizer(vocabulary)
    codebook, codec_metadata = load_speechtokenizer_codebook(
        args.codec_config, args.codec_checkpoint, args.rvq_layer
    )
    datasets = {
        split: PhonemeDataset(
            rows, args.token_root, args.phoneme_targets, split,
            args.rvq_layer, codebook, args.limit_per_split,
        )
        for split in ("train", "valid", "test")
    }
    loaders = {
        split: make_loader(datasets[split], tokenizer, args, shuffle=split == "train")
        for split in datasets
    }
    model = PhonemeCTCProbe(
        codec_metadata["embedding_dim"], len(tokenizer), args.bottleneck_dim,
        args.temporal_layers, args.dropout,
    ).to(args.device)
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history, best_per = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, criterion, args.device)
        valid = evaluate(model, loaders["valid"], criterion, tokenizer, args.device)
        record = {"epoch": epoch, "train_loss": train_loss, "valid": valid}
        history.append(record)
        print(json.dumps(record))
        if valid["per"] < best_per:
            best_per = valid["per"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "valid": valid},
                args.output_dir / "best.pt",
            )
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(
        model, loaders["test"], criterion, tokenizer, args.device,
        args.output_dir / "predictions.jsonl",
    )
    config = {
        **vars(args), "git_commit": git_commit(), "codec": codec_metadata,
        "model": {
            "architecture": "layernorm_bottleneck_temporal_conv_linear_ctc",
            "bottleneck_dim": args.bottleneck_dim,
            "temporal_layers": args.temporal_layers,
            "dropout": args.dropout,
        },
        "codec_frozen": True, "only_probe_parameters_optimized": True,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    results = {
        "probe_name": "phoneme_ctc", "rvq_layer": args.rvq_layer,
        "codec": codec_metadata["codec"], "seed": args.seed,
        "train_speakers": split_summary["train"]["speakers"],
        "valid_speakers": split_summary["valid"]["speakers"],
        "test_speakers": split_summary["test"]["speakers"],
        "train_utterances": len(datasets["train"]),
        "valid_utterances": len(datasets["valid"]),
        "test_utterances": len(datasets["test"]),
        "main_metric": "per", "main_metric_value": test["per"],
        "best_epoch": checkpoint["epoch"], "test": test,
        "split_summary": split_summary,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen individual-RVQ phoneme CTC probe")
    parser.add_argument("--token-index", type=Path, required=True)
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--phoneme-targets", type=Path, required=True)
    parser.add_argument("--phoneme-vocabulary", type=Path, required=True)
    parser.add_argument("--codec-config", type=Path, required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rvq-layer", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--limit-per-split", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
