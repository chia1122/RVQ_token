#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from rvq_asr.data import CTCBatchCollator, RVQTokenDataset
from rvq_asr.model import RVQTransformerCTC
from rvq_asr.text import CharacterTokenizer, ErrorRate, prediction_row


def read_index_dimensions(path: Path) -> tuple[int, int]:
    dimensions = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                dimensions.add((int(row["codebook_size"]), int(row["num_codebooks"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid token dimensions on index line {line_number}") from exc
    if not dimensions:
        raise ValueError("Token index is empty")
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent token dimensions in index: {sorted(dimensions)}")
    return next(iter(dimensions))


def parse_active_layers(value: str) -> list[int]:
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("active RVQ layers must be comma-separated integers") from exc
    if not layers or min(layers) < 1 or len(set(layers)) != len(layers):
        raise ValueError("active RVQ layers must be unique positive integers")
    return layers


def resolve_representation_metadata(
    num_rvq_layers: int, active_layers: list[int], requested_mode: str | None,
    condition_name: str | None, layer_fusion: str,
) -> dict:
    cumulative_layers = list(range(1, num_rvq_layers + 1))
    inferred_mode = (
        "cumulative" if active_layers == cumulative_layers
        else "individual" if len(active_layers) == 1 else "custom"
    )
    rvq_mode = requested_mode or inferred_mode
    if rvq_mode == "cumulative" and active_layers != cumulative_layers:
        raise ValueError("cumulative rvq_mode requires active layers Q1 through QK")
    if rvq_mode == "individual" and len(active_layers) != 1:
        raise ValueError("individual rvq_mode requires exactly one active layer")
    expected_condition = (
        f"individual_q{active_layers[0]}" if rvq_mode == "individual"
        else "cumulative_q1" if rvq_mode == "cumulative" and num_rvq_layers == 1
        else f"cumulative_q1_{num_rvq_layers}" if rvq_mode == "cumulative"
        else "custom_" + "_".join(f"q{layer}" for layer in active_layers)
    )
    if condition_name and rvq_mode != "custom" and condition_name != expected_condition:
        raise ValueError(
            f"condition_name={condition_name!r} does not match {expected_condition!r}"
        )
    condition = condition_name or expected_condition
    return {
        "rvq_mode": rvq_mode,
        "condition": condition,
        "effective_fusion": (
            "single_active_layer" if len(active_layers) == 1
            else "learned_weighted_sum" if layer_fusion == "learned"
            else "sqrt_normalized_sum"
        ),
    }


def set_seed(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_dataset(args, split, tokenizer):
    return RVQTokenDataset(
        args.token_index, args.token_root, split, args.num_rvq_layers, tokenizer
    )


def make_loader(args, dataset, codebook_size, shuffle=False):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=CTCBatchCollator(codebook_size),
        generator=generator,
    )


def move_batch(batch, device):
    for key in ("codes", "input_lengths", "targets", "target_lengths"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def compute_ctc_loss(criterion, logits, batch, output_lengths, loss_device):
    log_probs = logits.log_softmax(-1).transpose(0, 1)
    if loss_device == "cpu":
        log_probs = log_probs.cpu()
        targets = batch["targets"].cpu()
        output_lengths = output_lengths.cpu()
        target_lengths = batch["target_lengths"].cpu()
    else:
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
    return criterion(log_probs, targets, output_lengths, target_lengths)


def train_epoch(
    model, loader, optimizer, criterion, device, grad_clip, grad_accum_steps,
    ctc_loss_device,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        batch = move_batch(batch, device)
        logits, output_lengths = model(batch["codes"], batch["input_lengths"])
        if torch.any(output_lengths < batch["target_lengths"]):
            raise ValueError("CTC target is longer than time-reduced encoder output")
        loss = compute_ctc_loss(
            criterion, logits, batch, output_lengths, ctc_loss_device
        )
        group_start = (batch_index // grad_accum_steps) * grad_accum_steps
        group_size = min(grad_accum_steps, len(loader) - group_start)
        (loss / group_size).backward()
        if (batch_index + 1) % grad_accum_steps == 0 or batch_index + 1 == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.item())
    return total_loss / len(loader)


@torch.inference_mode()
def evaluate(
    model, loader, criterion, tokenizer, device,
    predictions_path=None, frame_output_dir=None, ctc_loss_device="cuda",
):
    model.eval()
    total_loss = 0.0
    overall = ErrorRate()
    overall_cer = ErrorRate()
    by_severity = defaultdict(ErrorRate)
    cer_by_severity = defaultdict(ErrorRate)
    by_condition = defaultdict(ErrorRate)
    cer_by_condition = defaultdict(ErrorRate)
    by_speaker = defaultdict(ErrorRate)
    cer_by_speaker = defaultdict(ErrorRate)
    predictions = []
    empty_hypotheses = 0
    reference_characters = 0
    hypothesis_characters = 0
    blank_frames = 0
    valid_frames = 0
    diagnostics = defaultdict(lambda: {
        "utterances": 0, "empty_hypotheses": 0, "blank_frames": 0,
        "valid_frames": 0, "reference_characters": 0,
        "hypothesis_characters": 0,
    })
    for batch in loader:
        batch = move_batch(batch, device)
        logits, output_lengths = model(batch["codes"], batch["input_lengths"])
        if torch.any(output_lengths < batch["target_lengths"]):
            raise ValueError("CTC target is longer than time-reduced encoder output")
        total_loss += float(compute_ctc_loss(
            criterion, logits, batch, output_lengths, ctc_loss_device
        ).item())
        predicted_ids = logits.argmax(-1).cpu()
        for index, length in enumerate(output_lengths.cpu().tolist()):
            frame_ids = predicted_ids[index, :length].tolist()
            hypothesis = tokenizer.decode_ctc(frame_ids)
            reference = batch["texts"][index]
            severity = batch["severities"][index]
            speaker = batch["speaker_ids"][index]
            condition = batch["conditions"][index]
            overall.update_words(reference, hypothesis)
            overall_cer.update_characters(reference, hypothesis)
            by_condition[condition].update_words(reference, hypothesis)
            cer_by_condition[condition].update_characters(reference, hypothesis)
            by_severity[severity].update_words(reference, hypothesis)
            cer_by_severity[severity].update_characters(reference, hypothesis)
            by_speaker[speaker].update_words(reference, hypothesis)
            cer_by_speaker[speaker].update_characters(reference, hypothesis)
            empty_hypotheses += int(not hypothesis)
            reference_characters += len(reference.replace(" ", ""))
            hypothesis_characters += len(hypothesis.replace(" ", ""))
            blank_frames += sum(token_id == tokenizer.blank_id for token_id in frame_ids)
            valid_frames += len(frame_ids)
            group_keys = (
                ("overall", "all"), ("condition", condition),
                ("severity", severity), ("speaker", speaker),
            )
            for group_key in group_keys:
                diagnostic = diagnostics[group_key]
                diagnostic["utterances"] += 1
                diagnostic["empty_hypotheses"] += int(not hypothesis)
                diagnostic["blank_frames"] += sum(
                    token_id == tokenizer.blank_id for token_id in frame_ids
                )
                diagnostic["valid_frames"] += len(frame_ids)
                diagnostic["reference_characters"] += len(reference.replace(" ", ""))
                diagnostic["hypothesis_characters"] += len(hypothesis.replace(" ", ""))
            predictions.append(prediction_row(
                batch["utt_ids"][index], speaker, condition, severity,
                reference, hypothesis,
            ))
            if frame_output_dir is not None:
                frame_output_dir.mkdir(parents=True, exist_ok=True)
                safe_utt_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", batch["utt_ids"][index])
                torch.save({
                    "utt_id": batch["utt_ids"][index],
                    "speaker_id": speaker,
                    "condition": condition,
                    "severity": severity,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "logits": logits[index, :length].detach().cpu().to(torch.float16),
                    "argmax_frame_ids": predicted_ids[index, :length].clone(),
                    "output_length": length,
                    "vocabulary": tokenizer.symbols,
                    "blank_id": tokenizer.blank_id,
                    "shape_order": "T,V",
                }, frame_output_dir / f"{safe_utt_id}.pt")
    if predictions_path:
        with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    word_groups = {("overall", "all"): overall}
    character_groups = {("overall", "all"): overall_cer}
    for group_type, word_values, character_values in (
        ("condition", by_condition, cer_by_condition),
        ("severity", by_severity, cer_by_severity),
        ("speaker", by_speaker, cer_by_speaker),
    ):
        for group_value, metric in word_values.items():
            word_groups[(group_type, group_value)] = metric
            character_groups[(group_type, group_value)] = character_values[group_value]
    group_order = {"overall": 0, "condition": 1, "severity": 2, "speaker": 3}
    grouped_metrics = []
    for group_key in sorted(word_groups, key=lambda key: (group_order[key[0]], key[1])):
        word_metric = word_groups[group_key]
        character_metric = character_groups[group_key]
        diagnostic = diagnostics[group_key]
        grouped_metrics.append({
            "group_type": group_key[0], "group_value": group_key[1],
            "utterances": diagnostic["utterances"],
            "reference_words": word_metric.reference_units,
            "reference_characters": character_metric.reference_units,
            "wer": word_metric.value, "cer": character_metric.value,
            **word_metric.counts_and_rates(),
            "empty_hypothesis_ratio": (
                diagnostic["empty_hypotheses"] / diagnostic["utterances"]
                if diagnostic["utterances"] else 0.0
            ),
            "ctc_blank_frame_ratio": (
                diagnostic["blank_frames"] / diagnostic["valid_frames"]
                if diagnostic["valid_frames"] else 0.0
            ),
            "hypothesis_reference_length_ratio": (
                diagnostic["hypothesis_characters"] / diagnostic["reference_characters"]
                if diagnostic["reference_characters"] else 0.0
            ),
        })
    return {
        "loss": total_loss / len(loader),
        "wer": overall.value,
        "cer": overall_cer.value,
        "reference_words": overall.reference_units,
        "reference_characters": reference_characters,
        "hypothesis_characters": hypothesis_characters,
        "hypothesis_reference_length_ratio": (
            hypothesis_characters / reference_characters if reference_characters else 0.0
        ),
        "empty_hypothesis_ratio": empty_hypotheses / len(predictions) if predictions else 0.0,
        "ctc_blank_frame_ratio": blank_frames / valid_frames if valid_frames else 0.0,
        "wer_by_condition": {key: value.value for key, value in sorted(by_condition.items())},
        "cer_by_condition": {key: value.value for key, value in sorted(cer_by_condition.items())},
        "wer_by_severity": {key: value.value for key, value in sorted(by_severity.items())},
        "cer_by_severity": {key: value.value for key, value in sorted(cer_by_severity.items())},
        "wer_by_speaker": {key: value.value for key, value in sorted(by_speaker.items())},
        "cer_by_speaker": {key: value.value for key, value in sorted(cer_by_speaker.items())},
        "groups": grouped_metrics,
    }


def main(args):
    set_seed(args.seed, args.deterministic)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    codebook_size, available_layers = read_index_dimensions(args.token_index)
    if args.num_rvq_layers > available_layers:
        raise ValueError(f"Requested {args.num_rvq_layers} RVQ layers, index has {available_layers}")
    active_layers = args.active_rvq_layers or list(range(1, args.num_rvq_layers + 1))
    if max(active_layers) > args.num_rvq_layers:
        raise ValueError(
            f"Active layer Q{max(active_layers)} exceeds --num-rvq-layers={args.num_rvq_layers}"
        )
    representation = resolve_representation_metadata(
        args.num_rvq_layers, active_layers, args.rvq_mode,
        args.condition_name, args.layer_fusion,
    )
    ctc_loss_device = args.ctc_loss_device
    if ctc_loss_device == "auto":
        ctc_loss_device = (
            "cpu" if args.deterministic or not args.device.startswith("cuda") else "cuda"
        )
    tokenizer = CharacterTokenizer()
    train_dataset = make_dataset(args, "train", tokenizer)
    if args.overfit_samples:
        if args.overfit_samples > len(train_dataset):
            raise ValueError(
                f"--overfit-samples={args.overfit_samples} exceeds train size {len(train_dataset)}"
            )
        selected = sorted(random.Random(args.seed).sample(range(len(train_dataset)), args.overfit_samples))
        train_dataset = Subset(train_dataset, selected)
        valid_dataset = train_dataset
        test_dataset = train_dataset
    else:
        valid_dataset = make_dataset(args, "valid", tokenizer)
        test_dataset = make_dataset(args, "test", tokenizer)
    train_loader = make_loader(args, train_dataset, codebook_size, shuffle=True)
    train_eval_loader = make_loader(args, train_dataset, codebook_size)
    valid_loader = make_loader(args, valid_dataset, codebook_size)
    test_loader = make_loader(args, test_dataset, codebook_size)
    model = RVQTransformerCTC(
        codebook_size=codebook_size,
        num_rvq_layers=args.num_rvq_layers,
        vocabulary_size=len(tokenizer),
        max_rvq_layers=available_layers,
        model_dim=args.model_dim,
        num_encoder_layers=args.encoder_layers,
        num_heads=args.heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        time_reduction=args.time_reduction,
        subsampling=args.subsampling,
        active_rvq_layers=[layer - 1 for layer in active_layers],
        layer_fusion=args.layer_fusion,
    ).to(args.device)
    embedding_parameters = [
        sum(parameter.numel() for parameter in embedding.parameters())
        for embedding in model.embeddings
    ]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    active_embedding_parameters = sum(
        embedding_parameters[layer - 1] for layer in active_layers
    )
    embedding_parameter_count = sum(embedding_parameters)
    if args.init_checkpoint:
        try:
            initial = torch.load(
                args.init_checkpoint, map_location=args.device, weights_only=False
            )
        except TypeError:
            initial = torch.load(args.init_checkpoint, map_location=args.device)
        state_dict = initial["model"] if isinstance(initial, dict) and "model" in initial else initial
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded initialization checkpoint: {args.init_checkpoint}")
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_score = (float("inf"), float("inf"), float("inf"))
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, args.device,
            args.grad_clip, args.grad_accum_steps, ctc_loss_device,
        )
        valid_metrics = evaluate(
            model, valid_loader, criterion, tokenizer, args.device,
            ctc_loss_device=ctc_loss_device,
        )
        record = {"epoch": epoch, "train_loss": train_loss, "valid": valid_metrics}
        if args.evaluate_train or args.overfit_samples:
            record["train"] = evaluate(
                model, train_eval_loader, criterion, tokenizer, args.device,
                ctc_loss_device=ctc_loss_device,
            )
        history.append(record)
        print(json.dumps(record))
        secondary_metric = "cer" if args.selection_metric == "wer" else "wer"
        monitor_score = (
            valid_metrics[args.selection_metric],
            valid_metrics[secondary_metric],
            valid_metrics["loss"],
        )
        if monitor_score < best_score:
            best_score = monitor_score
            torch.save({
                "model": model.state_dict(), "args": vars(args), "epoch": epoch,
                "valid_metrics": valid_metrics, "vocabulary": tokenizer.symbols,
            }, args.output_dir / "best.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    prediction_name = "overfit_predictions.jsonl" if args.overfit_samples else "test_predictions.jsonl"
    frame_output_dir = args.output_dir / "test_frame_outputs" if args.save_frame_outputs else None
    test_metrics = evaluate(
        model, test_loader, criterion, tokenizer, args.device,
        args.output_dir / prediction_name,
        frame_output_dir,
        ctc_loss_device,
    )
    results = {
        "mode": "overfit" if args.overfit_samples else "standard",
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "overfit_samples": args.overfit_samples,
        "best_epoch": checkpoint["epoch"], "num_rvq_layers": args.num_rvq_layers,
        "active_rvq_layers": active_layers, "layer_fusion": args.layer_fusion,
        "representation_mode": args.representation_mode,
        **representation,
        "parameter_counts": {
            "total": total_parameters,
            "trainable": trainable_parameters,
            "embedding_total": embedding_parameter_count,
            "active_embedding": active_embedding_parameters,
            "inactive_embedding": embedding_parameter_count - active_embedding_parameters,
            "non_embedding": total_parameters - embedding_parameter_count,
        },
        "normalized_layer_weights": model.normalized_layer_weights(),
        "deterministic": args.deterministic,
        "selection_metric": args.selection_metric,
        "ctc_loss_device": ctc_loss_device,
        "codebook_size": codebook_size, "test": test_metrics, "history": history,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(test_metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight Transformer CTC probe on RVQ tokens")
    parser.add_argument("--token-index", type=Path, required=True)
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--init-checkpoint", type=Path,
        help="Initialize the complete probe from a compatible pretraining checkpoint",
    )
    parser.add_argument("--num-rvq-layers", type=int, required=True)
    parser.add_argument(
        "--active-rvq-layers", type=parse_active_layers,
        help="Comma-separated 1-based layers to enable within the configured input, e.g. 1 or 5,6,7,8",
    )
    parser.add_argument(
        "--layer-fusion", choices=("sum", "learned"), default="sum",
        help="Fuse active layer embeddings by fixed normalized sum or learned softmax weights",
    )
    parser.add_argument(
        "--representation-mode", choices=("discrete_learned",),
        default="discrete_learned",
    )
    parser.add_argument("--rvq-mode", choices=("cumulative", "individual", "custom"))
    parser.add_argument("--condition-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--time-reduction", type=int, default=4)
    parser.add_argument("--subsampling", choices=("average", "conv"), default="conv")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--save-frame-outputs", action="store_true",
        help="Save pre-greedy test logits [T,V] and frame-level argmax IDs per utterance",
    )
    parser.add_argument(
        "--overfit-samples", type=int, default=0,
        help="Train and evaluate on the same deterministic subset of train samples",
    )
    parser.add_argument(
        "--evaluate-train", action="store_true",
        help="Compute train WER/CER after every epoch (slower)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Require deterministic PyTorch algorithms and deterministic cuDNN behavior",
    )
    parser.add_argument(
        "--selection-metric", choices=("wer", "cer"), default="wer",
        help="Primary validation metric used to select best.pt (default preserves earlier runs)",
    )
    parser.add_argument(
        "--ctc-loss-device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Compute CTC loss on this device; auto uses CPU for strict deterministic runs",
    )
    args = parser.parse_args()
    if args.overfit_samples < 0:
        parser.error("--overfit-samples cannot be negative")
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
