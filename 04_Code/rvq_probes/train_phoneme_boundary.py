#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, random, subprocess
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from rvq_probes.boundaries import match_boundary_frames
from rvq_probes.phoneme_boundary import BoundaryDataset, PhonemeBoundaryProbe, collate_boundary
from rvq_probes.representation import load_speechtokenizer_codebook
from rvq_probes.splits import load_index, validate_speaker_disjoint


def seed_all(seed):
    torch.set_num_threads(1)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return None


def loader(dataset, args, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, collate_fn=collate_boundary,
                      generator=torch.Generator().manual_seed(args.seed))


def train_epoch(model, data, optimizer, criterion, device):
    model.train(); total = 0.0
    for batch in data:
        x, y, mask = (batch[k].to(device) for k in ("representations", "targets", "mask"))
        loss = criterion(model(x)[mask], y[mask]); optimizer.zero_grad(set_to_none=True)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        total += float(loss)
    return total / len(data)


@torch.inference_mode()
def evaluate(model, data, criterion, device, threshold=0.5, predictions_path=None):
    model.eval(); losses, predictions = [], []
    totals = {0: {"tp": 0, "fp": 0, "fn": 0}, 1: {"tp": 0, "fp": 0, "fn": 0}}
    positives = frames = 0
    for batch in data:
        x, y, mask = (batch[k].to(device) for k in ("representations", "targets", "mask"))
        logits = model(x); losses.append(float(criterion(logits[mask], y[mask])))
        for i, length in enumerate(batch["lengths"].tolist()):
            pred = torch.where(logits[i, :length].sigmoid().cpu() >= threshold)[0].tolist()
            ref = batch["frames"][i]; positives += len(ref); frames += length
            metrics = {}
            for tolerance, name in ((0, "exact"), (1, "tolerant_1")):
                result = match_boundary_frames(ref, pred, tolerance)
                metrics[name] = result
                for key in totals[tolerance]: totals[tolerance][key] += result[key]
            row = batch["rows"][i]
            predictions.append({"utt_id": row["utt_id"], "reference_frames": ref,
                                "predicted_frames": pred, **metrics})
    output = {"loss": sum(losses)/len(losses), "positive_prevalence": positives/frames,
              "positive_frames": positives, "total_frames": frames}
    for tolerance, name in ((0, "exact"), (1, "tolerant_1")):
        c = totals[tolerance]; p = c["tp"]/(c["tp"]+c["fp"]) if c["tp"]+c["fp"] else 0.0
        r = c["tp"]/(c["tp"]+c["fn"]) if c["tp"]+c["fn"] else 0.0
        tn = frames - c["tp"] - c["fp"] - c["fn"]
        output[name] = {**c, "tn": tn, "precision": p, "recall": r,
                        "f1": 2*p*r/(p+r) if p+r else 0.0,
                        "confusion_matrix": {"tn": tn, "fp": c["fp"], "fn": c["fn"], "tp": c["tp"]}}
    if predictions_path:
        predictions_path.write_text("".join(json.dumps(x, sort_keys=True)+"\n" for x in predictions))
    return output


def main(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True); seed_all(args.seed)
    rows = load_index(args.token_index); summary = validate_speaker_disjoint(rows)
    codebook, codec = load_speechtokenizer_codebook(args.codec_config, args.codec_checkpoint, args.rvq_layer)
    datasets = {s: BoundaryDataset(rows, args.token_root, args.boundary_targets, s,
                args.rvq_layer, codebook, args.limit_per_split) for s in ("train", "valid", "test")}
    loaders = {s: loader(datasets[s], args, s == "train") for s in datasets}
    train_pos = sum(len(datasets["train"].targets[row["utt_id"]]["boundary_frames"]) for row in datasets["train"].rows)
    train_frames = sum(int(row["num_frames"]) for row in datasets["train"].rows)
    pos_weight = (train_frames-train_pos)/train_pos if train_pos else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=args.device))
    model = PhonemeBoundaryProbe(codec["embedding_dim"], args.hidden_dim, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history, best = [], -1.0
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, loaders["train"], optimizer, criterion, args.device)
        valid = evaluate(model, loaders["valid"], criterion, args.device, args.threshold)
        record = {"epoch": epoch, "train_loss": loss, "valid": valid}
        history.append(record)
        print(json.dumps(record), flush=True)
        if valid["tolerant_1"]["f1"] > best:
            best = valid["tolerant_1"]["f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "valid": valid}, args.output_dir/"best.pt")
    checkpoint = torch.load(args.output_dir/"best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, loaders["test"], criterion, args.device, args.threshold, args.output_dir/"predictions.jsonl")
    config = {**vars(args), "git_commit": git_commit(), "codec": codec, "codec_frozen": True,
              "only_probe_parameters_optimized": True, "pos_weight_from_training_only": pos_weight,
              "model": "layernorm_projection_local_conv_linear"}
    results = {"probe_name": "phoneme_boundary", "rvq_layer": args.rvq_layer,
               "codec": codec["codec"], "seed": args.seed,
               **{f"{s}_speakers": summary[s]["speakers"] for s in ("train","valid","test")},
               **{f"{s}_utterances": len(datasets[s]) for s in ("train","valid","test")},
               "main_metric": "tolerant_1_f1", "main_metric_value": test["tolerant_1"]["f1"],
               "best_epoch": checkpoint["epoch"], "test": test, "split_summary": summary}
    for name, value in (("config.json",config),("training_history.json",history),("results.json",results)):
        (args.output_dir/name).write_text(json.dumps(value, indent=2, default=str)+"\n")
    print(json.dumps(results, indent=2))


def parse_args():
    p=argparse.ArgumentParser(description="Frozen individual-RVQ phoneme boundary probe")
    for name in ("token-index","token-root","boundary-targets","codec-config","codec-checkpoint","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--rvq-layer",type=int,required=True); p.add_argument("--seed",type=int,default=1337)
    p.add_argument("--device",default="cuda"); p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--learning-rate",type=float,default=3e-4); p.add_argument("--weight-decay",type=float,default=1e-2)
    p.add_argument("--hidden-dim",type=int,default=64); p.add_argument("--dropout",type=float,default=.1)
    p.add_argument("--threshold",type=float,default=.5); p.add_argument("--limit-per-split",type=int,default=0)
    return p.parse_args()
if __name__ == "__main__": main(parse_args())
