#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

MODULES={"phoneme_ctc":"rvq_probes.train_phoneme_ctc","phoneme_boundary":"rvq_probes.train_phoneme_boundary","dysarthria":"rvq_probes.train_dysarthria"}
def main(a):
    rotations=sorted(path for path in a.rotations_root.glob("rotation_*") if path.is_dir() and (path / "tokens.jsonl").is_file())
    if not rotations: raise SystemExit(f"No rotations under {a.rotations_root}")
    for rotation in rotations:
        for layer in range(1,a.num_rvq_layers+1):
            output=a.output_root/rotation.name/f"q{layer}"
            if a.skip_completed and (output / "results.json").is_file():
                print(f"skip completed {output}", flush=True)
                continue
            command=[sys.executable,"-m",MODULES[a.probe],"--token-index",str(rotation/"tokens.jsonl"),
              "--token-root",str(a.token_root),"--codec-config",str(a.codec_config),"--codec-checkpoint",str(a.codec_checkpoint),
              "--output-dir",str(output),"--rvq-layer",str(layer),"--seed",str(a.seed),"--device",a.device]
            if a.probe=="phoneme_ctc": command += ["--phoneme-targets",str(a.phoneme_targets),"--phoneme-vocabulary",str(a.phoneme_vocabulary)]
            elif a.probe=="phoneme_boundary": command += ["--boundary-targets",str(a.boundary_targets)]
            print(" ".join(command),flush=True)
            if not a.dry_run: subprocess.run(command,check=True)
def parse_args():
    p=argparse.ArgumentParser(description="Run individual Q1..QN probes over formal speaker rotations")
    p.add_argument("--probe",choices=MODULES,required=True);p.add_argument("--rotations-root",type=Path,required=True)
    for n in ("token-root","codec-config","codec-checkpoint","output-root"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--phoneme-targets",type=Path);p.add_argument("--phoneme-vocabulary",type=Path);p.add_argument("--boundary-targets",type=Path)
    p.add_argument("--num-rvq-layers",type=int,default=8);p.add_argument("--skip-completed",action="store_true");p.add_argument("--seed",type=int,default=1337);p.add_argument("--device",default="cuda");p.add_argument("--dry-run",action="store_true")
    return p.parse_args()
if __name__=="__main__":main(parse_args())
