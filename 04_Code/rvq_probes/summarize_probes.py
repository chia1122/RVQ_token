#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,re,statistics
from pathlib import Path
KEYS={"phoneme_ctc":("test","per"),"phoneme_boundary":("test","tolerant_1","f1"),"dysarthria":("test","macro_f1")}
def nested(value,path):
    for key in path:value=value[key]
    return float(value)
def main(a):
    collected={i:{p:[] for p in KEYS} for i in range(1,a.num_rvq_layers+1)};rotation_rows=[];boundary_rows=[]
    for path in a.runs_root.rglob("results.json"):
        data=json.loads(path.read_text());probe=data.get("probe_name");layer=data.get("rvq_layer")
        if probe not in KEYS or layer not in collected:continue
        match=re.search(r"rotation[_-](\d+)",str(path));rotation=f"rotation_{int(match.group(1)):02d}" if match else "unknown"
        value=nested(data,KEYS[probe]);collected[layer][probe].append(value)
        rotation_rows.append({"rotation":rotation,"rvq_layer":layer,"probe":probe,"value":value})
        if probe=="phoneme_boundary":
            row={"rotation":rotation,"rvq_layer":f"Q{layer}"}
            for source in ("exact","tolerant_1"):
                for metric in ("precision","recall","f1"):row[f"{source}_{metric}"]=float(data["test"][source][metric])
            boundary_rows.append(row)
    with (a.output_dir/"probe_results_by_rotation.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["rotation","rvq_layer","probe","value"]);w.writeheader();w.writerows(sorted(rotation_rows,key=lambda x:(x["rotation"],x["rvq_layer"],x["probe"])))
    fields=["rvq_layer","phoneme_per_mean","phoneme_per_std","boundary_f1_mean","boundary_f1_std","dysarthria_macro_f1_mean","dysarthria_macro_f1_std"];rows=[]
    for layer,probes in collected.items():
        row={"rvq_layer":f"Q{layer}"}
        for probe,prefix in (("phoneme_ctc","phoneme_per"),("phoneme_boundary","boundary_f1"),("dysarthria","dysarthria_macro_f1")):
            values=probes[probe];row[prefix+"_mean"]=statistics.mean(values) if values else "";row[prefix+"_std"]=statistics.stdev(values) if len(values)>1 else ""
        rows.append(row)
    with (a.output_dir/"probe_summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    boundary_fields=["rotation","rvq_layer","exact_precision","exact_recall","exact_f1","tolerant_1_precision","tolerant_1_recall","tolerant_1_f1"]
    with (a.output_dir/"boundary_results_by_rotation.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=boundary_fields);w.writeheader();w.writerows(sorted(boundary_rows,key=lambda x:(x["rotation"],x["rvq_layer"])))
    summary_fields=["rvq_layer"]+[f"{kind}_{metric}_{stat}" for kind in ("exact","tolerant_1") for metric in ("precision","recall","f1") for stat in ("mean","std")];summary_rows=[]
    for layer in range(1,a.num_rvq_layers+1):
        selected=[row for row in boundary_rows if row["rvq_layer"]==f"Q{layer}"];row={"rvq_layer":f"Q{layer}"}
        for kind in ("exact","tolerant_1"):
            for metric in ("precision","recall","f1"):
                values=[item[f"{kind}_{metric}"] for item in selected];row[f"{kind}_{metric}_mean"]=statistics.mean(values) if values else "";row[f"{kind}_{metric}_std"]=statistics.stdev(values) if len(values)>1 else ""
        summary_rows.append(row)
    with (a.output_dir/"boundary_summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=summary_fields);w.writeheader();w.writerows(summary_rows)
def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--runs-root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--num-rvq-layers",type=int,default=8);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);return a
if __name__=="__main__":main(parse_args())
