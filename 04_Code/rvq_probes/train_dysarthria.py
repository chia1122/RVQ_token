#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random,subprocess
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from rvq_probes.dysarthria import DysarthriaDataset,DysarthriaProbe,collate_dysarthria
from rvq_probes.representation import load_speechtokenizer_codebook
from rvq_probes.splits import load_index,validate_speaker_disjoint

def seed_all(s): torch.set_num_threads(1);random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def git_commit():
    try:return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception:return None
def make_loader(d,a,shuffle=False):
    return DataLoader(d,batch_size=a.batch_size,shuffle=shuffle,num_workers=a.num_workers,
      collate_fn=collate_dysarthria,generator=torch.Generator().manual_seed(a.seed))
def class_metrics(labels,preds):
    cm=[[0,0],[0,0]]
    for y,p in zip(labels,preds):cm[y][p]+=1
    per=[]
    for c in (0,1):
        tp=cm[c][c];fp=sum(cm[r][c] for r in (0,1))-tp;fn=sum(cm[c])-tp
        pr=tp/(tp+fp) if tp+fp else 0.;re=tp/(tp+fn) if tp+fn else 0.
        per.append({"precision":pr,"recall":re,"f1":2*pr*re/(pr+re) if pr+re else 0.,"support":sum(cm[c])})
    return {"macro_f1":sum(x["f1"] for x in per)/2,"uar":sum(x["recall"] for x in per)/2,
      "balanced_accuracy":sum(x["recall"] for x in per)/2,"per_class":{"control":per[0],"dysarthric":per[1]},
      "confusion_matrix":{"labels":["control","dysarthric"],"counts":cm}}
def train_epoch(model,data,opt,criterion,device):
    model.train();total=0.
    for b in data:
        x,y=b["pooled"].to(device),b["labels"].to(device)
        loss=criterion(model(x),y);opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=float(loss)
    return total/len(data)
@torch.inference_mode()
def evaluate(model,data,criterion,device,path=None):
    model.eval();labels=[];preds=[];losses=[];rows=[]
    for b in data:
        x,y=b["pooled"].to(device),b["labels"].to(device)
        logits=model(x);losses.append(float(criterion(logits,y)));p=logits.argmax(1).cpu().tolist()
        labels.extend(y.cpu().tolist());preds.extend(p)
        rows.extend({"utt_id":r["utt_id"],"speaker_id":r["speaker_id"],"reference":int(z),"prediction":int(q)} for r,z,q in zip(b["rows"],y.cpu().tolist(),p))
    result={"loss":sum(losses)/len(losses),**class_metrics(labels,preds)}
    if path:path.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
    return result
def main(a):
    if a.device.startswith("cuda") and not torch.cuda.is_available():raise SystemExit("CUDA unavailable")
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True);seed_all(a.seed);rows=load_index(a.token_index);summary=validate_speaker_disjoint(rows)
    codebook,codec=load_speechtokenizer_codebook(a.codec_config,a.codec_checkpoint,a.rvq_layer)
    ds={s:DysarthriaDataset(rows,a.token_root,s,a.rvq_layer,codebook,a.limit_per_split) for s in ("train","valid","test")}
    loaders={s:make_loader(ds[s],a,s=="train") for s in ds};model=DysarthriaProbe(codec["embedding_dim"]).to(a.device)
    criterion=nn.CrossEntropyLoss();opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay)
    history=[];best=-1.
    for epoch in range(1,a.epochs+1):
        loss=train_epoch(model,loaders["train"],opt,criterion,a.device);valid=evaluate(model,loaders["valid"],criterion,a.device)
        record={"epoch":epoch,"train_loss":loss,"valid":valid};history.append(record);print(json.dumps(record),flush=True)
        if valid["macro_f1"]>best:best=valid["macro_f1"];torch.save({"model":model.state_dict(),"epoch":epoch,"valid":valid},a.output_dir/"best.pt")
    ck=torch.load(a.output_dir/"best.pt",map_location=a.device,weights_only=False);model.load_state_dict(ck["model"])
    test=evaluate(model,loaders["test"],criterion,a.device,a.output_dir/"predictions.jsonl")
    config={**vars(a),"git_commit":git_commit(),"codec":codec,"codec_frozen":True,"only_probe_parameters_optimized":True,"model":"masked_mean_layernorm_linear"}
    results={"probe_name":"dysarthria","rvq_layer":a.rvq_layer,"codec":codec["codec"],"seed":a.seed,
      **{f"{s}_speakers":summary[s]["speakers"] for s in ("train","valid","test")},
      **{f"{s}_utterances":len(ds[s]) for s in ds},"main_metric":"macro_f1","main_metric_value":test["macro_f1"],"best_epoch":ck["epoch"],"test":test,"split_summary":summary}
    for n,v in (("config.json",config),("training_history.json",history),("results.json",results)):(a.output_dir/n).write_text(json.dumps(v,indent=2,default=str)+"\n")
    print(json.dumps(results,indent=2))
def parse_args():
    p=argparse.ArgumentParser(description="Frozen individual-RVQ dysarthria probe")
    for n in ("token-index","token-root","codec-config","codec-checkpoint","output-dir"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--rvq-layer",type=int,required=True);p.add_argument("--seed",type=int,default=1337);p.add_argument("--device",default="cuda")
    p.add_argument("--epochs",type=int,default=30);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--weight-decay",type=float,default=1e-2);p.add_argument("--limit-per-split",type=int,default=0)
    return p.parse_args()
if __name__=="__main__":main(parse_args())
