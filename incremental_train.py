"""
incremental_train.py
- Consumes a retrain ZIP (export from app) or directory of traces
- Supports arch selection: --arch {lstm,gru,transformer}
- Supports KL regularizer weight (--kl-weight)
- Saves model to outdir and writes threshold.json
Usage:
 python incremental_train.py --input retrain_pkg.zip --arch transformer --model ae_model.pth --outdir models --epochs 20 --batch 64 --lr 1e-4 --kl-weight 0.01 --cuda
"""

import argparse
import zipfile
import tempfile
import shutil
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from model import build_model, load_trace, estimated_vocab_size, compute_kl_regularizer

def extract_traces_from_zip(zippath, outdir):
    with zipfile.ZipFile(zippath, "r") as zf:
        zf.extractall(outdir)
    traces = []
    for root, dirs, files in os.walk(outdir):
        for f in files:
            if f.endswith(".json"):
                traces.append(os.path.join(root, f))
    return traces

def collect_traces(input_path):
    p = Path(input_path)
    tmpdir = None
    traces = []
    if p.is_file() and zipfile.is_zipfile(str(p)):
        tmpdir = tempfile.mkdtemp(prefix="bfzdd_inc_")
        traces = extract_traces_from_zip(str(p), tmpdir)
    elif p.is_dir():
        for q in p.rglob("*.json"):
            traces.append(str(q))
    else:
        raise ValueError("Input must be a zip file or directory.")
    return traces, tmpdir

def seq_from_path(p):
    try:
        s = load_trace(str(p))
        if hasattr(s, "tolist"):
            s = s.tolist()
        s = list(s)
        s = [int(x) for x in s]
        return s
    except Exception:
        return None

def pad_seqs(seqs, pad_value=0):
    maxlen = max(len(s) for s in seqs)
    X = np.full((len(seqs), maxlen), pad_value, dtype=np.int64)
    for i,s in enumerate(seqs):
        X[i,:len(s)] = s
    return torch.LongTensor(X)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="retrain zip or directory")
    parser.add_argument("--arch", choices=["lstm","gru","transformer"], default="lstm")
    parser.add_argument("--model", default="ae_model.pth", help="base model to start from (optional)")
    parser.add_argument("--outdir", default="models", help="where to save new models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--cuda", action="store_true", help="use CUDA if available")
    args = parser.parse_args()

    traces, tmpdir = collect_traces(args.input)
    print(f"Found {len(traces)} JSON traces")
    seqs = []
    for t in traces:
        s = seq_from_path(t)
        if s:
            seqs.append(s)
    if len(seqs) == 0:
        raise RuntimeError("No valid sequences found")

    print(f"Loaded {len(seqs)} sequences, building model arch={args.arch}")
    vocab = estimated_vocab_size()
    model = build_model(args.arch, vocab_size=vocab)
    if Path(args.model).exists():
        print("Loading base model:", args.model)
        sd = torch.load(args.model, map_location="cpu")
        try:
            model.load_state_dict(sd)
        except Exception:
            ms = model.state_dict()
            for k,v in sd.items():
                if k in ms and ms[k].shape == v.shape:
                    ms[k] = v
            model.load_state_dict(ms)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.train()

    X = pad_seqs(seqs)
    ds = torch.utils.data.TensorDataset(X)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in dl:
            bx = batch[0].to(device).long()
            opt.zero_grad()
            logits = model(bx)
            B,T,V = logits.size()
            loss = criterion(logits.view(-1,V), bx.view(-1))
            if args.kl_weight and args.kl_weight > 0:
                kl = compute_kl_regularizer(logits.detach())
                loss = loss + args.kl_weight * kl
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        print(f"Epoch {epoch+1}/{args.epochs} loss={epoch_loss/max(1,len(dl)):.6f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    outname = f"ae_finetuned_{args.arch}_{ts}.pth"
    outpath = outdir / outname
    torch.save(model.state_dict(), str(outpath))
    # metadata
    meta = {"arch": args.arch, "vocab_size": vocab, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(str(outpath)+".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print("Saved fine-tuned model:", outpath)

    # threshold calibration
    model.eval()
    scores = []
    with torch.no_grad():
        for s in seqs:
            x = torch.LongTensor([s]).to(device)
            out = model(x)
            B,T,V = out.size()
            losses = nn.CrossEntropyLoss(reduction="none", ignore_index=0)(out.view(-1,V), x.view(-1)).cpu().numpy().reshape(B,T)[0]
            valid = [float(x) for (x,t) in zip(losses, x.cpu().numpy()[0]) if int(t)!=0]
            if valid:
                scores.append(float(np.mean(valid)))
    if len(scores) > 0:
        p95 = float(np.percentile(scores, 95))
        p99 = float(np.percentile(scores, 99))
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        threshold = {"mean": mean, "std": std, "p95": p95, "p99": p99, "suggested_threshold": p99}
        with open("threshold.json", "w") as fh:
            json.dump(threshold, fh, indent=2)
        print("Wrote threshold.json (suggested_threshold=p99)")

    if tmpdir:
        shutil.rmtree(tmpdir)

if __name__ == "__main__":
    main()
