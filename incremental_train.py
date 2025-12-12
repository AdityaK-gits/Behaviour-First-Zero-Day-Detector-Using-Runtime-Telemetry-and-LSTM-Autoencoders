# incremental_train.py
# Usage:
#   python incremental_train.py --input retrain_package.zip --model ae_model.pth --outdir models/ --epochs 10 --batch 32 --lr 1e-4
#
# This script:
# - extracts retrain_package.zip (or reads a directory)
# - loads traces (uploaded + optional replay)
# - loads ae_model.pth if provided, else initializes model
# - trains on GPU if available
# - saves versioned model in outdir and writes threshold.json

import os
import argparse
import zipfile
import json
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from model import AuditAutoencoder, load_trace, estimated_vocab_size

def read_traces_from_zip(zip_path, tmpdir):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)
    traces = []
    # uploaded/ and replay/ paths inside zip
    for root, dirs, files in os.walk(tmpdir):
        for f in files:
            if f.endswith(".json"):
                traces.append(os.path.join(root, f))
    return traces

def read_traces_from_dir(dir_path):
    traces = []
    for p in Path(dir_path).rglob("*.json"):
        traces.append(str(p))
    return traces

def seq_from_path(p):
    try:
        seq = load_trace(str(p))
        if hasattr(seq, "tolist"):
            seq = seq.tolist()
        seq = list(seq)
        seq = [int(x) for x in seq]
        return seq
    except Exception as e:
        print("Failed load_trace:", p, e)
        return None

def pad_sequences(seqs, pad_value=0):
    maxlen = max(len(s) for s in seqs)
    arr = np.full((len(seqs), maxlen), pad_value, dtype=np.int64)
    for i,s in enumerate(seqs):
        arr[i,:len(s)] = s
    return torch.LongTensor(arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to retrain zip or directory with traces")
    parser.add_argument("--model", default="ae_model.pth", help="Base model (.pth) to fine-tune")
    parser.add_argument("--outdir", default="models", help="Output directory for new model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    tmpdir = None
    traces = []
    if zipfile.is_zipfile(args.input):
        tmpdir = tempfile.mkdtemp(prefix="bfzdd_retrain_")
        traces = read_traces_from_zip(args.input, tmpdir)
    elif Path(args.input).is_dir():
        traces = read_traces_from_dir(args.input)
    else:
        raise ValueError("Input must be a zip file or a directory of traces")

    print(f"Found {len(traces)} trace files")

    seqs = []
    for t in traces:
        s = seq_from_path(t)
        if s:
            seqs.append(s)
    if len(seqs) == 0:
        raise RuntimeError("No valid sequences found")

    vb = estimated_vocab_size() if callable(estimated_vocab_size) else 200
    model = AuditAutoencoder(vocab_size=vb)
    if Path(args.model).exists():
        print("Loading base model:", args.model)
        state = torch.load(args.model, map_location="cpu")
        try:
            model.load_state_dict(state)
        except Exception:
            ms = model.state_dict()
            for k,v in state.items():
                if k in ms and ms[k].shape == v.shape:
                    ms[k] = v
            model.load_state_dict(ms)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.train()

    X = pad_sequences(seqs)
    dataset = torch.utils.data.TensorDataset(X)
    dl = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in dl:
            batch_x = batch[0].to(device).long()
            opt.zero_grad()
            logits = model(batch_x)
            B,T,V = logits.size()
            loss = criterion(logits.view(-1,V), batch_x.view(-1))
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} loss={epoch_loss/len(dl):.6f}")

    # save model
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    outpath = outdir / f"ae_model_finetuned_{ts}.pth"
    torch.save(model.state_dict(), str(outpath))
    print("Saved fine-tuned model to", outpath)

    # threshold calibration: compute mean per-sample recon loss and write threshold.json
    model.eval()
    benign_scores = []
    with torch.no_grad():
        for s in seqs:
            x = torch.LongTensor([s]).to(device)
            out = model(x)
            B,T,V = out.size()
            losses = nn.CrossEntropyLoss(reduction="none", ignore_index=0)(out.view(-1,V), x.view(-1))
            benign_scores.append(float(losses.mean().item()))
    p95 = float(np.percentile(benign_scores, 95))
    p99 = float(np.percentile(benign_scores, 99))
    mean_score = float(np.mean(benign_scores))
    std_score = float(np.std(benign_scores))
    threshold = {"mean": mean_score, "std": std_score, "p95": p95, "p99": p99, "suggested_threshold": p99}
    with open("threshold.json", "w") as fh:
        json.dump(threshold, fh, indent=2)
    print("Wrote threshold.json (suggested_threshold=p99)")

    # cleanup tempdir if created
    if tmpdir:
        shutil.rmtree(tmpdir)
        print("Removed temporary dir", tmpdir)

if __name__ == "__main__":
    main()
