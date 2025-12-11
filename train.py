# train.py
"""
Train an LSTM Autoencoder on benign traces and save ae_model.pth.

Usage:
    python train.py

Notes:
 - Expects tokenized traces (JSON event traces convertible by model.load_trace)
 - Model architecture is provided by model.AuditAutoencoder
 - Versioning: saves a model snapshot into models/ via versioning.save_model_version
"""

import os
import glob
import json
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# local model module
import model as model_mod
from model import AuditAutoencoder, load_trace

# versioning helper (optional)
try:
    import versioning
except Exception:
    versioning = None

# --- Config ---
DATA_DIR = Path("dataset/traces")
BENIGN_GLOB = str(DATA_DIR / "*benign*.json")
MODEL_OUT = "ae_model.pth"

BATCH_SIZE = 8
EPOCHS = 8
LR = 1e-3
MAX_LEN = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)

# --- Utilities ---
def estimate_vocab():
    if hasattr(model_mod, "estimated_vocab_size"):
        try:
            return int(model_mod.estimated_vocab_size())
        except Exception:
            pass
    return int(getattr(model_mod, "_next_token_index", 200))

VOCAB = estimate_vocab()
print(f"[train] Using VOCAB estimate = {VOCAB}, device = {DEVICE}")

def safe_load_seq(path):
    """
    Call load_trace(path) robustly and convert to list of ints.
    If load_trace signature differs, handle exceptions and return [] if cannot parse.
    """
    try:
        seq = load_trace(str(path))
        # convert numpy / tensor / list -> python list
        if hasattr(seq, "tolist"):
            seq = list(seq.tolist())
        elif isinstance(seq, (list, tuple)):
            seq = list(seq)
        else:
            seq = list(seq)
        seq = [int(x) for x in seq]
        return seq
    except TypeError:
        # try calling without any args if that function variant exists
        try:
            seq = load_trace(str(path))
            if hasattr(seq, "tolist"):
                seq = list(seq.tolist())
            else:
                seq = list(seq)
            return [int(x) for x in seq]
        except Exception as e:
            print(f"[train] load_trace failed for {path}: {e}")
            return []
    except Exception as e:
        print(f"[train] load_trace error for {path}: {e}")
        return []

def pad_or_truncate(seq, length=MAX_LEN, pad_value=0):
    if len(seq) >= length:
        return seq[:length]
    else:
        return seq + [pad_value] * (length - len(seq))

# Dataset wrapper
class TraceDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        seq = safe_load_seq(self.files[idx])
        seq = pad_or_truncate(seq, MAX_LEN)
        return torch.LongTensor(seq)

def collate_pad(batch):
    # all samples are MAX_LEN fixed already
    return torch.stack(batch, dim=0)

# --- Main training ---
def train():
    benign_files = sorted(glob.glob(BENIGN_GLOB))
    if not benign_files:
        print("[train] No benign traces found. Ensure traces exist in dataset/traces and are named with 'benign'.")
        return

    ds = TraceDataset(benign_files)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_pad(b))

    model = AuditAutoencoder(vocab_size=VOCAB).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"[train] Starting training on {len(benign_files)} samples, epochs={EPOCHS}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in dl:
            batch = batch.to(DEVICE)  # (B, T)
            optimizer.zero_grad()
            out = model(batch)        # (B, T, V)
            B, T, V = out.size()
            logits = out.view(B * T, V)
            targets = batch.view(B * T)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        avg = total_loss / max(1, steps)
        print(f"[train] Epoch {epoch}/{EPOCHS}  avg_loss = {avg:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"[train] Saved model -> {MODEL_OUT}")

    # Register version
    if versioning is not None:
        try:
            entry = versioning.save_model_version(MODEL_OUT, note="trained from train.py")
            print(f"[train] Registered model version: {entry['name']}")
        except Exception as e:
            print(f"[train] versioning.save_model_version failed: {e}")

if __name__ == "__main__":
    train()
