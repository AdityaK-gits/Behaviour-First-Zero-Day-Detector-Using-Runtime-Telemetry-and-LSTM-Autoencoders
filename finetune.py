# finetune.py
"""
Fine-tune existing ae_model.pth on JSON traces found in sandbox_traces/
Saves updated model as ae_model.pth (and registers a version).
Usage:
    python finetune.py
"""

import os
import glob
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# local model
import model as model_mod
from model import AuditAutoencoder, load_trace

# versioning helper (optional)
try:
    import versioning
except Exception:
    versioning = None

# --- Config ---
SANDBOX_DIR = Path("sandbox_traces")
TRACE_GLOB = str(SANDBOX_DIR / "*.json")
MODEL_PATH = "ae_model.pth"
MAX_LEN = 200
BATCH_SIZE = 8
EPOCHS = 4
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utilities ---
def estimate_vocab():
    if hasattr(model_mod, "estimated_vocab_size"):
        try:
            return int(model_mod.estimated_vocab_size())
        except Exception:
            pass
    return int(getattr(model_mod, "_next_token_index", 200))

VOCAB = estimate_vocab()
print(f"[finetune] Using VOCAB estimate = {VOCAB}, device = {DEVICE}")

def safe_load_seq(path):
    try:
        seq = load_trace(str(path))
        if hasattr(seq, "tolist"):
            seq = list(seq.tolist())
        elif isinstance(seq, (list, tuple)):
            seq = list(seq)
        else:
            seq = list(seq)
        return [int(x) for x in seq]
    except Exception as e:
        print(f"[finetune] load_trace error for {path}: {e}")
        return []

def pad_or_truncate(seq, length=MAX_LEN, pad_value=0):
    if len(seq) >= length:
        return seq[:length]
    else:
        return seq + [pad_value] * (length - len(seq))

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
    return torch.stack(batch, dim=0)

# --- Main ---
def finetune():
    files = sorted(glob.glob(TRACE_GLOB))
    if not files:
        print("[finetune] No sandbox traces found in sandbox_traces/.")
        return

    print(f"[finetune] Found {len(files)} traces. Using VOCAB={VOCAB}")

    ds = TraceDataset(files)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_pad(b))

    if not Path(MODEL_PATH).exists():
        print(f"[finetune] Base model {MODEL_PATH} not found. Aborting.")
        return

    device = DEVICE
    model = AuditAutoencoder(vocab_size=VOCAB).to(device)

    # Load state robustly (partial load on mismatch)
    state = torch.load(MODEL_PATH, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        print("[finetune] full state load failed; attempting partial safe load.")
        model_state = model.state_dict()
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
        model.load_state_dict(model_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        steps = 0
        for batch in dl:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            B, T, V = out.size()
            logits = out.view(B * T, V)
            targets = batch.view(B * T)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps += 1
        avg = total / max(1, steps)
        print(f"[finetune] Epoch {epoch}/{EPOCHS} avg_loss = {avg:.4f}")

    # Save updated model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[finetune] Saved finetuned model -> {MODEL_PATH}")

    # Register version
    if versioning is not None:
        try:
            entry = versioning.save_model_version(MODEL_PATH, note="finetuned from finetune.py")
            print(f"[finetune] Registered model version: {entry['name']}")
        except Exception as e:
            print(f"[finetune] versioning.save_model_version failed: {e}")

if __name__ == "__main__":
    finetune()
