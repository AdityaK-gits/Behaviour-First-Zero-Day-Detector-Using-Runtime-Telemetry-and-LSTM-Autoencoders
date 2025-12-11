# train.py
# Trains a simple LSTM Autoencoder on benign traces saved by dataset_gen.py
# Produces: ae_model.pth

import os, glob, json
import numpy as np
import torch
import torch.nn as nn
from model import AuditAutoencoder, load_trace  # uses your model.py loader
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "dataset/traces"
MODEL_PATH = "ae_model.pth"
VOCAB_SIZE = 100  # must match model.py vocab size

# Simple dataset: loads benign traces and returns token sequences
class TraceDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        seq = load_trace(f)  # load_trace should convert trace->token ids list
        return torch.LongTensor(seq)

def collate_pad(batch):
    # batch: list of LongTensor (L_i)
    lengths = [b.size(0) for b in batch]
    maxlen = max(lengths)
    out = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, b in enumerate(batch):
        out[i, :b.size(0)] = b
    return out, torch.LongTensor(lengths)

def train():
    # pick benign files
    benign_files = sorted(glob.glob(os.path.join(DATA_DIR, "benign_*.json")))
    if not benign_files:
        raise SystemExit("No benign traces found. Run dataset_gen.py first.")
    ds = TraceDataset(benign_files)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=lambda batch: collate_pad(batch))

    device = torch.device("cpu")
    model = AuditAutoencoder(vocab_size=VOCAB_SIZE)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 8
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        count = 0
        for x_batch, lengths in dl:
            x_batch = x_batch.to(device)
            optim.zero_grad()
            out = model(x_batch)  # (B, T, vocab)
            # flatten for CE
            B, T, V = out.size()
            logits = out.view(B*T, V)
            targets = x_batch.view(B*T)
            loss = criterion(logits, targets)
            loss.backward()
            optim.step()
            total_loss += loss.item() * B
            count += B
        print(f"Epoch {epoch+1}/{epochs} avg loss: {total_loss / max(1,count):.4f}")
    # save
    torch.save(model.state_dict(), MODEL_PATH)
    print("Saved model ->", MODEL_PATH)

if __name__ == "__main__":
    train()
