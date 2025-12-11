import torch
import torch.nn as nn
import json
import os

# Simple AuditAutoencoder stub compatible with the Streamlit app.
class AuditAutoencoder(nn.Module):
    def __init__(self, vocab_size=100, emb=64, hid=128, latent=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.enc = nn.LSTM(emb, hid, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hid, vocab_size)  # decoder outputs vocab logits

    def forward(self, x):
        # x: (B, T) long
        e = self.embed(x)
        out, _ = self.enc(e)
        logits = self.fc(out)  # (B, T, vocab)
        return logits

# Simple trace loader: maps events to token ids using a small dictionary,
# pads/truncates to length 128.
EVENT_VOCAB = {
    "open": 2,
    "io.open": 3,
    "socket": 4,
    "connect": 5,
    "os.system": 6,
    "subprocess": 7,
    "os.spawn": 8,
    "os.posix_spawn": 9,
    "shutil": 10,
    "os.remove": 11,
    "os.rename": 12,
    "os.mkdir": 13,
    "file_analysis": 14,
    "other": 1
}

def event_to_token(event_str):
    # basic mapping by prefix match
    for k,v in EVENT_VOCAB.items():
        if event_str.startswith(k):
            return v
    return EVENT_VOCAB["other"]

def load_trace(path, seq_len=128):
    """
    Load trace JSON and convert to list of token ids.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path,'r') as f:
        data = json.load(f)
    tokens = []
    for ev in data:
        e = ev.get("event","other")
        t = event_to_token(e)
        tokens.append(t)
    # pad/truncate
    if len(tokens) >= seq_len:
        return tokens[:seq_len]
    else:
        tokens = tokens + [0]*(seq_len - len(tokens))
        return tokens

if __name__ == "__main__":
    # quick test: create a dummy model and save state dict if run directly
    m = AuditAutoencoder(vocab_size=100)
    torch.save(m.state_dict(), "ae_model_sample.pth")
    print("Saved sample model state to ae_model_sample.pth")
