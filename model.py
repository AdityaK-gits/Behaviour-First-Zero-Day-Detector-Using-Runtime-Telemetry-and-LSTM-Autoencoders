"""
model.py

Provides:
- AuditAutoencoder: an LSTM-based autoencoder that returns logits shaped (B, T, V)
- load_trace(path, max_len=None): converts JSON runtime trace into list[int] token ids
- score_trace_with_model(model, seq_tokens): returns (avg_loss, per_token_losses_list)
- estimated_vocab_size(): helper to inspect current token mapping size

Design choices:
- index 0 == PAD token
- event->token mapping grows dynamically in-memory; estimate function exposes current size
- AuditAutoencoder is intentionally simple and robust (Embedding + LSTM + Linear projection)
"""

import json
import os
from pathlib import Path
from typing import List

# PyTorch imports
import torch
import torch.nn as nn

# -------------------------
# Global token mapping
# -------------------------
# 0 reserved for PAD
_event_to_idx = {
    # seed with some common events (helps stable vocab across runs)
    "open": 1,
    "io.open": 2,
    "socket": 3,
    "connect": 4,
    "os.system": 5,
    "subprocess": 6,
    "os.spawn": 7,
    "shutil": 8,
    "os.remove": 9,
    "os.rename": 10,
    "os.mkdir": 11,
    "file_analysis": 12,
}
_idx_to_event = {v: k for k, v in _event_to_idx.items()}
_next_token_index = max(_event_to_idx.values(), default=1) + 1


def _register_event_name(name: str):
    """Map an event name to an integer token. If unknown, add to mapping."""
    global _next_token_index
    if name in _event_to_idx:
        return _event_to_idx[name]
    # assign new id
    _event_to_idx[name] = _next_token_index
    _idx_to_event[_next_token_index] = name
    _next_token_index += 1
    return _event_to_idx[name]


def estimated_vocab_size() -> int:
    """Return current estimated vocab size (useful for scripts)."""
    # +1 to account for PAD=0
    return max(_next_token_index, 1)


# -------------------------
# Trace loading / tokenization
# -------------------------
def _extract_event_name(event_item):
    """
    Accepts an event item (string or dict). Returns a simple string 'event' key.
    For dicts: prefer event_item.get('event') else str(event_item).
    """
    if isinstance(event_item, str):
        return event_item
    if isinstance(event_item, dict):
        # Many of our traces have keys: "event", "args", "t"
        if "event" in event_item and isinstance(event_item["event"], str):
            return event_item["event"]
        # fallback to type or a summarised form
        # try args[0] or first key
        if "args" in event_item and event_item["args"]:
            # try to map to a coarse category (e.g., 'open', 'connect', etc.)
            maybe = event_item["args"][0]
            if isinstance(maybe, str):
                return str(maybe)
        # final fallback
        return str(event_item)
    # other types
    return str(event_item)


def load_trace(path: str, max_len: int = None) -> List[int]:
    """
    Load a JSON trace file and return a list of integer token ids.
    - path: file path to JSON (expected to be a list of events)
    - max_len: if provided, truncates/pads to this length (pads with 0)
    Returns list[int] (possibly empty).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # Could be not JSON or missing; return empty sequence
        return []

    # data expected to be a list of events
    seq = []
    if isinstance(data, dict):
        # sometimes we saved object with key 'events'
        if "events" in data and isinstance(data["events"], list):
            data = data["events"]
        elif "trace" in data and isinstance(data["trace"], list):
            data = data["trace"]
        else:
            # if not list, fallback to treating dict as single event
            data = [data]

    if not isinstance(data, list):
        return []

    for item in data:
        name = _extract_event_name(item)
        # normalize name: only take the prefix/event token (avoid arg proliferation)
        # Do simple normalization: keep string up to first whitespace or punctuation
        if isinstance(name, str):
            # lower-case and strip
            nm = name.strip().lower()
            # for readability compress sequences like 'open' or 'socket.connect' into tokens
            # We prefer 'open', 'socket', 'connect', 'file_analysis' etc.
            # If name contains '.', take left-most token
            if "." in nm:
                nm = nm.split(".")[0]
            # take first word
            nm = nm.split()[0]
        else:
            nm = str(name)

        token = _register_event_name(nm)
        seq.append(token)

    # optional truncation/padding
    if max_len is not None:
        if len(seq) >= max_len:
            return seq[:max_len]
        else:
            # pad with 0
            return seq + [0] * (max_len - len(seq))
    return seq


# -------------------------
# AuditAutoencoder model
# -------------------------
class AuditAutoencoder(nn.Module):
    """
    Simple per-timestep autoencoder:
    - Embedding -> LSTM (encoder-style) -> Linear projection to vocab logits per timestep.
    This is not a seq2seq with separate decoder; it is a simple reconstruction network.
    It outputs logits shaped (B, T, V) where V = vocab_size.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # embedding (pad_idx reserved)
        self.embedding = nn.Embedding(self.vocab_size + 1, self.emb_dim, padding_idx=self.pad_idx)
        # bidirectional may help but keep uni-directional for simplicity
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        # project hidden states back to vocab
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size + 1)

    def forward(self, x):
        """
        x: LongTensor shape (B, T)
        returns logits: (B, T, V) where V = vocab_size + 1 (including pad)
        """
        if x is None:
            return None
        emb = self.embedding(x)  # (B, T, emb_dim)
        out, (h, c) = self.lstm(emb)  # out: (B, T, hidden_dim)
        logits = self.fc(out)  # (B, T, V)
        return logits


# -------------------------
# Scoring helper
# -------------------------
def score_trace_with_model(model: nn.Module, seq_tokens, pad_idx: int = 0):
    """
    Score a sequence using the given model.
    - model: AuditAutoencoder (or compatible); model(x) -> (B,T,V) logits
    - seq_tokens: list[int] OR 1D tensor
    Returns:
    - avg_loss: float (mean per-token cross entropy, ignoring pad index)
    - per_token_losses: list[float] (length = T) with per-token loss values
    """
    import numpy as _np

    if model is None:
        raise RuntimeError("Model is None in score_trace_with_model")

    if isinstance(seq_tokens, list):
        if len(seq_tokens) == 0:
            return float("inf"), []
        x = torch.LongTensor([seq_tokens])
    elif isinstance(seq_tokens, torch.Tensor):
        if seq_tokens.dim() == 1:
            x = seq_tokens.long().unsqueeze(0)
        else:
            x = seq_tokens.long()
    else:
        # fallback cast
        try:
            x = torch.LongTensor([list(seq_tokens)])
        except Exception:
            raise ValueError("seq_tokens cannot be converted to tensor in score_trace_with_model")

    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
    x = x.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        out = model(x)  # (B, T, V)
        if out is None:
            raise RuntimeError("Model forward returned None in score")
        B, T, V = out.size()
        logits = out.view(B * T, V)
        targets = x.view(B * T)
        criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_idx)
        losses = criterion(logits, targets).cpu().numpy()  # (B*T,)
        per_token = losses.reshape(B, T)[0].tolist()  # get first batch row
        # ignore pad positions when computing average
        mask = (_np.array(targets.cpu()) != pad_idx).reshape(B, T)[0]
        valid_losses = _np.array(per_token)[mask]
        if valid_losses.size == 0:
            avg = float("inf")
        else:
            avg = float(valid_losses.mean())
        return float(avg), [float(x) for x in per_token]


# -------------------------
# If run as script, simple demo of loading a trace (not executed on import)
# -------------------------
if __name__ == "__main__":
    print("model.py debug: estimated_vocab_size =", estimated_vocab_size())
    p = Path("dataset/traces")
    if p.exists():
        files = list(p.glob("*.json"))
        if files:
            seq = load_trace(str(files[0]), max_len=100)
            print("Loaded trace sample tokens (len):", len(seq))
