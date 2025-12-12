"""
model.py - Upgraded BFZDD model module

Features:
- build_model(arch="lstm"|"gru"|"transformer", vocab_size=..., **kwargs)
- AuditAutoencoderLSTM / AuditAutoencoderGRU / AuditAutoencoderTransformer (forward -> logits (B,T,V))
- load_trace(path, max_len=None)  (keeps previous behavior)
- score_trace_with_model(model, seq_tokens, scoring="ce"|"smoothed_prob")
    returns (score, per_token_losses)
    - `scoring="ce"` returns average cross-entropy (old behavior)
    - `scoring="smoothed_prob"` returns smoothed negative log-prob score (recommended)
- compute_kl_regularizer(logits, prior="uniform")
- estimated_vocab_size()

Backwards compatible with previous UI and training scripts.
"""

from typing import List, Tuple, Optional, Union, Dict
import json
from pathlib import Path
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------
# Token mapping (same design)
# -------------------------
_event_to_idx = {
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


def _register_event_name(name: str) -> int:
    global _next_token_index
    if name in _event_to_idx:
        return _event_to_idx[name]
    _event_to_idx[name] = _next_token_index
    _idx_to_event[_next_token_index] = name
    _next_token_index += 1
    return _event_to_idx[name]


def estimated_vocab_size() -> int:
    """Estimate current vocab size (includes PAD=0)."""
    # +1 for pad token reserved at 0
    return max(_next_token_index, 1)


# -------------------------
# Trace loading (backwards-compatible)
# -------------------------
def _extract_event_name(event_item):
    if isinstance(event_item, str):
        return event_item
    if isinstance(event_item, dict):
        if "event" in event_item and isinstance(event_item["event"], str):
            return event_item["event"]
        if "args" in event_item and event_item["args"]:
            maybe = event_item["args"][0]
            if isinstance(maybe, str):
                return str(maybe)
        return str(event_item)
    return str(event_item)


def load_trace(path: str, max_len: Optional[int] = None) -> List[int]:
    """
    Load JSON trace and return token list. Pads/truncates if max_len provided.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, dict):
        if "events" in data and isinstance(data["events"], list):
            data = data["events"]
        elif "trace" in data and isinstance(data["trace"], list):
            data = data["trace"]
        else:
            data = [data]

    if not isinstance(data, list):
        return []

    seq = []
    for item in data:
        name = _extract_event_name(item)
        if isinstance(name, str):
            nm = name.strip().lower()
            if "." in nm:
                nm = nm.split(".")[0]
            nm = nm.split()[0]
        else:
            nm = str(name)
        token = _register_event_name(nm)
        seq.append(token)

    if max_len is not None:
        if len(seq) >= max_len:
            return seq[:max_len]
        else:
            return seq + [0] * (max_len - len(seq))
    return seq


# -------------------------
# Model architectures
# -------------------------
class BaseAutoencoder(nn.Module):
    """Common utilities for models."""
    def __init__(self):
        super().__init__()

    def freeze_embedding(self):
        if hasattr(self, "embedding"):
            for p in self.embedding.parameters():
                p.requires_grad = False

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError()


class AuditAutoencoderLSTM(BaseAutoencoder):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 128,
                 num_layers: int = 2, pad_idx: int = 0, dropout: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_idx = pad_idx
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.embedding = nn.Embedding(self.vocab_size + 1, self.emb_dim, padding_idx=self.pad_idx)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.num_layers,
                            batch_first=True, dropout=dropout, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_dim * self.num_directions, self.vocab_size + 1)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


class AuditAutoencoderGRU(BaseAutoencoder):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 128,
                 num_layers: int = 2, pad_idx: int = 0, dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_idx = pad_idx
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(self.vocab_size + 1, self.emb_dim, padding_idx=self.pad_idx)
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=self.num_layers,
                          batch_first=True, dropout=dropout, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_dim * self.num_directions, self.vocab_size + 1)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


class AuditAutoencoderTransformer(BaseAutoencoder):
    def __init__(self, vocab_size: int, emb_dim: int = 128, nhead: int = 4, nlayers: int = 2,
                 dim_feedforward: int = 256, pad_idx: int = 0, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_idx = pad_idx
        self.emb_dim = emb_dim
        self.max_len = max_len

        self.embedding = nn.Embedding(self.vocab_size + 1, self.emb_dim, padding_idx=self.pad_idx)
        self.pos_embedding = nn.Embedding(self.max_len, self.emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.emb_dim, self.vocab_size + 1)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        B, T = x.size()
        if T > self.max_len:
            # truncate inputs to max_len (caller should have padded/truncated)
            x = x[:, :self.max_len]
            T = self.max_len
        emb = self.embedding(x)  # (B,T,emb)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embedding(positions)
        h = emb + pos_emb
        h = self.dropout(h)
        out = self.transformer(h)  # (B,T,emb)
        logits = self.fc(out)
        return logits


# -------------------------
# Factory
# -------------------------
def build_model(arch: str = "lstm", vocab_size: int = None, **kwargs) -> nn.Module:
    if vocab_size is None:
        vocab_size = estimated_vocab_size()
    arch = (arch or "lstm").lower()
    if arch == "lstm":
        return AuditAutoencoderLSTM(vocab_size=vocab_size, **kwargs)
    elif arch == "gru":
        return AuditAutoencoderGRU(vocab_size=vocab_size, **kwargs)
    elif arch in ("transformer", "transformer-lite"):
        return AuditAutoencoderTransformer(vocab_size=vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown arch: {arch}")


# -------------------------
# KL regularizer & scoring helpers
# -------------------------
def compute_kl_regularizer(logits: torch.Tensor, prior: str = "uniform") -> torch.Tensor:
    """
    Compute a simple KL divergence regularizer between per-token predicted distribution and a prior.
    logits: (B,T,V)
    prior: "uniform" or "empirical" (empirical not implemented here)
    Returns: scalar tensor (mean KL over tokens)
    """
    with torch.no_grad():
        B, T, V = logits.size()
    probs = F.softmax(logits, dim=-1)  # (B,T,V)
    if prior == "uniform":
        # KL(probs || uniform) = sum p log (p / (1/V)) = sum p log p + log V
        kl_per = (probs * (probs + 1e-12).log()).sum(dim=-1) + math.log(V)
        return kl_per.mean()
    else:
        # fallback to uniform
        kl_per = (probs * (probs + 1e-12).log()).sum(dim=-1) + math.log(V)
        return kl_per.mean()


def _smoothed_negative_logprob(logits: torch.Tensor, targets: torch.LongTensor, smoothing_window: int = 3) -> Tuple[float, List[float]]:
    """
    Compute negative log-prob per token, then smooth with simple moving average (window size).
    logits: (1,T,V) or (B,T,V) but we process first batch element
    targets: (1,T) or (B,T)
    Returns (avg_score, per_token_scores_list)
    """
    with torch.no_grad():
        if logits.dim() == 3:
            probs = F.softmax(logits, dim=-1)  # (B,T,V)
        else:
            raise ValueError("logits expected shape (B,T,V)")
        B, T, V = probs.size()
        probs = probs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        # negative log prob per token for first batch item
        nlp = []
        for t in range(T):
            idx = int(targets_np[0, t])
            if idx == 0:
                # pad token: assign 0 nll (ignored later)
                nlp.append(0.0)
            else:
                p = float(probs[0, t, idx])
                p = max(p, 1e-12)
                nlp.append(-math.log(p))
        # smooth with moving average
        window = max(1, int(smoothing_window))
        smoothed = []
        for i in range(len(nlp)):
            lo = max(0, i - (window // 2))
            hi = min(len(nlp), lo + window)
            smoothed.append(sum(nlp[lo:hi]) / max(1, (hi - lo)))
        # compute avg only over non-pad tokens
        valid = [v for (v, tgt) in zip(smoothed, targets_np[0]) if int(tgt) != 0]
        avg = float(np.mean(valid)) if len(valid) > 0 else float("inf")
        return avg, smoothed


# -------------------------
# Scoring: preserve old signature but extend
# -------------------------
def score_trace_with_model(model: nn.Module, seq_tokens: Union[List[int], torch.Tensor],
                           pad_idx: int = 0, scoring: str = "smoothed_prob") -> Tuple[float, List[float]]:
    """
    Returns (score, per_token_losses)
    - scoring="ce" => average cross-entropy as previously (compatible)
    - scoring="smoothed_prob" => smoothed negative log-prob (better anomaly separation)
    """
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
        try:
            x = torch.LongTensor([list(seq_tokens)])
        except Exception:
            raise ValueError("seq_tokens cannot be converted to tensor")

    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
    x = x.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(x)  # (B,T,V)
        if scoring == "ce":
            B, T, V = logits.size()
            logits_view = logits.view(B * T, V)
            targets = x.view(B * T)
            criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_idx)
            losses = criterion(logits_view, targets).cpu().numpy().reshape(B, T)[0].tolist()
            mask = (targets.cpu().numpy().reshape(B, T)[0] != pad_idx)
            valid = [l for (l, m) in zip(losses, mask) if m]
            avg = float(np.mean(valid)) if len(valid) > 0 else float("inf")
            return avg, [float(x) for x in losses]
        else:
            # default: smoothed_prob
            targets = x
            avg, per_token = _smoothed_negative_logprob(logits, targets, smoothing_window=3)
            # also return raw per-token ce as second return to maintain backward-compatible shape
            return avg, [float(x) for x in per_token]


# -------------------------
# Utilities: save model with metadata
# -------------------------
def save_model_with_meta(state_dict: Dict, out_path: str, arch: str = "lstm", vocab_size: int = None):
    """
    Save state_dict to out_path and write out_path + ".meta.json" with arch & timestamp.
    """
    torch.save(state_dict, out_path)
    meta = {
        "arch": arch,
        "vocab_size": int(vocab_size) if vocab_size is not None else estimated_vocab_size(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    try:
        with open(out_path + ".meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
    except Exception:
        pass


# -------------------------
# Quick demo (safe)
# -------------------------
if __name__ == "__main__":
    print("model.py upgraded - estimated_vocab_size =", estimated_vocab_size())
    p = Path("dataset/traces")
    if p.exists():
        samples = list(p.glob("*.json"))
        if samples:
            seq = load_trace(str(samples[0]), max_len=200)
            print("Sample tokens len:", len(seq))
            m = build_model("lstm", vocab_size=estimated_vocab_size())
            out = m(torch.LongTensor([seq if len(seq)>0 else [0]*10]))
            print("Forward logits shape:", out.shape)
