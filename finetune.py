# finetune.py — robust fine-tune script (handles load_trace without max_len)
import os, glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# import model module safely
import model as model_mod
from model import AuditAutoencoder, load_trace

# ---- Determine VOCAB size with safe fallbacks ----
if hasattr(model_mod, "estimated_vocab_size"):
    VOCAB = model_mod.estimated_vocab_size()
elif hasattr(model_mod, "_next_token_index"):
    VOCAB = int(getattr(model_mod, "_next_token_index"))
else:
    VOCAB = 200
print(f"[INFO] Using VOCAB size = {VOCAB}")

# ---- Paths & params ----
DATA_DIR = "sandbox_traces"
MODEL_PATH = "ae_model.pth"
FINETUNED_PATH = "ae_model_finetuned.pth"
MAX_LEN = 200

# ---- Helpers to pad/truncate a sequence of token ids ----
def pad_or_truncate(seq, length=MAX_LEN, pad_value=0):
    if len(seq) >= length:
        return seq[:length]
    else:
        return seq + [pad_value] * (length - len(seq))

# ---- Dataset: call load_trace(file) (no max_len arg) and then pad/truncate ----
class TraceDataset(Dataset):
    def __init__(self, files, max_len=MAX_LEN):
        self.files = files
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # call load_trace with a single arg (some model.py versions don't accept max_len)
        seq = load_trace(self.files[idx])
        if not isinstance(seq, (list, tuple)):
            # if load_trace returns a numpy array / tensor, convert to list of ints
            try:
                seq = list(map(int, seq))
            except Exception:
                seq = [int(x) for x in seq]
        seq = pad_or_truncate(seq, self.max_len)
        return torch.LongTensor(seq)

# ---- Collate: create batch (fixed length already) ----
def collate_fixed(batch):
    # batch is list of LongTensor with same length (MAX_LEN)
    return torch.stack(batch, dim=0)

# ---- Main finetune logic ----
def finetune():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if not files:
        print("[ERROR] No sandbox traces found in", DATA_DIR)
        return

    print(f"[INFO] Found {len(files)} traces. Using VOCAB={VOCAB} (max token idx estimate).")

    ds = TraceDataset(files, max_len=MAX_LEN)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fixed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AuditAutoencoder(vocab_size=VOCAB).to(device)

    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Base model not found:", MODEL_PATH)
        return

    # load state dict (robust partial-load)
    state = torch.load(MODEL_PATH, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        print("[WARN] failed to load full state_dict into model; attempting partial load.")
        model_state = model.state_dict()
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
        model.load_state_dict(model_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    EPOCHS = 4
    print("[INFO] Starting fine-tuning...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in dl:
            batch = batch.to(device)               # (B, T)
            optimizer.zero_grad()
            out = model(batch)                     # (B, T, V)
            B, T, V = out.size()
            logits = out.view(B * T, V)            # (B*T, V)
            targets = batch.view(B * T)           # (B*T)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        print(f"[EPOCH] {epoch}/{EPOCHS}  avg_loss = {total_loss / max(1, batches):.4f}")

    # Save finetuned model
    torch.save(model.state_dict(), FINETUNED_PATH)
    try:
        os.replace(FINETUNED_PATH, MODEL_PATH)
        print("[SUCCESS] Saved finetuned model ->", MODEL_PATH)
    except Exception:
        print("[SUCCESS] Saved finetuned model ->", FINETUNED_PATH)
        print("[NOTICE] Could not overwrite base model directly; please replace ae_model.pth manually.")

if __name__ == "__main__":
    finetune()
