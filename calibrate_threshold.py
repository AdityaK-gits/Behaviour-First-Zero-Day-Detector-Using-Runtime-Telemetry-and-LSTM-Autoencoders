import torch
import torch.nn as nn
import glob
import os
import json
import numpy as np
from model import AuditAutoencoder, load_trace

MODEL_PATH = "ae_model.pth"
DATA_DIR = "dataset/traces"
OUTPUT_FILE = "threshold.json"

def calibrate():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train first.")
        return

    # Load Model
    model = AuditAutoencoder(vocab_size=100)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Load Benign Traces
    benign_files = glob.glob(os.path.join(DATA_DIR, "benign_*.json"))
    if not benign_files:
        print("No benign traces found.")
        return

    scores = []
    print(f"[*] Calibrating on {len(benign_files)} benign samples...")
    
    with torch.no_grad():
        for fpath in benign_files:
            try:
                seq = load_trace(fpath)
                x = torch.LongTensor([seq])
                out = model(x)
                loss = criterion(out.view(-1, out.size(-1)), x.view(-1))
                scores.append(loss.mean().item())
            except Exception as e:
                print(f"Error processing {fpath}: {e}")

    if not scores:
        print("No valid scores.")
        return

    scores = np.array(scores)
    
    # Compute stats
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    p95 = np.percentile(scores, 95)
    p99 = np.percentile(scores, 99)
    
    print(f"Stats: Mean={mean_score:.4f}, Std={std_score:.4f}")
    print(f"95th Percentile: {p95:.4f}")
    print(f"99th Percentile: {p99:.4f}")
    
    # Save
    data = {
        "mean": float(mean_score),
        "std": float(std_score),
        "p95": float(p95),
        "p99": float(p99),
        "suggested_threshold": float(p99)
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[*] Thresholds saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    calibrate()
