# versioning.py — simple local model version registry
import os, json, time, shutil, subprocess
from pathlib import Path
import hashlib

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
REGISTRY = MODELS_DIR / "versions.json"
if not REGISTRY.exists():
    REGISTRY.write_text("[]")

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit_hash():
    try:
        out = subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
        return out
    except Exception:
        return None

def save_model_version(source_model_path, name=None, note=None):
    ts = time.strftime("%Y%m%dT%H%M%S")
    commit = get_git_commit_hash()
    name = name or f"ae_model_{ts}.pth"
    dest = MODELS_DIR / name
    shutil.copy2(source_model_path, dest)
    sha = compute_sha256(dest)
    entry = {"name": name, "path": str(dest), "timestamp": ts, "git_commit": commit, "sha256": sha, "note": note}
    arr = json.loads(REGISTRY.read_text())
    arr.insert(0, entry)
    REGISTRY.write_text(json.dumps(arr, indent=2))
    return entry

def list_versions():
    return json.loads(REGISTRY.read_text())
