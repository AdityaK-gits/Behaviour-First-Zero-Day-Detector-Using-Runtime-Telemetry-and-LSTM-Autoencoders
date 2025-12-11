# app.py - BFZDD Streamlit dashboard (refactored, robust, with diagnostics)
import os
import sys
import json
import glob
import time
from pathlib import Path
from typing import List

import streamlit as st
import pandas as pd

# Diagnostics: print to logs so we can inspect runtime file layout on Streamlit Cloud
print("[DEBUG] APP START")
print("[DEBUG] process id:", os.getpid())
print("[DEBUG] cwd:", os.getcwd())
print("[DEBUG] python:", sys.version.replace('\n', ' '))
print("[DEBUG] top-level files:")
try:
    print("[DEBUG]", os.listdir("."))
except Exception as e:
    print("[DEBUG] listdir error:", e)

# Safety: check environment switch to disable code execution in public deployments
DISABLE_EXECUTION = os.environ.get("DISABLE_EXECUTION", "false").lower() in ("1", "true", "yes")
print(f"[DEBUG] DISABLE_EXECUTION={DISABLE_EXECUTION}")

# Constants
REPO_ROOT = Path(".")
DATASET_DIR = REPO_ROOT / "dataset"
TRACES_DIR = DATASET_DIR / "traces"
SCRIPTS_DIR = DATASET_DIR / "scripts"
MODEL_FILENAMES = ["ae_model.pth", "./ae_model.pth", str(REPO_ROOT / "ae_model.pth")]
THRESHOLD_PATH = REPO_ROOT / "threshold.json"
MAX_LEN = 200

# Import model module robustly
try:
    import model as model_mod
    from model import AuditAutoencoder, load_trace
    print("[DEBUG] Imported model module successfully")
except Exception as e:
    model_mod = None
    AuditAutoencoder = None
    load_trace = None
    print("[DEBUG] Could not import model module:", e)


# Utility: try to load a torch model file path (safe, returns None on failure)
def try_load_model_from_path(path: str):
    import torch
    try:
        print(f"[DEBUG] Attempting torch.load -> {path}")
        state = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[DEBUG] torch.load failed for {path}: {e}")
        return None

    # Build model if AuditAutoencoder available
    if AuditAutoencoder is None:
        print("[DEBUG] AuditAutoencoder not available in model.py, cannot construct model")
        return None

    # Determine vocab size safely
    vocab = 200
    if model_mod is not None:
        if hasattr(model_mod, "estimated_vocab_size"):
            try:
                vocab = model_mod.estimated_vocab_size()
            except Exception:
                vocab = int(getattr(model_mod, "_next_token_index", 200))
        else:
            vocab = int(getattr(model_mod, "_next_token_index", 200))

    try:
        model = AuditAutoencoder(vocab_size=vocab)
        # Attempt load; allow partial load if shapes mismatch
        try:
            model.load_state_dict(state)
        except Exception:
            print("[DEBUG] full state load failed, attempting partial load / shape-safe load")
            ms = model.state_dict()
            for k, v in state.items():
                if k in ms and ms[k].shape == v.shape:
                    ms[k] = v
            model.load_state_dict(ms)
        model.eval()
        return model
    except Exception as e:
        print(f"[DEBUG] Failed to instantiate or load AuditAutoencoder: {e}")
        return None


# Load thresholds if present
def load_thresholds():
    if THRESHOLD_PATH.exists():
        try:
            with open(THRESHOLD_PATH, "r") as f:
                data = json.load(f)
            print("[DEBUG] Thresholds loaded:", data)
            return data
        except Exception as e:
            print("[DEBUG] Failed to load threshold.json:", e)
            return {}
    else:
        print("[DEBUG] threshold.json not found")
        return {}


# Score a single trace sequence using the model.
# returns (avg_loss, per_token_losses_list)
def score_sequence_with_model(model, seq_tokens: List[int]):
    import torch
    import torch.nn as nn
    if model is None:
        raise RuntimeError("Model is None")

    # Ensure tensor shape (1, T)
    x = torch.LongTensor([seq_tokens])
    with torch.no_grad():
        out = model(x)  # expected (B, T, V)
        B, T, V = out.shape
        # cross-entropy expects (N, V) and targets (N,)
        logits = out.view(B * T, V)
        targets = x.view(B * T)
        criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
        losses = criterion(logits, targets).cpu().numpy()
        # reshape to (B, T)
        per_token = losses.reshape(B, T)[0].tolist()
        avg = float(sum(per_token) / len(per_token)) if len(per_token) > 0 else float("inf")
        return avg, per_token


# Helper to pad/truncate token lists
def pad_or_truncate(seq, length=MAX_LEN, pad_value=0):
    if len(seq) >= length:
        return seq[:length]
    else:
        return seq + [pad_value] * (length - len(seq))


# Safe wrapper for load_trace(file_path) - returns list of ints and the json-events list (if available)
def safe_load_trace_and_events(trace_path):
    # Try to load sequence tokens through load_trace if available
    events = []
    seq = []
    try:
        # if the trace file is JSON of events, load to get events mapping
        with open(trace_path, "r") as f:
            events = json.load(f)
    except Exception:
        events = []

    # Try to call load_trace() (model.py)
    if load_trace is not None:
        try:
            seq_raw = load_trace(str(trace_path))
            # convert sequences to plain python ints list
            if hasattr(seq_raw, "tolist"):
                seq = list(seq_raw.tolist())
            elif isinstance(seq_raw, (list, tuple)):
                seq = list(map(int, seq_raw))
            else:
                seq = list(map(int, seq_raw))
        except TypeError:
            # maybe load_trace needs different args; try without args or with only path
            try:
                seq = list(map(int, load_trace(str(trace_path))))
            except Exception as e:
                print("[DEBUG] load_trace call failed:", e)
                seq = []
        except Exception as e:
            print("[DEBUG] load_trace error:", e)
            seq = []
    else:
        print("[DEBUG] load_trace not defined in model.py")
        seq = []

    return seq, events


# Streamlit UI
st.set_page_config(layout="wide", page_title="BFZDD — Behaviour-First Zero-Day Detector")

st.title("BFZDD — Behaviour-First Zero-Day Detector")
st.markdown("**Warning:** Running uploaded scripts executes code. Only run inside an isolated VM. See VM_SAFETY.md")

# Sidebar - Model & Environment
st.sidebar.header("Model & Environment")

# Upload model file via sidebar (session only)
uploaded_model = st.sidebar.file_uploader("Upload ae_model.pth (optional)", type=["pth", "pt"], help="Upload a PyTorch .pth/.pt model file to use for scoring (session-only)")

# Show threshold status
thresholds = load_thresholds()
st.sidebar.success("Thresholds: Loaded" if thresholds else "Thresholds: Not found")

# Execution policy
if DISABLE_EXECUTION:
    st.sidebar.error("Execution: DISABLED (public deployment)")
else:
    st.sidebar.info("Execution: ENABLED (local/VM allowed)")

# Attempt to load model
model = None
model_source = None

# Priority: uploaded model (session)
if uploaded_model is not None:
    # save uploaded file to a local path for loading
    temp_model_path = "/tmp/ae_model_uploaded.pth"
    try:
        with open(temp_model_path, "wb") as out_f:
            out_f.write(uploaded_model.getbuffer())
        model = try_load_model_from_path(temp_model_path)
        if model is not None:
            model_source = f"uploaded:{temp_model_path}"
            st.sidebar.success("Model: Loaded (uploaded)")
        else:
            st.sidebar.warning("Uploaded model: failed to load")
    except Exception as e:
        st.sidebar.error(f"Failed saving uploaded model: {e}")

# Next: try repo root files
if model is None:
    for candidate in MODEL_FILENAMES:
        try:
            if os.path.exists(candidate):
                m = try_load_model_from_path(candidate)
                if m is not None:
                    model = m
                    model_source = f"repo:{candidate}"
                    st.sidebar.success(f"Model: Loaded ({candidate})")
                    break
                else:
                    print(f"[DEBUG] Candidate model {candidate} exists but failed to load")
            else:
                print(f"[DEBUG] Candidate model {candidate} does not exist")
        except Exception as e:
            print("[DEBUG] error checking candidate", candidate, e)

if model is None:
    st.sidebar.warning("Model: Not found — place ae_model.pth in repo or upload one above.")
else:
    st.sidebar.success(f"Model source: {model_source}")

# Main UI: Mode selection
mode = st.sidebar.selectbox("Mode", ["Dataset Review", "Live Analysis", "About & Safety"])

if mode == "About & Safety":
    st.header("About BFZDD")
    st.write(
        "Behaviour-First Zero-Day Detector — prototype demo. "
        "Only run uploaded scripts in an isolated VM. See VM_SAFETY.md for instructions."
    )
    st.subheader("Safety & Deployment Notes")
    st.markdown(
        "- Local VM only: Run the Streamlit app inside an isolated VM when analyzing unknown scripts.\n"
        "- Public deployments: set `DISABLE_EXECUTION=true` in the environment so uploaded scripts will not be executed.\n"
        "- Use snapshots and network isolation for any sandbox runs."
    )

elif mode == "Dataset Review":
    st.header("Dataset Review")
    # List traces
    trace_files = sorted(glob.glob(str(TRACES_DIR / "*.json")))
    trace_rows = []
    for t in trace_files:
        label = "unknown"
        if "benign" in os.path.basename(t).lower():
            label = "benign"
        if "malware" in os.path.basename(t).lower():
            label = "malicious"
        trace_rows.append({"file": os.path.basename(t), "path": t, "score": None, "label": label})

    df = pd.DataFrame(trace_rows)
    st.table(df)

    if model is None:
        st.info("No scores available (no model loaded).")
    else:
        # Score all traces
        st.info("Scoring traces with loaded model (this may take a few seconds)...")
        results = []
        for row in trace_rows:
            path = row["path"]
            seq, events = safe_load_trace_and_events(path)
            if not seq:
                results.append({"file": row["file"], "score": None, "label": row["label"]})
                continue
            seq = pad_or_truncate(seq, MAX_LEN)
            try:
                score, per_token = score_sequence_with_model(model, seq)
                results.append({"file": row["file"], "score": float(score), "label": row["label"]})
            except Exception as e:
                print("[DEBUG] scoring error for", path, e)
                results.append({"file": row["file"], "score": None, "label": row["label"]})

        res_df = pd.DataFrame(results)
        st.table(res_df)

        # Show histogram
        try:
            import plotly.express as px
            fig = px.histogram(res_df, x="score", color="label", nbins=20, title="Reconstruction error distribution")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.write("Plotly not available; showing simple scores.")
            st.write(res_df)

elif mode == "Live Analysis":
    st.header("Live Analysis")
    st.markdown("Select an existing script from `dataset/scripts` or upload a script. **DO NOT** run untrusted scripts outside a VM.")

    # choose sample script
    scripts = sorted(glob.glob(str(SCRIPTS_DIR / "*.py")))
    script_options = ["--none--"] + [os.path.basename(s) for s in scripts]
    chosen = st.selectbox("Choose a sample script (from dataset/scripts)", script_options)

    # choose precomputed trace
    traces = sorted(glob.glob(str(TRACES_DIR / "*.json")))
    trace_options = ["--none--"] + [os.path.basename(t) for t in traces]
    chosen_trace = st.selectbox("Or select precomputed trace (dataset/traces)", trace_options)

    upload_file = st.file_uploader("Or upload a Python script (optional)", type=["py"], help="Upload a script to run inside your own VM; do not run untrusted code here.")

    confirm_vm = st.checkbox("I confirm I will run this only in an isolated VM", value=False)

    # diagnostics toggle
    show_diag = st.checkbox("Show diagnostics (len/events sample)", False)

    # Execution controls
    can_execute = (not DISABLE_EXECUTION) and confirm_vm

    if not can_execute:
        st.warning("Execution is disabled (either DISABLE_EXECUTION is true or you have not confirmed VM usage).")
    else:
        st.success("Execution allowed: you confirmed and DISABLE_EXECUTION is false.")

    # Score using precomputed trace
    if st.button("Analyze selected trace"):
        chosen_path = None
        if chosen_trace != "--none--":
            chosen_path = TRACES_DIR / chosen_trace
        elif chosen != "--none--" and (SCRIPTS_DIR / chosen).exists():
            # If a script was selected but no trace, we cannot run it here (dangerous).
            st.warning("No precomputed trace selected. Please run the script in your VM using sandbox_runner.py to produce a trace.json and then upload it here.")
            chosen_path = None
        elif upload_file is not None:
            st.warning("You uploaded a script. Please run it in a VM and upload the produced trace.json instead of running it here.")
            chosen_path = None

        if chosen_path is not None and model is None:
            st.error("Model missing: upload ae_model.pth or place it in repo to enable scoring.")
            chosen_path = None

        if chosen_path is not None and chosen_path.exists():
            # Score the precomputed trace
            with st.spinner("Loading and scoring..."):
                seq, events = safe_load_trace_and_events(str(chosen_path))
                if not seq:
                    st.error("Trace parsing failed or trace empty.")
                else:
                    seq = pad_or_truncate(seq, MAX_LEN)
                    try:
                        score, per_token = score_sequence_with_model(model, seq)
                        st.metric("Anomaly Score", f"{score:.4f}")
                        # verdict
                        thr = thresholds.get("suggested_threshold")
                        if thr is not None:
                            verdict = "QUARANTINE (score > threshold)" if score > float(thr) else "OK (score <= threshold)"
                            st.error(f"Verdict: {verdict} (threshold = {thr})" if score > float(thr) else f"Verdict: OK (threshold = {thr})")
                        else:
                            st.info("No threshold configured (threshold.json missing).")

                        # Top anomalous events: pick top-k token losses
                        try:
                            import numpy as np
                            per_token = list(map(float, per_token))
                            arr = np.array(per_token)
                            # top 10 indices
                            topk = arr.argsort()[-10:][::-1]
                            rows = []
                            for idx in topk:
                                # map index to event if events list is loaded and same length
                                event_repr = None
                                if isinstance(events, list) and idx < len(events):
                                    e = events[idx]
                                    event_repr = json.dumps(e) if not isinstance(e, str) else str(e)
                                else:
                                    event_repr = f"token_index:{idx}"
                                rows.append({"idx": int(idx), "event": event_repr, "error": float(arr[idx])})
                            anomalous = pd.DataFrame(rows)
                            if anomalous.empty:
                                st.info("Top anomalous events: No entries to show.")
                            else:
                                st.subheader("Top anomalous events")
                                st.table(anomalous)
                        except Exception as e:
                            print("[DEBUG] top anomalous events calculation failed:", e)
                            st.info("Could not compute top anomalous events.")

                    except Exception as e:
                        st.error(f"Scoring failed: {e}")
        else:
            st.info("No precomputed trace selected. Pick a trace from the dropdown (dataset/traces) or upload one.")


# Footer diagnostics for quick debugging in UI (not a substitute for logs)
st.sidebar.markdown("---")
st.sidebar.write("Debug / quick info:")
try:
    files_here = os.listdir(".")
    st.sidebar.write(f"Repo files: {files_here[:12]}")
except Exception as e:
    st.sidebar.write("Could not list files:", e)

if st.sidebar.checkbox("Show DEBUG log (print)"):
    st.sidebar.text("Check app logs for [DEBUG] lines (Manage app -> Logs).")

# End of app
