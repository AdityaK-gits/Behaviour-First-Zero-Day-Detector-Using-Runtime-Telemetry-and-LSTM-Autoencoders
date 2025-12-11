# app.py — BFZDD Streamlit Dashboard (edited, safer + upload support)
# Replace your existing app.py with this file.
# Features added:
# - Upload ae_model.pth at runtime (session-only) and auto-load
# - Graceful handling when model is missing (no crashes)
# - DISABLE_EXECUTION env var respected (safe public deploys)
# - Friendly messages, logging and protected scoring
# - Disabled Run & Analyze when dangerous or missing model

import os
import sys
import glob
import json
import tempfile
import subprocess
import logging

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Import model utilities (must exist in repo)
# model.py must define AuditAutoencoder and load_trace(path)
from model import AuditAutoencoder, load_trace

# ---------------------------
# Configuration
# ---------------------------
MODEL_FILENAME = "ae_model.pth"          # repo model (persistent)
UPLOADED_MODEL_FILENAME = "ae_model_upload.pth"  # session/uploaded model (transient)
THRESH_PATH = "threshold.json"
DATA_DIR = "dataset/traces"
SCRIPTS_DIR = "dataset/scripts"
SANDBOX_RUNNER = "sandbox_runner.py"     # used only when DISABLE_EXECUTION is False

# If deploying publicly, set this env var in Streamlit Cloud: DISABLE_EXECUTION=true
DISABLE_EXECUTION = os.environ.get("DISABLE_EXECUTION", "false").lower() in ("1", "true", "yes")

# Configure logging to a file (useful for Streamlit Cloud logs)
LOGFILE = "app_runtime.log"
logging.basicConfig(filename=LOGFILE, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------
# Helper functions
# ---------------------------

@st.cache_resource
def _load_model_from_path(path):
    """Load a model state dict from given path into an AuditAutoencoder and return it.
    Returns None on failure."""
    try:
        model = AuditAutoencoder(vocab_size=100)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        logging.info(f"Loaded model from {path}")
        return model
    except Exception as e:
        logging.exception("Failed to load model from %s", path)
        return None

def load_model():
    """Try in order: uploaded model (session), then repo model (ae_model.pth)."""
    # uploaded model (if present) should override repo model
    if os.path.exists(UPLOADED_MODEL_FILENAME):
        m = _load_model_from_path(UPLOADED_MODEL_FILENAME)
        if m is not None:
            return m
    if os.path.exists(MODEL_FILENAME):
        m = _load_model_from_path(MODEL_FILENAME)
        if m is not None:
            return m
    return None

def load_thresholds():
    if not os.path.exists(THRESH_PATH):
        return None
    try:
        with open(THRESH_PATH) as f:
            return json.load(f)
    except Exception:
        logging.exception("Failed to load thresholds")
        return None

def score_trace_with_model(model, trace_seq):
    """Compute per-step CrossEntropyLoss and mean score. Raises on model errors."""
    if model is None:
        raise ValueError("Model is None")
    device = torch.device("cpu")
    x = torch.LongTensor([trace_seq]).to(device)  # (1, T)
    with torch.no_grad():
        out = model(x)  # (1, T, vocab)
    ce = nn.CrossEntropyLoss(reduction='none')
    logits = out.squeeze(0)   # (T, vocab)
    target = x.squeeze(0)    # (T,)
    losses = ce(logits, target).cpu().numpy()  # (T,)
    score = float(np.mean(losses))
    return score, losses

def compute_scores_for_dataset(model):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    rows = []
    for f in files:
        try:
            seq = load_trace(f)
            if model is None:
                score = None
            else:
                score, _ = score_trace_with_model(model, seq)
            label = "malicious" if "malware" in os.path.basename(f) else "benign"
            rows.append({"file": os.path.basename(f), "path": f, "score": score, "label": label})
        except Exception:
            logging.exception("Failed scoring dataset file: %s", f)
            rows.append({"file": os.path.basename(f), "path": f, "score": None, "label": "error"})
    return pd.DataFrame(rows)

def run_sandbox_script(script_path, output_trace="trace.json"):
    """Run sandbox_runner.py as a subprocess (blocking). Returns subprocess.CompletedProcess."""
    cmd = [sys.executable, SANDBOX_RUNNER, script_path, "--output", output_trace]
    logging.info("Executing sandbox: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logging.info("Sandbox finished. returncode=%s", proc.returncode)
    if proc.stdout:
        logging.info("Sandbox stdout: %s", proc.stdout[:2000])
    if proc.stderr:
        logging.info("Sandbox stderr: %s", proc.stderr[:2000])
    return proc

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="BFZDD Dashboard", layout="wide")
st.title("BFZDD — Behaviour-First Zero-Day Detector")
st.markdown("**Warning:** Running uploaded scripts executes code. Only run inside an isolated VM. See VM_SAFETY.md")

# Sidebar: model upload & status
st.sidebar.header("Model & Environment")

# Upload a model (session-only). This is optional and convenient on Streamlit Cloud.
uploaded_model_file = st.sidebar.file_uploader("Upload ae_model.pth (optional)", type=["pth", "pt"])
if uploaded_model_file is not None:
    try:
        # Save uploaded file to workspace and attempt to load
        with open(UPLOADED_MODEL_FILENAME, "wb") as out_f:
            out_f.write(uploaded_model_file.getbuffer())
        st.sidebar.info("Uploaded model saved for this session.")
        logging.info("User uploaded model to session.")
    except Exception:
        logging.exception("Failed to save uploaded model")
        st.sidebar.error("Failed to save uploaded model (see logs).")

# Load model (uploaded overrides repo file)
model = load_model()
thresholds = load_thresholds()

# Show status
if model is None:
    st.sidebar.warning("Model: Not found — place ae_model.pth in repo or upload one above.")
else:
    st.sidebar.success("Model: Loaded")

if thresholds is None:
    st.sidebar.warning("Thresholds: Not found")
else:
    st.sidebar.success("Thresholds: Loaded")

# Environment flag
if DISABLE_EXECUTION:
    st.sidebar.info("Execution: DISABLED (DISABLE_EXECUTION=true) — safe public mode")
else:
    st.sidebar.info("Execution: ENABLED (local/VM mode allowed)")

# Main navigation
mode = st.sidebar.selectbox("Mode", ["Dataset Review", "Live Analysis", "About & Safety"])

# ---------------------------
# Dataset Review
# ---------------------------
if mode == "Dataset Review":
    st.header("Dataset Review")
    if model is None:
        st.warning("Model not found. Place ae_model.pth in the repo or upload one via sidebar to enable scoring.")
    df = compute_scores_for_dataset(model)
    st.dataframe(df.sort_values(by="score", ascending=False).reset_index(drop=True))
    # show distribution where scores exist
    try:
        import plotly.express as px
        if df['score'].notnull().any():
            fig = px.histogram(df[df['score'].notnull()], x="score", color="label", nbins=40, marginal="box",
                               title="Reconstruction error distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scores available (no model loaded).")
    except Exception:
        st.write("Install plotly for nicer charts or view numeric summary.")
        st.write(df.groupby("label")["score"].describe())

# ---------------------------
# Live Analysis
# ---------------------------
elif mode == "Live Analysis":
    st.header("Live Analysis")
    st.markdown("Select an existing script from `dataset/scripts` or upload a script. **DO NOT** run untrusted scripts outside a VM.")
    col1, col2 = st.columns(2)

    # list available scripts
    scripts = sorted(glob.glob(os.path.join(SCRIPTS_DIR, "*.py"))) if os.path.exists(SCRIPTS_DIR) else []
    sel = col1.selectbox("Choose a sample script (from dataset/scripts)", ["--none--"] + [os.path.basename(s) for s in scripts])
    uploaded = col1.file_uploader("Or upload a Python script (optional)", type=["py"])

    run_in_vm = col1.checkbox("I confirm I will run this only in an isolated VM", value=False)

    # Decide disabling logic
    exec_disabled = DISABLE_EXECUTION or (model is None)
    if exec_disabled:
        if DISABLE_EXECUTION:
            exec_hint = "Execution disabled by server config (DISABLE_EXECUTION=true)."
        else:
            exec_hint = "Model missing: upload ae_model.pth or place it in repo to enable execution."
        col1.info(exec_hint)

    # Buttons
    execute = col1.button("Run & Analyze (Sandbox)", disabled=exec_disabled)
    trace_file_list = ["--none--"] + [os.path.basename(p) for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))]
    trace_file = col2.selectbox("Or select precomputed trace (dataset/traces)", trace_file_list)
    analyze_trace_btn = col2.button("Analyze selected trace", disabled=(model is None))

    # Handle Run & Analyze (sandbox)
    if execute:
        # double-check environment safety
        if DISABLE_EXECUTION:
            st.error("Execution disabled by server config. Enable execution locally in a VM to run sandbox.")
        elif not run_in_vm:
            st.error("You must confirm VM isolation before running scripts.")
        else:
            if uploaded is None and sel == "--none--":
                st.error("No script selected or uploaded.")
            else:
                with st.spinner("Running script in sandbox..."):
                    tmpdir = tempfile.mkdtemp()
                    if uploaded is not None:
                        script_path = os.path.join(tmpdir, uploaded.name)
                        with open(script_path, "wb") as f:
                            f.write(uploaded.getvalue())
                    else:
                        script_path = scripts[[os.path.basename(s) for s in scripts].index(sel)]
                    trace_out = os.path.join(tmpdir, "trace.json")
                    try:
                        proc = run_sandbox_script(script_path, output_trace=trace_out)
                    except Exception:
                        logging.exception("Sandbox runner invocation failed.")
                        st.error("Sandbox execution failed. Check app logs.")
                        proc = None

                    if proc is not None:
                        if proc.stdout:
                            st.text("Sandbox stdout:")
                            st.text(proc.stdout)
                        if proc.stderr:
                            st.text("Sandbox stderr:")
                            st.text(proc.stderr)

                        if os.path.exists(trace_out):
                            st.success("Trace produced. Loading and scoring...")
                            try:
                                seq = load_trace(trace_out)
                            except Exception:
                                logging.exception("Failed to load trace produced by sandbox")
                                st.error("Failed to parse produced trace. See logs.")
                                seq = None

                            if seq is not None:
                                try:
                                    score, losses = score_trace_with_model(model, seq)
                                except Exception:
                                    logging.exception("Scoring failed for sandbox trace")
                                    st.error("Scoring failed. See app logs for details.")
                                    score, losses = None, None

                                if score is not None:
                                    st.metric("Anomaly Score", f"{score:.4f}", delta=(score - thresholds.get("mean",0)) if thresholds else None)
                                    suggested = thresholds.get("suggested_threshold") if thresholds else None
                                    if suggested and score > suggested:
                                        st.error(f"Verdict: QUARANTINE (score > {suggested:.4f})")
                                    else:
                                        st.success("Verdict: Benign or below threshold")
                                    # top anomalous events
                                    idxs = np.argsort(-losses)[:5] if losses is not None else []
                                    try:
                                        with open(trace_out) as f:
                                            events = json.load(f)
                                    except Exception:
                                        events = []
                                    anomalous = []
                                    for i in idxs:
                                        if i < len(events):
                                            anomalous.append({"idx": int(i), "event": events[i].get("event"), "args": events[i].get("args"), "error": float(losses[i])})
                                    st.markdown("### Top anomalous events")
                                    if anomalous:
                                        st.table(pd.DataFrame(anomalous))
                                    else:
                                        st.write("No anomalous events detected or no losses available.")
                                else:
                                    st.warning("Score unavailable (scoring error).")
                        else:
                            st.error("Trace not produced. Check sandbox logs.")
    # Handle Analyze selected trace
    if analyze_trace_btn:
        if trace_file == "--none--":
            st.error("No trace selected.")
        else:
            trace_path = os.path.join(DATA_DIR, trace_file)
            try:
                seq = load_trace(trace_path)
            except Exception:
                logging.exception("Failed to load selected trace: %s", trace_path)
                st.error("Failed to parse selected trace. See logs.")
                seq = None

            if seq is not None:
                try:
                    score, losses = score_trace_with_model(model, seq)
                except Exception:
                    logging.exception("Scoring failed for selected trace")
                    st.error("Scoring failed. See app logs.")
                    score, losses = None, None

                if score is not None:
                    st.metric("Anomaly Score", f"{score:.4f}")
                    suggested = thresholds.get("suggested_threshold") if thresholds else None
                    if suggested and score > suggested:
                        st.error(f"Verdict: QUARANTINE (score > {suggested:.4f})")
                    else:
                        st.success("Verdict: Benign or below threshold")
                    try:
                        with open(trace_path) as f:
                            events = json.load(f)
                    except Exception:
                        events = []
                    idxs = np.argsort(-losses)[:5] if losses is not None else []
                    anomalous = []
                    for i in idxs:
                        if i < len(events):
                            anomalous.append({"idx": int(i), "event": events[i].get("event"), "args": events[i].get("args"), "error": float(losses[i])})
                    st.markdown("### Top anomalous events")
                    if anomalous:
                        st.table(pd.DataFrame(anomalous))
                    else:
                        st.write("No anomalous events detected or no losses available.")

# ---------------------------
# About & Safety Page
# ---------------------------
elif mode == "About & Safety":
    st.header("About BFZDD")
    st.write("Behaviour-First Zero-Day Detector — prototype demo.")
    st.subheader("Safety & Deployment Notes")
    st.markdown("""
    - **Local VM only:** To run scripts, start this Streamlit app inside an isolated VM (VirtualBox/Hyper-V) with networking disabled.
    - **Public deployments:** If you want to deploy publicly (Streamlit Cloud), set `DISABLE_EXECUTION=true` so uploaded scripts cannot be executed.
    - See VM_SAFETY.md for step-by-step VM setup.
    """)
    st.markdown("### Logs & troubleshooting")
    st.markdown(f"Application logs (recent) are written to `{LOGFILE}` in the app root. On Streamlit Cloud view 'Manage App -> Logs' to inspect failures.")

# Footer: quick status
st.sidebar.markdown("---")
st.sidebar.write("Model: " + ("Loaded" if model else "Not found"))
st.sidebar.write("Thresholds: " + ("Loaded" if thresholds else "Not found"))
