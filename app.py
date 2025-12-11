# app.py (BFZDD Streamlit Dashboard - safe-ready)
import streamlit as st
import os, glob, json, subprocess, tempfile, time, sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from model import AuditAutoencoder, load_trace

# Config
MODEL_PATH = "ae_model.pth"
THRESH_PATH = "threshold.json"
DATA_DIR = "dataset/traces"
SCRIPTS_DIR = "dataset/scripts"
SANDBOX_RUNNER = "sandbox_runner.py"

# Allow disabling execution via env var for safe public deploys
DISABLE_EXECUTION = os.environ.get("DISABLE_EXECUTION", "false").lower() in ("1","true","yes")

st.set_page_config(page_title="BFZDD Dashboard", layout="wide")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = AuditAutoencoder(vocab_size=100)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    model.eval()
    return model

def load_thresholds():
    if not os.path.exists(THRESH_PATH):
        return None
    with open(THRESH_PATH) as f:
        return json.load(f)

def score_trace_with_model(model, trace_seq):
    device = torch.device("cpu")
    x = torch.LongTensor([trace_seq]).to(device)
    with torch.no_grad():
        out = model(x)
    ce = nn.CrossEntropyLoss(reduction='none')
    logits = out.squeeze(0)
    target = x.squeeze(0)
    losses = ce(logits, target).cpu().numpy()
    score = float(np.mean(losses))
    return score, losses

def compute_scores_for_dataset(model):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    rows = []
    for f in files:
        try:
            seq = load_trace(f)
            score, _ = score_trace_with_model(model, seq)
            label = "malicious" if "malware" in os.path.basename(f) else "benign"
            rows.append({"file": os.path.basename(f), "path": f, "score": score, "label": label})
        except Exception as e:
            rows.append({"file": os.path.basename(f), "path": f, "score": None, "label": "error"})
    return pd.DataFrame(rows)

def run_sandbox_script(script_path, output_trace="trace.json"):
    cmd = [sys.executable, SANDBOX_RUNNER, script_path, "--output", output_trace]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc

st.title("BFZDD — Behaviour-First Zero-Day Detector")
st.markdown("**Warning:** Running uploaded scripts executes code. Only run inside an isolated VM. See VM_SAFETY.md")

model = load_model()
thresholds = load_thresholds()

sidebar_choice = st.sidebar.selectbox("Mode", ["Dataset Review", "Live Analysis", "About & Safety"])

if sidebar_choice == "Dataset Review":
    st.header("Dataset Review")
    if model is None:
        st.warning("Model not found. Place ae_model.pth in the repo or run training.")
    else:
        st.info("Computing scores for traces in `dataset/traces` (cached).")
        df = compute_scores_for_dataset(model)
        st.dataframe(df.sort_values(by="score", ascending=False).reset_index(drop=True))
        try:
            import plotly.express as px
            fig = px.histogram(df, x="score", color="label", nbins=40, marginal="box",
                               title="Reconstruction error distribution")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.write("Plotly not available — showing numeric summary.")
            st.write(df.groupby("label")["score"].describe())

elif sidebar_choice == "Live Analysis":
    st.header("Live Analysis")
    st.markdown("Select an existing script from `dataset/scripts` or upload a script. **DO NOT** run untrusted scripts outside a VM.")
    col1, col2 = st.columns(2)

    scripts = sorted(glob.glob(os.path.join(SCRIPTS_DIR, "*.py"))) if os.path.exists(SCRIPTS_DIR) else []
    sel = col1.selectbox("Choose a sample script (from dataset/scripts)", ["--none--"] + [os.path.basename(s) for s in scripts])
    uploaded = col1.file_uploader("Or upload a Python script (optional)", type=["py"])

    run_in_vm = col1.checkbox("I confirm I will run this only in an isolated VM", value=False)
    execute = col1.button("Run & Analyze (Sandbox)")

    trace_file = col2.selectbox("Or select precomputed trace (dataset/traces)", ["--none--"] + [os.path.basename(p) for p in sorted(glob.glob(os.path.join(DATA_DIR,"*.json")))])
    analyze_trace_btn = col2.button("Analyze selected trace")

    if execute:
        if DISABLE_EXECUTION:
            st.error("Execution is disabled in this deployment (DISABLE_EXECUTION=true).")
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
                    proc = run_sandbox_script(script_path, output_trace=trace_out)
                    st.text("Sandbox stdout/stderr:")
                    st.text(proc.stdout)
                    st.text(proc.stderr)
                    if os.path.exists(trace_out):
                        st.success("Trace produced. Loading and scoring...")
                        seq = load_trace(trace_out)
                        score, losses = score_trace_with_model(model, seq)
                        st.metric("Anomaly Score", f"{score:.4f}", delta=(score - thresholds.get("mean",0)) if thresholds else None)
                        suggested = thresholds.get("suggested_threshold") if thresholds else None
                        if suggested and score > suggested:
                            st.error(f"Verdict: QUARANTINE (score > {suggested:.4f})")
                        else:
                            st.success("Verdict: Benign or below threshold")
                        idxs = np.argsort(-losses)[:5]
                        try:
                            with open(trace_out) as f:
                                events = json.load(f)
                        except:
                            events = []
                        anomalous = []
                        for i in idxs:
                            if i < len(events):
                                anomalous.append({"idx": int(i), "event": events[i].get("event"), "args": events[i].get("args"), "error": float(losses[i])})
                        st.markdown("### Top anomalous events")
                        st.table(pd.DataFrame(anomalous))
                    else:
                        st.error("Trace not produced. Check sandbox logs.")

    if analyze_trace_btn:
        if trace_file == "--none--":
            st.error("No trace selected.")
        else:
            trace_path = os.path.join(DATA_DIR, trace_file)
            seq = load_trace(trace_path)
            score, losses = score_trace_with_model(model, seq)
            st.metric("Anomaly Score", f"{score:.4f}")
            suggested = thresholds.get("suggested_threshold") if thresholds else None
            if suggested and score > suggested:
                st.error(f"Verdict: QUARANTINE (score > {suggested:.4f})")
            else:
                st.success("Verdict: Benign or below threshold")
            with open(trace_path) as f:
                events = json.load(f)
            idxs = np.argsort(-losses)[:5]
            anomalous = []
            for i in idxs:
                if i < len(events):
                    anomalous.append({"idx": int(i), "event": events[i].get("event"), "args": events[i].get("args"), "error": float(losses[i])})
            st.markdown("### Top anomalous events")
            st.table(pd.DataFrame(anomalous))

elif sidebar_choice == "About & Safety":
    st.header("About BFZDD")
    st.write("Behaviour-First Zero-Day Detector — prototype demo.")
    st.subheader("Safety & Deployment Notes")
    st.markdown("""
    - **Local VM only:** To run scripts, start this Streamlit app inside an isolated VM (VirtualBox/Hyper-V) with networking disabled.
    - **Public deployments:** If you want to deploy publicly (Streamlit Cloud), set DISABLE_EXECUTION=true so uploaded scripts cannot be executed.
    - See VM_SAFETY.md for step-by-step VM setup.
    """)

st.sidebar.markdown("---")
st.sidebar.write("Model: " + ("Loaded" if model else "Not found"))
st.sidebar.write("Thresholds: " + ("Loaded" if thresholds else "Not found"))
