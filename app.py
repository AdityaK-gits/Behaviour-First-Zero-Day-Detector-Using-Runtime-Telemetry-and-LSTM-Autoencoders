# app.py  — BFZDD Dashboard (refactored with ROC, ConfMatrix, Trace viz, Heatmap, Versioning)
import os
import json
import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# ML
import torch

# local helpers (new)
import utils_viz
import versioning
from model import AuditAutoencoder, load_trace, score_trace_with_model

# Config
REPO_MODEL_NAME = "ae_model.pth"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Safety flag — when deploying public, set DISABLE_EXECUTION=true
DISABLE_EXECUTION = os.environ.get("DISABLE_EXECUTION", "false").lower() in ("1","true","yes")

st.set_page_config(page_title="BFZDD — Behaviour-First Zero-Day Detector", layout="wide")

# --- Sidebar ---
st.sidebar.title("Model & Environment")
st.sidebar.markdown("Upload `ae_model.pth` (optional). If you upload a model it will be used for the session only.")
uploaded_model = st.sidebar.file_uploader("Upload ae_model.pth (optional)", type=["pth","pt"])

# Model status placeholders
model_loaded = False
model_source = None
model_obj = None

# Load thresholds if present
THRESHOLD_FILE = Path("threshold.json")
thresholds = {}
if THRESHOLD_FILE.exists():
    try:
        thresholds = json.loads(THRESHOLD_FILE.read_text())
    except Exception:
        thresholds = {}

# Attempt to load model: prefer uploaded (session), then repo file
def try_load_model_from_path(path):
    try:
        # default vocab_size; model.load should accept partial loads
        m = AuditAutoencoder(vocab_size=200)
        state = torch.load(path, map_location="cpu")
        m.load_state_dict(state, strict=False)
        m.eval()
        return m
    except Exception as e:
        st.sidebar.error(f"Failed to load model from {path}: {e}")
        return None

# 1) Uploaded model
if uploaded_model is not None:
    with st.sidebar.spinner("Loading uploaded model..."):
        tmp = Path(tempfile.gettempdir()) / f"uploaded_ae_{int(time.time())}.pth"
        with open(tmp, "wb") as f:
            f.write(uploaded_model.getbuffer())
        model_obj = try_load_model_from_path(str(tmp))
        if model_obj:
            model_loaded = True
            model_source = f"uploaded:{uploaded_model.name}"
            st.sidebar.success("Uploaded model loaded for session.")

# 2) Repo model fallback
if (not model_loaded) and Path(REPO_MODEL_NAME).exists():
    with st.sidebar.spinner("Loading model from repo..."):
        model_obj = try_load_model_from_path(REPO_MODEL_NAME)
        if model_obj:
            model_loaded = True
            model_source = f"repo:{REPO_MODEL_NAME}"
            st.sidebar.success(f"Model: Loaded ({REPO_MODEL_NAME})")

if not model_loaded:
    st.sidebar.warning("Model: Not found — place ae_model.pth in repo or upload one above.")
else:
    st.sidebar.info(f"Model source:\n{model_source}")

# Show threshold summary
if thresholds:
    st.sidebar.success("Thresholds: Loaded")
    st.sidebar.markdown(f"**suggested_threshold**: `{thresholds.get('suggested_threshold', 'n/a')}`")

# Execution safety note
exec_status = "DISABLED (public)" if DISABLE_EXECUTION else "ENABLED (local/VM allowed)"
st.sidebar.info(f"Execution: {exec_status}")

# Model versions panel (v1)
st.sidebar.subheader("Model versions")
if (versioning.list_versions()):
    versions = versioning.list_versions()
    choices = [v["name"] for v in versions]
    chosen = st.sidebar.selectbox("Load saved model version (session only)", ["--none--"] + choices)
    if chosen != "--none--":
        entry = next((v for v in versions if v["name"] == chosen), None)
        if entry:
            if st.sidebar.button("Load version into session"):
                src = Path(entry["path"])
                if src.exists():
                    tmp_dest = Path(tempfile.gettempdir()) / f"ae_model_version_{int(time.time())}.pth"
                    shutil.copy2(src, tmp_dest)
                    m = try_load_model_from_path(str(tmp_dest))
                    if m:
                        model_obj = m
                        model_loaded = True
                        model_source = f"version:{chosen}"
                        st.sidebar.success(f"Loaded version {chosen} into session.")
                else:
                    st.sidebar.error("Model file missing on disk (version registry stale).")

# --- Main UI ---
st.title("BFZDD — Behaviour-First Zero-Day Detector")
st.markdown("**Warning:** Running uploaded scripts executes code. Only run uploaded scripts inside an isolated VM. See `VM_SAFETY.md`.")

# Tabs: Dataset Review / Live Analysis / Tools
tab = st.selectbox("Mode", ["Dataset Review", "Live Analysis", "Tools / Diagnostics"])

# Utility: load traces from dataset/traces
DATA_TRACES_DIR = Path("dataset/traces")
trace_files = sorted([str(p) for p in (DATA_TRACES_DIR.glob("*.json"))]) if DATA_TRACES_DIR.exists() else []

# ---------- Dataset Review ----------
if tab == "Dataset Review":
    st.header("Dataset Review")
    df_rows = []
    for p in trace_files:
        label = "benign" if "benign" in os.path.basename(p).lower() else "malicious"
        df_rows.append({"file": os.path.basename(p), "path": str(p), "score": None, "label": label})
    res_df = pd.DataFrame(df_rows)
    st.table(res_df[["file","path","label","score"]])

    if model_loaded:
        st.info("Scoring traces with loaded model (this may take a few seconds)...")
        scores = []
        for row in df_rows:
            seq = load_trace(row["path"])
            score, losses = score_trace_with_model(model_obj, seq)  # returns scalar + per-token loss list
            scores.append(score)
        res_df["score"] = scores
        res_df["pred"] = res_df["score"].apply(lambda s: "malicious" if (s is not None and s > thresholds.get("suggested_threshold", 1.0)) else "benign")
        st.table(res_df[["file","score","label"]])

        # Confusion matrix + PRF
        thr_input = st.number_input("Decision threshold (score > thr => malicious)", value=float(thresholds.get("suggested_threshold", 1.0)))
        res_df["pred_thr"] = res_df["score"].apply(lambda s: "malicious" if (s is not None and s > thr_input) else "benign")
        cm_df, prf_df = utils_viz.build_confusion_matrix_df(res_df["label"].tolist(), res_df["pred_thr"].tolist())
        st.subheader("Confusion Matrix")
        st.table(cm_df)
        st.subheader("Precision/Recall/F1")
        st.table(prf_df)

        # ROC
        st.subheader("ROC Curve")
        mask = res_df["score"].notnull()
        if mask.sum() >= 2:
            fpr, tpr, area = utils_viz.plot_roc_from_labels_scores(res_df["label"][mask].tolist(), res_df["score"][mask].tolist())
            fig = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {area:.3f})", labels={"x":"False Positive Rate","y":"True Positive Rate"})
            fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough scored samples for ROC curve.")

        # Reconstruction distribution
        st.subheader("Reconstruction error distribution")
        fig = px.histogram(res_df, x="score", color="label", nbins=20, title="Reconstruction error distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model loaded — upload or place ae_model.pth in repo to enable scoring.")

# ---------- Live Analysis ----------
if tab == "Live Analysis":
    st.header("Live Analysis")
    st.markdown("Select an existing script from `dataset/scripts` or upload a script (ONLY run in a VM).")
    # sample script dropdown
    script_paths = sorted([str(p) for p in Path("dataset/scripts").glob("*.py")]) if Path("dataset/scripts").exists() else []
    chosen_script = st.selectbox("Choose a sample script (from dataset/scripts)", ["--none--"] + [os.path.basename(x) for x in script_paths])
    uploaded_script = st.file_uploader("Or upload a Python script (optional)", type=["py"])
    run_in_vm_confirm = st.checkbox("I confirm I will run this only in an isolated VM", value=False)

    # Analyze precomputed trace option
    st.markdown("---")
    trace_choice = st.selectbox("Or select precomputed trace (dataset/traces)", ["--none--"] + [os.path.basename(x) for x in trace_files])
    if trace_choice != "--none--":
        trace_path = str(DATA_TRACES_DIR / trace_choice)
        with open(trace_path,"r") as f:
            events = json.load(f)
        st.success("Trace loaded for analysis.")
        # Show timeline
        fig, ev_df = utils_viz.plot_trace_timeline(events)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Top anomalous events (if model loaded and per-token losses available)")
        if model_loaded:
            seq = load_trace(trace_path)
            score, losses = score_trace_with_model(model_obj, seq)
            st.metric("Anomaly Score", f"{score:.4f}")
            # show top events if losses list and events align
            try:
                df_events = utils_viz.top_anomalous_events(events, losses)
                st.table(df_events)
            except Exception:
                st.info("No per-token losses available to compute top anomalous events.")
        else:
            st.info("Model not loaded — cannot score trace.")

    # Run & analyze script (exec disabled by default for safety)
    if st.button("Run & Analyze (Sandbox)"):
        if DISABLE_EXECUTION:
            st.error("Execution disabled by environment variable (DISABLE_EXECUTION). Upload model and run locally/VM.")
        elif not run_in_vm_confirm:
            st.error("Please confirm you will run this only in an isolated VM.")
        else:
            # choose script source
            if uploaded_script is None and chosen_script == "--none--":
                st.error("No script chosen or uploaded.")
            else:
                # safe execution path omitted — instruct user
                st.info("Sandbox execution should be performed inside your VM. Use sandbox_runner.py inside a VM and then upload trace.json here.")
                st.stop()

# ---------- Tools / Diagnostics ----------
if tab == "Tools / Diagnostics":
    st.header("Tools & Diagnostics")
    st.subheader("Model versioning")
    if st.button("Create version from repo model (save copy to models/)"):
        if Path(REPO_MODEL_NAME).exists():
            entry = versioning.save_model_version(REPO_MODEL_NAME, note="manual snapshot from app")
            st.success(f"Saved model as {entry['name']}")
        else:
            st.error("Repo model not found to create a version.")

    st.subheader("Event frequency heatmap (dataset traces)")
    if st.button("Compute event frequency heatmap"):
        if not trace_files:
            st.info("No traces found in dataset/traces")
        else:
            heat_df = utils_viz.build_event_freq_matrix(trace_files)
            # limit columns to top 25 event types
            sums = heat_df.sum(axis=0).sort_values(ascending=False)
            top_cols = list(sums.index[:25])
            df_small = heat_df[top_cols]
            fig = px.imshow(df_small, aspect="auto", labels=dict(x="Event Type", y="Trace", color="Count"), title="Event frequency heatmap (top event types)")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Developer Diagnostics")
    if st.checkbox("Show model file list"):
        st.write("Repo model exists:", Path(REPO_MODEL_NAME).exists())
        st.write("Model versions:", versioning.list_versions())

st.markdown("---")
st.caption("BFZDD — Behaviour-First Zero-Day Detector. Keep `ae_model.pth` in repository or upload one in sidebar for scoring.")
