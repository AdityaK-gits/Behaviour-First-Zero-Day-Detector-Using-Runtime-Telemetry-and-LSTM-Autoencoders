"""
app.py — BFZDD Streamlit dashboard (corrected & robust)

This version:
- imports model helpers safely from model.py
- if score_trace_with_model is missing, uses fallback (but model.py now includes it)
- loads model from uploaded file (session) or repo root ae_model.pth
- integrates with utils_viz and versioning (they must exist)
- safe default: DISABLE_EXECUTION env var prevents executing uploaded scripts
"""

import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# Diagnostics to logs (helpful on Streamlit Cloud)
print("[APP] start - cwd:", os.getcwd(), "python:", sys.version)

# Try to import model module helpers (robust)
try:
    from model import AuditAutoencoder, load_trace, score_trace_with_model, estimated_vocab_size
    print("[APP] Imported model helpers from model.py")
except Exception as e:
    # If import fails, fail gently — app will show helpful message in UI
    AuditAutoencoder = None
    load_trace = None
    score_trace_with_model = None
    estimated_vocab_size = lambda: 200
    print("[APP] Warning: failed to import model helpers:", e)

# Import visualization helpers (must be present as provided earlier)
try:
    import utils_viz
except Exception as e:
    utils_viz = None
    print("[APP] Warning: utils_viz import failed:", e)

# Import versioning helper (optional)
try:
    import versioning
except Exception:
    versioning = None

# Config
REPO_MODEL_NAME = "ae_model.pth"
DATA_TRACES_DIR = Path("dataset/traces")
THRESHOLD_PATH = Path("threshold.json")
DISABLE_EXECUTION = os.environ.get("DISABLE_EXECUTION", "false").lower() in ("1", "true", "yes")

st.set_page_config(page_title="BFZDD — Behaviour-First Zero-Day Detector", layout="wide")
st.title("BFZDD — Behaviour-First Zero-Day Detector")
st.markdown("**Note:** Do not run untrusted scripts in this public app. Use a VM and follow VM_SAFETY.md.")

# Sidebar: Model upload + status
st.sidebar.header("Model & Environment")
uploaded_model = st.sidebar.file_uploader("Upload ae_model.pth (optional)", type=["pth", "pt"])

# load thresholds if present
thresholds = {}
if THRESHOLD_PATH.exists():
    try:
        thresholds = json.loads(THRESHOLD_PATH.read_text())
    except Exception:
        thresholds = {}

if thresholds:
    st.sidebar.success("Thresholds loaded")
else:
    st.sidebar.info("No threshold.json found")

# Try to load model (uploaded -> repo)
model = None
model_source = None

def try_load_model_from_file(path: str):
    """Load a PyTorch model file into an AuditAutoencoder instance."""
    global AuditAutoencoder
    if AuditAutoencoder is None:
        return None, "model_class_missing"
    try:
        # use estimated_vocab_size if available
        vb = estimated_vocab_size() if callable(estimated_vocab_size) else 200
        m = AuditAutoencoder(vocab_size=vb)
        import torch
        state = torch.load(path, map_location="cpu")
        # attempt strict load; if fails, do partial safe load
        try:
            m.load_state_dict(state)
        except Exception:
            ms = m.state_dict()
            for k, v in state.items():
                if k in ms and ms[k].shape == v.shape:
                    ms[k] = v
            m.load_state_dict(ms)
        m.eval()
        return m, "ok"
    except Exception as e:
        print("[APP] model load failed:", e)
        return None, str(e)

# Uploaded model priority
if uploaded_model is not None:
    tmp_path = Path(tempfile.gettempdir()) / f"uploaded_ae_{int(time.time())}.pth"
    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        m, status = try_load_model_from_file(str(tmp_path))
        if m:
            model = m
            model_source = f"uploaded:{uploaded_model.name}"
            st.sidebar.success("Uploaded model loaded (session)")
        else:
            st.sidebar.error(f"Uploaded model failed to load: {status}")
    except Exception as e:
        st.sidebar.error("Failed saving uploaded model: " + str(e))

# Repo model fallback
if model is None and Path(REPO_MODEL_NAME).exists():
    m, status = try_load_model_from_file(REPO_MODEL_NAME)
    if m:
        model = m
        model_source = f"repo:{REPO_MODEL_NAME}"
        st.sidebar.success(f"Model loaded from repo ({REPO_MODEL_NAME})")
    else:
        st.sidebar.error(f"Failed to load repo model: {status}")

if model is None:
    st.sidebar.warning("Model not loaded. Upload a .pth or place ae_model.pth in repo root.")
else:
    st.sidebar.info(f"Model source: {model_source}")

# Sidebar: model versions (optional)
st.sidebar.markdown("---")
st.sidebar.subheader("Model versions")
if versioning is not None:
    vs = versioning.list_versions()
    if vs:
        names = [v["name"] for v in vs]
        chosen = st.sidebar.selectbox("Choose version to load (session)", ["--none--"] + names)
        if chosen != "--none--" and st.sidebar.button("Load version in session"):
            entry = next((v for v in vs if v["name"] == chosen), None)
            if entry and Path(entry["path"]).exists():
                tmpv = Path(tempfile.gettempdir()) / f"ae_ver_{int(time.time())}.pth"
                shutil.copy2(entry["path"], tmpv)
                m, status = try_load_model_from_file(str(tmpv))
                if m:
                    model = m
                    model_source = f"version:{chosen}"
                    st.sidebar.success(f"Loaded version {chosen} into session")
                else:
                    st.sidebar.error("Failed to load version: " + str(status))
    else:
        st.sidebar.write("No saved versions")
else:
    st.sidebar.write("Versioning not available")

# Main UI modes
mode = st.selectbox("Mode", ["Dataset Review", "Live Analysis", "Tools & Diagnostics"])

# Helper: load sequence safely (handles missing load_trace signature)
def safe_load_seq(trace_path, max_len=None):
    if load_trace is None:
        return []
    try:
        seq = load_trace(str(trace_path))
        # normalize to Python list of ints
        if hasattr(seq, "tolist"):
            seq = seq.tolist()
        seq = list(seq)
        seq = [int(x) for x in seq]
    except TypeError:
        # try again without max_len
        try:
            seq = load_trace(str(trace_path))
            if hasattr(seq, "tolist"):
                seq = seq.tolist()
            seq = list(seq)
            seq = [int(x) for x in seq]
        except Exception:
            seq = []
    except Exception:
        seq = []
    if max_len is not None:
        if len(seq) >= max_len:
            return seq[:max_len]
        else:
            return seq + [0] * (max_len - len(seq))
    return seq

# ---------- Dataset Review ----------
if mode == "Dataset Review":
    st.header("Dataset Review")
    trace_paths = sorted([str(p) for p in DATA_TRACES_DIR.glob("*.json")]) if DATA_TRACES_DIR.exists() else []
    if not trace_paths:
        st.info("No traces found in dataset/traces")
    else:
        rows = []
        for p in trace_paths:
            fname = os.path.basename(p)
            label = "benign" if "benign" in fname.lower() else ("malicious" if "malware" in fname.lower() else "unknown")
            rows.append({"file": fname, "path": p, "label": label})
        df = pd.DataFrame(rows)
        st.table(df)

        if model is None:
            st.info("Model not loaded; scoring disabled.")
        else:
            st.info("Scoring traces...")
            scores = []
            per_trace_losses = {}
            for r in rows:
                seq = safe_load_seq(r["path"], max_len=200)
                if not seq:
                    scores.append(None)
                    continue
                try:
                    score, losses = score_trace_with_model(model, seq)
                except Exception as e:
                    print("[APP] scoring error:", e)
                    score, losses = None, []
                scores.append(score)
                per_trace_losses[r["file"]] = losses
            df["score"] = scores
            # show table with score + label
            st.table(df[["file", "label", "score"]])

            # Confusion matrix + PRF
            if utils_viz is not None:
                thr = st.number_input("Decision threshold (score > thr => malicious)", value=float(thresholds.get("suggested_threshold", 1.0) if thresholds else 1.0))
                df_valid = df[df["score"].notnull()]
                if not df_valid.empty:
                    df_valid["pred"] = df_valid["score"].apply(lambda s: "malicious" if s > thr else "benign")
                    cm_df, prf_df = utils_viz.build_confusion_matrix_df(df_valid["label"].tolist(), df_valid["pred"].tolist())
                    st.subheader("Confusion matrix")
                    st.table(cm_df)
                    st.subheader("Precision / Recall / F1")
                    st.table(prf_df)

                    # ROC
                    try:
                        fpr, tpr, area = utils_viz.plot_roc_from_labels_scores(df_valid["label"].tolist(), df_valid["score"].tolist())
                        fig = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {area:.3f})", labels={"x":"FPR","y":"TPR"})
                        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        print("[APP] ROC plot failed:", e)
                else:
                    st.info("Not enough scored traces for confusion/ROC.")

# ---------- Live Analysis ----------
elif mode == "Live Analysis":
    st.header("Live Analysis")
    st.markdown("Pick a precomputed trace or upload a script (run scripts only in your VM).")
    # list scripts / traces
    scripts_dir = Path("dataset/scripts")
    script_list = sorted([p for p in scripts_dir.glob("*.py")]) if scripts_dir.exists() else []
    trace_list = sorted([p for p in DATA_TRACES_DIR.glob("*.json")]) if DATA_TRACES_DIR.exists() else []

    chosen_trace = st.selectbox("Precomputed trace", ["--none--"] + [os.path.basename(str(p)) for p in trace_list])
    uploaded_trace = st.file_uploader("Or upload a trace.json", type=["json"])
    if uploaded_trace is not None:
        # save temporary and treat as trace
        ttmp = Path(tempfile.gettempdir()) / f"uploaded_trace_{int(time.time())}.json"
        with open(ttmp, "wb") as f:
            f.write(uploaded_trace.getbuffer())
        trace_path = str(ttmp)
    elif chosen_trace != "--none--":
        trace_path = str(DATA_TRACES_DIR / chosen_trace)
    else:
        trace_path = None

    if trace_path:
        # load events and show timeline
        try:
            with open(trace_path, "r") as f:
                events = json.load(f)
        except Exception:
            events = []
        st.success(f"Loaded trace: {os.path.basename(trace_path)}")
        if utils_viz is not None:
            fig, evdf = utils_viz.plot_trace_timeline(events)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            # show raw event table head
            st.subheader("Event sample")
            if evdf is not None:
                st.table(evdf.head(200))
        # score if model loaded
        if model is not None:
            seq = safe_load_seq(trace_path, max_len=200)
            if seq:
                try:
                    score, losses = score_trace_with_model(model, seq)
                    st.metric("Anomaly Score", f"{score:.4f}")
                    thr_val = thresholds.get("suggested_threshold")
                    if thr_val is not None:
                        verdict = "QUARANTINE" if score > float(thr_val) else "OK"
                        st.write("Verdict:", verdict, f"(threshold={thr_val})")
                    # top anomalous events
                    if utils_viz is not None:
                        taf = utils_viz.top_anomalous_events(events, losses)
                        if not taf.empty:
                            st.subheader("Top anomalous events")
                            st.table(taf)
                except Exception as e:
                    st.error("Scoring failed: " + str(e))
        else:
            st.info("Model not loaded; scoring unavailable.")

# ---------- Tools & Diagnostics ----------
elif mode == "Tools & Diagnostics":
    st.header("Tools & Diagnostics")
    st.subheader("Model versioning")
    if st.button("Create snapshot (save repo model in models/)"):
        if Path(REPO_MODEL_NAME).exists():
            if versioning is not None:
                try:
                    entry = versioning.save_model_version(REPO_MODEL_NAME, note="snapshot from app")
                    st.success(f"Saved model version: {entry['name']}")
                except Exception as e:
                    st.error("Versioning save failed: " + str(e))
            else:
                st.info("versioning.py not present")
        else:
            st.info("Repo model not found (place ae_model.pth in repo root)")

    st.subheader("Event frequency heatmap")
    if st.button("Compute heatmap (dataset traces)"):
        if utils_viz is None:
            st.info("utils_viz.py missing")
        else:
            trace_paths = [str(p) for p in DATA_TRACES_DIR.glob("*.json")] if DATA_TRACES_DIR.exists() else []
            if not trace_paths:
                st.info("No traces found")
            else:
                heat_df = utils_viz.build_event_freq_matrix(trace_paths)
                sums = heat_df.sum(axis=0).sort_values(ascending=False)
                top_cols = list(sums.index[:25])
                df_small = heat_df[top_cols]
                fig = px.imshow(df_small, aspect="auto", labels=dict(x="Event Type", y="Trace", color="Count"), title="Event frequency heatmap")
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Developer diagnostics")
    if st.checkbox("Show repo root listing"):
        try:
            st.write(os.listdir("."))
        except Exception as e:
            st.write("listdir failed:", e)

st.caption("BFZDD - Behaviour-First Zero-Day Detector")
