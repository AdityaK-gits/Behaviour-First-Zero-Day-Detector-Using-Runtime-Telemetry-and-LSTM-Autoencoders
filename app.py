# app.py — BFZDD Streamlit dashboard (integrated Fine-Tune, save uploads + immediate replay inclusion, export retrain package)
# Overwrite your existing app.py with this file.
# NOTE: Persistence is local to the server. To persist across redeploys, commit files to GitHub manually.

import os
import sys
import json
import time
import tempfile
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# ML imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# start log
print("[APP] start - cwd:", os.getcwd(), "python:", sys.version)

# Robust imports from model.py / utils_viz / versioning
try:
    from model import AuditAutoencoder, load_trace, score_trace_with_model, estimated_vocab_size
    print("[APP] Imported model helpers from model.py")
except Exception as e:
    AuditAutoencoder = None
    load_trace = None
    score_trace_with_model = None
    estimated_vocab_size = lambda: 200
    print("[APP] Warning: failed to import model helpers:", e)

try:
    import utils_viz
except Exception as e:
    utils_viz = None
    print("[APP] Warning: utils_viz import failed:", e)

try:
    import versioning
except Exception:
    versioning = None

# Config and folders
REPO_MODEL_NAME = "ae_model.pth"
DATA_TRACES_DIR = Path("dataset/traces")
DATA_TRACES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
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

# ------------------------
# Model loading helpers
# ------------------------
model = None
model_source = None

def try_load_model_from_file(path: str):
    global AuditAutoencoder
    if AuditAutoencoder is None:
        return None, "model_class_missing"
    try:
        vb = estimated_vocab_size() if callable(estimated_vocab_size) else 200
        m = AuditAutoencoder(vocab_size=vb)
        state = torch.load(path, map_location="cpu")
        # try strict load, else partial align
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

tmp_uploaded_model_path = None
if uploaded_model is not None:
    try:
        tmp_uploaded_model_path = Path(tempfile.gettempdir()) / f"uploaded_ae_{int(time.time())}.pth"
        with open(tmp_uploaded_model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        m, status = try_load_model_from_file(str(tmp_uploaded_model_path))
        if m:
            model = m
            model_source = f"uploaded:{uploaded_model.name}"
            st.sidebar.success("Uploaded model loaded (session)")
        else:
            st.sidebar.error(f"Uploaded model failed to load: {status}")
    except Exception as e:
        st.sidebar.error("Failed saving uploaded model: " + str(e))

# fallback repo model
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

# versioning UI
st.sidebar.markdown("---")
st.sidebar.subheader("Model Versions")
if versioning is not None:
    try:
        vs = versioning.list_versions()
    except Exception as e:
        vs = []
        print("[APP] versioning.list_versions failed:", e)
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

# ----------------------------------------------------------------
# Session-state: keep track of saved uploads included in replay buffer
# ----------------------------------------------------------------
if "saved_upload_paths" not in st.session_state:
    st.session_state.saved_upload_paths = []  # list of Path strings
if "new_sequences_cache" not in st.session_state:
    st.session_state.new_sequences_cache = []  # sequences extracted from current session uploads

# Helper: robust load_trace wrapper -> tokens list
def safe_load_seq(trace_path, max_len=None):
    if load_trace is None:
        return []
    try:
        # many model.load_trace implementations accept only path
        seq = load_trace(str(trace_path))
        if hasattr(seq, "tolist"):
            seq = seq.tolist()
        seq = list(seq)
        seq = [int(x) for x in seq]
    except Exception:
        try:
            seq = load_trace(str(trace_path))
            if hasattr(seq, "tolist"):
                seq = seq.tolist()
            seq = list(seq)
            seq = [int(x) for x in seq]
        except Exception:
            seq = []
    if max_len is not None:
        if len(seq) >= max_len:
            return seq[:max_len]
        else:
            return seq + [0] * (max_len - len(seq))
    return seq

# -------------------------
# Main UI mode selector
# -------------------------
mode = st.selectbox("Mode", ["Dataset Review", "Live Analysis", "Tools & Diagnostics", "Fine-Tune Model"])

# -------------------------
# Dataset Review
# -------------------------
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
            df["score"] = scores
            st.table(df[["file", "label", "score"]])
            if utils_viz is not None:
                thr = st.number_input("Decision threshold (score > thr => malicious)", value=float(thresholds.get("suggested_threshold", 1.0) if thresholds else 1.0))
                df_valid = df[df["score"].notnull()].copy()
                if not df_valid.empty:
                    df_valid["pred"] = df_valid["score"].apply(lambda s: "malicious" if s > thr else "benign")
                    cm_df, prf_df = utils_viz.build_confusion_matrix_df(df_valid["label"].tolist(), df_valid["pred"].tolist())
                    st.subheader("Confusion matrix")
                    st.table(cm_df)
                    st.subheader("Precision / Recall / F1")
                    st.table(prf_df)
                    try:
                        fpr, tpr, area = utils_viz.plot_roc_from_labels_scores(df_valid["label"].tolist(), df_valid["score"].tolist())
                        fig = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {area:.3f})", labels={"x":"FPR","y":"TPR"})
                        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        print("[APP] ROC plot failed:", e)
                else:
                    st.info("Not enough scored traces for confusion/ROC.")

# -------------------------
# Live Analysis
# -------------------------
elif mode == "Live Analysis":
    st.header("Live Analysis")
    st.markdown("Pick a precomputed trace or upload a trace.json.")
    trace_list = sorted([p for p in DATA_TRACES_DIR.glob("*.json")]) if DATA_TRACES_DIR.exists() else []
    chosen_trace = st.selectbox("Precomputed trace", ["--none--"] + [os.path.basename(str(p)) for p in trace_list])
    uploaded_trace = st.file_uploader("Or upload a trace.json", type=["json"])
    if uploaded_trace is not None:
        ttmp = Path(tempfile.gettempdir()) / f"uploaded_trace_{int(time.time())}.json"
        with open(ttmp, "wb") as f:
            f.write(uploaded_trace.getbuffer())
        trace_path = str(ttmp)
    elif chosen_trace != "--none--":
        trace_path = str(DATA_TRACES_DIR / chosen_trace)
    else:
        trace_path = None

    if trace_path:
        try:
            with open(trace_path, "r") as f:
                events = json.load(f)
        except Exception:
            events = []
        st.success(f"Loaded trace: {os.path.basename(trace_path)}")
        if utils_viz is not None:
            try:
                fig, evdf = utils_viz.plot_trace_timeline(events)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                st.subheader("Event sample")
                if evdf is not None:
                    st.table(evdf.head(200))
            except Exception as e:
                st.warning("Trace visualization failed: " + str(e))
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
                    if utils_viz is not None:
                        taf = utils_viz.top_anomalous_events(events, losses)
                        if taf is not None and not taf.empty:
                            st.subheader("Top anomalous events")
                            st.table(taf)
                except Exception as e:
                    st.error("Scoring failed: " + str(e))
        else:
            st.info("Model not loaded; scoring unavailable.")

# -------------------------
# Tools & Diagnostics
# -------------------------
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

# -------------------------
# Fine-Tune Model (with immediate save+replay inclusion)
# -------------------------
elif mode == "Fine-Tune Model":
    st.header("🔧 Fine-Tune the Model (Incremental Learning)")
    st.markdown("Upload benign traces, save them into dataset/traces/ (one-click) and fine-tune here. Only upload benign traces.")

    if DISABLE_EXECUTION:
        st.warning("Model fine-tuning is disabled in this deployment (DISABLE_EXECUTION=true).")
    else:
        uploaded_files = st.file_uploader("Upload NEW benign trace files (JSON). You can upload multiple files.", accept_multiple_files=True, type=["json"])
        st.markdown("**Options:**")
        epochs = st.number_input("Training Epochs", min_value=1, max_value=10, value=3)
        lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.6f")
        batch_size = st.selectbox("Batch Size", options=[1,2,4,8,16,32], index=3)
        replay_fraction = st.slider("Replay Buffer % (include this % of old benign traces)", 0, 100, 30)
        max_files_limit = st.number_input("Max uploaded files (safety cap)", min_value=1, max_value=500, value=200)

        # Buttons: Save+Include, Export, Start training
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            save_and_include = st.button("💾 Save uploaded traces to dataset/traces & INCLUDE in replay (one-click)")
        with col2:
            export_manifest = st.button("📦 Export retrain package (ZIP + manifest)")
        with col3:
            start_training = st.button("🚀 Start Fine-Tuning (train in UI)")

        # ---- Save & Immediately include in replay buffer (in-memory)
        def safe_write_local_and_register(file_obj, dest_dir=DATA_TRACES_DIR):
            name = getattr(file_obj, "name", f"uploaded_{int(time.time()*1000)}.json")
            name = "".join(c for c in name if c.isalnum() or c in ("-", "_", "."))
            dest = dest_dir / name
            with open(dest, "wb") as fh:
                fh.write(file_obj.getbuffer())
            # record for immediate replay inclusion
            if str(dest) not in st.session_state.saved_upload_paths:
                st.session_state.saved_upload_paths.append(str(dest))
            return dest

        if save_and_include:
            if not uploaded_files:
                st.error("No uploaded files to save. Use the uploader above.")
            else:
                saved = []
                for f in uploaded_files:
                    try:
                        p = safe_write_local_and_register(f)
                        saved.append(str(p))
                    except Exception as e:
                        st.warning(f"Failed to save {getattr(f,'name',str(f))}: {e}")
                if saved:
                    st.success(f"Saved {len(saved)} files to dataset/traces/ and included them in replay buffer for this session.")
                    st.write(saved)

        # ---- Export retrain package
        if export_manifest:
            if not uploaded_files or len(uploaded_files) == 0:
                st.error("No uploaded files to export. Upload traces first.")
            else:
                include_replay = st.checkbox("Include existing benign traces from dataset/traces/ in the package", value=True)
                replay_paths = []
                if include_replay:
                    benigns = sorted([p for p in DATA_TRACES_DIR.glob("benign_*.json")])
                    max_include = len(benigns)
                    if max_include > 0:
                        replay_count = st.number_input("How many existing benign traces to include (0 = none)", min_value=0, max_value=max_include, value=min(20, max_include))
                        replay_paths = benigns[:replay_count]
                    else:
                        st.info("No existing benign traces found for inclusion.")
                mem_zip = BytesIO()
                with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    uploaded_names = []
                    for f in uploaded_files:
                        fname = "".join(c for c in f.name if c.isalnum() or c in ("-", "_", "."))
                        zf.writestr(f"uploaded/{fname}", f.getvalue())
                        uploaded_names.append(fname)
                    included = []
                    for p in replay_paths:
                        try:
                            zf.write(str(p), arcname=f"replay/{p.name}")
                            included.append(p.name)
                        except Exception as e:
                            st.warning(f"Could not include {p}: {e}")
                    manifest = {
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "uploaded_files": uploaded_names,
                        "replay_included": included,
                        "recommended_hyperparams": {"epochs": int(3), "learning_rate": float(1e-4), "batch_size": int(8), "replay_fraction": int(30)},
                        "notes": "Exported by BFZDD UI. Use incremental_train.py to consume on GPU/VM."
                    }
                    zf.writestr("manifest.json", json.dumps(manifest, indent=2))
                mem_zip.seek(0)
                bname = f"retrain_package_{int(time.time())}.zip"
                st.download_button("Download retrain package (zip)", data=mem_zip.getvalue(), file_name=bname, mime="application/zip")
                st.success("Retrain package ready for download.")

        # ---- Start fine-tuning within UI (uses saved_upload_paths for immediate replay inclusion)
        if start_training:
            if AuditAutoencoder is None or load_trace is None or score_trace_with_model is None:
                st.error("Model utilities missing (model.py). Cannot fine-tune here.")
            elif not Path(REPO_MODEL_NAME).exists() and model is None and tmp_uploaded_model_path is None:
                st.error("No base model found to fine-tune. Upload ae_model.pth or place one in repo root.")
            else:
                if not uploaded_files or len(uploaded_files) == 0:
                    st.error("Please upload at least one JSON benign trace to fine-tune.")
                elif len(uploaded_files) > max_files_limit:
                    st.error(f"Uploaded {len(uploaded_files)} files exceeds safety cap ({max_files_limit}). Reduce uploads.")
                else:
                    st.info("Preparing fine-tune dataset (this session includes saved uploads)...")
                    base_model_path = tmp_uploaded_model_path if tmp_uploaded_model_path and Path(tmp_uploaded_model_path).exists() else (REPO_MODEL_NAME if Path(REPO_MODEL_NAME).exists() else None)
                    try:
                        vb = estimated_vocab_size() if callable(estimated_vocab_size) else 200
                        finetune_model = AuditAutoencoder(vocab_size=vb)
                        finetune_model.load_state_dict(torch.load(base_model_path, map_location="cpu"), strict=False)
                        finetune_model.train()
                    except Exception as e:
                        st.error("Failed to load base model for fine-tuning: " + str(e))
                        st.stop()

                    # build sequence list from uploaded files (temp) and saved_upload_paths
                    new_sequences = []
                    for f in uploaded_files:
                        try:
                            tpath = Path(tempfile.gettempdir()) / f"upload_trace_{int(time.time()*1000)}_{f.name}"
                            with open(tpath, "wb") as fh:
                                fh.write(f.getbuffer())
                            seq = safe_load_seq(str(tpath), max_len=None)
                            if seq and isinstance(seq, list) and len(seq) > 0:
                                new_sequences.append(seq)
                        except Exception as e:
                            st.warning(f"Failed to parse {getattr(f,'name',str(f))}: {e}")

                    # also include any saved uploads (immediate include)
                    for sp in st.session_state.saved_upload_paths:
                        try:
                            seq = safe_load_seq(sp, max_len=None)
                            if seq and isinstance(seq, list) and len(seq) > 0:
                                # avoid duplicates
                                if seq not in new_sequences:
                                    new_sequences.append(seq)
                        except:
                            continue

                    if len(new_sequences) == 0:
                        st.error("No valid sequences extracted from uploads/saved uploads.")
                        st.stop()

                    # replay buffer: existing benign files in dataset/traces/ (excluding the ones we just saved if already included)
                    replay_sequences = []
                    benign_files = sorted([p for p in DATA_TRACES_DIR.glob("benign_*.json")]) if DATA_TRACES_DIR.exists() else []
                    for p in benign_files:
                        try:
                            seq = safe_load_seq(str(p), max_len=None)
                            if seq and isinstance(seq, list) and len(seq) > 0:
                                replay_sequences.append(seq)
                        except:
                            continue
                    k = int((replay_fraction / 100.0) * len(replay_sequences))
                    replay_sequences = replay_sequences[:k] if k > 0 else []

                    st.write(f"Using {len(new_sequences)} new sequences + {len(replay_sequences)} replay sequences (replay%={replay_fraction}%).")

                    combined = new_sequences + replay_sequences
                    def pad_batch_np(seqs, pad_value=0):
                        maxlen = max(len(s) for s in seqs)
                        arr = np.full((len(seqs), maxlen), pad_value, dtype=np.int64)
                        for i,s in enumerate(seqs):
                            arr[i,:len(s)] = s
                        return torch.LongTensor(arr)
                    dataset_tensor = pad_batch_np(combined)
                    ds = torch.utils.data.TensorDataset(dataset_tensor)
                    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

                    # training
                    opt = torch.optim.Adam(finetune_model.parameters(), lr=lr)
                    criterion = nn.CrossEntropyLoss(ignore_index=0)
                    progress_bar = st.progress(0)
                    loss_placeholder = st.empty()
                    loss_history = []
                    total_steps = int(epochs) * max(1, len(dl))
                    step = 0
                    for epoch in range(int(epochs)):
                        finetune_model.train()
                        epoch_loss = 0.0
                        batch_count = 0
                        for batch in dl:
                            batch_x = batch[0].long()
                            opt.zero_grad()
                            logits = finetune_model(batch_x)
                            B,T,V = logits.size()
                            loss = criterion(logits.view(-1, V), batch_x.view(-1))
                            loss.backward()
                            opt.step()
                            val = loss.item()
                            epoch_loss += val
                            batch_count += 1
                            step += 1
                            progress_bar.progress(int(step / total_steps * 100))
                        avg_epoch_loss = epoch_loss / max(1, batch_count)
                        loss_history.append(avg_epoch_loss)
                        fig, ax = plt.subplots()
                        ax.plot(loss_history, marker="o")
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("Loss")
                        ax.set_title("Fine-tune loss")
                        loss_placeholder.pyplot(fig)

                    st.success("🎉 Fine-tuning completed successfully!")

                    # save version
                    timestamp = time.strftime("%Y%m%dT%H%M%S")
                    version_name = f"ae_model_{timestamp}.pth"
                    version_path = MODELS_DIR / version_name
                    try:
                        torch.save(finetune_model.state_dict(), str(version_path))
                        st.success(f"Saved new model version: {version_name}")
                    except Exception as e:
                        st.warning("Failed saving versioned model: " + str(e))

                    # update repo model file if possible
                    try:
                        torch.save(finetune_model.state_dict(), REPO_MODEL_NAME)
                        st.info("Updated ae_model.pth with fine-tuned weights.")
                    except Exception as e:
                        st.warning("Could not overwrite ae_model.pth: " + str(e))

                    # register version via versioning.py
                    if versioning is not None:
                        try:
                            entry = versioning.save_model_version(str(version_path), note="fine-tuned from UI")
                            st.write("Version registered:", entry.get("name", "unknown"))
                        except Exception as e:
                            st.warning("Version registration failed: " + str(e))

                    # threshold recalibration
                    st.info("Recalibrating thresholds using benign traces...")
                    finetune_model.eval()
                    benign_scores = []
                    for p in DATA_TRACES_DIR.glob("benign_*.json") if DATA_TRACES_DIR.exists() else []:
                        try:
                            seq = safe_load_seq(str(p), max_len=None)
                            if not seq:
                                continue
                            x = torch.LongTensor([seq])
                            with torch.no_grad():
                                out = finetune_model(x)
                                B,T,V = out.size()
                                losses = nn.CrossEntropyLoss(reduction="none", ignore_index=0)(out.view(-1,V), x.view(-1))
                                mean_loss = float(losses.mean().item())
                                benign_scores.append(mean_loss)
                        except Exception:
                            continue
                    # include newly uploaded sequences in calibration
                    for seq in new_sequences:
                        try:
                            x = torch.LongTensor([seq])
                            with torch.no_grad():
                                out = finetune_model(x)
                                B,T,V = out.size()
                                losses = nn.CrossEntropyLoss(reduction="none", ignore_index=0)(out.view(-1,V), x.view(-1))
                                mean_loss = float(losses.mean().item())
                                benign_scores.append(mean_loss)
                        except Exception:
                            continue

                    if len(benign_scores) > 0:
                        p95 = float(np.percentile(benign_scores, 95))
                        p99 = float(np.percentile(benign_scores, 99))
                        mean_score = float(np.mean(benign_scores))
                        std_score = float(np.std(benign_scores))
                        suggested = p99
                        data = {"mean": mean_score, "std": std_score, "p95": p95, "p99": p99, "suggested_threshold": suggested}
                        try:
                            with open(THRESHOLD_PATH, "w") as fh:
                                json.dump(data, fh, indent=2)
                            st.success(f"Thresholds recalibrated and saved. suggested_threshold = {suggested:.4f}")
                        except Exception as e:
                            st.warning("Could not save threshold.json: " + str(e))
                    else:
                        st.warning("No benign scores found to recalibrate thresholds.")

                    st.info("Fine-tune workflow finished. Review Dataset Review for updated metrics.")

# Footer
st.markdown("---")
st.caption("BFZDD — Behaviour-First Zero-Day Detector — Fine-Tune + Save/Export (local persistence only)")
