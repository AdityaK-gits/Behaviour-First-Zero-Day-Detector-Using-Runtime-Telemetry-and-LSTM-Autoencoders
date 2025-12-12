"""
app.py — Final BFZDD Streamlit app (upgraded)
- Loads upgraded model.py (build_model + score_trace_with_model)
- Dataset Review, Live Analysis, Tools & Diagnostics, Fine-Tune (in-UI)
- New heatmaps: event frequency and anomalous-events heatmap (uses utils_viz)
- Backwards compatible with existing repo layout (dataset/traces, models/, ae_model.pth)
"""

import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import zipfile
from io import BytesIO

import torch
import torch.nn as nn
import numpy as np

# Logging
print("[APP] start - cwd:", os.getcwd(), "python:", sys.version)

# Import model helpers
try:
    from model import build_model, load_trace, score_trace_with_model, estimated_vocab_size, compute_kl_regularizer
    print("[APP] imported upgraded model module")
except Exception as e:
    build_model = None
    load_trace = None
    score_trace_with_model = None
    estimated_vocab_size = lambda: 200
    compute_kl_regularizer = None
    print("[APP] failed to import model module:", e)

# Import visualization helpers
try:
    import utils_viz
except Exception as e:
    utils_viz = None
    print("[APP] failed to import utils_viz:", e)

# Optional versioning helper
try:
    import versioning
except Exception:
    versioning = None

# Paths & config
REPO_MODEL_NAME = "ae_model.pth"
DATA_TRACES_DIR = Path("dataset/traces")
DATA_TRACES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD_PATH = Path("threshold.json")
DISABLE_EXECUTION = os.environ.get("DISABLE_EXECUTION", "false").lower() in ("1", "true", "yes")

st.set_page_config(page_title="BFZDD — Behaviour-First Zero-Day Detector", layout="wide")
st.title("BFZDD — Behaviour-First Zero-Day Detector")
st.markdown("**Safety note:** Do not run untrusted scripts on this public app. Use VM and follow VM_SAFETY.md.")

# Sidebar: model upload and options
st.sidebar.header("Model & Environment")
uploaded_model = st.sidebar.file_uploader("Upload ae_model.pth (optional)", type=["pth", "pt"])
default_arch = st.sidebar.selectbox("Default architecture for fine-tune", ["lstm", "gru", "transformer"])
st.sidebar.markdown("---")
if THRESHOLD_PATH.exists():
    try:
        thresholds = json.loads(THRESHOLD_PATH.read_text())
        st.sidebar.success("Thresholds loaded")
    except Exception:
        thresholds = {}
        st.sidebar.info("Threshold file unreadable")
else:
    thresholds = {}
    st.sidebar.info("No threshold.json found")

# Model loading helpers
model = None
model_meta = None
model_source = None
tmp_uploaded_model_path = None

def try_load_model(path: str, arch_hint: Optional[str] = None):
    if build_model is None:
        return None, "model_module_missing", None
    meta = None
    meta_path = str(path) + ".meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as fh:
                meta = json.load(fh)
        except Exception:
            meta = None
    arch = arch_hint or (meta.get("arch") if isinstance(meta, dict) and "arch" in meta else "lstm")
    vocab = estimated_vocab_size()
    try:
        m = build_model(arch, vocab_size=vocab)
        state = torch.load(path, map_location="cpu")
        try:
            m.load_state_dict(state)
        except Exception:
            ms = m.state_dict()
            for k, v in state.items():
                if k in ms and ms[k].shape == v.shape:
                    ms[k] = v
            m.load_state_dict(ms)
        m.eval()
        return m, "ok", meta
    except Exception as e:
        return None, str(e), meta

# Uploaded model priority
if uploaded_model is not None:
    try:
        tmp_uploaded_model_path = Path(tempfile.gettempdir()) / f"uploaded_ae_{int(time.time())}.pth"
        with open(tmp_uploaded_model_path, "wb") as fh:
            fh.write(uploaded_model.getbuffer())
        m, status, meta = try_load_model(str(tmp_uploaded_model_path), default_arch)
        if m:
            model = m
            model_meta = meta
            model_source = f"uploaded:{uploaded_model.name}"
            st.sidebar.success("Uploaded model loaded")
        else:
            st.sidebar.error(f"Uploaded model failed: {status}")
    except Exception as e:
        st.sidebar.error("Failed saving uploaded model: " + str(e))

# Repo fallback
if model is None and Path(REPO_MODEL_NAME).exists():
    m, status, meta = try_load_model(REPO_MODEL_NAME, default_arch)
    if m:
        model = m
        model_meta = meta
        model_source = f"repo:{REPO_MODEL_NAME}"
        st.sidebar.success("Model loaded from repo")
    else:
        st.sidebar.error(f"Failed to load repo model: {status}")

if model is None:
    st.sidebar.warning("No model loaded. Upload or place ae_model.pth in repo root.")
else:
    st.sidebar.info(f"Model source: {model_source}")

# Versioning UI (optional)
st.sidebar.markdown("---")
st.sidebar.subheader("Model versions")
if versioning is not None:
    try:
        vs = versioning.list_versions()
    except Exception:
        vs = []
    if vs:
        names = [v["name"] for v in vs]
        chosen = st.sidebar.selectbox("Load saved version", ["--none--"] + names)
        if chosen != "--none--" and st.sidebar.button("Load version in session"):
            entry = next((v for v in vs if v["name"] == chosen), None)
            if entry and Path(entry["path"]).exists():
                tmpv = Path(tempfile.gettempdir()) / f"ae_ver_{int(time.time())}.pth"
                shutil.copy2(entry["path"], tmpv)
                m, status, meta = try_load_model(str(tmpv), default_arch)
                if m:
                    model = m
                    model_meta = meta
                    model_source = f"version:{chosen}"
                    st.sidebar.success(f"Loaded version {chosen}")
                else:
                    st.sidebar.error("Failed to load version: " + str(status))
    else:
        st.sidebar.write("No saved versions")
else:
    st.sidebar.write("Versioning not available")

# safe load wrapper
def safe_load_seq(trace_path, max_len=None):
    if load_trace is None:
        return []
    try:
        seq = load_trace(str(trace_path), max_len=max_len)
    except TypeError:
        seq = load_trace(str(trace_path))
        if max_len is not None:
            if len(seq) >= max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [0] * (max_len - len(seq))
    except Exception:
        seq = []
    if isinstance(seq, list):
        return [int(x) for x in seq]
    return seq

# main UI modes
mode = st.selectbox("Mode", ["Dataset Review", "Live Analysis", "Tools & Diagnostics", "Fine-Tune Model"])

# ---------- Dataset Review ----------
if mode == "Dataset Review":
    st.header("Dataset Review")
    trace_paths = sorted([str(p) for p in DATA_TRACES_DIR.glob("*.json")]) if DATA_TRACES_DIR.exists() else []
    if not trace_paths:
        st.info("No traces in dataset/traces")
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
                    score, per_token = score_trace_with_model(model, seq, scoring="smoothed_prob")
                except Exception:
                    try:
                        score, per_token = score_trace_with_model(model, seq, scoring="ce")
                    except Exception:
                        score, per_token = None, []
                scores.append(score)
            df["score"] = scores
            st.table(df[["file", "label", "score"]])
            if utils_viz is not None:
                thr = st.number_input("Decision threshold (score > thr => malicious)", value=float((thresholds.get("suggested_threshold", 1.0) if thresholds else 1.0)))
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
                        fig = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {area:.3f})")
                        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

# ---------- Live Analysis ----------
elif mode == "Live Analysis":
    st.header("Live Analysis")
    st.markdown("Pick a precomputed trace or upload one (trace.json).")
    trace_list = sorted([p for p in DATA_TRACES_DIR.glob("*.json")]) if DATA_TRACES_DIR.exists() else []
    chosen_trace = st.selectbox("Precomputed trace", ["--none--"] + [os.path.basename(str(p)) for p in trace_list])
    uploaded_trace = st.file_uploader("Or upload a trace.json", type=["json"])
    if uploaded_trace is not None:
        ttmp = Path(tempfile.gettempdir()) / f"uploaded_trace_{int(time.time())}.json"
        with open(ttmp, "wb") as fh:
            fh.write(uploaded_trace.getbuffer())
        trace_path = str(ttmp)
    elif chosen_trace != "--none--":
        trace_path = str(DATA_TRACES_DIR / chosen_trace)
    else:
        trace_path = None

    if trace_path:
        try:
            with open(trace_path, "r") as fh:
                events = json.load(fh)
        except Exception:
            events = []
        st.success(f"Loaded trace: {Path(trace_path).name}")

        # plot timeline (with scores if available)
        if utils_viz is not None:
            try:
                # compute per-token scores if model available
                per_token_scores = None
                if model is not None:
                    seq = safe_load_seq(trace_path, max_len=200)
                    if seq:
                        try:
                            score, per_token_scores = score_trace_with_model(model, seq, scoring="smoothed_prob")
                        except Exception:
                            score, per_token_scores = score_trace_with_model(model, seq, scoring="ce")
                fig, evdf = utils_viz.plot_trace_timeline(events, scores=per_token_scores)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                if evdf is not None:
                    st.subheader("Event sample")
                    st.table(evdf.head(200))
            except Exception as e:
                st.warning("Trace visualization failed: " + str(e))

        # scoring + top anomalous events
        if model is not None:
            seq = safe_load_seq(trace_path, max_len=200)
            if seq:
                try:
                    score, per_token_scores = score_trace_with_model(model, seq, scoring="smoothed_prob")
                except Exception:
                    score, per_token_scores = score_trace_with_model(model, seq, scoring="ce")
                st.metric("Anomaly Score", f"{score:.4f}")
                thr_val = thresholds.get("suggested_threshold")
                if thr_val is not None:
                    verdict = "QUARANTINE" if score > float(thr_val) else "OK"
                    st.write("Verdict:", verdict, f"(threshold={thr_val})")
                if utils_viz is not None:
                    taf = utils_viz.top_anomalous_events(events, per_token_scores)
                    if taf is not None and not taf.empty:
                        st.subheader("Top anomalous events")
                        st.table(taf)
                    # show anomalous-events heatmap
                    heat_fig, heat_df = utils_viz.top_anomalous_events_heatmap(events, per_token_scores, top_k=25, n_buckets=20)
                    if heat_fig is not None:
                        st.subheader("Anomalous events heatmap")
                        st.plotly_chart(heat_fig, use_container_width=True)
                        with st.expander("Heatmap data (table)"):
                            st.dataframe(heat_df)

        else:
            st.info("Model not loaded; scoring unavailable.")

# ---------- Tools & Diagnostics ----------
elif mode == "Tools & Diagnostics":
    st.header("Tools & Diagnostics")
    st.subheader("Model snapshot / versioning")
    if st.button("Create snapshot (save repo model in models/)"):
        if Path(REPO_MODEL_NAME).exists():
            try:
                ts = time.strftime("%Y%m%dT%H%M%S")
                out = MODELS_DIR / f"ae_model_snapshot_{ts}.pth"
                shutil.copy2(REPO_MODEL_NAME, out)
                st.success(f"Saved snapshot {out.name}")
            except Exception as e:
                st.error("Snapshot failed: " + str(e))
        else:
            st.info("No ae_model.pth in repo root.")

    st.subheader("Event frequency heatmap (dataset)")
    if st.button("Show dataset event frequency heatmap"):
        if utils_viz is None:
            st.info("utils_viz.py missing")
        else:
            trace_paths = [str(p) for p in DATA_TRACES_DIR.glob("*.json")]
            if not trace_paths:
                st.info("No traces")
            else:
                fig = utils_viz.plot_event_heatmap(trace_paths, top_n_events=40, annotate=True, cluster=False)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Could not build heatmap")

    st.subheader("Developer diagnostics")
    if st.checkbox("Show repo root"):
        try:
            st.write(os.listdir("."))
        except Exception as e:
            st.write("listdir failed:", e)

# ---------- Fine-Tune Model ----------
elif mode == "Fine-Tune Model":
    st.header("Fine-Tune Model (in-UI)")
    st.markdown("Upload benign traces and fine-tune. Save uploaded traces to dataset/traces/ with one click.")

    if DISABLE_EXECUTION:
        st.warning("Fine-tuning disabled in this deployment (DISABLE_EXECUTION=true).")
    else:
        uploaded_files = st.file_uploader("Upload benign trace JSONs (multiple)", accept_multiple_files=True, type=["json"])
        arch = st.selectbox("Architecture", ["lstm", "gru", "transformer"], index=["lstm","gru","transformer"].index(default_arch))
        epochs = st.number_input("Epochs", min_value=1, max_value=20, value=3)
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.6f")
        batch_size = st.selectbox("Batch size", [1,2,4,8,16,32], index=3)
        replay_fraction = st.slider("Replay %", 0, 100, 30)
        kl_weight = st.number_input("KL-regularizer weight", min_value=0.0, max_value=1.0, value=0.0)
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            save_and_include = st.button("Save uploads → dataset/traces/ & include")
        with c2:
            export_pkg = st.button("Export retrain package (ZIP)")
        with c3:
            start_ft = st.button("Start Fine-Tuning (in-UI)")

        def write_upload(f):
            name = getattr(f, "name", f"uploaded_{int(time.time()*1000)}.json")
            name = "".join(c for c in name if c.isalnum() or c in ("-","_","."))
            dest = DATA_TRACES_DIR / name
            with open(dest, "wb") as fh:
                fh.write(f.getbuffer())
            return str(dest)

        if save_and_include:
            if not uploaded_files:
                st.error("No uploads.")
            else:
                saved = []
                for f in uploaded_files:
                    try:
                        p = write_upload(f)
                        saved.append(p)
                    except Exception as e:
                        st.warning(f"Failed to save: {e}")
                st.success(f"Saved {len(saved)} files to dataset/traces/ (they will be used in this training session if you start it)")

        if export_pkg:
            if not uploaded_files:
                st.error("Upload files first.")
            else:
                mem = BytesIO()
                with zipfile.ZipFile(mem, mode="w") as zf:
                    names = []
                    for f in uploaded_files:
                        fname = "".join(c for c in f.name if c.isalnum() or c in ("-","_","."))
                        zf.writestr(f"uploaded/{fname}", f.getvalue())
                        names.append(fname)
                    # include some replays
                    replays = sorted([p for p in DATA_TRACES_DIR.glob("benign_*.json")])[:50]
                    for p in replays:
                        zf.write(str(p), arcname=f"replay/{p.name}")
                    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "uploaded": names}
                    zf.writestr("manifest.json", json.dumps(manifest, indent=2))
                mem.seek(0)
                st.download_button("Download retrain ZIP", mem.getvalue(), file_name=f"retrain_pkg_{int(time.time())}.zip")

        if start_ft:
            if build_model is None:
                st.error("model.py missing.")
            elif not uploaded_files:
                st.error("Upload at least one benign trace.")
            else:
                seqs = []
                for f in uploaded_files:
                    tmp = Path(tempfile.gettempdir()) / f"upload_{int(time.time()*1000)}_{f.name}"
                    with open(tmp, "wb") as fh:
                        fh.write(f.getbuffer())
                    s = safe_load_seq(str(tmp), max_len=None)
                    if s:
                        seqs.append(s)
                # include replay sequences
                replays = []
                for p in sorted(DATA_TRACES_DIR.glob("benign_*.json")):
                    s = safe_load_seq(str(p), max_len=None)
                    if s:
                        replays.append(s)
                k = int((replay_fraction/100.0) * len(replays))
                replays = replays[:k] if k > 0 else []
                combined = seqs + replays
                if not combined:
                    st.error("No valid sequences found.")
                else:
                    maxlen = max(len(s) for s in combined)
                    arr = np.zeros((len(combined), maxlen), dtype=np.int64)
                    for i, s in enumerate(combined):
                        arr[i, :len(s)] = s
                    X = torch.LongTensor(arr)
                    ds = torch.utils.data.TensorDataset(X)
                    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
                    base_path = tmp_uploaded_model_path if tmp_uploaded_model_path and Path(tmp_uploaded_model_path).exists() else (REPO_MODEL_NAME if Path(REPO_MODEL_NAME).exists() else None)
                    if base_path is None:
                        st.error("No base model found. Upload or put ae_model.pth in repo root.")
                    else:
                        try:
                            m = build_model(arch, vocab_size=estimated_vocab_size())
                            state = torch.load(base_path, map_location="cpu")
                            try:
                                m.load_state_dict(state)
                            except Exception:
                                sd = m.state_dict()
                                for k0, v0 in state.items():
                                    if k0 in sd and sd[k0].shape == v0.shape:
                                        sd[k0] = v0
                                m.load_state_dict(sd)
                            m.train()
                        except Exception as e:
                            st.error("Model load failed: " + str(e))
                            st.stop()
                        opt = torch.optim.Adam(m.parameters(), lr=lr)
                        criterion = nn.CrossEntropyLoss(ignore_index=0)
                        progress = st.progress(0)
                        loss_ph = st.empty()
                        loss_history = []
                        total = int(epochs) * max(1, len(dl))
                        step = 0
                        for ep in range(int(epochs)):
                            epoch_loss = 0.0
                            for batch in dl:
                                bx = batch[0].long()
                                opt.zero_grad()
                                logits = m(bx)
                                B,T,V = logits.size()
                                loss = criterion(logits.view(-1,V), bx.view(-1))
                                if kl_weight and kl_weight > 0:
                                    kl = compute_kl_regularizer(logits)
                                    loss = loss + kl_weight * kl
                                loss.backward()
                                opt.step()
                                epoch_loss += float(loss.item())
                                step += 1
                                progress.progress(int(step / total * 100))
                            loss_history.append(epoch_loss / max(1, len(dl)))

                            # Plot training loss with Plotly (replaces matplotlib)
                            if len(loss_history) > 0:
                                df_loss = {"epoch": list(range(1, len(loss_history)+1)), "loss": loss_history}
                                fig_loss = px.line(df_loss, x="epoch", y="loss", markers=True, title="Fine-tune Loss")
                                fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
                                loss_ph.plotly_chart(fig_loss, use_container_width=True)

                        st.success("Fine-tune complete")
                        ts = time.strftime("%Y%m%dT%H%M%S")
                        outname = f"ae_model_{arch}_{ts}.pth"
                        outpath = MODELS_DIR / outname
                        try:
                            torch.save(m.state_dict(), str(outpath))
                            with open(str(outpath) + ".meta.json", "w") as fh:
                                json.dump({"arch": arch, "vocab_size": estimated_vocab_size(), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh, indent=2)
                            st.success(f"Saved {outname}")
                        except Exception as e:
                            st.warning("Save failed: " + str(e))
                        try:
                            torch.save(m.state_dict(), REPO_MODEL_NAME)
                            with open(REPO_MODEL_NAME + ".meta.json", "w") as fh:
                                json.dump({"arch": arch, "vocab_size": estimated_vocab_size()}, fh, indent=2)
                            st.info("Updated ae_model.pth in repo root (if writable).")
                        except Exception:
                            st.warning("Could not overwrite ae_model.pth.")
                        # recalibrate threshold
                        st.info("Recalibrating thresholds...")
                        m.eval()
                        benign_scores = []
                        with torch.no_grad():
                            for p in DATA_TRACES_DIR.glob("benign_*.json"):
                                s = safe_load_seq(str(p), max_len=None)
                                if not s:
                                    continue
                                x = torch.LongTensor([s])
                                out = m(x)
                                B,T,V = out.size()
                                losses_token = nn.CrossEntropyLoss(reduction="none", ignore_index=0)(out.view(-1,V), x.view(-1)).cpu().numpy().reshape(B,T)[0]
                                valid = [float(x) for (x,t) in zip(losses_token, x.numpy()[0]) if int(t)!=0]
                                if valid:
                                    benign_scores.append(float(np.mean(valid)))
                            # include uploaded seqs
                            for s in seqs:
                                x = torch.LongTensor([s])
                                out = m(x)
                                B,T,V = out.size()
                                losses_token = nn.CrossEntropyLoss(reduction="none", ignore_index=0)(out.view(-1,V), x.view(-1)).cpu().numpy().reshape(B,T)[0]
                                valid = [float(x) for (x,t) in zip(losses_token, x.numpy()[0]) if int(t)!=0]
                                if valid:
                                    benign_scores.append(float(np.mean(valid)))
                        if len(benign_scores) > 0:
                            p95 = float(np.percentile(benign_scores, 95))
                            p99 = float(np.percentile(benign_scores, 99))
                            mean = float(np.mean(benign_scores))
                            std = float(np.std(benign_scores))
                            data = {"mean": mean, "std": std, "p95": p95, "p99": p99, "suggested_threshold": p99}
                            try:
                                with open(THRESHOLD_PATH, "w") as fh:
                                    json.dump(data, fh, indent=2)
                                st.success("Thresholds recalibrated and saved")
                            except Exception:
                                st.warning("Could not write threshold.json")

st.caption("BFZDD — upgraded final app")
