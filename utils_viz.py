# utils_viz.py — helper plotting and analysis utilities
import os, json
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc

def build_confusion_matrix_df(y_true, y_pred, labels=["benign","malicious"]):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=[f"True:{l}" for l in labels], columns=[f"Pred:{l}" for l in labels])
    prf = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    prf_df = pd.DataFrame({
        "precision": prf[0],
        "recall": prf[1],
        "f1": prf[2],
        "support": prf[3]
    }, index=labels)
    return df_cm, prf_df

def plot_roc_from_labels_scores(labels, scores, positive_label="malicious"):
    y = [1 if l==positive_label else 0 for l in labels]
    fpr, tpr, thr = roc_curve(y, scores)
    area = auc(fpr, tpr)
    return fpr, tpr, area

def plot_trace_timeline(events):
    # events: list of dicts with "t","event","args"
    if not events:
        return None, None
    start = events[0].get("t", 0)
    rows = []
    for i, e in enumerate(events):
        t = e.get("t", start) - start
        rows.append({"index": i, "time": t, "event": e.get("event",""), "detail": json.dumps(e.get("args",""))})
    df = pd.DataFrame(rows)
    fig = px.scatter(df, x="time", y="index", hover_data=["event","detail"], title="Trace Timeline (time vs event index)")
    fig.update_yaxes(autorange="reversed")
    return fig, df

def top_anomalous_events(events, losses, top_k=10):
    # events: list, losses: list of floats (same length)
    if not events or not losses:
        return pd.DataFrame([], columns=["idx","event","args","error"])
    pairs = []
    for i,(e,l) in enumerate(zip(events, losses)):
        pairs.append((i, e.get("event",""), e.get("args",""), float(l)))
    pairs = sorted(pairs, key=lambda x: -x[3])[:top_k]
    df = pd.DataFrame(pairs, columns=["idx","event","args","error"])
    return df

def build_event_freq_matrix(trace_paths):
    rows = []
    all_types = set()
    for p in trace_paths:
        try:
            with open(p,"r") as f:
                events = json.load(f)
        except Exception:
            events = []
        counts = {}
        for e in events:
            et = e.get("event","unknown")
            counts[et] = counts.get(et,0) + 1
            all_types.add(et)
        rows.append((os.path.basename(p), counts))
    cols = sorted(list(all_types))
    data = []
    names = []
    for name, counts in rows:
        row = [counts.get(c,0) for c in cols]
        data.append(row)
        names.append(name)
    df = pd.DataFrame(data, index=names, columns=cols)
    return df
