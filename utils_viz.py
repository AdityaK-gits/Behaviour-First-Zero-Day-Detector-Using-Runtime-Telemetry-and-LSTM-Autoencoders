# utils_viz.py — helper plotting and analysis utilities (final)
import os
import json
from typing import List, Tuple, Optional, Sequence, Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc


def build_confusion_matrix_df(y_true: Sequence[str], y_pred: Sequence[str], labels: List[str] = ["benign", "malicious"]):
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


def plot_roc_from_labels_scores(labels: Sequence[str], scores: Sequence[float], positive_label: str = "malicious") -> Tuple[np.ndarray, np.ndarray, float]:
    y = [1 if l == positive_label else 0 for l in labels]
    fpr, tpr, thr = roc_curve(y, scores)
    area = auc(fpr, tpr)
    return fpr, tpr, area


def plot_trace_timeline(events: List[dict], scores: Optional[Sequence[float]] = None):
    if not events:
        return None, None
    start = events[0].get("t", 0)
    rows = []
    for i, e in enumerate(events):
        t = e.get("t", start) - start if "t" in e else i
        ev = e.get("event", "")
        args = e.get("args", "")
        try:
            detail = json.dumps(args) if args is not None else ""
        except Exception:
            detail = str(args)
        rows.append({"idx": i, "time": t, "event": ev, "detail": detail})
    df = pd.DataFrame(rows)
    if scores is not None and len(scores) == len(df):
        df["score"] = scores
        sc = np.array(scores, dtype=float)
        finite = np.isfinite(sc)
        if finite.any():
            mn, mx = sc[finite].min(), sc[finite].max()
            if mx > mn:
                df["score_norm"] = df["score"].apply(lambda x: (x - mn) / (mx - mn) if np.isfinite(x) else 0.0)
            else:
                df["score_norm"] = df["score"].apply(lambda x: 0.5 if np.isfinite(x) else 0.0)
        else:
            df["score_norm"] = 0.0
        fig = px.scatter(df, x="time", y="idx", color="score_norm", hover_data=["event", "detail", "score"],
                         title="Trace Timeline (time vs event index) — colored by anomaly score")
    else:
        fig = px.scatter(df, x="time", y="idx", hover_data=["event", "detail"], title="Trace Timeline (time vs event index)")
    fig.update_yaxes(autorange="reversed")
    return fig, df


def top_anomalous_events(events: List[dict], losses: Optional[Sequence[float]], top_k: int = 10, higher_is_anomalous: bool = True) -> pd.DataFrame:
    cols = ["idx", "score", "score_z", "event", "args"]
    if not events:
        return pd.DataFrame([], columns=cols)
    n = len(events)
    if losses is None or len(losses) != n:
        losses = [0.0] * n
    sc = np.array([float(x) if (x is not None and (isinstance(x, (int, float)) and np.isfinite(x))) else 0.0 for x in losses], dtype=float)
    if not higher_is_anomalous:
        sc = -sc
    finite = np.isfinite(sc)
    if finite.sum() > 1:
        mn = np.mean(sc[finite])
        sd = np.std(sc[finite])
        if sd == 0:
            zs = np.zeros_like(sc)
        else:
            zs = (sc - mn) / (sd + 1e-12)
    else:
        zs = np.zeros_like(sc)
    pairs = []
    for i, (e, s, z) in enumerate(zip(events, sc, zs)):
        evname = e.get("event", "") if isinstance(e, dict) else str(e)
        args = e.get("args", "") if isinstance(e, dict) else ""
        try:
            args_serial = json.dumps(args)
        except Exception:
            args_serial = str(args)
        pairs.append((i, float(s), float(z), evname, args_serial))
    pairs_sorted = sorted(pairs, key=lambda x: -x[1])[:top_k]
    df = pd.DataFrame(pairs_sorted, columns=cols)
    return df


def build_event_freq_matrix(trace_paths: List[str]) -> pd.DataFrame:
    rows = []
    all_types = set()
    for p in trace_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                events = json.load(f)
        except Exception:
            events = []
        counts = {}
        for e in events:
            if isinstance(e, dict):
                et = e.get("event", "unknown")
            else:
                try:
                    et = str(e)
                except Exception:
                    et = "unknown"
            counts[et] = counts.get(et, 0) + 1
            all_types.add(et)
        rows.append((os.path.basename(p), counts))
    cols = sorted(list(all_types))
    data = []
    names = []
    for name, counts in rows:
        row = [counts.get(c, 0) for c in cols]
        data.append(row)
        names.append(name)
    if len(data) == 0:
        return pd.DataFrame([], columns=cols)
    df = pd.DataFrame(data, index=names, columns=cols)
    return df


# --- Heatmap helpers (new) ---
import math
from typing import Sequence, Optional
from typing import List as ListType

def plot_event_heatmap(trace_paths: Sequence[str],
                       top_n_events: int = 50,
                       annotate: bool = True,
                       cluster: bool = False):
    df = build_event_freq_matrix(list(trace_paths))
    if df.empty:
        return None
    col_sums = df.sum(axis=0).sort_values(ascending=False)
    top_cols = list(col_sums.index[:top_n_events])
    df_small = df[top_cols].copy()
    if cluster:
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            if df_small.shape[0] > 1:
                lr = linkage(df_small.values, method="average")
                row_order = leaves_list(lr)
                df_small = df_small.iloc[row_order, :]
            if df_small.shape[1] > 1:
                lc = linkage(df_small.values.T, method="average")
                col_order = leaves_list(lc)
                df_small = df_small.iloc[:, col_order]
        except Exception:
            pass
    fig = go.Figure(data=go.Heatmap(
        z=df_small.values,
        x=list(df_small.columns),
        y=list(df_small.index),
        colorscale="Viridis",
        colorbar=dict(title="Count"),
        zmin=0
    ))
    fig.update_layout(title=f"Event frequency heatmap (top {len(df_small.columns)} events)",
                      xaxis_title="Event Type",
                      yaxis_title="Trace (file)")
    if annotate:
        annotations = []
        for i, row in enumerate(df_small.index):
            for j, col in enumerate(df_small.columns):
                val = df_small.iat[i, j]
                if val != 0:
                    annotations.append(dict(x=col, y=row, text=str(int(val)),
                                            font=dict(color="white", size=10),
                                            showarrow=False))
        fig.update_layout(annotations=annotations)
    return fig


def top_anomalous_events_heatmap(events: Sequence[dict],
                                 scores: Sequence[float],
                                 top_k: int = 25,
                                 n_buckets: int = 20,
                                 fill_value: float = 0.0):
    import pandas as pd
    if not events or not scores or len(events) != len(scores):
        return None, pd.DataFrame()
    T = len(events)
    types = []
    for e in events:
        if isinstance(e, dict):
            et = e.get("event", "unknown")
        else:
            et = str(e)
        types.append(str(et))
    bucket_idx = [min(n_buckets - 1, int(math.floor((i / max(1, T - 1)) * (n_buckets - 1)))) for i in range(T)]
    accum = {}
    counts = {}
    for et, b, s in zip(types, bucket_idx, scores):
        key = (et, b)
        accum[key] = accum.get(key, 0.0) + float(s)
        counts[key] = counts.get(key, 0) + 1
    type_totals = {}
    for (et, b), val in accum.items():
        type_totals[et] = type_totals.get(et, 0.0) + val
    sorted_types = sorted(type_totals.items(), key=lambda x: -x[1])
    top_types = [t for t, _ in sorted_types[:top_k]]
    import numpy as np
    heat = np.full((len(top_types), n_buckets), fill_value, dtype=float)
    for i, et in enumerate(top_types):
        for b in range(n_buckets):
            val = accum.get((et, b), 0.0)
            c = counts.get((et, b), 0)
            heat[i, b] = (val / c) if c > 0 else 0.0
    cols = [f"bucket_{i}" for i in range(n_buckets)]
    df_heat = pd.DataFrame(heat, index=top_types, columns=cols)
    fig = go.Figure(data=go.Heatmap(
        z=df_heat.values,
        x=cols,
        y=list(df_heat.index),
        colorscale="YlOrRd",
        colorbar=dict(title="Avg anomaly score")
    ))
    fig.update_layout(title=f"Anomalous events heatmap (Top {len(top_types)} event types over {n_buckets} time buckets)",
                      xaxis_title="Time bucket (trace progression)",
                      yaxis_title="Event Type")
    return fig, df_heat
