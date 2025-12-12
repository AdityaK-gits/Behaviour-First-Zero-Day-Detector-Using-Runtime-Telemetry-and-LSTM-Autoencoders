BFZDD — Behaviour-First Zero-Day Detector
AI-Driven Runtime Telemetry Analyzer Using LSTM / GRU / Transformer Autoencoders

BFZDD is a behaviour-first malware detection system that learns from runtime telemetry rather than signatures.
It uses sequence-model autoencoders to learn normal program behavior and detect anomalies in unknown or zero-day malware.

This project includes:

✅ LSTM / GRU / Transformer-lite autoencoders
✅ In-UI fine-tuning (no terminal needed)
✅ Replay buffer training
✅ KL-regularization for smoother learning
✅ Live anomaly scoring system
✅ Confusion matrix, ROC curve, event frequency heatmaps
✅ Anomalous-event heatmaps for explainability
✅ Model versioning & snapshots
✅ Full Streamlit dashboard
✅ Safe VM-based script execution model (DISABLE_EXECUTION flag)

📌 Features Overview
🔥 1. Behaviour-First Detection

Converts runtime events into a fixed token vocabulary

Learns normal behavior via sequence reconstruction

Computes anomaly scores using:

Cross-entropy loss

Smoothed probability scoring

(Optional) KL-regularization

🚀 2. Multiple Neural Architectures

Choose dynamically in UI:

LSTM Autoencoder

GRU Autoencoder

Transformer-Lite Encoder

Each architecture is compatible with incremental fine-tuning.

🎛 3. Streamlit Dashboard

The dashboard includes:

Dataset Review

View all traces in dataset/traces/

Auto-label benign/malicious based on filename

Score traces with anomaly models

Compute:

Confusion Matrix

Precision / Recall / F1

ROC Curve (AUC)

Live Analysis

Load precomputed traces or upload your own

View timeline plots

Heatmap of anomalous events

Model verdict (OK / QUARANTINE)

Tools & Diagnostics

One-click model snapshot system

Dataset event frequency heatmap

Repo file browser (safe mode)

Fine-Tune Model (No Terminal Needed!)

Upload JSON trace files and fine-tune directly inside the UI:

✔ Upload & Save → dataset/traces/
✔ Replay buffer integration
✔ Live training progress bar
✔ Plot of loss per epoch
✔ Automatic saving into models/
✔ Auto-update ae_model.pth in repo root
✔ Threshold recalibration (p95/p99 of benign set)
✔ Export retraining ZIP manifest for offline GPU training

All without touching the command prompt.

🧠 Model Architecture
Tokenization

Each runtime event is mapped to an integer token via load_trace():

Normalized event names

Dynamic vocab expansion

PAD=0 reserved

estimated_vocab_size() used for building models safely

Autoencoder Models

The final model.py defines:

build_model(arch="lstm"|"gru"|"transformer")

Embedding → Encoder → Projection to vocab logits

Trains to reconstruct the event sequence

High reconstruction error → anomaly

score_trace_with_model()

Returns:

avg_loss = anomaly score

per_token_losses for explainability

Supports "ce" and "smoothed_prob" scoring

compute_kl_regularizer()

Optional KL penalty for model stability.

📊 Explainability
✔ Per-event anomaly identification
✔ Top anomalous events table
✔ Heatmap of anomalous event clusters
✔ Frequency heatmap of global dataset events
✔ ROC, AUC, Confusion Matrix

Explainability is critical for malware analysis and BFZDD provides detailed event-level diagnostics.

Repository Structure: 
project/
│
├── app.py                     # Full Streamlit dashboard
├── model.py                   # Autoencoders + scoring utils
├── utils_viz.py               # Visualization & heatmap utilities
├── versioning.py              # Model snapshot/version helper (optional)
│
├── ae_model.pth               # Primary model
├── ae_model.pth.meta.json     # Saved metadata (arch, vocab, timestamp)
│
├── dataset/
│   └── traces/                # JSON traces (benign & malicious)
│
├── models/                    # Snapshots and fine-tuned versions
│
└── threshold.json             # Auto-calibrated anomaly thresholds

🧪 Running the App
🔧 Local Setup
pip install -r requirements.txt
streamlit run app.py

☁️ Streamlit Cloud

Just push the repo and deploy — no CLI needed.

🔐 Security Guidelines

Because this project deals with malware behavior simulation:

Do NOT run untrusted scripts on Streamlit Cloud.

Local execution must be inside a virtual machine.

DISABLE_EXECUTION flag ensures safety in public deployments.

🧩 Retrain Package

You can export a ZIP containing:

✔ Uploaded benign traces
✔ Replay dataset samples
✔ Manifest file for offline retraining

Useful for GPU-based fine-tuning outside Streamlit Cloud.

🏆 Why BFZDD Stands Out

Not signature-based — detects new malware families

Provides event-level explanations

Supports continual learning

Runs fully inside a UI

Modern ML architectures integrated

Clean, production-quality structure

Perfect for:

Cybersecurity research

Zero-day behavior analysis

AI/ML interviews

Internship & job applications

Demonstration of real applied AI

📞 Contact

Developer: Aditya Kolluru
Email: adityakolluru2004@gmail.com

Location: Bengaluru, India

link: https://iz222gve472hosjdwdeqvu.streamlit.app/
