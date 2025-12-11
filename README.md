🛡️ BFZDD — Behaviour-First Zero-Day Detector
AI-powered behavioural anomaly detection for zero-day malware using Python Audit Hooks + LSTM Autoencoders.
🔥 Overview

BFZDD is a lightweight, behaviour-first malware detection framework designed to identify zero-day attacks without relying on signatures or known malware samples.
Instead of analyzing file hashes or byte patterns, BFZDD captures runtime behaviour — such as file operations, socket events, and process execution — and uses an LSTM Autoencoder to detect abnormal activity.

This makes BFZDD resistant to:

Obfuscation

Polymorphism

AI-generated malware variants

Unknown attack patterns

🧩 Key Features
✔ Python Audit Hook Sandbox

Captures real-time behaviour: open(), socket, exec, file writes, and more.

✔ Polymorphic Malware Generator

Synthetic adversarial samples used for testing detection robustness.

✔ LSTM Autoencoder Detection Engine

Learns benign behaviour and flags anomalies using reconstruction error.

✔ Explainability

Highlights top anomalous events and computes entropy of written files.

✔ Streamlit Dashboard

Interactive UI for dataset review, live sandbox execution, and anomaly scoring.

✔ VM Safety Guide

Ensures safe execution of suspicious code inside an isolated virtual machine.

📁 Project Structure
BFZDD/
│
├── app.py                 # Streamlit dashboard
├── model.py               # AuditAutoencoder + trace loader
├── sandbox_runner.py      # Python audit-hook sandbox + entropy analysis
├── calibrate_threshold.py # Threshold calibration for detection
├── threshold.json         # Saved thresholds (generated)
├── requirements.txt       # Dependencies
├── VM_SAFETY.md           # Safety guidelines
├── README.md              
│
└── dataset/
    ├── traces/
    │     ├── benign_0.json
    │     ├── malware_0.json
    └── scripts/
          ├── benign_0.py

⚙️ Installation
git clone https://github.com/yourusername/BFZDD.git
cd BFZDD
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

🧪 Generating Model & Thresholds
Train model (optional):
python train.py

Calibrate thresholds:
python calibrate_threshold.py

🚀 Run the Streamlit Dashboard
streamlit run app.py


Runs in your browser at:

http://localhost:8501/

⚠️ Important Safety Warning

Running arbitrary scripts executes code.

👉 Always use a virtual machine
👉 Disable networking
👉 Follow VM_SAFETY.md carefully

Never run unknown code directly on your host machine.

📊 Results Example
Sample	Score	Verdict
benign_0.json	0.41	✔ Normal
malware_0.json	1.12	🔥 Quarantine

BFZDD successfully separates benign vs unknown-malicious behaviour.

🛠️ Future Improvements

Multi-process behaviour graphs (GNN-based detection)

Sysmon integration for deeper telemetry

Cross-platform sandboxing using eBPF

Real malware dataset evaluation inside safe lab environments

💡 Author

Aditya Kolluru
B.Tech CSE — Cybersecurity & AI Enthusiast
