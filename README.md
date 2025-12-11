🚀 BFZDD — Behaviour-First Zero-Day Detector
Runtime Telemetry + LSTM Autoencoders for Zero-Day Malware Detection

Author: Aditya Kolluru

🧩 Overview

BFZDD (Behaviour-First Zero-Day Detector) is an AI-powered, behavior-based malware detection system that identifies zero-day attacks using runtime telemetry instead of static signatures.

Traditional antiviruses fail against polymorphic & zero-day malware because they rely on known patterns. BFZDD learns benign program behavior and flags anomalous execution patterns using an LSTM Autoencoder.

This system includes:

✔ Runtime sandbox using Python Audit Hooks

✔ Synthetic benign & polymorphic malware generation
✔ LSTM Autoencoder anomaly detection
✔ Threshold calibration module
✔ Advanced visualization dashboard
✔ Live analysis of traces
✔ Confusion matrix, ROC curve, event frequency heatmaps
✔ Trace timeline visualizer
✔ Model versioning support
✔ Full cloud deployment (Streamlit)

This project is end-to-end, modular, and built to demonstrate industry-grade AI + cybersecurity engineering.

🧠 Key Features
🔒 1. Behaviour-Based Detection

Detects malware by observing behavior, not signatures.

📡 2. Runtime Telemetry Capture

Using Audit Hooks, BFZDD logs:

file opens, reads, writes

socket connections

subprocess creation

file deletion/renaming

entropy of written files

🤖 3. LSTM Autoencoder Model

Learns normal behavior → high reconstruction loss signals anomalies.

🧮 4. Threshold Calibration

Calibrates anomaly threshold using benign samples:

suggested_threshold = 99th percentile benign score

📊 5. Full Visualization Suite

Confusion Matrix

Precision / Recall / F1

ROC Curve with AUC

Event Frequency Heatmap

Trace Timeline

Top Anomalous Events

🧪 6. Live Trace Analysis

Upload or select a trace → get:

anomaly score

verdict (OK / QUARANTINE)

detailed anomaly breakdown

💾 7. Model Versioning

Save & load different model versions via versioning.py.

☁️ 8. Cloud Deployment

Runs on Streamlit Cloud with:

automatic model loading

optional user-uploaded .pth

static mode (script execution disabled for safety)

📁 Repository Structure
├── app.py                      # Streamlit dashboard
├── model.py                    # LSTM Autoencoder + scoring + trace loader
├── sandbox_runner.py           # Runtime telemetry capture
├── dataset_gen.py              # Synthetic dataset generator
├── polymorphic_gen.py          # Malware generator
├── train.py                    # Model training code
├── calibrate_threshold.py      # Threshold computation
├── utils_viz.py                # ROC, confusion matrix, heatmaps, timelines
├── versioning.py               # Model version management
│
├── ae_model.pth                # Trained model (repo-loaded)
├── threshold.json              # Threshold stats
│
├── dataset/
│   ├── traces/                 # Trace JSON samples
│   └── scripts/                # Benign & malicious scripts
│
├── models/                     # Saved historical models
├── VM_SAFETY.md                # Sandbox usage safety documentation
└── README.md

⚙️ How It Works
1️⃣ Generate Dataset
python dataset_gen.py


Generates:

benign traces

polymorphic malware traces

Stored under dataset/traces/.

2️⃣ Train Model
python train.py


Saves model as:

ae_model.pth

3️⃣ Calibrate Threshold
python calibrate_threshold.py


Generates:

threshold.json

4️⃣ Launch Dashboard

Local:

streamlit run app.py


Streamlit Cloud:
Add repo → Deploy.

🖥️ Streamlit Features
📊 Dataset Review

Score all dataset traces

Visualize confusion matrix

View Precision / Recall / F1

ROC Curve

🔍 Live Analysis

Upload or choose a trace

Shows:

timeline visualization

anomaly score

verdict

top anomalous events

📈 Tools & Diagnostics

Event frequency heatmap

Model version saving/loading

Repo inspection

🔬 Anomaly Detection Logic

Reconstruction loss for each token:

per_token_loss = CrossEntropy(reconstructed, original)


Final anomaly score:

anomaly_score = mean(per_token_loss)


If anomaly_score > threshold ⇒ malicious.

🧰 Technology Stack
Layer	Tools
ML	PyTorch (LSTM Autoencoder)
Visualization	Plotly, Streamlit
Runtime Telemetry	Python Audit Hooks
Deployment	Streamlit Cloud
Data Handling	JSON, Pandas
🛡 Security Guidelines

🚫 Never execute unknown scripts on Streamlit Cloud.
✔ Run malicious scripts ONLY in a Virtual Machine with:

no internet

snapshots enabled

isolated environment

See VM_SAFETY.md for instructions.

📊 Example Outputs
ROC Curve

Behavior-based separation of benign vs malicious sequences.

Confusion Matrix

Performance evaluation at any threshold.

Trace Timeline

Event-by-event behavioral visualization.

Anomalous Events Table

Pinpoints suspicious behavior tokens.

📢 Why BFZDD Matters

This project demonstrates:

AI for security

sequence modeling

anomaly detection

telemetry processing

real-world cybersecurity engineering

end-to-end full-stack ML pipeline

deployment & visualization

Comparable to the approach used in modern XDR (Extended Detection & Response) systems.

🚀 Future Enhancements

Transformer-based anomaly detector

Graph Neural Networks for behavior graphs

Cuckoo/Firecracker sandbox integration

Real-world malware datasets

Explainable AI for attack attribution

📝 Citation

“Behaviour-First Zero-Day Detector (BFZDD) by Aditya Kolluru (2025)”
working link : https://iz222gve472hosjdwdeqvu.streamlit.app/

