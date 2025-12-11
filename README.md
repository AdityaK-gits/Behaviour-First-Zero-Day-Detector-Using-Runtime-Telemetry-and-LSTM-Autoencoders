BFZDD - Behaviour-First Zero-Day Detector (starter repo)

Files:
- model.py         : AuditAutoencoder stub and load_trace()
- dataset/scripts/ : Example benign script(s)
- dataset/traces/  : Example trace JSON files (benign_0.json, malware_0.json)
- app.py           : Streamlit dashboard (safe-ready)
- requirements.txt : Python dependencies

Quick start (inside an isolated VM):
1. Create a virtualenv and install requirements.
2. Ensure ae_model.pth and threshold.json exist in repo root (see below to create).
3. Run: streamlit run app.py
4. In Live Analysis choose Analyze selected trace or run script after confirming VM checkbox.

WARNING: Do NOT run untrusted scripts on your host. Use VM_SAFETY.md instructions.

To generate a sample model file (ae_model.pth):
- Run `python -c "from model import AuditAutoencoder; import torch; m=AuditAutoencoder(); torch.save(m.state_dict(),'ae_model.pth'); print('saved')"`
