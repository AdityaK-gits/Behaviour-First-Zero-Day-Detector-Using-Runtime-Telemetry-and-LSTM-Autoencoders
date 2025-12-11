# dataset_gen.py
# Generates simple synthetic benign traces and a few polymorphic malicious traces
# Output: dataset/traces/*.json and dataset/scripts/*.py

import os
import json
import random
from pathlib import Path

OUT_TRACES = Path("dataset/traces")
OUT_SCRIPTS = Path("dataset/scripts")
OUT_TRACES.mkdir(parents=True, exist_ok=True)
OUT_SCRIPTS.mkdir(parents=True, exist_ok=True)

# small helper to write a script and corresponding trace-like json (simulated)
def write_script_and_trace(name, events, script_code):
    script_path = OUT_SCRIPTS / f"{name}.py"
    with open(script_path, "w") as f:
        f.write(script_code)
    trace_path = OUT_TRACES / f"{name}.json"
    with open(trace_path, "w") as t:
        json.dump(events, t, indent=2)

# Simple event templates to simulate audit hook output
def make_open(path, mode="w"):
    return {"t": 1.0, "event": "open", "args": [path, mode]}

def make_socket():
    return {"t": 1.0, "event": "socket", "args": ["AF_INET", "SOCK_STREAM"]}

def make_connect(ip="127.0.0.1", port=80):
    return {"t": 1.0, "event": "connect", "args": [ip, port]}

def make_file_analysis(path, size, entropy):
    return {"t": 1.0, "event": "file_analysis", "args": [path, size, f"{entropy:.4f}"]}

# Generate benign scripts/traces
for i in range(20):
    name = f"benign_{i}"
    # benign script: create a file, write some predictable content, no sockets
    script = f"""# benign_{i}.py
with open('out_{i}.txt','w') as f:
    f.write('hello benign {i}\\n'*10)
"""
    events = []
    events.append(make_open(f"out_{i}.txt", "w"))
    # small entropy
    events.append(make_file_analysis(f"out_{i}.txt", 100 + i, random.uniform(1.0, 3.0)))
    write_script_and_trace(name, events, script)

# Generate 6 polymorphic malicious-ish samples (random noise, sockets)
for i in range(6):
    name = f"malware_{i}"
    # malware script: random writes + a connect
    script = f"""# malware_{i}.py
with open('tmp_{i}.bin','wb') as f:
    f.write(b'\\x00\\xff' * {100+i})
import socket
s = socket.socket()
try:
    s.connect(('127.0.0.1', {8000 + i}))
    s.close()
except Exception:
    pass
"""
    events = []
    events.append(make_open(f"tmp_{i}.bin", "w"))
    # high entropy for "malicious" writes
    events.append(make_file_analysis(f"tmp_{i}.bin", 512 + i*10, random.uniform(6.0, 7.5)))
    events.append(make_socket())
    events.append(make_connect("127.0.0.1", 8000 + i))
    write_script_and_trace(name, events, script)

print("Generated dataset/traces and dataset/scripts (20 benign, 6 malware).")
