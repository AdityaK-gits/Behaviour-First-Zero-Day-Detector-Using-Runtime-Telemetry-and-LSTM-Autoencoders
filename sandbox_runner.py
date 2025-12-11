import sys
import json
import time
import runpy
import threading
import os
import argparse

# Global list to store trace events safely
trace_events = []
trace_lock = threading.Lock()

def audit_hook(event, args):
    """
    Capture audit events.
    Interested in: open, socket, connect, subprocess, os.system, etc.
    """
    # Filter for interesting events to reduce noise
    interesting_prefixes = (
        "open", "io.open", 
        "socket", "connect", 
        "os.system", "subprocess", "os.spawn", "os.posix_spawn",
        "shutil", "os.remove", "os.rename", "os.mkdir"
    )
    
    if not event.startswith(interesting_prefixes):
        return

    # Convert args to string-friendly format
    clean_args = []
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            clean_args.append(arg)
        elif isinstance(arg, bytes):
            try:
                clean_args.append(arg.decode('utf-8'))
            except:
                clean_args.append(str(arg))
        else:
            clean_args.append(str(arg))

    entry = {
        "t": time.time(),
        "event": event,
        "args": clean_args
    }
    
    with trace_lock:
        trace_events.append(entry)

import math

def calculate_entropy(filepath):
    """Calculates the Shannon entropy of a file."""
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        if not data:
            return 0
        
        entropy = 0
        for x in range(256):
            p_x = float(data.count(x)) / len(data)
            if p_x > 0:
                entropy += - p_x * math.log(p_x, 2)
        return entropy
    except Exception:
        return 0

def run_sandbox(target_script, output_trace="trace.json"):
    """
    Run the target script with the audit hook enabled.
    """
    # Clear previous traces
    global trace_events
    trace_events = []
    
    # Track touched files for post-execution analysis
    written_files = set()
    
    # We need a wrapper hook to capture written files?
    # Actually, the existing hook captures 'open'.
    # We can just process the trace_events at the end to find 'open' with 'w'
    
    # Add the hook
    sys.addaudithook(audit_hook)
    
    print(f"[*] Sandbox started. Running: {target_script}")
    start_time = time.time()
    
    try:
        # Execute the target script in this process
        runpy.run_path(target_script, run_name="__main__")
    except Exception as e:
        print(f"[!] Target script execution error: {e}")
    finally:
        end_time = time.time()
        print(f"[*] Execution finished in {end_time - start_time:.2f}s")
        
        # Post-processing: Calculate entropy for files opened for writing
        # Filter trace for open events with 'w', 'a', 'x', '+'
        for entry in trace_events:
            if entry["event"] == "open" and len(entry["args"]) >= 2:
                path = str(entry["args"][0])
                mode = str(entry["args"][1])
                if any(x in mode for x in ['w', 'a', 'x', '+']):
                     # resolve relative paths if possible (assuming cwd)
                     if not os.path.isabs(path):
                         path = os.path.abspath(path)
                     written_files.add(path)

        # Add entropy events
        for fpath in written_files:
            if os.path.exists(fpath) and os.path.isfile(fpath):
                ent = calculate_entropy(fpath)
                size = os.path.getsize(fpath)
                trace_events.append({
                    "t": time.time(),
                    "event": "file_analysis",
                    "args": [fpath, size, f"{ent:.4f}"]
                })

        # Save trace
        with open(output_trace, "w") as f:
            json.dump(trace_events, f, indent=2)
        print(f"[*] Trace saved to {output_trace} ({len(trace_events)} events)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFZDD Sandbox Runner")
    parser.add_argument("script", help="Path to the python script to analyze")
    parser.add_argument("--output", default="trace.json", help="Output JSON trace file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.script):
        print(f"Error: Script {args.script} not found.")
        sys.exit(1)
        
    run_sandbox(args.script, args.output)
