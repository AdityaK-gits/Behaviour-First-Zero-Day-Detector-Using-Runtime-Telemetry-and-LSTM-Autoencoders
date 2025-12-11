# Running BFZDD in a Virtual Machine (Safety Guide)

To ensure maximum safety when analyzing potential malware (even synthetic), it is recommended to run the sandbox component inside an isolated Virtual Machine (VM).

## 1. Safety First
- **Network Isolation**: Disable the network adapter (NIC) of the VM or set it to "Host-Only" to prevent malware from contacting external C2 servers.
- **Snapshots**: Always take a clean snapshot of your VM *before* running any analysis. Revert to this snapshot after every session.

## 2. Setting up the VM
1.  **Install VirtualBox** or **VMware Workstation Player**.
2.  **Create a Windows VM** (Windows 10/11) or Linux VM.
3.  **Install Python 3.9+** inside the VM.
4.  **Copy the BFZDD project** into the VM (e.g., via Shared Folders or USB, then disable sharing).

## 3. Running Analysis
Inside the VM, use the provided script to run the runner:

```cmd
cd C:\path\to\BFZDD
python sandbox_runner.py malware_sample.py --output trace.json
```

## 4. Retrieving Results
1.  After analysis, copy the `trace.json` file back to your host machine (enable Shared Folder briefly or use a USB drive).
2.  **Revert the VM** to the clean snapshot immediately.

## Automating with `run_in_vm.bat` (Conceptual)
If using VirtualBox, you can automate this:

```bat
@echo off
echo [*] Restoring Clean Snapshot...
VBoxManage snapshot "BFZDD_VM" restore "CleanState"
echo [*] Starting VM...
VBoxManage startvm "BFZDD_VM"
REM Use VBoxManage guestcontrol to execute script...
```
