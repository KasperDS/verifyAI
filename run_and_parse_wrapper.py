# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
from datetime import datetime


# Step 1: Import and run launcher
try:
    import launcher
except ImportError:
    print("[ERROR] Could not import launcher.py. Make sure this script is in the same folder.")
    sys.exit(1)

import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")


print("[INFO] Running Garak launcher...")
output_path = launcher.main()  # returns full garak_outputs/<timestamp>/... path

# Step 2: Derive matching parsed output path

# Extract timestamp from garak output folder name (e.g. 20250821_131400_xyz)
raw = Path(output_path).name.split("_")[0:2]
timestamp_compact = "_".join(raw)  # '20250821_131400'
timestamp_formatted = datetime.strptime(timestamp_compact, "%Y%m%d_%H%M%S").strftime("%Y%m%d-%H%M%S")

parsed_path = f"parsed_reports/{timestamp_formatted}"


print(f"[INFO] Parsing {output_path} into {parsed_path}")

# Step 3: Run the parser
subprocess.run([
    "python3", "json_parser.py",
    "--input", output_path,
    "--out", parsed_path
])
