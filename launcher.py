# SPDX-License-Identifier: Apache-2.0
import os
import yaml
import subprocess
from datetime import datetime




def main() -> str:
    ...
    # === CONFIG FILES ===
    MODELS_FILE = "models.yaml"
    PROBES_FILE = "probes.yaml"
    OUTPUT_DIR = "garak_outputs"

    # === LOAD CONFIGS ===
    def load_yaml(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    models_config = load_yaml(MODELS_FILE)
    probes_config = load_yaml(PROBES_FILE)

    model_version = models_config.get("version", "v0")
    probe_version = probes_config.get("version", "v0")
    models = models_config.get("models", [])
    probes = probes_config.get("probes", [])

    # === CREATE TIMESTAMPED OUTPUT FOLDER ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_models_{model_version}_probes_{probe_version}"
    run_output_path = os.path.abspath(os.path.join(OUTPUT_DIR, run_id))
    os.makedirs(run_output_path, exist_ok=True)

    # === RUN GARAK FOR EACH MODEL ===
    for model in models:
        # Handle if model is a dict
        if isinstance(model, dict):
            model_name = model.get("name", "unknown_model")
        else:
            model_name = model

        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        report_prefix = os.path.join(run_output_path, safe_model_name)

        # ✅ Ensure parent directory exists (needed!)
        os.makedirs(os.path.dirname(report_prefix), exist_ok=True)

        command = [
            "python", "-m", "garak",
            "--model_type", "huggingface",
            "--model_name", model_name,
            "--probes", ",".join(probes),
            "--report_prefix", report_prefix
        ]

        print(f"\n📊 Running benchmark for: {model_name}")
        subprocess.run(command)
        print(f"[DONE] Garak run output saved to: {run_output_path}")
    print(f"[DONE] Garak run output saved to: {run_output_path}")
    return run_output_path


if __name__ == "__main__":
    main()

