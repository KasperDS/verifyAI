# 🛡️ VerifyAI

**VerifyAI** is a modular, reproducible benchmarking pipeline for evaluating safety, bias, and alignment in open-source language models using [Garak](https://github.com/NVIDIA/garak).

Built to support researchers, developers, and policymakers in understanding how well models perform on real-world safety tests — across race, gender, compliance, slurs, jailbreaks, and more.

## 🚀 Features

- ✅ We benchmark models locally or in the cloud 
- ✅ Modular config system (models + probes)
- ✅ Generates structured `.jsonl` and `.html` reports
- ✅ Timestamped + versioned output folders
- ✅ Lightweight and reproducible (Python + YAML)

## 📁 Project Structure

```
VerifyAI/
├── launcher.py            # Python script to run Garak using configs
├── models.yaml            # List of models to evaluate
├── probes.yaml            # List of probes to run
├── requirements.txt       # Python dependencies
├── .gitignore             # Output and environment exclusions
└── README.md              # You're here!
```

## ⚙️ Setup

1. **Clone the repo**
    ```bash
    git clone https://github.com/YOUR_USERNAME/VerifyAI.git
    cd VerifyAI
    ```

2. **(Optional) Create a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🧪 Running a Benchmark

```bash
python run_and_parse_wrapper.py
```

This will:
- Load all models from `models.yaml`
- Run all probes from `probes.yaml`
- Output results in `garak_outputs/<timestamp>_model_<version>_probe_<version>/`

Each model will generate:
- A `.report.jsonl` (machine-readable)
- A `.report.html` (human-readable)
- A '.<timestamp> folder under parsed_reports including summary reports per model, and run

## 📊 Example Probes

These test for:
- 🔸 Social bias
- 🔸 Discrimination / slurs
- 🔸 Jailbreak vulnerability
- 🔸 Prompt injection handling
- 🔸 Toxicity refusal

Probes are defined in `probes.yaml` using Garak’s built-in plugins.

## 📬 Get Involved

VerifyAI is in early development. We’re building toward:

- 🔹 Public dashboards
- 🔹 Community-submitted probes
- 🔹 Periodic safety reports
- 🔹 CLI-based model testing

**Join the mission to make LLM safety transparent and accessible.**

> ✉️ Want updates or to contribute? [Join the list](https://tally.so/r/wdM4EK)

## 📄 License

This project is licensed under the **Apache 2.0 License**.  
See the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [NVIDIA Garak](https://github.com/NVIDIA/garak)
- Inspired by the open-source model verification ecosystem

## 🧠 About

VerifyAI was created to provide trustworthy, independent benchmarking of language models at scale — without sacrificing openness, usability, or depth.

```python
# SPDX-License-Identifier: Apache-2.0
```
