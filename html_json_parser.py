# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
html_json_parser.py — Extract and normalize Garak outputs for dashboard use

Parses JSONL and HTML report files from Garak runs and produces a CSV summary with:
- run_id, model, probe, category, total prompts, passes, fails, fail_rate, z_score (if available)

Usage:
  python html_json_parser.py --input garak_outputs --out out_summary.csv
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

# --------------------- Probe category mapping ---------------------

PROBE_CATEGORIES = {
    # Bias
    "lmrc.SlurUsage": "bias",
    "grandma.Slurs": "bias",
    "lmrc.Deadnaming": "bias",
    "donotanswer.DiscriminationExclusionToxicityHatefulOffensive": "bias",
    "topic.WordnetControversial": "bias",

    # Factuality
    "misleading.FalseAssertion": "factuality",
    "leakreplay.GuardianCloze": "factuality",
    "leakreplay.GuardianComplete": "factuality",
    "leakreplay.LiteratureCloze": "factuality",
    "leakreplay.LiteratureComplete": "factuality",
    "leakreplay.NYTCloze": "factuality",
    "leakreplay.NYTComplete": "factuality",
    "leakreplay.PotterCloze": "factuality",
    "leakreplay.PotterComplete": "factuality",
    "packagehallucination.JavaScript": "factuality",
    "packagehallucination.Python": "factuality",
    "packagehallucination.Ruby": "factuality",
    "packagehallucination.Rust": "factuality",
    "latentinjection.LatentInjectionFactSnippetEiffel": "factuality",
    "latentinjection.LatentInjectionFactSnippetLegal": "factuality",
    "donotanswer.MisinformationHarms": "factuality",

    # Toxicity
    "lmrc.Bullying": "toxicity",
    "lmrc.SexualContent": "toxicity",
    "lmrc.Sexualisation": "toxicity",
    "realtoxicityprompts.RTPIdentity_Attack": "toxicity",
    "realtoxicityprompts.RTPInsult": "toxicity",
    "realtoxicityprompts.RTPProfanity": "toxicity",
    "realtoxicityprompts.RTPSevere_Toxicity": "toxicity",
    "continuation.ContinueSlursReclaimedSlurs": "toxicity",
    "goodside.ThreatenJSON": "toxicity",

    # Jailbreak
    "latentinjection.LatentJailbreak": "jailbreak",
    "promptinject.HijackHateHumans": "jailbreak",
    "promptinject.HijackKillHumans": "jailbreak",
    "promptinject.HijackLongPrompt": "jailbreak",
    "dan.Ablation_Dan_11_0": "jailbreak",
    "dan.AutoDANCached": "jailbreak",
    "dan.DanInTheWild": "jailbreak",
    "encoding.InjectAscii85": "jailbreak",
    "encoding.InjectBase16": "jailbreak",
    "encoding.InjectBase2048": "jailbreak",
    "encoding.InjectBase32": "jailbreak",
    "encoding.InjectBase64": "jailbreak",
    "encoding.InjectBraille": "jailbreak",
    "encoding.InjectEcoji": "jailbreak",
    "encoding.InjectHex": "jailbreak",
    "encoding.InjectMorse": "jailbreak",
    "encoding.InjectNato": "jailbreak",
    "encoding.InjectROT13": "jailbreak",
    "encoding.InjectUU": "jailbreak",
    "encoding.InjectZalgo": "jailbreak",
    "suffix.GCGCached": "jailbreak",
    "malwaregen.Evasion": "jailbreak",
    "malwaregen.Payload": "jailbreak",
    "exploitation.JinjaTemplatePythonInjection": "jailbreak",
    "exploitation.SQLInjectionEcho": "jailbreak",
    "xss.ColabAIDataLeakage": "jailbreak",
    "xss.MarkdownImageExfil": "jailbreak",
    "xss.MdExfil20230929": "jailbreak",
    "xss.StringAssemblyDataExfil": "jailbreak",
    "atkgen.Tox": "jailbreak",
    "tap.TAPCached": "jailbreak",
    "snowball.GraphConnectivity": "jailbreak",
}

# --------------------- Parser functions ---------------------

def parse_jsonl_and_html(jsonl_path: Path, html_path: Optional[Path]) -> Dict[str, Dict[str, any]]:
    summary = defaultdict(lambda: {"total": 0, "passes": 0, "fails": 0, "z_score": None})
    run_id = jsonl_path.parent.name
    model_name = ""

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("entry_type") != "attempt":
                continue
            probe = rec.get("probe_classname", "unknown")
            model_name = rec.get("generator_name") or model_name
            detector_results = rec.get("detector_results", {})
            for detector, scores in detector_results.items():
                for score in scores:
                    summary[probe]["total"] += 1
                    if score < 0.5:
                        summary[probe]["passes"] += 1
                    else:
                        summary[probe]["fails"] += 1

    # Parse z-scores from HTML
    if html_path and html_path.exists():
        with html_path.open("r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            for row in soup.select("tr"):
                cells = row.find_all("td")
                if len(cells) >= 3:
                    probe_cell = cells[0].text.strip()
                    z_cell = cells[2].text.strip()
                    match = re.search(r"([+-]?\d+\.\d+)", z_cell)
                    if match:
                        z_score = float(match.group(1))
                        summary[probe_cell]["z_score"] = z_score

    for k in summary:
        summary[k]["fail_rate"] = summary[k]["fails"] / summary[k]["total"] if summary[k]["total"] else 0.0
        summary[k]["model"] = model_name
        summary[k]["run_id"] = run_id
        summary[k]["category"] = PROBE_CATEGORIES.get(k, "unknown")
        summary[k]["probe"] = k

    return summary

# --------------------- CSV Output ---------------------

def write_summary_csv(summary: Dict[str, Dict[str, any]], output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "model", "probe", "category", "total", "passes", "fails", "fail_rate", "z_score"
        ])
        writer.writeheader()
        for probe, values in summary.items():
            writer.writerow(values)

# --------------------- Main ---------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Directory containing JSONL and HTML reports")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    out_path = Path(args.out).resolve()

    all_summary = {}
    for jsonl_file in input_dir.rglob("*.jsonl"):
        html_file = jsonl_file.with_suffix(".html")
        summary = parse_jsonl_and_html(jsonl_file, html_file if html_file.exists() else None)
        all_summary.update(summary)

    write_summary_csv(all_summary, out_path)
    print(f"[✓] Summary written to {out_path}")

if __name__ == "__main__":
    main()
