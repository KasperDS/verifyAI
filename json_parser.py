# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
parse_garak.py — Normalize NVIDIA Garak JSONL reports for VerifyAI

Scans a directory of Garak runs (each run produces JSONL files) and
emits normalized CSVs and JSON aggregates that a frontend can consume.

Usage:
  python parse_garak.py --input garak_outputs --out out
  python parse_garak.py --input garak_outputs/2025-08-10T14-33-59 --out out/specific_run

Outputs:
  - normalized.csv: long-form rows of (run_id, model, probe, prompt, response, detector, score, passed, ...)
  - model_summary.csv: per model and probe aggregates
  - probe_summary.csv: probe aggregates across models
  - per_model/*.json: compact JSON for frontend cards (metrics + links)
  - per_run/*.json: run metadata (files, timestamps, models)
"""
import argparse
import csv
import glob
import json
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
except Exception as e:
    pd = None

# ----------------------- Helpers -----------------------

def coalesce(*vals, default=None):
    """Return first non-empty value in vals (not None / not '')."""
    for v in vals:
        if v is not None and v != "":
            return v
    return default

def as_str(x):
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)

def safe_float(x, default=None):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(x)
    except Exception:
        return default

def guess_run_id(path: Path) -> str:
    # Expect runs in garak_outputs/<timestamp>/filename.jsonl
    # If not, fall back to parent dir name or file stem.
    parts = path.parts
    if len(parts) >= 2:
        return path.parent.name
    return path.stem

# Common field name aliases we might encounter in Garak exports
FIELD_ALIASES = {
    "model": ["model", "model_name", "generator", "generator_name"],
    "probe": ["probe", "probe_name", "attack", "attack_name"],
    "detector": ["detector", "detector_name", "metric", "metric_name"],
    "score": ["score", "detector_score", "metric_score", "value"],
    "passed": ["passed", "pass", "is_pass", "success", "ok"],
    "prompt": ["prompt", "input", "attack_prompt", "question"],
    "response": ["response", "output", "answer", "model_response"],
    "case_id": ["case_id", "id", "example_id", "uid"],
    "timestamp": ["timestamp", "time", "created_at", "run_time"],
    "report_html": ["report_html", "html", "report_path_html"],
    "report_jsonl": ["report_jsonl", "jsonl", "report_path_jsonl"],
}

def first_field(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d:
            return d.get(k)
    return None

@dataclass
class NormalizedRow:
    run_id: str
    model: str
    probe: str
    case_id: str
    prompt: str
    response: str
    detector: str
    score: Optional[float]
    passed: Optional[bool]
    # Optional metadata
    timestamp: str = ""
    report_html: str = ""
    report_jsonl: str = ""
    file_path: str = ""

    @staticmethod
    def from_record(rec: Dict[str, Any], file_path: Path) -> List["NormalizedRow"]:
        rows = []
        if rec.get("entry_type") != "attempt":
            return []

        model = GLOBAL_CONTEXT.get("model") or "unknown"
        run_id = GLOBAL_CONTEXT.get("run_id") or guess_run_id(file_path)
        probe = rec.get("probe_classname", "unknown")

        prompt = rec.get("prompt", "")
        responses = rec.get("outputs", [])
        detector_scores = rec.get("detector_results", {})

        # Each detector returns a list of scores (1 per response)
        for i, response in enumerate(responses):
            for detector, scores in detector_scores.items():
                score = scores[i] if i < len(scores) else None
                passed = score == 0.0 if score is not None else None

                row = NormalizedRow(
                    run_id=run_id,
                    model=model,
                    probe=probe,
                    case_id=rec.get("uuid", "") + f"_{i}",
                    prompt=prompt,
                    response=response,
                    detector=detector,
                    score=score,
                    passed=passed,
                    timestamp="",  # not present per-attempt
                    report_html="",
                    report_jsonl=str(file_path),
                    file_path=str(file_path)
                )
                rows.append(row)

        return rows


# ----------------------- Track global context -----------------------

GLOBAL_CONTEXT = {
    "model": None,
    "probe": None,
    "run_id": None,
}



# ----------------------- Core Parser -----------------------

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"[WARN] {path}:{line_num} JSON decode error: {e}\n")
                continue

def collect_records(input_dir: Path) -> Tuple[List[NormalizedRow], Dict[str, Any]]:
    rows: List[NormalizedRow] = []
    run_meta: Dict[str, Any] = defaultdict(lambda: {"files": [], "models": set(), "probes": set()})
    jsonl_files = [p for p in input_dir.rglob("*") if p.name.endswith(".jsonl")]

    if not jsonl_files:
        sys.stderr.write(f"[WARN] No .jsonl files found under {input_dir}\n")

    for jf in jsonl_files:
        for rec in iter_jsonl(jf):

            # Update global context if setup/digest records
            if rec.get("entry_type") == "start_run setup":
                GLOBAL_CONTEXT["model"] = rec.get("plugins.model_name")
                GLOBAL_CONTEXT["run_id"] = rec.get("transient.run_id")
            elif rec.get("entry_type") == "digest":
                GLOBAL_CONTEXT["model"] = rec.get("model_name")

            normalized_rows = NormalizedRow.from_record(rec, jf)
            rows.extend(normalized_rows)

            # Add meta info if any rows were extracted
            for row in normalized_rows:
                meta = run_meta[row.run_id]
                meta["files"].append(str(jf))
                if row.model:
                    meta["models"].add(row.model)
                if row.probe:
                    meta["probes"].add(row.probe)

    # Convert sets to lists
    for k, v in run_meta.items():
        v["models"] = sorted(list(v["models"]))
        v["probes"] = sorted(list(v["probes"]))

    return rows, run_meta


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def to_dicts(rows: List[NormalizedRow]) -> List[Dict[str, Any]]:
    return [asdict(r) for r in rows]

def summarize_with_pandas(rows: List[NormalizedRow], out_dir: Path):
    if pd is None:
        sys.stderr.write("[WARN] pandas not installed; skipping summaries. Install pandas to enable.\n")
        return
    import pandas as _pd
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _pd.DataFrame([asdict(r) for r in rows])
    # Normalize missing booleans as False for aggregation where needed
    # but keep NaN too to see unknowns
    # Long-form normalized.csv
    df.to_csv(out_dir / "normalized.csv", index=False)

    # Model summary by probe
    grp_cols = ["run_id", "model", "probe"]
    agg = df.groupby(grp_cols).agg(
        total=("case_id", "count"),
        passes=("passed", lambda s: int(_pd.Series(s).fillna(False).astype(bool).sum())),
        fails=("passed", lambda s: int((~_pd.Series(s).fillna(False).astype(bool)).sum())),
        # average score per (run_id, model, probe, detector) doesn't make sense collapsed;
        # so we also compute by detector separately below.
    ).reset_index()
    agg["fail_rate"] = agg.apply(lambda r: (r["fails"] / r["total"]) if r["total"] else 0.0, axis=1)
    agg.sort_values(["run_id", "model", "probe"], inplace=True)
    agg.to_csv(out_dir / "model_summary.csv", index=False)

    # By detector within probe/model
    if "detector" in df.columns:
        det_grp_cols = ["run_id", "model", "probe", "detector"]
        det_agg = df.groupby(det_grp_cols).agg(
            total=("case_id", "count"),
            avg_score=("score", "mean"),
            passes=("passed", lambda s: int(_pd.Series(s).fillna(False).astype(bool).sum())),
            fails=("passed", lambda s: int((~_pd.Series(s).fillna(False).astype(bool)).sum())),
        ).reset_index()
        det_agg["fail_rate"] = det_agg.apply(lambda r: (r["fails"] / r["total"]) if r["total"] else 0.0, axis=1)
        det_agg.sort_values(det_grp_cols, inplace=True)
        det_agg.to_csv(out_dir / "detector_summary.csv", index=False)

    # Probe summary across models
    probe_grp = df.groupby(["probe"]).agg(
        total=("case_id", "count"),
        avg_score=("score", "mean"),
        pass_rate=("passed", lambda s: float(_pd.Series(s).fillna(False).astype(bool).mean())),
    ).reset_index()
    probe_grp.sort_values(["probe"], inplace=True)
    probe_grp.to_csv(out_dir / "probe_summary.csv", index=False)

    # Per-model compact JSON for frontend cards
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(exist_ok=True)
    for (run_id, model), sub in df.groupby(["run_id", "model"]):
        by_probe = sub.groupby("probe").agg(
            total=("case_id", "count"),
            pass_rate=("passed", lambda s: float(_pd.Series(s).fillna(False).astype(bool).mean()))
        ).reset_index()
        model_card = {
            "run_id": run_id,
            "model": model,
            "probes": [
                {"name": row["probe"], "total": int(row["total"]), "pass_rate": float(row["pass_rate"])}
                for _, row in by_probe.iterrows()
            ],
            "reports": {
                "jsonl_files": sorted({p for p in sub["file_path"].unique().tolist()}),
            },
        }
        with open(per_model_dir / f"{run_id}__{sanitize_filename(model)}.json", "w", encoding="utf-8") as f:
            json.dump(model_card, f, ensure_ascii=False, indent=2)

def sanitize_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)
    return safe[:200] if len(safe) > 200 else safe

def emit_run_meta(run_meta: Dict[str, Any], out_dir: Path):
    out = out_dir / "per_run"
    out.mkdir(parents=True, exist_ok=True)
    for run_id, meta in run_meta.items():
        payload = {"run_id": run_id, "files": meta["files"], "models": meta["models"], "probes": meta["probes"]}
        with open(out / f"{sanitize_filename(run_id)}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Directory that contains Garak .jsonl files (or run folders).")
    ap.add_argument("--out", type=str, required=True, help="Output directory for normalized CSV/JSON.")
    ap.add_argument("--no-pandas", action="store_true", help="Skip pandas-based summaries (only write normalized.csv via csv module).")
    args = ap.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, run_meta = collect_records(input_dir)

    # Always write normalized.csv via csv module for zero-deps path
    fieldnames = list(asdict(NormalizedRow("", "", "", "", "", "", "", None, None)).keys())
    fieldnames += ["timestamp", "report_html", "report_jsonl", "file_path"]
    # But our dataclass already includes those optional fields; ensure union
    fieldnames = list(dict.fromkeys(fieldnames))  # de-dupe preserving order

    write_csv(out_dir / "normalized.csv", fieldnames, (asdict(r) for r in rows))

    # Emit per-run metadata
    emit_run_meta(run_meta, out_dir)

    # Pandas summaries if available and not disabled
    if not args.no_pandas and pd is not None:
        summarize_with_pandas(rows, out_dir)
    elif not args.no_pandas and pd is None:
        sys.stderr.write("[INFO] Install pandas to enable aggregated summaries: pip install pandas\n")

    print(f"[OK] Parsed {len(rows)} records from {input_dir}. Outputs in {out_dir}")

if __name__ == "__main__":
    main()