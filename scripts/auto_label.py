"""Auto-label stages and escalations using LLM classification.

Reads metrics/stages.csv and metrics/escalations.csv, calls an LLM to
classify each row, writes labeled versions to metrics/stages_labeled.csv
and metrics/escalations_labeled.csv.

Stage axes:
  stage_type: data_prep | core_analysis | evaluation | synthesis
  paradigm:   mechanistic | predictive | statistical | descriptive | none
Escalation types: plan_review | result_review | direction | knowledge | correction

Usage: python scripts/auto_label.py [--model openrouter/openai/gpt-5.4]
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from prepare_scenario import _build_client, _load_openrouter_config

SKIP_SCENARIOS = {"live", "live_scenario", "phagecounting_full"}

# ── Stage labeling ──

STAGE_LABEL_PROMPT = """\
Classify this research stage on two orthogonal axes.

Axis 1 — workflow role (WHERE in the pipeline):
- data_prep: data loading, cleaning, formatting, QC, matrix assembly, cohort construction
- core_analysis: main statistical/modeling/fitting work (DE, classification, ODE fitting, parameter estimation)
- evaluation: validation, benchmarking, cross-validation, model comparison, external test
- synthesis: final summary, visualization compilation, artifact check, report generation

Axis 2 — computational paradigm (WHAT KIND of computation):
- mechanistic: ODE/PDE/dynamical-system modeling, parameter fitting to mechanistic equations, reaction networks, bifurcation/Hopf analysis, Turing patterns, reservoir computing dynamics
- predictive: supervised ML (classification, regression), cross-validation, feature selection for models, SHAP, hyperparameter tuning
- statistical: hypothesis testing (t-test, chi-square, PERMANOVA), differential expression/abundance, confidence intervals, power analysis, pure statistical inference
- descriptive: PCA/UMAP/ordination, clustering, network topology, enrichment analysis, visualization-centric analysis, longitudinal trend tracking
- none: pure data ingestion/cleaning/QC without analysis or modeling (use for data_prep stages that do not perform statistical/ML/mechanistic work)

Stage ID: {stage_id}
Conclusion title: {conclusion_title}

Return ONLY a JSON object with exactly these two fields, no other text:
{{"stage_type": "<one of 4 stage_type labels>", "paradigm": "<one of 5 paradigm labels>"}}"""

# ── Escalation labeling ──

ESCALATION_LABEL_PROMPT = """\
Classify this user-agent escalation exchange into the PRIMARY type.

Types:
- plan_review: user reviews a proposed plan and approves or requests changes
- result_review: user reviews completed stage results, confirms quality or flags issues
- direction: user decides what to do next (which goal, what stage, change course)
- knowledge: user injects domain-specific knowledge (parameters, thresholds, biological facts, equations)
- correction: user points out an error the agent made (wrong data, wrong method, bug)

Agent message (truncated): {agent_summary}

User response (truncated): {user_response}

Return ONLY the type label, nothing else."""

VALID_STAGE_TYPES = {"data_prep", "core_analysis", "evaluation", "synthesis"}
VALID_PARADIGMS = {"mechanistic", "predictive", "statistical", "descriptive", "none"}
VALID_ESC_TYPES = {"plan_review", "result_review", "direction", "knowledge", "correction"}


def classify(client: OpenAI, model: str, prompt: str, valid: set[str]) -> str:
    """Call LLM to classify, with retry and validation."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a classifier. Return only the label."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=20,
                temperature=0.0,
            )
            label = resp.choices[0].message.content.strip().lower().strip(".")
            if label in valid:
                return label
            # Try to extract from longer response
            for v in valid:
                if v in label:
                    return v
            print(f"  WARNING: invalid label '{label}', retrying", file=sys.stderr)
        except Exception as exc:
            print(f"  ERROR: {exc}, retrying in {2 ** attempt}s", file=sys.stderr)
            time.sleep(2 ** attempt)
    return "unknown"


def classify_stage(client: OpenAI, model: str, prompt: str) -> tuple[str, str]:
    """Classify a stage on both axes (stage_type + paradigm) in one LLM call.

    Returns (stage_type, paradigm). Falls back to ("unknown", "unknown") after 3 retries.
    """
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a classifier. Return ONLY a JSON object with fields stage_type and paradigm."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=80,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.strip("`").lstrip("json").strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Try to extract JSON substring
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    parsed = json.loads(raw[start:end + 1])
                else:
                    raise
            stage_type = str(parsed.get("stage_type", "")).strip().lower()
            paradigm = str(parsed.get("paradigm", "")).strip().lower()
            if stage_type in VALID_STAGE_TYPES and paradigm in VALID_PARADIGMS:
                return stage_type, paradigm
            print(f"  WARNING: invalid labels stage_type='{stage_type}' paradigm='{paradigm}', retrying", file=sys.stderr)
        except Exception as exc:
            print(f"  ERROR: {exc}, retrying in {2 ** attempt}s", file=sys.stderr)
            time.sleep(2 ** attempt)
    return "unknown", "unknown"


def label_stages(client: OpenAI, model: str) -> None:
    """Label all stages (both axes) and write stages_labeled.csv."""
    in_path = PROJECT_ROOT / "metrics" / "stages.csv"
    out_path = PROJECT_ROOT / "metrics" / "stages_labeled.csv"

    with open(in_path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["scenario"] not in SKIP_SCENARIOS]

    print(f"Labeling {len(rows)} stages on two axes (stage_type + paradigm)...")
    for i, row in enumerate(rows):
        if not row["conclusion_title"]:
            row["stage_type"] = "unknown"
            row["paradigm"] = "unknown"
            continue
        prompt = STAGE_LABEL_PROMPT.format(
            stage_id=row["stage_id"],
            conclusion_title=row["conclusion_title"],
        )
        stage_type, paradigm = classify_stage(client, model, prompt)
        row["stage_type"] = stage_type
        row["paradigm"] = paradigm
        print(f"  [{i+1}/{len(rows)}] {row['scenario']}/{row['stage_id']}: role={stage_type}, paradigm={paradigm}")

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


def label_escalations(client: OpenAI, model: str) -> None:
    """Label all escalations and write escalations_labeled.csv."""
    in_path = PROJECT_ROOT / "metrics" / "escalations.csv"
    out_path = PROJECT_ROOT / "metrics" / "escalations_labeled.csv"

    with open(in_path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["scenario"] not in SKIP_SCENARIOS]

    print(f"Labeling {len(rows)} escalations...")
    for i, row in enumerate(rows):
        prompt = ESCALATION_LABEL_PROMPT.format(
            agent_summary=row["agent_summary"][:300],
            user_response=row["user_response"][:300],
        )
        row["escalation_type"] = classify(client, model, prompt, VALID_ESC_TYPES)
        print(f"  [{i+1}/{len(rows)}] {row['scenario']} esc#{row['escalation_idx']}: {row['escalation_type']}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config = _load_openrouter_config()
    model = args.model or config["default_model"]
    if model.startswith("openrouter/"):
        model = model[len("openrouter/"):]
    client = _build_client(config["api_key"], config["proxy"])

    # First re-extract metrics to get latest data
    print("Re-extracting metrics...")
    os.system(f"{sys.executable} scripts/extract_metrics.py")
    print()

    label_stages(client, model)
    print()
    label_escalations(client, model)

    # Print summary
    print("\n=== Stage type (role) distribution ===")
    with open(PROJECT_ROOT / "metrics" / "stages_labeled.csv") as f:
        rows = list(csv.DictReader(f))
        types = [r["stage_type"] for r in rows]
    for t in sorted(set(types)):
        print(f"  {t}: {types.count(t)}")

    print("\n=== Stage paradigm distribution ===")
    paradigms = [r.get("paradigm", "unknown") for r in rows]
    for p in sorted(set(paradigms)):
        print(f"  {p}: {paradigms.count(p)}")

    print("\n=== Escalation type distribution ===")
    with open(PROJECT_ROOT / "metrics" / "escalations_labeled.csv") as f:
        types = [r["escalation_type"] for r in csv.DictReader(f)]
    for t in sorted(set(types)):
        print(f"  {t}: {types.count(t)}")


if __name__ == "__main__":
    main()
