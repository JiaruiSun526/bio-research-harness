"""Evaluate a workspace against an evaluation checklist and produce a scored report.

Usage:
    python scripts/evaluate_run.py \
        --workspace runs/phagecounting_20260410_002024 \
        --checklist evaluation_checklist.json \
        [--output evaluation_report.json] \
        [--model openrouter/openai/gpt-5.4]

Three check_type strategies:
    - pipeline:     file existence + non-empty + parseable (no LLM)
    - quantitative: file content → LLM extracts values → code compares tolerance
    - qualitative:  file content → LLM judges match against expected description
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_scenario import (
    PROJECT_ROOT,
    _build_client,
    _load_openrouter_config,
)

log = logging.getLogger(__name__)

MAX_LLM_RETRIES = 2
IMAGE_SIZE_LIMIT = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# LLM prompts (from scripts_for_assets/evaluate_checklist.py)
# ---------------------------------------------------------------------------

QUANTITATIVE_PROMPT = """\
You are evaluating whether a computational reproduction matches expected results.

## Check
- verify: {verify}
- expected: {expected}
- tolerance: {tolerance}

## Output file(s)
{file_previews}

Extract the relevant numerical value(s) from the output files and compare
against the expected value. Return JSON:
{{
  "verdict": "pass" | "partial" | "fail",
  "extracted_value": "<the value you found>",
  "reasoning": "<explain what you found and whether it matches>"
}}
- "pass": extracted value matches expected within tolerance
- "partial": value found but outside tolerance, OR correct direction/magnitude
- "fail": no matching value found, or completely wrong
"""

QUALITATIVE_PROMPT = """\
You are evaluating whether a computational reproduction matches expected results.

## Check
- verify: {verify}
- expected: {expected}

## Output file(s)
{file_previews}

Judge whether the output matches the expected description in kind, pattern,
or direction. Return JSON:
{{
  "verdict": "pass" | "partial" | "fail",
  "extracted_value": "<summary of what you found>",
  "reasoning": "<explain whether and how the output matches>"
}}
- "pass": output clearly matches the expected description
- "partial": output partially matches or is in the right direction
- "fail": output does not match or is missing
"""


# ---------------------------------------------------------------------------
# File discovery and reading
# ---------------------------------------------------------------------------


DISCOVERY_PROMPT = """\
You are matching evaluation checklist items to workspace artifacts.

## Workspace Manifest
{manifest}

## Checklist Item
- ID: {item_id}
- Verify: {verify}
- Expected: {expected}
- Output pattern hint: {pattern}

Find the workspace file(s) most relevant to this checklist item.
Return JSON:
{{
  "matched_files": ["path/to/file1.csv", "path/to/file2.png"],
  "reasoning": "brief explanation of why these files match"
}}
Return {{"matched_files": [], "reasoning": "..."}} if no relevant files exist.
Only return paths that appear in the manifest above.
"""


def build_workspace_manifest(workspace: Path) -> str:
    """Build a manifest of all workspace artifacts for LLM discovery.

    Combines stage manifest.json descriptions with conclusion summaries.
    """
    lines: list[str] = []

    # Stage conclusions
    for conclusion_path in sorted(workspace.rglob("conclusion.md")):
        stage_id = conclusion_path.parent.name
        title = conclusion_path.read_text(encoding="utf-8").split("\n")[0]
        lines.append(f"[conclusion] stages/{stage_id}/conclusion.md — {title}")

    # Artifact manifests
    for manifest_path in sorted(workspace.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for artifact_path, description in manifest.items():
            # Truncate long descriptions
            desc = description[:150] + "..." if len(description) > 150 else description
            lines.append(f"[artifact] {artifact_path} — {desc}")

    # Also list any files not in manifests (catch orphans from run_code)
    manifest_paths = set()
    for manifest_path in workspace.rglob("manifest.json"):
        try:
            manifest_paths.update(json.loads(manifest_path.read_text()).keys())
        except Exception:
            pass

    for f in sorted(workspace.rglob("*")):
        if not f.is_file():
            continue
        rel = str(f.relative_to(workspace))
        if rel in manifest_paths or f.name in ("manifest.json", "project_state.json",
                                                 "session.json", "data_catalog.json"):
            continue
        if "/outputs/" in rel and f.suffix in (".csv", ".json", ".png", ".md", ".txt"):
            lines.append(f"[file] {rel} ({f.stat().st_size} bytes)")

    return "\n".join(lines) if lines else "(empty workspace)"


def discover_files_for_item(
    client: OpenAI, model: str, item: dict[str, Any],
    workspace: Path, manifest: str,
) -> list[Path]:
    """Use LLM to find workspace files matching a checklist item."""
    prompt = DISCOVERY_PROMPT.format(
        manifest=manifest,
        item_id=item.get("id", "?"),
        verify=item.get("verify", ""),
        expected=item.get("expected", ""),
        pattern=item.get("output_pattern", ""),
    )
    result = _call_llm(client, model, "You are a file-matching assistant. Return only JSON.", prompt)
    matched_paths: list[Path] = []
    for rel_path in result.get("matched_files", []):
        full = workspace / rel_path
        if full.is_file():
            matched_paths.append(full)
        else:
            # Try without leading "stages/" etc
            for candidate in workspace.rglob(Path(rel_path).name):
                if candidate.is_file():
                    matched_paths.append(candidate)
                    break
    return matched_paths


def read_file_preview(path: Path, max_chars: int = 8000) -> str | list[dict[str, Any]]:
    """Read a file and return text preview or multimodal content for LLM."""
    suffix = path.suffix.lower()

    if suffix in (".png", ".jpg", ".jpeg", ".tiff", ".gif", ".bmp", ".webp"):
        if path.stat().st_size > IMAGE_SIZE_LIMIT:
            return f"[Image {path.name}: too large]"
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".tiff": "image/tiff", ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp"}
        b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        return [
            {"type": "text", "text": f"File: {path.name}"},
            {"type": "image_url", "image_url": {"url": f"data:{mime_map.get(suffix, 'image/png')};base64,{b64}"}},
        ]

    if suffix == ".pdf":
        if path.stat().st_size > IMAGE_SIZE_LIMIT:
            return f"[PDF {path.name}: too large]"
        b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        return [{"type": "file", "file": {"filename": path.name, "file_data": f"data:application/pdf;base64,{b64}"}}]

    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            text = path.read_text(errors="replace")
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            rows = list(reader)
            header = f"File: {path.name} ({len(rows)} rows × {len(rows[0]) if rows else 0} columns)\n"
            preview_rows = rows[:20] + ([["..."]] + rows[-5:] if len(rows) > 25 else [])
            preview = header + "\n".join(delimiter.join(r) for r in preview_rows)
            return preview[:max_chars] + (f"\n[truncated]" if len(preview) > max_chars else "")
        except Exception as exc:
            return f"File: {path.name} [parse error: {exc}]"

    if suffix == ".json":
        try:
            parsed = json.loads(path.read_text(errors="replace"))
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
            return f"File: {path.name}\n{formatted[:max_chars]}"
        except Exception:
            pass

    try:
        text = path.read_text(errors="replace")
        return f"File: {path.name}\n{text[:max_chars]}"
    except Exception:
        return f"File: {path.name} [binary, {path.stat().st_size} bytes]"


# ---------------------------------------------------------------------------
# Tolerance parsing
# ---------------------------------------------------------------------------


def parse_tolerance(tolerance_str: str | None) -> tuple[float, bool] | None:
    """Parse '±0.05' or '±5%'. Returns (value, is_percentage) or None."""
    if not tolerance_str or tolerance_str.strip().lower() in ("null", "none", ""):
        return None
    match = re.search(r"[±+\-]?\s*(\d+\.?\d*)\s*(%)?", tolerance_str)
    if not match:
        return None
    all_matches = re.findall(r"[±]\s*\d+\.?\d*\s*%?", tolerance_str)
    if len(all_matches) > 1:
        return None
    return (float(match.group(1)), match.group(2) is not None)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_pipeline(item: dict[str, Any], matched_files: list[Path]) -> dict[str, Any]:
    """Pipeline check: file existence + non-empty + parseable (no LLM)."""
    pattern = item.get("output_pattern", "")
    if not matched_files:
        return {"verdict": "fail", "matched_files": [], "extracted_value": None,
                "reasoning": f"No files matched: {pattern}", "score": 0.0}

    rel = [f.name for f in matched_files]
    empty = [f for f in matched_files if f.stat().st_size == 0]
    if len(empty) == len(matched_files):
        return {"verdict": "fail", "matched_files": rel, "extracted_value": None,
                "reasoning": f"All files empty: {rel}", "score": 0.0}

    details: list[str] = []
    ok = False
    for f in matched_files:
        if f.stat().st_size == 0:
            continue
        suffix = f.suffix.lower()
        try:
            if suffix in (".csv", ".tsv"):
                rows = list(csv.reader(io.StringIO(f.read_text(errors="replace")),
                                       delimiter="\t" if suffix == ".tsv" else ","))
                details.append(f"{f.name}: {len(rows)}×{len(rows[0]) if rows else 0}")
                ok = True
            elif suffix == ".json":
                data = json.loads(f.read_text(errors="replace"))
                details.append(f"{f.name}: JSON {type(data).__name__}")
                ok = True
            elif suffix in (".png", ".jpg", ".jpeg", ".pdf", ".svg"):
                details.append(f"{f.name}: {suffix} {f.stat().st_size}B")
                ok = True
            else:
                n = f.read_text(errors="strict").count("\n")
                details.append(f"{f.name}: {n} lines")
                ok = True
        except Exception as exc:
            details.append(f"{f.name}: error {exc}")

    verdict = "pass" if ok else "partial"
    return {"verdict": verdict, "matched_files": rel, "extracted_value": None,
            "reasoning": "; ".join(details), "score": 1.0 if ok else 0.5}


def _call_llm(client: OpenAI, model: str, system: str,
              user_content: str | list[dict[str, Any]]) -> dict[str, Any]:
    """Call LLM with retry, parse JSON response."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=2048, temperature=0.1)
            content = resp.choices[0].message.content or ""
            return _parse_llm_json(content)
        except Exception as exc:
            if attempt < MAX_LLM_RETRIES:
                time.sleep(2 ** (attempt + 1))
                log.warning("LLM attempt %d failed: %s", attempt + 1, exc)
            else:
                return {"verdict": "fail", "extracted_value": None,
                        "reasoning": f"LLM failed: {exc}"}
    return {"verdict": "fail", "extracted_value": None, "reasoning": "unreachable"}


def _parse_llm_json(content: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling code fences."""
    s = content.strip()
    for attempt_str in [s, re.sub(r"^```\w*\n?", "", re.sub(r"\n?```$", "", s))]:
        try:
            return json.loads(attempt_str)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"verdict": "fail", "extracted_value": None, "reasoning": f"Unparseable: {s[:200]}"}


def _build_file_previews(matched_files: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    """Build text and multimodal previews from matched files."""
    text_parts: list[str] = []
    mm_parts: list[dict[str, Any]] = []
    has_mm = False
    for f in matched_files:
        preview = read_file_preview(f)
        if isinstance(preview, list):
            has_mm = True
            mm_parts.extend(preview)
        else:
            text_parts.append(preview)
            mm_parts.append({"type": "text", "text": preview})
    return "\n\n---\n\n".join(text_parts) or "(no text)", mm_parts if has_mm else []


def check_quantitative(client: OpenAI, model: str, item: dict[str, Any],
                       matched_files: list[Path]) -> dict[str, Any]:
    """Quantitative check: LLM extracts values, compares to expected."""
    text_preview, mm_parts = _build_file_previews(matched_files)
    prompt = QUANTITATIVE_PROMPT.format(
        verify=item.get("verify", ""), expected=item.get("expected", ""),
        tolerance=item.get("tolerance", "N/A"), file_previews=text_preview)
    user_content: Any = mm_parts + [{"type": "text", "text": prompt}] if mm_parts else prompt
    result = _call_llm(client, model, "You are an evaluation judge. Return only JSON.", user_content)
    verdict = result.get("verdict", "fail")
    if verdict not in ("pass", "partial", "fail"):
        verdict = "fail"
    return {"verdict": verdict, "matched_files": [f.name for f in matched_files],
            "extracted_value": result.get("extracted_value"),
            "reasoning": result.get("reasoning", ""),
            "score": {"pass": 1.0, "partial": 0.5, "fail": 0.0}[verdict]}


def check_qualitative(client: OpenAI, model: str, item: dict[str, Any],
                      matched_files: list[Path]) -> dict[str, Any]:
    """Qualitative check: LLM judges pattern/direction match."""
    text_preview, mm_parts = _build_file_previews(matched_files)
    prompt = QUALITATIVE_PROMPT.format(
        verify=item.get("verify", ""), expected=item.get("expected", ""),
        file_previews=text_preview)
    user_content: Any = mm_parts + [{"type": "text", "text": prompt}] if mm_parts else prompt
    result = _call_llm(client, model, "You are an evaluation judge. Return only JSON.", user_content)
    verdict = result.get("verdict", "fail")
    if verdict not in ("pass", "partial", "fail"):
        verdict = "fail"
    return {"verdict": verdict, "matched_files": [f.name for f in matched_files],
            "extracted_value": result.get("extracted_value"),
            "reasoning": result.get("reasoning", ""),
            "score": {"pass": 1.0, "partial": 0.5, "fail": 0.0}[verdict]}


# ---------------------------------------------------------------------------
# Item-level and checklist-level evaluation
# ---------------------------------------------------------------------------


def evaluate_item(client: OpenAI, model: str, item: dict[str, Any],
                  workspace: Path, manifest: str) -> dict[str, Any]:
    """Evaluate a single checklist item using LLM-based file discovery."""
    item_id = item.get("id", "?")
    check_type = item.get("check_type", "pipeline")

    log.info("Evaluating %s [%s] verify=%s", item_id, check_type,
             item.get("verify", "")[:60])

    # LLM-based discovery: find relevant files semantically
    matched = discover_files_for_item(client, model, item, workspace, manifest)
    log.info("  Discovered %d files for %s", len(matched), item_id)

    if check_type == "pipeline":
        return check_pipeline(item, matched)
    if not matched:
        return {"verdict": "fail", "matched_files": [], "extracted_value": None,
                "reasoning": f"No relevant files found in workspace for: {item.get('verify', '')[:100]}",
                "score": 0.0}
    if check_type == "quantitative":
        return check_quantitative(client, model, item, matched)
    if check_type == "qualitative":
        return check_qualitative(client, model, item, matched)
    return check_pipeline(item, matched)


def aggregate_scores(proof_flows: list[dict[str, Any]],
                     scoring: dict[str, Any]) -> dict[str, Any]:
    """Compute weighted aggregate scores."""
    weights = scoring.get("weights", {"critical": 3, "important": 2, "supplementary": 1})
    total_w, total_max = 0.0, 0.0
    counts = {"pass": 0, "partial": 0, "fail": 0}
    by_priority: dict[str, dict[str, Any]] = {}

    for pf in proof_flows:
        for item in pf.get("items", []):
            p = item.get("priority", "supplementary")
            v = item.get("verdict", "fail")
            s = item.get("score", 0.0)
            w = weights.get(p, 1)
            total_w += s * w
            total_max += w
            counts[v] = counts.get(v, 0) + 1
            if p not in by_priority:
                by_priority[p] = {"sum": 0.0, "n": 0, "pass": 0, "partial": 0, "fail": 0}
            by_priority[p]["sum"] += s
            by_priority[p]["n"] += 1
            by_priority[p][v] = by_priority[p].get(v, 0) + 1

    return {
        "total_score": round(total_w / total_max, 4) if total_max else 0.0,
        "items_total": sum(counts.values()),
        **{f"items_{k}": v for k, v in counts.items()},
        "by_priority": {p: {"score": round(d["sum"] / d["n"], 4) if d["n"] else 0,
                            **{k: d[k] for k in ("pass", "partial", "fail")}}
                        for p, d in by_priority.items()},
    }


def evaluate_checklist(checklist: dict[str, Any], workspace: Path,
                       client: OpenAI, model: str) -> dict[str, Any]:
    """Evaluate all items in the checklist. Returns full report."""
    scoring = checklist.get("scoring", {
        "weights": {"critical": 3, "important": 2, "supplementary": 1}})
    weights = scoring.get("weights", {"critical": 3, "important": 2, "supplementary": 1})

    # Build workspace manifest once for all items
    manifest = build_workspace_manifest(workspace)
    log.info("Workspace manifest: %d lines", manifest.count("\n") + 1)

    evaluated: list[dict[str, Any]] = []
    for pf in checklist.get("proof_flows", []):
        pf_items: list[dict[str, Any]] = []
        pf_wsum, pf_wtot = 0.0, 0.0
        for item in pf.get("items", []):
            result = evaluate_item(client, model, item, workspace, manifest)
            w = weights.get(item.get("priority", "supplementary"), 1)
            pf_wsum += result["score"] * w
            pf_wtot += w
            pf_items.append({"id": item.get("id"), "priority": item.get("priority"),
                             "check_type": item.get("check_type"),
                             "verify": item.get("verify"), "expected": item.get("expected"),
                             **result})
        evaluated.append({"id": pf.get("id"), "name": pf.get("name"),
                         "score": round(pf_wsum / pf_wtot, 4) if pf_wtot else 0.0,
                         "items": pf_items})

    return {"paper": checklist.get("paper", {}),
            "summary": aggregate_scores(evaluated, scoring),
            "proof_flows": evaluated}


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------


def print_summary(report: dict[str, Any]) -> None:
    """Print a compact evaluation summary to stderr."""
    s = report["summary"]
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"EVALUATION REPORT — {report.get('paper', {}).get('title', 'unknown')}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Overall score: {s['total_score']:.1%} ({s['items_pass']}P / "
          f"{s['items_partial']}A / {s['items_fail']}F out of {s['items_total']})", file=sys.stderr)
    for p, d in s.get("by_priority", {}).items():
        print(f"  {p}: {d.get('score', 0):.1%} ({d.get('pass',0)}P/{d.get('partial',0)}A/{d.get('fail',0)}F)",
              file=sys.stderr)
    print(f"\nPer proof flow:", file=sys.stderr)
    for pf in report.get("proof_flows", []):
        n = len(pf.get("items", []))
        print(f"  {pf['id']} ({pf['name'][:50]}): {pf['score']:.1%} [{n} items]", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a workspace against an evaluation checklist")
    parser.add_argument("--workspace", type=Path, required=True, help="Workspace directory to evaluate")
    parser.add_argument("--checklist", type=Path, required=True, help="evaluation_checklist.json")
    parser.add_argument("--output", type=Path, default=None, help="Output report JSON (default: workspace/evaluation_report.json)")
    parser.add_argument("--model", type=str, default=None, help="LLM model for quantitative/qualitative checks")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stderr)

    if not args.workspace.is_dir():
        log.error("Workspace not found: %s", args.workspace)
        sys.exit(1)
    if not args.checklist.is_file():
        log.error("Checklist not found: %s", args.checklist)
        sys.exit(1)

    config = _load_openrouter_config()
    model = args.model or config["default_model"]
    if model.startswith("openrouter/"):
        model = model[len("openrouter/"):]
    client = _build_client(config["api_key"], config["proxy"])

    checklist = json.loads(args.checklist.read_text(encoding="utf-8"))
    log.info("Loaded checklist: %d proof flows, %d items",
             len(checklist.get("proof_flows", [])),
             sum(len(pf.get("items", [])) for pf in checklist.get("proof_flows", [])))

    report = evaluate_checklist(checklist, args.workspace.resolve(), client, model)

    output_path = args.output or (args.workspace / "evaluation_report.json")
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Report written to: %s", output_path)

    print_summary(report)


if __name__ == "__main__":
    main()
