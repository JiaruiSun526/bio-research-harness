"""Extract structured metrics from session.json files for thesis analysis.

Scans runs/ for the latest successful run per scenario and extracts:
  - metrics/run_overview.csv     — one row per run
  - metrics/stages.csv           — one row per stage (with type column for manual annotation)
  - metrics/escalations.csv      — one row per escalation (with type column for annotation)
  - metrics/failed_tasks.csv     — one row per failed subagent task (with failure_mode column)

Usage: python scripts/extract_metrics.py [--runs-dir runs/]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_latest_runs(runs_dir: Path) -> dict[str, Path]:
    """Find the latest run directory per scenario name."""
    runs: dict[str, Path] = {}
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if not d.is_dir() or not (d / "session.json").is_file():
            continue
        # Scenario name = dir name minus the timestamp suffix
        parts = d.name.rsplit("_", 2)
        if len(parts) >= 3:
            name = "_".join(parts[:-2])
        else:
            name = d.name
        if name not in runs:
            runs[name] = d
    return runs


_AGGREGATED_COUNTER_KEYS = (
    "turn_count",
    "tool_call_count",
    "escalation_count",
    "prompt_tokens",
    "completion_tokens",
)


def aggregate_resume_history(session: dict) -> dict:
    """Sum per-segment counters across a resume chain.

    Post-fix sessions store per-segment counters in ``resume_history``. Each
    segment corresponds to one ``run()`` or ``resume()`` invocation. Summing
    them gives the total work for the scenario regardless of how many times
    execution was interrupted and resumed.

    Pre-fix sessions have no ``resume_history``; in that case we fall back to
    the final ``run_result`` (which only reflects the LAST segment — the best
    we can recover from those older artifacts).

    Returns a dict with all keys from ``_AGGREGATED_COUNTER_KEYS`` plus
    ``segment_count`` (number of segments; 1 for fresh, N+1 for N resumes,
    0 is not expected but returned as 0 for safety).
    """

    history = session.get("resume_history") or []
    if history:
        totals: dict[str, int] = {k: 0 for k in _AGGREGATED_COUNTER_KEYS}
        for segment in history:
            for key in _AGGREGATED_COUNTER_KEYS:
                totals[key] += int(segment.get(key, 0) or 0)
        totals["segment_count"] = len(history)
        return totals

    fallback = session.get("run_result") or {}
    totals = {k: int(fallback.get(k, 0) or 0) for k in _AGGREGATED_COUNTER_KEYS}
    totals["segment_count"] = 1 if fallback else 0
    return totals


def extract_run_overview(session: dict, run_dir: Path) -> dict:
    """Extract one-row summary for a run."""
    rr = session.get("run_result", {})
    ps = session.get("project_state", {})
    stages = ps.get("stages", {})
    conv = session.get("main_conversation", [])
    subs = session.get("subagent_runs", [])
    agg = aggregate_resume_history(session)

    n_stages = len(stages)
    n_concluded = sum(1 for s in stages.values() if s.get("has_conclusion"))
    # turn count: prefer aggregated (covers all segments); fall back to counting
    # assistant messages in the (possibly latest-segment-only) conversation.
    n_turns = agg["turn_count"] or sum(1 for m in conv if m.get("role") == "assistant")
    # Count artifacts from disk (more reliable than session.json artifact_paths
    # which may be empty for runs using older harness versions).
    n_artifacts = 0
    n_pngs = 0
    n_csvs = 0
    stages_dir = run_dir / "stages"
    if stages_dir.is_dir():
        for f in stages_dir.rglob("*"):
            if f.is_file() and "outputs" in str(f.relative_to(stages_dir)):
                if f.name in ("manifest.json",):
                    continue
                n_artifacts += 1
                if f.suffix == ".png":
                    n_pngs += 1
                elif f.suffix == ".csv":
                    n_csvs += 1

    # Check for FINISH_SIGNAL
    has_finish = any(
        "FINISH_SIGNAL" in m.get("content", "")
        for m in conv if m.get("role") == "tool"
    )

    return {
        "scenario": run_dir.name.rsplit("_", 2)[0] if len(run_dir.name.rsplit("_", 2)) >= 3 else run_dir.name,
        "run_dir": run_dir.name,
        "stop_reason": rr.get("stop_reason", ""),
        "finish_verified": has_finish,
        "segment_count": agg["segment_count"],
        "stages": n_stages,
        "stages_concluded": n_concluded,
        "turns": n_turns,
        "escalations": agg["escalation_count"],
        "subagent_tasks": len(subs),
        "artifacts_total": n_artifacts,
        "artifacts_png": n_pngs,
        "artifacts_csv": n_csvs,
        "prompt_tokens": agg["prompt_tokens"],
        "completion_tokens": agg["completion_tokens"],
        "model": session.get("model", ""),
    }


def extract_stages(session: dict, scenario: str, run_dir: Path) -> list[dict]:
    """Extract one row per stage."""
    ps = session.get("project_state", {})
    stages = ps.get("stages", {})
    subs = session.get("subagent_runs", [])

    rows = []
    for stage_id, stage_data in stages.items():
        # Count tasks and artifacts for this stage
        stage_tasks = [s for s in subs if s.get("stage_id") == stage_id]
        n_tasks = len(stage_tasks)
        n_success = sum(1 for t in stage_tasks if t.get("task_result", {}).get("status") == "success")
        n_fail = sum(1 for t in stage_tasks if t.get("task_result", {}).get("status") != "success")
        n_artifacts = sum(
            len(t.get("task_result", {}).get("artifact_paths", []))
            for t in stage_tasks
        )

        # Read conclusion title if exists
        conclusion_path = run_dir / "stages" / stage_id / "conclusion.md"
        conclusion_title = ""
        if conclusion_path.is_file():
            conclusion_title = conclusion_path.read_text(encoding="utf-8").split("\n")[0].strip("# ").strip()

        rows.append({
            "scenario": scenario,
            "stage_id": stage_id,
            "stage_order": list(stages.keys()).index(stage_id) + 1,
            "has_conclusion": stage_data.get("has_conclusion", False),
            "conclusion_title": conclusion_title,
            "tasks_total": n_tasks,
            "tasks_success": n_success,
            "tasks_fail": n_fail,
            "artifacts": n_artifacts,
            "stage_type": "",  # For manual annotation
        })
    return rows


def extract_escalations(session: dict, scenario: str) -> list[dict]:
    """Extract one row per escalation (escalate_to_user call + response)."""
    conv = session.get("main_conversation", [])
    rows = []
    esc_idx = 0

    for i, msg in enumerate(conv):
        for tc in msg.get("tool_calls", []):
            if tc.get("name") != "escalate_to_user":
                continue
            args = tc.get("arguments", {})
            tc_id = tc.get("id", "")

            # Find matching tool result
            user_response = ""
            for j in range(i + 1, min(i + 10, len(conv))):
                if conv[j].get("role") == "tool" and conv[j].get("tool_call_id") == tc_id:
                    user_response = conv[j].get("content", "")
                    break

            esc_idx += 1
            rows.append({
                "scenario": scenario,
                "escalation_idx": esc_idx,
                "stage_id": args.get("stage_id", ""),
                "agent_summary": args.get("summary", ""),
                "user_response": user_response,
                "has_finish_signal": "FINISH_SIGNAL" in user_response,
                "escalation_type": "",  # For manual annotation
            })
    return rows


def extract_failed_tasks(session: dict, scenario: str) -> list[dict]:
    """Extract one row per failed subagent task."""
    subs = session.get("subagent_runs", [])
    rows = []
    for sub in subs:
        tr = sub.get("task_result", {})
        if tr.get("status") == "success":
            continue
        rows.append({
            "scenario": scenario,
            "stage_id": sub.get("stage_id", ""),
            "task_id": tr.get("task_id", ""),
            "status": tr.get("status", ""),
            "summary": tr.get("summary", "")[:200],
            "error": (tr.get("error") or "")[:300],
            "blockers": "; ".join(tr.get("blockers", [])),
            "failure_mode": "",  # For manual annotation
        })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write rows to CSV, creating parent dirs if needed."""
    if not rows:
        print(f"  (no data for {path.name})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name}: {len(rows)} rows")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "metrics"

    runs = find_latest_runs(args.runs_dir)
    print(f"Found {len(runs)} scenarios in {args.runs_dir}")
    for name, d in sorted(runs.items()):
        print(f"  {name}: {d.name}")

    all_overview: list[dict] = []
    all_stages: list[dict] = []
    all_escalations: list[dict] = []
    all_failures: list[dict] = []

    for scenario, run_dir in sorted(runs.items()):
        session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))

        all_overview.append(extract_run_overview(session, run_dir))
        all_stages.extend(extract_stages(session, scenario, run_dir))
        all_escalations.extend(extract_escalations(session, scenario))
        all_failures.extend(extract_failed_tasks(session, scenario))

    print(f"\nWriting to {out_dir}/")
    write_csv(out_dir / "run_overview.csv", all_overview)
    write_csv(out_dir / "stages.csv", all_stages)
    write_csv(out_dir / "escalations.csv", all_escalations)
    write_csv(out_dir / "failed_tasks.csv", all_failures)
    print("Done.")


if __name__ == "__main__":
    main()
