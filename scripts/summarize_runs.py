"""Summarize all runs into a markdown table.

Usage: python scripts/summarize_runs.py [--output experiment_log.md]

Scans runs/ for session.json (completed runs) and project_state.json
(crashed runs), extracts key metrics, and outputs a markdown summary.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


def _extract_run_info(run_dir: Path) -> dict:
    """Extract metrics from a single run directory."""
    name = run_dir.name
    # Parse scenario name and timestamp from directory name
    parts = name.rsplit("_", 2)
    if len(parts) >= 3:
        scenario = name[: name.rfind("_", 0, name.rfind("_"))]
        timestamp = parts[-2] + "_" + parts[-1]
    else:
        scenario = name
        timestamp = ""

    info = {
        "dir": name,
        "scenario": scenario,
        "timestamp": timestamp,
        "stop_reason": "",
        "stages": 0,
        "stages_completed": 0,
        "tools": 0,
        "turns": 0,
        "escalations": 0,
        "model": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "has_session": False,
        "artifacts": [],
    }

    session_path = run_dir / "session.json"
    ps_path = run_dir / "project_state.json"

    if session_path.is_file():
        info["has_session"] = True
        s = json.loads(session_path.read_text())
        rr = s.get("run_result", {})
        ps = s.get("project_state", {})
        info["stop_reason"] = rr.get("stop_reason", "?")
        info["stages"] = len(ps.get("stages", {}))
        info["stages_completed"] = rr.get("stages_completed", 0)
        info["tools"] = rr.get("tool_call_count", 0)
        info["turns"] = rr.get("turn_count", 0)
        info["escalations"] = rr.get("escalation_count", 0)
        info["model"] = s.get("model", "")
        info["prompt_tokens"] = rr.get("prompt_tokens", 0)
        info["completion_tokens"] = rr.get("completion_tokens", 0)
    elif ps_path.is_file():
        # No session.json — distinguish "still running" from "crashed"
        import subprocess as _sp
        try:
            _sp.check_output(["pgrep", "-f", f"run_validation.*{scenario}"], text=True)
            info["stop_reason"] = "running"
        except _sp.CalledProcessError:
            info["stop_reason"] = "crashed"
        ps_data = json.loads(ps_path.read_text())
        info["stages"] = len(ps_data.get("stages", {}))

    # Count output artifacts
    stages_dir = run_dir / "stages"
    if stages_dir.is_dir():
        for f in stages_dir.rglob("outputs/*"):
            if f.is_file() and f.name != "manifest.json":
                info["artifacts"].append(str(f.relative_to(run_dir)))

    return info


def summarize() -> str:
    """Generate markdown summary of all runs."""
    if not RUNS_DIR.is_dir():
        return "No runs/ directory found."

    runs = sorted(RUNS_DIR.iterdir())
    infos = [_extract_run_info(r) for r in runs if r.is_dir()]

    # Group by scenario
    by_scenario: dict[str, list[dict]] = {}
    for info in infos:
        by_scenario.setdefault(info["scenario"], []).append(info)

    lines = ["# Experiment Run Log", "", f"Auto-generated from `runs/` — {len(infos)} total runs.", ""]

    # Summary table
    lines.append("## Summary by Scenario")
    lines.append("")
    lines.append("| Scenario | Runs | Completed | Best Run | Stages | Tools | Artifacts |")
    lines.append("|----------|------|-----------|----------|--------|-------|-----------|")

    for scenario, scenario_runs in sorted(by_scenario.items()):
        total = len(scenario_runs)
        completed = [r for r in scenario_runs if r["stop_reason"] == "completed"]
        # Best run: completed with most stages, or most tools as tiebreak
        all_with_session = [r for r in scenario_runs if r["has_session"]]
        if all_with_session:
            best = max(all_with_session, key=lambda r: (r["stages"], r["tools"]))
        else:
            best = max(scenario_runs, key=lambda r: r["stages"])

        lines.append(
            f"| {scenario} | {total} | {len(completed)} | "
            f"`{best['dir']}` | {best['stages']} | {best['tools']} | "
            f"{len(best['artifacts'])} |"
        )

    # Detailed table
    lines.append("")
    lines.append("## All Runs (chronological)")
    lines.append("")
    lines.append("| Run | Stop | Stages | Tools | Turns | Esc | Model | Tokens |")
    lines.append("|-----|------|--------|-------|-------|-----|-------|--------|")

    for info in infos:
        model_short = info["model"].split("/")[-1] if info["model"] else "?"
        tokens = f"{info['prompt_tokens']//1000}K+{info['completion_tokens']//1000}K" if info["prompt_tokens"] else "—"
        lines.append(
            f"| {info['dir']} | {info['stop_reason']} | "
            f"{info['stages']} | {info['tools']} | {info['turns']} | "
            f"{info['escalations']} | {model_short} | {tokens} |"
        )

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    md = summarize()
    output = PROJECT_ROOT / "experiment_log.md"
    if len(sys.argv) > 2 and sys.argv[1] == "--output":
        output = Path(sys.argv[2])
    output.write_text(md, encoding="utf-8")
    print(f"Written to {output}")
    print(md)
