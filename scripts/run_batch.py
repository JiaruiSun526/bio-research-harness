"""Batch runner: parallel workers, retry on failure, log everything.

Usage: nohup python scripts/run_batch.py > batch_run.log 2>&1 &

Scenarios are split into N_WORKERS groups for parallel execution.
Each worker runs its group serially with up to MAX_ATTEMPTS retries.
"""

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    import tomli
except ModuleNotFoundError:
    import tomllib as tomli  # type: ignore[no-redef]

# ---- Configuration ----
MAX_ATTEMPTS = 8
N_WORKERS = 5
PROVIDER = "openrouter"
USER_PROVIDER = "mimo"

# 5 parallel workers, 2 scenarios each. Balanced by estimated runtime.
WORKER_GROUPS: list[list[dict]] = [
    [  # W1: complex + small
        {"toml": "scenarios/phagecounting.toml", "mode": "fresh"},
        {"toml": "scenarios/mock_tcell.toml", "mode": "fresh"},
    ],
    [  # W2: complex + small
        {"toml": "scenarios/lyme_diagnostics_v2.toml", "mode": "fresh"},
        {"toml": "scenarios/kaggle_pancancer.toml", "mode": "fresh"},
    ],
    [  # W3: complex + medium
        {"toml": "scenarios/insilico_immunotherapy.toml", "mode": "fresh"},
        {"toml": "scenarios/conn2res.toml", "mode": "fresh"},
    ],
    [  # W4: medium + medium
        {"toml": "scenarios/crc_survival.toml", "mode": "fresh"},
        {"toml": "scenarios/turing_patterns.toml", "mode": "fresh"},
    ],
    [  # W5: medium + medium
        {"toml": "scenarios/gmwi2.toml", "mode": "fresh"},
        {"toml": "scenarios/qshgm.toml", "mode": "fresh"},
    ],
]


def log(msg: str, worker: int | None = None) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"W{worker}" if worker is not None else "MAIN"
    print(f"[{ts}] [{prefix}] {msg}", flush=True)


def scenario_name(toml_path: str) -> str:
    full = PROJECT_ROOT / toml_path
    config = tomli.loads(full.read_text())
    return config["name"]


def latest_run_dir(name: str) -> Path | None:
    runs_dir = PROJECT_ROOT / "runs"
    candidates = sorted(runs_dir.glob(f"{name}_*"), key=lambda p: p.name, reverse=True)
    return candidates[0] if candidates else None


def check_run_result(run_dir: Path) -> str:
    session = run_dir / "session.json"
    if not session.is_file():
        return "crashed"
    data = json.loads(session.read_text())
    return data.get("run_result", {}).get("stop_reason", "unknown")


def run_scenario(task: dict, worker: int) -> str:
    """Run a single scenario with retries. Returns final status."""
    toml_path = task["toml"]
    mode = task.get("mode", "auto")
    name = scenario_name(toml_path)

    log(f"START {name} (mode={mode})", worker)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"Attempt {attempt}/{MAX_ATTEMPTS} for {name}", worker)

        prev_dir = latest_run_dir(name)
        cmd = [
            sys.executable, "scripts/run_validation.py", toml_path,
            "--provider", PROVIDER, "--user-provider", USER_PROVIDER,
        ]

        if mode == "fresh":
            pass  # no --resume
        elif prev_dir and (prev_dir / "session.json").is_file():
            prev_stop = check_run_result(prev_dir)
            if prev_stop == "completed":
                log(f"{name} already completed, skipping.", worker)
                return "completed"
            log(f"Resuming from {prev_dir.name} (prev: {prev_stop})", worker)
            cmd.extend(["--resume", str(prev_dir)])

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        elapsed = int(time.time() - t0)

        # After first attempt, switch to auto for retries
        mode = "auto"

        run_dir = latest_run_dir(name)
        if run_dir is None:
            log(f"{name} ERROR: no run dir after {elapsed}s", worker)
            time.sleep(10)
            continue

        stop = check_run_result(run_dir)
        log(f"{name} → {stop} in {elapsed}s (dir: {run_dir.name})", worker)

        if stop in ("completed", "max_turns_reached"):
            return stop
        time.sleep(10)

    log(f"{name} FAILED after {MAX_ATTEMPTS} attempts", worker)
    return "failed"


def run_worker(worker_id: int, tasks: list[dict]) -> dict[str, str]:
    """Run a group of scenarios serially. Returns {name: status}."""
    # Stagger start: each worker waits worker_id * 5 seconds to avoid
    # simultaneous workspace creation and initial API bursts.
    time.sleep(worker_id * 5)
    results: dict[str, str] = {}
    log(f"Starting with {len(tasks)} scenarios", worker_id)
    for task in tasks:
        name = scenario_name(task["toml"])
        status = run_scenario(task, worker_id)
        results[name] = status
    log(f"Finished all {len(tasks)} scenarios", worker_id)
    return results


def main() -> None:
    total = sum(len(g) for g in WORKER_GROUPS)
    log(f"Batch run: {total} scenarios across {N_WORKERS} parallel workers")
    log(f"Provider: {PROVIDER} | User: {USER_PROVIDER} | Max attempts: {MAX_ATTEMPTS}")

    all_results: dict[str, str] = {}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(run_worker, i + 1, group): i + 1
            for i, group in enumerate(WORKER_GROUPS)
        }
        for future in as_completed(futures):
            worker_id = futures[future]
            worker_results = future.result()
            all_results.update(worker_results)
            log(f"Worker {worker_id} done: {worker_results}", None)

    elapsed = int(time.time() - t0)
    log(f"ALL DONE in {elapsed}s ({elapsed // 3600}h {(elapsed % 3600) // 60}m)")
    log("Results:")
    for name, status in sorted(all_results.items()):
        log(f"  {name}: {status}")

    subprocess.run([sys.executable, "scripts/summarize_runs.py"], cwd=str(PROJECT_ROOT))
    log("Updated experiment_log.md")


if __name__ == "__main__":
    main()
