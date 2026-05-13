#!/bin/bash
# Batch runner: sequentially run scenarios, retry on failure, log everything.
# Usage: nohup bash scripts/run_batch.sh > batch_run.log 2>&1 &
#
# Each scenario gets up to MAX_ATTEMPTS tries. If a run produces session.json
# with stop_reason != "completed", it attempts --resume from that workspace.
# Fresh retry if no session.json (hard crash).

set -u
cd "$(dirname "$0")/.."

MAX_ATTEMPTS=3
PROVIDER="openrouter"
USER_PROVIDER="mimo"
LOG_FILE="batch_run.log"

# ---- Scenarios to run (edit this list) ----
SCENARIOS=(
    "scenarios/insilico_immunotherapy.toml"
    "scenarios/lyme_diagnostics_v2.toml"
    "scenarios/qshgm.toml"
    "scenarios/turing_patterns.toml"
)

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(timestamp)] $*"; }

# Extract scenario name from TOML 'name' field (not filename).
# This must match the run directory prefix created by run_validation.py.
scenario_name() {
    local toml_path="$1"
    python3 -c "import tomli; print(tomli.loads(open('${toml_path}').read())['name'])"
}

# Find the latest run directory for a scenario
latest_run_dir() {
    local name="$1"
    ls -dt runs/${name}_* 2>/dev/null | head -1
}

# Check if a run completed successfully
check_run_result() {
    local run_dir="$1"
    if [ ! -f "$run_dir/session.json" ]; then
        echo "crashed"
        return
    fi
    python3 -c "
import json
s = json.load(open('$run_dir/session.json'))
print(s.get('run_result', {}).get('stop_reason', 'unknown'))
"
}

run_scenario() {
    local scenario_toml="$1"
    local name
    name=$(scenario_name "$scenario_toml")

    log "=========================================="
    log "SCENARIO: $name"
    log "=========================================="

    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        log "Attempt $attempt/$MAX_ATTEMPTS for $name"

        # Check if previous run exists and can be resumed
        local prev_dir
        prev_dir=$(latest_run_dir "$name")
        local cmd

        if [ -n "$prev_dir" ] && [ -f "$prev_dir/session.json" ]; then
            local prev_stop
            prev_stop=$(check_run_result "$prev_dir")
            if [ "$prev_stop" = "completed" ]; then
                log "$name already completed in $prev_dir, skipping."
                return 0
            fi
            # Has session.json but not completed — resume
            log "Resuming from $prev_dir (previous stop: $prev_stop)"
            cmd="python scripts/run_validation.py $scenario_toml --resume $prev_dir --provider $PROVIDER --user-provider $USER_PROVIDER"
        else
            # Fresh run (no session.json or no previous run)
            log "Starting fresh run"
            cmd="python scripts/run_validation.py $scenario_toml --provider $PROVIDER --user-provider $USER_PROVIDER"
        fi

        log "CMD: $cmd"
        local start_time
        start_time=$(date +%s)

        if eval "$cmd"; then
            local end_time elapsed
            end_time=$(date +%s)
            elapsed=$(( end_time - start_time ))
            log "$name finished in ${elapsed}s"
        else
            local end_time elapsed
            end_time=$(date +%s)
            elapsed=$(( end_time - start_time ))
            log "$name failed after ${elapsed}s (exit code $?)"
        fi

        # Check result
        local run_dir
        run_dir=$(latest_run_dir "$name")
        if [ -z "$run_dir" ]; then
            log "ERROR: No run directory found for $name"
            continue
        fi

        local stop_reason
        stop_reason=$(check_run_result "$run_dir")
        log "Result: stop_reason=$stop_reason (dir: $run_dir)"

        if [ "$stop_reason" = "completed" ]; then
            log "$name SUCCEEDED"
            break
        elif [ "$stop_reason" = "max_turns_reached" ]; then
            log "$name hit max_turns — has partial results, moving on"
            break
        else
            log "$name stopped with $stop_reason, will retry"
            sleep 10  # brief cooldown before retry
        fi
    done

    log "$name done after $attempt attempt(s)"
    log ""
}

# ---- Main ----
log "Batch run starting with ${#SCENARIOS[@]} scenarios"
log "Provider: $PROVIDER | User: $USER_PROVIDER | Max attempts: $MAX_ATTEMPTS"
log ""

for scenario in "${SCENARIOS[@]}"; do
    run_scenario "$scenario"
done

log "=========================================="
log "ALL SCENARIOS DONE"
log "=========================================="

# Generate summary
python scripts/summarize_runs.py
log "Updated experiment_log.md"
