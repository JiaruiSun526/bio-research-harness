#!/bin/bash
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

PROVIDER="openrouter"
USER_PROVIDER="mimo"

# ---- Step 1: Wait for current batch ----
BATCH_PID=${1:-0}
if [ "$BATCH_PID" -gt 0 ] && kill -0 "$BATCH_PID" 2>/dev/null; then
    log "Waiting for current batch (PID $BATCH_PID) to finish..."
    while kill -0 "$BATCH_PID" 2>/dev/null; do sleep 30; done
    log "Current batch finished."
fi

# ---- Step 2: Generate missing TOMLs ----
for args in \
    "--name conn2res --paper papers/conn2res_neuro/paper.pdf --data-dir papers/conn2res_neuro/data --max-turns 50" \
    "--name gmwi2 --paper papers/gmwi2_gut_microbiome/paper.pdf --data-dir papers/gmwi2_gut_microbiome/data --max-turns 50"
do
    name=$(echo "$args" | grep -o '\-\-name [^ ]*' | awk '{print $2}')
    toml="scenarios/${name}.toml"
    if [ -f "$toml" ]; then
        log "TOML already exists: $toml, skipping generation"
    else
        log "Generating $toml ..."
        python scripts/prepare_scenario.py $args
        if [ $? -ne 0 ]; then
            log "ERROR: Failed to generate $toml"
        else
            log "Generated $toml"
        fi
    fi
done

# ---- Step 3: Run all pending scenarios ----
run_one() {
    local toml="$1"
    local max_attempts=3
    local name
    name=$(python3 -c "import tomli; print(tomli.loads(open('$toml').read())['name'])")

    log "=========================================="
    log "SCENARIO: $name"
    log "=========================================="

    for attempt in $(seq 1 $max_attempts); do
        log "Attempt $attempt/$max_attempts for $name"

        # Find latest run
        local prev_dir
        prev_dir=$(ls -dt runs/${name}_* 2>/dev/null | head -1)
        local cmd="python scripts/run_validation.py $toml --provider $PROVIDER --user-provider $USER_PROVIDER"

        if [ -n "$prev_dir" ] && [ -f "$prev_dir/session.json" ]; then
            local prev_stop
            prev_stop=$(python3 -c "import json; print(json.load(open('$prev_dir/session.json')).get('run_result',{}).get('stop_reason','?'))")
            if [ "$prev_stop" = "completed" ]; then
                log "$name already completed in $(basename $prev_dir), skipping."
                return
            fi
            log "Resuming from $(basename $prev_dir) (previous: $prev_stop)"
            cmd="$cmd --resume $prev_dir"
        else
            log "Starting fresh run"
        fi

        log "CMD: $cmd"
        local t0=$(date +%s)
        eval "$cmd"
        local elapsed=$(( $(date +%s) - t0 ))
        log "$name finished attempt in ${elapsed}s"

        # Check result
        prev_dir=$(ls -dt runs/${name}_* 2>/dev/null | head -1)
        if [ -z "$prev_dir" ]; then
            log "ERROR: No run directory for $name"
            sleep 10; continue
        fi

        local stop="crashed"
        [ -f "$prev_dir/session.json" ] && \
            stop=$(python3 -c "import json; print(json.load(open('$prev_dir/session.json')).get('run_result',{}).get('stop_reason','?'))")
        log "Result: stop=$stop ($(basename $prev_dir))"

        case "$stop" in
            completed)     log "$name SUCCEEDED"; return ;;
            max_turns*)    log "$name max_turns — partial results, moving on"; return ;;
            *)             log "$name stopped ($stop), retrying..."; sleep 10 ;;
        esac
    done
    log "$name done after $max_attempts attempts"
}

# Run scenarios in order
for toml in \
    scenarios/conn2res.toml \
    scenarios/gmwi2.toml \
    scenarios/lyme_diagnostics_v2.toml \
    scenarios/qshgm.toml
do
    [ -f "$toml" ] && run_one "$toml"
done

log "=========================================="
log "OVERNIGHT BATCH DONE"
log "=========================================="
python scripts/summarize_runs.py
log "Updated experiment_log.md"
