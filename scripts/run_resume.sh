#!/bin/bash
# Resume loop: keeps resuming max_turns_reached scenarios until all
# are completed or max passes exhausted.
# Chain after run_batch.py:
#   nohup python -u scripts/run_batch.py > batch_run.log 2>&1 && \
#   nohup bash scripts/run_resume.sh > resume_run.log 2>&1 &
#
# Or standalone: nohup bash scripts/run_resume.sh > resume_run.log 2>&1 &

cd "$(dirname "$0")/.."
PROVIDER="openrouter"
USER_PROVIDER="mimo"
MAX_PASSES=5

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# Map scenario name (from dir) to toml path
find_toml() {
    local name="$1"
    for candidate in "scenarios/${name}.toml" "scenarios/${name}_v2.toml"; do
        [ -f "$candidate" ] && echo "$candidate" && return
    done
}

# Get latest run dir for a scenario name
latest_dir() {
    ls -dt runs/${1}_* 2>/dev/null | head -1
}

log "=== Resume loop starting (max $MAX_PASSES passes) ==="

for pass in $(seq 1 $MAX_PASSES); do
    log "--- Pass $pass/$MAX_PASSES ---"

    any_resumed=false

    # Scan ALL scenario run dirs (latest per name)
    for toml in scenarios/*.toml; do
        name=$(python3 -c "import tomli; print(tomli.loads(open('$toml').read())['name'])")
        dir=$(latest_dir "$name")
        [ -z "$dir" ] && continue
        [ ! -f "$dir/session.json" ] && continue

        stop=$(python3 -c "import json; print(json.load(open('$dir/session.json')).get('run_result',{}).get('stop_reason',''))")

        if [ "$stop" = "completed" ]; then
            log "SKIP $name: already completed"
            continue
        fi

        if [ "$stop" = "max_turns_reached" ] || [ "$stop" = "in_progress" ]; then
            log "RESUME $name (stop=$stop) from $dir"
            python -u scripts/run_validation.py "$toml" \
                --provider "$PROVIDER" --user-provider "$USER_PROVIDER" \
                --resume "$dir"

            # Check result
            new_dir=$(latest_dir "$name")
            new_stop=$(python3 -c "import json; print(json.load(open('$new_dir/session.json')).get('run_result',{}).get('stop_reason',''))" 2>/dev/null)
            log "$name → $new_stop (dir: $(basename $new_dir))"
            any_resumed=true
        fi
    done

    if [ "$any_resumed" = false ]; then
        log "No scenarios need resuming. Done."
        break
    fi

    log "Pass $pass complete. Checking if more resumes needed..."
    sleep 10
done

log "=== Resume loop finished ==="
