#!/bin/bash
# Wait for batch to finish, then run resume loop
cd "$(dirname "$0")/.."
BATCH_PID=$1

echo "[$(date)] Waiting for batch PID $BATCH_PID to finish..."
while kill -0 $BATCH_PID 2>/dev/null; do
    sleep 30
done
echo "[$(date)] Batch finished. Starting resume loop..."

bash scripts/run_resume.sh
