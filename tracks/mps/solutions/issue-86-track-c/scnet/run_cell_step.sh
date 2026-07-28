#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: run_cell_step.sh CELL_ID" >&2
  exit 2
fi

: "${TRACK_C_RUN_JSON_ABS:?missing TRACK_C_RUN_JSON_ABS}"
: "${TRACK_C_OUTPUT_ROOT_ABS:?missing TRACK_C_OUTPUT_ROOT_ABS}"
: "${TRACK_C_SOLUTION_DIRECTORY:?missing TRACK_C_SOLUTION_DIRECTORY}"
: "${TRACK_C_JULIA_PROJECT:?missing TRACK_C_JULIA_PROJECT}"
: "${TRACK_C_CORES_PER_WORKER:?missing TRACK_C_CORES_PER_WORKER}"
: "${TRACK_C_MEMORY_PER_WORKER_MB:?missing TRACK_C_MEMORY_PER_WORKER_MB}"

cell_id="$1"
export JULIA_NUM_THREADS="$TRACK_C_CORES_PER_WORKER"
export OPENBLAS_NUM_THREADS="$TRACK_C_CORES_PER_WORKER"
export OMP_NUM_THREADS="$TRACK_C_CORES_PER_WORKER"

if srun --exact --exclusive --nodes=1 --ntasks=1 \
    --cpus-per-task="$TRACK_C_CORES_PER_WORKER" \
    --mem="${TRACK_C_MEMORY_PER_WORKER_MB}M" \
    --cpu-bind=cores --unbuffered \
    julia --project="$TRACK_C_JULIA_PROJECT" \
      "$TRACK_C_SOLUTION_DIRECTORY/track_c.jl" cell \
      "$TRACK_C_RUN_JSON_ABS" "$cell_id" "$TRACK_C_OUTPUT_ROOT_ABS"; then
  exit 0
else
  status="$?"
fi

echo "Track C cell $cell_id failed with status $status" >&2
exit 1
