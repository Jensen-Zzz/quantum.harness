#!/bin/bash

set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: packed_worker.sh RUN_SPEC.json OUTPUT_DIR RESOURCE_CLASS WORKERS CORES_PER_WORKER" >&2
  exit 2
fi

run_spec="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
output_directory="$2"
resource_class="$3"
workers="$4"
cores_per_worker="$5"
solution_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$solution_directory/../../../.." && pwd)"
julia_project="$repo_root/julia-env"

if (( workers < 1 || cores_per_worker < 1 )); then
  echo "worker and core counts must be positive" >&2
  exit 2
fi
if (( workers * cores_per_worker > 128 )); then
  echo "packed layout exceeds the 128-core node" >&2
  exit 2
fi

mkdir -p "$output_directory"
mapfile -t pending_indices < <(
  julia --project="$julia_project" \
    "$solution_directory/pending_cells.jl" \
    "$run_spec" "$output_directory" "$resource_class"
)

if (( ${#pending_indices[@]} == 0 )); then
  echo "no pending class-$resource_class cells"
  exit 0
fi

echo "launching ${#pending_indices[@]} class-$resource_class cells: ${workers} workers x ${cores_per_worker} cores"
export JULIA_NUM_THREADS="$cores_per_worker"
export OPENBLAS_NUM_THREADS="$cores_per_worker"
export OMP_NUM_THREADS="$cores_per_worker"
export MKL_NUM_THREADS="$cores_per_worker"

pids=()
for index in "${pending_indices[@]}"; do
  srun --exclusive --nodes=1 --ntasks=1 \
    --cpus-per-task="$cores_per_worker" --cpu-bind=cores --unbuffered \
    julia --project="$julia_project" \
      "$solution_directory/run_cell.jl" \
      "$run_spec" "$index" "$output_directory" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done

julia --project="$julia_project" \
  "$solution_directory/collect.jl" "$run_spec" "$output_directory"

if (( failures > 0 )); then
  echo "$failures cell step(s) failed; successful manifests were retained" >&2
  exit 1
fi
