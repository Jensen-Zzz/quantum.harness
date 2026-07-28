#!/bin/bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: packed_worker.sh RUN.json OUTPUT_ROOT standard|large short|long" >&2
  exit 2
fi

run_json="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
output_root="$2"
resource_class="$3"
duration_class="$4"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
solution_directory="$(cd "$script_directory/.." && pwd)"
repo_root="$(cd "$solution_directory/../../../.." && pwd)"
julia_project="$repo_root/julia-env"

case "$resource_class" in
  standard)
    workers=8
    cores_per_worker=16
    ;;
  large)
    workers=4
    cores_per_worker=32
    ;;
  *)
    echo "resource class must be standard or large" >&2
    exit 2
    ;;
esac

case "$duration_class" in
  short|long) ;;
  *)
    echo "duration class must be short or long" >&2
    exit 2
    ;;
esac

mkdir -p "$output_root"
pending_ids=()
while IFS= read -r cell_id; do
  [[ -n "$cell_id" ]] && pending_ids+=("$cell_id")
done < <(
  julia --project="$julia_project" "$solution_directory/track_c.jl" \
    pending "$run_json" "$resource_class" "$duration_class"
)

if (( ${#pending_ids[@]} == 0 )); then
  echo "no pending $resource_class/$duration_class Track C cells"
  exit 0
fi

allocated_cpus="${SLURM_CPUS_PER_TASK:-128}"
(( workers * cores_per_worker <= allocated_cpus )) || {
  echo "packed worker layout exceeds allocation" >&2
  exit 2
}

allocated_memory_mb="${SLURM_MEM_PER_NODE:-491520}"
memory_per_worker=$((allocated_memory_mb / workers))
echo "Track C: ${#pending_ids[@]} cells, ${workers}x${cores_per_worker} cores, ${memory_per_worker} MB/worker"

export TRACK_C_RUN_JSON_ABS="$run_json"
export TRACK_C_OUTPUT_ROOT_ABS="$output_root"
export TRACK_C_SOLUTION_DIRECTORY="$solution_directory"
export TRACK_C_JULIA_PROJECT="$julia_project"
export TRACK_C_CORES_PER_WORKER="$cores_per_worker"
export TRACK_C_MEMORY_PER_WORKER_MB="$memory_per_worker"

set +e
printf '%s\n' "${pending_ids[@]}" |
  xargs -n 1 -P "$workers" bash "$script_directory/run_cell_step.sh"
worker_status="${PIPESTATUS[1]}"
set -e

if (( worker_status != 0 )); then
  echo "one or more Track C cells failed; completed manifests were retained" >&2
  exit 1
fi
