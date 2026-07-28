#!/bin/bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: packed_critical_worker.sh CRITICAL_SPEC.json OUTPUT_ROOT" >&2
  exit 2
fi

critical_spec="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
output_root="$2"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
solution_directory="$(cd "$script_directory/.." && pwd)"
repo_root="$(cd "$solution_directory/../../../.." && pwd)"
julia_project="$repo_root/julia-env"
workers=4
cores_per_worker=32

mkdir -p "$output_root"
pending_ids=()
while IFS= read -r cell_id; do
  [[ -n "$cell_id" ]] && pending_ids+=("$cell_id")
done < <(
  julia --project="$julia_project" "$solution_directory/track_c.jl" \
    critical-pending "$critical_spec" "$output_root"
)

if (( ${#pending_ids[@]} == 0 )); then
  echo "no pending Track C critical scans"
  exit 0
fi

allocated_cpus="${SLURM_CPUS_PER_TASK:-128}"
(( workers * cores_per_worker <= allocated_cpus )) || {
  echo "critical worker layout exceeds allocation" >&2
  exit 2
}
allocated_memory_mb="${SLURM_MEM_PER_NODE:-491520}"
memory_per_worker=$((allocated_memory_mb / workers))
echo "Track C critical: ${#pending_ids[@]} scans, ${workers}x${cores_per_worker} cores, ${memory_per_worker} MB/worker"

export TRACK_C_CRITICAL_SPEC_ABS="$critical_spec"
export TRACK_C_CRITICAL_OUTPUT_ROOT_ABS="$output_root"
export TRACK_C_SOLUTION_DIRECTORY="$solution_directory"
export TRACK_C_JULIA_PROJECT="$julia_project"
export TRACK_C_CORES_PER_WORKER="$cores_per_worker"
export TRACK_C_MEMORY_PER_WORKER_MB="$memory_per_worker"

set +e
printf '%s\n' "${pending_ids[@]}" |
  xargs -n 1 -P "$workers" bash "$script_directory/run_critical_step.sh"
worker_status="${PIPESTATUS[1]}"
set -e

if (( worker_status != 0 )); then
  echo "one or more Track C critical scans failed; checkpoints were retained" >&2
  exit 1
fi
