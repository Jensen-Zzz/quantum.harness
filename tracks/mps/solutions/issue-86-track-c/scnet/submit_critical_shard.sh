#!/bin/bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: submit_critical_shard.sh CRITICAL_SPEC.json OUTPUT_ROOT" >&2
  exit 2
fi

critical_spec="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
output_root="$2"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
solution_directory="$(cd "$script_directory/.." && pwd)"
repo_root="$(cd "$solution_directory/../../../.." && pwd)"

[[ -s "$critical_spec" ]] || {
  echo "critical spec not found: $critical_spec" >&2
  exit 2
}
mkdir -p "$output_root"
output_root="$(cd "$output_root" && pwd)"

active_jobs="$(squeue -h -u "$USER" -t PD,R -o '%A')"
if [[ -n "$active_jobs" ]]; then
  echo "SCNet already has an active job for $USER: $active_jobs" >&2
  exit 1
fi
sinfo -h -p xhacnormalb -o '%P %a %l %D %t' || exit 1

mkdir -p "$repo_root/tracks/mps/results/issue-86-track-c"
cd "$repo_root"
sbatch \
  --export=ALL,TRACK_C_CRITICAL_SPEC="$critical_spec",TRACK_C_CRITICAL_OUTPUT_ROOT="$output_root" \
  "$script_directory/run_critical_packed.sbatch"
