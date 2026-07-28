#!/bin/bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: submit.sh RUN.json OUTPUT_ROOT standard|large short|long" >&2
  exit 2
fi

run_json="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
output_root="$2"
resource_class="$3"
duration_class="$4"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
solution_directory="$(cd "$script_directory/.." && pwd)"
repo_root="$(cd "$solution_directory/../../../.." && pwd)"

[[ -s "$run_json" ]] || {
  echo "run.json not found: $run_json" >&2
  exit 2
}
[[ "$resource_class" == "standard" || "$resource_class" == "large" ]] ||
  { echo "resource class must be standard or large" >&2; exit 2; }
[[ "$duration_class" == "short" || "$duration_class" == "long" ]] ||
  { echo "duration class must be short or long" >&2; exit 2; }
mkdir -p "$output_root"
output_root="$(cd "$output_root" && pwd)"

active_jobs="$(squeue -h -u "$USER" -t PD,R -o '%A')"
if [[ -n "$active_jobs" ]]; then
  echo "SCNet already has an active job for $USER: $active_jobs" >&2
  echo "wait for it to settle before submitting the next Track C class" >&2
  exit 1
fi
sinfo -h -p xhacnormalb -o '%P %a %l %D %t' || exit 1
mkdir -p "$repo_root/tracks/mps/results/issue-86-track-c"
cd "$repo_root"

sbatch \
  --export=ALL,TRACK_C_RUN_JSON="$run_json",TRACK_C_OUTPUT_ROOT="$output_root",TRACK_C_RESOURCE="$resource_class",TRACK_C_DURATION="$duration_class" \
  "$script_directory/run_packed.sbatch"
