#!/bin/bash

set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: submit_critical.sh OUTPUT.json SIGMA L INITIAL_GAMMA CHI" >&2
  exit 2
fi

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
solution_directory="$(cd "$script_directory/.." && pwd)"
repo_root="$(cd "$solution_directory/../../../.." && pwd)"
output="$1"
mkdir -p "$(dirname "$output")"
output="$(cd "$(dirname "$output")" && pwd)/$(basename "$output")"

active_jobs="$(squeue -h -u "$USER" -t PD,R -o '%A')"
if [[ -n "$active_jobs" ]]; then
  echo "SCNet already has an active job for $USER: $active_jobs" >&2
  exit 1
fi
sinfo -h -p xhacnormalb -o '%P %a %l %D %t' || exit 1
mkdir -p "$repo_root/tracks/mps/results/issue-86-track-c"
cd "$repo_root"

sbatch \
  --export=ALL,TRACK_C_CRITICAL_OUTPUT="$output",TRACK_C_SIGMA="$2",TRACK_C_L="$3",TRACK_C_INITIAL_GAMMA="$4",TRACK_C_CHI="$5" \
  "$script_directory/run_critical.sbatch"
