#!/bin/bash
#SBATCH --job-name=qh86-c-critical
#SBATCH --partition=xhacnormalb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
#SBATCH --time=23:50:00
#SBATCH --output=tracks/mps/results/issue-86-track-c/slurm-%x-%j.out

set -euo pipefail

: "${TRACK_C_CRITICAL_SPEC:?set TRACK_C_CRITICAL_SPEC}"
: "${TRACK_C_CRITICAL_OUTPUT_ROOT:?set TRACK_C_CRITICAL_OUTPUT_ROOT}"
: "${TRACK_C_SCNET_DIRECTORY:?set TRACK_C_SCNET_DIRECTORY}"

bash "$TRACK_C_SCNET_DIRECTORY/packed_critical_worker.sh" \
  "$TRACK_C_CRITICAL_SPEC" "$TRACK_C_CRITICAL_OUTPUT_ROOT"
