#!/bin/bash
#SBATCH --job-name=qh86-track-c
#SBATCH --partition=xhacnormalb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
#SBATCH --time=23:50:00
#SBATCH --output=tracks/mps/results/issue-86-track-c/slurm-%x-%j.out

set -euo pipefail

: "${TRACK_C_RUN_JSON:?set TRACK_C_RUN_JSON to an existing run.json}"
: "${TRACK_C_OUTPUT_ROOT:?set TRACK_C_OUTPUT_ROOT}"
: "${TRACK_C_RESOURCE:?set TRACK_C_RESOURCE to standard or large}"
: "${TRACK_C_DURATION:?set TRACK_C_DURATION to short or long}"
: "${TRACK_C_SCNET_DIRECTORY:?set TRACK_C_SCNET_DIRECTORY}"

bash "$TRACK_C_SCNET_DIRECTORY/packed_worker.sh" \
  "$TRACK_C_RUN_JSON" "$TRACK_C_OUTPUT_ROOT" \
  "$TRACK_C_RESOURCE" "$TRACK_C_DURATION"
