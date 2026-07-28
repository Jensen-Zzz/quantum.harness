#!/bin/bash
#SBATCH --job-name=qh86-c-gammac
#SBATCH --partition=xhacnormalb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=23:50:00
#SBATCH --output=tracks/mps/results/issue-86-track-c/slurm-%x-%j.out

set -euo pipefail

: "${TRACK_C_CRITICAL_OUTPUT:?missing TRACK_C_CRITICAL_OUTPUT}"
: "${TRACK_C_SIGMA:?missing TRACK_C_SIGMA}"
: "${TRACK_C_L:?missing TRACK_C_L}"
: "${TRACK_C_INITIAL_GAMMA:?missing TRACK_C_INITIAL_GAMMA}"
: "${TRACK_C_CHI:=96}"
: "${TRACK_C_SOLUTION_DIRECTORY:?missing TRACK_C_SOLUTION_DIRECTORY}"

mkdir -p "$(dirname "$TRACK_C_CRITICAL_OUTPUT")"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$JULIA_NUM_THREADS"
export OMP_NUM_THREADS="$JULIA_NUM_THREADS"

srun --exact --unbuffered \
  julia --project="$TRACK_C_SOLUTION_DIRECTORY/../../../../julia-env" \
    "$TRACK_C_SOLUTION_DIRECTORY/track_c.jl" critical \
    "$TRACK_C_CRITICAL_OUTPUT" "$TRACK_C_SIGMA" "$TRACK_C_L" \
    "$TRACK_C_INITIAL_GAMMA" "$TRACK_C_CHI"
