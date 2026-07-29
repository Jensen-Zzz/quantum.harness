#!/bin/bash

set -euo pipefail

test_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
solution_directory="$(cd "$test_directory/.." && pwd)"
temporary_directory="$(mktemp -d)"
trap 'rm -rf "$temporary_directory"' EXIT

fake_scnet_directory="$temporary_directory/repository-scnet"
fake_solution_directory="$temporary_directory/repository-solution"
fake_binary_directory="$temporary_directory/bin"
spool_directory="$temporary_directory/slurm-spool"
mkdir -p \
  "$fake_scnet_directory" "$fake_solution_directory" \
  "$fake_binary_directory" "$spool_directory"

printf '%s\n' \
  '#!/bin/bash' \
  'printf "critical-worker:%s:%s\n" "$1" "$2"' \
  > "$fake_scnet_directory/packed_critical_worker.sh"
printf '%s\n' \
  '#!/bin/bash' \
  'printf "dynamics-worker:%s:%s:%s:%s\n" "$1" "$2" "$3" "$4"' \
  > "$fake_scnet_directory/packed_worker.sh"
chmod +x \
  "$fake_scnet_directory/packed_critical_worker.sh" \
  "$fake_scnet_directory/packed_worker.sh"
printf '%s\n' \
  '#!/bin/bash' \
  'printf "julia:%s\n" "$*"' \
  > "$fake_binary_directory/julia"
printf '%s\n' \
  '#!/bin/bash' \
  'while [[ "$1" == --* ]]; do shift; done' \
  'exec "$@"' \
  > "$fake_binary_directory/srun"
chmod +x "$fake_binary_directory/julia" "$fake_binary_directory/srun"
touch "$fake_solution_directory/track_c.jl"

# Slurm executes a copied batch script from its spool directory, so BASH_SOURCE
# cannot be used to locate repository-side worker scripts.
cp "$solution_directory/scnet/run_critical_packed.sbatch" \
  "$spool_directory/critical-job.sh"
critical_output="$(
  TRACK_C_SCNET_DIRECTORY="$fake_scnet_directory" \
  TRACK_C_CRITICAL_SPEC="/remote/spec.json" \
  TRACK_C_CRITICAL_OUTPUT_ROOT="/remote/critical-output" \
  bash "$spool_directory/critical-job.sh"
)"
[[ "$critical_output" == \
   "critical-worker:/remote/spec.json:/remote/critical-output" ]]

cp "$solution_directory/scnet/run_packed.sbatch" \
  "$spool_directory/dynamics-job.sh"
dynamics_output="$(
  TRACK_C_SCNET_DIRECTORY="$fake_scnet_directory" \
  TRACK_C_RUN_JSON="/remote/run.json" \
  TRACK_C_OUTPUT_ROOT="/remote/dynamics-output" \
  TRACK_C_RESOURCE="standard" \
  TRACK_C_DURATION="short" \
  bash "$spool_directory/dynamics-job.sh"
)"
[[ "$dynamics_output" == \
   "dynamics-worker:/remote/run.json:/remote/dynamics-output:standard:short" ]]

cp "$solution_directory/scnet/run_critical.sbatch" \
  "$spool_directory/single-critical-job.sh"
single_output_path="$temporary_directory/single.json"
single_output="$(
  PATH="$fake_binary_directory:$PATH" \
  TRACK_C_SOLUTION_DIRECTORY="$fake_solution_directory" \
  TRACK_C_CRITICAL_OUTPUT="$single_output_path" \
  TRACK_C_SIGMA="1.875" \
  TRACK_C_L="64" \
  TRACK_C_INITIAL_GAMMA="1.29" \
  TRACK_C_CHI="96" \
  TRACK_C_TARGET_WIDTH="0.002" \
  SLURM_CPUS_PER_TASK="32" \
  bash "$spool_directory/single-critical-job.sh"
)"
[[ "$single_output" == *"$fake_solution_directory/track_c.jl critical"* ]]

printf '%s\n' \
  '#!/bin/bash' \
  'if [[ "$*" == *" pending "* ]]; then printf "%s\n" "test-cell"; fi' \
  > "$fake_binary_directory/julia"
chmod +x "$fake_binary_directory/julia"
touch "$temporary_directory/run.json"

standard_worker_output="$(
  PATH="$fake_binary_directory:$PATH" \
  SLURM_CPUS_PER_TASK="64" \
  SLURM_MEM_PER_NODE="245760" \
  bash "$solution_directory/scnet/packed_worker.sh" \
    "$temporary_directory/run.json" \
    "$temporary_directory/standard-output" standard short
)"
[[ "$standard_worker_output" == *"Track C: 1 cells, 4x16 cores, 61440 MB/worker"* ]]

large_worker_output="$(
  PATH="$fake_binary_directory:$PATH" \
  SLURM_CPUS_PER_TASK="64" \
  SLURM_MEM_PER_NODE="245760" \
  bash "$solution_directory/scnet/packed_worker.sh" \
    "$temporary_directory/run.json" \
    "$temporary_directory/large-output" large short
)"
[[ "$large_worker_output" == *"Track C: 1 cells, 2x32 cores, 122880 MB/worker"* ]]

printf '%s\n' "slurm-spool-paths-ok"
