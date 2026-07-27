from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "skills/using-slurm/profiles/scnet.toml"
SBATCH = ROOT / "tracks/mps/solutions/issue-86/run_full.sbatch"
PACKED_WORKER = ROOT / "tracks/mps/solutions/issue-86/packed_worker.sh"
GENERATE_SPEC = ROOT / "tracks/mps/solutions/issue-86/generate_run_spec.jl"
RUN_CELL = ROOT / "tracks/mps/solutions/issue-86/run_cell.jl"
COLLECT = ROOT / "tracks/mps/solutions/issue-86/collect.jl"
FORMAL_ANALYSIS = ROOT / "tracks/mps/solutions/issue-86/analyze_formal.jl"


def test_scnet_profile_targets_xhacnormalb_cpu_node():
    profile = PROFILE.read_text()

    assert 'default_partition = "xhacnormalb"' in profile
    assert 'name = "xhacnormalb"' in profile
    assert 'class = "cpu"' in profile
    assert "cores = 128" in profile
    assert 'memory = "512000M"' in profile
    assert 'gpu = ' not in profile


def test_issue86_sbatch_packs_a_full_cpu_node():
    script = SBATCH.read_text()

    assert "#SBATCH --partition=xhacnormalb" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --cpus-per-task=128" in script
    assert "#SBATCH --mem=480G" in script
    assert "packed_worker.sh" in script
    assert "HARNESS_RUN_SPEC" in script


def test_packed_worker_is_resumable_and_pins_each_worker():
    script = PACKED_WORKER.read_text()

    assert "pending_cells.jl" in script
    assert 'RESOURCE_CLASS' in script
    assert "srun --exclusive" in script
    assert "--cpu-bind=cores" in script
    assert "OPENBLAS_NUM_THREADS" in script
    assert "run_cell.jl" in script


def test_run_spec_entrypoints_are_separate_from_the_solver():
    assert "build_run_spec" in GENERATE_SPEC.read_text()
    assert "execute_cell" in RUN_CELL.read_text()
    assert "collect_cell_results" in COLLECT.read_text()
    analysis = FORMAL_ANALYSIS.read_text()
    assert "fit_crossing_sequence" in analysis
    assert "conservative_error_budget" in analysis
    assert "adaptive_run_spec.json" in analysis
