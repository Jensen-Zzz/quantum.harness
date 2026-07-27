using JSON
using Test
using TOML

if !isdefined(Main, :Issue86TrackB)
    include(joinpath(@__DIR__, "..", "src", "Issue86TrackB.jl"))
end
using .Issue86TrackB

@testset "Run spec expansion and resource classes" begin
    config = Dict(
        "sweeps" => Any[
            Dict(
                "model" => "nn",
                "lengths" => [16],
                "gammas" => [0.99, 1.0],
                "chis" => [64, 128],
                "excited" => false,
            ),
            Dict(
                "model" => "long_range",
                "sigmas" => [1.75],
                "lengths" => [128],
                "gammas" => [1.5609],
                "chis" => [128, 256],
                "poles" => [16],
            ),
        ],
    )

    spec = build_run_spec(config; run_id = "unit-run", stage = "stage-test")
    @test spec["metadata"]["run_id"] == "unit-run"
    @test spec["metadata"]["stage"] == "stage-test"
    @test length(spec["cells"]) == 6
    @test [cell["cell_id"] for cell in spec["cells"]] ==
        ["stage-test-0001", "stage-test-0002", "stage-test-0003",
         "stage-test-0004", "stage-test-0005", "stage-test-0006"]
    @test [cell["resource_class"] for cell in spec["cells"]] ==
        ["A", "B", "A", "B", "C", "D"]
    @test all(cell["params"]["tolerance"] == 1.0e-8 for cell in spec["cells"])
end

@testset "Production stage configurations have the planned cells" begin
    config_directory = joinpath(@__DIR__, "..", "configs")
    expectations = Dict(
        "calibration.toml" => (1, Dict("A" => 1)),
        "stage1.toml" => (75, Dict("A" => 70, "B" => 5)),
        "stage2-baseline.toml" => (60, Dict("A" => 60)),
        "stage2-systematics.toml" => (32, Dict("A" => 20, "B" => 12)),
        "stage2-contingency.toml" => (6, Dict("C" => 6)),
        "stage2-chi256.toml" => (6, Dict("D" => 6)),
    )

    for (filename, (expected_total, expected_classes)) in expectations
        config = TOML.parsefile(joinpath(config_directory, filename))
        spec = build_run_spec(config; run_id = filename, stage = splitext(filename)[1])
        counts = Dict{String, Int}()
        for cell in spec["cells"]
            counts[cell["resource_class"]] =
                get(counts, cell["resource_class"], 0) + 1
        end
        @test length(spec["cells"]) == expected_total
        @test counts == expected_classes
    end
end

@testset "Completed cells are excluded from resume" begin
    config = Dict(
        "sweeps" => Any[
            Dict(
                "model" => "nn",
                "lengths" => [8],
                "gammas" => [0.99, 1.0, 1.01],
                "chis" => [64],
            ),
        ],
    )
    spec = build_run_spec(config; run_id = "resume-run", stage = "stage1")

    mktempdir() do directory
        first_manifest = joinpath(directory, "cells", "stage1-0001", "manifest.json")
        mkpath(dirname(first_manifest))
        open(first_manifest, "w") do io
            JSON.print(io, Dict("status" => "success", "result" => Dict("E0" => -1.0)))
        end

        failed_manifest = joinpath(directory, "cells", "stage1-0002", "manifest.json")
        mkpath(dirname(failed_manifest))
        open(failed_manifest, "w") do io
            JSON.print(io, Dict("status" => "failed", "error" => "test failure"))
        end

        @test pending_cell_indices(spec, directory) == [2, 3]
        @test pending_cell_indices(spec, directory; resource_class = "B") == Int[]

        collected = collect_cell_results(spec, directory)
        @test length(collected) == 1
        @test collected[1]["E0"] == -1.0
        @test collected[1]["cell_id"] == "stage1-0001"
    end
end

@testset "Cell execution writes an atomic resumable manifest" begin
    config = Dict(
        "sweeps" => Any[
            Dict(
                "model" => "nn",
                "lengths" => [8],
                "gammas" => [1.0],
                "chis" => [64],
            ),
        ],
    )
    spec = build_run_spec(config; run_id = "cell-run", stage = "stage1")
    calls = Ref(0)
    fake_solver = function (; kwargs...)
        calls[] += 1
        return Dict{String, Any}("E0" => -8.0, "runtime" => 0.01)
    end

    mktempdir() do directory
        first = execute_cell(spec, 1, directory; solver = fake_solver)
        second = execute_cell(spec, 1, directory; solver = fake_solver)
        manifest_path = joinpath(directory, "cells", "stage1-0001", "manifest.json")
        manifest = JSON.parsefile(manifest_path)

        @test calls[] == 1
        @test first["E0"] == second["E0"] == -8.0
        @test manifest["status"] == "success"
        @test manifest["cell_id"] == "stage1-0001"
        @test manifest["resource_class"] == "A"
        @test manifest["runtime"]["julia_threads"] >= 1
        @test isempty(filter(name -> occursin(".tmp-", name), readdir(dirname(manifest_path))))
    end
end
