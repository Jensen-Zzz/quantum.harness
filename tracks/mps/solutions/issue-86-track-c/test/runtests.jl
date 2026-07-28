using Test
using LinearAlgebra
using Random
using TensorKit
using TensorKit: ℂ
using MPSKit

const TRACK_C_ROOT = normpath(joinpath(@__DIR__, ".."))
const TRACK_C_MODULE = joinpath(TRACK_C_ROOT, "src", "Issue86TrackC.jl")

@testset "Track C module boundary" begin
    @test isfile(TRACK_C_MODULE)
end

include(TRACK_C_MODULE)
using .Issue86TrackC

@testset "G0 OBC SOE and dense Hamiltonian agreement" begin
    approximation = fit_obc_soe(256, 1.0, 16)
    @test approximation.max_relative_error <= 1.0e-6
    relative_errors = [
        abs(obc_soe_coupling(approximation, r) / obc_coupling(1.0, r) - 1)
        for r in 1:255
    ]
    @test maximum(relative_errors) <= 1.0e-6

    L = 4
    gamma = 0.73
    small_fit = fit_obc_soe(L, 1.0, 16)
    exact = exact_obc_mpo(L, 1.0, gamma)
    compressed = soe_obc_mpo(L, 1.0, gamma, small_fit)
    exact_dense = convert(TensorMap, exact)
    compressed_dense = convert(TensorMap, compressed)
    @test norm(exact_dense - compressed_dense) / norm(exact_dense) <= 1.0e-6
    dense_matrix = Matrix(block(exact_dense, Trivial()))
    reference_matrix = Matrix(ed_hamiltonian(obc_coupling_matrix(L, 1.0), gamma))
    @test dense_matrix ≈ reference_matrix atol = 1.0e-12
end

@testset "ED split-operator convention and conserved parity" begin
    Random.seed!(86)
    L = 5
    gamma = 0.8
    dt = 1.0e-3
    couplings = obc_coupling_matrix(L, 1.0)
    H = ed_hamiltonian(couplings, gamma)
    diagonal = ed_zz_diagonal(couplings)
    state = normalize!(randn(ComplexF64, 1 << L))
    split_state = copy(state)
    ed_split_step!(split_state, diagonal, gamma, dt, L)
    exact_state = exp(-1im * dt * Matrix(H)) * state
    @test norm(split_state - exact_state) <= 1.0e-8
    @test norm(split_state) ≈ 1.0 atol = 1.0e-13

    plus_x = fill(ComplexF64(inv(sqrt(1 << L))), 1 << L)
    initial_parity = parity_expectation(plus_x, L)
    ed_split_step!(plus_x, diagonal, gamma, 0.05, L)
    @test initial_parity ≈ 1.0 atol = 1.0e-13
    @test parity_expectation(plus_x, L) ≈ initial_parity atol = 1.0e-12

    ferromagnet = zeros(ComplexF64, 1 << L)
    ferromagnet[1] = 1
    correlations = ed_correlations(ferromagnet, L)
    @test correlations == ones(L - 1)
    @test kink_density_from_zz([correlations[1]]) == 0.0
    local_correlations = ed_nearest_neighbor_correlations(ferromagnet, L)
    @test local_correlations == ones(L - 1)
    @test bulk_kink_density_from_zz(local_correlations) == 0.0
end

@testset "ED trajectory artifacts and idempotent resume" begin
    mktempdir() do directory
        L = 4
        cell = make_cell(
            sigma = 1.0, L = L, gamma_c = 1.0, T = 0.1,
            chi = 16, dt = 0.05, poles = 16, seed = 86,
        )
        plus_x = fill(ComplexF64(inv(sqrt(1 << L))), 1 << L)
        result = run_ed_trajectory(
            cell, directory; initial_state = plus_x, checkpoint_fraction = 0.5,
        )
        @test result["status"] == "completed"
        @test isfile(joinpath(directory, "trajectory.csv"))
        @test isfile(joinpath(directory, "manifest.json"))
        @test isfile(joinpath(directory, "checkpoint.bin"))
        @test haskey(result, "final_kink_density_bulk")
        @test result["norm_drift"] <= 1.0e-12
        @test result["parity_drift"] <= 1.0e-12

        before = readlines(joinpath(directory, "trajectory.csv"))
        rm(joinpath(directory, "manifest.json"))
        recovered = run_ed_trajectory(
            cell, directory; initial_state = plus_x, checkpoint_fraction = 0.5,
        )
        @test recovered["status"] == "completed"
        @test readlines(joinpath(directory, "trajectory.csv")) == before

        resumed = run_ed_trajectory(
            cell, directory; initial_state = plus_x, checkpoint_fraction = 0.5,
        )
        @test resumed["status"] == "completed"
        @test readlines(joinpath(directory, "trajectory.csv")) == before
    end
end

@testset "MPS observables and TDVP trajectory artifacts" begin
    mktempdir() do directory
        Random.seed!(86)
        L = 4
        state = FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^4)
        correlations = mps_correlations(state)
        @test length(correlations) == L - 1
        @test all(value -> -1 <= value <= 1, correlations)
        @test abs(mps_parity(state)) <= 1 + 1.0e-12

        cell = make_cell(
            sigma = 1.0, L = L, gamma_c = 1.0, T = 0.1,
            chi = 16, dt = 0.05, poles = 16, seed = 86,
        )
        result = run_tdvp_trajectory(
            cell, directory; initial_state = state, checkpoint_fraction = 0.5,
        )
        @test result["status"] == "completed"
        @test result["method"] == "MPSKit-TDVP"
        @test isfile(joinpath(directory, "trajectory.csv"))
        @test isfile(joinpath(directory, "manifest.json"))
        @test haskey(result, "final_kink_density_bulk")
        @test result["norm_drift"] <= 1.0e-8
    end
end

@testset "Conventions and fixed cell interface" begin
    @test tau_q(32) == 16.0
    @test gamma_at_time(0, 32, 1.4) == 2.8
    @test gamma_at_time(32, 32, 1.4) == 0.0
    @test_throws ArgumentError tau_q(0)
    @test_throws DomainError gamma_at_time(33, 32, 1.4)

    @test obc_coupling(1.0, 1) == 1.0
    @test obc_coupling(1.0, 2) == 0.25
    J = obc_coupling_matrix(4, 1.0)
    @test J[1, 4] ≈ 1 / 9
    @test J[1, 2] == J[3, 4] == 1.0
    @test all(iszero, diag(J))

    @test kink_density_from_zz([1.0, -1.0, 1.0]) ≈ 1 / 3
    @test kink_density_from_zz(ones(5)) == 0.0
    edge_walls = ones(10)
    edge_walls[[1, 10]] .= -1
    @test kink_density_from_zz(edge_walls) ≈ 0.2
    @test bulk_kink_density_from_zz(edge_walls) == 0.0
    @test bulk_kink_density_from_zz(-ones(10)) == 1.0
    @test_throws ArgumentError bulk_kink_density_from_zz(
        ones(10); boundary_fraction = 0.5,
    )
    @test correlation_kink_estimator(2.0) ≈ (1 - exp(-0.5)) / 2

    cell = make_cell(
        sigma = 1.0, L = 64, gamma_c = 1.3, T = 16.0,
        chi = 96, dt = 0.05, poles = 16, seed = 86,
    )
    @test sort!(collect(keys(cell))) ==
          sort!(["sigma", "L", "Gamma_c", "T", "tau_Q", "chi", "dt", "poles", "seed"])
    @test cell["tau_Q"] == 8.0
    @test parameter_hash(cell) == parameter_hash(Dict(reverse(collect(cell))))
    modified = copy(cell)
    modified["chi"] = 128
    @test parameter_hash(cell) != parameter_hash(modified)
end

@testset "Atomic checkpoint and append-safe trajectory" begin
    mktempdir() do directory
        checkpoint = joinpath(directory, "checkpoint.bin")
        payload = Dict("step" => 20, "time" => 1.0, "state" => ComplexF64[1, 2im])
        atomic_checkpoint(checkpoint, payload)
        @test isfile(checkpoint)
        @test load_checkpoint(checkpoint) == payload
        @test isempty(filter(name -> occursin(".tmp", name), readdir(directory)))

        trajectory = joinpath(directory, "trajectory.csv")
        append_trajectory_row(
            trajectory;
            time = 0.0, gamma = 2.0, kink_density = 0.0,
            kink_density_bulk = 0.0,
            correlations = [1.0, 0.5], norm = 1.0, max_bond = 4,
        )
        append_trajectory_row(
            trajectory;
            time = 1.0, gamma = 1.0, kink_density = 0.1,
            kink_density_bulk = 0.05,
            correlations = [0.8, 0.4], norm = 1.0, max_bond = 8,
        )
        lines = readlines(trajectory)
        @test length(lines) == 3
        @test lines[1] ==
              "time,Gamma,kink_density,kink_density_bulk,C(r),norm,max_bond"
    end
end

@testset "Pre-registered KZM analysis gates" begin
    T = Float64[4, 8, 16, 32, 64, 128]
    exact_curve = 0.24 .* T .^ (-0.5)
    fit = fit_power_law(T, exact_curve; indices = 2:5)
    @test fit["mu"] ≈ 0.5 atol = 1.0e-12
    @test fit["prefactor"] ≈ 0.24 atol = 1.0e-12
    @test fit["max_relative_residual"] < 1.0e-12

    g1 = evaluate_g1(
        Dict("final_kink_ed" => 0.2, "final_kink_tdvp" => 0.201,
             "trajectory_ed" => [0.0, 0.1, 0.2],
             "trajectory_tdvp" => [0.0, 0.102, 0.201])
    )
    @test g1["passed"]

    g2 = evaluate_g2(T, exact_curve; exact_prefactor = 0.24)
    @test g2["passed"]
    @test g2["fit_indices"] == [2, 3, 4, 5]

    g3 = evaluate_g3(T, exact_curve; paper_mu = 0.50)
    @test g3["passed"]
    @test g3["acceptance"] == "relaxed"

    bad_curve = copy(exact_curve)
    bad_curve[4] *= 1.5
    @test !evaluate_g3(T, bad_curve; paper_mu = 0.50)["passed"]
end

@testset "Critical-field refinement and correlation estimator" begin
    evaluator = gamma -> 2.0 - (gamma - 1.231)^2
    scan = locate_critical_field(evaluator, 1.23; initial_half_width = 0.03)
    @test length(scan["coarse"]) == 7
    @test length(scan["fine"]) == 5
    @test scan["Gamma_high"] - scan["Gamma_low"] <= 0.01 + 10eps()
    @test scan["Gamma_low"] <= 1.231 <= scan["Gamma_high"]
    tight_scan = locate_critical_field(
        evaluator, 1.23;
        initial_half_width = 0.03,
        target_width = 0.002,
    )
    @test tight_scan["Gamma_high"] - tight_scan["Gamma_low"] <=
          0.002 + 10eps()
    @test length(tight_scan["refinements"]) >= 2

    xi = 3.2
    correlations = exp.(-collect(1:12) ./ xi)
    estimate = fit_correlation_length(correlations; distances = 2:10)
    @test estimate["xi"] ≈ xi atol = 1.0e-12
    @test estimate["n_corr"] ≈ correlation_kink_estimator(xi) atol = 1.0e-12
end

@testset "Finite-time collapse gate" begin
    mu = 0.62
    rows = Dict{String, Any}[]
    for L in (32, 64, 128), T in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
        x = (T / 2) / L^(1 / mu)
        y = 0.3 / (1 + x^0.7)
        push!(rows, Dict(
            "L" => L, "T" => T, "tau_Q" => T / 2,
            "kink_density" => y / L,
        ))
    end
    collapse = fit_collapse_mu(rows; mu_grid = 0.50:0.002:0.75)
    @test collapse["mu"] ≈ mu atol = 0.02
    @test evaluate_collapse(collapse, mu + 0.03)["passed"]
    @test !evaluate_collapse(collapse, mu + 0.08)["passed"]
end

@testset "Sprint mu curve, systematic envelope, and theory shape" begin
    rows = Dict{String, Any}[]
    expected_mu = Dict(1.75 => 0.54, 1.8 => 0.56, 1.875 => 0.59,
                       1.95 => 0.62, 2.0 => 0.64)
    for (sigma, mu) in expected_mu, T in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
        bulk = 0.2 * T^(-mu)
        push!(rows, Dict(
            "sigma" => sigma,
            "L" => 64,
            "T" => T,
            "variant" => "baseline",
            "kink_density_bulk" => bulk,
            "kink_density" => 0.2 * T^(-(mu + 0.005)),
            "correlation_kink_density" => 0.2 * T^(-(mu - 0.005)),
        ))
    end
    for variant in ("chi", "dt", "poles", "seed", "gamma_low", "gamma_high"),
            T in (16.0, 64.0)
        delta_mu = variant == "chi" ? 0.01 : -0.005
        value = 0.2 * T^(-expected_mu[1.875]) * (T / 16)^(-delta_mu)
        push!(rows, Dict(
            "sigma" => 1.875,
            "L" => 64,
            "T" => T,
            "variant" => variant,
            "kink_density_bulk" => value,
            "kink_density" => value,
            "correlation_kink_density" => value,
        ))
    end

    curve = fit_sprint_mu_curve(rows)
    @test curve["1.8"]["mu"] ≈ expected_mu[1.8] atol = 1.0e-12
    @test curve["1.875"]["fit_passed"]
    @test curve["1.95"]["endpoint_mu_delta"] <= 1.0e-12

    envelope = sprint_systematic_envelope(
        rows, 1.875; collapse_delta = 0.02,
    )
    @test envelope["systematic_mu_delta"] ≈ 0.02 atol = 1.0e-12
    @test envelope["passed"]
    @test envelope["gamma_mu_delta"] <= 0.01 + 1.0e-12

    observed = Dict(
        string(sigma) => Dict(
            "mu" => expected_mu[sigma],
            "total_error" => 0.005,
        )
        for sigma in (1.8, 1.875, 1.95)
    )
    theories = Dict(
        "continuous" => Dict(
            "1.8" => 0.56, "1.875" => 0.59, "1.95" => 0.62,
        ),
        "plateau" => Dict(
            "1.8" => 0.50, "1.875" => 0.50, "1.95" => 0.50,
        ),
    )
    comparison = compare_theory_shapes(observed, theories)
    @test comparison["status"] == "favored"
    @test comparison["favored"] == "continuous"
    @test all(
        winner == "continuous" for winner in values(comparison["winners"])
    )

    wide_observed = deepcopy(observed)
    for result in values(wide_observed)
        result["total_error"] = 0.1
    end
    @test compare_theory_shapes(wide_observed, theories)["status"] ==
          "inconclusive"
end

@testset "Stage expansion and escalation policy" begin
    g1 = stage_cells("G1"; gamma_c = 1.3)
    @test length(g1) == 6
    @test Set(cell["L"] for cell in g1) == Set([12, 16, 20])
    @test Set(cell["T"] for cell in g1) == Set([8.0, 32.0])
    size_resolved = stage_cells(
        "G1";
        gamma_c = Dict("1.0:12" => 1.21, "1.0:16" => 1.22, "1.0:20" => 1.23),
    )
    @test Set(
        (cell["L"], cell["Gamma_c"]) for cell in size_resolved
    ) == Set([(12, 1.21), (16, 1.22), (20, 1.23)])

    g2 = stage_cells("G2"; gamma_c = 1.0)
    @test length(g2) == 6
    @test all(cell["L"] == 256 && cell["chi"] == 64 for cell in g2)
    @test all(resource_class(cell) == "standard" for cell in g2)

    @test select_poles(8.0e-7, 16) == 16
    @test select_poles(2.0e-6, 16) == 24
    @test_throws ErrorException select_poles(2.0e-6, 24)

    @test resource_class(make_cell(
        sigma = 1, L = 64, gamma_c = 1, T = 16, chi = 96,
        dt = 0.05, poles = 16, seed = 1,
    )) == "standard"
    @test resource_class(make_cell(
        sigma = 1, L = 128, gamma_c = 1, T = 16, chi = 96,
        dt = 0.05, poles = 16, seed = 1,
    )) == "large"

    sprint_gamma = Dict{String, Any}(
        "1.0:12" => 1.2,
        "1.0:16" => 1.2,
        "1.0:20" => 1.2,
        "1.0:64" => 1.2,
    )
    for key in (
        "1.75:64", "1.8:64", "1.875:32", "1.875:64",
        "1.875:128", "1.95:64", "2.0:64",
    )
        sprint_gamma[key] = Dict(
            "Gamma_c" => 1.3,
            "Gamma_low" => 1.299,
            "Gamma_high" => 1.301,
        )
    end
    shard_a = stage_cells("sprint-A"; gamma_c = sprint_gamma)
    shard_b = stage_cells("sprint-B"; gamma_c = sprint_gamma)
    @test length(shard_a) == 49
    @test length(shard_b) == 48
    @test all(cell["shard_id"] == "A" for cell in shard_a)
    @test all(cell["shard_id"] == "B" for cell in shard_b)
    @test count(
        cell -> get(cell, "execution_mode", "") == "ed-tdvp", shard_a
    ) == 7
    @test count(
        cell -> get(cell, "component_stage", "") == "G2", shard_a
    ) == 6
    @test all(
        cell["Gamma_c"] == 1.0
        for cell in shard_a
        if get(cell, "component_stage", "") == "G2"
    )
    @test count(
        cell -> get(cell, "component_stage", "") == "sprint-size-1875",
        shard_a,
    ) == 14
    @test count(
        cell -> get(cell, "component_stage", "") == "sprint-trend", shard_b
    ) == 24
    @test isempty(
        intersect(
            Set(parameter_hash(cell) for cell in shard_a),
            Set(parameter_hash(cell) for cell in shard_b),
        ),
    )
    @test length(unique(parameter_hash(cell) for cell in shard_a)) == 49
    @test length(unique(parameter_hash(cell) for cell in shard_b)) == 48
end

@testset "Gamma map and independent shard campaign merge" begin
    scan(sigma, L; source = "source-sha", low = 1.299, high = 1.301) =
        Dict{String, Any}(
            "status" => "completed",
            "sigma" => sigma,
            "L" => L,
            "Gamma_c" => (low + high) / 2,
            "Gamma_low" => low,
            "Gamma_high" => high,
            "parameter_hash" => parameter_hash(Dict("sigma" => sigma, "L" => L)),
            "code_revision" => "commit",
            "code_source_sha256" => source,
            "parameters" => Dict("target_width" => 0.002),
        )
    gamma_map = build_gamma_map([
        scan(1.875, 64),
        scan(1.8, 64),
    ])
    @test gamma_map["kind"] == "track-c-gamma-map"
    @test gamma_map["source_sha256"] == "source-sha"
    @test gamma_map["entries"]["1.875:64"]["Gamma_c"] ≈ 1.3
    @test haskey(gamma_map, "map_hash")
    @test_throws ErrorException build_gamma_map([
        scan(1.875, 64),
        scan(1.8, 64; source = "different-source"),
    ])
    @test_throws ErrorException build_gamma_map([
        scan(1.875, 64; low = 1.29, high = 1.31),
    ])
    @test_throws ErrorException build_gamma_map([
        scan(1.875, 64),
        scan(1.875, 64),
    ])

    sprint_gamma = Dict{String, Any}(
        "1.0:12" => 1.2,
        "1.0:16" => 1.2,
        "1.0:20" => 1.2,
        "1.0:64" => 1.2,
    )
    for key in (
        "1.75:64", "1.8:64", "1.875:32", "1.875:64",
        "1.875:128", "1.95:64", "2.0:64",
    )
        sprint_gamma[key] = Dict(
            "Gamma_c" => 1.3,
            "Gamma_low" => 1.299,
            "Gamma_high" => 1.301,
        )
    end
    run_a = build_run_spec("sprint-A"; gamma_c = sprint_gamma)
    run_b = build_run_spec("sprint-B"; gamma_c = sprint_gamma)
    campaign = merge_campaign_runs(run_a, run_b)
    @test campaign["kind"] == "track-c-sprint-campaign"
    @test length(campaign["cells"]) == 97
    @test campaign["cell_counts"] == Dict("A" => 49, "B" => 48)

    wrong_source = deepcopy(run_b)
    wrong_source["code_source_sha256"] = "different"
    @test_throws ErrorException merge_campaign_runs(run_a, wrong_source)
    wrong_map = deepcopy(run_b)
    wrong_map["gamma_map_hash"] = "different"
    @test_throws ErrorException merge_campaign_runs(run_a, wrong_map)
    duplicate = deepcopy(run_b)
    push!(duplicate["cells"], deepcopy(first(run_a["cells"])))
    @test_throws ErrorException merge_campaign_runs(run_a, duplicate)
end

@testset "Critical scan shards are balanced and registered" begin
    estimates = Dict{String, Any}(
        "1.0:12" => 1.2,
        "1.0:16" => 1.2,
        "1.0:20" => 1.2,
        "1.0:64" => 1.2,
        "1.75:64" => 1.3,
        "1.8:64" => 1.3,
        "1.875:32" => 1.3,
        "1.875:64" => 1.3,
        "1.875:128" => 1.3,
        "1.95:64" => 1.3,
        "2.0:64" => 1.3,
    )
    critical_a = build_critical_shard_spec("A", estimates)
    critical_b = build_critical_shard_spec("B", estimates)
    @test length(critical_a["cells"]) == 7
    @test length(critical_b["cells"]) == 4
    @test all(cell["shard_id"] == "A" for cell in critical_a["cells"])
    @test all(cell["shard_id"] == "B" for cell in critical_b["cells"])
    @test count(
        cell -> cell["target_width"] == 0.002, critical_a["cells"]
    ) == 3
    @test all(cell["target_width"] == 0.002 for cell in critical_b["cells"])
    @test all(
        cell["initial_half_width"] == 0.2
        for cell in vcat(critical_a["cells"], critical_b["cells"])
    )
    @test length(unique(cell["id"] for cell in vcat(
        critical_a["cells"], critical_b["cells"],
    ))) == 11
    @test length(unique(cell["parameter_hash"] for cell in vcat(
        critical_a["cells"], critical_b["cells"],
    ))) == 11
end

@testset "run.json is the resumable source of truth" begin
    mktempdir() do directory
        path = joinpath(directory, "run.json")
        run = build_run_spec("G1"; gamma_c = 1.3, created_at = "2026-07-28T00:00:00")
        @test run["schema_version"] == 1
        @test run["physics"]["boundary"] == "open"
        @test run["physics"]["J"] == 1.0
        @test run["protocol"]["tau_Q_definition"] == "T/2"
        @test length(run["cells"]) == 6
        @test all(cell["status"] == "pending" for cell in run["cells"])
        @test length(unique(cell["id"] for cell in run["cells"])) == 6

        write_run_json(path, run)
        restored = read_run_json(path)
        @test restored["cells"][1]["parameters"]["tau_Q"] ==
              restored["cells"][1]["parameters"]["T"] / 2

        cell_id = restored["cells"][1]["id"]
        running = update_cell_status!(path, cell_id, "running")
        lease_token = running["lease"]["token"]
        update_cell_status!(
            path, cell_id, "completed";
            result = Dict("kink_density" => 0.123),
            lease_token,
        )
        updated = read_run_json(path)
        target = only(filter(cell -> cell["id"] == cell_id, updated["cells"]))
        @test target["status"] == "completed"
        @test target["result"]["kink_density"] == 0.123
        @test_throws ErrorException update_cell_status!(
            path, cell_id, "completed";
            result = Dict("kink_density" => 9.9),
            lease_token = "stale-token",
        )
        @test length(pending_cells(updated)) == 5

        updated["cells"][2]["status"] = "running"
        updated["cells"][2]["lease"] = Dict("owner" => "slurm:finished-job")
        @test length(pending_cells(updated)) == 5
    end
end

@testset "Stage summaries preserve registered gates" begin
    g2_run = build_run_spec("G2"; gamma_c = 1.0)
    for cell in g2_run["cells"]
        T = cell["parameters"]["T"]
        cell["status"] = "completed"
        cell["result"] = Dict(
            "tdvp" => Dict("final_kink_density" => inv(2pi) * T^(-0.5)),
            "numerical_gate" => Dict("tdvp" => Dict("passed" => true)),
        )
    end
    g2_summary = analyze_run(g2_run)
    @test g2_summary["status"] == "passed"
    @test g2_summary["gate"] == "G2"

    baseline = Dict(
        16.0 => Dict("final_kink_density" => 0.10),
        64.0 => Dict("final_kink_density" => 0.05),
    )
    sentinels = [
        Dict(
            "parameters" => Dict("T" => 16.0, "chi" => 64, "dt" => 0.05),
            "final_kink_density" => 0.101,
        ),
        Dict(
            "parameters" => Dict("T" => 64.0, "chi" => 64, "dt" => 0.05),
            "final_kink_density" => 0.052,
        ),
        Dict(
            "parameters" => Dict("T" => 16.0, "chi" => 96, "dt" => 0.025),
            "final_kink_density" => 0.10,
        ),
        Dict(
            "parameters" => Dict("T" => 64.0, "chi" => 96, "dt" => 0.025),
            "final_kink_density" => 0.05,
        ),
    ]
    convergence = evaluate_g3_sentinels(baseline, sentinels)
    @test convergence["upgrade_T"] == [64.0]
    @test isempty(convergence["dt_failed_T"])
    @test !convergence["passed"]

    g3_run = build_run_spec("G3"; gamma_c = 1.0)
    g3_run["stage"] = "G3-converged"
    for (index, cell) in enumerate(g3_run["cells"])
        T = cell["parameters"]["T"]
        cell["status"] = "completed"
        cell["result"] = Dict(
            "tdvp" => Dict("final_kink_density" => 0.16 * T^(-0.5)),
            "numerical_gate" => Dict(
                "tdvp" => Dict("passed" => index != 1),
            ),
        )
    end
    g3_run["fallback_validation"] = Dict("status" => "passed")
    g3_summary = analyze_run(g3_run)
    @test g3_summary["literature_gate_passed"]
    @test !g3_summary["numerical_gates_passed"]
    @test g3_summary["status"] == "failed"
end
