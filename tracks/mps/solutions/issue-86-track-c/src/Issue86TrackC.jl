module Issue86TrackC

using JSON
using Dates
using KrylovKit
using LinearAlgebra
using MPSKit
using Printf
using Random
using Serialization
using SHA
using Sockets
using SparseArrays
using Statistics
using TensorKit
using TensorKit: ℂ
using UUIDs

const TRACK_B_SOURCE = normpath(
    joinpath(@__DIR__, "..", "..", "issue-86", "src", "Issue86TrackB.jl")
)
include(TRACK_B_SOURCE)
using .Issue86TrackB: SOEApproximation, exact_mpo, fit_power_law_soe, pauli_x, pauli_z

"""
Independent implementation of the Track C minimum reproduction for Challenge #86.

The module deliberately imports no Track B run state. Shared numerical conventions
are copied through explicit, tested interfaces only.
"""

export append_trajectory_row
export analyze_sprint_campaign
export analyze_run
export atomic_checkpoint
export bulk_kink_density_from_zz
export build_critical_shard_spec
export build_gamma_map
export build_run_spec
export correlation_kink_estimator
export campaign_rows
export dmrg_entropy_point
export evaluate_g1
export evaluate_g2
export evaluate_g3
export evaluate_collapse
export evaluate_g3_sentinels
export execute_cell
export ed_correlations
export ed_nearest_neighbor_correlations
export ed_hamiltonian
export ed_ground_state
export ed_split_step!
export ed_zz_diagonal
export exact_obc_mpo
export fit_power_law
export fit_sprint_mu_curve
export fit_obc_soe
export fit_collapse_mu
export fit_correlation_length
export gamma_at_time
export kink_density_from_zz
export load_checkpoint
export locate_critical_field
export make_cell
export mps_correlations
export mps_nearest_neighbor_correlations
export mps_parity
export merge_campaign_runs
export compare_theory_shapes
export obc_coupling
export obc_coupling_matrix
export obc_soe_coupling
export parameter_hash
export parity_expectation
export pending_cells
export read_run_json
export run_ed_trajectory
export run_critical_scan
export run_g0
export run_tdvp_trajectory
export resource_class
export select_poles
export soe_obc_mpo
export sprint_systematic_envelope
export stage_cells
export tau_q
export update_cell_status!
export write_run_json

const CELL_KEYS = (
    "sigma", "L", "Gamma_c", "T", "tau_Q", "chi", "dt", "poles", "seed",
)
const DEFAULT_TIMES = Float64[4, 8, 16, 32, 64, 128]

function tau_q(T::Real)
    T > 0 || throw(ArgumentError("T must be positive"))
    return Float64(T) / 2
end

function gamma_at_time(t::Real, T::Real, gamma_c::Real)
    tau_q(T)
    0 <= t <= T || throw(DomainError(t, "time must lie in [0,T]"))
    gamma_c > 0 || throw(ArgumentError("Gamma_c must be positive"))
    return 2Float64(gamma_c) * (1 - Float64(t) / Float64(T))
end

function obc_coupling(sigma::Real, distance::Integer)
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    distance >= 1 || throw(ArgumentError("distance must be positive"))
    return Float64(distance)^(-(1 + Float64(sigma)))
end

function obc_coupling_matrix(L::Integer, sigma::Real)
    L >= 2 || throw(ArgumentError("L must be at least two"))
    couplings = zeros(Float64, L, L)
    for i in 1:(L - 1), j in (i + 1):L
        couplings[i, j] = couplings[j, i] = obc_coupling(sigma, j - i)
    end
    return couplings
end

function _obc_fit_candidate(
        L::Integer, sigma::Real, poles::Integer,
        xmin::Real, xmax::Real
    )
    alpha = 1 + Float64(sigma)
    distances = collect(1:(L - 1))
    exact = Float64.(distances) .^ (-alpha)
    xs = collect(range(Float64(xmin), Float64(xmax); length = poles))
    lambdas = exp.(-exp.(xs))
    basis = [lambda^(distance - 1) for distance in distances, lambda in lambdas]
    relative_basis = basis ./ exact
    amplitudes = relative_basis \ ones(length(exact))
    relative_errors = basis * amplitudes ./ exact .- 1
    return SOEApproximation(
        alpha, Int(poles), Int(L - 1), Float64(xmin), Float64(xmax),
        amplitudes, lambdas, maximum(abs, relative_errors),
        sqrt(mean(abs2, relative_errors)),
    )
end

"""
Fit the finite OBC distance set directly in relative error. The implementation
shares Track B's audited `SOEApproximation` representation while keeping the
open-boundary target and all Track C run state independent.
"""
function fit_obc_soe(L::Integer, sigma::Real, poles::Integer)
    L >= 3 || throw(ArgumentError("L must be at least three"))
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    poles in (16, 24) || throw(ArgumentError("Track C supports 16 or 24 poles"))
    best = nothing
    for xmin in -10.0:0.2:-1.0, xmax in -0.5:0.2:7.0
        xmax > xmin || continue
        candidate = _obc_fit_candidate(L, sigma, poles, xmin, xmax)
        if isnothing(best) ||
                candidate.max_relative_error < best.max_relative_error
            best = candidate
        end
    end
    return best::SOEApproximation
end

function obc_soe_coupling(approximation::SOEApproximation, distance::Integer)
    1 <= distance <= approximation.dmax ||
        throw(ArgumentError("distance lies outside the fitted OBC interval"))
    return sum(
        approximation.amplitudes .* approximation.lambdas .^ (distance - 1)
    )
end

function exact_obc_mpo(L::Integer, sigma::Real, gamma::Real)
    return exact_mpo(obc_coupling_matrix(L, sigma), gamma)
end

function soe_obc_mpo(
        L::Integer, sigma::Real, gamma::Real,
        approximation::SOEApproximation
    )
    isapprox(approximation.alpha, 1 + sigma; atol = 100eps(Float64)) ||
        throw(ArgumentError("SOE alpha does not match sigma"))
    L - 1 <= approximation.dmax ||
        throw(ArgumentError("SOE fit does not cover the requested chain"))
    levels = approximation.poles + 2
    finish = levels
    X = FiniteMPO(pauli_x())[1]
    Z = FiniteMPO(pauli_z())[1]
    matrices = Vector{Matrix{Any}}(undef, L)

    for site in 1:L
        W = Matrix{Any}(undef, levels, levels)
        fill!(W, missing)
        W[1, 1] = ComplexF64(1)
        W[finish, finish] = ComplexF64(1)
        W[1, finish] = -Float64(gamma) * X
        for pole in 1:approximation.poles
            channel = 1 + pole
            W[1, channel] = -approximation.amplitudes[pole] * Z
            W[channel, channel] = ComplexF64(approximation.lambdas[pole])
            W[channel, finish] = Z
        end
        matrices[site] =
            site == 1 ? W[1:1, :] :
            site == L ? W[:, finish:finish] : W
    end
    return FiniteMPOHamiltonian(matrices)
end

ed_hamiltonian(couplings::AbstractMatrix, gamma::Real) =
    Issue86TrackB.ed_hamiltonian(couplings, gamma)

function ed_zz_diagonal(couplings::AbstractMatrix)
    L = size(couplings, 1)
    size(couplings, 2) == L ||
        throw(DimensionMismatch("coupling matrix must be square"))
    dimension = 1 << L
    diagonal = zeros(Float64, dimension)
    Threads.@threads for basis in 0:(dimension - 1)
        energy = 0.0
        for i in 1:(L - 1), j in (i + 1):L
            zi = iszero(basis & (1 << (i - 1))) ? 1.0 : -1.0
            zj = iszero(basis & (1 << (j - 1))) ? 1.0 : -1.0
            energy -= couplings[i, j] * zi * zj
        end
        diagonal[basis + 1] = energy
    end
    return diagonal
end

function _apply_x_rotation!(
        state::AbstractVector{<:Complex}, gamma::Real, dt::Real, L::Integer
    )
    cosine = cos(Float64(gamma) * Float64(dt))
    sine = sin(Float64(gamma) * Float64(dt))
    for site in 0:(L - 1)
        mask = 1 << site
        Threads.@threads for basis in 0:(length(state) - 1)
            iszero(basis & mask) || continue
            partner = basis | mask
            left = state[basis + 1]
            right = state[partner + 1]
            state[basis + 1] = cosine * left + 1im * sine * right
            state[partner + 1] = 1im * sine * left + cosine * right
        end
    end
    return state
end

function ed_split_step!(
        state::AbstractVector{<:Complex}, zz_diagonal::AbstractVector,
        gamma::Real, dt::Real, L::Integer
    )
    length(state) == 1 << L ||
        throw(DimensionMismatch("state dimension does not match L"))
    length(zz_diagonal) == length(state) ||
        throw(DimensionMismatch("ZZ diagonal dimension does not match state"))
    state .*= exp.(-0.5im * Float64(dt) .* zz_diagonal)
    _apply_x_rotation!(state, gamma, dt, L)
    state .*= exp.(-0.5im * Float64(dt) .* zz_diagonal)
    return state
end

function parity_expectation(state::AbstractVector, L::Integer)
    length(state) == 1 << L ||
        throw(DimensionMismatch("state dimension does not match L"))
    complement = (1 << L) - 1
    partial = zeros(ComplexF64, Threads.maxthreadid())
    Threads.@threads for basis in 0:(length(state) - 1)
        partial[Threads.threadid()] +=
            conj(state[basis + 1]) * state[xor(basis, complement) + 1]
    end
    return real(sum(partial))
end

function ed_correlations(state::AbstractVector, L::Integer)
    length(state) == 1 << L ||
        throw(DimensionMismatch("state dimension does not match L"))
    probabilities = abs2.(state)
    partial = zeros(Float64, Threads.maxthreadid(), L - 1)
    Threads.@threads for basis in 0:(length(state) - 1)
        thread = Threads.threadid()
        probability = probabilities[basis + 1]
        for distance in 1:(L - 1)
            pair_sum = 0.0
            for i in 1:(L - distance)
                left = (basis >> (i - 1)) & 1
                right = (basis >> (i + distance - 1)) & 1
                pair_sum += left == right ? 1.0 : -1.0
            end
            partial[thread, distance] += probability * pair_sum
        end
    end
    correlations = vec(sum(partial; dims = 1))
    for distance in 1:(L - 1)
        correlations[distance] /= L - distance
    end
    return correlations
end

function ed_nearest_neighbor_correlations(state::AbstractVector, L::Integer)
    length(state) == 1 << L ||
        throw(DimensionMismatch("state dimension does not match L"))
    probabilities = abs2.(state)
    partial = zeros(Float64, Threads.maxthreadid(), L - 1)
    Threads.@threads for basis in 0:(length(state) - 1)
        thread = Threads.threadid()
        probability = probabilities[basis + 1]
        for site in 1:(L - 1)
            left = (basis >> (site - 1)) & 1
            right = (basis >> site) & 1
            partial[thread, site] += probability * (left == right ? 1.0 : -1.0)
        end
    end
    return vec(sum(partial; dims = 1))
end

function _ed_action(
        state::AbstractVector, zz_diagonal::AbstractVector,
        gamma::Real, L::Integer
    )
    result = Vector{ComplexF64}(undef, length(state))
    Threads.@threads for basis in 0:(length(state) - 1)
        value = zz_diagonal[basis + 1] * state[basis + 1]
        for site in 0:(L - 1)
            value -= gamma * state[xor(basis, 1 << site) + 1]
        end
        result[basis + 1] = value
    end
    return result
end

function ed_ground_state(
        couplings::AbstractMatrix, gamma::Real;
        tolerance::Real = 1.0e-10, seed::Integer = 86
    )
    L = size(couplings, 1)
    if L <= 16
        values, vectors, _ = Issue86TrackB.ed_lowest(
            ed_hamiltonian(couplings, gamma); number = 1, tolerance
        )
        return Float64(first(values)), ComplexF64.(first(vectors))
    end
    L <= 20 || throw(ArgumentError("full-Hilbert-space ED is restricted to L <= 20"))
    Random.seed!(seed)
    diagonal = ed_zz_diagonal(couplings)
    initial = normalize!(randn(ComplexF64, 1 << L))
    operator = state -> _ed_action(state, diagonal, gamma, L)
    values, vectors, _ = eigsolve(
        operator, initial, 1, :SR;
        ishermitian = true, tol = tolerance, krylovdim = 30, maxiter = 500,
    )
    return real(first(values)), ComplexF64.(first(vectors))
end

function _cell_couplings(cell::AbstractDict)
    L = Int(cell["L"])
    sigma = cell["sigma"]
    if sigma == "NN"
        couplings = zeros(Float64, L, L)
        for site in 1:(L - 1)
            couplings[site, site + 1] = couplings[site + 1, site] = 1.0
        end
        return couplings
    end
    return obc_coupling_matrix(L, Float64(sigma))
end

function _write_manifest(path::AbstractString, manifest::AbstractDict)
    write_run_json(path, manifest)
    return manifest
end

function _resource_manifest()
    return Dict{String, Any}(
        "hostname" => get(ENV, "HOSTNAME", gethostname()),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", nothing),
        "slurm_array_task_id" => get(ENV, "SLURM_ARRAY_TASK_ID", nothing),
        "julia_threads" => Threads.nthreads(),
    )
end

function _trajectory_kink_values(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 2 || return Float64[]
    return Float64[parse(Float64, split(line, ',')[3]) for line in lines[2:end]]
end

function _correlation_length_summary(correlations::AbstractVector)
    length(correlations) >= 2 || return nothing
    first_distance = length(correlations) >= 3 ? 2 : 1
    last_distance = min(10, length(correlations))
    try
        return fit_correlation_length(
            correlations; distances = first_distance:last_distance
        )
    catch
        return nothing
    end
end

"""
Run the full-Hilbert-space split-operator reference trajectory.

Rows and checkpoints are emitted at the same approximately 5% cadence. A
completed manifest is idempotent; an interrupted run resumes from its atomic
checkpoint without duplicating trajectory rows.
"""
function run_ed_trajectory(
        cell::AbstractDict, output_directory::AbstractString;
        initial_state = nothing,
        checkpoint_fraction::Real = 0.05,
    )
    0 < checkpoint_fraction <= 1 ||
        throw(ArgumentError("checkpoint_fraction must lie in (0,1]"))
    mkpath(output_directory)
    manifest_path = joinpath(output_directory, "manifest.json")
    trajectory_path = joinpath(output_directory, "trajectory.csv")
    checkpoint_path = joinpath(output_directory, "checkpoint.bin")
    expected_hash = parameter_hash(cell)

    if isfile(manifest_path)
        manifest = JSON.parsefile(manifest_path)
        manifest["parameter_hash"] == expected_hash ||
            error("existing manifest parameter hash does not match requested cell")
        manifest["status"] == "completed" && return manifest
    end

    L = Int(cell["L"])
    L <= 20 || throw(ArgumentError("ED trajectory is restricted to L <= 20"))
    T = Float64(cell["T"])
    requested_dt = Float64(cell["dt"])
    steps = round(Int, T / requested_dt)
    steps >= 1 || throw(ArgumentError("trajectory needs at least one time step"))
    isapprox(steps * requested_dt, T; atol = 100eps(T), rtol = 0) ||
        throw(ArgumentError("T must be an integer multiple of dt"))
    dt = T / steps
    checkpoint_every = max(1, ceil(Int, checkpoint_fraction * steps))
    couplings = _cell_couplings(cell)
    diagonal = ed_zz_diagonal(couplings)
    started = time()

    if isfile(checkpoint_path)
        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint["parameter_hash"] == expected_hash ||
            error("checkpoint parameter hash does not match requested cell")
        state = checkpoint["state"]
        start_step = Int(checkpoint["step"])
        initial_norm = Float64(checkpoint["initial_norm"])
        initial_parity = Float64(checkpoint["initial_parity"])
        max_norm_drift = Float64(get(checkpoint, "max_norm_drift", 0.0))
        max_parity_drift = Float64(get(checkpoint, "max_parity_drift", 0.0))
    else
        isfile(trajectory_path) &&
            error("trajectory exists without a checkpoint; preserve it and choose a new output directory")
        if isnothing(initial_state)
            _, state = ed_ground_state(
                couplings, 2Float64(cell["Gamma_c"]);
                tolerance = 1.0e-10, seed = Int(cell["seed"]),
            )
        else
            state = normalize!(ComplexF64.(copy(initial_state)))
        end
        start_step = 0
        initial_norm = real(dot(state, state))
        initial_parity = parity_expectation(state, L)
        max_norm_drift = 0.0
        max_parity_drift = 0.0
        correlations = ed_correlations(state, L)
        nearest = ed_nearest_neighbor_correlations(state, L)
        append_trajectory_row(
            trajectory_path;
            time = 0.0,
            gamma = gamma_at_time(0.0, T, cell["Gamma_c"]),
            kink_density = kink_density_from_zz(nearest),
            kink_density_bulk = bulk_kink_density_from_zz(nearest),
            correlations,
            norm = initial_norm,
            max_bond = 0,
        )
        atomic_checkpoint(checkpoint_path, Dict{String, Any}(
            "parameter_hash" => expected_hash,
            "step" => 0,
            "state" => state,
            "initial_norm" => initial_norm,
            "initial_parity" => initial_parity,
            "max_norm_drift" => max_norm_drift,
            "max_parity_drift" => max_parity_drift,
        ))
    end

    final_correlations = ed_correlations(state, L)
    for step in (start_step + 1):steps
        midpoint = (step - 0.5) * dt
        gamma_midpoint = gamma_at_time(midpoint, T, cell["Gamma_c"])
        ed_split_step!(state, diagonal, gamma_midpoint, dt, L)
        if step % checkpoint_every == 0 || step == steps
            current_time = step * dt
            final_correlations = ed_correlations(state, L)
            nearest = ed_nearest_neighbor_correlations(state, L)
            current_norm = real(dot(state, state))
            current_parity = parity_expectation(state, L)
            max_norm_drift = max(max_norm_drift, abs(current_norm - initial_norm))
            max_parity_drift =
                max(max_parity_drift, abs(current_parity - initial_parity))
            append_trajectory_row(
                trajectory_path;
                time = current_time,
                gamma = gamma_at_time(current_time, T, cell["Gamma_c"]),
                kink_density = kink_density_from_zz(nearest),
                kink_density_bulk = bulk_kink_density_from_zz(nearest),
                correlations = final_correlations,
                norm = current_norm,
                max_bond = 0,
            )
            atomic_checkpoint(checkpoint_path, Dict{String, Any}(
                "parameter_hash" => expected_hash,
                "step" => step,
                "state" => state,
                "initial_norm" => initial_norm,
                "initial_parity" => initial_parity,
                "max_norm_drift" => max_norm_drift,
                "max_parity_drift" => max_parity_drift,
            ))
            @printf(
                "ED Track C L=%d T=%.6g step=%d/%d kink=%.8g\n",
                L, T, step, steps, kink_density_from_zz(nearest),
            )
            flush(stdout)
        end
    end

    final_norm = real(dot(state, state))
    final_parity = parity_expectation(state, L)
    final_nearest = ed_nearest_neighbor_correlations(state, L)
    final_kink = kink_density_from_zz(final_nearest)
    final_bulk_kink = bulk_kink_density_from_zz(final_nearest)
    correlation_summary = _correlation_length_summary(final_correlations)
    manifest = Dict{String, Any}(
        "status" => "completed",
        "method" => "ED-Strang-split",
        "parameter_hash" => expected_hash,
        "parameters" => cell,
        "code_revision" => _git_revision(),
        "code_source_sha256" => _source_hash(),
        "completed_at" => _utc_timestamp(),
        "resources" => _resource_manifest(),
        "runtime_seconds" => time() - started,
        "final_kink_density" => final_kink,
        "final_kink_density_bulk" => final_bulk_kink,
        "boundary_fraction" => 0.1,
        "correlation_length" =>
            isnothing(correlation_summary) ? nothing : correlation_summary["xi"],
        "correlation_kink_density" =>
            isnothing(correlation_summary) ? nothing : correlation_summary["n_corr"],
        "checkpoint_kink_density" => _trajectory_kink_values(trajectory_path),
        "final_correlations" => final_correlations,
        "norm_drift" => max(max_norm_drift, abs(final_norm - initial_norm)),
        "parity_initial" => initial_parity,
        "parity_final" => final_parity,
        "parity_drift" =>
            max(max_parity_drift, abs(final_parity - initial_parity)),
        "checkpoint_every_steps" => checkpoint_every,
    )
    _write_manifest(manifest_path, manifest)
    return manifest
end

function mps_correlations(
        state::FiniteMPS;
        max_distance::Integer = length(state) - 1,
    )
    L = length(state)
    1 <= max_distance <= L - 1 ||
        throw(ArgumentError("max_distance must lie in 1:L-1"))
    ZZ = pauli_z() ⊗ pauli_z()
    correlations = zeros(Float64, max_distance)
    counts = zeros(Int, max_distance)
    for site in 1:(L - 1)
        last_site = min(L, site + max_distance)
        values = correlator(state, ZZ, site, (site + 1):last_site)
        for (offset, value) in enumerate(values)
            correlations[offset] += real(value)
            counts[offset] += 1
        end
    end
    correlations ./= counts
    return correlations
end

function mps_nearest_neighbor_correlations(state::FiniteMPS)
    L = length(state)
    ZZ = pauli_z() ⊗ pauli_z()
    return Float64[
        real(only(correlator(state, ZZ, site, (site + 1):(site + 1))))
        for site in 1:(L - 1)
    ]
end

function mps_parity(state::FiniteMPS)
    local_x = FiniteMPO(pauli_x())[1]
    parity_operator = FiniteMPO(fill(local_x, length(state)))
    return real(expectation_value(state, parity_operator))
end

function run_g0(
        output_path::AbstractString;
        L::Integer = 4,
        audit_L::Integer = 256,
        sigma::Real = 1.0,
        gamma::Real = 0.73,
    )
    audit_L >= L || throw(ArgumentError("audit_L must cover the small-chain check"))
    audit_approximation = fit_obc_soe(audit_L, sigma, 16)
    effective_poles = select_poles(
        audit_approximation.max_relative_error, 16
    )
    if effective_poles == 24
        audit_approximation = fit_obc_soe(audit_L, sigma, 24)
        select_poles(audit_approximation.max_relative_error, 24)
    end
    approximation = fit_obc_soe(L, sigma, effective_poles)
    exact = exact_obc_mpo(L, sigma, gamma)
    compressed = soe_obc_mpo(L, sigma, gamma, approximation)
    exact_tensor = convert(TensorMap, exact)
    compressed_tensor = convert(TensorMap, compressed)
    dense_mpo = Matrix(block(exact_tensor, Trivial()))
    dense_reference = Matrix(
        ed_hamiltonian(obc_coupling_matrix(L, sigma), gamma)
    )
    mpo_relative_error =
        norm(exact_tensor - compressed_tensor) / norm(exact_tensor)
    dense_relative_error =
        norm(dense_mpo - dense_reference) / norm(dense_reference)
    report = Dict{String, Any}(
        "gate" => "G0",
        "status" =>
            audit_approximation.max_relative_error <= 1.0e-6 &&
            mpo_relative_error <= 1.0e-6 &&
            dense_relative_error <= 1.0e-12 ? "passed" : "failed",
        "L" => Int(L),
        "coupling_audit_L" => Int(audit_L),
        "sigma" => Float64(sigma),
        "Gamma" => Float64(gamma),
        "effective_poles" => effective_poles,
        "coupling_max_relative_error" =>
            audit_approximation.max_relative_error,
        "soe_vs_exact_mpo_relative_error" => mpo_relative_error,
        "exact_mpo_vs_dense_relative_error" => dense_relative_error,
        "boundary" => "open",
        "pauli_eigenvalues" => [-1, 1],
        "code_revision" => _git_revision(),
        "code_source_sha256" => _source_hash(),
        "completed_at" => _utc_timestamp(),
    )
    _write_manifest(output_path, report)
    report["status"] == "passed" || error("G0 consistency gate failed")
    return report
end

function dmrg_entropy_point(
        sigma::Real, L::Integer, gamma::Real, chi::Integer,
        approximation::SOEApproximation;
        seed::Integer = 86,
        initial_state = nothing,
    )
    Random.seed!(seed)
    trial = isnothing(initial_state) ?
        FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^min(8, chi)) :
        deepcopy(initial_state)
    H = soe_obc_mpo(L, sigma, gamma, approximation)
    algorithm = DMRG2(;
        tol = 1.0e-10,
        maxiter = 50,
        verbosity = 1,
        trscheme = truncrank(chi),
    )
    state, environments, residual = find_groundstate(trial, H, algorithm)
    return Dict{String, Any}(
        "Gamma" => Float64(gamma),
        "half_chain_entropy" => entropy(state, fld(L, 2)),
        "energy" => real(expectation_value(state, H, environments)),
        "residual" => real(residual),
        "max_bond_dimension" => _max_bond_dimension(state),
    ), state
end

function run_critical_scan(
        output_path::AbstractString;
        sigma::Real,
        L::Integer,
        initial_estimate::Real,
        chi::Integer = 96,
        poles::Integer = 16,
        seed::Integer = 86,
        initial_half_width::Real = 0.03,
        target_width::Real = 0.01,
    )
    if L >= 32 && !haskey(ENV, "SLURM_JOB_ID") &&
            get(ENV, "TRACK_C_ALLOW_LOCAL", "0") != "1"
        error(
            "critical scan L>=32 requires Slurm; set TRACK_C_ALLOW_LOCAL=1 only for an intentional local run"
        )
    end
    scan_parameters = Dict{String, Any}(
        "sigma" => Float64(sigma),
        "L" => Int(L),
        "initial_estimate" => Float64(initial_estimate),
        "chi" => Int(chi),
        "poles" => Int(poles),
        "seed" => Int(seed),
        "initial_half_width" => Float64(initial_half_width),
        "target_width" => Float64(target_width),
    )
    scan_hash = parameter_hash(scan_parameters)
    checkpoint_path = output_path * ".checkpoint.bin"
    if isfile(output_path)
        previous = JSON.parsefile(output_path)
        if get(previous, "status", "") == "completed"
            previous["parameter_hash"] == scan_hash ||
                error("completed critical scan parameters do not match")
            return previous
        end
    end
    approximation = fit_obc_soe(L, sigma, poles)
    effective_poles = select_poles(approximation.max_relative_error, poles)
    if effective_poles != poles
        approximation = fit_obc_soe(L, sigma, effective_poles)
        select_poles(approximation.max_relative_error, effective_poles)
    end
    if isfile(checkpoint_path)
        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint["parameter_hash"] == scan_hash ||
            error("critical-scan checkpoint parameters do not match")
        cache = checkpoint["cache"]
        state = checkpoint["state"]
    else
        cache = Dict{Float64, Dict{String, Any}}()
        state = nothing
    end
    evaluator = function (gamma)
        key = Float64(gamma)
        haskey(cache, key) && return cache[key]["half_chain_entropy"]
        point, state_next = dmrg_entropy_point(
            sigma, L, key, chi, approximation;
            seed, initial_state = state,
        )
        state = state_next
        cache[key] = point
        atomic_checkpoint(checkpoint_path, Dict{String, Any}(
            "parameter_hash" => scan_hash,
            "cache" => cache,
            "state" => state,
        ))
        partial = Dict{String, Any}(
            "status" => "running",
            "parameter_hash" => scan_hash,
            "parameters" => scan_parameters,
            "sigma" => Float64(sigma),
            "L" => Int(L),
            "initial_estimate" => Float64(initial_estimate),
            "effective_poles" => effective_poles,
            "mpo_max_relative_error" => approximation.max_relative_error,
            "points" => sort!(collect(values(cache)); by = row -> row["Gamma"]),
        )
        _write_manifest(output_path, partial)
        @printf(
            "Gamma_c scan sigma=%.6g L=%d Gamma=%.8g S_half=%.10g\n",
            sigma, L, key, point["half_chain_entropy"],
        )
        flush(stdout)
        return point["half_chain_entropy"]
    end
    scan = locate_critical_field(
        evaluator, initial_estimate; initial_half_width, target_width
    )
    scan["status"] = "completed"
    scan["parameter_hash"] = scan_hash
    scan["parameters"] = scan_parameters
    scan["sigma"] = Float64(sigma)
    scan["L"] = Int(L)
    scan["chi"] = Int(chi)
    scan["effective_poles"] = effective_poles
    scan["mpo_max_relative_error"] = approximation.max_relative_error
    scan["points"] = sort!(collect(values(cache)); by = row -> row["Gamma"])
    scan["code_revision"] = _git_revision()
    scan["code_source_sha256"] = _source_hash()
    scan["completed_at"] = _utc_timestamp()
    _write_manifest(output_path, scan)
    return scan
end

function _max_bond_dimension(state::FiniteMPS)
    return maximum(dim(left_virtualspace(state, site)) for site in 1:length(state))
end

function _cell_mpo(cell::AbstractDict, gamma::Real, approximation)
    L = Int(cell["L"])
    if cell["sigma"] == "NN"
        return exact_mpo(_cell_couplings(cell), gamma)
    end
    return soe_obc_mpo(L, Float64(cell["sigma"]), gamma, approximation)
end

function _track_c_approximation(cell::AbstractDict)
    cell["sigma"] == "NN" && return nothing, 0.0, 1
    L = Int(cell["L"])
    sigma = Float64(cell["sigma"])
    requested_poles = Int(cell["poles"])
    requested_poles in (16, 24) ||
        throw(ArgumentError("long-range Track C cells require 16 or 24 poles"))
    approximation = fit_obc_soe(L, sigma, requested_poles)
    effective_poles = select_poles(
        approximation.max_relative_error, requested_poles
    )
    if effective_poles != requested_poles
        approximation = fit_obc_soe(L, sigma, effective_poles)
        select_poles(approximation.max_relative_error, effective_poles)
    end
    return approximation, approximation.max_relative_error, effective_poles
end

"""
Run midpoint-Hamiltonian TDVP2, switching to one-site TDVP after the requested
bond cap has remained saturated for three consecutive checkpoints.
"""
function run_tdvp_trajectory(
        cell::AbstractDict, output_directory::AbstractString;
        initial_state = nothing,
        checkpoint_fraction::Real = 0.05,
    )
    0 < checkpoint_fraction <= 1 ||
        throw(ArgumentError("checkpoint_fraction must lie in (0,1]"))
    mkpath(output_directory)
    manifest_path = joinpath(output_directory, "manifest.json")
    trajectory_path = joinpath(output_directory, "trajectory.csv")
    checkpoint_path = joinpath(output_directory, "checkpoint.bin")
    expected_hash = parameter_hash(cell)

    if isfile(manifest_path)
        manifest = JSON.parsefile(manifest_path)
        manifest["parameter_hash"] == expected_hash ||
            error("existing manifest parameter hash does not match requested cell")
        manifest["status"] == "completed" && return manifest
    end

    L = Int(cell["L"])
    chi = Int(cell["chi"])
    T = Float64(cell["T"])
    requested_dt = Float64(cell["dt"])
    steps = round(Int, T / requested_dt)
    steps >= 1 || throw(ArgumentError("trajectory needs at least one time step"))
    isapprox(steps * requested_dt, T; atol = 100eps(T), rtol = 0) ||
        throw(ArgumentError("T must be an integer multiple of dt"))
    dt = T / steps
    checkpoint_every = max(1, ceil(Int, checkpoint_fraction * steps))
    approximation, mpo_error, effective_poles = _track_c_approximation(cell)
    started = time()

    if isfile(checkpoint_path)
        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint["parameter_hash"] == expected_hash ||
            error("checkpoint parameter hash does not match requested cell")
        state = checkpoint["state"]
        start_step = Int(checkpoint["step"])
        initial_norm = Float64(checkpoint["initial_norm"])
        initial_parity = Float64(checkpoint["initial_parity"])
        cap_checkpoints = Int(checkpoint["cap_checkpoints"])
        one_site = Bool(checkpoint["one_site"])
        max_norm_drift = Float64(get(checkpoint, "max_norm_drift", 0.0))
        max_parity_drift = Float64(get(checkpoint, "max_parity_drift", 0.0))
    else
        isfile(trajectory_path) &&
            error("trajectory exists without a checkpoint; preserve it and choose a new output directory")
        if isnothing(initial_state)
            Random.seed!(Int(cell["seed"]))
            initial_bond = min(8, chi)
            trial = FiniteMPS(
                randn, ComplexF64, L, ℂ^2, ℂ^initial_bond
            )
            H_initial = _cell_mpo(
                cell, 2Float64(cell["Gamma_c"]), approximation
            )
            algorithm = DMRG2(;
                tol = 1.0e-10,
                maxiter = 50,
                verbosity = 1,
                trscheme = truncrank(chi),
            )
            state, _, _ = find_groundstate(trial, H_initial, algorithm)
        else
            state = deepcopy(initial_state)
        end
        start_step = 0
        initial_norm = real(dot(state, state))
        initial_parity = mps_parity(state)
        cap_checkpoints = 0
        one_site = false
        max_norm_drift = 0.0
        max_parity_drift = 0.0
        correlations = mps_correlations(state; max_distance = min(10, L - 1))
        nearest = mps_nearest_neighbor_correlations(state)
        append_trajectory_row(
            trajectory_path;
            time = 0.0,
            gamma = gamma_at_time(0.0, T, cell["Gamma_c"]),
            kink_density = kink_density_from_zz(nearest),
            kink_density_bulk = bulk_kink_density_from_zz(nearest),
            correlations,
            norm = initial_norm,
            max_bond = _max_bond_dimension(state),
        )
        atomic_checkpoint(checkpoint_path, Dict{String, Any}(
            "parameter_hash" => expected_hash,
            "step" => 0,
            "state" => state,
            "initial_norm" => initial_norm,
            "initial_parity" => initial_parity,
            "cap_checkpoints" => cap_checkpoints,
            "one_site" => one_site,
            "max_norm_drift" => max_norm_drift,
            "max_parity_drift" => max_parity_drift,
        ))
    end

    final_correlations = mps_correlations(
        state;
        max_distance = start_step == steps ? L - 1 : min(10, L - 1),
    )
    maximum_bond_seen = _max_bond_dimension(state)
    for step in (start_step + 1):steps
        midpoint = (step - 0.5) * dt
        gamma_midpoint = gamma_at_time(midpoint, T, cell["Gamma_c"])
        H_midpoint = _cell_mpo(cell, gamma_midpoint, approximation)
        algorithm = one_site ? TDVP() : TDVP2(; trscheme = truncrank(chi))
        state, _ = timestep(state, H_midpoint, 0.0, dt, algorithm)
        maximum_bond_seen = max(maximum_bond_seen, _max_bond_dimension(state))

        if step % checkpoint_every == 0 || step == steps
            current_bond = _max_bond_dimension(state)
            cap_checkpoints = current_bond >= chi ? cap_checkpoints + 1 : 0
            one_site = one_site || cap_checkpoints >= 3
            current_time = step * dt
            final_correlations = mps_correlations(
                state;
                max_distance = step == steps ? L - 1 : min(10, L - 1),
            )
            nearest = mps_nearest_neighbor_correlations(state)
            current_norm = real(dot(state, state))
            current_parity = mps_parity(state)
            max_norm_drift = max(max_norm_drift, abs(current_norm - initial_norm))
            max_parity_drift =
                max(max_parity_drift, abs(current_parity - initial_parity))
            append_trajectory_row(
                trajectory_path;
                time = current_time,
                gamma = gamma_at_time(current_time, T, cell["Gamma_c"]),
                kink_density = kink_density_from_zz(nearest),
                kink_density_bulk = bulk_kink_density_from_zz(nearest),
                correlations = final_correlations,
                norm = current_norm,
                max_bond = current_bond,
            )
            atomic_checkpoint(checkpoint_path, Dict{String, Any}(
                "parameter_hash" => expected_hash,
                "step" => step,
                "state" => state,
                "initial_norm" => initial_norm,
                "initial_parity" => initial_parity,
                "cap_checkpoints" => cap_checkpoints,
                "one_site" => one_site,
                "max_norm_drift" => max_norm_drift,
                "max_parity_drift" => max_parity_drift,
            ))
            @printf(
                "TDVP Track C L=%d T=%.6g step=%d/%d kink=%.8g chi=%d mode=%s\n",
                L, T, step, steps,
                kink_density_from_zz(nearest),
                current_bond, one_site ? "TDVP1" : "TDVP2",
            )
            flush(stdout)
        end
    end

    final_norm = real(dot(state, state))
    final_parity = mps_parity(state)
    final_nearest = mps_nearest_neighbor_correlations(state)
    final_kink = kink_density_from_zz(final_nearest)
    final_bulk_kink = bulk_kink_density_from_zz(final_nearest)
    correlation_summary = _correlation_length_summary(final_correlations)
    manifest = Dict{String, Any}(
        "status" => "completed",
        "method" => "MPSKit-TDVP",
        "parameter_hash" => expected_hash,
        "parameters" => cell,
        "code_revision" => _git_revision(),
        "code_source_sha256" => _source_hash(),
        "completed_at" => _utc_timestamp(),
        "resources" => _resource_manifest(),
        "runtime_seconds" => time() - started,
        "effective_poles" => effective_poles,
        "mpo_max_relative_error" => mpo_error,
        "final_kink_density" => final_kink,
        "final_kink_density_bulk" => final_bulk_kink,
        "boundary_fraction" => 0.1,
        "correlation_length" =>
            isnothing(correlation_summary) ? nothing : correlation_summary["xi"],
        "correlation_kink_density" =>
            isnothing(correlation_summary) ? nothing : correlation_summary["n_corr"],
        "checkpoint_kink_density" => _trajectory_kink_values(trajectory_path),
        "final_correlations" => final_correlations,
        "norm_drift" => max(max_norm_drift, abs(final_norm - initial_norm)),
        "parity_initial" => initial_parity,
        "parity_final" => final_parity,
        "parity_drift" =>
            max(max_parity_drift, abs(final_parity - initial_parity)),
        "max_bond_dimension" => maximum_bond_seen,
        "switched_to_one_site_tdvp" => one_site,
        "checkpoint_every_steps" => checkpoint_every,
    )
    _write_manifest(manifest_path, manifest)
    return manifest
end

function kink_density_from_zz(nearest_neighbour_correlations::AbstractVector)
    isempty(nearest_neighbour_correlations) &&
        throw(ArgumentError("at least one nearest-neighbour correlation is required"))
    return sum(1 .- real.(nearest_neighbour_correlations)) /
           (2length(nearest_neighbour_correlations))
end

function bulk_kink_density_from_zz(
        nearest_neighbour_correlations::AbstractVector;
        boundary_fraction::Real = 0.1,
    )
    0 <= boundary_fraction < 0.5 ||
        throw(ArgumentError("boundary_fraction must lie in [0,0.5)"))
    count = length(nearest_neighbour_correlations)
    count >= 1 ||
        throw(ArgumentError("at least one nearest-neighbour correlation is required"))
    trim = floor(Int, Float64(boundary_fraction) * count)
    first_bulk = trim + 1
    last_bulk = count - trim
    first_bulk <= last_bulk ||
        throw(ArgumentError("boundary trim removes every nearest-neighbour bond"))
    return kink_density_from_zz(
        view(nearest_neighbour_correlations, first_bulk:last_bulk)
    )
end

function correlation_kink_estimator(correlation_length::Real)
    correlation_length > 0 || throw(ArgumentError("correlation length must be positive"))
    return (1 - exp(-inv(Float64(correlation_length)))) / 2
end

function make_cell(;
        sigma, L::Integer, gamma_c::Real, T::Real, chi::Integer,
        dt::Real, poles::Integer, seed::Integer
    )
    L >= 2 || throw(ArgumentError("L must be at least two"))
    gamma_c > 0 || throw(ArgumentError("Gamma_c must be positive"))
    chi >= 1 || throw(ArgumentError("chi must be positive"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    poles >= 1 || throw(ArgumentError("poles must be positive"))
    return Dict{String, Any}(
        "sigma" => sigma isa Real ? Float64(sigma) : String(sigma),
        "L" => Int(L),
        "Gamma_c" => Float64(gamma_c),
        "T" => Float64(T),
        "tau_Q" => tau_q(T),
        "chi" => Int(chi),
        "dt" => Float64(dt),
        "poles" => Int(poles),
        "seed" => Int(seed),
    )
end

function _canonical_value(value)
    if value isa AbstractDict
        pairs = sort!(collect(value); by = pair -> string(first(pair)))
        return "{" * join(
            (repr(string(key)) * ":" * _canonical_value(item) for (key, item) in pairs),
            ",",
        ) * "}"
    elseif value isa AbstractVector
        return "[" * join((_canonical_value(item) for item in value), ",") * "]"
    elseif value isa AbstractString
        return repr(String(value))
    elseif value isa Real
        return repr(value)
    elseif isnothing(value)
        return "null"
    else
        return repr(value)
    end
end

parameter_hash(parameters::AbstractDict) =
    bytes2hex(sha256(codeunits(_canonical_value(parameters))))

function atomic_checkpoint(path::AbstractString, payload)
    mkpath(dirname(path))
    temporary = path * ".tmp." * string(getpid())
    open(temporary, "w") do io
        serialize(io, payload)
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 ||
            error("failed to fsync checkpoint")
    end
    mv(temporary, path; force = true)
    return path
end

load_checkpoint(path::AbstractString) = open(deserialize, path)

function append_trajectory_row(
        path::AbstractString;
        time::Real, gamma::Real, kink_density::Real, kink_density_bulk::Real,
        correlations::AbstractVector, norm::Real, max_bond::Integer
    )
    mkpath(dirname(path))
    header =
        "time,Gamma,kink_density,kink_density_bulk,C(r),norm,max_bond\n"
    existing = isfile(path) ? read(path, String) : header
    lines = filter(!isempty, split(chomp(existing), '\n'))
    if length(lines) >= 2
        last_time = parse(Float64, first(split(last(lines), ',')))
        isapprox(last_time, time; atol = 100eps(max(abs(Float64(time)), 1.0)), rtol = 0) &&
            return path
        last_time < time ||
            error("trajectory time must increase monotonically")
    end
    encoded_correlations = join(
        (@sprintf("%.16g", real(value)) for value in correlations), ";"
    )
    row = @sprintf(
        "%.16g,%.16g,%.16g,%.16g,%s,%.16g,%d\n",
        time, gamma, kink_density, kink_density_bulk,
        encoded_correlations, norm, max_bond,
    )
    temporary = path * ".tmp." * string(getpid())
    open(temporary, "w") do io
        write(io, existing)
        endswith(existing, '\n') || println(io)
        write(io, row)
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 ||
            error("failed to fsync trajectory")
    end
    mv(temporary, path; force = true)
    return path
end

function fit_power_law(
        times::AbstractVector, values::AbstractVector;
        indices = eachindex(times)
    )
    length(times) == length(values) ||
        throw(DimensionMismatch("times and values must have equal length"))
    selected = collect(indices)
    length(selected) >= 2 || throw(ArgumentError("at least two fit points are required"))
    x = log.(Float64.(times[selected]))
    y = log.(Float64.(values[selected]))
    all(isfinite, x) && all(isfinite, y) ||
        throw(ArgumentError("power-law inputs must be finite and positive"))
    design = hcat(ones(length(x)), x)
    intercept, slope = design \ y
    predicted = exp.(intercept .+ slope .* x)
    residual = maximum(abs.((predicted .- Float64.(values[selected])) ./
                            Float64.(values[selected])))
    return Dict{String, Any}(
        "mu" => -slope,
        "prefactor" => exp(intercept),
        "max_relative_residual" => residual,
        "indices" => selected,
    )
end

function _sprint_sigma_key(sigma)
    return string(Float64(sigma))
end

function _sprint_baseline_rows(rows::AbstractVector, sigma; L::Integer = 64)
    return filter(rows) do row
        Float64(row["sigma"]) == Float64(sigma) &&
            Int(row["L"]) == L &&
            get(row, "variant", "baseline") == "baseline"
    end
end

function fit_sprint_mu_curve(
        rows::AbstractVector;
        estimator::AbstractString = "kink_density_bulk",
    )
    results = Dict{String, Any}()
    sigmas = sort!(unique(Float64(row["sigma"]) for row in rows if
        Int(row["L"]) == 64 && get(row, "variant", "baseline") == "baseline"))
    primary_times = Float64[8, 16, 32, 64]
    for sigma in sigmas
        baseline = _sprint_baseline_rows(rows, sigma)
        by_time = Dict(Float64(row["T"]) => row for row in baseline)
        all(haskey(by_time, T) for T in primary_times) ||
            error("sigma=$sigma is missing a registered fit time")
        values = Float64[by_time[T][estimator] for T in primary_times]
        fit = fit_power_law(primary_times, values)
        without_left = fit_power_law(primary_times[2:end], values[2:end])["mu"]
        without_right =
            fit_power_law(primary_times[1:end-1], values[1:end-1])["mu"]
        endpoint_delta = maximum(abs.([
            without_left - fit["mu"], without_right - fit["mu"],
        ]))
        results[_sprint_sigma_key(sigma)] = Dict{String, Any}(
            "mu" => fit["mu"],
            "prefactor" => fit["prefactor"],
            "max_relative_residual" => fit["max_relative_residual"],
            "endpoint_mu_delta" => endpoint_delta,
            "fit_times" => primary_times,
            "fit_passed" =>
                fit["max_relative_residual"] <= 0.10 && endpoint_delta <= 0.03,
        )
    end
    return results
end

function _two_point_mu(first_row, second_row, estimator::AbstractString)
    first_time = Float64(first_row["T"])
    second_time = Float64(second_row["T"])
    first_time < second_time || throw(ArgumentError("times must increase"))
    first_value = Float64(first_row[estimator])
    second_value = Float64(second_row[estimator])
    first_value > 0 && second_value > 0 ||
        throw(ArgumentError("kink estimates must be positive"))
    return -log(second_value / first_value) / log(second_time / first_time)
end

function sprint_systematic_envelope(
        rows::AbstractVector, sigma;
        collapse_delta::Real = 0.0,
    )
    selected = filter(
        row -> Float64(row["sigma"]) == Float64(sigma) &&
               Int(row["L"]) == 64,
        rows,
    )
    baselines = Dict(
        Float64(row["T"]) => row for row in selected
        if get(row, "variant", "baseline") == "baseline"
    )
    all(haskey(baselines, T) for T in (16.0, 64.0)) ||
        error("systematic envelope requires T=16 and T=64 baselines")
    baseline_mu = _two_point_mu(
        baselines[16.0], baselines[64.0], "kink_density_bulk"
    )
    variant_mu_delta = Dict{String, Float64}()
    max_kink_drift = 0.0
    variants = sort!(unique(
        String(get(row, "variant", "baseline")) for row in selected
        if get(row, "variant", "baseline") != "baseline"
    ))
    for variant in variants
        variant_rows = Dict(
            Float64(row["T"]) => row for row in selected
            if get(row, "variant", "baseline") == variant
        )
        all(haskey(variant_rows, T) for T in (16.0, 64.0)) || continue
        mu = _two_point_mu(
            variant_rows[16.0], variant_rows[64.0], "kink_density_bulk"
        )
        variant_mu_delta[variant] = abs(mu - baseline_mu)
        for T in (16.0, 64.0)
            reference = Float64(baselines[T]["kink_density_bulk"])
            observed = Float64(variant_rows[T]["kink_density_bulk"])
            max_kink_drift = max(
                max_kink_drift, abs(observed - reference) / abs(reference)
            )
        end
    end
    main_baseline = _sprint_baseline_rows(rows, sigma)
    by_time = Dict(Float64(row["T"]) => row for row in main_baseline)
    primary_times = Float64[8, 16, 32, 64]
    bulk_fit = fit_power_law(
        primary_times,
        Float64[by_time[T]["kink_density_bulk"] for T in primary_times],
    )
    estimator_deltas = Float64[]
    for estimator in ("kink_density", "correlation_kink_density")
        all(haskey(by_time[T], estimator) &&
            !isnothing(by_time[T][estimator]) for T in primary_times) || continue
        fit = fit_power_law(
            primary_times,
            Float64[by_time[T][estimator] for T in primary_times],
        )
        push!(estimator_deltas, abs(fit["mu"] - bulk_fit["mu"]))
    end
    gamma_mu_delta = maximum(
        [get(variant_mu_delta, "gamma_low", 0.0),
         get(variant_mu_delta, "gamma_high", 0.0)];
        init = 0.0,
    )
    systematic_mu_delta = maximum(vcat(
        collect(values(variant_mu_delta)),
        estimator_deltas,
        [abs(Float64(collapse_delta))],
    ); init = 0.0)
    return Dict{String, Any}(
        "baseline_two_point_mu" => baseline_mu,
        "variant_mu_delta" => variant_mu_delta,
        "estimator_mu_delta" =>
            isempty(estimator_deltas) ? 0.0 : maximum(estimator_deltas),
        "gamma_mu_delta" => gamma_mu_delta,
        "finite_size_mu_delta" => abs(Float64(collapse_delta)),
        "max_kink_relative_drift" => max_kink_drift,
        "systematic_mu_delta" => systematic_mu_delta,
        "passed" => max_kink_drift <= 0.02 &&
                    gamma_mu_delta <= 0.03 &&
                    systematic_mu_delta <= 0.05,
    )
end

function _theory_subset_result(observed, prediction, keys)
    offset = mean(
        Float64(observed[key]["mu"]) - Float64(prediction[key]) for key in keys
    )
    residuals = Dict(
        key => Float64(observed[key]["mu"]) -
               (Float64(prediction[key]) + offset)
        for key in keys
    )
    loss = sum(abs2, values(residuals))
    return Dict{String, Any}(
        "offset" => offset,
        "loss" => loss,
        "residuals" => residuals,
    )
end

function compare_theory_shapes(
        observed::AbstractDict, theories::AbstractDict;
        internal_sigmas = ("1.8", "1.875", "1.95"),
    )
    sigma_keys = String.(collect(internal_sigmas))
    all(haskey(observed, key) for key in sigma_keys) ||
        error("observed trend is missing an internal sigma")
    length(theories) == 2 ||
        throw(ArgumentError("exactly two theory curves are required"))
    theory_names = sort!(String.(collect(Base.keys(theories))))
    all(all(haskey(theories[name], key) for key in sigma_keys)
        for name in theory_names) ||
        error("theory curve is missing an internal sigma")
    subsets = Dict{String, Vector{String}}("full" => sigma_keys)
    for omitted in sigma_keys
        subsets["omit_$omitted"] = filter(!=(omitted), sigma_keys)
    end
    winners = Dict{String, String}()
    details = Dict{String, Any}()
    for (label, subset) in subsets
        results = Dict(
            name => _theory_subset_result(observed, theories[name], subset)
            for name in theory_names
        )
        losses = Dict(name => Float64(results[name]["loss"]) for name in theory_names)
        ordered = sort!(theory_names; by = name -> (losses[name], name))
        winners[label] = first(ordered)
        details[label] = results
    end
    stable_winner = length(unique(values(winners))) == 1 ?
        first(values(winners)) : nothing
    nonoverlap_count = 0
    if !isnothing(stable_winner)
        loser = only(filter(!=(stable_winner), theory_names))
        loser_result = details["full"][loser]
        nonoverlap_count = count(sigma_keys) do key
            abs(Float64(loser_result["residuals"][key])) >
                Float64(observed[key]["total_error"])
        end
    end
    favored = !isnothing(stable_winner) && nonoverlap_count >= 2
    return Dict{String, Any}(
        "status" => favored ? "favored" : "inconclusive",
        "favored" => favored ? stable_winner : nothing,
        "winners" => winners,
        "nonoverlap_count_for_loser" => nonoverlap_count,
        "details" => details,
    )
end

function campaign_rows(campaign::AbstractDict)
    rows = Dict{String, Any}[]
    for cell in campaign["cells"]
        get(cell, "status", "") == "completed" || continue
        result = cell["result"]
        result isa AbstractDict || error("completed cell has no result")
        manifest = haskey(result, "tdvp") ? result["tdvp"] : result
        haskey(manifest, "final_kink_density_bulk") ||
            error("completed sprint cell lacks the bulk kink estimator")
        parameters = cell["parameters"]
        push!(rows, Dict{String, Any}(
            "cell_id" => cell["id"],
            "sigma" => parameters["sigma"],
            "L" => parameters["L"],
            "T" => parameters["T"],
            "tau_Q" => parameters["tau_Q"],
            "variant" => get(parameters, "variant", "baseline"),
            "shard_id" => get(parameters, "shard_id", nothing),
            "component_stage" => get(parameters, "component_stage", nothing),
            "kink_density_bulk" => manifest["final_kink_density_bulk"],
            "kink_density" => manifest["final_kink_density"],
            "correlation_kink_density" =>
                get(manifest, "correlation_kink_density", nothing),
        ))
    end
    return rows
end

function _campaign_numerical_pass(cell::AbstractDict)
    result = cell["result"]
    result isa AbstractDict || return false
    gates = get(result, "numerical_gate", nothing)
    gates isa AbstractDict || return false
    return all(Bool(gate["passed"]) for gate in values(gates))
end

function _campaign_reproduction_gates(campaign::AbstractDict)
    cells = campaign["cells"]
    g2_cells = filter(
        cell -> get(cell["parameters"], "component_stage", "") == "G2",
        cells,
    )
    g3_cells = filter(
        cell -> get(cell["parameters"], "component_stage", "") == "G3",
        cells,
    )
    g3_sentinels = filter(
        cell -> get(cell["parameters"], "component_stage", "") == "G3-sentinels",
        cells,
    )
    g2_complete = length(g2_cells) == 6 &&
                  all(cell["status"] == "completed" for cell in g2_cells)
    g3_complete = length(g3_cells) == 6 &&
                  all(cell["status"] == "completed" for cell in g3_cells)
    sentinel_complete = length(g3_sentinels) == 4 &&
        all(cell["status"] == "completed" for cell in g3_sentinels)
    g2_gate = Dict{String, Any}("passed" => false)
    if g2_complete
        ordered = sort!(g2_cells; by = cell -> cell["parameters"]["T"])
        times = Float64[cell["parameters"]["T"] for cell in ordered]
        values = Float64[
            cell["result"]["tdvp"]["final_kink_density"] for cell in ordered
        ]
        g2_gate = evaluate_g2(times, values; exact_prefactor = inv(2pi))
        g2_gate["passed"] &= all(_campaign_numerical_pass, g2_cells)
    end
    g3_gate = Dict{String, Any}("passed" => false, "literature_passed" => false)
    if g3_complete
        ordered = sort!(g3_cells; by = cell -> cell["parameters"]["T"])
        times = Float64[cell["parameters"]["T"] for cell in ordered]
        values = Float64[
            cell["result"]["tdvp"]["final_kink_density"] for cell in ordered
        ]
        literature = evaluate_g3(times, values; paper_mu = 0.50)
        baseline = Dict(
            Float64(cell["parameters"]["T"]) =>
                Float64(cell["result"]["tdvp"]["final_kink_density"])
            for cell in g3_cells
        )
        max_sentinel_drift = 0.0
        if sentinel_complete
            for cell in g3_sentinels
                T = Float64(cell["parameters"]["T"])
                observed =
                    Float64(cell["result"]["tdvp"]["final_kink_density"])
                max_sentinel_drift = max(
                    max_sentinel_drift,
                    abs(observed - baseline[T]) / abs(baseline[T]),
                )
            end
        else
            max_sentinel_drift = Inf
        end
        numerical = all(_campaign_numerical_pass, g3_cells) &&
                    all(_campaign_numerical_pass, g3_sentinels) &&
                    max_sentinel_drift <= 0.02
        g3_gate = Dict{String, Any}(
            "passed" => numerical,
            "literature_passed" => Bool(literature["passed"]),
            "mu" => literature["mu"],
            "max_sentinel_relative_drift" => max_sentinel_drift,
        )
    end
    return Dict{String, Any}("G2" => g2_gate, "G3" => g3_gate)
end

function analyze_sprint_campaign(
        campaign::AbstractDict;
        theories = nothing,
    )
    missing = String[
        cell["id"] for cell in campaign["cells"]
        if cell["status"] != "completed"
    ]
    if !isempty(missing)
        return Dict{String, Any}(
            "status" => "incomplete",
            "missing_cell_ids" => missing,
            "completed_cells" => length(campaign["cells"]) - length(missing),
            "total_cells" => length(campaign["cells"]),
        )
    end
    rows = campaign_rows(campaign)
    sprint_rows = filter(rows) do row
        row["component_stage"] in (
            "sprint-size-1875", "sprint-trend",
            "sprint-sentinels", "sprint-gamma",
        )
    end
    curve = fit_sprint_mu_curve(sprint_rows)
    collapse_rows = Dict{String, Any}[
        Dict(
            "L" => row["L"],
            "T" => row["T"],
            "tau_Q" => row["tau_Q"],
            "kink_density" => row["kink_density_bulk"],
        )
        for row in sprint_rows
        if Float64(row["sigma"]) == 1.875 &&
           get(row, "variant", "baseline") == "baseline" &&
           Float64(row["T"]) in (8.0, 16.0, 32.0, 64.0)
    ]
    collapse = fit_collapse_mu(collapse_rows)
    collapse_delta = abs(
        Float64(collapse["mu"]) - Float64(curve["1.875"]["mu"])
    )
    collapse_gate = evaluate_collapse(collapse, curve["1.875"]["mu"])
    envelopes = Dict(
        key => sprint_systematic_envelope(
            sprint_rows, parse(Float64, key); collapse_delta
        )
        for key in ("1.8", "1.875", "1.95")
    )
    reproduction = _campaign_reproduction_gates(campaign)
    prerequisites_passed =
        Bool(reproduction["G2"]["passed"]) &&
        Bool(reproduction["G3"]["passed"])
    trend_ready = prerequisites_passed &&
        Bool(collapse_gate["passed"]) &&
        all(Bool(curve[key]["fit_passed"]) for key in ("1.8", "1.875", "1.95")) &&
        all(Bool(envelopes[key]["passed"]) for key in ("1.8", "1.875", "1.95"))
    observed = Dict(
        key => Dict(
            "mu" => curve[key]["mu"],
            "total_error" => envelopes[key]["systematic_mu_delta"],
        )
        for key in ("1.8", "1.875", "1.95")
    )
    comparison = isnothing(theories) ? nothing :
        compare_theory_shapes(observed, theories)
    status = !prerequisites_passed ? "provisional" :
             !trend_ready ? "inconclusive" :
             isnothing(comparison) ? "ready-for-theory" :
             comparison["status"]
    return Dict{String, Any}(
        "status" => status,
        "mu_curve" => curve,
        "collapse_1875" => merge(
            Dict("finite_size_mu_delta" => collapse_delta),
            collapse,
            collapse_gate,
        ),
        "systematic_envelopes" => envelopes,
        "reproduction_gates" => reproduction,
        "trend_ready" => trend_ready,
        "theory_comparison" => comparison,
    )
end

function evaluate_g1(result::AbstractDict)
    ed = Float64(result["final_kink_ed"])
    tdvp = Float64(result["final_kink_tdvp"])
    relative = abs(tdvp - ed) / max(abs(ed), eps(Float64))
    trajectory_difference = maximum(abs.(
        Float64.(result["trajectory_tdvp"]) .- Float64.(result["trajectory_ed"])
    ))
    return Dict{String, Any}(
        "passed" => relative <= 0.01 && trajectory_difference <= 0.01,
        "final_relative_difference" => relative,
        "trajectory_max_absolute_difference" => trajectory_difference,
    )
end

function _endpoint_stability(times, values, primary)
    reference = fit_power_law(times, values; indices = primary)["mu"]
    without_left = fit_power_law(times, values; indices = primary[2:end])["mu"]
    without_right = fit_power_law(times, values; indices = primary[1:(end - 1)])["mu"]
    return maximum(abs.([without_left - reference, without_right - reference]))
end

function evaluate_g2(
        times::AbstractVector, values::AbstractVector;
        exact_prefactor::Real
    )
    primary = [2, 3, 4, 5]
    fit = fit_power_law(times, values; indices = primary)
    prefactor_error = abs(fit["prefactor"] - exact_prefactor) / abs(exact_prefactor)
    endpoint_delta = _endpoint_stability(times, values, primary)
    return Dict{String, Any}(
        "passed" => abs(fit["mu"] - 0.5) <= 0.02 &&
                    prefactor_error <= 0.05 && endpoint_delta <= 0.03,
        "mu" => fit["mu"],
        "prefactor" => fit["prefactor"],
        "prefactor_relative_error" => prefactor_error,
        "endpoint_mu_delta" => endpoint_delta,
        "fit_indices" => primary,
    )
end

function evaluate_g3(
        times::AbstractVector, values::AbstractVector;
        paper_mu::Real = 0.50
    )
    primary = [2, 3, 4, 5]
    fit = fit_power_law(times, values; indices = primary)
    return Dict{String, Any}(
        "passed" => fit["max_relative_residual"] <= 0.10 &&
                    abs(fit["mu"] - paper_mu) <= 0.10,
        "acceptance" => "relaxed",
        "mu" => fit["mu"],
        "paper_mu" => Float64(paper_mu),
        "max_relative_residual" => fit["max_relative_residual"],
        "fit_indices" => primary,
    )
end

function locate_critical_field(
        evaluator::Function, initial_estimate::Real;
        initial_half_width::Real = 0.03,
        target_width::Real = 0.01,
        max_refinements::Integer = 12,
    )
    initial_half_width > 0 ||
        throw(ArgumentError("initial_half_width must be positive"))
    target_width > 0 || throw(ArgumentError("target_width must be positive"))
    max_refinements >= 1 ||
        throw(ArgumentError("max_refinements must be positive"))
    coarse_gammas = collect(range(
        Float64(initial_estimate - initial_half_width),
        Float64(initial_estimate + initial_half_width);
        length = 7,
    ))
    coarse_values = Float64[evaluator(gamma) for gamma in coarse_gammas]
    coarse_peak = argmax(coarse_values)
    1 < coarse_peak < length(coarse_gammas) ||
        error("coarse entropy maximum lies at scan boundary; widen or recenter the scan")
    encode = (gammas, values) -> [
        Dict("Gamma" => gamma, "half_chain_entropy" => value)
        for (gamma, value) in zip(gammas, values)
    ]
    bracket_low = coarse_gammas[coarse_peak - 1]
    bracket_high = coarse_gammas[coarse_peak + 1]
    refinements = Any[]
    fine_gammas = Float64[]
    fine_values = Float64[]
    fine_peak = 0
    for _ in 1:max_refinements
        fine_gammas = collect(range(bracket_low, bracket_high; length = 5))
        fine_values = Float64[evaluator(gamma) for gamma in fine_gammas]
        fine_peak = argmax(fine_values)
        1 < fine_peak < length(fine_gammas) ||
            error("fine entropy maximum lies at scan boundary; recenter the scan")
        bracket_low = fine_gammas[fine_peak - 1]
        bracket_high = fine_gammas[fine_peak + 1]
        push!(refinements, encode(fine_gammas, fine_values))
        bracket_high - bracket_low <= target_width + 100eps(Float64) && break
    end
    bracket_high - bracket_low <= target_width + 100eps(Float64) ||
        error("critical-field interval remains wider than target_width")
    return Dict{String, Any}(
        "Gamma_c" => fine_gammas[fine_peak],
        "Gamma_low" => bracket_low,
        "Gamma_high" => bracket_high,
        "target_width" => Float64(target_width),
        "coarse" => encode(coarse_gammas, coarse_values),
        "fine" => encode(fine_gammas, fine_values),
        "refinements" => refinements,
    )
end

function fit_correlation_length(
        correlations::AbstractVector;
        distances = eachindex(correlations)
    )
    selected = collect(distances)
    length(selected) >= 2 ||
        throw(ArgumentError("at least two distances are required"))
    values = abs.(Float64.(correlations[selected]))
    all(>(0), values) ||
        throw(ArgumentError("selected correlations must be nonzero"))
    x = Float64.(selected)
    intercept, slope = hcat(ones(length(x)), x) \ log.(values)
    slope < 0 || throw(ArgumentError("selected correlations do not decay exponentially"))
    xi = -inv(slope)
    predicted = exp.(intercept .+ slope .* x)
    return Dict{String, Any}(
        "xi" => xi,
        "n_corr" => correlation_kink_estimator(xi),
        "max_relative_residual" =>
            maximum(abs.((predicted .- values) ./ values)),
        "distances" => selected,
    )
end

function fit_collapse_mu(
        rows::AbstractVector;
        mu_grid = 0.2:0.002:1.2,
        polynomial_degree::Integer = 3,
    )
    length(rows) >= polynomial_degree + 2 ||
        throw(ArgumentError("insufficient rows for collapse fit"))
    best = nothing
    for mu in mu_grid
        mu > 0 || continue
        log_x = Float64[
            log(Float64(row["tau_Q"]) / Float64(row["L"])^(1 / mu))
            for row in rows
        ]
        log_y = Float64[
            log(Float64(row["L"]) * Float64(row["kink_density"]))
            for row in rows
        ]
        all(isfinite, log_x) && all(isfinite, log_y) || continue
        center = mean(log_x)
        scale = std(log_x)
        scale > 0 || continue
        normalized = (log_x .- center) ./ scale
        design = hcat(
            (normalized .^ degree for degree in 0:polynomial_degree)...
        )
        coefficients = design \ log_y
        residuals = log_y - design * coefficients
        score = sqrt(mean(abs2, residuals))
        candidate = Dict{String, Any}(
            "mu" => Float64(mu),
            "log_rmse" => score,
            "polynomial_degree" => polynomial_degree,
            "log_x_center" => center,
            "log_x_scale" => scale,
            "coefficients" => coefficients,
        )
        if isnothing(best) || score < best["log_rmse"]
            best = candidate
        end
    end
    isnothing(best) && error("no valid collapse candidate")
    return best::Dict{String, Any}
end

function evaluate_collapse(collapse::AbstractDict, single_size_mu::Real)
    difference = abs(Float64(collapse["mu"]) - Float64(single_size_mu))
    return Dict{String, Any}(
        "passed" => difference <= 0.05,
        "collapse_mu" => Float64(collapse["mu"]),
        "single_size_mu" => Float64(single_size_mu),
        "absolute_difference" => difference,
    )
end

function evaluate_g3_sentinels(
        baseline::AbstractDict, sentinels::AbstractVector
    )
    differences = Dict(
        "chi" => Dict{Float64, Vector{Float64}}(),
        "dt" => Dict{Float64, Vector{Float64}}(),
    )
    for sentinel in sentinels
        T = Float64(sentinel["parameters"]["T"])
        haskey(baseline, T) || throw(KeyError("missing G3 baseline T=$T"))
        reference = Float64(baseline[T]["final_kink_density"])
        observed = Float64(sentinel["final_kink_density"])
        difference = abs(observed - reference) / max(abs(reference), eps(Float64))
        chi = Int(sentinel["parameters"]["chi"])
        dt = Float64(sentinel["parameters"]["dt"])
        kind = chi == 64 && isapprox(dt, 0.05) ? "chi" :
            chi == 96 && isapprox(dt, 0.025) ? "dt" :
            throw(ArgumentError("unrecognized G3 sentinel chi=$chi dt=$dt"))
        push!(get!(differences[kind], T, Float64[]), difference)
    end
    maximum_by_kind = Dict(
        kind => Dict(
            string(T) => maximum(values) for (T, values) in by_time
        )
        for (kind, by_time) in differences
    )
    chi_upgrade_T = sort!([
        T for (T, values) in differences["chi"] if maximum(values) > 0.02
    ])
    dt_failed_T = sort!([
        T for (T, values) in differences["dt"] if maximum(values) > 0.02
    ])
    return Dict{String, Any}(
        "passed" => isempty(chi_upgrade_T) && isempty(dt_failed_T),
        "threshold" => 0.02,
        "maximum_relative_difference" => maximum_by_kind,
        "upgrade_T" => chi_upgrade_T,
        "chi_upgrade_T" => chi_upgrade_T,
        "dt_failed_T" => dt_failed_T,
    )
end

function _completed_results(run::AbstractDict)
    cells = filter(cell -> cell["status"] == "completed", run["cells"])
    return cells, length(cells) == length(run["cells"])
end

function _all_numerical_gates_pass(cells)
    return all(cells) do cell
        gates = values(cell["result"]["numerical_gate"])
        all(gate["passed"] for gate in gates)
    end
end

function analyze_run(run::AbstractDict)
    stage = String(run["stage"])
    cells, complete = _completed_results(run)
    if !complete
        return Dict{String, Any}(
            "gate" => stage,
            "status" => "incomplete",
            "completed_cells" => length(cells),
            "total_cells" => length(run["cells"]),
        )
    end

    if stage == "G1"
        checks = Dict{String, Any}[]
        for cell in cells
            result = cell["result"]
            push!(checks, merge(
                Dict("cell_id" => cell["id"]),
                evaluate_g1(Dict(
                    "final_kink_ed" => result["ed"]["final_kink_density"],
                    "final_kink_tdvp" => result["tdvp"]["final_kink_density"],
                    "trajectory_ed" => result["ed"]["checkpoint_kink_density"],
                    "trajectory_tdvp" =>
                        result["tdvp"]["checkpoint_kink_density"],
                )),
            ))
        end
        passed = all(check["passed"] for check in checks) &&
                 _all_numerical_gates_pass(cells)
        return Dict{String, Any}(
            "gate" => "G1",
            "status" => passed ? "passed" : "failed",
            "checks" => checks,
        )
    elseif stage == "G2"
        ordered = sort!(cells; by = cell -> Float64(cell["parameters"]["T"]))
        times = Float64[cell["parameters"]["T"] for cell in ordered]
        values = Float64[
            cell["result"]["tdvp"]["final_kink_density"] for cell in ordered
        ]
        gate = evaluate_g2(times, values; exact_prefactor = inv(2pi))
        gate["passed"] = gate["passed"] && _all_numerical_gates_pass(cells)
        gate["gate"] = "G2"
        gate["status"] = gate["passed"] ? "passed" : "failed"
        gate["exact_prefactor_axis_T"] = inv(2pi)
        return gate
    elseif stage in ("G3", "G3-L128", "G3-converged")
        ordered = sort!(cells; by = cell -> Float64(cell["parameters"]["T"]))
        times = Float64[cell["parameters"]["T"] for cell in ordered]
        values = Float64[
            cell["result"]["tdvp"]["final_kink_density"] for cell in ordered
        ]
        gate = evaluate_g3(times, values; paper_mu = 0.50)
        literature_gate_passed = Bool(gate["passed"])
        numerical_gates_passed = _all_numerical_gates_pass(cells)
        gate["literature_gate_passed"] = literature_gate_passed
        gate["numerical_gates_passed"] = numerical_gates_passed
        gate["passed"] = literature_gate_passed && numerical_gates_passed
        if stage == "G3-converged" && haskey(run, "fallback_validation")
            fallback = run["fallback_validation"]
            if fallback["status"] == "passed" && numerical_gates_passed
                gate["L64_screening_status"] =
                    gate["passed"] ? "passed" : "failed_finite_size_screen"
                gate["fallback_validation"] = fallback
                gate["passed"] = true
            end
        end
        gate["gate"] = stage
        gate["status"] = gate["passed"] ? "passed" : "failed"
        return gate
    elseif stage == "floor-collapse"
        return Dict{String, Any}(
            "gate" => stage,
            "status" => "completed",
            "next" => "merge with G3 and floor-l64 using the collapse command",
        )
    end
    return Dict{String, Any}(
        "gate" => stage,
        "status" => "completed",
        "completed_cells" => length(cells),
    )
end

function _gamma_lookup(gamma_c, sigma, L::Integer)
    gamma_c isa Real && return Float64(gamma_c)
    gamma_c isa AbstractDict ||
        throw(ArgumentError("gamma_c must be a scalar or sigma-keyed dictionary"))
    sigma == "NN" && L == 256 && return 1.0
    key = sigma isa AbstractString ? String(sigma) : string(Float64(sigma))
    size_key = "$key:$L"
    haskey(gamma_c, size_key) && return gamma_c[size_key]
    haskey(gamma_c, key) && return gamma_c[key]
    short_key = sigma isa Real && isinteger(sigma) ? string(Int(sigma)) : key
    short_size_key = "$short_key:$L"
    haskey(gamma_c, short_size_key) && return gamma_c[short_size_key]
    haskey(gamma_c, short_key) && return gamma_c[short_key]
    throw(KeyError("missing Gamma_c for sigma=$sigma, L=$L"))
end

function _gamma_for(gamma_c, sigma, L::Integer)
    entry = _gamma_lookup(gamma_c, sigma, L)
    entry isa Real && return Float64(entry)
    entry isa AbstractDict ||
        throw(ArgumentError("Gamma_c map entry must be a number or dictionary"))
    haskey(entry, "Gamma_c") || throw(KeyError("Gamma_c"))
    return Float64(entry["Gamma_c"])
end

function _gamma_interval(gamma_c, sigma, L::Integer)
    entry = _gamma_lookup(gamma_c, sigma, L)
    if entry isa Real
        value = Float64(entry)
        return value, value
    end
    entry isa AbstractDict ||
        throw(ArgumentError("Gamma_c map entry must be a number or dictionary"))
    low = Float64(entry["Gamma_low"])
    high = Float64(entry["Gamma_high"])
    low <= high || throw(ArgumentError("Gamma_c interval is reversed"))
    return low, high
end

function _tag_sprint_cell!(
        cell::AbstractDict, shard_id::AbstractString, variant::AbstractString,
        gamma_c;
        execution_mode::AbstractString = "tdvp",
        component_stage::AbstractString,
    )
    low, high = _gamma_interval(gamma_c, cell["sigma"], Int(cell["L"]))
    cell["Gamma_c_interval"] = [low, high]
    cell["variant"] = String(variant)
    cell["shard_id"] = String(shard_id)
    cell["execution_mode"] = String(execution_mode)
    cell["component_stage"] = String(component_stage)
    return cell
end

function _sprint_cell(;
        sigma, L::Integer, gamma_c, T::Real, chi::Integer, dt::Real,
        poles::Integer, seed::Integer, shard_id::AbstractString,
        variant::AbstractString = "baseline", gamma_variant = nothing,
        component_stage::AbstractString,
    )
    nominal = _gamma_for(gamma_c, sigma, L)
    low, high = _gamma_interval(gamma_c, sigma, L)
    actual_gamma = isnothing(gamma_variant) ? nominal :
        gamma_variant == "low" ? low :
        gamma_variant == "high" ? high :
        throw(ArgumentError("gamma_variant must be low, high, or nothing"))
    cell = make_cell(;
        sigma, L, gamma_c = actual_gamma, T, chi, dt, poles, seed
    )
    cell["Gamma_c_nominal"] = nominal
    cell["Gamma_c_interval"] = [low, high]
    cell["variant"] = String(variant)
    cell["shard_id"] = String(shard_id)
    cell["execution_mode"] = "tdvp"
    cell["component_stage"] = String(component_stage)
    return cell
end

function _sprint_size_cells(gamma_c; seed::Integer)
    cells = Dict{String, Any}[]
    for (L, times, chi) in (
            (32, (8.0, 16.0, 32.0, 64.0), 96),
            (64, Tuple(DEFAULT_TIMES), 96),
            (128, (8.0, 16.0, 32.0, 64.0), 128),
        ), T in times
        push!(cells, _sprint_cell(;
            sigma = 1.875, L, gamma_c, T, chi, dt = 0.05,
            poles = 16, seed, shard_id = "A",
            component_stage = "sprint-size-1875",
        ))
    end
    return cells
end

function _sprint_trend_cells(gamma_c; seed::Integer)
    return [
        _sprint_cell(;
            sigma, L = 64, gamma_c, T, chi = 96, dt = 0.05,
            poles = 16, seed, shard_id = "B",
            component_stage = "sprint-trend",
        )
        for sigma in (1.75, 1.8, 1.95, 2.0), T in DEFAULT_TIMES
    ][:]
end

function _sprint_sentinel_cells(gamma_c; seed::Integer)
    cells = Dict{String, Any}[]
    for T in (16.0, 64.0)
        push!(cells, _sprint_cell(;
            sigma = 1.875, L = 64, gamma_c, T, chi = 128, dt = 0.05,
            poles = 16, seed, shard_id = "A", variant = "chi",
            component_stage = "sprint-sentinels",
        ))
        push!(cells, _sprint_cell(;
            sigma = 1.875, L = 64, gamma_c, T, chi = 96, dt = 0.025,
            poles = 16, seed, shard_id = "A", variant = "dt",
            component_stage = "sprint-sentinels",
        ))
        push!(cells, _sprint_cell(;
            sigma = 1.875, L = 64, gamma_c, T, chi = 96, dt = 0.05,
            poles = 24, seed, shard_id = "A", variant = "poles",
            component_stage = "sprint-sentinels",
        ))
        push!(cells, _sprint_cell(;
            sigma = 1.875, L = 64, gamma_c, T, chi = 96, dt = 0.05,
            poles = 16, seed = 87, shard_id = "A", variant = "seed",
            component_stage = "sprint-sentinels",
        ))
    end
    for sigma in (1.75, 2.0), T in (16.0, 64.0)
        push!(cells, _sprint_cell(;
            sigma, L = 64, gamma_c, T, chi = 96, dt = 0.025,
            poles = 16, seed, shard_id = "B", variant = "dt",
            component_stage = "sprint-sentinels",
        ))
        push!(cells, _sprint_cell(;
            sigma, L = 64, gamma_c, T, chi = 96, dt = 0.05,
            poles = 24, seed, shard_id = "B", variant = "poles",
            component_stage = "sprint-sentinels",
        ))
    end
    for sigma in (1.8, 1.95), T in (16.0, 64.0)
        push!(cells, _sprint_cell(;
            sigma, L = 64, gamma_c, T, chi = 128, dt = 0.05,
            poles = 16, seed, shard_id = "B", variant = "chi",
            component_stage = "sprint-sentinels",
        ))
        push!(cells, _sprint_cell(;
            sigma, L = 64, gamma_c, T, chi = 96, dt = 0.025,
            poles = 16, seed, shard_id = "B", variant = "dt",
            component_stage = "sprint-sentinels",
        ))
    end
    return cells
end

function _sprint_gamma_cells(gamma_c; seed::Integer)
    cells = Dict{String, Any}[]
    for (sigma, shard) in ((1.875, "A"), (1.8, "B"), (1.95, "B")),
            T in (16.0, 64.0), endpoint in ("low", "high")
        push!(cells, _sprint_cell(;
            sigma, L = 64, gamma_c, T, chi = 96, dt = 0.05,
            poles = 16, seed, shard_id = shard,
            variant = "gamma_$endpoint", gamma_variant = endpoint,
            component_stage = "sprint-gamma",
        ))
    end
    return cells
end

function stage_cells(stage::AbstractString; gamma_c, seed::Integer = 86)
    if stage == "G1"
        return [
            make_cell(; sigma = 1.0, L,
                      gamma_c = _gamma_for(gamma_c, 1.0, L),
                      T, chi = 64, dt = 0.025,
                      poles = 16, seed)
            for L in (12, 16, 20), T in (8.0, 32.0)
        ][:]
    elseif stage == "G1-dt-sentinel"
        return [
            make_cell(;
                sigma = 1.0, L = 16,
                gamma_c = _gamma_for(gamma_c, 1.0, 16),
                T = 32.0, chi = 64, dt = 0.0125,
                poles = 16, seed,
            ),
        ]
    elseif stage == "G2"
        return [
            make_cell(; sigma = "NN", L = 256,
                      gamma_c = _gamma_for(gamma_c, "NN", 256),
                      T, chi = 64, dt = 0.05,
                      poles = 1, seed)
            for T in DEFAULT_TIMES
        ]
    elseif stage == "G3"
        return [
            make_cell(; sigma = 1.0, L = 64,
                      gamma_c = _gamma_for(gamma_c, 1.0, 64),
                      T, chi = 96, dt = 0.05,
                      poles = 16, seed)
            for T in DEFAULT_TIMES
        ]
    elseif stage == "G3-sentinels"
        return [
            make_cell(;
                sigma = 1.0, L = 64,
                gamma_c = _gamma_for(gamma_c, 1.0, 64),
                T, chi, dt, poles = 16, seed,
            )
            for T in (16.0, 64.0), (chi, dt) in ((64, 0.05), (96, 0.025))
        ][:]
    elseif stage == "G3-L128"
        return [
            make_cell(;
                sigma = 1.0, L = 128,
                gamma_c = _gamma_for(gamma_c, 1.0, 128),
                T, chi = 128, dt = 0.05, poles = 16, seed,
            )
            for T in DEFAULT_TIMES
        ]
    elseif stage == "floor-l64"
        return [
            make_cell(; sigma, L = 64,
                      gamma_c = _gamma_for(gamma_c, sigma, 64),
                      T, chi = 96, dt = 0.05,
                      poles = 16, seed)
            for sigma in (1.25, 1.5), T in DEFAULT_TIMES
        ][:]
    elseif stage == "floor-collapse"
        return [
            make_cell(; sigma, L,
                      gamma_c = _gamma_for(gamma_c, sigma, L), T,
                      chi = L == 128 ? 128 : 96, dt = 0.05,
                      poles = 16, seed)
            for sigma in (1.0, 1.25, 1.5), L in (32, 128), T in DEFAULT_TIMES
        ][:]
    elseif stage == "sprint-size-1875"
        return _sprint_size_cells(gamma_c; seed)
    elseif stage == "sprint-trend"
        return _sprint_trend_cells(gamma_c; seed)
    elseif stage == "sprint-sentinels"
        return _sprint_sentinel_cells(gamma_c; seed)
    elseif stage == "sprint-gamma"
        return _sprint_gamma_cells(gamma_c; seed)
    elseif stage in ("sprint-A", "sprint-B")
        shard = last(stage)
        cells = Dict{String, Any}[]
        if shard == 'A'
            for component in ("G1", "G1-dt-sentinel", "G2", "G3", "G3-sentinels")
                for cell in stage_cells(component; gamma_c, seed)
                    mode = component in ("G1", "G1-dt-sentinel") ?
                        "ed-tdvp" : "tdvp"
                    push!(cells, _tag_sprint_cell!(
                        cell, "A", "baseline", gamma_c;
                        execution_mode = mode,
                        component_stage = component,
                    ))
                end
            end
            append!(cells, _sprint_size_cells(gamma_c; seed))
        else
            append!(cells, _sprint_trend_cells(gamma_c; seed))
        end
        append!(
            cells,
            filter(
                cell -> cell["shard_id"] == string(shard),
                _sprint_sentinel_cells(gamma_c; seed),
            ),
        )
        append!(
            cells,
            filter(
                cell -> cell["shard_id"] == string(shard),
                _sprint_gamma_cells(gamma_c; seed),
            ),
        )
        return cells
    end
    throw(ArgumentError("unknown stage: $stage"))
end

function select_poles(max_relative_error::Real, current_poles::Integer)
    max_relative_error <= 1.0e-6 && return Int(current_poles)
    current_poles == 16 && return 24
    current_poles == 24 &&
        error("24-pole OBC SOE still exceeds the 1e-6 coupling-error gate")
    throw(ArgumentError("pole escalation only supports 16 to 24"))
end

function resource_class(cell::AbstractDict)
    large_long_range = cell["sigma"] != "NN" && Int(cell["L"]) >= 128
    return large_long_range || Int(cell["chi"]) >= 128 ? "large" : "standard"
end

function _git_revision()
    try
        return readchomp(`git rev-parse HEAD`)
    catch
        return "unknown"
    end
end

_utc_timestamp() = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS")
_source_hash() = bytes2hex(sha256(read(@__FILE__)))

function _gamma_map_key(sigma, L::Integer)
    sigma_key = sigma isa AbstractString ? String(sigma) : string(Float64(sigma))
    return "$sigma_key:$L"
end

function build_gamma_map(scans::AbstractVector)
    isempty(scans) && throw(ArgumentError("at least one critical scan is required"))
    sources = Set(String(scan["code_source_sha256"]) for scan in scans)
    length(sources) == 1 ||
        error("critical scans were produced by different source revisions")
    revisions = Set(String(scan["code_revision"]) for scan in scans)
    length(revisions) == 1 ||
        error("critical scans were produced by different git revisions")
    entries = Dict{String, Any}()
    for scan in scans
        get(scan, "status", "") == "completed" ||
            error("critical scan is not completed")
        low = Float64(scan["Gamma_low"])
        high = Float64(scan["Gamma_high"])
        target = Float64(get(
            scan["parameters"], "target_width", get(scan, "target_width", 0.01)
        ))
        high >= low || error("critical scan interval is reversed")
        high - low <= target + 100eps(Float64) ||
            error("critical scan interval exceeds its registered target")
        key = _gamma_map_key(scan["sigma"], Int(scan["L"]))
        haskey(entries, key) && error("duplicate critical scan for $key")
        entries[key] = Dict{String, Any}(
            "Gamma_c" => Float64(scan["Gamma_c"]),
            "Gamma_low" => low,
            "Gamma_high" => high,
            "target_width" => target,
            "scan_parameter_hash" => String(scan["parameter_hash"]),
        )
    end
    map_hash = parameter_hash(entries)
    return Dict{String, Any}(
        "schema_version" => 1,
        "kind" => "track-c-gamma-map",
        "created_at" => _utc_timestamp(),
        "code_revision" => only(revisions),
        "source_sha256" => only(sources),
        "map_hash" => map_hash,
        "entries" => entries,
    )
end

function build_critical_shard_spec(
        shard_id::AbstractString, initial_estimates::AbstractDict
    )
    shard_id in ("A", "B") ||
        throw(ArgumentError("critical shard must be A or B"))
    definitions = if shard_id == "A"
        [
            (1.0, 12, 64, 0.01),
            (1.0, 16, 64, 0.01),
            (1.0, 20, 64, 0.01),
            (1.0, 64, 96, 0.005),
            (1.875, 32, 96, 0.002),
            (1.875, 64, 96, 0.002),
            (1.875, 128, 128, 0.002),
        ]
    else
        [
            (1.75, 64, 96, 0.002),
            (1.8, 64, 96, 0.002),
            (1.95, 64, 96, 0.002),
            (2.0, 64, 96, 0.002),
        ]
    end
    cells = Dict{String, Any}[]
    for (sigma, L, chi, target_width) in definitions
        scan_parameters = Dict{String, Any}(
            "sigma" => sigma,
            "L" => L,
            "initial_estimate" => _gamma_for(initial_estimates, sigma, L),
            "chi" => chi,
            "poles" => 16,
            "seed" => 86,
            "initial_half_width" => 0.2,
            "target_width" => target_width,
        )
        hash = parameter_hash(scan_parameters)
        parameters = Dict{String, Any}(
            scan_parameters...,
            "shard_id" => String(shard_id),
            "parameter_hash" => hash,
        )
        parameters["id"] = "critical-" * lowercase(shard_id) * "-" * hash[1:12]
        push!(cells, parameters)
    end
    return Dict{String, Any}(
        "schema_version" => 1,
        "kind" => "track-c-critical-shard",
        "shard_id" => String(shard_id),
        "created_at" => _utc_timestamp(),
        "code_revision" => _git_revision(),
        "code_source_sha256" => _source_hash(),
        "cells" => cells,
    )
end

function merge_campaign_runs(run_a::AbstractDict, run_b::AbstractDict)
    run_a["stage"] == "sprint-A" ||
        error("first campaign shard must be sprint-A")
    run_b["stage"] == "sprint-B" ||
        error("second campaign shard must be sprint-B")
    run_a["code_source_sha256"] == run_b["code_source_sha256"] ||
        error("campaign shards use different source SHA values")
    run_a["code_revision"] == run_b["code_revision"] ||
        error("campaign shards use different git revisions")
    run_a["gamma_map_hash"] == run_b["gamma_map_hash"] ||
        error("campaign shards use different Gamma_c maps")
    cells = vcat(deepcopy(run_a["cells"]), deepcopy(run_b["cells"]))
    hashes = String[cell["parameter_hash"] for cell in cells]
    length(unique(hashes)) == length(hashes) ||
        error("campaign shards contain duplicate parameter hashes")
    ids = String[cell["id"] for cell in cells]
    length(unique(ids)) == length(ids) ||
        error("campaign shards contain duplicate cell ids")
    return Dict{String, Any}(
        "schema_version" => 1,
        "kind" => "track-c-sprint-campaign",
        "created_at" => _utc_timestamp(),
        "code_revision" => run_a["code_revision"],
        "code_source_sha256" => run_a["code_source_sha256"],
        "gamma_map_hash" => run_a["gamma_map_hash"],
        "shards" => Dict(
            "A" => get(run_a, "run_id", "sprint-A"),
            "B" => get(run_b, "run_id", "sprint-B"),
        ),
        "cell_counts" => Dict(
            "A" => length(run_a["cells"]),
            "B" => length(run_b["cells"]),
        ),
        "cells" => cells,
    )
end

function build_run_spec(
        stage::AbstractString;
        gamma_c,
        seed::Integer = 86,
        created_at::AbstractString = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
    )
    gamma_entries = gamma_c isa AbstractDict && haskey(gamma_c, "entries") ?
        gamma_c["entries"] : gamma_c
    gamma_map_hash = gamma_c isa AbstractDict && haskey(gamma_c, "map_hash") ?
        String(gamma_c["map_hash"]) :
        gamma_entries isa AbstractDict ? parameter_hash(gamma_entries) :
        parameter_hash(Dict("Gamma_c" => Float64(gamma_entries)))
    cells = map(stage_cells(stage; gamma_c = gamma_entries, seed)) do parameters
        hash = parameter_hash(parameters)
        Dict{String, Any}(
            "id" => lowercase(stage) * "-" * hash[1:12],
            "parameters" => parameters,
            "parameter_hash" => hash,
            "resource_class" => resource_class(parameters),
            "status" => "pending",
            "result" => nothing,
            "error" => nothing,
            "lease" => nothing,
        )
    end
    return Dict{String, Any}(
        "schema_version" => 1,
        "track" => "C",
        "stage" => String(stage),
        "created_at" => String(created_at),
        "code_revision" => _git_revision(),
        "code_source_sha256" => _source_hash(),
        "gamma_map_hash" => gamma_map_hash,
        "sources" => Dict(
            "challenge" => "https://github.com/QuantumBFS/quantum.harness/issues/86",
            "paper" => "https://arxiv.org/abs/1612.07437",
        ),
        "physics" => Dict(
            "hamiltonian" => "-sum_{i<j} |i-j|^(-(1+sigma)) ZiZj - Gamma(t) sum_i Xi",
            "J" => 1.0,
            "hbar" => 1.0,
            "boundary" => "open",
            "interaction" => "ferromagnetic",
            "pauli_normalization" => "eigenvalues +/-1",
        ),
        "protocol" => Dict(
            "Gamma_initial" => "2*Gamma_c",
            "Gamma_final" => 0.0,
            "ramp" => "linear",
            "paper_horizontal_axis" => "T",
            "tau_Q_definition" => "T/2",
        ),
        "acceptance" => Dict(
            "mpo_max_relative_error" => 1.0e-6,
            "norm_drift" => 1.0e-8,
            "parity_drift" => 1.0e-6,
            "fit_indices" => [2, 3, 4, 5],
            "scientific_failure_may_change_fit_window" => false,
        ),
        "cells" => cells,
    )
end

function write_run_json(path::AbstractString, run::AbstractDict)
    mkpath(dirname(path))
    temporary = path * ".tmp." * string(getpid())
    open(temporary, "w") do io
        JSON.print(io, run, 2)
        println(io)
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 ||
            error("failed to fsync JSON artifact")
    end
    mv(temporary, path; force = true)
    return path
end

read_run_json(path::AbstractString) = JSON.parsefile(path)

function update_cell_status!(
        path::AbstractString, cell_id::AbstractString, status::AbstractString;
        result = nothing, error = nothing, lease_token = nothing
    )
    status in ("pending", "running", "completed", "failed", "blocked") ||
        throw(ArgumentError("unsupported cell status: $status"))
    lock_path = path * ".lock"
    mkpath(dirname(lock_path))
    return open(lock_path, "w") do lock_io
        ccall(:flock, Cint, (Cint, Cint), Base.fd(lock_io), 2) == 0 ||
            error("failed to acquire run.json lock")
        try
            run = read_run_json(path)
            matches = filter(cell -> cell["id"] == cell_id, run["cells"])
            length(matches) == 1 || throw(KeyError(cell_id))
            cell = only(matches)
            if status in ("completed", "failed", "blocked")
                cell["status"] == "running" ||
                    Base.error(
                        "cell $cell_id cannot transition to $status from $(cell["status"])"
                    )
                active_lease = get(cell, "lease", nothing)
                expected_token = isnothing(active_lease) ?
                    nothing : get(active_lease, "token", nothing)
                expected_token == lease_token ||
                    Base.error("cell $cell_id lease token no longer matches")
            end
            cell["status"] = String(status)
            cell["result"] = result
            cell["error"] = error
            cell["updated_at"] = _utc_timestamp()
            cell["lease"] = status == "running" ? Dict(
                "owner" => _lease_identity(),
                "token" => string(uuid4()),
                "started_at" => _utc_timestamp(),
                "hostname" => gethostname(),
                "pid" => getpid(),
            ) : nothing
            write_run_json(path, run)
            return cell
        finally
            ccall(:flock, Cint, (Cint, Cint), Base.fd(lock_io), 8)
        end
    end
end

function pending_cells(
        run::AbstractDict;
        resource = nothing,
        duration = nothing,
    )
    cells = filter(run["cells"]) do cell
        cell["status"] in ("pending", "failed") ||
            (cell["status"] == "running" && _running_lease_is_stale(cell))
    end
    !isnothing(resource) &&
        (cells = filter(cell -> cell["resource_class"] == resource, cells))
    if duration == "short"
        cells = filter(cell -> Float64(cell["parameters"]["T"]) <= 64, cells)
    elseif duration == "long"
        cells = filter(cell -> Float64(cell["parameters"]["T"]) > 64, cells)
    elseif !isnothing(duration)
        throw(ArgumentError("duration must be short, long, or nothing"))
    end
    return cells
end

_lease_identity() = haskey(ENV, "SLURM_JOB_ID") ?
    "slurm:" * ENV["SLURM_JOB_ID"] :
    "local:" * gethostname() * ":" * string(getpid())

function _running_lease_is_stale(cell::AbstractDict)
    lease = get(cell, "lease", nothing)
    isnothing(lease) && return true
    owner = String(get(lease, "owner", ""))
    owner == _lease_identity() && return false
    if startswith(owner, "slurm:")
        job_id = split(owner, ':'; limit = 2)[2]
        return !_slurm_job_is_live(job_id)
    end
    started_at = get(lease, "started_at", nothing)
    isnothing(started_at) && return true
    try
        age = now(UTC) - DateTime(String(started_at))
        return age > Hour(1)
    catch
        return true
    end
end

function _slurm_job_is_live(job_id::AbstractString)
    occursin(r"^[0-9]+(?:_[0-9]+)?$", job_id) || return false
    isnothing(Sys.which("squeue")) && return false
    try
        state = strip(readchomp(`squeue -h -j $job_id -o %T`))
        return state in ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING")
    catch
        return true
    end
end

function _claim_cell!(path::AbstractString, cell_id::AbstractString)
    lock_path = path * ".lock"
    return open(lock_path, "w") do lock_io
        ccall(:flock, Cint, (Cint, Cint), Base.fd(lock_io), 2) == 0 ||
            error("failed to acquire run.json lock")
        try
            run = read_run_json(path)
            matches = filter(cell -> cell["id"] == cell_id, run["cells"])
            length(matches) == 1 || throw(KeyError(cell_id))
            cell = only(matches)
            cell["status"] == "completed" &&
                error("cell $cell_id is already completed")
            if cell["status"] == "running" && !_running_lease_is_stale(cell)
                error("cell $cell_id is already claimed by the current allocation")
            end
            cell["status"] in ("pending", "failed", "running") ||
                error("cell $cell_id cannot be claimed from status $(cell["status"])")
            cell["status"] = "running"
            cell["error"] = nothing
            cell["updated_at"] = _utc_timestamp()
            cell["lease"] = Dict(
                "owner" => _lease_identity(),
                "token" => string(uuid4()),
                "started_at" => _utc_timestamp(),
                "hostname" => gethostname(),
                "pid" => getpid(),
            )
            write_run_json(path, run)
            return deepcopy(cell)
        finally
            ccall(:flock, Cint, (Cint, Cint), Base.fd(lock_io), 8)
        end
    end
end

function _numerical_gate(manifest::AbstractDict)
    norm_ok = Float64(manifest["norm_drift"]) <= 1.0e-8
    parity_ok = Float64(manifest["parity_drift"]) <= 1.0e-6
    mpo_ok = !haskey(manifest, "mpo_max_relative_error") ||
             Float64(manifest["mpo_max_relative_error"]) <= 1.0e-6
    return Dict{String, Any}(
        "passed" => norm_ok && parity_ok && mpo_ok,
        "norm_ok" => norm_ok,
        "parity_ok" => parity_ok,
        "mpo_ok" => mpo_ok,
    )
end

function execute_cell(
        run_path::AbstractString, cell_id::AbstractString,
        output_root::AbstractString,
    )
    run = read_run_json(run_path)
    matches = filter(cell -> cell["id"] == cell_id, run["cells"])
    length(matches) == 1 || throw(KeyError(cell_id))
    cell_record = only(matches)
    parameters = cell_record["parameters"]
    stage = String(run["stage"])
    cell_directory = joinpath(output_root, cell_id)
    production_cell =
        Int(parameters["L"]) >= 12 || Float64(parameters["T"]) >= 4
    if production_cell && !haskey(ENV, "SLURM_JOB_ID") &&
            get(ENV, "TRACK_C_ALLOW_LOCAL", "0") != "1"
        error(
            "production Track C cells require Slurm; set TRACK_C_ALLOW_LOCAL=1 only for an intentional local run"
        )
    end
    claimed_cell = _claim_cell!(run_path, cell_id)
    lease_token = claimed_cell["lease"]["token"]

    try
        execution_mode = get(
            parameters, "execution_mode",
            stage in ("G1", "G1-dt-sentinel") ? "ed-tdvp" : "tdvp",
        )
        result = if execution_mode == "ed-tdvp"
            ed = run_ed_trajectory(
                parameters, joinpath(cell_directory, "ed")
            )
            tdvp = run_tdvp_trajectory(
                parameters, joinpath(cell_directory, "tdvp")
            )
            Dict{String, Any}(
                "ed" => ed,
                "tdvp" => tdvp,
                "numerical_gate" => Dict(
                    "ed" => _numerical_gate(ed),
                    "tdvp" => _numerical_gate(tdvp),
                ),
            )
        elseif execution_mode == "tdvp"
            tdvp = run_tdvp_trajectory(
                parameters, joinpath(cell_directory, "tdvp")
            )
            Dict{String, Any}(
                "tdvp" => tdvp,
                "numerical_gate" => Dict("tdvp" => _numerical_gate(tdvp)),
            )
        else
            error("unsupported execution_mode: $execution_mode")
        end
        update_cell_status!(
            run_path, cell_id, "completed"; result, lease_token
        )
        return result
    catch exception
        message = sprint(showerror, exception)
        update_cell_status!(
            run_path, cell_id, "failed"; error = message, lease_token
        )
        rethrow()
    end
end

end
