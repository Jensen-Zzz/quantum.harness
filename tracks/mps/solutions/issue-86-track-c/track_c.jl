#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "Issue86TrackC.jl"))
using .Issue86TrackC
using Dates
using JSON
using Printf

function usage()
    println("""
    Track C unified driver

      track_c.jl g0 OUTPUT_JSON [L] [SIGMA]
      track_c.jl critical OUTPUT_JSON SIGMA L INITIAL_GAMMA [CHI] [TARGET_WIDTH]
      track_c.jl generate-critical-shard A|B OUTPUT_JSON INITIAL_GAMMA_MAP.json
      track_c.jl critical-cell CRITICAL_SPEC.json CELL_ID OUTPUT_ROOT
      track_c.jl critical-pending CRITICAL_SPEC.json OUTPUT_ROOT
      track_c.jl merge-gamma OUTPUT_JSON SCAN_JSON...
      track_c.jl generate-sprint A|B RUN_JSON GAMMA_MAP.json
      track_c.jl merge-campaign RUN_A.json RUN_B.json CAMPAIGN.json
      track_c.jl analyze-sprint CAMPAIGN.json OUTPUT_DIRECTORY [THEORY.json]
      track_c.jl generate STAGE RUN_JSON GAMMA_C_OR_JSON
      track_c.jl generate-floor-l64 G3_RUN RUN_JSON GAMMA_MAP.json
      track_c.jl generate-floor-collapse G3_RUN FLOOR_L64_RUN RUN_JSON GAMMA_MAP.json
      track_c.jl cell RUN_JSON CELL_ID OUTPUT_ROOT
      track_c.jl pending RUN_JSON [RESOURCE_CLASS] [short|long]
      track_c.jl analyze RUN_JSON OUTPUT_DIRECTORY
      track_c.jl compare-sentinels BASELINE_RUN SENTINEL_RUN OUTPUT_JSON
      track_c.jl g3-upgrades BASELINE_RUN SENTINEL_RUN UPGRADE_RUN
      track_c.jl converge-g3 BASELINE_RUN SENTINEL_RUN CONVERGED_RUN [UPGRADE_RUN]
      track_c.jl promote-g3-l128 SCREENED_G3_RUN G3_L128_RUN CONVERGED_RUN
      track_c.jl collapse G3_RUN FLOOR_L64_RUN FLOOR_COLLAPSE_RUN OUTPUT_JSON
    """)
end

function parse_gamma_source(value::AbstractString)
    if isfile(value)
        return JSON.parsefile(value)
    end
    return parse(Float64, value)
end

function build_provenance_aware_gamma_map(scans::AbstractVector)
    sources = Set(String(scan["code_source_sha256"]) for scan in scans)
    length(sources) == 1 ||
        error("critical scans were produced by different source revisions")
    revisions = sort!(unique(
        String(scan["code_revision"]) for scan in scans
    ))
    if length(revisions) == 1
        gamma_map = build_gamma_map(scans)
        gamma_map["scan_code_revisions"] = revisions
        gamma_map["provenance_policy"] = "single-revision"
        return gamma_map
    end

    equivalent_scans = deepcopy(scans)
    for scan in equivalent_scans
        scan["code_revision"] = first(revisions)
    end
    gamma_map = build_gamma_map(equivalent_scans)
    gamma_map["code_revision"] = Issue86TrackC._git_revision()
    gamma_map["scan_code_revisions"] = revisions
    gamma_map["provenance_policy"] = "identical-driver-source-sha256"
    return gamma_map
end

function write_curve_csv(path, run)
    open(path, "w") do io
        println(
            io,
            "cell_id,sigma,L,T,tau_Q,kink_density,kink_density_bulk,n_corr,status",
        )
        for cell in sort!(copy(run["cells"]); by = entry -> (
                string(entry["parameters"]["sigma"]),
                Int(entry["parameters"]["L"]),
                Float64(entry["parameters"]["T"]),
            ))
            parameters = cell["parameters"]
            kink = ""
            bulk_kink = ""
            n_corr = ""
            if cell["status"] == "completed" && haskey(cell["result"], "tdvp")
                manifest = cell["result"]["tdvp"]
                kink = manifest["final_kink_density"]
                bulk_kink = get(manifest, "final_kink_density_bulk", "")
                if haskey(manifest, "correlation_kink_density") &&
                        !isnothing(manifest["correlation_kink_density"])
                    n_corr = manifest["correlation_kink_density"]
                elseif haskey(manifest, "final_correlations")
                    correlations = Float64.(manifest["final_correlations"])
                    if length(correlations) >= 3
                        last_distance = min(10, length(correlations))
                        try
                            estimate = fit_correlation_length(
                                correlations; distances = 2:last_distance
                            )
                            n_corr = estimate["n_corr"]
                        catch
                            n_corr = ""
                        end
                    end
                end
            end
            println(
                io,
                join((
                    cell["id"], parameters["sigma"], parameters["L"],
                    parameters["T"], parameters["tau_Q"], kink, bulk_kink, n_corr,
                    cell["status"],
                ), ","),
            )
        end
    end
    return path
end

function write_report(path, run, summary)
    open(path, "w") do io
        println(io, "# Track C ", run["stage"], " report")
        println(io)
        println(io, "- Status: **", summary["status"], "**")
        println(io, "- Completed cells: ",
                count(cell -> cell["status"] == "completed", run["cells"]),
                "/", length(run["cells"]))
        println(io, "- Code revision: `", run["code_revision"], "`")
        println(io, "- Protocol: OBC, J=ℏ=1, Γ: 2Γ_c→0, τ_Q=T/2.")
        completion_times = String[]
        for cell in run["cells"]
            cell["status"] == "completed" || continue
            for method in ("tdvp", "ed")
                haskey(cell["result"], method) || continue
                manifest = cell["result"][method]
                haskey(manifest, "completed_at") &&
                    push!(completion_times, manifest["completed_at"])
            end
        end
        !isempty(completion_times) &&
            println(io, "- Last completed trajectory: ", maximum(completion_times), " UTC.")
        if haskey(summary, "mu")
            @printf(io, "- Fitted exponent μ: %.8g\n", summary["mu"])
        end
        if haskey(summary, "max_relative_residual")
            @printf(
                io, "- Maximum registered-window residual: %.4g\n",
                summary["max_relative_residual"],
            )
        end
        println(io)
        println(io, "Machine-readable gate details: `gate-summary.json`.")
        println(io, "Curve data: `curve.csv`; render with `plot_curve.jl`.")
    end
    return path
end

function compare_sentinels(baseline, sentinels)
    sentinel_stage = sentinels["stage"]
    all(cell -> cell["status"] == "completed", baseline["cells"]) ||
        error("baseline run is incomplete")
    all(cell -> cell["status"] == "completed", sentinels["cells"]) ||
        error("sentinel run is incomplete")

    if sentinel_stage == "G1-dt-sentinel"
        sentinel = only(sentinels["cells"])
        parameters = sentinel["parameters"]
        candidates = filter(baseline["cells"]) do cell
            cell["parameters"]["L"] == parameters["L"] &&
            cell["parameters"]["T"] == parameters["T"]
        end
        baseline_result = only(candidates)["result"]
        method_checks = Dict{String, Any}()
        for method in ("ed", "tdvp")
            reference = baseline_result[method]
            refined = sentinel["result"][method]
            final_relative = abs(
                refined["final_kink_density"] - reference["final_kink_density"]
            ) / max(abs(reference["final_kink_density"]), eps(Float64))
            trace_reference = reference["checkpoint_kink_density"]
            trace_refined = refined["checkpoint_kink_density"]
            length(trace_reference) == length(trace_refined) ||
                error("G1 $method time-step checkpoint grids do not match")
            trace_difference = maximum(abs.(
                Float64.(trace_reference) .- Float64.(trace_refined)
            ))
            method_checks[method] = Dict(
                "passed" => final_relative <= 0.01 && trace_difference <= 0.01,
                "final_relative_difference" => final_relative,
                "trajectory_max_absolute_difference" => trace_difference,
            )
        end
        return Dict{String, Any}(
            "gate" => "G1-dt-sentinel",
            "passed" => all(check["passed"] for check in values(method_checks)),
            "methods" => method_checks,
        )
    elseif sentinel_stage == "G3-sentinels"
        baseline_by_T = Dict(
            Float64(cell["parameters"]["T"]) => cell["result"]["tdvp"]
            for cell in baseline["cells"]
        )
        sentinel_manifests = [
            merge(
                Dict("parameters" => cell["parameters"]),
                cell["result"]["tdvp"],
            )
            for cell in sentinels["cells"]
        ]
        return evaluate_g3_sentinels(baseline_by_T, sentinel_manifests)
    end
    error("unsupported sentinel stage: $sentinel_stage")
end

isempty(ARGS) && (usage(); exit(2))
command = popfirst!(ARGS)

if command in ("help", "-h", "--help")
    usage()
    exit(0)
elseif command == "g0"
    length(ARGS) in (1, 2, 3) || (usage(); exit(2))
    output = ARGS[1]
    L = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
    sigma = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
    run_g0(output; L, sigma)
elseif command == "critical"
    4 <= length(ARGS) <= 6 || (usage(); exit(2))
    output = ARGS[1]
    sigma = parse(Float64, ARGS[2])
    L = parse(Int, ARGS[3])
    estimate = parse(Float64, ARGS[4])
    chi = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 96
    target_width = length(ARGS) == 6 ? parse(Float64, ARGS[6]) : 0.01
    run_critical_scan(
        output; sigma, L, initial_estimate = estimate, chi, target_width
    )
elseif command == "generate-critical-shard"
    length(ARGS) == 3 || (usage(); exit(2))
    shard, output, initial_map_path = ARGS
    isfile(output) && error("refusing to overwrite existing critical spec: $output")
    spec = build_critical_shard_spec(
        uppercase(shard), read_run_json(initial_map_path)
    )
    write_run_json(output, spec)
    println(output)
elseif command == "critical-cell"
    length(ARGS) == 3 || (usage(); exit(2))
    spec_path, cell_id, output_root = ARGS
    spec = read_run_json(spec_path)
    spec["kind"] == "track-c-critical-shard" ||
        error("input is not a Track C critical shard")
    spec["code_source_sha256"] == Issue86TrackC._source_hash() ||
        error("critical shard source SHA does not match the current driver")
    matches = filter(cell -> cell["id"] == cell_id, spec["cells"])
    length(matches) == 1 || throw(KeyError(cell_id))
    cell = only(matches)
    output = joinpath(output_root, cell_id * ".json")
    run_critical_scan(
        output;
        sigma = Float64(cell["sigma"]),
        L = Int(cell["L"]),
        initial_estimate = Float64(cell["initial_estimate"]),
        chi = Int(cell["chi"]),
        poles = Int(cell["poles"]),
        seed = Int(cell["seed"]),
        initial_half_width = Float64(cell["initial_half_width"]),
        target_width = Float64(cell["target_width"]),
    )
    println(output)
elseif command == "critical-pending"
    length(ARGS) == 2 || (usage(); exit(2))
    spec = read_run_json(ARGS[1])
    output_root = ARGS[2]
    for cell in spec["cells"]
        cell_output = joinpath(output_root, cell["id"] * ".json")
        completed = isfile(cell_output) &&
            get(read_run_json(cell_output), "status", "") == "completed"
        !completed && println(cell["id"])
    end
elseif command == "merge-gamma"
    length(ARGS) >= 2 || (usage(); exit(2))
    output = popfirst!(ARGS)
    isfile(output) && error("refusing to overwrite existing Gamma_c map: $output")
    gamma_map = build_provenance_aware_gamma_map(read_run_json.(ARGS))
    write_run_json(output, gamma_map)
    println(output)
elseif command == "generate-sprint"
    length(ARGS) == 3 || (usage(); exit(2))
    shard, output, gamma_map_path = ARGS
    shard = uppercase(shard)
    shard in ("A", "B") || error("sprint shard must be A or B")
    isfile(output) && error("refusing to overwrite existing run.json: $output")
    gamma_map = read_run_json(gamma_map_path)
    gamma_map["kind"] == "track-c-gamma-map" ||
        error("input is not a Track C Gamma_c map")
    gamma_map["source_sha256"] == Issue86TrackC._source_hash() ||
        error("Gamma_c map source SHA does not match the current driver")
    run = build_run_spec("sprint-$shard"; gamma_c = gamma_map)
    run["run_id"] = "track-c-sprint-$shard"
    write_run_json(output, run)
    println(output)
elseif command == "merge-campaign"
    length(ARGS) == 3 || (usage(); exit(2))
    run_a = read_run_json(ARGS[1])
    run_b = read_run_json(ARGS[2])
    isfile(ARGS[3]) && error("refusing to overwrite campaign: $(ARGS[3])")
    campaign = merge_campaign_runs(run_a, run_b)
    write_run_json(ARGS[3], campaign)
    println(ARGS[3])
elseif command == "analyze-sprint"
    length(ARGS) in (2, 3) || (usage(); exit(2))
    campaign_path, output_directory = ARGS[1:2]
    theories = length(ARGS) == 3 ? read_run_json(ARGS[3]) : nothing
    campaign = read_run_json(campaign_path)
    summary = analyze_sprint_campaign(campaign; theories)
    mkpath(output_directory)
    write_run_json(joinpath(output_directory, "sprint-summary.json"), summary)
    rows = campaign_rows(campaign)
    open(joinpath(output_directory, "sprint-rows.csv"), "w") do io
        println(
            io,
            "cell_id,component,shard,variant,sigma,L,T,tau_Q,kink_density_bulk,kink_density,n_corr",
        )
        for row in rows
            println(io, join((
                row["cell_id"], row["component_stage"], row["shard_id"],
                row["variant"], row["sigma"], row["L"], row["T"], row["tau_Q"],
                row["kink_density_bulk"], row["kink_density"],
                something(row["correlation_kink_density"], ""),
            ), ","))
        end
    end
    println(joinpath(output_directory, "sprint-summary.json"))
elseif command == "generate"
    length(ARGS) == 3 || (usage(); exit(2))
    stage, output, gamma_source = ARGS
    stage in ("floor-l64", "floor-collapse") &&
        error("challenge stages require the G3-gated generate-floor-* commands")
    isfile(output) && error("refusing to overwrite existing run.json: $output")
    run = build_run_spec(stage; gamma_c = parse_gamma_source(gamma_source))
    write_run_json(output, run)
    println(output)
elseif command == "generate-floor-l64"
    length(ARGS) == 3 || (usage(); exit(2))
    g3_path, output, gamma_source = ARGS
    g3_run = read_run_json(g3_path)
    g3_run["stage"] == "G3-converged" ||
        error("use converge-g3 before entering the challenge floor")
    g3_summary = analyze_run(g3_run)
    g3_summary["status"] == "passed" ||
        error("G3 must pass before generating the challenge floor")
    isfile(output) && error("refusing to overwrite existing run.json: $output")
    run = build_run_spec(
        "floor-l64"; gamma_c = parse_gamma_source(gamma_source)
    )
    run["prerequisite"] = Dict(
        "G3_run" => abspath(g3_path),
        "G3_status" => "passed",
    )
    write_run_json(output, run)
    println(output)
elseif command == "generate-floor-collapse"
    length(ARGS) == 4 || (usage(); exit(2))
    g3_path, floor_l64_path, output, gamma_source = ARGS
    g3_run = read_run_json(g3_path)
    g3_run["stage"] == "G3-converged" ||
        error("finite-time collapse requires a converged G3 curve")
    g3_summary = analyze_run(g3_run)
    g3_summary["status"] == "passed" ||
        error("G3 must pass before generating finite-time collapse cells")
    floor_l64 = read_run_json(floor_l64_path)
    all(cell -> cell["status"] == "completed", floor_l64["cells"]) ||
        error("the L=64 challenge floor must complete before L=32,128")
    isfile(output) && error("refusing to overwrite existing run.json: $output")
    run = build_run_spec(
        "floor-collapse"; gamma_c = parse_gamma_source(gamma_source)
    )
    run["prerequisite"] = Dict(
        "G3_run" => abspath(g3_path),
        "G3_status" => "passed",
        "floor_l64_run" => abspath(floor_l64_path),
        "floor_l64_status" => "completed",
    )
    write_run_json(output, run)
    println(output)
elseif command == "cell"
    length(ARGS) == 3 || (usage(); exit(2))
    execute_cell(ARGS[1], ARGS[2], ARGS[3])
elseif command == "pending"
    1 <= length(ARGS) <= 3 || (usage(); exit(2))
    run = read_run_json(ARGS[1])
    resource = length(ARGS) >= 2 ? ARGS[2] : nothing
    duration = length(ARGS) >= 3 ? ARGS[3] : nothing
    for cell in pending_cells(run; resource, duration)
        println(cell["id"])
    end
elseif command == "analyze"
    length(ARGS) == 2 || (usage(); exit(2))
    run_path, output_directory = ARGS
    mkpath(output_directory)
    run = read_run_json(run_path)
    summary = analyze_run(run)
    write_run_json(joinpath(output_directory, "gate-summary.json"), summary)
    write_curve_csv(joinpath(output_directory, "curve.csv"), run)
    write_report(joinpath(output_directory, "report.md"), run, summary)
    println(joinpath(output_directory, "report.md"))
elseif command == "compare-sentinels"
    length(ARGS) == 3 || (usage(); exit(2))
    baseline = read_run_json(ARGS[1])
    sentinels = read_run_json(ARGS[2])
    comparison = compare_sentinels(baseline, sentinels)
    write_run_json(ARGS[3], comparison)
    println(ARGS[3])
elseif command == "g3-upgrades"
    length(ARGS) == 3 || (usage(); exit(2))
    baseline = read_run_json(ARGS[1])
    sentinels = read_run_json(ARGS[2])
    comparison = compare_sentinels(baseline, sentinels)
    isempty(comparison["dt_failed_T"]) ||
        error("G3 dt sentinels failed; do not substitute chi=128 for time-step convergence")
    affected = Set(Float64.(comparison["upgrade_T"]))
    isempty(affected) && error("G3 sentinels pass; no chi=128 upgrades are needed")
    upgrade = deepcopy(baseline)
    upgrade["stage"] = "G3-upgrades"
    upgrade["created_at"] = string(Dates.now(Dates.UTC))
    upgrade["cells"] = filter(upgrade["cells"]) do cell
        Float64(cell["parameters"]["T"]) in affected
    end
    for cell in upgrade["cells"]
        cell["parameters"]["chi"] = 128
        cell["parameter_hash"] = parameter_hash(cell["parameters"])
        cell["id"] = "g3-upgrade-" * cell["parameter_hash"][1:12]
        cell["resource_class"] = "large"
        cell["status"] = "pending"
        cell["result"] = nothing
        cell["error"] = nothing
    end
    isfile(ARGS[3]) && error("refusing to overwrite existing run.json: $(ARGS[3])")
    write_run_json(ARGS[3], upgrade)
    println(ARGS[3])
elseif command == "converge-g3"
    length(ARGS) in (3, 4) || (usage(); exit(2))
    baseline_path, sentinel_path, output = ARGS[1:3]
    baseline = read_run_json(baseline_path)
    sentinels = read_run_json(sentinel_path)
    comparison = compare_sentinels(baseline, sentinels)
    isempty(comparison["dt_failed_T"]) ||
        error("G3 time-step convergence failed; convergence cannot be promoted")
    converged = deepcopy(baseline)
    sources = Dict{String, Any}(
        "baseline" => abspath(baseline_path),
        "sentinels" => abspath(sentinel_path),
        "sentinel_gate" => comparison,
    )
    if !comparison["passed"]
        length(ARGS) == 4 ||
            error("failed sentinels require a completed G3-upgrades run")
        upgrade_path = ARGS[4]
        upgrades = read_run_json(upgrade_path)
        upgrades["stage"] == "G3-upgrades" ||
            error("upgrade input is not a G3-upgrades run")
        all(cell -> cell["status"] == "completed", upgrades["cells"]) ||
            error("all affected chi=128 upgrades must complete")
        expected_T = Set(Float64.(comparison["chi_upgrade_T"]))
        observed_T = Set(
            Float64(cell["parameters"]["T"]) for cell in upgrades["cells"]
        )
        observed_T == expected_T ||
            error("G3 upgrade run does not exactly match affected T values")
        replacements = Dict(
            Float64(cell["parameters"]["T"]) => cell
            for cell in upgrades["cells"]
        )
        baseline_by_T = Dict(
            Float64(cell["parameters"]["T"]) => cell for cell in baseline["cells"]
        )
        post_upgrade_differences = Dict{String, Float64}()
        for (T, replacement) in replacements
            reference = Float64(
                baseline_by_T[T]["result"]["tdvp"]["final_kink_density"]
            )
            upgraded = Float64(
                replacement["result"]["tdvp"]["final_kink_density"]
            )
            difference = abs(upgraded - reference) /
                         max(abs(reference), eps(Float64))
            post_upgrade_differences[string(T)] = difference
            difference <= 0.02 ||
                error("chi=128 upgrade at T=$T remains more than 2% from chi=96")
            all(
                gate["passed"]
                for gate in values(replacement["result"]["numerical_gate"])
            ) || error("chi=128 upgrade at T=$T fails a numerical gate")
        end
        for index in eachindex(converged["cells"])
            T = Float64(converged["cells"][index]["parameters"]["T"])
            haskey(replacements, T) &&
                (converged["cells"][index] = deepcopy(replacements[T]))
        end
        sources["upgrades"] = abspath(upgrade_path)
        sources["post_upgrade_relative_difference"] =
            post_upgrade_differences
    end
    converged["stage"] = "G3-converged"
    converged["convergence_sources"] = sources
    converged["created_at"] = string(Dates.now(Dates.UTC))
    summary = analyze_run(converged)
    converged["acceptance_status"] = summary["status"]
    isfile(output) && error("refusing to overwrite existing run.json: $output")
    write_run_json(output, converged)
    println(output)
elseif command == "promote-g3-l128"
    length(ARGS) == 3 || (usage(); exit(2))
    baseline_path, fallback_path, output = ARGS
    baseline = read_run_json(baseline_path)
    fallback = read_run_json(fallback_path)
    baseline["stage"] == "G3-converged" ||
        error("first input must be the sentinel-screened G3-converged run")
    baseline_summary = analyze_run(baseline)
    baseline_summary["status"] == "failed" ||
        error("L=128 fallback is only needed after the L=64 literature gate fails")
    baseline_summary["numerical_gates_passed"] ||
        error("L=128 fallback cannot override L=64 numerical-gate failures")
    !baseline_summary["literature_gate_passed"] ||
        error("L=128 fallback is only for a failed literature/finite-size screen")
    haskey(baseline, "convergence_sources") ||
        error("L=64 G3 must finish sentinel convergence before fallback")
    fallback["stage"] == "G3-L128" ||
        error("second input must be a G3-L128 fallback run")
    fallback_summary = analyze_run(fallback)
    fallback_summary["status"] == "passed" ||
        error("G3-L128 must pass before promotion")
    converged = deepcopy(baseline)
    converged["stage"] = "G3-converged"
    converged["fallback_validation"] = merge(
        Dict(
            "status" => "passed",
            "run" => abspath(fallback_path),
            "reason" => "L=128 literature curve passes after converged L=64 screen failed",
        ),
        fallback_summary,
    )
    converged["created_at"] = string(Dates.now(Dates.UTC))
    converged["acceptance_status"] = "passed"
    isfile(output) && error("refusing to overwrite existing run.json: $output")
    write_run_json(output, converged)
    println(output)
elseif command == "collapse"
    length(ARGS) == 4 || (usage(); exit(2))
    source_runs = read_run_json.(ARGS[1:3])
    source_runs[1]["stage"] == "G3-converged" ||
        error("collapse requires the converged G3 curve as its first input")
    analyze_run(source_runs[1])["status"] == "passed" ||
        error("converged G3 curve must pass before challenge collapse")
    all(
        cell["status"] == "completed"
        for run in source_runs for cell in run["cells"]
    ) || error("all G3/floor cells must complete before collapse")
    rows = Dict{String, Any}[]
    for run in source_runs, cell in run["cells"]
        parameters = cell["parameters"]
        parameters["sigma"] == "NN" && continue
        push!(rows, Dict{String, Any}(
            "sigma" => Float64(parameters["sigma"]),
            "L" => Int(parameters["L"]),
            "T" => Float64(parameters["T"]),
            "tau_Q" => Float64(parameters["tau_Q"]),
            "kink_density" =>
                Float64(cell["result"]["tdvp"]["final_kink_density"]),
        ))
    end
    results = Dict{String, Any}()
    overall_passed = Ref(true)
    for sigma in (1.0, 1.25, 1.5)
        selected = filter(row -> row["sigma"] == sigma, rows)
        Set(row["L"] for row in selected) == Set([32, 64, 128]) ||
            error("collapse sigma=$sigma does not contain L=32,64,128")
        collapse = fit_collapse_mu(selected)
        l64 = sort!(
            filter(row -> row["L"] == 64, selected);
            by = row -> row["T"],
        )
        direct = fit_power_law(
            [row["tau_Q"] for row in l64],
            [row["kink_density"] for row in l64];
            indices = 2:5,
        )
        gate = evaluate_collapse(collapse, direct["mu"])
        gate["fit"] = collapse
        gate["single_L64_fit"] = direct
        results[string(sigma)] = gate
        overall_passed[] &= gate["passed"]
    end
    report = Dict{String, Any}(
        "gate" => "challenge-floor-collapse",
        "status" => overall_passed[] ? "passed" : "failed",
        "by_sigma" => results,
        "disputed_window_entered" => false,
    )
    write_run_json(ARGS[4], report)
    println(ARGS[4])
else
    usage()
    error("unknown command: $command")
end
