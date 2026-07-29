using Test

if !isdefined(Main, :Issue86TrackC)
    const PROVENANCE_TRACK_C_ROOT = normpath(joinpath(@__DIR__, ".."))
    include(joinpath(PROVENANCE_TRACK_C_ROOT, "src", "Issue86TrackC.jl"))
    using .Issue86TrackC
else
    const PROVENANCE_TRACK_C_ROOT = TRACK_C_ROOT
end

@testset "Gamma-map CLI accepts source-identical scan revisions" begin
    scan(sigma, revision) = Dict{String, Any}(
        "status" => "completed",
        "sigma" => sigma,
        "L" => 64,
        "Gamma_c" => 1.3,
        "Gamma_low" => 1.299,
        "Gamma_high" => 1.301,
        "parameter_hash" => Issue86TrackC.parameter_hash(
            Dict("sigma" => sigma, "L" => 64),
        ),
        "code_revision" => revision,
        "code_source_sha256" => "source-sha",
        "parameters" => Dict("target_width" => 0.002),
    )

    mktempdir() do directory
        scan_a_path = joinpath(directory, "scan-a.json")
        scan_b_path = joinpath(directory, "scan-b.json")
        output_path = joinpath(directory, "gamma-map.json")
        Issue86TrackC.write_run_json(scan_a_path, scan(1.875, "revision-a"))
        Issue86TrackC.write_run_json(scan_b_path, scan(1.8, "revision-b"))
        project = normpath(joinpath(
            PROVENANCE_TRACK_C_ROOT, "../../../..", "julia-env",
        ))
        driver = joinpath(PROVENANCE_TRACK_C_ROOT, "track_c.jl")
        command = `$(Base.julia_cmd()) --project=$project $driver merge-gamma $output_path $scan_a_path $scan_b_path`
        process = run(pipeline(
            ignorestatus(command); stdout = devnull, stderr = devnull,
        ))
        @test success(process)
        if success(process)
            gamma_map = Issue86TrackC.read_run_json(output_path)
            @test gamma_map["source_sha256"] == "source-sha"
            @test gamma_map["scan_code_revisions"] ==
                  ["revision-a", "revision-b"]
            @test gamma_map["provenance_policy"] ==
                  "identical-driver-source-sha256"
        end
    end
end
