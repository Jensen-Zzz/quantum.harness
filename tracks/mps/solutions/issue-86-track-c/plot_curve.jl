#!/usr/bin/env julia

using DelimitedFiles
using Plots

length(ARGS) == 2 || error("usage: plot_curve.jl CURVE.csv OUTPUT.png")
curve_path, output_path = ARGS
rows = readdlm(curve_path, ',', Any, '\n'; header = true)[1]

plot(
    xlabel = "T",
    ylabel = "final kink density",
    xscale = :log10,
    yscale = :log10,
    legend = :bottomleft,
    framestyle = :box,
)

groups = Dict{Tuple{String, Int}, Vector{Tuple{Float64, Float64}}}()
for row in eachrow(rows)
    isempty(string(row[6])) && continue
    key = (string(row[2]), Int(row[3]))
    push!(
        get!(groups, key, Tuple{Float64, Float64}[]),
        (Float64(row[4]), Float64(row[6])),
    )
end

for ((sigma, L), points) in sort!(collect(groups); by = first)
    sort!(points; by = first)
    plot!(
        first.(points), last.(points);
        marker = :circle, label = "σ=$sigma, L=$L",
    )
end

if any(first(key) == "NN" for key in keys(groups))
    times = 10.0 .^ range(log10(4), log10(128); length = 200)
    plot!(
        times, inv(2pi) ./ sqrt.(times);
        linestyle = :dash, color = :black, label = "NN exact",
    )
end

mkpath(dirname(output_path))
savefig(output_path)
println(output_path)
