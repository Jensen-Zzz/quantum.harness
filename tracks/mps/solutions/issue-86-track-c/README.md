# Challenge #86 — Track C minimum reproduction

This directory is an independent open-boundary implementation of Track C. It
reuses Track B's audited Pauli operators and SOE result type, but it does not
read or modify Track B run data.

The fixed convention is

`H(t) = -Σᵢ<ⱼ |i-j|^(-(1+σ)) ZᵢZⱼ - Γ(t)ΣᵢXᵢ`,

with `J=ℏ=1`, open boundaries, and a linear ramp `Γ: 2Γ_c → 0`. The paper
horizontal axis is `T`; the nearest-neighbour exact formula uses `τ_Q=T/2`.

## One driver

All commands use the repository Julia environment:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl COMMAND ...
```

## Three-day two-node sprint

The sprint uses two independent repository clones. Shard A contains 49 cells
(G1/G2/G3 plus the σ=1.875 size and systematic axis); shard B contains 48
cells (the L=64 σ trend and its controls). The shards never share a `run.json`
or checkpoint directory and are merged by source SHA, Gamma-map hash, cell id,
and parameter hash.

The reviewed initial estimates and immutable A/B critical-scan specifications
are committed under `configs/sprint/`, so both collaborators receive identical
files with a normal Git pull:

```text
configs/sprint/initial-gamma-sprint.json
configs/sprint/critical-A.json
configs/sprint/critical-B.json
```

The initial-estimate JSON contains the exact keys required by the two critical
shards:

```text
1.0:12  1.0:16  1.0:20  1.0:64
1.75:64  1.8:64  1.875:32  1.875:64  1.875:128
1.95:64  2.0:64
```

Values are scan centers only. Track B periodic values may seed the window
entries, but the accepted values always come from the Track C OBC entropy
scans. Each sprint spec registers a conservative `initial_half_width=0.2`
because the OBC finite-size peak can differ materially from the periodic
anchor; the final accepted interval still obeys its stage-specific target.

Run the committed critical shards independently:

```bash
bash tracks/mps/solutions/issue-86-track-c/scnet/submit_critical_shard.sh \
  tracks/mps/solutions/issue-86-track-c/configs/sprint/critical-A.json \
  tracks/mps/results/issue-86-track-c/critical-A
```

The collaborator replaces `critical-A.json`/`critical-A` with
`critical-B.json`/`critical-B` in a separate clone/account. Each node packs
`4 workers × 32 cores`. After both result directories are returned, merge all
eleven completed scan JSON files:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  merge-gamma gamma-map-sprint.json \
  tracks/mps/results/issue-86-track-c/critical-A/*.json \
  tracks/mps/results/issue-86-track-c/critical-B/*.json
```

`merge-gamma` rejects different source revisions, duplicate `(σ,L)` scans,
reversed intervals, and intervals wider than their registered target. Commit
the resulting small Gamma map, then generate the two immutable dynamics
shards from that same file:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  generate-sprint A sprint-A.json gamma-map-sprint.json
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  generate-sprint B sprint-B.json gamma-map-sprint.json
```

Run each shard with the existing packed launcher. Submit `standard/short`,
`large/short`, `standard/long`, and `large/long` sequentially within each
account:

```bash
bash tracks/mps/solutions/issue-86-track-c/scnet/submit.sh \
  sprint-A.json tracks/mps/results/issue-86-track-c/sprint-A standard short
```

After both `run.json` files have been returned:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  merge-campaign sprint-A.json sprint-B.json campaign.json
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  analyze-sprint campaign.json sprint-report theory-curves.json
```

The optional theory file contains exactly two named curves, each with keys
`1.8`, `1.875`, and `1.95`. Without it, the analyzer still produces the
measured μ curve, σ=1.875 collapse, and systematic envelopes, with status
`ready-for-theory` when every numerical gate passes.

Start with the non-negotiable consistency gate:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  g0 tracks/mps/results/issue-86-track-c/g0.json
```

Locate `Γ_c` for every `(σ,L)` before generating dynamics. `INITIAL_GAMMA`
must be read from the paper curve; the driver deliberately does not invent a
default. Production scans (`L≥32`) are guarded against accidental local runs:

```bash
bash tracks/mps/solutions/issue-86-track-c/scnet/submit_critical.sh \
  OUTPUT.json SIGMA L INITIAL_GAMMA 96
```

Generate a stage after the corresponding critical fields have been accepted:

```bash
# A single Gamma_c is enough for G1, G2, or G3.
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  generate G1 RUN.json GAMMA_C

# Multi-size stages accept exact "sigma:L" keys, with sigma-only keys as an
# explicit shared-Gamma fallback:
# {"1.0:32": 1.23, "1.0:128": 1.24, "1.25:32": 1.34, ...}.
```

Supported stages are:

- `G1`, `G1-dt-sentinel`
- `G2`
- `G3`, `G3-sentinels`, `G3-L128`
- `floor-l64`, `floor-collapse`

Challenge-stage generation is gated by completed reproduction data:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  generate-floor-l64 G3_RUN.json FLOOR_L64_RUN.json gamma-map.json

julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  generate-floor-collapse G3_RUN.json FLOOR_L64_RUN.json \
  FLOOR_COLLAPSE_RUN.json gamma-map.json
```

The `cell` command refuses production-sized local work. For an intentional
small smoke, set the explicit escape hatch:

```bash
TRACK_C_ALLOW_LOCAL=1 julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  cell RUN.json CELL_ID OUTPUT_ROOT
```

The normal science route is SCNet. Submit `T≤64` and `T=128` separately:

```bash
bash tracks/mps/solutions/issue-86-track-c/scnet/submit.sh \
  RUN.json OUTPUT_ROOT standard short
bash tracks/mps/solutions/issue-86-track-c/scnet/submit.sh \
  RUN.json OUTPUT_ROOT standard long
```

Use `large` for `χ=128` or `L=128`. A standard allocation packs
`8 workers × 16 cores`; a large allocation packs `4 × 32`. Because the active
SCNet account allows only one node/job at a time, submit the next class after
the previous job settles.

## Artifacts and recovery

`run.json` is the only stage-level source of truth. Every cell writes:

- `trajectory.csv`: time, Γ, full-chain kink density, bulk kink density,
  averaged `C(r)`, norm, maximum bond;
- `manifest.json`: parameter hash, revision, resources, final observables and
  numerical-gate values;
- `checkpoint.bin`: atomic state checkpoint at approximately 5% intervals.

Completed cells are idempotent. Failed cells remain selectable by `pending`;
re-submitting a class skips completed cells and resumes valid checkpoints.
Updates to `run.json` are protected by a file lock for packed workers.

Analyze and render after a stage completes:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  analyze RUN.json REPORT_DIRECTORY
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/plot_curve.jl \
  REPORT_DIRECTORY/curve.csv REPORT_DIRECTORY/curve.png
```

The registered G2/G3 fit window is always points 2–5. The analysis never
changes the window after seeing results. G3 sentinel deviations above 2%
identify only the affected `T` values for `χ=128` reruns. A converged G3
failure routes to `G3-L128`; another failure pauses challenge production.

Compare either sentinel stage to its baseline:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  compare-sentinels BASELINE_RUN.json SENTINEL_RUN.json sentinel-summary.json
```

For a failed G3 sentinel check, generate a run containing only affected
`χ=128` points:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  g3-upgrades G3_RUN.json G3_SENTINELS_RUN.json G3_UPGRADES_RUN.json
```

Create the single converged G3 curve consumed downstream. Omit the final
argument when sentinels pass:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  converge-g3 G3_RUN.json G3_SENTINELS_RUN.json G3_CONVERGED_RUN.json \
  G3_UPGRADES_RUN.json
```

Affected `T` values are overlaid from the `χ=128` run; all other points retain
the baseline result. A failed `Δt=0.025` sentinel blocks promotion instead of
being disguised as a bond-dimension problem; each `χ=128` replacement must
also return within 2% of the baseline before the merged curve is accepted.
Challenge-stage generation refuses an unconverged G3 run.

If the sentinel-screened L=64 curve is numerically converged but misses the
relaxed literature gate, run `G3-L128`. A passing fallback is promoted while
retaining the L=64 points needed by the three-size collapse:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  promote-g3-l128 SCREENED_G3_RUN.json G3_L128_RUN.json \
  G3_CONVERGED_RUN.json
```

After the `L=32,128` floor run completes, combine it with the already computed
`L=64` curves:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/track_c.jl \
  collapse G3_RUN.json FLOOR_L64_RUN.json FLOOR_COLLAPSE_RUN.json collapse.json
```

This command enforces `|μ_collapse-μ_L64|≤0.05` for σ=1, 1.25, and 1.5. It
does not generate or accept points in the disputed `7/4<σ<2` window.

## Tests

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86-track-c/test/runtests.jl
```

The first full run compiles MPSKit's dense-MPO and TDVP paths and can take
several minutes. It covers OBC/SOE/dense consistency, ED splitting, parity and
norm preservation, atomic recovery, a real small-chain TDVP trajectory, all
registered fit gates, and finite-time collapse.
