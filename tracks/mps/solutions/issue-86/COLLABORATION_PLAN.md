# Issue #86 Track B reproduction and collaborator handoff

This document is the operational plan for reproducing Track B of
[QuantumBFS/quantum.harness#86](https://github.com/QuantumBFS/quantum.harness/issues/86).
Track C / issue #100 is out of scope.

## Repository state

- Repository: <https://github.com/Jensen-Zzz/quantum.harness>
- Branch: `codex/issue-86-reproduction`
- Solver and scheduler commit used to generate the Stage 2 run specification:
  `16b62684c898525ff5b736a93e9f653cb6470548`
- Stage 1 data publication commit: `486fafb84437708d755fc773579553e35b408b1f`
- Main implementation:
  `tracks/mps/solutions/issue-86/`
- Published Stage 1 data:
  `tracks/mps/results/issue-86-stage1/`
- Published Stage 1 formal audit:
  `tracks/mps/results/issue-86-formal-stage1/`
- Ready-to-run Stage 2 specification:
  `tracks/mps/results/issue-86-stage2-first-pass/run_spec.json`

The Stage 2 specification names the solver commit rather than the later
data-only commit. Checking out the branch head is safe: no solver code changed
between those commits.

## Scientific target

The Hamiltonian convention is

\[
H=-\sum_{i<j}J_L(|i-j|)\sigma_i^z\sigma_j^z
  -\Gamma\sum_i\sigma_i^x,
\]

\[
J_L(r)=L^{-(1+\sigma)}
\left[
\zeta(1+\sigma,r/L)+\zeta(1+\sigma,1-r/L)
\right].
\]

This is the bare-\(J=1\), periodic-image Hurwitz-zeta convention used by the
challenge and Shiratani--Todo. The primary observables and references are:

| target | reference |
|---|---:|
| nearest-neighbour \(\Gamma_c\) | \(1\) |
| nearest-neighbour dynamic exponent \(z\) | \(1\) |
| \(\Gamma_c(\sigma=7/4)\) | \(1.5609(3)\) |
| \(\Gamma_c(\sigma=2)\) | \(1.4208(2)\) |

Critical points are obtained from crossings of

\[
R_L=\xi_L/L,\qquad
\xi_L/L=\frac{1}{2\pi}
\sqrt{S(0)/S(2\pi/L)-1}.
\]

The numerical method is finite-chain DMRG with MPSKit. A sparse exact
diagonalization implementation is the independent oracle for \(L\le16\).
Long-range interactions use a sum-of-exponentials MPO with explicit pole
residual auditing.

## Completed work

Stage 1 contains 75 successful cells. The raw CSV has 75 matching rows and all
manifests record successful completion.

| audit | result | gate | status |
|---|---:|---:|---|
| NN crossing, \(L=32/64,\chi=64\) | \(0.99933978\) | \(|\Gamma_c-1|<0.005\) | pass |
| NN gap exponent, \(\chi=128\), all sizes | \(z=1.00051779\) | \(0.95<z<1.05\) | pass |
| NN gap exponent, remove \(L=16\) | \(z=1.00030733\) | \(0.95<z<1.05\) | pass |
| \(\sigma=1.75\), \(L=32/64,\chi=64,P=16\) | \(1.56407317\) | within 1% | pass, 0.203% high |
| \(\sigma=2\), \(L=32/64,\chi=64,P=16\) | \(1.42368967\) | within 1% | pass, 0.203% high |

All 75 convergence residuals are below \(10^{-8}\). All ten excited-state
normalized variances are below \(10^{-10}\). Some finite-\(\chi\) ground-state
normalized variances, particularly \(L=64,\chi=64\), remain above
\(10^{-10}\); they are retained as diagnostics and must not be hidden by a
fit.

Stage 1 therefore establishes the preliminary 1% reproduction but is not yet
the formal reproduction. The formal audit identifies three missing evidence
classes for each long-range anchor:

1. four baseline size-pair crossings;
2. a \(\chi=128\), \(L=32/64\) crossing;
3. a \(P=12\), \(L=32/64\) crossing for the MPO truncation audit.

The NN \(L=16\) energy gate passes, but its correlation ratio differs from ED
by about \(1.3\times10^{-5}\). Stage 2 includes a tighter
`tolerance=1e-11, maxiter=80` diagnostic. This diagnostic is not used to move
the crossing; it checks whether the previous early DMRG stop caused the
correlator error.

## Stage 2 first pass

Use `configs/stage2-first-pass.toml` and the committed
`issue-86-stage2-first-pass/run_spec.json`. It contains 67 cells and reuses
the completed Stage 1 cells instead of recomputing them.

| component | cells | resource class |
|---|---:|---|
| missing \(P=16,\chi=64,L=8,24,48\) baseline sizes | 30 | A |
| \(P=12,\chi=64,L=32,64\) MPO audit | 20 | A |
| first adaptive \(P=16,\chi=64,L=32,64\) midpoints | 4 | A |
| tight NN--ED diagnostic | 1 | A |
| \(P=16,\chi=128,L=32,64\) MPS audit | 12 | B |
| total | 67 | 54 A + 13 B |

Do not launch `stage2-baseline.toml` and `stage2-systematics.toml` in addition
to this first pass; that would duplicate cells already covered by Stage 1 and
the incremental specification.

## Environment and validation

Clone and validate before submitting expensive work:

```bash
git clone --branch codex/issue-86-reproduction \
  https://github.com/Jensen-Zzz/quantum.harness.git
cd quantum.harness

make skills
make install julia
make install mpskit

PATH="$HOME/.juliaup/bin:$PATH" \
  julia --project=julia-env \
  tracks/mps/solutions/issue-86/test/runtests.jl

pytest -q scripts/tests/test_issue86_slurm.py
```

The currently validated suite contains 136 Julia tests and 8 Slurm launcher
tests. Do not proceed to the full scan if the Hamiltonian, ED--MPO, pole
convergence, or DMRG--ED tests fail.

## Running on a collaborator cluster

The calculation is CPU-only. Each DMRG point is single-node; parallelism is
across independent parameter points. GPU nodes and cross-node tensor
parallelism are not needed.

`run_full.sbatch` automatically scales the worker count to the allocated
CPUs:

| allocation | class A layout | class B layout |
|---:|---:|---:|
| 128 CPUs | 16 workers x 8 CPUs | 8 workers x 16 CPUs |
| 64 CPUs | 8 workers x 8 CPUs | 4 workers x 16 CPUs |

Suggested allocations:

| job | cells | preferred request | conservative time limit |
|---|---:|---:|---:|
| Stage 2 A | 54 | 128 CPUs, 240--480 GB | 2 hours |
| Stage 2 B | 13 | 128 CPUs, 240--480 GB | 4 hours |

On the SCNet AMD EPYC 7742 node, the expected compute times are approximately
25--50 minutes for A and 30--90 minutes for B at 128 CPUs. At 64 CPUs,
allow approximately 40--80 minutes for A and 45--120 minutes for B.

The script contains SCNet defaults, but command-line `sbatch` options override
them. On a different Slurm cluster, replace `<partition>` and choose memory
consistent with local policy:

```bash
RUN_SPEC=tracks/mps/results/issue-86-stage2-first-pass/run_spec.json
OUT_DIR=tracks/mps/results/issue-86-stage2-first-pass

sbatch \
  --partition=<partition> --nodes=1 --cpus-per-task=128 \
  --mem=480G --time=02:00:00 --job-name=qh86-s2-A \
  --output="$OUT_DIR/slurm-%x-%j.out" \
  --export=ALL,HARNESS_RUN_SPEC="$RUN_SPEC",HARNESS_COMMAND=stage2-first-pass:A \
  tracks/mps/solutions/issue-86/run_full.sbatch

sbatch \
  --partition=<partition> --nodes=1 --cpus-per-task=128 \
  --mem=480G --time=04:00:00 --job-name=qh86-s2-B \
  --output="$OUT_DIR/slurm-%x-%j.out" \
  --export=ALL,HARNESS_RUN_SPEC="$RUN_SPEC",HARNESS_COMMAND=stage2-first-pass:B \
  tracks/mps/solutions/issue-86/run_full.sbatch
```

The A and B jobs may run concurrently. Do not submit two copies of the same
resource class into the same output directory: the current resumability
contract skips completed cells but does not provide a distributed claim lock
between two simultaneously starting jobs.

Every cell writes an atomic
`cells/<cell-id>/manifest.json`. A node failure or wall-time limit therefore
requires only rerunning the same command; successful cells will be skipped.
Do not save MPS wavefunctions unless a failed point specifically requires
state-level diagnosis.

## Collection and formal analysis

After A and B finish, confirm that all 67 Stage 2 manifests are successful,
then combine Stage 1 and Stage 2:

```bash
julia --project=julia-env \
  tracks/mps/solutions/issue-86/analyze_formal.jl \
  tracks/mps/results/issue-86-formal \
  tracks/mps/results/issue-86-stage1 \
  tracks/mps/results/issue-86-stage2-first-pass
```

Inspect:

- `formal_summary.json`;
- `crossings.csv` and `crossings.json`;
- the two finite-size extrapolation plots;
- `adaptive_run_spec.json`;
- variance, residual, \(\chi\)-drift, and pole-drift diagnostics.

If `adaptive_run_spec.json` contains cells, run it with the same packed
launcher and repeat the formal analysis. Continue midpoint refinement until
the crossing bracket is at most `0.001` and the crossing changes by less than
`1e-4`. Preserve every failed or superseded manifest.

## Formal acceptance and escalation

A long-range anchor is formally reproduced only if all of the following hold:

1. four baseline pairs \((8,16),(16,32),(24,48),(32,64)\) are available;
2. the crossing bracket is at most `0.001` and adaptive movement is below
   \(10^{-4}\);
3. increasing \(\chi\) and pole count reduces the observed drift;
4. the conservative error budget includes interpolation, finite-size fit
   envelope, maximum \(\chi\) drift, and maximum pole drift;
5. the final interval overlaps \(1.5609(3)\) or \(1.4208(2)\);
6. production residuals are below \(10^{-8}\), and any normalized variance
   above \(10^{-10}\) is resolved or explicitly retained as a failed gate.

Use the fit

\[
\Gamma_\times(L,2L)=\Gamma_c+aL^{-\omega}
\]

with both the full size set and a fit omitting the smallest size. Do not use a
fit to conceal a failed pole, variance, or ED gate.

Only if the first pass still does not cover the literature interval, run
`stage2-contingency.toml`: \(L=128,\chi=128,P=16\), three crossing-neighbourhood
points per sigma. This is resource class C and may add roughly 12--24
single-node hours on an EPYC 7742 node. Use `stage2-chi256.toml` only if the
\(\chi=128\) audit remains visibly unconverged; it is a last resort and can
add two to four node-days.

## Result handback

Return the complete `tracks/mps/results/issue-86-stage2-first-pass/`
directory, including cell manifests and Slurm logs. The expected data volume
is small because no wavefunctions are stored. Before pushing:

1. verify the number of successful manifests and inspect failed cells;
2. run the formal analyzer;
3. scan logs for credentials or site-private tokens;
4. commit the Stage 2 directory with `git add -f` because `tracks/**/results/`
   is ignored by default;
5. publish on a separate collaborator branch or pull request to avoid
   overwriting another active result upload.

The final article package should include raw CSV/JSON, all manifests,
crossing plots, NN gap scaling, finite-size extrapolations, the
\(\chi/P\)-drift table, the resource-consumption table, and an explicit label
of either “formal reproduction” or “pipeline validation / finite-size
preliminary result.”
