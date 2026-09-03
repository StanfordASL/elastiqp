# ElastiQP benchmarks

Benchmarks, cross-solver comparisons, and the tests that compare
[ElastiQP](https://github.com/StanfordASL/elastiqp) against external QP
solvers. This is a standalone CMake project so that the core `elastiqp`
repo stays dependency-free (Eigen only): everything that needs PIQP,
ProxQP, Pinocchio, qpax, or scipy lives here.

## Build

```bash
cmake -B build            # needs Eigen3; fetches PIQP + ProxSuite tarballs
cmake --build build -j
ctest --test-dir build    # cross-solver tests
```

How dependencies are resolved (each in exactly one place):

| dependency | where | how |
|---|---|---|
| Eigen3 | `CMakeLists.txt` | `find_package` |
| elastiqp | `cmake/elastiqp.cmake` | source tree: `-DELASTIQP_SOURCE_DIR=<path>`, else auto-detected at `../` (this repo nested in the elastiqp tree) or `../elastiqp` (sibling checkout), else a git fetch pinned to `ELASTIQP_GIT_TAG`. Header-only, consumed via `add_subdirectory`, which also exposes elastiqp's dev-only `elastiqp::testing` headers (problem generators, IPM reference). |
| PIQP v0.6.3, ProxSuite v0.7.3 | `cmake/external_solvers.cmake` | pinned, hash-verified release tarballs, header-only; nothing vendored, nothing built. `-DELASTIQP_BENCH_EXTERNAL_SOLVERS=OFF` for an Eigen-only build of the core benchmarks. |
| Pinocchio | `CMakeLists.txt` | `find_package`, optional; only to regenerate `data/robot_sequences.bin` and run the robot-builder tests. |
| Python (+scipy) | `CMakeLists.txt` | optional; packs the Maros-Meszaros set at build time. |

Python side: `pip install -r requirements.txt` (elastiqp itself from PyPI or
`pip install -e "../elastiqp[jax]"`).

## Layout

```
core/         Eigen-only benchmarks (elastiqp alone)
external/     cross-solver benchmarks (PIQP, ProxQP)
experiments/  paper experiments (robot_solver_comparison)
tests/        elastiqp vs external solvers; IPM-reference vs PIQP; robot builders
robotics/     Pinocchio problem builders + the robot-sequence generator
common/       qp_io.hpp (EQPS robot-sequence format)
data/         committed robot_sequences.bin (replayed by every robot benchmark)
python/       JAX-side experiments (qpax), plotting, Maros-Meszaros in Python
tools/        Maros-Meszaros .mat -> packed binary converter
results/      CSV/figure output (gitignored)
cmake/        dependency resolution (see table above)
```

## Tests (`tests/`)

| ctest name | checks |
|---|---|
| `elastiqp_bench.piqp_cross_validation` | `elastiqp::Solver` vs vanilla PIQP: strict feasible problems (t exactly 0), the expanded `(n+p)` elastic formulation on infeasible ones, per-row penalties, PSD Q, hard equalities, exact-penalty threshold |
| `elastiqp_bench.ipm_reference_validation` | elastiqp's test-only IPM reference (the oracle its own test suite uses) vs vanilla PIQP, plus its warm-start / Ruiz / relaxation behaviour |
| `elastiqp_bench.robot_control` | (Pinocchio) closed-loop diff-ik / OSC / WBC builders converge, equalities hold under conflict, and match PIQP on the expanded formulation |

## Core benchmarks (`core/`, Eigen only)

The robot benchmarks replay the committed `data/robot_sequences.bin`, so
Pinocchio is NOT required to run them:

| binary | measures |
|---|---|
| `bench_random_qp` | cold solve / `relax(kappa)` / KKT vjp timings on random dense elastic QP families |
| `bench_collision_2d` | differentiable collision distance (qpax closest-point QP), cold-vs-warm solver over a 2D sweep; deliberate small-scale contrast to `bench_diff_robot` |
| `bench_robot_control` | per-tick solve-time distribution, cold vs warm, on realistic robot control loops (diff-ik / arm-osc / hum-wbc / biman-ik) |
| `bench_diff_robot` | cost of differentiability at robot scale: relax(kappa), KKT vjp, amortization vs the forward solve |
| `bench_relax_warm` | relax() warm chain vs cold retraction start across structure / penalty / drift composition on drifting random QPs (regime map for `relax(warm=true)` and the predicted-flip gate) |
| `bench_fwd_warm` | forward solve() warm vs cold start on the same drifting-QP grid; tracks activity changes, factorization reuse, and BCL cold-reset firings (warm-start pathology watch) |
| `bench_bcl_strategies` | the saturation-creep failure regime (isolated from `bench_relax_warm` @ b995082) replayed under each BCL strategy generation, proxqp parity through the shipped elastic BCL, plus mixed per-row penalty cells (alt/spike/dip w = {10, 1e4}); validates the `bcl_split` / `bcl_mu_jump` / `bcl_warm_eta` / `cold_reset_limit` defaults and the shallowest-first jump target |
| `bench_eq_elastic` | hard equalities vs the folded `[A; -A]` elastic pair |

## Cross-solver benchmarks (`external/`, `experiments/`)

| binary | measures |
|---|---|
| `bench_robot_multisolver` | every solver route (elastic, hard, l1-slack, l2 closest-feasible) on the robot sequences, feasible AND conflict variants |
| `robot_solver_comparison` | paper table: every route on the robot sequences, feasible + conflict variant (one row tightened past what the remaining constraints admit, via an auxiliary LP); `--csv results` |
| `bench_condensed_vs_expanded` | ElastiQP's condensed `O(n^3 + p n^2)` formulation vs vanilla PIQP on the expanded `(n+p)` elastic problem |
| `bench_proxqp_closest` | elastic (l1) resolution vs ProxQP's `primal_infeasibility_solving` (l2 closest-feasible): shift structure and cost |
| `bench_maros_meszaros` | Maros-Meszaros small dense subset (n <= 200): elastiqp / piqp-hard / piqp-expanded / proxqp-hard (needs python3 + scipy at build time to pack the data) |

Fairness rule: every comparison table is produced entirely within one
harness — the C++ binaries time C++ APIs on identical in-memory data. The
only Python-side comparison is `python/run_diff_experiment.py`, where the
counterpart (qpax) is JAX-only, so both sides go through JAX.

Timing benchmarks print human-readable tables by default and write CSVs
with `--csv <dir>` (conventionally `results/`, which is gitignored).
`python/plot_results.py` renders the figures from those CSVs.

## Codegen (-march=native) comparisons

The build does not duplicate targets per arch. For a codegen comparison,
configure a second build tree with `-DCMAKE_CXX_FLAGS=-march=native`, run
the same benchmarks with `--csv`, and diff on the `arch` column baked into
every CSV (`generic` vs `native`, `+simde` appended when ProxQP was built
with vectorization).

## Robot sequence data

`data/robot_sequences.bin` (EQPS v2, float32, ~10 MB) is the canonical
committed artifact: four control-loop QP sequences generated by
`robotics/gen_robot_sequences.cc` from Pinocchio sample models, with the
closed loops driven by a warm-started `elastiqp::Solver`. The benchmarks
replay this file; Pinocchio is only needed to REgenerate it:

```bash
# any Pinocchio works: pip install pin / conda / ROS; then
cmake -B build -DCMAKE_PREFIX_PATH=<pinocchio prefix>
cmake --build build -j --target regen_robot_sequences   # writes into data/
ctest --test-dir build -R robot_control                  # builder correctness
./build/bench_robot_control                              # replay sanity
```

Regeneration is not bit-reproducible across machines (FMA availability and
Eigen version perturb the closed loop). The generator prints per-sequence
fingerprints; "verified" regeneration means the trajectory statistics agree
to a few significant digits and `bench_robot_control` reports 0 fails —
matching hashes mean byte-identical data. Current committed file:

```
diff-ik   500 ticks  n=  6 m=  0 p= 24  hash=06e1349ed626e304  sum|q|=2.111816e+02  sum|h|=5.749404e+04
arm-osc   500 ticks  n=  6 m=  0 p= 24  hash=50cad1dc3c672ba1  sum|q|=1.187094e+03  sum|h|=1.741511e+05
hum-wbc   250 ticks  n= 46 m= 18 p=132  hash=4761dc012d8a9bde  sum|q|=4.127335e+02  sum|h|=2.095248e+05
biman-ik  500 ticks  n= 12 m=  6 p= 48  hash=41e79c0d4a2283a9  sum|q|=3.737436e+02  sum|h|=8.131677e+04
```

(biman-ik was appended 2026-08-22 by merging a fresh generation into the
committed file, leaving the three original sequences byte-identical; a full
regeneration reproduces their trajectory statistics but not their bytes,
because solver-default changes since the original run perturb the closed
loops at rounding level.)

## Python layer (`python/`)

- `run_diff_experiment.py` — elastiqp.jax vs qpax gradient accuracy,
  timing across robot scales, and kappa-smoothing bias. qpax's only home
  in the suite (both sides through JAX = fair).
- `run_maros_meszaros.py` — Maros-Meszaros through the Python APIs
  (elastiqp / piqp / proxqp / qpax); reads the .mat files from the
  proxsuite tarball fetched into `build*/_deps`, or `--data-dir`.
- `run_robot_qpax.py` — qpax on the robot sequences (JAX side of the paper
  table).
- `plot_results.py` — figures from the CSVs in `results/`.
- `bench_common.py` — EQPS reader, timing/residual/CSV helpers.
