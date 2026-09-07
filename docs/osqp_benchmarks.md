# OSQP benchmark classes: ElastiQP vs PIQP / ProxQP (2026-09-07)

Runs: `_dense_eps1e-6` (before the Eq QP fix) and `_dense_eps1e-6_eqfix`
(after), same pack, same protocol, 341 s and 346 s wall.

Motivation: avoid tuning to Maros-Meszaros alone. The OSQP paper's
benchmark suite (osqp_benchmarks, cloned at the repo root) generates seven
random problem classes -- Random QP, Eq QP, Portfolio, Lasso, SVM, Huber,
Control -- as sparse OSQP-form QPs, l <= Ax <= u. This note runs the three
ElastiQP backends, PIQP and ProxQP on dense conversions of those exact
generators (same code, same seeds). The suite's own Maros-Meszaros copy is
not used (covered by docs/maros_full_set.md).

## Setup

- `elastiqp_benchmarks/tools/convert_osqp_benchmarks.py ../osqp_benchmarks
  <pack>`: instantiates the osqp_benchmarks problem classes (cvxpy stubbed),
  packs them densely into the MMQP format `bench_maros_meszaros` reads, and
  writes `<pack>.index.csv` (class, dim, seed, n_vars, rows, eq/ineq rows,
  nnz). `--preset osqp` reproduces the paper's 20-dimension x 10-seed
  ladders; the default `dense` preset is what ran here.
- `elastiqp_benchmarks/python/run_osqp_benchmarks.py --pack <pack>
  --work-dir <dir> --cores 0,1,2,3`: one subprocess per (problem, route)
  under a 300 s limit, four worker threads each pinned to a core pulling
  whole problems from a queue (all routes of a problem time on the same
  core), PIQP 1e-9 reference cached per problem (`--ref-cache`), resumable.
  Writes `results/osqp_benchmarks_{results,summary,drops,log}<sfx>.*`;
  `plot_results.py` adds `osqp_benchmarks_{profile,scaling}<sfx>.{png,svg}`.
- Same protocol as the Maros-Meszaros tables: eps_abs 1e-6, eps_rel 0,
  penalty 10x the largest reference dual, cold starts, Ruiz on for the
  ElastiQP routes; `ok` = reported success AND hard violation <= 1e-4 AND
  relative objective error <= 1e-4.

Dense preset: 6 log-spaced dimensions x 3 seeds per class (Huber 5 x 3),
123 problems, 200-2100 variables / up to 3200 rows at the top of each
ladder (the paper goes to n = 2000 for Random QP = 20000 rows, Huber
n = 200 = 60000 variables; those are out of reach for dense solvers and
not the point here). Wall time for the whole run: 341 s on 4 P-cores
(Core Ultra 7 258V). Because four solves run concurrently, absolute times
of the largest instances are inflated by memory contention -- the DAS
SVM n=1616 solve took 52 s alone vs 72-104 s in the run, PIQP 5.7 vs
7.3 s -- so the largest-size ratios below overstate DAS by up to ~1.5x.
Within a problem every route ran on the same core.

## Headline (after the Eq QP fix below; `_eqfix` files)

123 problems, all five routes solve 123/123. Shifted geometric mean
(shift 1 ms):

| group (problems) | PIQP | ProxQP | ElastiQP-DAS | ElastiQP-PDAL | ElastiQP-IPM |
|---|---|---|---|---|---|
| all (123) | 61 ms (1.33x) | 82 ms (1.79x) | 80 ms (1.74x) | **46 ms (1.00x)** | 83 ms (1.81x) |
| Random QP (18) | 4.5 (1.04x) | 9.5 (2.21x) | **4.3 (1.00x)** | 4.7 (1.10x) | 5.2 (1.21x) |
| Eq QP (18) | **3.2 (1.00x)** | 3.6 (1.14x) | 3.4 (1.05x) | 3.7 (1.14x) | 3.6 (1.14x) |
| Portfolio (18) | 172 (1.18x) | 228 (1.58x) | 319 (2.21x) | **145 (1.00x)** | 369 (2.55x) |
| Lasso (18) | **61 (1.00x)** | 169 (2.77x) | 77 (1.27x) | 65 (1.06x) | 84 (1.38x) |
| SVM (18) | 299 (2.08x) | 414 (2.89x) | 762 (5.31x) | **143 (1.00x)** | 315 (2.20x) |
| Huber (15) | 1165 (1.53x) | 1253 (1.64x) | 2432 (3.19x) | **762 (1.00x)** | 1287 (1.69x) |
| Control (18) | 67 (2.43x) | 43 (1.59x) | 36 (1.33x) | **27 (1.00x)** | 129 (4.71x) |

Before the fix (the un-suffixed `_dense_eps1e-6` files) the only rows that
differed were Eq QP -- PDAL 6.4 ms (1.98x) and IPM 6.5 ms (1.99x) -- and
the all-problems sgm (PDAL 49 ms, IPM 88 ms). Every other cell moved by
less than run-to-run noise.

Accuracy: max hard violation 9.9e-7 (PDAL), 9.4e-8 (DAS), 2e-12 (IPM);
max relative objective error 3.5e-7 (PDAL). Nothing near the 1e-4 gates.

Per-(class, dimension) means, iteration counts and the piqp ratios are in
`results/osqp_benchmarks_results_dense_eps1e-6_eqfix.csv`; the scaling grid
`results/osqp_benchmarks_scaling_dense_eps1e-6_eqfix.png` shows the same.

## Drops worth knowing about

1. **Eq QP: PDAL and IPM 3.6x (n=356) to 7.3x (n=1509) slower than
   PIQP/ProxQP/DAS, with 0 iterations -- FIXED 2026-09-07.** Both backends
   take `solve_no_inequalities()` when p = 0, which solved the (n+m) KKT
   system with `Eigen::ColPivHouseholderQr`, an unblocked O((n+m)^3)
   pivoted QR: 2.45 s at n+m = 2263. Ruiz and the equality consistency
   check were not the cause. Replaced by `SolveEqualityQP` in
   `include/elastiqp/common.hpp`, shared by both backends: the range-space
   solve the DAS working set already does implicitly -- LLT of Q + rho I,
   S = A K^-1 A' (+ delta I only when the equality certificate says A has
   dependent rows, or an LLT of S fails), LLT of S, then iterative
   refinement against the unregularized KKT residual so the proximal rho
   leaves no bias (one step for a PD Q; a few for a singular Q with a PD
   reduced Hessian). The pivoted QR stays as the fallback when Q is
   indefinite or S cannot be regularized. Mean solve time, ms:

   | n | PIQP | DAS | PDAL before | PDAL after | IPM before | IPM after |
   |---|---|---|---|---|---|---|
   | 89 | 0.25 | 0.21 | 0.45 | 0.21 | 0.52 | 0.21 |
   | 356 | 8.0 | 9.0 | 28.7 | 10.7 | 28.2 | 10.7 |
   | 1509 | 471 | 544 | 3841 | 689 | 3831 | 684 |

   The remaining 1.45x at n=1509 is setup (Ruiz + the certificate, ~180 ms)
   plus forming Y = K^-1 A' and S. Tests: `test_solvers.cc` "equality-only
   range-space solve" (dependent consistent rows = full-rank answer,
   singular Q pinned by equalities, n=1500/m=750 vs a blocked LU of the
   KKT matrix), all three backends. Robot control benchmark
   (`bench_robot_control`, same core, before vs after binaries): identical
   iteration and failure counts on all 32 scenario x mode rows, mean time
   ratio geomean 0.99 (range 0.94-1.03, noise). Robot problems always have
   p > 0, so this path is never reached there.

2. **DAS on problems with a large optimal active set**: SVM (2000 active
   of 3200 rows at n=1616), Huber (1437 active), Portfolio (1669 active =
   the whole long-only box). Iterations track the active-set size
   (one add per iteration from cold), and each add/remove is O(k^2)-O(k^3)
   in the current active count, so DAS goes from parity at 200 variables
   to 11.6x PIQP on SVM n=1616 (9.2x measured alone), 3.6x on Huber
   n=2107, 2.9x on Portfolio n=1616. These are the three `drops` rows
   (>= 10x). Same mechanism as the Maros-Meszaros large-n note
   (`remove_row` O(k^3) vs DAQP's rank-one); expected for a dual
   active-set method, and irrelevant at robot scale, but it is the one
   class of problem where DAS is not competitive cold. Random QP (rows =
   10n, ~4 active per variable at optimum) shows the mild version: 0.4-0.5x
   PIQP at n <= 40, 2x at n=309.

3. **IPM 2-2.5x PIQP on Control and Portfolio (4.7x on the Control sgm),
   ~1.4x on Random QP**, at equal iteration counts (8-13). Control rows are
   almost all two-sided boxes, which the elastic split doubles (p = 3158
   elastic rows vs PIQP's 1579 two-sided rows), so every IPM factorization
   carries twice the constraint block. Structural cost of the elastic
   formulation, not a convergence problem.

4. **PDAL cold on small Random QP**: 1.5-2.1x PIQP at n <= 40 (20-43 outer
   iterations against PIQP's 11-13). Sub-millisecond either way; at
   n >= 105 PDAL is at or below PIQP.

Where ElastiQP is ahead: PDAL is the fastest route on Portfolio, Lasso
(tied with PIQP), SVM (0.35-0.7x PIQP), Huber (0.5-0.8x) and Control
(0.3-0.55x); DAS is fastest on small Random QP, small Eq QP and Control
up to n=1579 (0.46x PIQP, faster than PDAL below 384 variables).

## Reproduce

    cd elastiqp_benchmarks
    ../.venv/bin/python tools/convert_osqp_benchmarks.py ../osqp_benchmarks \
        ../untracked/osqp_bench/osqp_dense.bin           # 123 problems, 1.5 GB
    ../.venv/bin/python python/run_osqp_benchmarks.py \
        --pack ../untracked/osqp_bench/osqp_dense.bin \
        --work-dir ../untracked/osqp_bench/work_dense --cores 0,1,2,3
    (cd python && ../../.venv/bin/python -c "import plot_results as p; p.plot_osqp_benchmarks()")

`--preset osqp --max-vars 3000` on the converter gives the paper's ladders
truncated to dense-tractable sizes (10 seeds); `--summary-only` on the
runner rebuilds the tables from a finished work dir.
