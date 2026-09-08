# OSQP benchmark classes: ElastiQP vs PIQP / ProxQP (2026-09-07)

Runs: `_dense_eps1e-6` (before the Eq QP fix) and `_dense_eps1e-6_eqfix`
(after), same 123-problem pack, 341 s and 346 s wall; `_osqp_eps1e-6` is
the paper's ladders (860 problems, 6198 s), see the end. DAQP was added to
both `_eqfix` and `_osqp` afterwards, and `_rank1` re-runs DAS with the
rank-one removal (sections at the end).

Motivation: avoid tuning to Maros-Meszaros alone. The OSQP paper's
benchmark suite (the `third_party/osqp_benchmarks` submodule of
elastiqp_benchmarks) generates seven
random problem classes -- Random QP, Eq QP, Portfolio, Lasso, SVM, Huber,
Control -- as sparse OSQP-form QPs, l <= Ax <= u. This note runs the three
ElastiQP backends, PIQP and ProxQP on dense conversions of those exact
generators (same code, same seeds). The suite's own Maros-Meszaros copy is
not used (covered by docs/maros_full_set.md).

## Setup

- `elastiqp_benchmarks/tools/convert_osqp_benchmarks.py
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
  `perf_profile.py` renders `osqp_benchmarks_profile<sfx>.{png,svg}`.
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

## Paper ladders: `--preset osqp` (`_osqp_eps1e-6` files)

The OSQP paper's own dimension ladders (20 log-spaced dimensions x 10
seeds per class), truncated to what dense solvers can hold (`--max-vars
3100 --max-rows 4000`): 860 problems, 10 to 3010 variables, 6198 s wall on
4 P-cores with a 600 s per-solve limit. Huber keeps only its smallest
dimension (n=10 is already 3010 variables); Random QP stops at n=304
(3040 rows); everything else keeps 8 to 20 dimensions.

All five routes solve 860/860; max hard violation 9.8e-7 (ProxQP), max
relative objective error 8.6e-7. Shifted geometric mean (shift 1 ms):

| group (problems) | PIQP | ProxQP | ElastiQP-DAS | ElastiQP-PDAL | ElastiQP-IPM |
|---|---|---|---|---|---|
| all (860) | 84 ms (1.29x) | 112 ms (1.71x) | 112 ms (1.70x) | **65 ms (1.00x)** | 121 ms (1.85x) |
| Random QP (150) | 3.7 (1.13x) | 8.2 (2.46x) | **3.3 (1.00x)** | 4.1 (1.23x) | 4.5 (1.34x) |
| Eq QP (200) | **3.1 (1.00x)** | 3.6 (1.16x) | 3.2 (1.04x) | 3.5 (1.14x) | 3.5 (1.13x) |
| Portfolio (120) | 1716 (1.25x) | 2185 (1.60x) | 4486 (3.28x) | **1369 (1.00x)** | 3688 (2.69x) |
| Lasso (100) | **1292 (1.00x)** | 6196 (4.80x) | 2482 (1.92x) | 1472 (1.14x) | 1932 (1.50x) |
| SVM (80) | 4990 (1.72x) | 6605 (2.27x) | 44811 (15.4x) | **2905 (1.00x)** | 4988 (1.72x) |
| Huber (10) | 21804 (1.28x) | 21691 (1.27x) | 130885 (7.68x) | **17036 (1.00x)** | 21508 (1.26x) |
| Control (200) | 96 (2.55x) | 62 (1.65x) | 50 (1.34x) | **38 (1.00x)** | 192 (5.11x) |

Same picture as the 3-seed dense preset, with the trends extended:

- PDAL is fastest on every class but Random QP, Eq QP and Lasso, where it
  is within 1.14-1.23x of the winner. Its ratio to PIQP falls with size
  on Random QP (1.7x at n=10, 0.63x at n=304), SVM (0.54-0.68x), Huber
  (0.77x), Portfolio (0.73-0.89x), Control (0.27-0.58x).
- DAS: fastest on Random QP up to n=146 and on Control up to n=109
  (0.42-0.71x PIQP), but its cold cost on large optimal active sets is the
  only real drop in the suite: SVM 5.4x PIQP at n=1010 rising to 16.4x at
  n=2020 (258 s), Huber 6x, Portfolio 1.8x -> 4.1x from k=5 to k=28. All
  32 `drops` rows are SVM x DAS.
- IPM tracks PIQP on SVM, Huber and Lasso (1.0-1.5x) and pays ~2x on the
  two-sided-box classes Control (1.8-2.4x) and Portfolio (1.9-2.4x), as
  before.
- Eq QP after the range-space fix: PDAL/IPM 0.8-1.1x PIQP up to n=211,
  drifting to 1.48x at n=2009 (DAS 1.26x); the gap is the explicit
  Y = K^-1 A' and S = A Y products, which PIQP's KKT LDLT avoids. Nothing
  further planned there.

## DAQP (added 2026-09-07, both packs)

`bench_maros_meszaros --solvers daqp` (DAQP v0.9.1 on the hard two-sided
problem, l == u rows pinned active, same 1e-6 tolerance and ok gate) run
on both packs against the cached references; 30 s for the 123-problem
pack, 271 s for the 860-problem one. Solves everything; the most accurate
route in the suite (max violation 1.1e-8, max objective error 6.5e-10).
Shifted geometric mean, paper ladders (860 problems):

| group | DAQP | PDAL | PIQP | DAS |
|---|---|---|---|---|
| all | **48 ms** | 65 (1.35x) | 84 (1.74x) | 112 (2.31x) |
| Random QP | **2.5** | 4.1 (1.62x) | 3.7 (1.49x) | 3.3 (1.32x) |
| Eq QP | 4.6 (1.48x) | 3.5 (1.14x) | **3.1** | 3.2 (1.04x) |
| Portfolio | **345** | 1369 (3.97x) | 1716 (4.98x) | 4486 (13.0x) |
| Lasso | **811** | 1472 (1.81x) | 1292 (1.59x) | 2482 (3.06x) |
| SVM | 2928 (1.01x) | **2905** | 4990 (1.72x) | 44811 (15.4x) |
| Huber | **16526** | 17036 (1.03x) | 21804 (1.32x) | 130885 (7.9x) |
| Control | **34** | 38 (1.10x) | 96 (2.80x) | 50 (1.47x) |

(The 123-problem pack gives the same ordering: DAQP 29 ms, PDAL 46,
PIQP 61, DAS 80.)

The DAS comparison is the informative one. At the largest size of every
class the two take the same number of iterations -- they walk the same
dual active-set path (SVM n=2020: DAQP 2059, DAS 2477; Portfolio k=28:
2952 vs 2917; Huber: 2046 vs 2063; Lasso: 38 vs 38) -- so the entire gap
is per-iteration cost: DAQP 10.4 s vs DAS 258 s on SVM (25x), 10.2 vs
117 s on Portfolio (11x), 16.5 vs 132 s on Huber (8x). DAQP updates its
LDL' of the working-set Gram matrix by rank-one operations on add and
remove; DAS refactors the trailing block on removal (O(k^3), see the
Maros-Meszaros large-n note) and forms M = C R^-1 for all rows up front.
The iteration counts also say the active-set path itself is fine; it is
the linear algebra per step, which is a bounded engineering gap rather
than an algorithmic one. On robot-scale working sets (k <= ~40) it does
not show: DAS is 1.2-1.9x DAQP cold there (docs/daqp_comparison.md) and
wins warm.

DAQP loses only Eq QP (1.48x PIQP), where its proximal-point outer loop
runs 1-2 extra factorizations that the direct KKT solvers skip.

## DAS rank-one removal (2026-09-07, `_rank1` files)

`das::Solver::remove_row` used to shift the Gram matrix and recompute rows
r.. of the working-set LDL' from scratch (`refactor_from(r)`, O(k^3) in
scalar loops). Deleting row r leaves the trailing block's factor as
L33 D3 L33' + d_r l32 l32', a positive rank-one update, so it is now done
with the Gill-Golub-Murray-Saunders C1 sweep (DAQP's `remove_constraint`)
in O((k-r)^2); the Gram matrix is still kept and `refactor_from` remains
the fallback when a pivot lands at or below `sing_tol`, which keeps the
"only the last pivot can be singular" invariant the LDP relies on. The
elastic mechanics (multiplier caps, saturation into `uS_`, hard and
equality rows) sit outside the factorization and are untouched.

Validation: every DAS iteration count is identical before and after on
all 983 problems of both packs (the working-set path does not change),
all solved, max violation 1.0e-7; C++ suite and Python tests pass; robot
control (`bench_robot_control`, same core, two repeats): identical
iterations and fail counts, DAS mean-time geomean ratio 0.98 (range
0.95-1.00), i.e. neutral to slightly better.

DAS shifted geomean before -> after (paper ladders, 860 problems; DAQP
for reference):

| group | DAS before | DAS after | speedup | DAQP | DAS/DAQP |
|---|---|---|---|---|---|
| all | 112 ms | 89 ms | 1.25x | 48 | 1.85x |
| Random QP | 3.3 | 2.8 | 1.17x | 2.5 | 1.13x |
| Eq QP | 3.2 | 3.3 | 1.0x | 4.6 | 0.72x |
| Portfolio | 4486 | 3359 | 1.34x | 345 | 9.7x |
| Lasso | 2482 | 2417 | 1.03x | 811 | 3.0x |
| SVM | 44811 | 8336 | 5.4x | 2928 | 2.85x |
| Huber | 130885 | 48666 | 2.7x | 16526 | 2.9x |
| Control | 50 | 52 | 1.0x | 34 | 1.53x |

Solo (one core) at the largest dense-pack sizes: SVM n=1616 52.6 -> 9.1 s,
Huber n=2107 19.3 -> 10.6 s, Portfolio n=1616 14.0 -> 6.7 s. The 32
`drops` rows (SVM x DAS >= 10x PIQP) are gone; DAS moves from last to
fourth overall.

What is left (2-10x DAQP on Portfolio, Huber, Lasso, SVM at equal
iteration counts) was profiled with per-phase timers on Portfolio n=1616
(1665 iterations, 44 removals; -O3 -march=native, 4.6 s total):

| phase | s | work per iteration |
|---|---|---|
| KKT scan `mu = M'u - d` | 1.82 | GEMV over all m+p = 3217 rows (41 MB streamed) |
| `compute_csp` | 0.90 | k dots `M_i . uS` (21 MB streamed) + LDL' solve |
| `add_row` | 0.81 | Gram column (k dots) + one row of L |
| `u = uS - M_W lam` | 0.59 | streams the working-set columns again |
| removals | 0.06 | now negligible |

So the remaining cost is streaming the constraint matrix three times per
iteration, memory bound at ~38 GB/s, not arithmetic. The scalar LDL'
triangular loops were replaced by Eigen `triangularView` solves anyway
(one `ldl_solve` helper used by `compute_csp`, the refinement, the
singular step and the equality certificate; `refactor_from` reads the
symmetric Gram column): identical iterations on all 123 dense-pack
problems, DAS 8% faster overall (Huber 0.82x, Portfolio 0.85x, SVM
0.85x), robot control identical iterations and 0.98 time ratio.

Why DAQP is still 3-10x ahead on these classes is structural, not loop
quality: the bench hands DAQP the HARD problem, where 0 <= x <= 1
(Portfolio), the slack bounds (SVM, Huber) are simple bounds that DAQP
keeps out of M entirely, and each two-sided row is one row. On Portfolio
DAQP's M has 18 general rows against our 3217 elastic rows, so its KKT
scan is ~180x cheaper; on SVM 1616 rows against 3200. ElastiQP's DAS sees
every bound as a general elastic row (and each two-sided constraint as
two) by design of the elastic interface. Closing that gap is the shelved
`experimental-das-box-bounds` branch (sparse/identity rows kept out of M;
hum-wbc -8..-11% but biman-ik +4..8%), i.e. a formulation change with a
mixed robot-scale record, not a tuning of the current solver. Left as is.

## Final tables (2026-09-07, `_final` files: every route as it stands)

DAS re-run with the rank-one removal and the Eigen triangular solves,
merged with the other five routes' existing rows (references shared, so
the penalties are identical). All six routes solve every problem on both
packs; no `drops` rows remain. Shifted geometric mean, ms (x best):

Paper ladders, 860 problems:

| group (problems) | DAQP | ElastiQP-PDAL | PIQP | ElastiQP-DAS | ProxQP | ElastiQP-IPM |
|---|---|---|---|---|---|---|
| all (860) | **48 (1.00x)** | 65 (1.35x) | 84 (1.74x) | 82 (1.70x) | 112 (2.32x) | 121 (2.51x) |
| Random QP (150) | **2.5** | 4.1 (1.62x) | 3.7 (1.49x) | 2.7 (1.09x) | 8.2 (3.24x) | 4.5 (1.77x) |
| Eq QP (200) | 4.6 (1.48x) | 3.5 (1.14x) | **3.1** | 3.3 (1.06x) | 3.6 (1.16x) | 3.5 (1.13x) |
| Portfolio (120) | **345** | 1369 (3.97x) | 1716 (4.98x) | 2921 (8.48x) | 2185 (6.34x) | 3688 (10.7x) |
| Lasso (100) | **811** | 1472 (1.81x) | 1292 (1.59x) | 1849 (2.28x) | 6196 (7.64x) | 1932 (2.38x) |
| SVM (80) | 2928 (1.01x) | **2905** | 4990 (1.72x) | 7094 (2.44x) | 6605 (2.27x) | 4988 (1.72x) |
| Huber (10) | **16526** | 17036 (1.03x) | 21804 (1.32x) | 35842 (2.17x) | 21691 (1.31x) | 21508 (1.30x) |
| Control (200) | **34** | 38 (1.10x) | 96 (2.80x) | 51 (1.48x) | 62 (1.81x) | 192 (5.60x) |

Dense preset, 123 problems:

| group (problems) | DAQP | ElastiQP-PDAL | PIQP | ElastiQP-DAS | ProxQP | ElastiQP-IPM |
|---|---|---|---|---|---|---|
| all (123) | **29 (1.00x)** | 46 (1.59x) | 61 (2.12x) | 59 (2.05x) | 82 (2.85x) | 83 (2.89x) |
| Random QP (18) | **3.4** | 4.7 (1.40x) | 4.4 (1.32x) | 3.6 (1.07x) | 9.5 (2.82x) | 5.2 (1.54x) |
| Eq QP (18) | 4.5 (1.42x) | 3.7 (1.14x) | **3.2** | 3.3 (1.04x) | 3.6 (1.14x) | 3.6 (1.14x) |
| Portfolio (18) | **31** | 145 (4.69x) | 172 (5.55x) | 221 (7.14x) | 228 (7.39x) | 369 (12.0x) |
| Lasso (18) | **33** | 65 (1.98x) | 61 (1.86x) | 67 (2.06x) | 169 (5.17x) | 84 (2.56x) |
| SVM (18) | **89** | 143 (1.61x) | 299 (3.35x) | 269 (3.01x) | 414 (4.64x) | 315 (3.53x) |
| Huber (15) | **551** | 762 (1.38x) | 1165 (2.11x) | 1570 (2.85x) | 1253 (2.27x) | 1287 (2.34x) |
| Control (18) | **26** | 27 (1.07x) | 67 (2.61x) | 37 (1.44x) | 43 (1.70x) | 129 (5.06x) |

Max hard violation / relative objective error (860): DAQP 1.1e-8 /
6.5e-10, DAS 9.9e-8 / 2.5e-8, IPM 0 / 2.4e-7, PDAL 9.5e-7 / 8.6e-7,
PIQP 1.6e-7 / 8.6e-7, ProxQP 9.8e-7 / 5.4e-7. Figures:
`results/osqp_benchmarks_{profile,scaling}_{osqp,dense}_eps1e-6_final.{png,svg}`.
DAS overall on the paper ladders across the day: 112 ms -> 89 (rank-one
removal) -> 82 (triangular solves); SVM 44811 -> 8336 -> 7094.

## Reproduce

    cd elastiqp_benchmarks
    ../.venv/bin/python tools/convert_osqp_benchmarks.py \
        ../untracked/osqp_bench/osqp_dense.bin           # 123 problems, 1.5 GB
    ../.venv/bin/python python/run_osqp_benchmarks.py \
        --pack ../untracked/osqp_bench/osqp_dense.bin \
        --work-dir ../untracked/osqp_bench/work_dense --cores 0,1,2,3
    ../.venv/bin/python python/perf_profile.py results/osqp_benchmarks_results_osqp_eps1e-6_final.csv --svg

`--preset osqp --max-vars 3000` on the converter gives the paper's ladders
truncated to dense-tractable sizes (10 seeds); `--summary-only` on the
runner rebuilds the tables from a finished work dir.
