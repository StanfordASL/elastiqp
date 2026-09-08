# DAQP vs ElastiQP on the robot control loops (2026-09-05)

DAQP (dual active-set, Arnström et al.) is the default QP backend of several
IK libraries, so it was added to `elastiqp_benchmarks/external/
robot_solver_comparison.cc` as a route family. This note records what was
measured and what it means for the paper. Raw data:
`elastiqp_benchmarks/results/robot_solver_comparison_daqp_20260905.csv`.

## Setup

* DAQP v0.9.1, pinned tarball, built from its C sources into a static
  `bench::daqp` target with upstream's default flags (PROFILING, FASTER_MATH).
* Routes, each cold and warm (working set seeded from the previous
  multipliers, as DAQP's Eigen wrapper does):
  * `daqp-hard`: the strict QP. Equalities are rows flagged
    ACTIVE|IMMUTABLE; DAQP eliminates them itself when there are enough.
  * `daqp-soft`: every inequality row flagged SOFT. DAQP's soft constraint is
    a *quadratic* slack penalty (dual Hessian + rho_soft I on the soft rows),
    i.e. an l2 relaxation with a single global weight.
  * `daqp-slack`: the expanded l1 formulation x' = [t; x], t >= 0 as simple
    bounds, singular Hessian handled by DAQP's proximal-point outer loop
    (`--daqp-slack-reg` adds an explicit slack regularization instead).
* Unit rows of G (dampers, velocity/acceleration boxes) are passed as DAQP
  simple bounds for the leading variables that have one, as an IK library
  passing lb/ub would. Every tick is a full update (Cholesky of Q, M = G R^-1),
  timed like the other routes' update()+solve().
* Same machine and binary as the pinned table; the existing solver rows
  reproduced within noise.

## Feasible sequences (mean us per tick)

| scenario | elastiqp cold / warm | daqp-hard cold / warm | daqp-soft cold | piqp-hard |
|---|---|---|---|---|
| diff-ik (n=6, p=24) | 0.5 / 0.9 | 0.2 / 0.2 | 0.2 | 11.6 |
| arm-osc (n=6, p=24) | 1.2 / 1.0 | 0.2 / 0.2 | 0.2 | 15.8 |
| biman-ik (n=12, m=6, p=48) | 3.0 / 3.0 | 2.8 / 2.8 | 3.2 | 37.0 |
| hum-wbc (n=46, m=18, p=132) | 77 / 34 | 42 / 41 | 52 | 552 |

* On the small IK problems DAQP is the fastest solver measured, 2-5x under
  ElastiQP, with one active-set iteration per tick warm or cold.
* On hum-wbc DAQP beats ElastiQP cold but not warm. Its warm solve takes one
  iteration yet still costs 41 us: the per-tick Cholesky of Q, the
  formation of M = G R^-1 and the equality elimination are DAQP's floor
  whenever the problem data change. ElastiQP warm reuses its factorization.

## Conflict sequences (one row tightened past what the rest admit)

| scenario | elastiqp warm: us, violated rows / l1 | daqp-hard | daqp-soft warm: us, violated rows (spurious) / l1 |
|---|---|---|---|
| diff-ik | 4.1, 1 / 1.55 | infeasible on every tick | 0.7, 2 (1) / 1.55 |
| arm-osc | 3.8, 1 / 5.0 | infeasible | 0.9, 6.8 (5.8) / 25.8 |
| biman-ik | 18, 1 / 1.56 | infeasible | 3.5, 3.9 (2.9) / 1.68 |
| hum-wbc | 115, 1 / 2.5 | infeasible (75 us, 50 it) | 57, 28 (27) / 147 |

* `daqp-hard` returns nothing usable on a conflict tick (dual unbounded).
* `daqp-soft` is fast but smears the violation over many rows: on hum-wbc 28
  rows move, 27 of which did not need to, and the total violation is 60x the
  l1 answer. The row structure matches ProxQP's closest-feasible mode (also
  l2). Sweeping rho_soft from 1e-4 to 1e-10 does not change it: no weight
  recovers the sparse, exact-penalty shift, because the penalty is l2.
* The expanded l1 form in DAQP is exact (1 row, same l1 as ElastiQP) but
  costs 20 us on diff-ik and 3 ms on hum-wbc, 25-136 iterations cold: every
  slack bound t >= 0 is active in the feasible case and each is a rank-1
  working-set update, and the singular slack block forces the prox loop.

## What to say in the paper

1. Active-set methods are the right tool for small, feasible, warm-startable
   IK. ElastiQP is within 2-5x of DAQP there and both are microseconds.
2. Under conflict, active-set hard mode fails outright, and the soft mode
   available in DAQP is l2, with the violation spread and magnitude penalty
   that implies. The exact-penalty (l1) argument is what distinguishes
   ElastiQP, not raw speed.
3. Structural pros/cons of dual active-set worth stating: exact termination
   with no parameters; iteration count is combinatorial (scales with active
   set changes, unpredictable timing); early termination yields a primal
   infeasible point; dense O(n^2) per row, so n up to ~100; needs H > 0
   (semidefinite / LP through a proximal loop, i.e. the same prox term as
   ProxQP/ElastiQP).
4. An elastic dual active-set is natural: the dual of the l1-elastic QP is the
   strict dual with the multipliers boxed to [0, w], so the working set gains
   a third "saturated" state (lambda = w, row leaves the working set and
   contributes a fixed force). Prototyped and measured below.

## Elastic DAQP prototype (`elastiqp_benchmarks/common/elastic_daqp.hpp`, now `include/elastiqp/das.hpp`, the `das` backend)

Single-header, Eigen-only re-implementation of DAQP's core (least-distance
problem in u = R x + R^-T q, recursive LDL' of the working-set Gram matrix,
row normalization, singular-direction step for dependent working sets,
iterative refinement) on the l1-elastic problem directly:

* the dual is box-constrained, 0 <= lambda_i <= w_i |M_i|, so a row is
  inactive (lambda = 0), active (in the working set) or saturated
  (lambda = w, out of the working set, contributing the fixed shift
  -w_i M_i' to u). A row enters the working set when violated (inactive) or
  strictly satisfied (saturated); the line search is blocked by either bound
  and the blocking row leaves to the bound it hit. Equalities are free rows
  that never leave; hard rows are w = inf. With every row elastic the dual is
  bounded, so the method terminates with a solution for any data.
* outer proximal-point loop as in `daqp_prox`: if the Cholesky of Q finds a
  pivot ratio below zero_tol, Q + eps I is factored and
  x_{k+1} = argmin{elastic QP with Q + eps I, q - eps x_k} is iterated on the
  warm working set (only the LDP right-hand side changes, the LDL' is reused)
  until eps |x_{k+1} - x_k| <= eta. Positive-definite Q takes one round.
* warm start keeps working set, saturated set and multipliers; when only
  q, h, b change the LDL' is reused as is.

Test (`elastiqp_bench.elastic_daqp`, all passing): random elastic QPs
(feasible, conflicting pairs, with equalities, mixed penalties 1/1e4,
rank-deficient Q, LP) match ElastiQP's objective to 1e-8 with KKT residuals
1e-10..1e-8 (the LP and rank-deficient cases take 2 prox rounds); the
`test_bcl_creep` cells as a 100-tick warm chain; the `test_gap_creep`
pinch/release tick (4 iterations in, 2 out, duals leave the cap).

Creep cells (warm chain, per tick; ElastiQP at eps 1e-5 with Ruiz):

| cell | edaqp us / active-set it (cold tick, warm worst) | elastiqp us / Newton it | max x diff |
|---|---|---|---|
| feas uniform 1e4, n=14 p=100 | 9 / 11 (141, 43) | 117 / 17.7 | 1.3e-5 |
| degen uniform 1e4, n=14 p=100 | 20 / 25 (291, 109) | 158 / 23.2 | 3.6e-5 |
| infeas spike 10/1e4, n=30 m=8 p=200 | 36 / 22 (467, 83) | 310 / 23.9 | 6.8e-6 |

The regime that makes BCL creep (rows drifting across the saturation
boundary) costs a dual active-set method one working-set change per row
crossing: no mu schedule, no stall. The cold start pays one change per
active row (141-467), which is the combinatorial cost the paper should name.
One caveat on tolerances: DAQP-style tolerances are on the normalized rows,
so with w = 1e4 an inactive row violated by 3e-5 shows up as a 0.3
complementarity product in the full KKT residual; the primal and dual
residuals themselves are 1e-8 to 1e-5 (addressed below by stating the
tolerances in user units).

### Robustness additions (Ruiz, equality certificate, cycle guard)

All on by default, each covered by `elastiqp_bench.elastic_daqp`:

* Ruiz equilibration, ElastiQP's pass (column/row factors 1/sqrt(max-norm),
  cost scaling, penalties scaled with their rows), at setup and refreshed
  when a matrix update drifts the scaled norms more than 4x from 1. On the
  same problem reparametrized with column scales 10^(+-3) and row scales
  10^(+-2), the bare method hits the iteration cap (1104 changes, wrong
  answer); with Ruiz it solves in 67 changes to 1e-11 in x. The LDP's own
  row normalization does not help there: the column and cost scaling is
  what conditions R and M = C R^-1.
* Equality consistency certificate: dependent equality rows surface as a
  singular pivot while the working set is built; before any iteration they
  are checked against the least-norm solution of the independent ones, and
  inconsistent data returns kInfeasible with `eq_infeasibility()` (0.5 on
  the test's shifted duplicate row, 0 iterations spent).
* DAQP's cycle guard: the dual objective must increase between dual-feasible
  points; after cycle_tol stalls the working set is refactored from scratch
  in a different pivot order, a second stall is kNumerics; a pivot below
  refactor_tol at optimality triggers the same refactor before accepting.
  The duplicated-rows case (dependent working sets at every step) solves
  without it firing; the guard is insurance, not a hot path.

Cost: 0.3-0.5 us per tick on the 6-DoF loops and ~5 us on hum-wbc for the
per-tick drift check and scaled copies (the refresh never fires on the
robot data).

### Tolerances in user units, factorization reuse

* Tolerances: `eps_abs` / `eps_rel` are stated on the user-frame row
  violation G_i x - h_i (and on a saturated row's slack), converted per row
  into the normalized LDP units (tol_i = eps scale_i dr_i); the row with the
  largest user-unit violation enters the working set. The proximal loop
  stops when the user-frame stationarity residual of the unshifted problem
  is below `eps_abs` as well (`eta_prox` overrides that tolerance when set). On the creep
  cells the inactive-row violations drop from 3e-5 to 0 at the same
  iteration counts, x matches ElastiQP to 1e-8, and the full KKT residual
  (complementarity included) goes from 0.3 to 1.5e-4. Complementarity is
  bounded by penalty_i * eps, which is what a user-unit constraint
  tolerance implies for an l1-elastic row.
* Factorization reuse: the Cholesky of Q is kept whenever Q is unchanged;
  set_G / set_A diff their rows and only the changed rows are re-solved
  against R in one batched triangular solve; the working-set LDL' is kept
  unless one of its rows changed; q / h / b updates touch only the LDP
  right-hand side. `reuse_factorization = false` forces the full path. On a
  hum-wbc-sized chain (n=46, m=18, p=132) with constant Q and a fifth of
  the G rows changing per tick: 43 us/tick against 74 with a full
  refactorization every tick; with only q/h changing, 34 against 117.
  Identical solutions (2e-10). The robot loops below change Q every tick,
  so the reuse does not show there: forming M = G R^-1 is about half of the
  per-tick floor at that size and is unavoidable when R changes. The
  remaining per-iteration cost (~5 us at n=46, p=132) is Eigen small-op
  overhead in the prototype and the next lever.

Robot loops, all of the above on (`edaqp` = elastic DAQP):

| scenario | variant | elastiqp cold / warm us | edaqp cold / warm us (it) | daqp-hard cold / warm |
|---|---|---|---|---|
| diff-ik | feasible | 0.7 / 1.3 | 1.2 / 1.1 (1.0) | 0.2 / 0.2 |
| diff-ik | conflict | 7.9 / 4.1 | 1.5 / 1.2 (4.5 / 1.0) | infeasible |
| arm-osc | feasible | 1.2 / 1.0 | 1.1 / 1.1 (1.0) | 0.2 / 0.2 |
| arm-osc | conflict | 8.0 / 3.8 | 1.9 / 1.2 (6.6 / 1.0) | infeasible |
| biman-ik | feasible | 3.0 / 3.0 | 3.5 / 3.5 (1.0) | 2.8 / 2.8 |
| biman-ik | conflict | 38 / 18 | 5.7 / 3.8 (6.9 / 1.1) | infeasible |
| hum-wbc | feasible | 74 / 33 | 49 / 46 (3.9 / 1.0) | 40 / 40 |
| hum-wbc | conflict | 439 / 110 | 228 / 59 (69 / 1.6) | infeasible (74, 50 it) |

Tail latencies (max us over the sequence, warm): hum-wbc conflict edaqp 323
vs elastiqp 494; biman-ik conflict 13 vs 40; hum-wbc feasible 93 vs 77.

Every conflict tick returns the same l1 answer as ElastiQP (1 violated row,
identical l1 shift). The elastic working set costs nothing over DAQP-hard on
the feasible sequences (the saturated shift is a vector add), and on the
conflict sequences the warm elastic DAQP is 2-6x faster than warm ElastiQP.
Its floor on hum-wbc is the same 40 us per-tick setup as DAQP (Cholesky of Q,
M = G R^-1, no equality elimination in the prototype) plus ~5 us of Ruiz
bookkeeping, which ElastiQP warm undercuts on the feasible sequence. The gap
to DAQP-hard on the 6-DoF problems is Eigen overhead and the per-tick
scaling pass in the prototype, not the method.

## Maros-Meszaros small dense subset (`bench_maros_meszaros`, edaqp route)

Same harness as the paper's Table III route (n <= 200, elastic form with
penalty 10x the hard problem's largest dual, cold, eps 1e-6, Ruiz on for
both), CSV `elastiqp_benchmarks/results/maros_meszaros_results_edaqp_20260905.csv`.

* Robustness: edaqp 36/36 converged and 34 matched the hard reference
  objective to 1e-5, identical to ElastiQP (the two unmatched instances,
  QPCBOEI2 and QSCAGR7, are the ones where the PIQP reference itself fails).
  ProxQP-hard: 35/36, 33 matched. Every objective gap is 1e-9 or better.
  Singular P instances go through the proximal loop without incident.
* Speed: geometric mean 2.0x faster than ElastiQP over the 36 problems.
  Large wins where p >> n (DUALC*, KSIP: 5-27x; HS118 7x), typical 1.2-2.5x,
  slower on DUAL2/3/4 (0.5-0.9x: n ~ 100, many active rows, 2-3 PDAL
  iterations suffice) and QISRAEL (0.9x, 715 working-set changes against
  292 Newton steps). Against piqp-hard it is faster on 28 of 36.
* Read with the same caveat as the rest of the note: cold solves, iteration
  count combinatorial (QISRAEL, QSHARE2B, QADLITTL at 160-715 changes).


## Full sweep: edaqp through the remaining relevant tests and benchmarks

Everything in `tests/` and `elastiqp_benchmarks/` that tests the elastic QP
being solved (as opposed to PDAL internals: `relax()`, the KKT VJP, the BCL
schedule, `set_warm_start`, the IPM reference's own tests) now runs the
prototype too. Raw outputs and CSVs:
`elastiqp_benchmarks/results/edaqp_sweep_20260905/`.

* `test_elastic_daqp_oracle` (new, ctest `elastiqp_bench.elastic_daqp_oracle`):
  the non-PDAL cells of `test_pdal.cc`, `test_lp.cc`, `test_ruiz.cc` and
  `test_piqp_cross_validation.cc` with the same generators, seeds and
  thresholds, oracle = IPM reference or vanilla PIQP, tolerances 1e-8.
  **69/69 checks pass**; 51 timed cells, edaqp geomean 2.6x faster than
  elastiqp cold, faster by >1.5x on 43, slower by >1.5x on 1 (n=58 m=15
  p=400 infeasible, 1.55x: cold start with ~500 working-set changes).
  Semantics differences surfaced: reconstructed slacks are ~1e-15 rather
  than identically 0 (active rows satisfied to roundoff instead of the
  PDAL's exact clamp); `check_eq_consistency=false` on inconsistent
  equalities reports kSolved on the consistent subset (dependent row
  dropped) where the PDAL runs to kMaxIter; no `set_warm_start`.
  Added `Solver::rescaled()` so a Ruiz refresh is observable
  (`scaling_drift()` reads 1 after one).
* `bench_random_qp` (cold, eps 1e-5): faster on every family at n <= 30
  (1.4-2.4x); at n=100 p=200 faster on feasible (1.1-1.4x), slower on the
  infeasible families (0.65-0.8x; 314-380 changes vs 41-64 Newton steps).
* `bench_condensed_vs_expanded` (cold, eps 1e-8): faster on all 36 cells,
  2-25x feasible, 1.3-4.4x infeasible; the margin shrinks with n (n=58
  infeasible p >= 200: 1.26-1.5x). Penalty sweep at n=30 p=200: faster at
  penalty 1 and 100, **slower at 1e4 and 1e6 (1400/1790 vs 1050 us, 635-643
  changes)**, and the full KKT residual there is penalty x eps
  (complementarity on active rows, 2e-6 at eps 1e-8 / penalty 1e4; x itself
  matches elastiqp to 1e-10, eps 1e-10 brings it to 3e-10). Rank-deficient
  Q: 2.5x / 1.7x faster at n=14/30, 0.9x at n=58 p=500 (2 prox rounds,
  1164 changes); KKT 1e-8..1e-7 there, measured when `eta_prox` was a fixed
  1e-6 default (it now follows `eps_abs`).
* `bench_proxqp_closest` (eps 1e-6): 3.1x / 2.8x / 1.25x faster than
  elastiqp at n=14/30/58, identical violation structure (nnz, spurious, l1,
  linf all equal to elastiqp's); feasible overhead table 5-8x faster.
* `bench_collision_2d` (n=4, p=8, forward only): warm 0.65 us vs 1.94,
  cold 2.2 vs 3.4; worst |x - elastiqp| 3e-5 at penalty 1e4 / eps 1e-5.
* `bench_robot_control` (eps 1e-6, all four sequences): same picture as
  the paper table. Warm: diff-ik 1.2 vs 1.0 us, arm-osc 1.2 vs 1.1,
  biman-ik 3.7 vs 3.0, hum-wbc 47.8 vs 33.0 (per-tick Cholesky + M floor).
  Cold hum-wbc 52 vs 75 us. Tails: hum-wbc cold max 139 vs 387 us, warm
  max 100 vs 77, p95 50 vs 43. 0 fails. hum-wbc's Q has eigenvalues down to
  1e-6, so x differs from elastiqp's by up to 0.3 at equal objective
  (2e-10) -- both are valid at eps 1e-6.
* `bench_fwd_warm` (100-tick drifting random QPs, 3 structures x 3 sizes x
  4 sigmas x 2 penalties + drift composition + accuracy tier, eps 1e-5):
  **0 fails in 104 cells** (elastiqp also 0). Warm edaqp is faster than
  warm elastiqp in every cell, 1.65-13x (median ~4x); at penalty 1e4 the
  elastiqp warm chain costs 90-2200 us/tick vs 4-1300 for edaqp. Cold:
  faster at n <= 30 and penalty 10, but at n=58 p=400 with penalty 1e4 the
  cold start needs 900-1230 changes and costs 4-8.8 ms vs 2.9-3.3 ms for
  elastiqp (0.35-0.7x) -- the one consistent loss. Warm-vs-cold |dx| is
  at the 1e-5..1e-9 level throughout, same as elastiqp's.
* `bench_eq_elastic`: hard equalities are exact (||Ax-b|| 1e-11..1e-13 vs
  1e-7..1e-9 for the PDAL, no mu_eq schedule). edaqp-hard vs elastiqp-hard:
  hum-wbc 0.69x cold / 1.39x warm; biman-ik 1.2x; random n=20/n=60 chains
  0.17-0.47x (2-6x faster). Folding equalities into `[A; -A]` elastic pairs
  costs edaqp 1.1-1.8x (vs 1.1-5.7x for the PDAL) and is insensitive to
  the weight w.
* Not run: `test_robot_control` (needs Pinocchio; its problems are the
  replayed sequences above), the Python
  bindings/JAX/torch tests and `bench_relax_warm` / `bench_diff_robot` /
  `bench_bcl_strategies` (PDAL-specific).

Summary of where edaqp loses to the PDAL: warm hum-wbc-size loops where the
per-tick setup floor dominates (1.2-1.45x slower), and cold starts at
n >= 58 with many conflicting rows or a large penalty (up to 2.9x slower,
iteration count in the hundreds to ~1200). Nothing failed to solve.

## Maros-Meszaros, every solver on one profile (2026-09-05)

`bench_maros_meszaros` now also runs `daqp` (DAQP v0.9.1 on the hard
two-sided problem, unit rows mapped to its simple bounds, cold one-shot
`daqp_quadprog`) and can write the Python runner's CSV schema
(`--py-csv`); `run_maros_meszaros.py --solvers qpax-hard,qpax-elastic
--merge-into <csv>` appends the qpax rows and rewrites the summary. So the
C++ solvers are timed in C++ and only qpax goes through Python. Both halves
pinned with `taskset -c 2`, eps 1e-6, n <= 200 subset (36 problems), penalty
= 10x the largest reference dual (the C++ harness now follows the Python
rule exactly, no upper clamp: QPCBOEI2's duals exceed 1e7 and the old 1e8
cap left a 2e-3 violation). Files:
`results/maros_meszaros_all_{results,summary}_n200_eps1e-6.csv`,
`results/maros_meszaros_all_profile_n200_eps1e-6.{png,svg}`,
`results/maros_meszaros_all_cpp_stdout_n200_eps1e-6.txt`.

| solver | solved | sgm ms (shift 1) | vs best |
|---|---|---|---|
| daqp | 36/36 | 0.255 | 1.00 |
| edaqp | 36/36 | 0.601 | 2.36 |
| piqp | 36/36 | 0.809 | 3.17 |
| elastiqp | 36/36 | 0.875 | 3.43 |
| proxqp | 35/36 | 1.308 | 5.13 |
| ipm (elastic IPM reference) | 35/36 | 1.310 | 5.14 |
| qpax-elastic | 28/36 | 5.23 | 20.5 |
| piqp-expanded | 35/36 | 6.65 | 26.1 |
| qpax-hard | 17/36 | 26.4 | 103 |

(`ipm` = `include/elastiqp/ipm.hpp` (formerly `tests/support/ipm_reference.hpp`), the original elastic-PIQP
experiment, cold on the same elastic form, same eps and Ruiz as elastiqp;
added in a second pinned pass, so the other rows moved by a few percent.)

* DAQP on the hard problem is the fastest solver on all 36 instances
  (tau = 1 everywhere): these are feasible, small, bounds-heavy QPs, the
  active-set method's home turf, and its simple-bound handling halves the
  row count the elastic routes carry (each two-sided row becomes two
  one-sided elastic rows). edaqp sits between it and PIQP.
* Relative to the pinned Python-timed run
  (`maros_meszaros_py_summary_n200_eps1e-6.csv`): elastiqp 0.84 vs 1.98 ms
  and proxqp 1.27 vs 1.63 ms are the C++-vs-bindings gap (the Python route
  builds the one-sided form and crosses nanobind per solve); piqp 0.78 vs
  0.62 ms is the C++ harness's two-sided interface with duplicated rows
  dropped differently -- within the run-to-run band. qpax rows are
  identical in method to the pinned run (28/36 and 17/36 solved, same
  instances).
* The elastic IPM reference lands on top of ProxQP: 35/36 (QPCBOEI2 hits
  the 250-iteration cap at penalty 1e8), geomean 1.6x slower than
  elastiqp and 1.4x slower than PIQP. Per problem it splits cleanly by
  iteration count: where the PDAL needs hundreds of semismooth-Newton steps
  (QSCAGR7 505, QISRAEL 292, QSHARE2B 207, QADLITTL 159, KSIP 33 at
  p = 1001) the IPM's 14-37 iterations win by 1.2-4.5x; where the PDAL
  finishes in 2-20 steps (DUAL*, CVXQP*, QRECIPE) the IPM's fixed 10-30
  iterations lose by 2.5-7.4x. Against PIQP on the hard problem it is
  1.1-2.6x slower on 30 of 36: same iteration counts, but each elastic
  KKT solve carries the p slack rows and the penalty-scaled conditioning.
* Success per the Python rule (reported success, hard violation <= 1e-4,
  relative objective error <= 1e-4): elastiqp and edaqp now pass all 36
  (QPCBOEI2 needed the unclamped penalty), proxqp fails QRECIPE.
