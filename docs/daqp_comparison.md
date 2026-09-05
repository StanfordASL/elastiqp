# DAQP vs ElastiQP on the robot control loops (2026-09-05)

DAQP (dual active-set, Arnström et al.) is the default QP backend of several
IK libraries, so it was added to `elastiqp_benchmarks/experiments/
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

## Elastic DAQP prototype (`elastiqp_benchmarks/common/elastic_daqp.hpp`)

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
  largest user-unit violation enters the working set. `eta_prox` is the
  user-frame stationarity residual of the unshifted problem. On the creep
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

