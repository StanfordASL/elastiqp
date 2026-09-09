# Sparse elastic PDAL: design, MPC benchmarks, failure modes (2026-09-09)

`include/elastiqp/sparse_pdal.hpp` is the elastic PDAL method of
`pdal.hpp` on `Eigen::SparseMatrix` data, written for multiple-shooting
MPC and the other block-sparse QPs of robot control. This note records
what was built, how it was validated, what it costs against sparse PIQP
and sparse ProxQP (OSQP problem classes, larger Control ladder, three
closed-loop robot MPC scenarios), which of the dense backend's tricks
transfer, and which failure modes are new. Benchmarks and generators live
in the `elastiqp_benchmarks` repo (`bench_sparse_qp`, `bench_mpc`,
`robotics/mpc_models.hpp`) and `tests/support/mpc_gen.hpp`.

## 1. What is the same, what is different

The method is unchanged: three-state classification of every inequality
row on the unclamped multiplier estimate (inactive / active / saturated),
semismooth Newton with the exact piecewise-quadratic line search, the
elastic BCL outer loop with its split revert, saturation and release jumps
and no cold reset (`docs/elastic_bcl.md`). `sparse_pdal::Settings` derives
from `pdal::Settings`, so every BCL knob is shared. Ruiz equilibration,
the equality-consistency certificate, the `p == 0` path, `set_*` updates
with factorization caching and automatic re-equilibration all have sparse
implementations. Not implemented: `relax()` / the KKT VJP, Python
bindings.

The linear algebra is where the two differ:

**Augmented, not condensed.** The dense backend condenses onto the SPD
`K = Q + rho I + A'A/mu_eq + G_J'G_J/mu_in` (an `n x n` LLT). Its sparse
counterpart would square the dynamics matrix (fill) and carry a condition
number of order `1/mu`. The sparse backend factors the quasidefinite
augmented system

```
[ Q + rho I    A'        G_J'   ] [dx]   [rhs_x]
[ A         -mu_eq I      0     ] [dy] = [rhs_y]
[ G_J          0      -mu_in I  ] [dz]   [rhs_z]
```

with `Eigen::SimplicialLDLT` (AMD ordering). Quasidefinite matrices admit
an LDL' under any symmetric permutation without pivoting (Vanderbei), so
the ordering is computed **once** at `setup()` for the pattern with every
inequality row present; a row that is not active keeps its column with
the coupling values zeroed and a `-1` on the diagonal, so it decouples
exactly and the symbolic analysis is never redone. Numeric refills are
`O(nnz)` index gathers. The LLT-instead-of-LDLT observation of the dense
backend does not carry over: the augmented system is indefinite by
construction, but the same convexity restriction is what lets the LDL' run
with a static ordering and no pivoting (ProxQP's sparse LDL' carries
dynamic regularization for the general case).

**Woodbury instead of rank-one updates.** Eigen's sparse Cholesky has no
rank-one updates, and a refactorization on every active-set change would
cost a factorization per Newton step (the dense backend pays `O(n^2)`
per flip through `LLT::rankUpdate`). Instead the factored matrix `K0`
(active set `J0`) is kept and the current Newton matrix is written as
`K0 + U S U'`, `U = [G_F'; 0; 0]`, `S = diag(+-1/mu_in)` over the rows
`F` whose state differs from `J0`. A solve costs one solve with `K0` plus
`|F|` cached solves (`V = K0^-1 U`, grown incrementally as rows flip) and
a small dense LU; the rows in `F` get their dual step from the row
equation afterwards. Refactorization happens when mu or the data change
or `|F|` exceeds the budget (auto: the mean column count of `L`, clamped
to `[8, 512]`, the point where `|F|` cached solves cost about a
refactorization). On the closed-loop MPC problems below this halves the
factorization count of a cold solve (6 vs 12 on the test's conflict
instance) and lets a warm tick run without any.

**Iterative refinement, relative exit.** The LDL' of a matrix whose
diagonal spans `rho = 1e-6` to `-mu_eq = -1e-9` loses digits; each Newton
solve is refined against the current Newton matrix (`K0 + USU'`, one
sparse product per check). The exit test is relative
(`refine_tol = 1e-11` times `1 + |rhs|`): with the absolute `1e-15` test
it never fired and refinement cost 20-30% of every MPC tick for identical
iteration counts; with the relative test the check is usually all that is
paid (4-8%). `refine_iters = 0` was never observed to change an iteration
count on these problems, so the steps are a safety net for badly scaled
data.

**Two sparse-specific defaults.** Both are settings of `pdal::Settings`
(dense default: off) and are validated in Section 5:

- `mu_eq_init = 1e-5` (dense: `1e-3`). The augmented form pays nothing in
  conditioning for a small equality penalty, and a dynamics-dominated
  problem otherwise spends 3-4 BCL rounds walking `mu_eq` down.
- `warm_keep_mu = true`, `warm_keep_mu_max_iters = 25`. A warm solve keeps
  the penalties where the previous solve left them **together with the
  factorization they belong to**: on a quiet control tick (vector-only
  drift, unchanged active set) the cached factorization is reused
  outright. After a matrix update (a refactorization is due anyway) and
  after a tick that needed more than 25 inner iterations the penalties
  reset to their initial values (Section 6).

## 2. Validation

`tests/test_sparse_pdal.cc` (`elastiqp.sparse_pdal`): random dense
problems through `sparseView()` (feasible, conflicting, with equalities,
per-row penalties, Ruiz on, `p == 0` with a singular `Q`, inconsistent and
rank-deficient equalities) against the IPM oracle and the dense PDAL,
agreement 1e-8 to 1e-10; a multiple-shooting MPC instance cold with a
feasible and an out-of-box initial state (dense PDAL agreement 1e-11 to
1e-15); Woodbury vs refactor-every-change (same point to 2e-15, half the
factorizations); a 40-tick closed loop through a box conflict (every
tick converges, sparse warm matches dense warm to 5e-10, worst elastic
KKT residual 2.5e-8); matrix updates with and without a pattern change.
The OSQP-class run below checks every solve against a 1e-9 sparse-PIQP
reference of the hard problem (violation and relative objective error
below 1e-4): 123/123 and 18/18 pass.

## 3. OSQP problem classes (sparse solvers)

`python/run_osqp_benchmarks.py --bench build/bench_sparse_qp` on the
123-problem dense preset of `docs/osqp_benchmarks.md` (same pack, same
PIQP references, eps 1e-6, penalty 10x the largest reference dual, cold,
Ruiz on for ElastiQP), four routes: sparse elastic PDAL, dense elastic
PDAL, sparse PIQP (`sparse_ldlt`; its block-tridiagonal `sparse_multistage`
solver needs a BLASFEO build and is not available here), sparse ProxQP.
Four problems ran concurrently on four P-cores (every route of a problem
on the same core), so absolute times carry the same contention as the
dense note. Shifted geometric mean, shift 1 ms
(`results/osqp_benchmarks_summary_sparse_eps1e-6.csv`):

| group (problems) | sparse PDAL | dense PDAL | sparse PIQP | sparse ProxQP |
|---|---|---|---|---|
| all (123) | 8.8 ms (3.5x) | 49.3 (19.5x) | **2.5 (1.0x)** | 27.5 (10.9x) |
| Random QP (18) | 8.2 (2.2x) | 5.0 (1.4x) | **3.7** | 15.9 (4.3x) |
| Eq QP (18) | 4.9 (1.2x) | **4.1** | 5.6 (1.4x) | 5.9 (1.5x) |
| Portfolio (18) | 10.5 (6.2x) | 154 (92x) | **1.7** | 124 (74x) |
| Lasso (18) | 6.8 (7.6x) | 72.6 (81x) | **0.9** | 26.6 (30x) |
| SVM (18) | 16.7 (11.8x) | 149 (106x) | **1.4** | 104 (73x) |
| Huber (15) | 12.6 (8.9x) | 782 (554x) | **1.4** | 46.9 (33x) |
| **Control (18)** | 6.6 (1.1x) | 30.1 (5.1x) | **5.9** | 7.1 (1.2x) |

All four routes solve 123/123. On Control, the class this backend is for,
the sparse elastic PDAL is within 10% of sparse PIQP, ahead of sparse
ProxQP and 4.5x ahead of the dense PDAL; per problem it is the fastest
route at the two largest sizes:

| problem | n | sparse PIQP | sparse ProxQP | sparse PDAL | dense PDAL |
|---|---|---|---|---|---|
| control-n24 (3 seeds) | 384 | 3.2-3.9 ms / 7-9 it | 1.7-3.8 / 4-8 | 3.1-3.9 / 4-9 | 21-27 |
| control-n46 | 736 | 17.5-17.8 / 8 | 18.5-36.9 / 10-14 | 16.8-21.6 / 10-13 | 156-180 |
| control-n99 | 1579 | 216-246 / 9-10 | 428-487 / 12-15 | 209-249 / 14-16 | 1544-1815 |

On the other classes the sparse PDAL trails sparse PIQP by 6-12x. This is
not the linear algebra: SVM n=606 takes 84 Newton steps and 37
factorizations against PIQP's 9 iterations, because the optimal active
set (2000 of 3200 rows) is built up through large active-set changes on
every Newton step. The dense backend hides the same iteration count behind
`O(n^2)` rank-one updates; the sparse Woodbury budget cannot: raising it
from the automatic 8 to 32 / 128 / 512 flips reduces the factorizations
(37 -> 15 -> 7 -> 6) but the `|F|`-column dense products cost more than the
(cheap, low-fill) refactorizations they save, and the solve time grows
from 17 to 22 / 103 / 133 ms. PIQP's dense-vs-sparse gap on these classes
(`docs/dense_benchmark_caveats`) is an IPM's 9 factorizations vs a PDAL's
tens; an elastic IPM on sparse data would inherit it.

**Larger Control ladder** (`tools/convert_osqp_benchmarks.py --preset
control_xl --sparse`, 16n variables, 27n rows, sparse MMQS pack; dense
solvers out of reach; `results/osqp_control_xl_results.csv`, one core,
concurrent with other runs up to n=300):

| dim | n | sparse PIQP | sparse PDAL | sparse ProxQP |
|---|---|---|---|---|
| 100 | 1600 | 124-152 ms / 8-9 it | 98-114 / 9-11 | 189-234 / 10-12 |
| 150 | 2400 | 373-411 / 8-9 | 286-337 / 11-15 | 641-722 / 11-15 |
| 200 | 3200 | 930-1021 / 9-10 | 634-780 / 11-18 | 1503-2360 / 11-17 |
| 300 | 4800 | 3019-3386 / 9-10 | 1937-2296 / 11-16 | 999588 (one instance, 12 it; dropped after) |
| 400 | 6400 | 8043-10616 / 10 | 4984-5766 / 14-18 | - |
| 500 | 8000 | 15739-17600 / 10-11 | 9105-11759 / 16-20 | - |

The sparse PDAL is 0.55-0.75x sparse PIQP at every size (more Newton
steps, 3 factorizations against PIQP's 9-11, the rest absorbed by the
Woodbury correction), and sparse ProxQP became pathological at n=4800
(1000 s for 12 iterations) and was dropped from the larger instances. All
solves certified. Note that OSQP's Control instances have dense
`nx x nx` dynamics blocks (`nx = dim`), so the factor has
`nnz(L) ~ 300 dim^2` and both solvers are far from real-time above
`dim ~ 50`; they measure the linear algebra, not an MPC loop.

## 4. Closed-loop robot MPC (`bench_mpc`)

`robotics/mpc_models.hpp` builds three multiple-shooting MPC scenarios
from closed-form models (no Pinocchio), stage-interleaved variables
`[x_0; u_0; ...; x_N]`, hard dynamics, elastic boxes and general rows,
200 closed-loop ticks with process noise, disturbances, and stretches
where the elastic rows must saturate:

- **quadrotor**: hover-linearized 12-state / 4-rotor model at 20 Hz,
  thrust / tilt / velocity boxes and a keep-out wall `p_x <= 1` that the
  position reference crosses and re-crosses (92 of 200 ticks in
  conflict), a gust at tick 130. `N = 20`: n=332, m=252, p=391.
- **manipulator**: 7-DoF joint-space MPC on a double integrator with the
  Panda joint / velocity / acceleration limits, a sinusoidal joint
  reference that overshoots two joint limits by 20% every cycle (43
  conflict ticks). `N = 20`: n=434, m=294, p=868.
- **quadruped**: single-rigid-body ("convex MPC") model, 12 states and 12
  foot forces at 33 Hz, trot gait with 10-stage phases, friction pyramids
  as general rows, per-stage `B_k` from the predicted footholds (matrix
  update every tick), a lateral and a backward push, a stretch of reduced
  force limits during which the two stance feet cannot carry the body
  (the elastic bounds go tight but are never violated: the height error is
  a soft cost, so these two scenarios have no conflict ticks and measure
  the schedule handling). Two formulations of the contact schedule: **quadruped**
  pins the swing feet with fixed-pattern hard equality rows (all-zero
  rows for the stance feet, explicit zeros kept in the pattern) and
  relaxes their elastic rows; **quadruped-ineq** is the textbook
  `0 <= f_z <= 0` through the bound rows. `N = 20`: n=492, m=492 (252),
  p=480.

Every route replays the state sequence recorded with the sparse PDAL
(shifted warm start), so all routes are timed on identical QPs; the
per-tick time includes the data update (`set_b` / `update()`) and the
solve. Routes: sparse PDAL cold (equality-constrained initial guess every
tick), warm (previous iterate, cached factor, kept penalties), shift
(previous iterate shifted one stage through `set_warm_start`), the same
two with the penalties reset every tick (`-reset`, ProxQP's warm start);
dense PDAL and DAS warm / shift; sparse ProxQP cold / warm and sparse PIQP
cold on the **hard** problem, which is infeasible on the conflict ticks.
eps 1e-6, penalties 1e2 (boxes) / 1e3 (inputs, pyramids) / 5e1 (wall).
One P-core, nothing else running (`results/mpc/`). Sparse ProxQP is
absent from the pinned quadruped: its sparse backend does not converge on
the all-zero equality rows (20,000 iterations and 35 s per tick), and its
warm route is absent from the two largest quadruped-ineq horizons for
the same reason (tens of seconds per tick).

Per-tick solve time, mean / p95 in ms, `N = 20`, 200 ticks (conflict ticks: quadrotor 92, manipulator 43, quadruped 0, quadruped-ineq 0); `fail` = not converged, `inf` = declared infeasible by a hard solver:

| route | quadrotor | manipulator | quadruped | quadruped-ineq |
|---|---|---|---|---|
| spdal-cold | 0.53 / 1.11 | 1.26 / 1.88 | 0.76 / 2.72 | 2.01 / 4.53 |
| spdal-warm | 0.19 / 0.50 | 0.65 / 1.28 | 0.90 / 2.33 | 2.04 / 3.58 |
| spdal-shift | 0.26 / 0.63 | 0.74 / 1.70 | 0.52 / 0.98 | 1.03 / 1.58 |
| spdal-warm-reset | 0.50 / 1.13 | 1.17 / 1.86 | 0.91 / 2.37 | 2.08 / 3.67 |
| spdal-shift-reset | 0.56 / 1.32 | 1.05 / 1.69 | 0.53 / 1.00 | 1.04 / 1.58 |
| pdal-warm | 5.97 / 9.89 | 17.31 / 25.37 | 47.99 / 64.41 | 39.01 / 54.86 |
| pdal-shift | 6.74 / 12.20 | 15.32 / 22.77 | 42.30 / 51.05 | 27.86 / 39.58 |
| pdal-warm-mu | 1.25 / 2.79 | 4.97 / 9.84 | 45.17 / 61.32 | 38.23 / 53.80 |
| pdal-shift-mu | 1.46 / 3.28 | 5.88 / 12.10 | 42.36 / 51.06 | 27.78 / 39.07 |
| das-warm | 0.42 / 0.91 | 1.28 / 1.96 | 14.69 / 21.61 | 15.37 / 20.18 |
| das-shift | 2.66 / 3.37 | 4.91 / 5.72 | 13.75 / 16.03 | 10.90 / 16.72 |
| proxqp-cold | 1.82 / 4.51 (57 inf) | 2.20 / 3.73 (43 inf) | - | 11.05 / 34.89 (1 inf) |
| proxqp-warm | 1.07 / 4.16 (65 inf) | 2.19 / 4.11 (165 inf) | - | 46.85 / 110.33 (72 inf) |
| piqp-cold | 3.25 / 11.86 (50 fail, 6 inf) | 3.46 / 14.04 (43 fail) | 0.99 / 1.18 | 0.73 / 0.92 |

Newton steps per tick (mean / max) and factorizations per tick of the sparse routes:

| route | quadrotor | manipulator | quadruped | quadruped-ineq |
|---|---|---|---|---|
| spdal-cold | 11.2 / 32 it, 3.5 fac | 16.3 / 29 it, 5.2 fac | 7.0 / 52 it, 2.5 fac | 16.8 / 69 it, 9.7 fac |
| spdal-warm | 4.4 / 22 it, 0.1 fac | 8.7 / 27 it, 0.7 fac | 8.6 / 48 it, 3.0 fac | 17.2 / 58 it, 10.8 fac |
| spdal-shift | 6.3 / 24 it, 0.1 fac | 10.1 / 30 it, 0.9 fac | 4.7 / 19 it, 1.3 fac | 7.6 / 35 it, 4.7 fac |
| spdal-warm-reset | 11.2 / 40 it, 2.3 fac | 15.4 / 30 it, 4.1 fac | 8.6 / 48 it, 3.0 fac | 17.2 / 58 it, 10.8 fac |
| spdal-shift-reset | 12.6 / 43 it, 2.7 fac | 14.2 / 28 it, 3.1 fac | 4.7 / 19 it, 1.3 fac | 7.6 / 35 it, 4.7 fac |

Horizon sweep (`results/mpc/mpc_sweep.txt`), mean per-tick time:

`quadrotor`, mean per-tick ms (n at each horizon: N=10 n=172, N=20 n=332, N=40 n=652, N=80 n=1292):

| route | N=10 | N=20 | N=40 | N=80 |
|---|---|---|---|---|
| spdal-cold | 0.20 | 0.58 | 1.78 | 4.28 |
| spdal-warm | 0.06 | 0.20 | 0.79 | 1.92 |
| spdal-shift | 0.08 | 0.28 | 1.50 | 3.36 |
| das-warm | 0.06 | 0.41 | 3.27 | 23.05 |
| proxqp-warm | 0.32 (27 inf) | 1.15 (65 inf) | 4.85 (92 inf) | 36.84 (129 inf) |
| piqp-cold | 0.57 (14 fail, 5 inf) | 3.42 (50 fail, 6 inf) | 7.60 (51 fail, 6 inf) | 15.78 (52 fail, 5 inf) |

`manipulator`, mean per-tick ms (n at each horizon: N=10 n=224, N=20 n=434, N=40 n=854, N=80 n=1694):

| route | N=10 | N=20 | N=40 | N=80 |
|---|---|---|---|---|
| spdal-cold | 0.44 | 1.26 | 3.53 | 9.84 |
| spdal-warm | 0.19 | 0.65 | 2.43 | 10.10 |
| spdal-shift | 0.25 | 0.74 | 1.57 | 3.25 |
| das-warm | 0.18 | 1.25 | 10.31 | - |
| proxqp-warm | 57.68 (3 fail, 107 inf) | 2.21 (165 inf) | 8.76 (177 inf) | 46.36 (198 inf) |
| piqp-cold | 1.68 (42 fail) | 3.47 (43 fail) | 7.31 (45 fail) | 15.42 (45 fail) |

`quadruped`, mean per-tick ms (n at each horizon: N=10 n=252, N=20 n=492, N=40 n=972, N=80 n=1932):

| route | N=10 | N=20 | N=40 | N=80 |
|---|---|---|---|---|
| spdal-cold | 0.34 | 0.82 | 1.71 | 3.84 |
| spdal-warm | 0.34 | 0.97 | 2.40 | 5.00 |
| spdal-shift | 0.24 | 0.56 | 1.23 | 2.21 |
| das-warm | 2.05 | 15.36 | 118.41 | - |
| piqp-cold | 0.48 | 1.07 | 2.23 | 4.54 |

`quadruped-ineq`, mean per-tick ms (n at each horizon: N=10 n=252, N=20 n=492, N=40 n=972, N=80 n=1932):

| route | N=10 | N=20 | N=40 | N=80 |
|---|---|---|---|---|
| spdal-cold | 0.80 | 1.93 | 4.49 | 11.23 |
| spdal-warm | 0.67 | 1.96 | 5.65 | 14.87 |
| spdal-shift | 0.42 | 0.98 | 2.44 | 5.24 |
| das-warm | 2.04 | 15.29 | 129.90 | - |
| proxqp-warm | 7.02 (46 inf) | 46.67 (72 inf) | - | - |
| piqp-cold | 0.34 | 0.72 | 1.60 | 3.34 |


## 5. Where the sparse backend's time goes, and the stage-wise question

`SimplicialLDLT` on the augmented pattern of the three scenarios, all
rows active, one P-core (analyze once, then factorize / solve):

| scenario, stage size | N | KKT size | nnz(L) | factorize | solve | GFlop/s |
|---|---|---|---|---|---|---|
| quadrotor, 16 | 20 / 40 / 80 / 160 | 975 / 1915 / 3795 / 7555 | 3.9k / 8.3k / 16.9k / 34k | 28 / 73 / 152 / 344 us | 8.5 / 19 / 49 / 93 us | 1.3-1.4 |
| manipulator, 21 | 20 / 40 / 80 / 160 | 1596 / 3136 / 6216 / 12376 | 2.3k / 4.5k / 9.0k / 18k | 29 / 60 / 116 / 246 us | 14 / 33 / 70 / 162 us | 0.13 |
| quadruped, 24 | 20 / 40 / 80 / 160 | 1464 / 2904 / 5784 / 11544 | 5.6k / 11.6k / 23.9k / 48k | 61 / 121 / 271 / 549 us | 15 / 35 / 70 / 147 us | 0.9-1.0 |

Everything is linear in the horizon and the fill is tiny (`nnz(L)` is 1.5
to 4.5x the KKT size): with AMD the factor is effectively block-banded
already, and a factorization costs about four solves. A cold quadrotor
tick at `N = 40` (3.1 ms, 19 Newton steps, 5.4 factorizations) therefore
spends roughly 13% factorizing, ~45% in the 2-4 solves per Newton step
(the refinement check, the Woodbury columns) and the rest in products,
residuals and the line search over `p` rows. A stage-wise (Riccati /
block-tridiagonal) factorization would not reduce the flop count much;
its gain would be the dense-block efficiency an unblocked simplicial code
lacks (the 0.13 GFlop/s of the manipulator's diagonal-block structure is
the extreme case), which is what PIQP's `sparse_multistage` and HPIPM
buy with BLASFEO. The realistic ceiling from that route is 1.5x per tick
here; reducing Newton steps (Section 6) was worth more, and the per-row
loops (classification, line-search breakpoints, residuals) are the next
target for a robot-sized implementation.

Two cheaper levers were measured. `mu_eq_init = 1e-5` instead of `1e-3`
cuts the cold solve of the quadrotor from 15.4 to 11.3 Newton steps
(4.4 -> 3.5 factorizations) and of the quadruped from 15.5 to 7.1 (3.8 ->
2.5), the manipulator 17.0 -> 16.3; `1e-7` is marginally better again
(10.5 / 6.7) and was not taken. On the OSQP Control class it changes
nothing (the BCL rounds there are gated by `eta_ext`, not by `mu_eq`,
and cost one cheap Newton step each). The Woodbury budget is discussed in
Section 3.

## 6. Warm starting and failure modes specific to the sparse / MPC setting

**The unshifted warm start is no better than cold.** In single-timestep
control consecutive QPs differ by a small drift and the previous solution
is a good iterate. In MPC the next solution is approximately the previous
one *shifted by one stage*; the unshifted iterate is off by one stage
along the whole horizon. On the test's random MPC chain (40 ticks through
a box conflict) the plain warm start needs 238 iterations against 175
cold and 167 shifted (`mpc_gen::ShiftWarmStart`, duplicating the last
stage). On the robot scenarios the picture is more nuanced: with kept
penalties the shift helps the quadruped (schedule changes every tick) and
hurts the quadrotor, because the shifted duals re-identify the active set
(flips = Woodbury work or a refactorization) where the unshifted ones
were continuous. The off-by-a-stage error grows with the horizon: on the
manipulator at `N = 80` the plain warm start needs 29.7 Newton steps and
8.1 factorizations per tick, no better than cold (28.6 / 8.3), while the
shifted one needs 10.3 / 1.5 (3.3 ms against 10.1 ms per tick).

**Keeping the penalties across ticks is the big lever, with a guard.** The
dense backend resets `mu_eq`, `mu_in` to their initial values on every
warm solve (ProxQP's behaviour). For the sparse backend that means the
cached factor is never valid at the first Newton step of a tick, and each
tick pays at least two factorizations (the reset and the first BCL
shrink). Keeping the previous penalties (`warm_keep_mu`) makes a quiet
tick reuse its factor: quadrotor `N = 20`, 266 -> 99 us mean, 9.2 -> 3.4
Newton steps, 2.0 -> 0.1 factorizations per tick. The failure mode this
creates is a **penalty cascade**: a tick that struggles (the quadruped at
a phase change) drives `mu` to its floor (`mu_in = 1e-8`, `mu_eq =
1e-9`), the next tick inherits floor-level penalties, and at that
stiffness the first inner loop re-identifies the active set one row per
Newton step (27 steps and 6 factorizations at `bcl 0` in the trace), ends
at the floor again, and so on for hundreds of ticks (100-300 Newton steps
per tick on both quadruped formulations). Flooring the kept `mu` at a
threshold does not help: the quadrotor and manipulator sit at the floor on
145 and 174 of 200 ticks and solve there in 5 and 8 steps, so any
threshold that breaks the cascade also destroys their factor reuse
(quadrotor 4.5 -> 11.9 steps). What separates the two is the *previous
solve's cost*: `warm_keep_mu_max_iters = 25` (reset the penalties after a
tick that needed more than 25 inner iterations) keeps quadrotor 4.5 /
manipulator 8.9 and brings the quadruped to 11.7 (shift 5.8) and
quadruped-ineq to 26 (shift 12.7); 10 is too eager (quadrotor 11.2), 50
lets the cascade back in (quadruped-ineq 39). The second guard is the
factorization itself: when a matrix update has invalidated the cached
factor (the quadruped's `B_k` change every tick) there is nothing to
reuse and the inherited deep penalties only cost inner iterations
(quadruped warm 9.3 Newton steps kept vs 8.4 reset, shift 5.7 vs 4.7), so
the penalties are kept only while the factor is (`matrix_dirty_` resets
them).

**Degenerate two-sided rows from a contact schedule chatter.** The
textbook convex-MPC schedule pins a swing foot's `f_z` through `0 <= f_z
<= 0` on the bound rows, with the pyramid rows then at `0 <= 0`. For the
three-state classification these are pairs of opposite rows with a
non-unique dual split; the estimate `z~` flips them between states at
every Newton step. Cold solves take 22 Newton steps and 12 factorizations
per tick (quadruped-ineq) against 15 / 4 with the schedule expressed as
hard equalities with a fixed pattern (`Eu_k u_k = 0` with all-zero rows
for stance feet, `mpc_gen::SetStageEqU`; the certificate strips all-zero
rows before its rank test); with warm starts the difference is 26 vs 12
Newton steps, and the shifted warm start 12.7 vs 5.8. The dense DAS
suffers the same way (36 steps per tick on the ineq form). PIQP does
not care (9-12 iterations on both), and sparse ProxQP declares 24 of 60
feasible ticks infeasible on the ineq form. Recommendation: express
schedule pins as equalities; the sparse backend's fixed pattern with
explicit zeros makes that free.

**The elastic BCL changes transfer, and matter little here.** Ablation on
the four scenarios at `N = 20` (200 ticks, iterations per tick, cold /
warm / shift; `results/mpc/ablation/`):

| setting | quadrotor | manipulator | quadruped | quadruped-ineq |
|---|---|---|---|---|
| default | 11.2 / 4.4 / 6.3 | 16.3 / 8.7 / 10.1 | 7.0 / 8.6 / 4.7 | 16.8 / 17.2 / 7.6 |
| bcl-proxqp | 12.1 / 4.7 / 6.9 | 17.7 / 9.1 / 10.3 | 7.0 / 8.5 / 4.7 | 16.8 / 17.2 / 7.2 |
| no-bcl-split | 12.2 / 4.6 / 6.4 | 17.6 / 9.0 / 10.7 | 7.1 / 8.4 / 4.7 | 16.8 / 17.2 / 7.2 |
| no-sat-jump | 11.6 / 4.3 / 6.5 | 16.8 / 8.7 / 9.8 | 7.1 / 8.6 / 4.7 | 16.9 / 17.2 / 7.1 |
| no-release-jump | 11.2 / 4.5 / 6.7 | 16.1 / 8.7 / 10.1 | 7.1 / 8.8 / 4.6 | 16.8 / 17.2 / 7.5 |
| cold-reset | 11.2 / 4.4 / 6.3 | 16.3 / 8.7 / 10.4 | 7.0 / 8.6 / 4.7 | 16.8 / 17.2 / 7.6 |
| no-warm-eta | 11.2 / 4.4 / 6.3 | 16.3 / 8.8 / 10.1 | 7.0 / 8.6 / 4.7 | 17.0 / 17.3 / 7.5 |
| mu-eq-init-1e-3 | 15.4 / 4.6 / 6.4 | 17.0 / 8.8 / 9.8 | 15.3 / 15.2 / 11.6 | 21.6 / 21.9 / 13.0 |
| keep-mu-always | - / 4.4 / 6.3 | - / 8.8 / 10.8 | - / 8.6 / 4.7 | - / 17.2 / 7.6 |

Nothing fails under any setting, unlike the dense conflict studies: these
scenarios' conflicts are resolved through the dynamics block (large
equality residuals dominate `r_prim`), and the inner active-set
identification, not the outer loop, is the expensive part. ProxQP's
unmodified outer loop costs 5-20% more iterations; the cold reset by
itself only adds iteration spikes (max 88 instead of 36 on the quadrotor
warm start). The saturation and release jumps are neutral to slightly
positive.

**Hard solvers on conflict ticks.** Sparse PIQP hits its 250-iteration cap
on 50-52 of the quadrotor's 92 conflict ticks (its infeasibility
certificate fires on only 5-6) and on every conflict tick of the
manipulator (42-45); sparse ProxQP declares infeasibility on 57-65 (cold /
warm) of the quadrotor's conflict ticks and, warm-started, on 107-198 of
the manipulator's 200 ticks (more than are actually infeasible), with
single ticks of up to 4.9 s (840 iterations per tick on average at
`N = 10`). On the degenerate quadruped-ineq form it also declares 46-72
feasible ticks infeasible. The elastic routes converge on every tick of
every scenario and horizon and their worst elastic KKT residual stays at
1e-6 to 1e-7. PIQP is the fastest route on the quadruped-ineq form
(0.34-3.3 ms against the sparse PDAL's shifted 0.42-5.2 ms): an IPM does
not care about degenerate rows, the active-set-like PDAL does.

## 7. Summary and recommendations

- On the OSQP Control class and the larger Control ladder the sparse
  elastic PDAL is at parity with sparse PIQP (0.55-1.1x its time) and
  2-3x faster than sparse ProxQP, all solves certified; the dense PDAL is
  5-10x slower at n ~ 1600 and out of reach beyond.
- On closed-loop robot MPC the sparse backend is the fastest route at
  every horizon on the quadrotor (kept penalties: 0.06-1.9 ms per tick,
  DAS 0.06-23 ms, sparse ProxQP 0.3-37 ms), the manipulator (shifted warm
  start beyond `N = 20`) and, with the equality-pinned schedule, the
  quadruped (shifted, 0.24-2.2 ms against sparse PIQP's 0.48-4.5 ms); the
  hard solvers fail or go infeasible on 50-100% of the conflict ticks.
  With the textbook degenerate schedule, sparse PIQP is 1.2-1.6x faster.
- The dense backend's rank-one flip updates, LLT, and reset-every-tick
  warm start do not carry over; the Woodbury correction, static-ordering
  LDL', small `mu_eq_init` and the guarded `warm_keep_mu` are the sparse
  replacements. Every one of them is a setting on `pdal::Settings`, so the
  dense backend can be run the same way for comparison.
- Open items: a blocked (BLASFEO-style) stage-wise factorization for the
  last 1.5x; per-row loop vectorization; an explicit MPC interface (stage
  data in, shifted warm start built in); `relax()` and bindings for the
  sparse backend; the sparse elastic IPM (PIQP's sparse_multistage is the
  bar) for the classes where the PDAL is iteration-bound.
