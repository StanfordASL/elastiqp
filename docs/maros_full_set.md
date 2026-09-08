# Maros-Meszaros beyond n <= 200: full dense-tractable set (2026-09-07)

The paper's Table III uses the 21-problem robotics-scale subset and the
exploratory tables the 36-problem n <= 200 subset. This note runs every
Maros-Meszaros problem a dense solver can hold in memory -- the 107 problems
with n < 10000 (n up to 5427; the 33 with n >= 10000 need 2 to 278 GB dense
and are out of reach for every dense solver here) -- against the same
external solvers, and uses the failures to find defects in the ElastiQP
backends. Robot control (n <= ~60) remains the design point; nothing below
changes small-scale behavior (checked, see "Small-scale parity").

Harness: `python/run_maros_meszaros_full.py` (elastiqp_benchmarks) drives
`bench_maros_meszaros --problem <one> --solvers <one route>` as one process
per (problem, solver) under a 600 s wall-clock limit (1800 s for the PIQP
1e-9 reference, which is computed once per problem and cached with
`--ref-cache` so every solver -- the qpax routes of `run_maros_meszaros.py`
included -- sees the identical penalty). Two streams run in parallel, C++
pinned to one core and qpax to another. eps_abs = 1e-6, eps_rel = 0,
penalty = 10x the largest reference dual, cold starts, Ruiz on for the
ElastiQP routes. `ok` = reported success AND hard violation <= 1e-4 AND
relative objective error <= 1e-4 (when the reference certified). Results:
`results/maros_meszaros_all_{results,summary,profile,driver_stdout}_n6000_eps1e-6.*`.

## Headline (before the DAS fixes below)

107 problems, shifted geometric mean (shift 1 ms, failures at the slowest
observed time):

| solver | solved | sgm | vs best | timeouts |
|---|---|---|---|---|
| DAQP | 98 | 0.12 s | 1.0x | 0 |
| PIQP | 104 | 0.17 s | 1.5x | 0 |
| ElastiQP-PDAL | 101 | 0.19 s | 1.6x | 0 |
| ProxQP | 93 | 0.34 s | 2.9x | 3 |
| ElastiQP-DAS | 86 | 0.37 s | 3.1x | 4 |
| ElastiQP-IPM | 96 | 0.40 s | 3.4x | 0 |
| qpax-elastic | 51 | 6.2 s | 53x | 21 |
| qpax-hard | 37 | 27 s | 232x | 5 |

By size tier (solved / problems):

| tier | # | DAQP | PIQP | DAS | PDAL | IPM | ProxQP | qpax-el | qpax-hard |
|---|---|---|---|---|---|---|---|---|---|
| n <= 200 | 36 | 36 | 36 | 36 | 36 | 35 | 35 | 29 | 17 |
| 200 < n <= 1000 | 37 | 34 | 37 | 34 | 36 | 32 | 32 | 13 | 8 |
| 1000 < n <= 2500 | 21 | 16 | 18 | 8 | 16 | 16 | 15 | 7 | 7 |
| 2500 < n <= 5427 | 13 | 12 | 13 | 8 | 13 | 13 | 11 | 2 | 5 |

Four references did not certify at 1e-9 within 1000 PIQP iterations
(QGFRDXPN, QSHELL, QPILOTNO, STADAT1); those rows are judged on
feasibility only.

## Pairwise: where one solver of a pair fails and the other succeeds

ElastiQP-DAS failed, DAQP ok (15): QBORE3D, QSCTAP1, QSCSD1, QSTANDAT,
QSCSD6, QSHIP04S, QSHELL, QSCTAP2, QSHIP04L, QSHIP08S, QSCTAP3, QSCSD8,
AUG3DQP, QSHIP08L (timeout), QSHIP12L (timeout). All but QSHELL end in
`Status::kNumerics` and return x = 0; every one is LP-like (rank(P) far
below n, e.g. QSTANDAT 138/1075) and goes through the proximal loop.
QSHELL returns `kInfeasible` at 0 iterations.

DAQP failed, ElastiQP-DAS ok (3): QCAPRI, QFORPLAN, QETAMACR (DAQP
`not_converged` after 600-1400 iterations, violation 0.2-44).

Both failed (6): Q25FV47, QGFRDXPN, QPILOTNO, QSIERRA, STADAT1, STADAT3.

ElastiQP-IPM failed, PIQP ok (8): QPCBOEI2, QCAPRI, QFORPLAN, QBANDM,
QFFFFF80, QSCFXM2, QSCFXM3, YAO -- every one at the 250-iteration cap.
PIQP failed, IPM ok: none. Both: QGFRDXPN, QPILOTNO, QSHELL.

ElastiQP-PDAL failed, ProxQP ok (2): QSHELL, QPILOTNO (both uncertified
references; PDAL's violation is 6e-11 and 9e-8, it is the dual/gap
tolerance at penalties of 9e7 and 1e9 it does not reach).
ProxQP failed, PDAL ok (10): QRECIPE, QCAPRI, QETAMACR, QFFFFF80,
QSCFXM2, QSCFXM3, STADAT2, QSIERRA, STADAT3 (timeout), QSHIP12L (timeout).
Both: QFORPLAN, QGFRDXPN, STADAT1, YAO.

## Mechanisms and fixes: ElastiQP-DAS

Diagnosed with a per-problem settings sweep (`das_diag`, scratch tool) and
the `ELASTIQP_DAS_DEBUG` trace. Three distinct defects, all in
robustness machinery, none in the active-set method itself.

### 1. Cycle guard with a relative progress tolerance (13 of the 15)

The guard declared a stall when the dual objective rose by less than
`progress_tol * max(1, |dual|)` with `progress_tol = 1e-14`. Under a small
proximal shift the dual carries an offset of order |R^-T q|^2 (1.5e10 on
QSTANDAT, 9.4e13 on QSHIP04S), so the threshold was 1e-4 to 1 while the
genuine per-iteration gain was 1e-5 (a few ulps). The trace shows the
working set growing by one row every iteration -- no cycle is possible --
while `stalled` counts to 10, the working set is refactored, counts to 10
again and the solve aborts as `kNumerics`. DAQP's test is absolute
(`fval - best_fval < 1e-14` on a quantity of the same magnitude), i.e.
effectively "bit-identical", which is why DAQP passes these problems.

Fix (das.hpp): absolute progress test (DAQP parity), and a stall
is counted only when a row left the working set since the previous check
(a cycle needs a removal). Solves QSTANDAT, QBORE3D, QSCSD1, QSCTAP1,
QSCSD6, AUG3DQP, QSCTAP2 with the default shift; `cycle_tol = 100` alone
did the same, confirming the diagnosis.

### 2. Degenerate vertex the guard cannot pass (QSHIP04S and the largest)

With the guard fixed, QSHIP04S still stops at iteration 1532 with 1454 of
1458 rows active: the dual is bit-stationary (jitter of +-10 ulps at 9e13)
across removals -- a genuinely degenerate vertex. `eps_prox = 1e-4`
instead of 1e-6 solves it directly (the larger shift changes the dual
geometry and breaks the ties); DAQP's LP path does the same adaptively
(`eps *= 10` when an inner solve stalls).

Fix: proximal escalation. When the inner method returns `kNumerics` inside
the proximal loop, the shift is raised x100 (up to `prox_escalations = 3`
times), R and the row images are refactored, the working set is rebuilt
from the current states and the loop continues from the last dual-feasible
point as the new center. Runs only on the failure path. QSHIP04S: solved,
1533 iterations, one escalation.

### 3. Equality-consistency certificate false positive (QSHELL, QSIERRA)

QSHELL has 784 equality rows of rank 783 (exactly dependent, consistent:
PIQP and DAQP solve it). The certificate compares the dropped row's
residual at the least-norm solution of the kept rows with eps_abs. That
least-norm solve runs in the M = C R^-1 metric, where the 1/sqrt(eps)
amplification of the null-space components of Q makes the kept rows'
Gram matrix ill-conditioned (pivots just above `sing_tol`), and its
roundoff alone is 1.2e-5 > 1e-6: `eq_infeasibility()` = 1.2e-5 at
eps = 1e-6, 1.0e-6 at 1e-4, 1.9e-9 at 1e-2. QGFRDXPN is the same effect
one step further: A has full rank (smallest singular value 0.04) yet rows
are dropped as dependent in the M metric, and DAQP fails it identically
(0 iterations, its immutable active set is singular).

Fix: the certificate also evaluates the KEPT rows at the least-norm point;
when their residual exceeds the tolerance the solve cannot distinguish
inconsistency from conditioning, `eq_infeasibility()` stays 0 and the solve
proceeds with the dependent rows dropped. QSHELL: solved (violation 1.5e-5
on the dropped row, the conditioning floor). The inconsistent-equality
tests still certify (test_ruiz.cc, test_das.cc).

### Not fixed: per-iteration cost at scale

DAS needs about the same number of active-set iterations as DAQP (QSCTAP2
2868 vs 2460, QSHIP04S 1532 vs 1604) but 10x the time per iteration at
n ~ 2000: `remove_row()` shifts the Gram matrix and recomputes the LDL'
from the removed position (O(k^3) worst case) where DAQP applies rank-one
downdates (O(k^2)). Irrelevant at n <= 60, the reason DAS times out on
QSHIP08L / QSHIP12L / STADAT3 and takes 480 s on QSCTAP2.

## Mechanisms: ElastiQP-IPM (diagnosed, not fixed)

All eight IPM-only failures hit the 250-iteration cap. Probing QPCBOEI2,
QBANDM and QSCFXM2 through the Python bindings: with Ruiz OFF the IPM
solves QPCBOEI2 in 42 iterations and QBANDM in 66; with Ruiz ON it stalls
on both (QSCFXM2 fails either way). The `ELASTIQP_IPM_DEBUG` trace shows
two frozen states, both PIQP-logic ported faithfully but interacting badly
with the scaled frame:

* QPCBOEI2: complementarity is driven to mu = 4e-16 with full steps while
  the primal residual sits at 31.7. The regularized subproblem is solved
  exactly, but the proximal center nu is only moved when the true primal
  residual drops 5% -- it never does -- and the "finetune" that lowers the
  regularization floor is gated on the prox-infeasibility measure
  `delta * |nu - z|`, which is 31.7 here (z is 2.4e10 in the scaled frame
  at penalty 1.3e9). Deadlock: PIQP would eventually declare primal
  infeasibility; the elastic problem never is.
* QBANDM: the dual residual sits at 1e-3 = rho * |x - xi| with rho pinned
  to 1e-8, because the condensed LLT failed repeatedly and each retry
  raised `reg_limit` (x10, capped at eps_abs). In the unscaled frame the
  factorization succeeds and the floor reaches 1e-13.

Both point at the same place: thresholds (`infeasibility_threshold`,
`reg_limit` handling, the LLT failure test) tuned for O(1) data applied
to Ruiz-scaled quantities with the exact-penalty magnitudes of this set
(1e5 to 1e9). Suggested follow-ups: force the proximal-center update when
the regularized subproblem is converged (mu below eps) regardless of
residual progress; compare the prox-infeasibility measures in the user
frame; and check the factor-failure path (Ruiz makes the scaled Q tiny
relative to rho).

## ElastiQP-PDAL vs ProxQP

PDAL fails only the two problems whose reference did not certify, and only
on the dual/gap tolerance at penalties of 9e7 and 1e9 (violation 6e-11 and
9e-8). ProxQP fails 10 that PDAL solves, most with violations of 1e3 to
2e8 after hitting its inner-iteration budget; no ElastiQP defect here.

## After the DAS fixes (same harness, DAS route rerun over all 107)

| solver | solved | sgm | vs best |
|---|---|---|---|
| DAQP | 98 | 0.12 s | 1.0x |
| ElastiQP-DAS | 100 (was 86) | 0.26 s (was 0.37) | 2.2x (was 3.1x) |
| PIQP | 104 | 0.17 s | 1.5x |
| ElastiQP-PDAL | 101 | 0.19 s | 1.6x |

DAS by tier: 36/36, 37/37 (was 34), 17/21 (was 8), 10/13 (was 8). The
remaining seven DAS failures are five 600 s timeouts (QSCTAP3, QSHIP08L,
QSHIP12L, STADAT1, STADAT3 -- the per-iteration cost item above; three of
them DAQP solves in 30-470 s) and the two both-fail problems Q25FV47 and
QGFRDXPN (QGFRDXPN now ends at violation 1e-10 but uncertified). DAS now
solves QSIERRA and QPILOTNO, which DAQP does not: DAS fails and DAQP
succeeds on 3 problems (all timeouts), DAQP fails and DAS succeeds on 5.

## Small-scale parity

`bench_robot_control` built against the pre-fix and post-fix header, run
back to back on one core, twice: identical iteration counts on all eight
sequences (cold and warm), timings within run-to-run noise (hum-wbc cold
70.3 vs 70.3 us, 75.0 vs 80.7 us in the two runs; the pre-fix pair itself
spans 70.3-75.0). None of the three changes touches a non-failure path.
All 11 repo tests pass.
