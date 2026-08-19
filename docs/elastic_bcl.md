# BCL in the elastic setting: a departure from ProxQP

*Why ProxQP's outer-loop safeguards mistranslate under the elastic
formulation, what failed, and the block-split BCL that follows.*

## ProxQP's BCL rationale

ProxQP's outer loop follows the classic BCL (LANCELOT) recipe: after
each proximal subproblem, compare the primal constraint violation
`p_k` against a tolerance `eta_bcl`. If `p_k <= eta_bcl`, the step is
"good" — keep the implicit multiplier update and tighten the
tolerances. Otherwise the step is "bad" — **revert the multipliers**
(duals estimated at an infeasible point are untrustworthy) and
**shrink mu** so the heavier penalization *enforces feasibility*. The
convergence argument needs one of two things to happen: the
multipliers converge (good steps), or the penalty grows without bound
and forces feasibility regardless (bad steps). A last-resort cold
reset of over-tightened mu backs this up on problems that may be
genuinely infeasible. All of this machinery is, at its core,
**infeasibility-fighting medicine**.

## What "primal violation" means for elastic rows

ElastiQP's elastic problem is feasible by construction: every
inequality row carries a slack `t >= 0` that absorbs any violation at
cost `penalty * t`. The solver's inequality-block "primal residual" is
the violation beyond the reconstructed slack, `r - t` with
`t = max(r + mu_in (z - w), 0)` — and this quantity does **not**
measure distance to feasibility. It measures **multiplier lag**: a row
whose data step moved it toward elastic saturation (true `z* = w`,
violation `r > 0`) reconstructs `t = 0` for as long as
`mu_in (w - z) > r`, and reports a "violation" of `r` that no amount
of penalization can remove. Only dual progress can — the PMM update
moves `z` toward its box target by `~r/mu_in` per outer round, so
`1/mu_in` is effectively the **dual step size**, not a penalty
weight.

The two framings prescribe the same *mu direction* (shrink), which is
why the inherited schedule mostly works. They disagree about the
multipliers, and that is where it breaks:

* The bad-step **z revert** discards the one kind of progress that can
  clear an inequality-block residual. The result, on warm-started
  solves whose drift flips weakly-active rows toward saturation, is a
  creep-and-revert cycle: each round takes one dual step of `r/mu`,
  the bad-step branch throws it away, and net progress survives only
  through the "good" rounds.
* The **cold reset** compounds the category error: "penalty at the
  floor and still infeasible" cannot mean infeasibility here — it can
  only mean the dual step size finally got large enough to work. The
  reset destroys it and re-slows the dual by five orders of magnitude,
  producing a limit cycle that runs the outer loop to `kMaxIter`
  (measured: 13–22% of ticks on small-drift, high-penalty warm-started
  trajectories, at *any* tolerance). Instrumented across 720 random
  and 144 Maros–Meszaros cold solves, the reset never fired once — in
  the elastic setting it fires **only** where it is harmful, and it is
  now disabled by default (`Settings::cold_reset_limit = 0`).

## The exception: equalities are still hard

ElastiQP does not elasticize the equality block `Ax = b` (in robot
control these are dynamics constraints, feasible by construction —
though a user *can* supply inconsistent equalities). For that block
the equality residual is a true feasibility measure and the classic
BCL logic is doing its original, legitimate job.

## Block-split BCL

The residual the BCL thresholds see is `max(in_res, eq_res)`, treated
uniformly. The elastic-native correction splits only the part that is
genuinely about trust in the multipliers — the bad-step **revert**:

* **Equality block** (hard): revert `y` on a bad step, but only when
  `eq_res` is itself the offender — the classical "duals estimated at
  an infeasible point are untrustworthy" safeguard, applied where
  infeasibility is actually possible.
* **Inequality block** (elastic): never revert `z`; its movement is
  always progress toward the box target.
* **Both mu still shrink in lockstep** on a bad step, reinterpreted as
  dual step-size increases (`1/mu` is each block's dual step length).
  An earlier variant that gated the `mu_eq` shrink on `eq_res`
  measurably slowed equality-dual convergence on cold infeasible+eq
  problems (+14–18% iterations) — the shrink is doing dual-acceleration
  work for the equality block too, independent of the feasibility
  framing.

Measured (`Settings::bcl_split`, now the default; set false for
ProxQP-parity outer-loop behavior): the warm-start saturation-creep
cells improve ~5–8% mean / 10–15% max iterations on top of the reset
removal; cold infeasible solves improve 6–9%; large-drift
high-penalty cells improve up to 18%; feasible/degenerate cells are
unchanged (±3%); genuinely inconsistent equalities fail identically
to classic BCL (same status, same irreducible residual, comparable
iterations — though at the default `eps_rel = 0` they are now caught
before the loop by the ingestion-time consistency certificate,
`Settings::check_eq_consistency`, and return `kInfeasible` without
spending iterations). The full test suite (including the pinned
finite-difference vs VJP cross-validation) and every benchmark
(random-QP families, robot control, differentiability, collision,
warm-start grids) pass unchanged with the split enabled.

A further split of the *thresholds* was evaluated and rejected
(2026-08-15). Per-block `eta_eq`/`eta_ineq` ladders — each block
classified against its own threshold, tightened on its own good rounds,
reset on its own bad rounds, with the y-revert gated on the equality
ladder — were benchmarked in two keyings over the full warm drift grid
(structure x penalty x sigma), cold families, and the robot replay:

* `eta_eq` keyed to `mu_eq`: wins 5–18% at small drift (sigma 1e-4)
  but loses 8–15% at large drift and on cold high-penalty families —
  the aggressive equality tightening (`mu_eq^0.9` per good round)
  drives both effects.
* both etas keyed to `mu_in` (pure per-block classification): the
  effects shrink toward neutral, net slightly negative (+2–6% on most
  warm cells, small wins only at sigma 1e-4).

Conclusion: the shared worst-block ladder is load-bearing, not an
inherited artifact — an inequality-lag "bad" round's mu shrink is
useful *global* dual acceleration, and letting each block keep its own
tightening schedule mostly de-synchronizes the ladder from the mu
schedule that actually governs progress. The threshold split is
rejected; only the *revert* is split (above).

## Creep-resolving mu jump

The second adopted departure (2026-08-15, `Settings::bcl_mu_jump`,
default on) cashes in the dual-step reinterpretation directly. Classic
BCL must shrink mu blindly — a fixed `mu_update_factor` per bad round —
because for a hard QP no target depth exists. In the elastic reading the
target is explicit: a row lagging toward saturation with violation `r`
and dual gap `w - z` snaps once `mu_in <= r / (w - z)`, all known at the
iterate. On an inequality-driven bad step whose residual is *stalled*
(improved by less than 20% this round — a working ladder shrinks it ~10x,
so slow improvement is the creep signature), the solver jumps `mu_in` to
the *shallowest* target among the rows that are individually above
tolerance and individually stalled (same 20% margin per row), instead of
paying one inner solve plus refactorization per 10x rung. The classic
shrink is the ceiling, `mu_min_in` the floor, and `mu_eq` follows by the
same ratio.

The shallowest-first rule is a 2026-08-18 refinement driven by mixed
per-row penalties. The original variant jumped to the single
worst-residual row's target. With mixed penalties (measured on
1-stiff-in-8 `w = {10, 1e4}` mixes, the `spike` cells of
`bench_bcl_strategies`) a stiff row far from saturation can win that
argmax while the soft majority is still settling; its target prices the
travel to saturation — 3-5 decades of mu — for a row that will end
interior, and the whole problem over-tightens: +40-60% iterations and
~2x worst-tick versus the plain split ladder, i.e. the jump was WORSE
than its own fallback there. Two candidate fixes were measured. A
per-row stall gate alone (jump only off a row whose own residual
stalled) did not help — the stiff argmax rows in the mixed cells are
genuinely stalled too, they are just converging to an interior dual,
which is indistinguishable from creep at the iterate. Selecting the
shallowest stalled target fixes it structurally: depth is self-pacing
(rows resolve shallowest-first and leave the candidate set; a genuinely
creeping row gets its full depth on the next round), the `spike`
regression disappears (−18 to −38%), the uniform-penalty creep wins are
retained, and across the full strategy grid the jump is never worse
than the split ladder beyond noise (worst cell +2.1%, sub-iteration).
On the `bench_fwd_warm` grid the refinement is net +0.5%: cost-drift
creep cells improve up to −29%, mid-drift (sigma 1e-2) infeasible cells
pay +5-7%.

Measured (5-seed warm drift grid, cold families, robot replay, eps 1e-5
and 1e-8): −7 to −20% iterations on every high-penalty (1e4) warm cell
and on all sigma=1e-4 creep cells; ~0% elsewhere warm; ≤ +4% on cold
feasible solves (sub-iteration per solve); robot sequences unchanged; no
failures introduced. Rejected variants, all measured worse: resolving
*all* lagging rows per jump (+20-46% nearly everywhere — too deep too
early), an ungated jump (+3-6% at low-penalty mid drift), and shrinking
`mu_eq` by only the classic factor while `mu_in` jumps (+25-58% on
high-penalty warm cells — the lockstep is essential, re-confirming the
gating result above from the other direction).

## Warm-start eta seeding

The third departure (`Settings::bcl_warm_eta`, default on) treats the
other half of the warm-tick replay. A warm solve resets mu to the inits
*and* restarts the BCL ladder at `eta_ext_init ~ 0.79` — but the iterate
is near-converged, so a small-drift tick spends several "good" rounds
merely tightening eta down to its actual residual before any mu action
starts. With the seed, a warm start sets
`eta_ext = min(eta_ext_init, 0.5 * pri0)` — but only when that skips at
least a decade of ladder: a marginal seed at moderate drift forces bad
steps while `x` is still far from the new optimum, over-tightening mu
early (measured +5-15% warm on badly row-scaled tight-eps cells without
the guard). Measured on top of the jump: −9 to −14% on warm
saturation-creep cells, −1 to −4% broadly, ~0 at large drift, cold
untouched.

A fixed deeper warm `mu_in_init` was evaluated for the same regime and
rejected: −22-28% on creep cells but +30-120% at large drift and
+20-83% cold. The right depth is drift-dependent — which is exactly what
the seeded eta plus the stall-gated jump discover per tick, at the cost
of one probing round.

## Validation

`benchmarks/bench_bcl_strategies.cc` isolates the failure regime that
motivated all of the above (penalty 1e4, sigma 1e-4/1e-3 qh drift, all
three structures — first seen as forward warm-solve failures in
`bench_relax_warm` at commit b995082) and replays it under each
strategy generation via the settings flags, on the exact original
trajectories (same generator, same seeds). Measured ladder on the
worst cell (degen n=14 p=100, sigma 1e-4, 100 warm ticks):

| generation | fails | reset ticks | mean it | worst tick |
|---|---|---|---|---|
| proxqp parity (uncapped reset) | 19 | 41 | 101.5 | 269 |
| reset capped at 1 | 0 | 47 | 67.8 | 149 |
| reset off | 0 | 0 | 38.5 | 77 |
| + block split | 0 | 0 | 35.6 | 73 |
| + mu jump | 0 | 0 | 26.2 | 52 |
| + eta seed (shipped) | 0 | 0 | 25.0 | 52 |

The fail counts under proxqp parity reproduce b995082 exactly (19
degen, 18 feas), and are identical at eps 1e-5 and 1e-8: the limit
cycle fires above `cold_reset_residual`, so the production tolerance
is equally exposed. In the control cells (penalty 10; sigma 1e-1) the
reset never fires and the three reset variants are identical, while
the elastic departures stay neutral-to-better.

The benchmark also carries a mixed-penalty group (per-row
`w = {10, 1e4}`: `alt` 50/50, `spike` 1 stiff in 8, `dip` 1 soft in 8),
added 2026-08-18 for warm control loops with heterogeneous penalties.
Two findings. First, the 50/50 mix is the one regime where the classic
BCL fails even with the reset off (1 tick to kMaxIter on the feas
sigma-1e-4 cell) — the block split itself, not just the reset removal,
is load-bearing there. Second, the `spike` mix exposed the
worst-residual-row jump regression fixed by the shallowest-first rule
(see the mu-jump section). Ladder on the worst mixed cell
(degen `spike`, sigma 1e-4, mean it / worst tick): split 34.1 / 76,
worst-residual jump 46.4 / 99, shallowest-first shipped 33.2 / 80.

The regression test `elastiqp.bcl_creep` (tests/test_bcl_creep.cc) pins
the shipped behavior on the two worst uniform cells plus the `spike`
mixed cell, with bounds that discriminate every rejected generation
(proxqp parity, capped reset, worst-residual-row jump).
