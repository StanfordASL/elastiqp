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


