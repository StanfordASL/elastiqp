# The unified log-barrier PDAL: implementation notes

*Notes on the 2026-08 rework that made the barrier smoothing of
`docs/log_barrier_pdal.tex` the mechanism of the **forward pass**, so that
the forward solve and the differentiation point are two stopping points of
one method. Companion to `docs/log_barrier_pdal.tex` (the math note, stated
for a standard hard QP) and successor to the relax()-only story in
`docs/pdal_differentiability.md`.*

## What changed and why

Previously the forward pass was an active-set PDAL (ProxQP-style
three-state semismooth Newton with an exact piecewise line search), and
differentiability was bolted on afterwards: `relax(kappa)` walked the
converged tight solution to the κ-central point with a separate smooth
Newton corrector. Two solvers, one story each.

Now **every inner subproblem of the forward pass is the smooth barrier
subproblem** of the note: the slack indicators are replaced by
`-κ Σ log s`, each slack/dual pair lives on the exact-complementarity
manifold through the retraction

```
z = b_κ(v),   s = b_κ(-v),   b_κ(v) = (v + sqrt(v² + 4κ))/2,   z⊙s = κ,
```

and the outer BCL loop anneals κ (`kappa_init → kappa_min`, one shrink per
good step, doubled when the inner Newton dispatches a round in ≤2 steps)
alongside the usual mu schedule — inexact path following with
BCL-controlled subproblem accuracy, exactly the outer-schedule option the
note's §5 prescribes. `relax(kappa)` is now literally the same machinery
held at a fixed κ (see "Unification" below). In the κ → 0 limit `b_κ`
becomes the orthant projection and the smooth solver degenerates into the
active-set method it replaced; the condensed Newton matrix converges to
the same `K = Q + ρI + (1/μ_eq)A'A + (1/μ_in)G_act'G_act`.

Termination and every reported quantity remain on the **tight** elastic
KKT with the hard-projected certificate reconstruction, so the external
contract (exact-zero slacks on feasible problems,
`penalty − z_t − z_ineq = 0` identically, warm-start semantics, the
FFI/JAX interface, `relax()` state isolation) is unchanged. The barrier
bias at `kappa_min = 1e-13` (duality gap ~ `2pκ`) sits far below any
practical tolerance.

## The elastic barrier subproblem

Stated in the elastic variables — this is the adaptation of the note's
hard-QP §2/§4 that the task called for. Each row carries two pairs:
`(s1, z1)` for `t ≥ 0` and `(s2, z2)` for `Gx − t ≤ h`. The inner
subproblem at centers `(xᵏ, tᵏ, yᵏ, z2ᵏ)` is the root system

```
F1 = Qx + q + A'y + G'z2 + ρ(x − xᵏ)
F2 = w − z1 − z2 + ρ(t − tᵏ)            (t-stationarity)
F3 = Ax − b + μ_eq(yᵏ − y)
F4 = s1 − t                              (hard: z_t is barrier-slaved)
F5 = Gx + s2 − h − t + μ_in(z2ᵏ − z2)
```

with both pairs on the κ-manifold. **The single most important modeling
decision:** only the row multiplier `z2` carries a dual proximal term.
Proxing `z1` as well (the "symmetric" choice) anchors `z1 ≈ w` through its
center and forces `z2 ≈ 0` via `F2` — the row then exerts no feasibility
pull and cold solves crawl (we measured round-0 primal residuals of 66
vs 0.18 for the tight solver before this was fixed). With `z1` slaved
(`s1 = t`, `z1 = κ/t` exactly), the κ → 0 limit of the row update is
*exactly* the tight solver's clamp `z = clamp(zᵏ + (Gx−h)/μ, 0, w)`, active
rows get the tight active weight `1/μ_in` (no spurious factor 2), and
saturated/inactive rows drop out of `K` as before.

Newton on this system condenses onto the usual n×n SPD shape. With
`B = b'_κ(v)`, `C = b'_κ(−v)`, per pair `D̂1 = C1`, `β1 = B1/C1 = z1/s1`,
`D̂2 = C2 + μ_in B2`, `β2 = B2/D̂2 ≤ 1/μ_in`:

```
E = ρ + β1 + β2,    Λ = β2(ρ + β1)/E,
K = Q + ρI + (1/μ_eq)A'A + G'diag(Λ)G,
```

the note's boundedness claim in elastic form: the proxed pair keeps
`β2 ≤ 1/μ_in` uniformly in κ, and the unbounded `β1 = z1/s1` only enters
through `Λ ≤ β2` and through `β1·F4` products whose `F4` is zero to
*relative* round-off (below).

## The closed-form row update (the note's §3, elastically)

The multiplier/slack blocks have a closed-form update for fixed `x` — the
elastic version of the note's smooth multiplier update, and the smooth
counterpart of the active-set solver's dual snap. Writing `d = s − μz`,
the proximal pair equation `F5` gives the paired retraction at the scaled
parameter `mk = μ_in κ`:

```
s2 = b_mk(d2),   μ_in z2 = b_mk(−d2),   d2 = (t − r) − μ_in z2ᵏ,   r = Gx − h,
```

(so `z2 s2 = κ` stays exact to round-off), and eliminating both pairs
leaves ONE strictly increasing scalar equation per row,

```
F2(t) = w − κ/t − z2(t) + ρ(t − tᵏ) = 0,   t > 0,
```

solved by bracketed Newton whose out-of-bracket fallback is the frozen-z2
quadratic model `ρt² + (w − z2 − ρtᵏ)t − κ = 0` (closed-form positive
root; it jumps between the `t ~ κ/w` inactive regime and the `t ~
violation` regime in one step). The stop tolerance is round-off-level
relative — a looser one leaves a systematic `tol·penalty` offset in `z2`
that floors the dual residual at penalty ~ 1e5.

This is what lets the coupled Newton absorb arbitrarily large PMM
multiplier updates: without it, a big dual update makes any line search
crawl across the retraction kinks one row at a time (the first
implementation did exactly that — 30–45 inner iterations per subproblem).

## Inner loop: Newton on a smooth convex reduced function

With the rows and the equality dual eliminated in closed form at every
evaluation point (`y*(x) = yᵏ + (Ax − b)/μ_eq` is linear), the subproblem
is an unconstrained smooth **convex** minimization in `x` alone; Danskin
gives `∇Φ = F1` reduced and `∇²Φ = K` (the condensed step *is* the reduced
Newton step, by the implicit function theorem through the same
elimination). Each iteration:

1. pair weights + `Λ`, factorization (cached, see below), condensed solve
   with **one step of iterative refinement** against the staged `K` —
   at `cond(K) ~ 1e10+` the plain LLT solve's `cond·ε` error otherwise
   becomes the dual-residual floor in weakly curved directions (e.g.
   barely regularized contact-force variables that appear in no
   inequality row);
2. an **exact line search** on the convex slice: the derivative
   `g(α) = Φ'(x + α dx)·dx` is strictly increasing, so its root is found
   by bracketed regula falsi with Wolfe-style acceptance
   (`|g(α)| ≤ 0.5|g(0)|`, usually a single `g(1)` evaluation), and α may
   exceed 1 — the smooth analogue of proxsuite's exact piecewise search.
   Trial evaluations re-solve the rows loosely (1e-9); `g(0)` is free.
   This matters: a merit line search on `‖F‖²` is NOT convex along the
   direction and gets trapped in the first per-row basin (measured
   α ~ 0.01–0.06 crawls);
3. accept, re-solve rows at full precision, measure.

A joint (x, t) slice (searching along `dt` too, one sqrt per row per
trial) was tried and is **wrong** for barrier variables: inactive rows sit
at `t ~ κ/w ≈ 1e-18`, so any negative `dt` puts the positivity wall at
α ≈ 0. The row-re-solve slice is the correct geometry.

## Certificate numerics: maps vs iterates

The barrier coordinates re-derive the duals from `x` at gain `1/μ`
(`y = yᵏ + (Ax−b)/μ_eq`, `z` through the row solve). Re-derived ("map")
duals re-inject primal evaluation round-off amplified by `1/μ` at every
measurement — an additive dual **iterate**, as in ordinary PDAL, does not.
Three mechanisms restore iterate behavior where it matters:

- **y is additive**: `y += α·dy` with `dy = (A dx + F3)/μ_eq` (identical to
  the closed form in exact arithmetic when y starts consistent, but
  accumulates instead of re-deriving). The μ floors were raised to 1e-6
  (from proxsuite's 1e-8/1e-9) because machine-ε/μ must stay below
  eps_abs for map-based evaluations.
- **the certificate z is the linearized dual update** `zc = z2(pre-step)
  + α·B2∘dv2`: the Newton step zeroes the *measured* F1 — including
  whatever noise the row-map z carried, since that noise was part of the
  residual the step corrected. Bad-step reverts restore the round-start
  certificate snapshot, never a re-derived map value.
- **decided rows snap to their exact bound** in the certificate: a
  saturated dual left ε below `penalty` becomes a penalty-scaled
  duality-gap bias. "Decided" is read off the barrier itself —
  tight-saturated ⟺ `z1 = κ/t` vanishes as κ → 0 (slack bounded away
  from 0), threshold `min(κ_s·1e6, 1e-10(1+w))`; degenerate ties keep
  `z1, z2 = O(1)` and are robustly excluded (snapping a tie stalls
  convergence — found the hard way on the penalty-equals-dual test
  instance).

Two BCL endgame guards complete the picture: `eta_ext` is floored at
`0.1·eps_abs` (otherwise it decays to denormals, every endgame step reads
"bad", and μ collapses to the floor exactly when the last digits are being
ground out), and a **dual push** shrinks μ *without* the bad-step revert
when the primal is at tolerance but the dual residual has stalled clearly
above it (the dual prox contracts like `σ·μ` per round; a weakly curved
dual direction can otherwise freeze — observed frozen at 6.12e-7 on a
humanoid WBC instance with 1e-6-regularized contact forces).

## Factorization cache

`Λ` varies continuously, so the old active-set cache key is replaced by a
per-row weight fingerprint: reuse the factor while

```
|ΔΛ_i|·‖G_i‖² ≤ kkt_cache_tol · (Λ_i‖G_i‖² + min diag K)   for all i
```

(relative freeze per row, absolute allowance anchored to the smallest
curvature — anchoring to the largest lets mid-weight rows drift by
amounts that are large against weakly curved directions). Steps on a
reused factor are inexact-Newton steps safeguarded by the line search.
As κ → 0 decided rows' weights freeze, so settled warm re-solves skip
factorizations like an active-set cache; annealing rounds refactor every
step, which is the main structural cost vs the old solver.

## Unification with relax()

`relax(kappa)` now shares the machinery rather than duplicating it:
`barrier_kkt_fill` (residuals F1–F5, prox terms optional),
`condense_weights` (E⁻¹, Λ), and `barrier_direction` (the condensed
Newton solve) serve both. relax() is the `μ_in = 0`, centers-at-the-
iterate instance: its pair weights are `β = z/s` (qpax's elastic weights,
unbounded near the boundary — which is why Ruiz matters for large
penalties there), the forward pass's are the μ-bounded ones. Its iterate,
tolerance semantics, and state isolation are unchanged, and it still
lands on the identical κ-central point as the IPM reference (cross-checked
to ~1e-10 by `tests/test_pdal.cc`). The backward pass (`_kkt_bwd` /
`kkt_vjp.hpp`) is untouched. So the story is now: **one barrier method;
solve() follows the central path to `kappa_min`, relax() holds it at the
differentiation target.**

## Settings added

```
kappa_init = 1e-2          # cold-start smoothing level
kappa_update_factor = 1e-2 # shrink per good BCL step (squared when the
                           # round took <= 2 Newton steps)
kappa_min = 1e-13          # forward-pass endpoint; bias ~ 2 p kappa
kappa_warm_scale = 1e-1    # warm start kappa = scale * entry residual,
                           # clamped to [kappa_min, kappa_init];
                           # 0 => control-loop mode (see below)
kkt_cache_tol = 1e-9       # factorization-reuse fingerprint tolerance
mu_min_eq = mu_min_in = 1e-6  # raised floors (map-noise bound, see above)
```

## Performance vs the previous (active-set) solver

Same machine, same benchmarks, library defaults. `bench_random_qp`
(cold solves, medians):

| family, n/m/p          | tight: µs / iters | barrier: µs / iters |
|---|---|---|
| feasible 10/0/20       | 19 / 14   | 43 / 15   |
| feasible 100/0/200     | 3304 / 32 | 8705 / 35 |
| feasible+eq 100/20/200 | 2274 / 24 | 6073 / 27 |
| infeasible 10/0/20     | 23 / 20   | 107 / 20  |
| infeasible 100/0/200   | 5727 / 66 | 15723 / 62 |
| infeasible+eq 100/20/200 | 3893 / 45 | 11421 / 46 |

`bench_robot_control` (eps 1e-6, warm ticks):

| sequence | tight warm: µs / iters | barrier warm: µs / iters |
|---|---|---|
| diff-ik  | 1.3 / 1.0  | 8.4 / 2.0 |
| arm-osc  | 1.2 / 1.0  | 6.6 / 2.1 |
| hum-wbc  | 29 / 2.1   | 874 / 14.7 (defaults) — **107 / 2.1** with `kappa_warm_scale = 0` |

`relax()` and the KKT VJP timings are unchanged (the differentiation
increment is identical).

Reading of the numbers:

- **Iteration counts match or beat the active-set solver** across the
  board (the smoothing + annealing globalizes well; badly scaled cold
  problems improve outright: the Ruiz-drift stress test's cold solves
  dropped from ~116 to ~35 iterations/tick).
- **Per-iteration cost is higher, ~2.5–5x wall time on cold solves.** Two
  causes, in order: the per-row scalar solves (row updates dominate the
  profile, ~55–65% at p=60 — every line-search trial re-solves all rows,
  ~20ns/row-evaluation but called ~6x per Newton step), and
  refactorization every annealing step (Λ moves; the active-set cache hit
  more often mid-solve).
- **Warm control loops**: with defaults the κ anneal costs drifted warm
  starts several rounds. The entry-residual-matched warm κ cannot
  distinguish "large-scale problem, stable active set" (wants κ_min;
  humanoid WBC) from "ill-scaled problem, structural drift" (wants
  κ_init; the Ruiz drift test) using any single residual scalar we
  tried — absolute, relative, and power-law maps each favor one and
  regress the other. The shipped default is the conservative one;
  `kappa_warm_scale = 0` is the documented control-loop mode and restores
  ~2-iteration warm solves (measured 107µs vs the old 29µs on hum-wbc —
  the residual 3.7x is per-iteration cost, mostly the uncached
  factorization).

Performance leads worth pursuing (not done):

- Vectorize the first row-solve sweep across rows (Eigen array ops +
  scalar cleanup for the unconverged tail); the scalar loops are the
  single largest cost.
- A per-row scaled entry-distance measure for the warm-start κ rule
  (the scalar heuristic is the blocker for making the aggressive warm
  path the default).
- Λ-cache with rank-1 factor updates for the few rows that move during
  annealing (would restore mid-solve cache hits).
- relax() could adopt the closed-form row update as its initializer
  (it is qpax's per-row elastic relaxation) — its Newton already
  converges in 2–9 steps, so the win is small.

## Validation

The full suite passes unchanged (`ctest`: pdal_cross_validation,
ipm_reference_validation, bindings, jax_ffi, robot_control): PIQP
cross-checks on feasible/infeasible/expanded forms, IPM-reference
agreement including the relaxed point, warm-start economies, Ruiz paths,
degenerate ties, penalty thresholds, FD-vs-VJP gradient checks, and the
robot control loops (including the humanoid conflict cases at 1e-8
absolute tolerance with 1e5 penalties, which exercised most of the
certificate numerics above).
