# Differentiability of ElastiQP's PDAL method via log-barrier smoothing

*Notes on the implementation of `elastiqp::Solver::relax()` and the
differentiable JAX path (`target_kappa > 0`).*

*Historical note: when this work was done, ElastiQP shipped a secondary
proximal interior-point (IPM) backend, and the IPM's `relax()` was the
existing differentiability path this work replaces. This result made the
IPM backend redundant, and it has since been removed from the library; it
survives as a test-only reference implementation
(`tests/ipm_reference.hpp`, not installed or distributed) that
cross-validates both the solver and the relaxed point. References to "the
IPM" below mean that reference implementation.*

## Summary

The PDAL solver is now differentiable. `Solver::relax(kappa)` walks the
converged solution to the **kappa-relaxed central point** of the elastic QP
— the point satisfying the elastic KKT conditions with complementarity
`s_t ⊙ z_t = s_ineq ⊙ z_ineq = kappa` — which is *the same point*
`IpmSolver::relax(kappa)` produces. The existing implicit-KKT backward pass
(`_kkt_bwd` in `python/elastiqp/jax.py`, formerly `_ipm_bwd`) is evaluated
there unchanged, so:

- `elastiqp.jax.solve(..., target_kappa=1e-3)` is now compatible with
  `jax.grad` / `jax.vjp`, under `jit` and `vmap`.
- PDAL and IPM smoothed gradients agree to ~1e-7 on every parameter
  (`Q, q, A, b, G, h, penalty`), and both match finite differences of their
  relaxed solution maps.
- The fast forward pass is untouched: the relaxation only runs on the
  differentiation path (JAX `custom_vjp` fwd), and `relax()` does not
  disturb the solver's warm-start iterate or factorization cache.
- Relaxed complementarity is **exact to round-off** by construction (see
  below) — the IPM, which walks the central path, only lands within its
  tolerance.

The mechanism is the log-barrier trick of `docs/log_barrier_admm_note.tex`,
adapted to the elastic form: the barrier's closed-form slack prox
`b_κ(v) = (v + sqrt(v² + 4κ))/2` provides an *implicit-complementarity
parametrization* of the slack/dual pairs, and a Newton corrector in those
coordinates converges from the PDAL warm start in a handful of iterations.

## Background

### The IPM's existing differentiability

The IPM (`tests/ipm_reference.hpp`) follows qpax: after `solve()`,
`relax(kappa)` re-solves from the
optimum toward the same KKT system with complementarity `s ⊙ z = κ` (Newton
steps toward the kappa-hyperbola with fraction-to-boundary step limiting,
one KKT factorization per step). The Python backward pass then implicitly
differentiates the *relaxed* KKT system: the constant `κ` offsets drop out
of both Jacobians, so only the evaluation point moves. At that point every
complementarity margin is bounded below by ~κ, which keeps the backward
solve well-conditioned near degenerate (weakly-active) constraints, at the
cost of an O(κ) bias.

### The log-barrier ADMM note

`docs/log_barrier_admm_note.tex` considers a standard slack-form QP, replaces the
indicator `I(s ≥ 0)` with the barrier `-κ Σ log s_i`, and applies two-block
ADMM. Its key observations, all load-bearing here:

1. The slack subproblem has the closed-form prox
   `b_μ(w) = (w + sqrt(w² + 4μ))/2` with `μ = κ/ρ` (note eq. 12).
2. After the paired slack + dual update, the scaled dual and slack satisfy
   `u = b_μ(v)`, `s = b_μ(-v)` — and therefore `u ⊙ s = μ`, i.e.
   `z ⊙ s = κ`, **exactly at every iteration**, not only at convergence
   (note eqs. 14–15).
3. `b_μ' ∈ (0, 1)` is bounded (note eq. 13), which makes the slack map
   smooth — the property that gives smooth derivative evaluations.
4. The note's own caveat: exactness of the update does *not* mean unrolled
   ADMM gradients are well-conditioned; "implicit differentiation of the
   converged fixed point is another option." That option is what we
   implement.

## Making the trick work for the elastic QP / PDAL

### The elastic barrier subproblem and its fixed point

Stack the elastic problem in `X = (x, t)`:

```
min  ½xᵀQx + qᵀx + penaltyᵀt
s.t. Ax = b,   [G  -I] X ≤ h   (slack s_ineq),   [0  -I] X ≤ 0   (slack s_t)
```

and replace both slack indicators with `-κ Σ log`. The stationarity
conditions of the barrier problem are precisely the elastic KKT with
`z ⊙ s = κ` on both blocks — the IPM's relaxation target:

```
F1:  Qx + q + Aᵀy + Gᵀz_ineq = 0
F2:  penalty - z_t - z_ineq  = 0
F3:  Ax - b                  = 0
F4:  s_t   = t
F5:  s_ineq = h + t - Gx
     z_t ⊙ s_t = z_ineq ⊙ s_ineq = κ,   all four > 0
```

So *any* solver for this system yields the same differentiation point, and
the existing backward pass applies verbatim. The question is only how to
reach it cheaply from the PDAL solution.

### What we tried first: literal ADMM iteration

Applying the note's ADMM directly to the stacked form works out neatly on
paper. The `(t,t)` block of `H_ρ = Q̃ + ρÃᵀÃ + ρG̃ᵀG̃` is just `2ρI`, so the
X-step condenses onto an n×n SPD system `K = Q + ρAᵀA + (ρ/2)GᵀG` — **one
factorization for the entire relaxation**, with each sweep just a triangular
solve plus matvecs, and `z ⊙ s = κ` exact after every sweep.

Numerically (numpy prototype, warm-started from the PDAL solution and
validated against `IpmSolver::relax`):

- The fixed point is correct: agreement with the IPM relaxed point to
  ~1e-10 whenever the iteration converged.
- But convergence is **linearly slow in the tail**. Small well-conditioned
  instances took 100–250 sweeps to reach 1e-10; an n=14, p=60 instance with
  many conflicting rows contracted at ~0.977/sweep — thousands of sweeps.
  ADMM's linear rate is dominated by the weakly-active rows the smoothing
  exists to handle, which is exactly backwards.
- Per-row penalties `ρ_i = z_i/s_i` (the "self-scaled" choice, which makes
  every row sit at the maximally-smooth point `w = 0`, `b_μ' = ½` of its
  retraction, and whose condensed weight `ρ₁ρ₂/(ρ₁+ρ₂)` is exactly qpax's
  elastic weight `1/(s₁/z₁ + s₂/z₂)`) speed up the good cases but go
  unstable for small κ, where the penalties reach `z²/κ`.
- A subtlety worth recording: solving the X-step *directly* (for `X` from a
  large right-hand side) floors the attainable residual at
  `cond(K)·eps ≈ 1e-9`; the X-step must be taken in **delta form** (solve
  for the step against the AL gradient, which is residual-sized near the
  fixed point) to reach 1e-11.

### What we shipped: Newton in the retraction coordinates

The note's §3 supplies the better vehicle. Since one ADMM slack+dual sweep
*is* the paired retraction, parametrize each slack/dual pair by a single
free variable `v` per row:

```
z = b_κ(v),    s = b_κ(-v),    v = z - s
```

Then `z ⊙ s = κ` and `z, s > 0` hold **identically for every v** (note
identity `b_κ(v)·b_κ(-v) = κ`), the complementarity rows disappear from the
system, and what remains — F1–F5 above in the unknowns
`(x, t, y, v_t, v_ineq)` — is smooth and unconstrained. We run plain Newton
on it (`relax()` in `elastiqp.hpp`):

- **Warm start**: from the tight PDAL certificate through the same
  retraction, `v_t = z_t - s_t`, `v_ineq = z_ineq - s_ineq`. The starting
  point already sits on the κ-manifold; only stationarity/feasibility
  residuals (size O(κ)) remain.
- **Condensation**: with `D = b'(v)/b'(-v) = z/s` per row, eliminating
  `dv_t, dv_ineq, dt` collapses each Newton step onto the same n×n SPD
  shape both solvers already use,

  ```
  K = Q + ρI + (1/δ)AᵀA + Gᵀ diag(Λ) G,
  Λ = D_ineq (ρ + D_t) / (ρ + D_t + D_ineq)  →  z_t z_ineq / (z_t s_ineq + z_ineq s_t)
  ```

  which is again qpax's elastic weight — the same matrix the IPM's relax
  factors, reached from the opposite (AL) direction. One factorization per
  Newton iteration.
- **No boundary safeguards**: since positivity is built into the
  parametrization there is no fraction-to-boundary step limit; the only
  safeguard is a backtracking line search. The line search **must accept on
  the 2-norm merit `½‖F‖₂²`**, not the max-norm: the Newton step is a
  guaranteed descent direction for the former (slope `-FᵀF < 0` at exact
  Jacobians) but not the latter, and a max-norm-monotone search measurably
  stalls (100+ rejected halvings) whenever a full step trades residual
  between rows — which is exactly what happens when starting anywhere but
  the tight certificate. Termination stays on the max-norm.
- **Regularization**: small fixed prox terms (`relax_reg = 1e-9` on the
  primal diagonal and the equality dual, escalated ×100 on factorization
  failure). The prox centers sit at the current iterate, so — as in the
  IPM's relax — the regularization damps the step without perturbing the
  fixed point.
- **Numerics**: `b_κ` and `1 - b_κ'` use the cancellation-free branches
  (`2κ/(sqrt(v²+4κ) - v)` for `v < 0`; `2κ/(r(r+|v|))` for the derivative
  complement), as emphasized in the note.
- **Ruiz**: iterates live in the equilibrated frame; each `s ⊙ z` pair picks
  up only the cost factor (slacks scale with the row, duals against it), so
  the scaled-space target is `c·κ` and termination is on unscaled residuals
  — identical to the IPM's handling.
- **State isolation**: unlike `IpmSolver::relax` (which moves its iterate),
  the PDAL relax runs on a separate workspace and factorization. The tight
  iterate and the factorization cache survive, so a control loop can
  `solve() → relax() → set_*() → solve()` and still get its zero/low-cost
  warm re-solves.

In practice this converges in **2–9 Newton iterations** across
κ ∈ [1e-9, 1e-2] on every instance we generated (feasible, infeasible with
many conflicting rows, degenerate/weakly-active, m = 0, p ≫ n), where the
literal ADMM iteration needed hundreds-to-thousands of sweeps or diverged
when over-scaled.

### Warm starting relax() across control ticks

In a control loop (`solve() → relax() → set_*() → solve() → relax() → …`)
the previous tick's relaxed iterate is an obvious candidate start for the
next relaxation: it persists in the relax workspace, and because the
v-parametrization is kappa-agnostic — *any* v maps onto the current κ
manifold exactly — a stale iterate is always a *valid* start, even after
data or κ changes. `Settings::relax_warm_start` (default on) enables this.

Making it *profitable* turned out to be the interesting part. What the
experiments showed (all measured on drifting control-loop instances,
n up to 58, p up to 400):

- **The retraction start is a strong incumbent.** Its x, t, y satisfy the
  new tight KKT to solver tolerance, and its v *sign pattern encodes the
  new smoothed configuration exactly*, so its (often larger) residual is
  concentrated in the weakly-active rows where Newton contracts fastest.
- **Residual norms do not predict the winner.** A stale iterate with a
  10× smaller residual can still converge slower: its error is spread
  across every densely-coupled equation, and — the failure mode that
  matters — rows whose smoothed configuration changed must traverse the
  curved region of `b_κ` near `v = 0` (curvature scale `√κ`), where full
  Newton steps stall against the line search. This is the retraction-space
  version of the classic "IPMs can't warm start across active-set changes"
  phenomenon.
- **The winning criterion is dimensional.** The stale iterate is adopted
  only if its residual is below both `0.1×` the retraction start's residual
  *and* `0.02·√κ` — i.e. well inside the retraction's curvature basin.
  Calibration was sharp: residuals under `0.02·√κ` won consistently
  (2–3× fewer iterations), `~0.1·√κ` was a coin flip, and above that the
  stale start reliably lost. Selection costs three residual evaluations
  (a few matvecs), no factorization.
- **Bounded regret backstop.** If an adopted warm start has not dropped the
  residual by two orders within 4 iterations, it is abandoned for the
  retraction start; wasted work is capped and counted in `Solution::iters`.
  With the selection rule above this rarely fires.

Measured over drifting ticks (κ = 1e-3, tol 1e-10, drift 1e-4 — a typical
high-rate control loop; "cold" = `relax_warm_start = false`):

| instance | cold relax | warm relax |
|---|---|---|
| n=30, m=8, p=200 | 382 µs/tick (10.3 it) | 163 µs/tick (4.1 it) |
| n=58, m=15, p=400 | 1277 µs/tick (7.8 it) | 540 µs/tick (3.2 it) |

At drift ≥ 1e-2 (or κ = 1e-6, where the basin `√κ = 1e-3` is far tighter
than any realistic drift) the selection falls back to the retraction start
and warm behaves identically to cold — across every (instance, κ, drift)
combination swept, warm start was never worse. Repeated `relax()` on an
unchanged problem converges in 0 iterations. The relaxed point itself is
unchanged by warm starting (verified to ~1e-11); only the start moves. The
JAX FFI path is cold-start-per-call (functional purity) and does not use
this.

### Limitation shared with the IPM

Rows with `penalty_i = 0` admit no relaxed point: `z_t + z_ineq = 0` cannot
hold with `z > 0` (the barrier subproblem in `t_i` is unbounded). The
relaxation cannot converge for such rows — drop them from the problem
instead. The IPM's relax has the same limit; the elastic KKT itself is fine
with zero-penalty rows in the *tight* solve.

## Comparison with the IPM's differentiability

| | IPM `relax` | PDAL `relax` (this work) |
|---|---|---|
| Target point | κ-central point | identical κ-central point |
| Backward pass | implicit KKT (`_kkt_bwd`) | same code, same formulas |
| Mechanism | Newton toward the κ-hyperbola along the central path | Newton in retraction coordinates `v` |
| Complementarity at result | within solver tolerance | exact to round-off (`z⊙s = κ` by construction) |
| Positivity | fraction-to-boundary step limiting (`tau`) | automatic (range of `b_κ`) |
| Cost per iteration | 1 KKT factorization | 1 KKT factorization (same condensed shape) |
| Typical iterations (warm) | ~2–8 | ~2–9 (1–4 with tick-to-tick warm start) |
| Solver state after | iterate moves to relaxed point | tight iterate and factorization cache preserved |
| Exact gradients (κ = 0) | supported — iterates stay strictly interior | **not defined** — the PDAL certificate sits exactly on the boundary (`s` or `z` exactly 0), so the exact-KKT backward solve divides by zero. `jax.grad` at `target_kappa=0` raises a `TypeError` naming the fix. |

Since both solvers produce the same relaxed point, the gradients agree to
the tolerance of the relax solves; we measure ≤1.5e-7 max elementwise
difference across all seven parameters at κ = 1e-3, and ~1e-10 agreement of
the relaxed points themselves in the C++ cross-checks.

Timing on random dense instances (i7 laptop, cold solves, κ = 1e-3, tol
1e-10; the relax increment is what differentiation adds):

| instance | PDAL solve | + relax | IPM solve | + relax |
|---|---|---|---|---|
| n=30, m=8, p=200 | 1300 µs | +190 µs (9 it) | 620 µs | +260 µs (7 it) |
| n=58, m=15, p=400 | 5100 µs | +1130 µs (7 it) | 2540 µs | +1330 µs (8 it) |

(The PDAL's advantage is warm-started re-solves in control loops — ~5–15 µs
on these sizes — which the cold-start JAX FFI does not exercise; cold,
heavily-conflicted instances can favor the IPM as above.) The relaxation
overhead is comparable between the two methods, so the practical win is
that the IPM stopped being *required* for anything but κ = 0 exact
gradients or tighter equality residuals — which is what let us remove it
from the library entirely and keep it only as the test-suite reference.

## Workflow

Unchanged from the IPM pattern, now on `elastiqp::Solver`:

- **C++ / Python bindings**: `solver.solve()` for the forward pass;
  `solver.relax(kappa)` only when a differentiation point is needed. The
  returned `Solution` is the relaxed certificate; `solver`'s warm state
  still holds the tight solution. Across control ticks, repeated `relax()`
  calls warm start from the previous relaxed iterate automatically (see
  above; disable with `Settings::relax_warm_start = false`).
- **JAX**: `elastiqp.jax.solve(..., target_kappa=1e-3)`. The primal path
  always runs the plain tight solve (relaxation skipped); the `custom_vjp`
  fwd runs solve + relax and stashes the relaxed point; the bwd solves the
  transposed KKT system there. `Result.converged` folds in the relaxation
  status on the differentiated path, so a failed relaxation (e.g. an
  unreachable κ) is never silent. The FFI handler signature is
  `(..., target_kappa) -> (tight block, relaxed block, info[3])`.

## Validation

- `tests/test_pdal.cc`: PDAL relax vs the IPM reference's relax agreement
  (κ ∈ {1e-2, 1e-3, 1e-6}, with equalities and conflicts), round-off-exact
  complementarity, iterate/warm-start preservation across relax; relax warm
  start across drifting ticks (fewer iterations, same relaxed point), under
  a κ change, and graceful fallback after a full problem jump.
- `tests/test_jax_ffi.py`: smoothed gradients vs central finite differences
  of the relaxed solution map on all seven parameters (plus degenerate,
  m = 0, and batched instances); κ → 0 convergence to the tight
  derivative; exactness of `s ⊙ z = κ`; value unchanged by κ on the fwd
  path; failed relaxation reported through `converged`; grads under `jit`,
  `vmap`, and invariant to Ruiz; `TypeError` at `target_kappa=0`.

## Possible follow-ups

- The retraction Jacobian is bounded (`b' ∈ (0,1)`), which is the property
  the Arrizabalaga et al. single-precision IPM builds on — a float32 relax
  path may be feasible if that ever matters for deployment.
- The literal ADMM sweep (one factorization total) could still pay off for
  very large n where factorizations dominate, if paired with acceleration;
  the measured linear-rate tail is the blocker.
