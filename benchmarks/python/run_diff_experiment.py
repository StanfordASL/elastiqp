#!/usr/bin/env python
"""Differentiability experiments: elastiqp.jax vs qpax reverse-mode
gradients through the elastic QP.

qpax introduced differentiable QP solving with kappa-relaxed implicit
differentiation; ElastiQP inherits the idea but condenses the backward
pass exactly like the forward solver (O(pn^2 + (n+m)^3) instead of
factoring the full (n+3p+m) KKT Jacobian of the expanded problem). Three
experiments, all on synthetic well-conditioned elastic QPs with active and
inactive constraints (a random subset of constraints is made active at the
optimum), loss L(theta) = 0.5 ||x*(theta)||^2:

1. accuracy  gradient of L w.r.t. every problem matrix vs central finite
             differences of the tight solution map (elastiqp at
             target_kappa = 1e-7 -- the VJP differentiates the RELAXED
             map, so it approaches the tight gradient as kappa -> 0 and
             the comparison carries an O(kappa) smoothing bias by
             design), and elastiqp vs qpax on the expanded formulation.
2. timing    forward-only (target_kappa = 0, no relaxation) vs
             value+gradient (target_kappa = 1e-3: forward + relax + vjp)
             wall time across robot scales (jitted, compile excluded),
             elastiqp native vs qpax on the expanded (n+p)-variable
             formulation (per-constraint penalties; qpax's built-in
             elastic mode supports only a scalar penalty and no
             equalities). Includes a naive dense-KKT backward baseline
             (the O((n+3p+m)^3) factorization the condensed pass
             replaces).
3. kappa     gradient bias of kappa-smoothed differentiation
             (target_kappa > 0, both solvers) vs the exact gradient of
             the tight map (central finite differences).

Outputs: results/diff_{accuracy,timing,kappa}.csv
"""

from __future__ import annotations

import time
from functools import partial

import numpy as np

import bench_common as bc

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import qpax  # noqa: E402
import elastiqp.jax as ejax  # noqa: E402

EPS = 1e-8  # tight forward tolerance so FD checks are clean
ACC_KAPPA = 1e-7  # small smoothing for the accuracy experiment
TIMING_KAPPA = 1e-3  # the recommended robotics setting


def make_problem(n, p, m, seed, active_frac=0.3):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    Q = B @ B.T + 0.5 * np.eye(n)
    q = rng.standard_normal(n)
    G = rng.standard_normal((p, n))
    A = rng.standard_normal((m, n)) if m else np.zeros((0, n))
    x0 = rng.standard_normal(n)
    b = A @ x0 if m else np.zeros(0)
    # Margins: a fraction of constraints active/violated at x0, rest slack.
    margin = rng.uniform(0.1, 1.0, p)
    active = rng.random(p) < active_frac
    h = G @ x0 + np.where(active, -0.05, margin)
    penalty = np.full(p, 10.0)
    return (jnp.array(Q), jnp.array(q), jnp.array(A), jnp.array(b),
            jnp.array(G), jnp.array(h), jnp.array(penalty))


def expand_for_qpax(Q, q, A, b, G, h, penalty):
    """Expanded (n+p)-variable l1-slack formulation (per-row penalties)."""
    n, p = q.size, h.size
    H = jnp.zeros((n + p, n + p)).at[:n, :n].set(Q)
    g = jnp.concatenate([q, penalty])
    Ae = jnp.hstack([A, jnp.zeros((A.shape[0], p))])
    Ge = jnp.block([[G, -jnp.eye(p)],
                    [jnp.zeros((p, n)), -jnp.eye(p)]])
    he = jnp.concatenate([h, jnp.zeros(p)])
    return H, g, Ae, b, Ge, he


def elqp_loss(kappa):
    """target_kappa = 0 is forward-only; differentiation needs kappa > 0."""
    def loss(Q, q, A, b, G, h, penalty):
        sol = ejax.solve(Q, q, G, h, penalty,
                         A=A if A.shape[0] else None,
                         b=b if b.shape[0] else None,
                         eps_abs=EPS, target_kappa=kappa)
        return 0.5 * jnp.sum(sol.x ** 2)
    return loss


def qpax_loss(kappa=1e-8):
    def loss(Q, q, A, b, G, h, penalty):
        H, g, Ae, be, Ge, he = expand_for_qpax(Q, q, A, b, G, h, penalty)
        x = qpax.solve_qp_primal(H, g, Ae, be, Ge, he, solver_tol=EPS,
                                 max_iter=250, target_kappa=kappa)
        return 0.5 * jnp.sum(x[:q.size] ** 2)
    return loss


# ------------------------------------------------- naive dense-KKT backward

@partial(jax.custom_vjp, nondiff_argnums=())
def _naive_solve_x(Q, q, A, b, G, h, penalty):
    x, *_ = ejax.solve(Q, q, G, h, penalty,
                       A=A if A.shape[0] else None,
                       b=b if b.shape[0] else None,
                       eps_abs=EPS)
    return x


def _naive_fwd(Q, q, A, b, G, h, penalty):
    sol = ejax.solve(Q, q, G, h, penalty,
                     A=A if A.shape[0] else None,
                     b=b if b.shape[0] else None,
                     eps_abs=EPS)
    res = (Q, A, G, h, sol.x, sol.t, sol.y, sol.z_t, sol.z_ineq)
    return sol.x, res


def _naive_bwd(res, xb):
    """The backward pass elastiqp.jax's condensed one replaces: assemble the
    full dense (n+3p+m)-dimensional Jacobian K = dF/dw of the elastic KKT
    conditions (same F as elastiqp.jax._kkt_bwd) and pay one
    O((n+3p+m)^3) solve of K' u = wbar, instead of condensing to the
    (n+m) saddle system. Differentiates at the TIGHT certificate, which is
    fine on these synthetic instances (strict complementarity holds by
    construction); the condensed pass differentiates the kappa-relaxed
    point, so the two agree to O(kappa).
    """
    Q, A, G, h, x, t, y, z1, z2 = res
    n, p, m = x.shape[-1], t.shape[-1], y.shape[-1]
    Zn = lambda r, c: jnp.zeros((r, c), Q.dtype)  # noqa: E731
    # Rows: g1 (n), g2 (p), g3 (m), g4 (p), g5 (p); cols: x, t, y, z1, z2.
    K = jnp.block([
        [Q, Zn(n, p), A.T, Zn(n, p), G.T],
        [Zn(p, n), Zn(p, p), Zn(p, m), -jnp.eye(p), -jnp.eye(p)],
        [A, Zn(m, p), Zn(m, m), Zn(m, p), Zn(m, p)],
        [Zn(p, n), jnp.diag(z1), Zn(p, m), jnp.diag(t), Zn(p, p)],
        [jnp.diag(z2) @ G, -jnp.diag(z2), Zn(p, m), Zn(p, p),
         jnp.diag(G @ x - t - h)],
    ])
    wbar = jnp.concatenate([xb, jnp.zeros(p), jnp.zeros(m),
                            jnp.zeros(p), jnp.zeros(p)])
    u = jnp.linalg.solve(K.T, wbar)
    u1, u2, u3 = u[:n], u[n:n + p], u[n + p:n + p + m]
    u5 = u[n + 2 * p + m:]
    v5 = z2 * u5
    qb = -u1
    penalty_b = -u2
    bb = u3
    hb = v5
    Qb = -0.5 * (jnp.outer(u1, x) + jnp.outer(x, u1))
    Ab = -(jnp.outer(y, u1) + jnp.outer(u3, x))
    Gb = -(jnp.outer(z2, u1) + jnp.outer(v5, x))
    return Qb, qb, Ab, bb, Gb, hb, penalty_b


_naive_solve_x.defvjp(_naive_fwd, _naive_bwd)


def naive_loss():
    def loss(Q, q, A, b, G, h, penalty):
        return 0.5 * jnp.sum(_naive_solve_x(Q, q, A, b, G, h, penalty) ** 2)
    return loss


def finite_diff(loss_fn, args, idx, eps=1e-6):
    """Central finite differences of loss w.r.t. args[idx]."""
    args = [np.array(a) for a in args]
    base = args[idx]
    grad = np.zeros_like(base)
    it = np.nditer(base, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        for sgn in (+1, -1):
            pert = [a.copy() for a in args]
            pert[idx][i] += sgn * eps
            val = float(loss_fn(*[jnp.array(a) for a in pert]))
            grad[i] += sgn * val / (2 * eps)
    return grad


def rel_err(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = max(1.0, float(np.abs(b).max()))
    return float(np.abs(a - b).max() / denom)


def experiment_accuracy(rows):
    n, p, m = 8, 20, 3
    names = ["Q", "q", "A", "b", "G", "h", "penalty"]
    # FD ground truth samples the tight map (the loss's VALUE is the tight
    # solution regardless of target_kappa), so it is the same for both
    # solvers; the VJPs carry their respective O(kappa) smoothing bias.
    fd_loss = elqp_loss(0.0)
    for seed in range(3):
        args = make_problem(n, p, m, seed)
        el_grads = jax.grad(elqp_loss(ACC_KAPPA),
                            argnums=tuple(range(7)))(*args)
        qx_grads = jax.grad(qpax_loss(), argnums=tuple(range(7)))(*args)
        for i, name in enumerate(names):
            fd = finite_diff(fd_loss, args, i)
            rows.append([seed, name, n, p, m,
                         rel_err(el_grads[i], fd),
                         rel_err(qx_grads[i], fd),
                         rel_err(el_grads[i], qx_grads[i])])
        print(f"accuracy seed {seed}: worst elqp-vs-fd "
              f"{max(r[5] for r in rows[-7:]):.2e}, qpax-vs-fd "
              f"{max(r[6] for r in rows[-7:]):.2e}")


def time_jit(fn, args, reps=20):
    out = fn(*args)
    jax.block_until_ready(out)  # compile
    t0 = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fn(*args))
    return 1e6 * (time.perf_counter() - t0) / reps


def experiment_timing(rows):
    scales = [(6, 24, 0), (14, 100, 0), (30, 200, 0), (46, 132, 18),
              (58, 500, 0)]
    for n, p, m in scales:
        args = make_problem(n, p, m, seed=0)
        # Forward-only runs the plain solve; value+grad additionally pays
        # relax(target_kappa) in the forward pass and the condensed vjp in
        # the backward -- the full cost of differentiability.
        el_fwd = time_jit(jax.jit(elqp_loss(0.0)), args)
        el_vg = time_jit(jax.jit(jax.value_and_grad(elqp_loss(TIMING_KAPPA))),
                         args)
        nv = naive_loss()
        nv_vg = time_jit(jax.jit(jax.value_and_grad(nv)), args)
        # The naive backward must agree with the condensed one (to O(kappa)).
        g_el = jax.grad(elqp_loss(TIMING_KAPPA), argnums=1)(*args)
        g_nv = jax.grad(nv, argnums=1)(*args)
        naive_err = rel_err(g_nv, g_el)
        try:
            qx = qpax_loss()
            qx_fwd = time_jit(jax.jit(qx), args)
            qx_vg = time_jit(jax.jit(jax.value_and_grad(qx)), args)
        except Exception as e:  # noqa: BLE001
            print(f"qpax failed at n={n} p={p}: {e}")
            qx_fwd = qx_vg = float("nan")
        rows.append([n, p, m, el_fwd, el_vg, nv_vg, qx_fwd, qx_vg,
                     el_vg / el_fwd, nv_vg / el_fwd,
                     qx_vg / qx_fwd if qx_fwd else float("nan"), naive_err])
        print(f"timing n={n:3d} p={p:3d} m={m:2d}: elqp {el_fwd:8.1f} -> "
              f"{el_vg:8.1f} us | naive vg {nv_vg:9.1f} us "
              f"(vs condensed err {naive_err:.1e}) | qpax {qx_fwd:9.1f} -> "
              f"{qx_vg:9.1f} us")


def experiment_kappa(rows):
    n, p, m = 8, 20, 3
    for seed in range(3):
        args = make_problem(n, p, m, seed)
        # Exact gradient of the tight map: central FD w.r.t. q.
        exact = finite_diff(elqp_loss(0.0), args, 1)
        for kappa in (1e-7, 1e-5, 1e-3):
            g_el = jax.grad(elqp_loss(kappa), argnums=1)(*args)
            g_qx = jax.grad(qpax_loss(kappa), argnums=1)(*args)
            rows.append([seed, kappa, rel_err(g_el, exact),
                         rel_err(g_qx, exact), rel_err(g_el, g_qx)])
        print(f"kappa seed {seed} done")


def main():
    acc, timing, kap = [], [], []
    experiment_accuracy(acc)
    bc.write_csv(bc.RESULTS_DIR / "diff_accuracy.csv",
                 ["seed", "wrt", "n", "p", "m", "elqp_vs_fd_relerr",
                  "qpax_vs_fd_relerr", "elqp_vs_qpax_relerr"], acc)
    experiment_timing(timing)
    bc.write_csv(bc.RESULTS_DIR / "diff_timing.csv",
                 ["n", "p", "m", "elqp_fwd_us", "elqp_valgrad_us",
                  "naive_valgrad_us", "qpax_fwd_us", "qpax_valgrad_us",
                  "elqp_grad_ratio", "naive_grad_ratio", "qpax_grad_ratio",
                  "naive_vs_condensed_relerr"], timing)
    experiment_kappa(kap)
    bc.write_csv(bc.RESULTS_DIR / "diff_kappa.csv",
                 ["seed", "kappa", "elqp_bias_relerr", "qpax_bias_relerr",
                  "elqp_vs_qpax_relerr"], kap)


if __name__ == "__main__":
    main()
