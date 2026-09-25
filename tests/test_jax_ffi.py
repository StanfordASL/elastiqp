"""ElastiQP JAX FFI test cases"""

import sys

import jax

jax.config.update("jax_enable_x64", True)

import elastiqp
import elastiqp.jax
import jax.numpy as jnp
import numpy as np

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  {name:<52s} {detail:<24s} {'OK' if ok else 'FAIL'}")


def random_qp(seed, n, m, p):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    P0 = rng.standard_normal((n, n))
    Q = P0.T @ P0 + 1e-3 * np.eye(n)
    A = rng.standard_normal((m, n))
    G = rng.standard_normal((p, n))
    s = np.abs(rng.standard_normal(p))
    z = np.abs(rng.standard_normal(p))
    mask = rng.random(p) < 0.5
    s[mask] = 0.0
    z[~mask] = 0.0
    h = G @ x + s
    b = A @ x
    y = rng.standard_normal(m)
    q = -Q @ x - G.T @ z - A.T @ y
    return Q, q, A, b, G, h, x


def make_infeasible(G, h, n_conflicts, gap=1.0):
    G, h = G.copy(), h.copy()
    for k in range(n_conflicts):
        i, j = 2 * k, 2 * k + 1
        if j >= len(h):
            break
        G[j] = -G[i]
        h[j] = -(h[i] + gap)
    return G, h


# Agreement between the FFI library and the nanobind extension, which run
# the same C++ on the same inputs. They are separate shared objects, so the
# last bit can differ (XLA also runs the callback on its own thread pool,
# perturbing the FP environment / buffer alignment). The solver's exact
# line search and discrete active-set test amplify a last-bit difference to
# ~1e-11 over a dozen Newton steps -- still two orders under the 1e-8
# solver tolerance, and each library is bit-deterministic with itself.
FFI_NB_TOL = 1e-9


def main():
    print("FFI vs nanobind: same C++ code, same answers (every backend)")
    Q, q, A, b, G, h, x_star = random_qp(0, 14, 0, 40)
    for method in elastiqp.jax.METHODS:
        sol = elastiqp.jax.solve(Q, q, G, h, 10.0, method=method)
        nb_sol = elastiqp.solve(Q, q, G, h, 10.0, method=method)
        dx = np.abs(np.asarray(sol.x) - nb_sol.x).max()
        check(
            f"inequality-only n=14 p=40 [{method}]",
            int(sol.converged) == 1 and dx < FFI_NB_TOL,
            f"|dx|={dx:.1e}",
        )
    sol = elastiqp.jax.solve(Q, q, G, h, 10.0)

    Q, q, A, b, G, h, x_star = random_qp(1, 14, 4, 60)
    # eq < 1e-8 needs a tighter solve than the 1e-5 default.
    sol = elastiqp.jax.solve(Q, q, G, h, 1e3, A=A, b=b, eps_abs=1e-8)
    nb_sol = elastiqp.solve(Q, q, G, h, 1e3, A=A, b=b, eps_abs=1e-8)
    dx = np.abs(np.asarray(sol.x) - nb_sol.x).max()
    eq = np.abs(A @ np.asarray(sol.x) - b).max()
    check(
        "with equalities n=14 m=4 p=60",
        int(sol.converged) == 1 and dx < FFI_NB_TOL and eq < 1e-8,
        f"|dx|={dx:.1e} eq={eq:.1e}",
    )
    check(
        "recovers known optimum",
        np.abs(np.asarray(sol.x) - x_star).max() < 1e-5,
        f"|dx*|={np.abs(np.asarray(sol.x) - x_star).max():.1e}",
    )

    print("Infeasible inequalities: equalities hold, slacks activate")
    G2, h2 = make_infeasible(G, h, 15)
    sol = elastiqp.jax.solve(Q, q, G2, h2, 10.0, A=A, b=b)
    eq = np.abs(A @ np.asarray(sol.x) - b).max()
    check(
        "Ax=b under conflict",
        int(sol.converged) == 1 and eq < 1e-8 and float(jnp.max(sol.t)) > 0.1,
        f"eq={eq:.1e} max_t={float(jnp.max(sol.t)):.2f}",
    )

    print("Under jit")
    penalty_vec = jnp.full(60, 10.0)

    @jax.jit
    def controller(q_k, h_k, b_k):
        s = elastiqp.jax.solve(Q, q_k, G2, h_k, penalty_vec, A=A, b=b_k)
        return s.x, s.converged

    x_jit, conv = controller(jnp.asarray(q), jnp.asarray(h2), jnp.asarray(b))
    dx = np.abs(np.asarray(x_jit) - np.asarray(sol.x)).max()
    check("jit matches eager", int(conv) == 1 and dx == 0.0, f"|dx|={dx:.1e}")

    # jit with different static options recompiles and respects them
    @jax.jit
    def loose(q_k):
        return elastiqp.jax.solve(Q, q_k, G2, h2, 10.0, A=A, b=b, eps_abs=1e-4).iters

    check(
        "static eps_abs attribute honored",
        int(loose(jnp.asarray(q))) <= int(sol.iters),
        f"{int(loose(jnp.asarray(q)))} <= {int(sol.iters)}",
    )

    print("Under vmap (sequential)")
    rng = np.random.default_rng(5)
    batch_q = jnp.asarray(q + 0.05 * rng.standard_normal((8, len(q))))
    vsolve = jax.vmap(lambda q_k: elastiqp.jax.solve(Q, q_k, G2, h2, 10.0, A=A, b=b).x)
    xs = vsolve(batch_q)
    ref = np.stack(
        [
            elastiqp.solve(Q, np.asarray(batch_q[i]), G2, h2, 10.0, A=A, b=b).x
            for i in range(8)
        ]
    )
    dx = np.abs(np.asarray(xs) - ref).max()
    check(
        "batch of 8 matches per-problem solves",
        xs.shape == (8, 14) and dx < FFI_NB_TOL,
        f"|dx|={dx:.1e}",
    )

    print("Composability: FFI output feeds jnp computation inside jit")

    @jax.jit
    def objective(q_k):
        s = elastiqp.jax.solve(Q, q_k, G2, h2, 10.0, A=A, b=b)
        return 0.5 * s.x @ Q @ s.x + q_k @ s.x + 10.0 * jnp.sum(s.t)

    obj = float(objective(jnp.asarray(q)))
    x_np = np.asarray(sol.x)
    obj_ref = 0.5 * x_np @ Q @ x_np + q @ x_np + 10.0 * np.asarray(sol.t).sum()
    check("objective matches", abs(obj - obj_ref) < 1e-9, f"obj={obj:.4f}")

    print("Smoothed gradients vs finite differences of the relaxed map")
    # Lightly-constrained instance: active set (2 eqs + a few active/elastic
    # ineqs) is smaller than n, so the solution is NOT pinned to a vertex and
    # every parameter (including q, Q, penalty) has a nonzero gradient. A
    # fully-pinned vertex solution would make x locally independent of q.
    # The seed is chosen for healthy strict-complementarity margins (~0.2):
    # a near-degenerate instance turns solver error into O(1e-3) gradient
    # error at small kappa. Tight solver tolerance so FD noise
    # (~eps_solve/step) stays small.
    #
    # The smoothed gradient is by definition the derivative of the RELAXED
    # solution map x_r(theta) (the value returned is still the tight x). So
    # the ground truth is central finite differences of a loss evaluated on
    # x_r, which we access through the raw FFI call -- an entirely
    # solver-independent check, run on every parameter. The loss is LINEAR
    # in (x, t): the VJP's cotangent is evaluated at the tight value while
    # the FD samples the relaxed map, and only a constant cotangent makes
    # the two the same function (a nonlinear loss term would differ by
    # O(kappa * loss curvature) by construction, not by error).
    kappa = 1e-3
    Qf, qf, Af, bf, Gf, hf, _ = random_qp(22, 12, 2, 8)
    Gf, hf = make_infeasible(Gf, hf, 2)  # active elastic slacks
    pen = 10.0 + jnp.linspace(0.0, 5.0, 8)
    rngd = np.random.default_rng(17)
    w_loss = jnp.asarray(rngd.standard_normal(12))
    w_t = jnp.asarray(0.3 * rngd.standard_normal(8))
    args = tuple(jnp.asarray(v) for v in (Qf, qf, Af, bf, Gf, hf, pen))

    def loss_smooth(Q_, q_, A_, b_, G_, h_, penalty_, kap=kappa):
        s = elastiqp.jax.solve(
            Q_,
            q_,
            G_,
            h_,
            penalty_,
            A=A_,
            b=b_,
            method="pdal",
            eps_abs=1e-11,
            max_iter=300,
            target_kappa=kap,
        )
        return w_loss @ s.x + w_t @ s.t

    def loss_relaxed(Q_, q_, A_, b_, G_, h_, penalty_, kap=kappa):
        out = elastiqp.jax._ffi_solve(
            Q_,
            q_,
            A_,
            b_,
            G_,
            h_,
            penalty_,
            1e-11,
            300,
            False,
            kap,
            "pdal",
        )
        xr, tr = out[5], out[6]
        return w_loss @ xr + w_t @ tr

    grads = jax.jit(jax.grad(loss_smooth, argnums=tuple(range(7))))(*args)

    # FD steps: the relaxed map is computed to ~1e-9 (relax tol amplified by
    # the O(1/kappa) conditioning of the relaxed KKT), so its difference
    # quotient needs the larger step; the smoothing keeps the curvature
    # error at ~step^2/kappa, still well under tolerance. The tight map is
    # solved to ~1e-11 and takes the smaller step.
    step_r = 1e-4
    step = 1e-5
    names = ["Q", "q", "A", "b", "G", "h", "penalty"]
    for k, name in enumerate(names):
        v = np.asarray(rngd.standard_normal(args[k].shape))
        v /= np.linalg.norm(v)
        pert = list(args)
        pert[k] = args[k] + step_r * v
        f_plus = float(loss_relaxed(*pert))
        pert[k] = args[k] - step_r * v
        f_minus = float(loss_relaxed(*pert))
        fd = (f_plus - f_minus) / (2 * step_r)
        an = float(jnp.sum(grads[k] * v))
        err = abs(fd - an) / max(1.0, abs(fd))
        # Require the gradient to be genuinely nonzero so the check has teeth.
        check(
            f"d/d{name} (directional, jit)",
            err < 1e-4 and abs(an) > 1e-4,
            f"an={an:+.5f} fd={fd:+.5f}",
        )

    print("kappa -> 0: smoothed gradients converge to the tight derivative")
    # Away from active-set changes the tight solution map is differentiable
    # and the smoothed gradient converges to its derivative as kappa -> 0
    # (the O(kappa) bias vanishes). Ground truth is central finite
    # differences of a loss on the TIGHT solution (plain solves, no
    # relaxation anywhere).
    v = np.asarray(rngd.standard_normal(12))
    v /= np.linalg.norm(v)

    def loss_tight(q_):
        s = elastiqp.jax.solve(
            args[0],
            q_,
            args[4],
            args[5],
            args[6],
            A=args[2],
            b=args[3],
            method="pdal",
            eps_abs=1e-11,
            max_iter=300,
        )
        return w_loss @ s.x + w_t @ s.t

    fd_tight = (
        float(loss_tight(args[1] + step * v)) - float(loss_tight(args[1] - step * v))
    ) / (2 * step)
    kappas = (1e-3, 1e-5, 1e-7)
    g_q = lambda kap: jax.grad(lambda q_: loss_smooth(args[0], q_, *args[2:], kap=kap))(
        args[1]
    )
    errs = [abs(float(jnp.sum(g_q(kap) * v)) - fd_tight) for kap in kappas]
    check(
        "directional error decreases in kappa, -> 0",
        errs[0] > errs[-1] and errs[-1] < 1e-4,
        " ".join(f"{e:.1e}" for e in errs),
    )

    # The smoothing payoff: on a near-degenerate instance (weakly active
    # constraint), the exact implicit gradient is ill-conditioned, but the
    # smoothed gradient still matches the relaxed map's true derivative.
    Qd, qd, Ad, bd, Gd, hd, _ = random_qp(9, 12, 2, 8)
    Gd, hd = make_infeasible(Gd, hd, 2)
    pend = 10.0 + jnp.linspace(0.0, 5.0, 8)
    argsd = tuple(jnp.asarray(vv) for vv in (Qd, qd, Ad, bd, Gd, hd, pend))

    g_d = jax.grad(lambda q_: loss_smooth(argsd[0], q_, *argsd[2:]))(argsd[1])
    fd = (
        float(loss_relaxed(argsd[0], argsd[1] + step_r * v, *argsd[2:]))
        - float(loss_relaxed(argsd[0], argsd[1] - step_r * v, *argsd[2:]))
    ) / (2 * step_r)
    an = float(jnp.sum(g_d * v))
    check(
        "smoothed grad well-conditioned at degenerate instance",
        abs(fd - an) / max(1.0, abs(fd)) < 1e-4,
        f"an={an:+.5f} fd={fd:+.5f}",
    )

    # Value is unaffected by kappa: still the tight solution. Compare the
    # differentiation (fwd) path, which runs the relaxation, against the
    # value-only path, which skips it.
    v_smooth, _ = jax.value_and_grad(lambda q_: loss_smooth(args[0], q_, *args[2:]))(
        args[1]
    )
    v_tight = loss_tight(args[1])
    dv = abs(float(v_smooth) - float(v_tight))
    check("value unchanged by kappa (fwd path)", dv == 0.0, f"|dv|={dv:.1e}")

    # The retraction z = b_kappa(v), s = b_kappa(-v) enforces s.z = kappa by
    # construction, so the relaxed complementarity is exact to round-off.
    # The status must be observable: info[2] reports the relax convergence
    # (the backward pass differentiates at that point, so a stalled
    # relaxation would otherwise mean silently wrong gradients). s_t = t and
    # s_in = h + t - Gx are the slacks of t >= 0 and Gx - t <= h; s_in is
    # reconstructed from x, so it carries the O(tol) primal residual.
    out = elastiqp.jax._ffi_solve(*args, 1e-11, 300, False, kappa, "pdal")
    xr, tr, z_t_r, z_r, info = out[5], out[6], out[8], out[9], out[10]
    s_in_r = args[5] + tr - args[4] @ xr
    comp = max(
        float(jnp.max(jnp.abs(tr * z_t_r - kappa))),
        float(jnp.max(jnp.abs(s_in_r * z_r - kappa))),
    )
    check(
        "relaxed point satisfies s.z = kappa, and reports so",
        comp < 1e-13 and int(info[2]) == 1,
        f"|s.z-k|={comp:.1e}",
    )

    # ...and a relaxation that cannot reach its target must not fail
    # silently: an absurd kappa converges the solve (info[0]) but stalls the
    # relaxation (info[2]), and solve() folds that into converged on the
    # differentiated path.
    out = elastiqp.jax._ffi_solve(*args, 1e-11, 300, False, 1e8, "pdal")
    tight_ok, relax_bad = float(out[10][0]) == 1.0, float(out[10][2]) == 0.0

    def conv_smooth(q_):
        s = elastiqp.jax.solve(
            args[0],
            q_,
            args[4],
            args[5],
            args[6],
            A=args[2],
            b=args[3],
            method="pdal",
            eps_abs=1e-11,
            max_iter=300,
            target_kappa=1e8,
        )
        return jnp.sum(s.x), s.converged

    (_, conv), _ = jax.value_and_grad(conv_smooth, has_aux=True)(args[1])
    check(
        "failed relaxation is reported via converged",
        tight_ok and relax_bad and int(conv) == 0,
        f"info={[int(vv) for vv in out[10]]} conv={int(conv)}",
    )

    # Smoothed gradient through vmap: the batched _kkt_bwd path (batched
    # einsums + batched saddle solve) with a relaxed evaluation point is
    # otherwise never exercised.
    batch_dq = jnp.asarray(0.05 * rngd.standard_normal((4, 12)))
    qs_b = args[1] + batch_dq

    def batched_smooth(qs):
        return jnp.sum(jax.vmap(lambda q_: loss_smooth(args[0], q_, *args[2:]))(qs))

    g_b = jax.jit(jax.grad(batched_smooth))(qs_b)
    g_ref = jnp.stack(
        [
            jax.grad(lambda q_: loss_smooth(args[0], q_, *args[2:]))(qs_b[i])
            for i in range(4)
        ]
    )
    db = float(jnp.abs(g_b - g_ref).max())
    check(
        "smoothed d/dq through vmap matches per-problem",
        g_b.shape == (4, 12) and db < 1e-9,
        f"|dg|={db:.1e}",
    )
    # ...and the batched gradient is right, not just self-consistent: FD of
    # the summed per-problem RELAXED losses (the smoothed gradient is the
    # relaxed map's derivative, so the tight-value loss is not the right
    # FD target).
    batched_relaxed = lambda qs: sum(
        float(loss_relaxed(args[0], qs[i], *args[2:])) for i in range(4)
    )
    fd_b = (
        batched_relaxed(qs_b + step_r * jnp.asarray(v))
        - batched_relaxed(qs_b - step_r * jnp.asarray(v))
    ) / (2 * step_r)
    an_b = float(jnp.sum(g_b * jnp.asarray(v)))
    check(
        "batched d/dq matches FD of the relaxed losses",
        abs(fd_b - an_b) / max(1.0, abs(fd_b)) < 1e-4,
        f"an={an_b:+.5f} fd={fd_b:+.5f}",
    )

    # No equality constraints: the backward pass degenerates from an (n+m)
    # saddle system to a plain n x n solve.
    Q0, q0, _, _, G0, h0, _ = random_qp(30, 12, 0, 8)
    G0, h0 = make_infeasible(G0, h0, 2)
    pen0 = 10.0 + jnp.linspace(0.0, 5.0, 8)
    A0 = jnp.zeros((0, 12))
    b0 = jnp.zeros((0,))
    a0 = tuple(jnp.asarray(vv) for vv in (Q0, q0, A0, b0, G0, h0, pen0))

    g_m0 = jax.jit(jax.grad(lambda q_: loss_smooth(a0[0], q_, *a0[2:])))(a0[1])
    fd = (
        float(loss_relaxed(a0[0], a0[1] + step_r * v, *a0[2:]))
        - float(loss_relaxed(a0[0], a0[1] - step_r * v, *a0[2:]))
    ) / (2 * step_r)
    an = float(jnp.sum(g_m0 * v))
    check(
        "smoothed d/dq with no equalities (m=0)",
        abs(fd - an) / max(1.0, abs(fd)) < 1e-4 and abs(an) > 1e-4,
        f"an={an:+.5f} fd={fd:+.5f}",
    )

    print("Infeasible QP: jit, vmap, ruiz; grad needs kappa")
    Qx, qx, Ax, bx, Gx, hx, _ = random_qp(21, 14, 4, 60)
    Gx, hx = make_infeasible(Gx, hx, 15)
    nb_ref = elastiqp.solve(Qx, qx, Gx, hx, 10.0, A=Ax, b=bx)
    ps = elastiqp.jax.solve(Qx, qx, Gx, hx, 10.0, A=Ax, b=bx)
    dx = float(jnp.abs(ps.x - jnp.asarray(nb_ref.x)).max())
    check(
        "matches nanobind on infeasible QP",
        int(ps.converged) == 1 and dx < FFI_NB_TOL,
        f"|dx|={dx:.1e}",
    )

    jit_solve = jax.jit(lambda q_: elastiqp.jax.solve(Qx, q_, Gx, hx, 10.0, A=Ax, b=bx))
    dj = float(jnp.abs(jit_solve(qx).x - ps.x).max())
    check("jit matches eager (infeasible)", dj == 0.0, f"|dx|={dj:.1e}")

    qs = qx + 0.01 * np.random.default_rng(5).standard_normal((8, 14))
    batch = jax.vmap(lambda q_: elastiqp.jax.solve(Qx, q_, Gx, hx, 10.0, A=Ax, b=bx).x)(
        qs
    )
    worst = max(
        float(
            jnp.abs(
                batch[i] - elastiqp.jax.solve(Qx, qs[i], Gx, hx, 10.0, A=Ax, b=bx).x
            ).max()
        )
        for i in range(8)
    )
    check("vmap matches per-problem", worst == 0.0, f"|dx|={worst:.1e}")

    scale = 10.0 ** np.random.default_rng(6).uniform(-3, 3, 60)
    rs = elastiqp.jax.solve(
        Qx, qx, Gx * scale[:, None], hx * scale, 10.0 / scale, A=Ax, b=bx, ruiz=True
    )
    dr = float(jnp.abs(rs.x - jnp.asarray(nb_ref.x)).max())
    check(
        "ruiz=True on rescaled rows",
        int(rs.converged) == 1 and dr < 1e-4,
        f"|dx|={dr:.1e}",
    )

    # The exact (kappa=0) KKT derivative is undefined at the boundary
    # certificate: grad with an explicit target_kappa=0 must fail with a
    # message that names the fix, not a generic JAX internal. (The default
    # target_kappa=1e-3 is differentiable out of the box.)
    try:
        jax.grad(
            lambda q_: jnp.sum(
                elastiqp.jax.solve(
                    Qx, q_, Gx, hx, 10.0, A=Ax, b=bx, method="pdal", target_kappa=0.0
                ).x
            )
        )(qx)
        msg = None
    except TypeError as e:
        msg = str(e)
    check("grad at kappa=0 raises", msg is not None, "")
    check(
        "...with a message pointing at target_kappa",
        msg is not None and "target_kappa" in msg and "not differentiable" in msg,
        "",
    )

    # With the default target_kappa (1e-3), grad works out of the box on the
    # pdal and ipm backends.
    for method in ("pdal", "ipm"):
        g_def = jax.grad(
            lambda q_: jnp.sum(
                elastiqp.jax.solve(Qx, q_, Gx, hx, 10.0, A=Ax, b=bx, method=method).x
            )
        )(qx)
        check(
            f"grad with default target_kappa is finite [{method}]",
            bool(jnp.all(jnp.isfinite(g_def))),
            "",
        )
    # ...and the two backends agree on it (same relaxed KKT point).
    g_pd, g_ip = (
        jax.grad(
            lambda q_: (
                loss_smooth(args[0], q_, *args[2:])
                if m == "pdal"
                else w_loss
                @ elastiqp.jax.solve(
                    args[0],
                    q_,
                    args[4],
                    args[5],
                    args[6],
                    A=args[2],
                    b=args[3],
                    method="ipm",
                    eps_abs=1e-11,
                    max_iter=300,
                    target_kappa=kappa,
                ).x
                + w_t
                @ elastiqp.jax.solve(
                    args[0],
                    q_,
                    args[4],
                    args[5],
                    args[6],
                    A=args[2],
                    b=args[3],
                    method="ipm",
                    eps_abs=1e-11,
                    max_iter=300,
                    target_kappa=kappa,
                ).t
            )
        )(args[1])
        for m in ("pdal", "ipm")
    )
    dpi = float(jnp.abs(g_pd - g_ip).max())
    check("pdal and ipm smoothed d/dq agree", dpi < 1e-5, f"|dg|={dpi:.1e}")

    print("Active-set backend: forward only")
    a_sol = elastiqp.jax.solve(Qx, qx, Gx, hx, 10.0, A=Ax, b=bx, method="das")
    a_ref = elastiqp.solve(Qx, qx, Gx, hx, 10.0, A=Ax, b=bx, method="das")
    da = float(jnp.abs(a_sol.x - jnp.asarray(a_ref.x)).max())
    check(
        "as forward matches nanobind",
        int(a_sol.converged) == 1 and da < FFI_NB_TOL,
        f"|dx|={da:.1e}",
    )
    a_jit = jax.jit(
        lambda q_: elastiqp.jax.solve(Qx, q_, Gx, hx, 10.0, A=Ax, b=bx, method="das").x
    )(qx)
    check("as under jit", float(jnp.abs(a_jit - a_sol.x).max()) == 0.0, "")
    try:
        jax.grad(
            lambda q_: jnp.sum(
                elastiqp.jax.solve(Qx, q_, Gx, hx, 10.0, A=Ax, b=bx, method="das").x
            )
        )(qx)
        msg = None
    except TypeError as e:
        msg = str(e)
    check(
        "grad with method='das' raises, pointing at pdal/ipm",
        msg is not None and "pdal" in msg and "not differentiable" in msg,
        "",
    )

    print("Explicit warm start: drifting sequence, every backend")
    # The warm-started FFI call seeds a FRESH solver from the previous
    # Result (no persistent state), so it must (a) agree with the cold
    # solve, (b) save iterations against it, and (c) match the nanobind
    # solver's set_warm_start bit for bit (same C++). A large enough drift
    # so the working set actually changes between ticks (active-set
    # changes are what the warm start has to get right, not a no-op
    # re-verify).
    rng = np.random.default_rng(11)
    n, m, p, ticks = 20, 5, 70, 12
    Qw, q0, Aw, b0, Gw, h0, _ = random_qp(5, n, m, p)
    Gw, h0 = make_infeasible(Gw, h0, p // 4)
    Qw, Aw, Gw = jnp.asarray(Qw), jnp.asarray(Aw), jnp.asarray(Gw)
    drift = [
        (
            q0 + 0.2 * np.cumsum(rng.standard_normal((ticks, n)), 0)[k],
            h0 + 0.2 * np.cumsum(rng.standard_normal((ticks, p)), 0)[k],
            b0 + 0.05 * np.cumsum(rng.standard_normal((ticks, m)), 0)[k],
        )
        for k in range(ticks)
    ]
    for method in ("das", "pdal"):  # the ipm backend has no warm start
        nb_cls = {"das": elastiqp.das, "pdal": elastiqp.pdal}[method]
        prev = None
        warm_iters = cold_iters = 0
        worst_dx = worst_nb = 0.0
        all_conv = True
        ws_changed = 0
        for q_k, h_k, b_k in drift:
            cs = elastiqp.jax.solve(Qw, q_k, Gw, h_k, 10.0, A=Aw, b=b_k, method=method)
            ws = (
                elastiqp.jax.solve(
                    Qw, q_k, Gw, h_k, 10.0, A=Aw, b=b_k, method=method, warm_start=prev
                )
                if prev is not None
                else cs
            )
            if prev is not None:
                ws_changed += int(
                    np.count_nonzero(
                        (np.asarray(prev.z) > 1e-6) != (np.asarray(ws.z) > 1e-6)
                    )
                )
                warm_iters += int(ws.iters)
                cold_iters += int(cs.iters)
                worst_dx = max(worst_dx, float(jnp.abs(ws.x - cs.x).max()))
                # nanobind parity: same seed, same C++
                nb = nb_cls.Solver()
                nb.setup(
                    np.asarray(Qw),
                    q_k,
                    np.asarray(Gw),
                    h_k,
                    10.0,
                    A=np.asarray(Aw),
                    b=b_k,
                )
                x0, y0, z0 = (
                    np.asarray(prev.x),
                    np.asarray(prev.y),
                    np.asarray(prev.z),
                )
                nb.set_warm_start(x0, y0, z0)
                nbs = nb.solve()
                worst_nb = max(worst_nb, float(np.abs(nbs.x - np.asarray(ws.x)).max()))
            all_conv &= int(ws.converged) == 1 and int(cs.converged) == 1
            prev = ws
        check(
            f"[{method}] warm matches cold across {ticks} ticks",
            all_conv and worst_dx < 1e-4,
            f"|dx|={worst_dx:.1e} ws-flips={ws_changed}",
        )
        check(
            f"[{method}] warm saves iterations",
            warm_iters < cold_iters,
            f"{warm_iters} vs {cold_iters}",
        )
        check(
            f"[{method}] matches nanobind set_warm_start",
            worst_nb < FFI_NB_TOL,
            f"|dx|={worst_nb:.1e}",
        )
    check("warm sequence has working-set changes", ws_changed > 0, f"{ws_changed}")
    try:
        elastiqp.jax.solve(
            Qw, drift[0][0], Gw, drift[0][1], 10.0, A=Aw, b=drift[0][2],
            method="ipm", warm_start=prev,
        )
        ipm_raised = False
    except ValueError:
        ipm_raised = True
    check("warm_start with method='ipm' raises", ipm_raised, "")

    # Cross-backend seed: a PDAL result warm-starts the active set (and
    # vice versa); the tuple form is accepted too.
    q_k, h_k, b_k = drift[0]
    p_sol = elastiqp.jax.solve(Qw, q_k, Gw, h_k, 10.0, A=Aw, b=b_k, method="pdal")
    a_warm = elastiqp.jax.solve(
        Qw,
        q_k,
        Gw,
        h_k,
        10.0,
        A=Aw,
        b=b_k,
        method="das",
        warm_start=(p_sol.x, p_sol.y, p_sol.z),
    )
    a_cold = elastiqp.jax.solve(Qw, q_k, Gw, h_k, 10.0, A=Aw, b=b_k, method="das")
    dxa = float(jnp.abs(a_warm.x - a_cold.x).max())
    check(
        "das seeded from a pdal result (tuple form)",
        int(a_warm.converged) == 1
        and dxa < 1e-4
        and int(a_warm.iters) < int(a_cold.iters),
        f"|dx|={dxa:.1e} iters {int(a_warm.iters)} vs {int(a_cold.iters)}",
    )

    # Under jit + lax.scan with the Result as the carry: the shape JAX
    # users will write.
    qs = jnp.asarray(np.stack([d[0] for d in drift]))
    hs = jnp.asarray(np.stack([d[1] for d in drift]))
    bs = jnp.asarray(np.stack([d[2] for d in drift]))

    @jax.jit
    def rollout(qs, hs, bs):
        def step(state, data):
            q_k, h_k, b_k = data
            r = elastiqp.jax.solve(
                Qw, q_k, Gw, h_k, 10.0, A=Aw, b=b_k, warm_start=state
            )
            return r, (r.x, r.iters, r.converged)

        init = elastiqp.jax.solve(Qw, qs[0], Gw, hs[0], 10.0, A=Aw, b=bs[0])
        _, out = jax.lax.scan(step, init, (qs, hs, bs))
        return out

    xs_scan, it_scan, cv_scan = rollout(qs, hs, bs)
    xs_cold = np.stack(
        [
            np.asarray(elastiqp.jax.solve(Qw, d[0], Gw, d[1], 10.0, A=Aw, b=d[2]).x)
            for d in drift
        ]
    )
    dxs = float(np.abs(np.asarray(xs_scan) - xs_cold).max())
    check(
        "warm start as a lax.scan carry under jit",
        bool(cv_scan.all()) and dxs < 1e-4,
        f"|dx|={dxs:.1e} iters/tick={float(it_scan.mean()):.1f}",
    )

    # vmap over a batch of problems, each with its own warm state.
    v_cold = jax.vmap(
        lambda q_, h_, b_: elastiqp.jax.solve(Qw, q_, Gw, h_, 10.0, A=Aw, b=b_)
    )(qs, hs, bs)
    v_warm = jax.vmap(
        lambda q_, h_, b_, s: elastiqp.jax.solve(
            Qw, q_, Gw, h_, 10.0, A=Aw, b=b_, warm_start=s
        )
    )(qs, hs, bs, v_cold)
    dxv = float(jnp.abs(v_warm.x - v_cold.x).max())
    check(
        "warm start under vmap",
        bool((v_warm.converged == 1).all())
        and dxv < 1e-6
        and int(v_warm.iters.max()) <= 1,
        f"|dx|={dxv:.1e} max iters={int(v_warm.iters.max())}",
    )

    try:
        jax.grad(
            lambda q_: jnp.sum(
                elastiqp.jax.solve(
                    Qw, q_, Gw, h_k, 10.0, A=Aw, b=b_k, method="pdal", warm_start=p_sol
                ).x
            )
        )(q_k)
        msg = None
    except TypeError as e:
        msg = str(e)
    check(
        "grad with warm_start raises",
        msg is not None and "warm_start" in msg and "not differentiable" in msg,
        "",
    )
    try:
        elastiqp.jax.solve(
            Qw,
            q_k,
            Gw,
            h_k,
            10.0,
            A=Aw,
            b=b_k,
            warm_start=(p_sol.x, p_sol.y, p_sol.z[:-1]),
        )
        msg = None
    except ValueError as e:
        msg = str(e)
    check(
        "warm_start shape mismatch raises", msg is not None and "warm_start" in msg, ""
    )

    # The solver terminates on and returns unscaled quantities, so the VJP
    # (which differentiates the KKT at the returned point) must not see the
    # equilibration at all.
    def loss_rz(q_):
        s = elastiqp.jax.solve(
            args[0],
            q_,
            args[4],
            args[5],
            args[6],
            A=args[2],
            b=args[3],
            method="pdal",
            eps_abs=1e-11,
            max_iter=300,
            ruiz=True,
            target_kappa=kappa,
        )
        return w_loss @ s.x + w_t @ s.t

    g_rz = jax.grad(loss_rz)(args[1])
    dg = float(jnp.linalg.norm(g_rz - grads[1]))
    check("smoothed gradients invariant to ruiz", dg < 1e-5, f"|dg|={dg:.1e}")

    n_fail = RESULTS.count(False)
    print(f"\n{'All jax ffi tests passed.' if n_fail == 0 else f'{n_fail} FAILURES'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
