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
# perturbing the FP environment / buffer alignment). The IPM walks a central
# path and stays at ULP level; the PDAL solver's exact line search and
# discrete active-set test amplify a last-bit difference to ~1e-11 over a
# dozen Newton steps -- still two orders under the 1e-8 solver tolerance,
# and each library is bit-deterministic with itself.
FFI_NB_TOL_PDAL = 1e-9
FFI_NB_TOL_IPM = 1e-12


def main():
    print("FFI vs nanobind: same C++ code, same answers")
    Q, q, A, b, G, h, x_star = random_qp(0, 14, 0, 40)
    sol = elastiqp.jax.solve(Q, q, G, h, 10.0)
    nb_sol = elastiqp.solve(Q, q, G, h, 10.0)
    dx = np.abs(np.asarray(sol.x) - nb_sol.x).max()
    check(
        "inequality-only n=14 p=40",
        int(sol.converged) == 1 and dx < FFI_NB_TOL_PDAL,
        f"|dx|={dx:.1e}",
    )

    Q, q, A, b, G, h, x_star = random_qp(1, 14, 4, 60)
    sol = elastiqp.jax.solve(Q, q, G, h, 1e3, A=A, b=b)
    nb_sol = elastiqp.solve(Q, q, G, h, 1e3, A=A, b=b)
    dx = np.abs(np.asarray(sol.x) - nb_sol.x).max()
    eq = np.abs(A @ np.asarray(sol.x) - b).max()
    check(
        "with equalities n=14 m=4 p=60",
        int(sol.converged) == 1 and dx < FFI_NB_TOL_PDAL and eq < 1e-8,
        f"|dx|={dx:.1e} eq={eq:.1e}",
    )
    check(
        "recovers known optimum",
        np.abs(np.asarray(sol.x) - x_star).max() < 1e-5,
        f"|dx*|={np.abs(np.asarray(sol.x) - x_star).max():.1e}",
    )

    isol = elastiqp.jax.solve(Q, q, G, h, 1e3, A=A, b=b, backend="ipm")
    inb = elastiqp.solve(Q, q, G, h, 1e3, A=A, b=b, backend="ipm")
    idx = np.abs(np.asarray(isol.x) - inb.x).max()
    check(
        "backend='ipm' agrees at ULP level",
        int(isol.converged) == 1 and idx < FFI_NB_TOL_IPM,
        f"|dx|={idx:.1e}",
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
        xs.shape == (8, 14) and dx < FFI_NB_TOL_PDAL,
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

    print("Gradients: implicit differentiation vs central finite differences")
    # Lightly-constrained instance: active set (2 eqs + a few active/elastic
    # ineqs) is smaller than n, so the solution is NOT pinned to a vertex and
    # every parameter (including q, Q, penalty) has a nonzero gradient. A
    # fully-pinned vertex solution would make x locally independent of q.
    # The seed is chosen for healthy strict-complementarity margins (~0.2):
    # implicit gradients are undefined at weakly-active constraints, and a
    # near-degenerate instance turns solver error into O(1e-3) gradient error.
    # Tight solver tolerance so FD noise (~eps_solve/step) stays small.
    Qf, qf, Af, bf, Gf, hf, _ = random_qp(22, 12, 2, 8)
    Gf, hf = make_infeasible(Gf, hf, 2)  # active elastic slacks
    pen = 10.0 + jnp.linspace(0.0, 5.0, 8)
    rngd = np.random.default_rng(17)
    w_loss = jnp.asarray(rngd.standard_normal(12))

    def loss(Q, q, A, b, G, h, penalty):
        s = elastiqp.jax.solve(
            Q, q, G, h, penalty, A=A, b=b, backend="ipm", eps_abs=1e-11, max_iter=300
        )
        return w_loss @ s.x + 0.1 * jnp.sum(s.t**2)

    args = tuple(jnp.asarray(v) for v in (Qf, qf, Af, bf, Gf, hf, pen))
    grads = jax.jit(jax.grad(loss, argnums=tuple(range(7))))(*args)

    step = 1e-5
    names = ["Q", "q", "A", "b", "G", "h", "penalty"]
    for k, name in enumerate(names):
        v = np.asarray(rngd.standard_normal(args[k].shape))
        v /= np.linalg.norm(v)
        pert = list(args)
        pert[k] = args[k] + step * v
        f_plus = float(loss(*pert))
        pert[k] = args[k] - step * v
        f_minus = float(loss(*pert))
        fd = (f_plus - f_minus) / (2 * step)
        an = float(jnp.sum(grads[k] * v))
        err = abs(fd - an) / max(1.0, abs(fd))
        # Require the gradient to be genuinely nonzero so the check has teeth.
        check(
            f"d/d{name} (directional, jit)",
            err < 1e-4 and abs(an) > 1e-4,
            f"an={an:+.5f} fd={fd:+.5f}",
        )

    # Gradient through vmap: sum of losses over a batch of shifted q's.
    batch_dq = jnp.asarray(0.05 * rngd.standard_normal((4, 12)))

    def batched_loss(q0):
        losses = jax.vmap(
            lambda dq: loss(
                args[0], q0 + dq, args[2], args[3], args[4], args[5], args[6]
            )
        )(batch_dq)
        return jnp.sum(losses)

    g_vmap = jax.grad(batched_loss)(args[1])
    v = np.asarray(rngd.standard_normal(12))
    v /= np.linalg.norm(v)
    fd = (
        float(batched_loss(args[1] + step * v))
        - float(batched_loss(args[1] - step * v))
    ) / (2 * step)
    an = float(jnp.sum(g_vmap * v))
    check(
        "d/dq through vmap",
        abs(fd - an) / max(1.0, abs(fd)) < 1e-4,
        f"an={an:+.5f} fd={fd:+.5f}",
    )

    print("Kappa-relaxed (smoothed) gradients")
    kappa = 1e-3

    # The smoothed gradient is by definition the derivative of the RELAXED
    # solution map x_r(theta) (the value returned is still the tight x). So
    # it must match finite differences of a loss evaluated on x_r, which we
    # access through the raw FFI call.
    def loss_relaxed(q_):
        out = elastiqp.jax._ffi_ipm_solve(
            args[0],
            q_,
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            1e-11,
            300,
            False,
            "sequential",
            kappa,
        )
        xr, tr = out[5], out[6]
        return w_loss @ xr + 0.1 * jnp.sum(tr**2)

    def loss_smooth(q_, kap=kappa):
        s = elastiqp.jax.solve(
            args[0],
            q_,
            args[4],
            args[5],
            args[6],
            A=args[2],
            b=args[3],
            backend="ipm",
            eps_abs=1e-11,
            max_iter=300,
            target_kappa=kap,
        )
        return w_loss @ s.x + 0.1 * jnp.sum(s.t**2)

    g_smooth = jax.jit(jax.grad(loss_smooth))(args[1])
    v = np.asarray(rngd.standard_normal(12))
    v /= np.linalg.norm(v)
    fd = (
        float(loss_relaxed(args[1] + step * v))
        - float(loss_relaxed(args[1] - step * v))
    ) / (2 * step)
    an = float(jnp.sum(g_smooth * v))
    check(
        "smoothed grad == FD of relaxed map",
        abs(fd - an) / max(1.0, abs(fd)) < 1e-4,
        f"an={an:+.5f} fd={fd:+.5f}",
    )

    # Value is unaffected by kappa: still the tight solution. Compare the
    # differentiation (fwd) path, which runs the relaxation, against the
    # value-only path, which skips it.
    v_smooth, _ = jax.value_and_grad(loss_smooth)(args[1])
    v_tight = loss(*args)
    dv = abs(float(v_smooth) - float(v_tight))
    check("value unchanged by kappa (fwd path)", dv == 0.0, f"|dv|={dv:.1e}")

    # kappa -> 0 recovers the exact gradient
    g_exact = jax.grad(loss, argnums=1)(*args)
    kappas = (1e-3, 1e-5, 1e-7)
    errs = [
        float(jnp.linalg.norm(jax.grad(loss_smooth)(args[1], kap) - g_exact))
        for kap in kappas
    ]
    check(
        "kappa -> 0 recovers exact gradient",
        errs[0] > errs[-1] and errs[-1] < 1e-5,
        " ".join(f"{e:.1e}" for e in errs),
    )

    # The smoothing payoff: on a near-degenerate instance (weakly active
    # constraint), the exact implicit gradient is ill-conditioned, but the
    # smoothed gradient still matches the relaxed map's true derivative.
    Qd, qd, Ad, bd, Gd, hd, _ = random_qp(9, 12, 2, 8)
    Gd, hd = make_infeasible(Gd, hd, 2)
    pend = 10.0 + jnp.linspace(0.0, 5.0, 8)
    argsd = tuple(jnp.asarray(vv) for vv in (Qd, qd, Ad, bd, Gd, hd, pend))

    def loss_relaxed_d(q_):
        out = elastiqp.jax._ffi_ipm_solve(
            argsd[0],
            q_,
            argsd[2],
            argsd[3],
            argsd[4],
            argsd[5],
            argsd[6],
            1e-11,
            300,
            False,
            "sequential",
            kappa,
        )
        return w_loss @ out[5] + 0.1 * jnp.sum(out[6] ** 2)

    def loss_smooth_d(q_):
        s = elastiqp.jax.solve(
            argsd[0],
            q_,
            argsd[4],
            argsd[5],
            argsd[6],
            A=argsd[2],
            b=argsd[3],
            backend="ipm",
            eps_abs=1e-11,
            max_iter=300,
            target_kappa=kappa,
        )
        return w_loss @ s.x + 0.1 * jnp.sum(s.t**2)

    g_d = jax.grad(loss_smooth_d)(argsd[1])
    fd = (
        float(loss_relaxed_d(argsd[1] + step * v))
        - float(loss_relaxed_d(argsd[1] - step * v))
    ) / (2 * step)
    an = float(jnp.sum(g_d * v))
    check(
        "smoothed grad well-conditioned at degenerate instance",
        abs(fd - an) / max(1.0, abs(fd)) < 1e-4,
        f"an={an:+.5f} fd={fd:+.5f}",
    )

    # The relaxation must actually reach the kappa-hyperbola, and its status
    # must be observable: info[2] reports the relax convergence (the backward
    # pass differentiates at that point, so a stalled relaxation would
    # otherwise mean silently wrong gradients). s1 = t and s2 = h + t - Gx
    # are the slacks of t >= 0 and Gx - t <= h.
    out = elastiqp.jax._ffi_ipm_solve(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        1e-11,
        300,
        False,
        "sequential",
        kappa,
    )
    xr, tr, z1r, z2r, info = out[5], out[6], out[8], out[9], out[10]
    s2r = args[5] + tr - args[4] @ xr
    comp = max(
        float(jnp.max(jnp.abs(tr * z1r - kappa))),
        float(jnp.max(jnp.abs(s2r * z2r - kappa))),
    )
    check(
        "relaxed point satisfies s.z = kappa, and reports so",
        comp < 1e-11 and int(info[2]) == 1,
        f"|s.z-k|={comp:.1e}",
    )

    # ...and a relaxation that cannot reach its target must not fail
    # silently: an absurd kappa under a tight iteration budget converges the
    # solve (info[0]) but stalls the relaxation (info[2]), and solve() folds
    # that into converged on the differentiated path.
    out = elastiqp.jax._ffi_ipm_solve(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        1e-11,
        60,
        False,
        "sequential",
        1e8,
    )
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
            backend="ipm",
            eps_abs=1e-11,
            max_iter=60,
            target_kappa=1e8,
        )
        return jnp.sum(s.x), s.converged

    (_, conv), _ = jax.value_and_grad(conv_smooth, has_aux=True)(args[1])
    check(
        "failed relaxation is reported via converged",
        tight_ok and relax_bad and int(conv) == 0,
        f"info={[int(v) for v in out[10]]} conv={int(conv)}",
    )

    # Smoothed gradient through vmap: the batched _ipm_bwd path (batched
    # einsums + batched saddle solve) with a relaxed evaluation point is
    # otherwise never exercised -- the plain vmap-grad test runs at kappa=0
    # and the kappa tests above are unbatched.
    qs_b = args[1] + batch_dq

    def batched_smooth(qs):
        return jnp.sum(jax.vmap(loss_smooth)(qs))

    g_b = jax.jit(jax.grad(batched_smooth))(qs_b)
    g_ref = jnp.stack([jax.grad(loss_smooth)(qs_b[i]) for i in range(4)])
    db = float(jnp.abs(g_b - g_ref).max())
    check(
        "smoothed d/dq through vmap matches per-problem",
        g_b.shape == (4, 12) and db < 1e-9,
        f"|dg|={db:.1e}",
    )

    # No equality constraints: the backward pass degenerates from an (n+m)
    # saddle system to a plain n x n solve.
    Q0, q0, _, _, G0, h0, _ = random_qp(30, 12, 0, 8)
    G0, h0 = make_infeasible(G0, h0, 2)
    pen0 = 10.0 + jnp.linspace(0.0, 5.0, 8)
    a0 = tuple(jnp.asarray(vv) for vv in (Q0, q0, G0, h0, pen0))

    def loss_m0(q_):
        s = elastiqp.jax.solve(
            a0[0], q_, a0[2], a0[3], a0[4], backend="ipm", eps_abs=1e-11, max_iter=300
        )
        return w_loss @ s.x + 0.1 * jnp.sum(s.t**2)

    g_m0 = jax.jit(jax.grad(loss_m0))(a0[1])
    v = np.asarray(rngd.standard_normal(12))
    v /= np.linalg.norm(v)
    fd = (float(loss_m0(a0[1] + step * v)) - float(loss_m0(a0[1] - step * v))) / (
        2 * step
    )
    an = float(jnp.sum(g_m0 * v))
    check(
        "d/dq with no equalities (m=0)",
        abs(fd - an) / max(1.0, abs(fd)) < 1e-4 and abs(an) > 1e-4,
        f"an={an:+.5f} fd={fd:+.5f}",
    )

    print("backend='pdal' (default): forward-only, and grad says so")
    Qx, qx, Ax, bx, Gx, hx, _ = random_qp(21, 14, 4, 60)
    Gx, hx = make_infeasible(Gx, hx, 15)
    ipm_ref = elastiqp.jax.solve(Qx, qx, Gx, hx, 10.0, A=Ax, b=bx, backend="ipm")
    ps = elastiqp.jax.solve(Qx, qx, Gx, hx, 10.0, A=Ax, b=bx)
    dx = float(jnp.abs(ps.x - ipm_ref.x).max())
    check(
        "default backend matches backend='ipm' on infeasible QP",
        int(ps.converged) == 1 and dx < 1e-5,
        f"|dx|={dx:.1e}",
    )

    jit_pdal = jax.jit(lambda q_: elastiqp.jax.solve(Qx, q_, Gx, hx, 10.0, A=Ax, b=bx))
    dj = float(jnp.abs(jit_pdal(qx).x - ps.x).max())
    check("jit matches eager (infeasible pdal)", dj == 0.0, f"|dx|={dj:.1e}")

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
    dr = float(jnp.abs(rs.x - ipm_ref.x).max())
    check(
        "ruiz=True on rescaled rows",
        int(rs.converged) == 1 and dr < 1e-4,
        f"|dx|={dr:.1e}",
    )

    # The whole point of wrapping the PDAL target in a custom_vjp: grad must
    # fail with a message that names the fix, not a generic JAX internal.
    try:
        jax.grad(
            lambda q_: jnp.sum(elastiqp.jax.solve(Qx, q_, Gx, hx, 10.0, A=Ax, b=bx).x)
        )(qx)
        msg = None
    except TypeError as e:
        msg = str(e)
    check("grad through backend='pdal' raises", msg is not None, "")
    check(
        "...with a message pointing at backend='ipm'",
        msg is not None and "backend='ipm'" in msg and "not differentiable" in msg,
        "",
    )

    def raises_value_error(fn):
        try:
            fn()
        except ValueError:
            return True
        except Exception:
            return False
        return False

    check(
        "unknown backend raises",
        raises_value_error(
            lambda: elastiqp.jax.solve(Qx, qx, Gx, hx, 10.0, backend="nope")
        ),
        "",
    )
    check(
        "target_kappa with backend='pdal' raises",
        raises_value_error(
            lambda: elastiqp.jax.solve(Qx, qx, Gx, hx, 10.0, target_kappa=1e-3)
        ),
        "",
    )
    rs_ipm = elastiqp.jax.solve(
        Qx,
        qx,
        Gx * scale[:, None],
        hx * scale,
        10.0 / scale,
        A=Ax,
        b=bx,
        backend="ipm",
        ruiz=True,
    )
    dri = float(jnp.abs(rs_ipm.x - ipm_ref.x).max())
    check(
        "ruiz=True with backend='ipm' on rescaled rows",
        int(rs_ipm.converged) == 1 and dri < 1e-4,
        f"|dx|={dri:.1e}",
    )

    # Both backends terminate on and return unscaled quantities, so the VJP
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
            backend="ipm",
            eps_abs=1e-11,
            max_iter=300,
            ruiz=True,
        )
        return w_loss @ s.x + 0.1 * jnp.sum(s.t**2)

    g_rz = jax.grad(loss_rz)(args[1])
    dg = float(jnp.linalg.norm(g_rz - grads[1]))
    check("ipm gradients invariant to ruiz", dg < 1e-5, f"|dg|={dg:.1e}")

    n_fail = RESULTS.count(False)
    print(f"\n{'All jax ffi tests passed.' if n_fail == 0 else f'{n_fail} FAILURES'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
