"""ElastiQP PyTorch interface test cases"""

import sys

import elastiqp
import elastiqp.torch
import numpy as np
import torch

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


T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))
maxabs = lambda a: float(torch.as_tensor(a).abs().max())


def main():
    print("torch vs nanobind: same C++ code, same answers (every backend)")
    Q, q, A, b, G, h, x_star = random_qp(0, 14, 0, 40)
    for method in elastiqp.torch.METHODS:
        sol = elastiqp.torch.solve(Q, q, G, h, 10.0, method=method)
        nb_sol = elastiqp.solve(Q, q, G, h, 10.0, method=method)
        dx = maxabs(sol.x - T(nb_sol.x))
        # Same shared object, same thread: bit-identical.
        check(
            f"inequality-only n=14 p=40 [{method}]",
            int(sol.converged) == 1 and dx == 0.0 and sol.x.dtype == torch.float64,
            f"|dx|={dx:.1e}",
        )
    sol = elastiqp.torch.solve(Q, q, G, h, 10.0)
    check("y empty without equalities", tuple(sol.y.shape) == (0,), "")

    Q, q, A, b, G, h, x_star = random_qp(1, 14, 4, 60)
    sol = elastiqp.torch.solve(Q, q, G, h, 1e3, A=A, b=b, eps_abs=1e-8)
    nb_sol = elastiqp.solve(Q, q, G, h, 1e3, A=A, b=b, eps_abs=1e-8)
    dx = maxabs(sol.x - T(nb_sol.x))
    eq = maxabs(T(A) @ sol.x - T(b))
    check(
        "with equalities n=14 m=4 p=60",
        int(sol.converged) == 1 and dx == 0.0 and eq < 1e-8,
        f"|dx|={dx:.1e} eq={eq:.1e}",
    )
    check(
        "recovers known optimum",
        maxabs(sol.x - T(x_star)) < 1e-5,
        f"|dx*|={maxabs(sol.x - T(x_star)):.1e}",
    )

    print("Infeasible inequalities: equalities hold, slacks activate")
    G2, h2 = make_infeasible(G, h, 15)
    sol = elastiqp.torch.solve(Q, q, G2, h2, 10.0, A=A, b=b)
    eq = maxabs(T(A) @ sol.x - T(b))
    check(
        "Ax=b under conflict",
        int(sol.converged) == 1 and eq < 1e-8 and float(sol.t.max()) > 0.1,
        f"eq={eq:.1e} max_t={float(sol.t.max()):.2f}",
    )

    print("Input handling")
    s32 = elastiqp.torch.solve(
        T(Q).float(), T(q).float(), T(G2).float(), T(h2).float(), 10.0,
        A=T(A).float(), b=T(b).float(),
    )
    ref32 = elastiqp.solve(
        T(Q).float().double().numpy(), T(q).float().double().numpy(),
        T(G2).float().double().numpy(), T(h2).float().double().numpy(), 10.0,
        A=T(A).float().double().numpy(), b=T(b).float().double().numpy(),
    )
    check(
        "float32 inputs upcast, float64 out",
        s32.x.dtype == torch.float64 and maxabs(s32.x - T(ref32.x)) == 0.0,
        "",
    )
    pen_vec = torch.full((60,), 10.0, dtype=torch.float64)
    sv = elastiqp.torch.solve(Q, q, G2, h2, pen_vec, A=A, b=b)
    check("vector penalty matches scalar", maxabs(sv.x - sol.x) == 0.0, "")
    try:
        elastiqp.torch.solve(Q, q, G2, h2, 10.0, A=A)
        bad = False
    except ValueError:
        bad = True
    check("A without b raises", bad, "")

    print("Batched: leading batch dims broadcast, one solve per entry")
    rng = np.random.default_rng(5)
    batch_q = T(q + 0.05 * rng.standard_normal((8, len(q))))
    sb = elastiqp.torch.solve(Q, batch_q, G2, h2, 10.0, A=A, b=b)
    ref = np.stack(
        [
            elastiqp.solve(Q, batch_q[i].numpy(), G2, h2, 10.0, A=A, b=b).x
            for i in range(8)
        ]
    )
    dx = maxabs(sb.x - T(ref))
    check(
        "batch of 8 (shared Q) matches per-problem solves",
        tuple(sb.x.shape) == (8, 14) and tuple(sb.converged.shape) == (8,) and dx == 0.0,
        f"|dx|={dx:.1e}",
    )
    # Nested batch dims, and batched Q too.
    Qb = T(Q).expand(2, 4, 14, 14)
    sn = elastiqp.torch.solve(Qb, batch_q.reshape(2, 4, 14), G2, h2, 10.0, A=A, b=b)
    check(
        "nested batch (2, 4)",
        tuple(sn.x.shape) == (2, 4, 14) and maxabs(sn.x.reshape(8, 14) - sb.x) == 0.0,
        "",
    )

    print("torch.vmap")
    xs = torch.vmap(lambda q_: elastiqp.torch.solve(Q, q_, G2, h2, 10.0, A=A, b=b).x)(
        batch_q
    )
    check("vmap matches batched", maxabs(xs - sb.x) == 0.0, "")

    print("torch.compile: the solve is an opaque custom op")

    @torch.compile(fullgraph=True)
    def controller(q_k, h_k, b_k):
        s = elastiqp.torch.solve(Q, q_k, G2, h_k, 10.0, A=A, b=b_k)
        return s.x, s.converged

    x_c, conv = controller(T(q), T(h2), T(b))
    dx = maxabs(x_c - sol.x)
    check("compile(fullgraph) matches eager", int(conv) == 1 and dx == 0.0, f"|dx|={dx:.1e}")

    print("Composability: solver output feeds torch computation")
    obj = 0.5 * sol.x @ T(Q) @ sol.x + T(q) @ sol.x + 10.0 * sol.t.sum()
    x_np = sol.x.numpy()
    obj_ref = 0.5 * x_np @ Q @ x_np + q @ x_np + 10.0 * sol.t.numpy().sum()
    check("objective matches", abs(float(obj) - obj_ref) < 1e-9, f"obj={float(obj):.4f}")

    print("Smoothed gradients vs finite differences of the relaxed map")
    # See tests/test_jax_ffi.py for the rationale (instance choice, linear
    # loss, FD steps): identical setup, so the two interfaces are pinned to
    # the same ground truth.
    kappa = 1e-3
    Qf, qf, Af, bf, Gf, hf, _ = random_qp(22, 12, 2, 8)
    Gf, hf = make_infeasible(Gf, hf, 2)  # active elastic slacks
    pen = 10.0 + np.linspace(0.0, 5.0, 8)
    rngd = np.random.default_rng(17)
    w_loss = T(rngd.standard_normal(12))
    w_t = T(0.3 * rngd.standard_normal(8))
    args = tuple(T(v) for v in (Qf, qf, Af, bf, Gf, hf, pen))

    def loss_smooth(Q_, q_, A_, b_, G_, h_, penalty_, kap=kappa, method="pdal", **kw):
        s = elastiqp.torch.solve(
            Q_, q_, G_, h_, penalty_, A=A_, b=b_, method=method,
            eps_abs=1e-11, max_iter=300, target_kappa=kap, **kw,
        )
        return s.x @ w_loss + s.t @ w_t

    def loss_relaxed(Q_, q_, A_, b_, G_, h_, penalty_, kap=kappa):
        out = torch.ops.elastiqp.solve(
            Q_, q_, A_, b_, G_, h_, penalty_, 1e-11, 300, False, kap, "pdal"
        )
        xr, tr = out[5], out[6]
        return xr @ w_loss + tr @ w_t

    leaves = tuple(a.clone().requires_grad_(True) for a in args)
    loss_smooth(*leaves).backward()
    grads = tuple(l.grad for l in leaves)

    step_r = 1e-4
    step = 1e-5
    names = ["Q", "q", "A", "b", "G", "h", "penalty"]
    for k, name in enumerate(names):
        v = T(rngd.standard_normal(tuple(args[k].shape)))
        v /= torch.linalg.norm(v)
        pert = list(args)
        pert[k] = args[k] + step_r * v
        f_plus = float(loss_relaxed(*pert))
        pert[k] = args[k] - step_r * v
        f_minus = float(loss_relaxed(*pert))
        fd = (f_plus - f_minus) / (2 * step_r)
        an = float((grads[k] * v).sum())
        err = abs(fd - an) / max(1.0, abs(fd))
        check(
            f"d/d{name} (directional)",
            err < 1e-4 and abs(an) > 1e-4,
            f"an={an:+.5f} fd={fd:+.5f}",
        )

    # Only the leaves that require grad get one; the rest are untouched.
    q_only = args[1].clone().requires_grad_(True)
    loss_smooth(args[0], q_only, *args[2:]).backward()
    check(
        "grad w.r.t. q alone matches the all-leaves run",
        maxabs(q_only.grad - grads[1]) < 1e-12,
        "",
    )

    print("kappa -> 0: smoothed gradients converge to the tight derivative")
    v = T(rngd.standard_normal(12))
    v /= torch.linalg.norm(v)

    def loss_tight(q_):
        with torch.no_grad():
            s = elastiqp.torch.solve(
                args[0], q_, args[4], args[5], args[6], A=args[2], b=args[3],
                method="pdal", eps_abs=1e-11, max_iter=300,
            )
        return s.x @ w_loss + s.t @ w_t

    fd_tight = (
        float(loss_tight(args[1] + step * v)) - float(loss_tight(args[1] - step * v))
    ) / (2 * step)

    def g_q(kap, base=args, **kw):
        q_ = base[1].clone().requires_grad_(True)
        loss_smooth(base[0], q_, *base[2:], kap=kap, **kw).backward()
        return q_.grad

    errs = [abs(float((g_q(kap) * v).sum()) - fd_tight) for kap in (1e-3, 1e-5, 1e-7)]
    check(
        "directional error decreases in kappa, -> 0",
        errs[0] > errs[-1] and errs[-1] < 1e-4,
        " ".join(f"{e:.1e}" for e in errs),
    )

    # Near-degenerate instance: smoothed gradient still matches the relaxed map.
    Qd, qd, Ad, bd, Gd, hd, _ = random_qp(9, 12, 2, 8)
    Gd, hd = make_infeasible(Gd, hd, 2)
    argsd = tuple(T(vv) for vv in (Qd, qd, Ad, bd, Gd, hd, pen))
    g_d = g_q(kappa, base=argsd)
    fd = (
        float(loss_relaxed(argsd[0], argsd[1] + step_r * v, *argsd[2:]))
        - float(loss_relaxed(argsd[0], argsd[1] - step_r * v, *argsd[2:]))
    ) / (2 * step_r)
    an = float((g_d * v).sum())
    check(
        "smoothed grad well-conditioned at degenerate instance",
        abs(fd - an) / max(1.0, abs(fd)) < 1e-4,
        f"an={an:+.5f} fd={fd:+.5f}",
    )

    # Value is the tight solution whether or not we differentiate.
    q_ = args[1].clone().requires_grad_(True)
    v_smooth = float(loss_smooth(args[0], q_, *args[2:]).detach())
    v_tight = float(loss_tight(args[1]))
    check("value unchanged by kappa (grad path)", v_smooth == v_tight, f"|dv|={abs(v_smooth - v_tight):.1e}")

    # The relaxation runs only when differentiating: a no-grad solve and a
    # grad-enabled solve report the same iters, and the relaxed block of the
    # raw op is a copy of the tight block at kappa = 0.
    out0 = torch.ops.elastiqp.solve(*args, 1e-11, 300, False, 0.0, "pdal")
    check(
        "kappa=0: relaxed block == tight block",
        all(torch.equal(out0[i], out0[i + 5]) for i in range(5)),
        "",
    )
    out = torch.ops.elastiqp.solve(*args, 1e-11, 300, False, kappa, "pdal")
    xr, tr, z1r, z2r, info = out[5], out[6], out[8], out[9], out[10]
    s2r = args[5] + tr - args[4] @ xr
    comp = max(maxabs(tr * z1r - kappa), maxabs(s2r * z2r - kappa))
    check(
        "relaxed point satisfies s.z = kappa, and reports so",
        comp < 1e-13 and int(info[2]) == 1,
        f"|s.z-k|={comp:.1e}",
    )

    # A stalled relaxation is reported through converged on the grad path.
    out = torch.ops.elastiqp.solve(*args, 1e-11, 300, False, 1e8, "pdal")
    tight_ok, relax_bad = float(out[10][0]) == 1.0, float(out[10][2]) == 0.0
    q_ = args[1].clone().requires_grad_(True)
    s = elastiqp.torch.solve(
        args[0], q_, args[4], args[5], args[6], A=args[2], b=args[3],
        method="pdal", eps_abs=1e-11, max_iter=300, target_kappa=1e8,
    )
    check(
        "failed relaxation is reported via converged",
        tight_ok and relax_bad and int(s.converged) == 0,
        f"info={[int(vv) for vv in out[10]]} conv={int(s.converged)}",
    )

    print("Batched gradients")
    batch_dq = T(0.05 * rngd.standard_normal((4, 12)))
    qs_b = (args[1] + batch_dq).requires_grad_(True)
    loss_smooth(args[0], qs_b, *args[2:]).sum().backward()
    g_b = qs_b.grad
    g_ref = torch.stack([g_q(kappa, base=(args[0], qs_b[i].detach()) + args[2:]) for i in range(4)])
    db = maxabs(g_b - g_ref)
    check(
        "batched d/dq matches per-problem",
        tuple(g_b.shape) == (4, 12) and db < 1e-9,
        f"|dg|={db:.1e}",
    )
    batched_relaxed = lambda qs: sum(
        float(loss_relaxed(args[0], qs[i], *args[2:])) for i in range(4)
    )
    fd_b = (
        batched_relaxed(qs_b.detach() + step_r * v) - batched_relaxed(qs_b.detach() - step_r * v)
    ) / (2 * step_r)
    an_b = float((g_b * v).sum())
    check(
        "batched d/dq matches FD of the relaxed losses",
        abs(fd_b - an_b) / max(1.0, abs(fd_b)) < 1e-4,
        f"an={an_b:+.5f} fd={fd_b:+.5f}",
    )
    # ...and autograd through the vmap rule, and the functional transforms.
    qs_v = qs_b.detach().clone().requires_grad_(True)
    torch.vmap(lambda q_: loss_smooth(args[0], q_, *args[2:]))(qs_v).sum().backward()
    check("backward through vmap matches batched", maxabs(qs_v.grad - g_b) < 1e-9, "")
    # torch.func transforms run the torch-native VJP (LU in torch instead
    # of Eigen), so agreement with the C++ path is to round-off, not bits.
    # All seven data gradients are exercised, including m=0 below.
    g_f = torch.func.grad(loss_smooth, argnums=tuple(range(7)))(*args)
    df = max(maxabs(a - c) for a, c in zip(g_f, grads))
    check("torch.func.grad (all args) matches backward", df < 1e-9, f"|dg|={df:.1e}")
    g_v = torch.vmap(
        torch.func.grad(lambda q_: loss_smooth(args[0], q_, *args[2:]))
    )(qs_b.detach())
    check("vmap(grad) matches batched backward", maxabs(g_v - g_b) < 1e-9, "")
    solve_x = lambda q_: elastiqp.torch.solve(
        args[0], q_, args[4], args[5], args[6], A=args[2], b=args[3],
        method="pdal", eps_abs=1e-11, max_iter=300,
    ).x
    J = torch.func.jacrev(solve_x)(args[1])
    q_j = args[1].clone().requires_grad_(True)
    (solve_x(q_j) @ w_loss).backward()
    check(
        "jacrev matches VJP",
        tuple(J.shape) == (12, 12) and maxabs(w_loss @ J - q_j.grad) < 1e-9,
        "",
    )

    # Gradient under torch.compile.
    q_c = args[1].clone().requires_grad_(True)
    torch.compile(lambda q_: loss_smooth(args[0], q_, *args[2:]))(q_c).backward()
    check("compiled backward matches eager", maxabs(q_c.grad - grads[1]) < 1e-12, "")

    # No equality constraints: (n+m) saddle system degenerates to n x n.
    Q0, q0, _, _, G0, h0, _ = random_qp(30, 12, 0, 8)
    G0, h0 = make_infeasible(G0, h0, 2)
    a0 = tuple(T(vv) for vv in (Q0, q0, np.zeros((0, 12)), np.zeros(0), G0, h0, pen))
    g_m0 = g_q(kappa, base=a0)
    fd = (
        float(loss_relaxed(a0[0], a0[1] + step_r * v, *a0[2:]))
        - float(loss_relaxed(a0[0], a0[1] - step_r * v, *a0[2:]))
    ) / (2 * step_r)
    an = float((g_m0 * v).sum())
    check(
        "smoothed d/dq with no equalities (m=0)",
        abs(fd - an) / max(1.0, abs(fd)) < 1e-4 and abs(an) > 1e-4,
        f"an={an:+.5f} fd={fd:+.5f}",
    )
    g_f0 = torch.func.grad(lambda q_: loss_smooth(a0[0], q_, *a0[2:]))(a0[1])
    check("torch.func.grad with no equalities (m=0)", maxabs(g_f0 - g_m0) < 1e-9, "")

    print("Infeasible QP: ruiz; grad needs kappa")
    Qx, qx, Ax, bx, Gx, hx, _ = random_qp(21, 14, 4, 60)
    Gx, hx = make_infeasible(Gx, hx, 15)
    nb_ref = elastiqp.solve(Qx, qx, Gx, hx, 10.0, A=Ax, b=bx)
    scale = 10.0 ** np.random.default_rng(6).uniform(-3, 3, 60)
    rs = elastiqp.torch.solve(
        Qx, qx, Gx * scale[:, None], hx * scale, 10.0 / scale, A=Ax, b=bx, ruiz=True
    )
    dr = maxabs(rs.x - T(nb_ref.x))
    check("ruiz=True on rescaled rows", int(rs.converged) == 1 and dr < 1e-4, f"|dx|={dr:.1e}")

    try:
        q_ = T(qx).requires_grad_(True)
        elastiqp.torch.solve(Qx, q_, Gx, hx, 10.0, A=Ax, b=bx, method="pdal", target_kappa=0.0)
        msg = None
    except TypeError as e:
        msg = str(e)
    check("grad at kappa=0 raises", msg is not None, "")
    check(
        "...with a message pointing at target_kappa",
        msg is not None and "target_kappa" in msg and "not differentiable" in msg,
        "",
    )
    with torch.no_grad():
        s0 = elastiqp.torch.solve(Qx, q_, Gx, hx, 10.0, A=Ax, b=bx, method="pdal", target_kappa=0.0)
    check("...but solves fine under no_grad", int(s0.converged) == 1, "")

    for method in ("pdal", "ipm"):
        q_ = T(qx).requires_grad_(True)
        elastiqp.torch.solve(Qx, q_, Gx, hx, 10.0, A=Ax, b=bx, method=method).x.sum().backward()
        check(f"grad with default target_kappa is finite [{method}]", bool(torch.isfinite(q_.grad).all()), "")
    g_ipm = g_q(kappa, method="ipm")
    dpi = maxabs(g_ipm - grads[1])
    check("pdal and ipm smoothed d/dq agree", dpi < 1e-5, f"|dg|={dpi:.1e}")

    print("Active-set backend: forward only")
    a_sol = elastiqp.torch.solve(Qx, qx, Gx, hx, 10.0, A=Ax, b=bx, method="das")
    a_ref = elastiqp.solve(Qx, qx, Gx, hx, 10.0, A=Ax, b=bx, method="das")
    check("as forward matches nanobind", int(a_sol.converged) == 1 and maxabs(a_sol.x - T(a_ref.x)) == 0.0, "")
    with torch.no_grad():
        a_b = elastiqp.torch.solve(Qx, T(qx).expand(3, -1), Gx, hx, 10.0, A=Ax, b=bx, method="das")
    check("as batched", tuple(a_b.x.shape) == (3, 14) and maxabs(a_b.x[1] - a_sol.x) == 0.0, "")
    try:
        q_ = T(qx).requires_grad_(True)
        elastiqp.torch.solve(Qx, q_, Gx, hx, 10.0, A=Ax, b=bx, method="das")
        msg = None
    except TypeError as e:
        msg = str(e)
    check(
        "grad with method='das' raises, pointing at pdal/ipm",
        msg is not None and "pdal" in msg and "not differentiable" in msg,
        "",
    )

    g_rz = g_q(kappa, ruiz=True)
    dg = float(torch.linalg.norm(g_rz - grads[1]))
    check("smoothed gradients invariant to ruiz", dg < 1e-5, f"|dg|={dg:.1e}")

    n_fail = RESULTS.count(False)
    print(f"\n{'All torch tests passed.' if n_fail == 0 else f'{n_fail} FAILURES'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
