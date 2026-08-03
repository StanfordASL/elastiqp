"""ElastiQP python bindings test cases"""

import sys

import elastiqp
import numpy as np

print(f"testing {elastiqp.__file__}")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  {name:<52s} {detail:<24s} {'OK' if ok else 'FAIL'}")


def random_qp(seed, n, m, p):
    """Feasible QP with a known optimizer (strict problem)."""
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
    """Conflicting inequality pairs: g'x <= c and -g'x <= -(c+gap)."""
    G, h = G.copy(), h.copy()
    for k in range(n_conflicts):
        i, j = 2 * k, 2 * k + 1
        if j >= len(h):
            break
        G[j] = -G[i]
        h[j] = -(h[i] + gap)
    return G, h


def main():
    print("solve: feasible => known strict optimum, t ~ 0")
    Q, q, A, b, G, h, x_star = random_qp(1, 10, 0, 30)
    sol = elastiqp.solve(Q, q, G, h, 1e3)
    dx = np.abs(sol.x - x_star).max()
    check(
        "n=10 p=30",
        sol.converged == 1 and dx < 1e-5 and sol.t.max() < 1e-6,
        f"|dx|={dx:.1e}",
    )
    check("y empty without equalities", sol.y.shape == (0,), f"shape={sol.y.shape}")

    print("solve: infeasible => slacks activate")
    G2, h2 = make_infeasible(G, h, 5)
    sol = elastiqp.solve(Q, q, G2, h2, 10.0)
    check(
        "n=10 p=30 conflicting",
        sol.converged == 1 and sol.t.max() > 0.1,
        f"max_t={sol.t.max():.2f}",
    )

    print("solve with hard equalities")
    Q, q, A, b, G, h, x_star = random_qp(2, 14, 4, 60)
    sol = elastiqp.solve(Q, q, G, h, 1e3, A=A, b=b)
    dx = np.abs(sol.x - x_star).max()
    eq = np.abs(A @ sol.x - b).max()
    check(
        "feasible => strict optimum, Ax=b",
        sol.converged == 1 and dx < 1e-5 and eq < 1e-8 and sol.t.max() < 1e-6,
        f"|dx|={dx:.1e} eq={eq:.1e}",
    )
    G2, h2 = make_infeasible(G, h, 15)
    cold = elastiqp.solve(Q, q, G2, h2, 10.0, A=A, b=b)
    eq = np.abs(A @ cold.x - b).max()
    check(
        "infeasible ineqs => Ax=b still holds",
        cold.converged == 1 and eq < 1e-8 and cold.t.max() > 0.1,
        f"eq={eq:.1e} max_t={cold.t.max():.2f}",
    )
    check("y has equality-dual shape", cold.y.shape == (4,), f"shape={cold.y.shape}")

    print("per-constraint penalty vector overload")
    pen = np.where(np.arange(60) % 2 == 0, 5.0, 100.0)
    vsol = elastiqp.solve(Q, q, G2, h2, pen, A=A, b=b)
    check("vector penalty accepted", vsol.converged == 1, "")

    print("Solver class: cold solve == free function")
    solver = elastiqp.Solver()
    solver.setup(Q, q, G2, h2, 10.0, A=A, b=b)
    sol = solver.solve()
    dx = np.abs(sol.x - cold.x).max()
    check(
        "identical iterates",
        sol.converged == 1 and dx == 0.0 and sol.iters == cold.iters,
        f"|dx|={dx:.1e}",
    )
    check("status enum", sol.status == elastiqp.Status.Solved, str(sol.status))
    check(
        "solution scalars finite",
        np.isfinite(
            [sol.primal_obj, sol.primal_res, sol.dual_res, sol.duality_gap]
        ).all(),
        f"obj={sol.primal_obj:.3f}",
    )

    print("Solver: settings round-trip")
    solver2 = elastiqp.Solver()
    solver2.settings.eps_abs = 1e-6
    solver2.settings.eps_duality_gap_abs = 1e-6
    solver2.settings.warm_start = False
    check(
        "read back",
        solver2.settings.eps_abs == 1e-6
        and solver2.settings.warm_start is False
        and solver2.settings.max_outer_iter == 250
        and solver2.settings.max_iter_in == 1500
        and solver2.settings.mu_in_init == 1e-1
        and solver2.settings.ruiz is False,
        "",
    )
    # The C++ static_assert in test_pdal.cc pins these defaults; this checks
    # the other half -- that the bindings expose the shared block symmetrically,
    # since each settings class re-declares its fields by hand.
    shared = (
        "eps_abs",
        "eps_rel",
        "check_duality_gap",
        "eps_duality_gap_abs",
        "eps_duality_gap_rel",
        "max_factor_retries",
        "warm_start",
    )
    pdal_d, ipm_d = elastiqp.Settings(), elastiqp.IpmSettings()
    check(
        "shared termination defaults agree across backends",
        all(getattr(pdal_d, f) == getattr(ipm_d, f) for f in shared),
        f"{len(shared)} fields",
    )

    print("Solver: warm start across a drifting sequence (q,h,b)")
    rng = np.random.default_rng(7)
    n, m, p, ticks = 30, 8, 100, 10
    Q, q0, A, b0, G, h0, _ = random_qp(3, n, m, p)
    G, h0 = make_infeasible(G, h0, p // 4)
    warm = elastiqp.Solver()
    warm.setup(Q, q0, G, h0, 10.0, A=A, b=b0)
    cold_iters = warm_iters = 0
    worst_dx = worst_eq = 0.0
    all_conv = True
    q_k, h_k, b_k = q0.copy(), h0.copy(), b0.copy()
    first_sol = None
    first_x = None
    for _ in range(ticks):
        q_k += 0.01 * rng.standard_normal(n)
        h_k += 0.01 * rng.standard_normal(p)
        b_k += 0.01 * rng.standard_normal(m)
        warm.update(q=q_k, h=h_k, b=b_k)
        ws = warm.solve()
        if first_sol is None:
            first_sol = ws  # held across later solves on the same object
            first_x = ws.x.copy()
        cs = elastiqp.solve(Q, q_k, G, h_k, 10.0, A=A, b=b_k)
        all_conv &= ws.converged == 1 and cs.converged == 1
        warm_iters += ws.iters
        cold_iters += cs.iters
        worst_dx = max(worst_dx, np.abs(ws.x - cs.x).max())
        worst_eq = max(worst_eq, np.abs(A @ ws.x - b_k).max())
    check(
        "warm matches cold, equalities hold",
        all_conv and worst_dx < 1e-6 and worst_eq < 1e-8,
        f"|dx|={worst_dx:.1e} eq={worst_eq:.1e}",
    )
    check(
        "warm saves iterations",
        warm_iters < cold_iters,
        f"{warm_iters} vs {cold_iters}",
    )
    check(
        "held solution snapshot not invalidated",
        np.abs(first_sol.x - first_x).max() == 0.0,
        "",
    )

    print("Solver.update: equivalence with set_*, validation")
    Qu, qu, Au, bu, Gu, hu, _ = random_qp(5, 12, 3, 40)
    upd, setr = elastiqp.Solver(), elastiqp.Solver()
    for s in (upd, setr):
        s.setup(Qu, qu, Gu, hu, 10.0, A=Au, b=bu)
        s.solve()
    q_new = qu + 0.05
    h_new = hu + 0.05
    upd.update(q=q_new, h=h_new)
    setr.set_q(q_new)
    setr.set_h(h_new)
    su, ss = upd.solve(), setr.solve()
    check(
        "update(q=, h=) == set_q/set_h",
        su.converged == 1 and np.abs(su.x - ss.x).max() == 0.0 and su.iters == ss.iters,
        f"iters={su.iters}",
    )
    upd.update(penalty=25.0)
    setr.set_penalty(np.full(40, 25.0))
    su, ss = upd.solve(), setr.solve()
    check(
        "scalar penalty == vector penalty",
        su.converged == 1 and np.abs(su.x - ss.x).max() == 0.0,
        "",
    )

    def raises(exc, fn):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    check(
        "wrong-size q raises",
        raises(ValueError, lambda: upd.update(q=np.zeros(13))),
        "",
    )
    check(
        "wrong-shape G raises",
        raises(ValueError, lambda: upd.update(G=np.zeros((40, 13)))),
        "",
    )
    check(
        "wrong-size penalty vector raises",
        raises(ValueError, lambda: upd.update(penalty=np.zeros(39))),
        "",
    )
    no_eq = elastiqp.Solver()
    no_eq.setup(Qu, qu, Gu, hu, 10.0)
    check(
        "A/b update without equalities raises",
        raises(ValueError, lambda: no_eq.update(A=Au, b=bu)),
        "",
    )
    check(
        "update before setup raises",
        raises(RuntimeError, lambda: elastiqp.Solver().update(q=qu)),
        "",
    )
    check(
        "positional args rejected (keyword-only)",
        raises(TypeError, lambda: upd.update(Qu)),
        "",
    )
    upd.settings.warm_start = False  # cold solves: identical data => same x
    state = upd.solve()
    try:
        # h is invalid, so the (valid, different) q must not be applied.
        upd.update(q=q_new - 1.0, h=np.zeros(39))
    except ValueError:
        pass
    check(
        "failed update leaves data untouched",
        np.abs(upd.solve().x - state.x).max() == 0.0,
        "",
    )

    print("Solver: factorizations(), explicit set_warm_start, ruiz flag")
    Qp, qp_, Ap, bp, Gp, hp, xp = random_qp(11, 20, 5, 60)
    Gp, hp = make_infeasible(Gp, hp, 15)
    pdal = elastiqp.Solver()
    pdal.setup(Qp, qp_, Gp, hp, 10.0, A=Ap, b=bp)
    pdal.solve()
    check("factorizations() exposed", isinstance(pdal.factorizations(), int), "")
    check(
        "penalty - z_t - z_ineq == 0 (to rounding)",
        np.abs(
            10.0 - np.asarray(pdal.solution().z_t) - np.asarray(pdal.solution().z_ineq)
        ).max()
        < 1e-12,
        "",
    )
    seed = pdal.solution()
    pdal.settings.warm_start = False
    pdal.set_warm_start(np.asarray(seed.x), np.asarray(seed.y), np.asarray(seed.z_ineq))
    es = pdal.solve()
    check(
        "set_warm_start seeds the solve",
        es.converged == 1 and es.iters <= 2,
        f"iters={es.iters}",
    )
    rng = np.random.default_rng(3)
    scale = 10.0 ** rng.uniform(-3, 3, 60)
    rz = elastiqp.solve(
        Qp,
        qp_,
        Gp * scale[:, None],
        hp * scale,
        10.0 / scale,
        A=Ap,
        b=bp,
        eps_abs=1e-8,
        ruiz=True,
    )
    dx = np.abs(np.asarray(rz.x) - np.asarray(seed.x)).max()
    check(
        "ruiz=True solves the row-rescaled problem",
        rz.converged == 1 and dx < 1e-4,
        f"|dx|={dx:.1e}",
    )

    print("solve(backend=): dispatch and validation")
    ref = elastiqp.solve(Qp, qp_, Gp, hp, 10.0, A=Ap, b=bp, eps_abs=1e-8)
    ipm = elastiqp.solve(Qp, qp_, Gp, hp, 10.0, A=Ap, b=bp, eps_abs=1e-8, backend="ipm")
    dx = np.abs(np.asarray(ipm.x) - np.asarray(ref.x)).max()
    check(
        "backend='ipm' matches the default backend",
        ipm.converged == 1 and ref.converged == 1 and dx < 1e-5,
        f"|dx|={dx:.1e}",
    )
    check(
        "backend='ipm' holds equalities tighter",
        np.abs(Ap @ np.asarray(ipm.x) - bp).max() < 1e-10,
        f"eq={np.abs(Ap @ np.asarray(ipm.x) - bp).max():.1e}",
    )
    check(
        "unknown backend raises",
        raises(
            ValueError, lambda: elastiqp.solve(Qp, qp_, Gp, hp, 10.0, backend="nope")
        ),
        "",
    )
    rz_ipm = elastiqp.solve(
        Qp,
        qp_,
        Gp * scale[:, None],
        hp * scale,
        10.0 / scale,
        A=Ap,
        b=bp,
        eps_abs=1e-8,
        backend="ipm",
        ruiz=True,
    )
    dx = np.abs(np.asarray(rz_ipm.x) - np.asarray(seed.x)).max()
    check(
        "ruiz=True with backend='ipm' solves the row-rescaled problem",
        rz_ipm.converged == 1 and dx < 1e-4,
        f"|dx|={dx:.1e}",
    )

    print("IpmSolver: settings, warm start, kappa relaxation")
    ipm_solver = elastiqp.IpmSolver()
    ipm_solver.settings.eps_abs = 1e-8
    ipm_solver.settings.eps_duality_gap_abs = 1e-8
    check(
        "IpmSettings round-trip",
        ipm_solver.settings.eps_abs == 1e-8
        and ipm_solver.settings.max_iter == 250
        and ipm_solver.settings.warm_start is True
        and ipm_solver.settings.warm_start_fraction == 0.1,
        "",
    )
    ipm_solver.setup(Qp, qp_, Gp, hp, 10.0, A=Ap, b=bp)
    q_d, h_d, b_d = qp_.copy(), hp.copy(), bp.copy()
    warm_iters = cold_iters = 0
    worst = 0.0
    all_ok = True
    for _ in range(10):
        q_d = q_d + 0.01 * rng.standard_normal(20)
        h_d = h_d + 0.01 * rng.standard_normal(60)
        b_d = b_d + 0.01 * rng.standard_normal(5)
        ipm_solver.update(q=q_d, h=h_d, b=b_d)
        ws = ipm_solver.solve()
        cs = elastiqp.solve(
            Qp, q_d, Gp, h_d, 10.0, A=Ap, b=b_d, eps_abs=1e-8, backend="ipm"
        )
        all_ok &= ws.converged == 1 and cs.converged == 1
        warm_iters += ws.iters
        cold_iters += cs.iters
        worst = max(worst, np.abs(np.asarray(ws.x) - np.asarray(cs.x)).max())
    check(
        "warm beats cold and matches",
        all_ok and worst < 1e-4 and warm_iters < cold_iters,
        f"iters {warm_iters} vs {cold_iters}",
    )
    kappa = 1e-3
    rsol = ipm_solver.relax(kappa)
    comp = np.concatenate(
        [
            np.asarray(rsol.s_t) * np.asarray(rsol.z_t),
            np.asarray(rsol.s_ineq) * np.asarray(rsol.z_ineq),
        ]
    )
    check(
        "relaxed point on s.z = kappa hyperbola",
        rsol.converged == 1 and np.abs(comp - kappa).max() < 1e-8,
        f"comp_err={np.abs(comp - kappa).max():.1e}",
    )

    print("IpmSolver.warm_start_from: cross-backend handoff")
    fast = elastiqp.Solver()
    fast.setup(Qp, q_d, Gp, h_d, 10.0, A=Ap, b=b_d)
    fast_sol = fast.solve()
    hand = elastiqp.IpmSolver()
    hand.setup(Qp, q_d, Gp, h_d, 10.0, A=Ap, b=b_d)
    hand.warm_start_from(fast_sol)
    hs = hand.solve()
    fresh = elastiqp.solve(
        Qp, q_d, Gp, h_d, 10.0, A=Ap, b=b_d, eps_abs=1e-8, backend="ipm"
    )
    dx = np.abs(np.asarray(hs.x) - np.asarray(fresh.x)).max()
    check(
        "seeded IPM solve matches a cold one, in fewer iterations",
        hs.converged == 1 and dx < 1e-6 and hs.iters < fresh.iters,
        f"|dx|={dx:.1e} iters {hs.iters} vs {fresh.iters}",
    )
    kr = hand.relax(kappa)
    comp = np.concatenate(
        [
            np.asarray(kr.s_t) * np.asarray(kr.z_t),
            np.asarray(kr.s_ineq) * np.asarray(kr.z_ineq),
        ]
    )
    check(
        "relax() works off the handed-off solution",
        kr.converged == 1 and np.abs(comp - kappa).max() < 1e-8,
        f"comp_err={np.abs(comp - kappa).max():.1e}",
    )
    check(
        "dimension mismatch raises",
        raises(ValueError, lambda: elastiqp.IpmSolver().warm_start_from(fast_sol)),
        "",
    )

    print("Return types and shapes")
    s = elastiqp.solve(
        Q[:14, :14],
        q0[:14],
        np.eye(14),
        np.ones(14),
        5.0,
        A=np.zeros((0, 14)),
        b=np.zeros(0),
    )
    ok = (
        s.x.dtype == np.float64
        and s.x.shape == (14,)
        and s.y.shape == (0,)
        and all(v.shape == (14,) for v in (s.t, s.s_t, s.s_ineq, s.z_t, s.z_ineq))
        and isinstance(s.converged, int)
        and isinstance(s.iters, int)
        and s.status == elastiqp.Status.Solved
        and np.isfinite([s.primal_obj, s.primal_res, s.dual_res, s.duality_gap]).all()
    )
    check("Solution dtypes/shapes incl. empty equality block", ok, "")

    n_fail = RESULTS.count(False)
    print(f"\n{'All binding tests passed.' if n_fail == 0 else f'{n_fail} FAILURES'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
