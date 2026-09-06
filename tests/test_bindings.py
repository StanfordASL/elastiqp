"""ElastiQP python bindings test cases"""

import os
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
    # eq < 1e-8 needs a tighter solve than the 1e-5 default.
    sol = elastiqp.solve(Q, q, G, h, 1e3, A=A, b=b, eps_abs=1e-8)
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
    print("Solver: warm start across a drifting sequence (q,h,b)")
    rng = np.random.default_rng(7)
    n, m, p, ticks = 30, 8, 100, 10
    Q, q0, A, b0, G, h0, _ = random_qp(3, n, m, p)
    G, h0 = make_infeasible(G, h0, p // 4)
    warm = elastiqp.Solver()
    # The warm-vs-cold agreement threshold (1e-6) needs tighter solves
    # than the 1e-5 default.
    warm.settings.eps_abs = 1e-8
    warm.settings.eps_duality_gap_abs = 1e-8
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
        cs = elastiqp.solve(Q, q_k, G, h_k, 10.0, A=A, b=b_k, eps_abs=1e-8)
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
    # Ruiz scaling is computed at setup(); matrix updates keep it (exact,
    # drifting), and solve() re-equilibrates past settings.ruiz_refresh_ratio
    # without losing the warm start. reequilibrate() is the manual form.
    rq = elastiqp.Solver()
    rq.settings.ruiz = True
    rq.settings.eps_abs = 1e-8
    rq.setup(Qp, qp_, Gp, hp, 10.0, A=Ap, b=bp)
    rq.solve()
    drift0 = rq.scaling_drift()
    rq.set_G(Gp * scale[:, None])
    rq.set_h(hp * scale)
    rq.set_penalty(10.0 / scale)
    drift1 = rq.scaling_drift()
    rs = rq.solve()
    drift2 = rq.scaling_drift()
    dx = np.abs(np.asarray(rs.x) - np.asarray(seed.x)).max()
    check(
        "solve() auto re-equilibrates after a drifting set_G",
        abs(drift0 - 1.0) < 0.01
        and drift1 > rq.settings.ruiz_refresh_ratio
        and drift2 <= rq.settings.ruiz_refresh_ratio
        and rs.converged == 1
        and dx < 1e-4,
        f"drift {drift0:.2f} -> {drift1:.1f} -> {drift2:.2f} |dx|={dx:.1e}",
    )
    rq.reequilibrate()
    check("reequilibrate() no-op when equilibrated", rq.solve().iters == 0, "")
    # Objective scale through set_Q: x invariant, duals scale, refresh fires;
    # and the relax() warm iterate is carried through the refresh
    rq.set_G(Gp)
    rq.set_h(hp)
    rq.set_penalty(np.full(60, 10.0))
    base = rq.solve()
    rbase = rq.relax(1e-3)
    alpha = 1e5
    rq.set_Q(alpha * Qp)
    rq.set_q(alpha * qp_)
    rq.set_penalty(np.full(60, alpha * 10.0))
    drift_q = rq.scaling_drift()
    rs = rq.solve()
    rr = rq.relax(1e-3)
    fresh = elastiqp.Solver()
    fresh.settings.ruiz = True
    fresh.settings.eps_abs = 1e-8
    fresh.setup(alpha * Qp, alpha * qp_, Gp, hp, alpha * 10.0, A=Ap, b=bp)
    fresh.solve()
    fr = fresh.relax(1e-3)
    dx = np.abs(np.asarray(rs.x) - np.asarray(base.x)).max()
    dz = np.abs(np.asarray(rs.z_ineq) - alpha * np.asarray(base.z_ineq)).max() / alpha
    rdx = np.abs(np.asarray(rr.x) - np.asarray(fr.x)).max()
    check(
        "set_Q objective scale: refresh, x invariant, duals scale",
        drift_q > rq.settings.ruiz_refresh_ratio
        and rq.scaling_drift() <= rq.settings.ruiz_refresh_ratio
        and rs.converged == 1
        and dx < 1e-5
        and dz < 1e-5,
        f"drift {drift_q:.1e} |dx|={dx:.1e} |dz|/a={dz:.1e}",
    )
    check(
        "relax() through the refresh matches fresh setup",
        rbase.converged == 1 and rr.converged == 1 and fr.converged == 1 and rdx < 1e-5,
        f"|dx|={rdx:.1e}",
    )

    print("Solver.relax: kappa relaxation (smoothed differentiation point)")
    kappa = 1e-3
    tight = pdal.solution()
    rsol = pdal.relax(kappa)
    comp = np.concatenate(
        [
            np.asarray(rsol.s_t) * np.asarray(rsol.z_t),
            np.asarray(rsol.s_ineq) * np.asarray(rsol.z_ineq),
        ]
    )
    moved = np.abs(np.asarray(rsol.x) - np.asarray(tight.x)).max()
    check(
        "relaxed point on s.z = kappa hyperbola",
        rsol.converged == 1 and np.abs(comp - kappa).max() < 1e-10 and moved > 0.0,
        f"comp_err={np.abs(comp - kappa).max():.1e}",
    )
    pdal.settings.warm_start = True  # warm re-solve from the tight iterate
    again = pdal.solve()
    check(
        "relax leaves the tight iterate untouched",
        again.converged == 1 and again.iters <= 2,
        f"iters={again.iters}",
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

    print("Hard case: warm-start stale duals (OSCBF Panda teleop, step 8945)")
    # Captured pathology: a primal-feasible warm start carrying oversized
    # duals from a constraint-conflict transition. Residuals pass on round
    # one; the duality gap decays at ~0.78/round -- under the former
    # 0.8-stall gate -- so bcl_release_jump never fired and the warm solve took
    # 30 iters vs 13 cold. The projected-horizon gate (bcl_release_jump_horizon)
    # fires on round one instead: 5 iters measured (bounds hold 2x headroom).
    d = np.load(os.path.join(os.path.dirname(__file__), "data", "warmstart_stale_duals.npz"))
    tol = float(d["solver_tol"])

    def hard_case_solver():
        s = elastiqp.Solver()
        s.settings.eps_abs = tol
        s.settings.eps_duality_gap_abs = tol
        s.setup(d["P"], d["q"], d["G"], d["h"], d["penalties"])
        return s

    cold = hard_case_solver().solve()
    check(
        "cold reference converges",
        cold.converged == 1 and cold.iters <= 26,
        f"iters={cold.iters}",
    )
    ws = hard_case_solver()
    ws.set_warm_start(d["warm_x"], d["warm_y"], d["warm_z_ineq"])
    warm = ws.solve()
    dx = np.abs(warm.x - d["expected_x"]).max()
    check(
        "warm solve converges accurately",
        warm.converged == 1 and dx < 10 * tol,
        f"|dx|={dx:.1e}",
    )
    check(
        "warm release jump beats logged 30 iters",
        warm.iters <= 10 and warm.iters < cold.iters,
        f"iters={warm.iters} (logged 30)",
    )
    # Informational: the repro still discriminates with the jump off.
    off = hard_case_solver()
    off.settings.bcl_release_jump = False
    off.set_warm_start(d["warm_x"], d["warm_y"], d["warm_z_ineq"])
    noj = off.solve()
    print(f"  [info] bcl_release_jump=False: iters={noj.iters} (logged 30)")

    print("Hard case: warm-start dual overshoot (OSCBF Panda teleop, step 10155)")
    # Captured spike, still open: the previous step's dual on row 28 is 2.4x
    # the new optimum's and two rows join the active set, so the warm solve
    # took 15 iters vs 14 cold (median 2). Warm-starting from the same stale x
    # with the correct duals converges in 2 iters, so the stale duals are the
    # cost. Pinned as a regression guard (bounds hold ~1.3x headroom); tighten
    # the warm-iters bound once a fix lands.
    d2 = np.load(os.path.join(os.path.dirname(__file__), "data", "warmstart_dual_overshoot.npz"))
    tol2 = float(d2["solver_tol"])

    def overshoot_solver():
        s = elastiqp.Solver()
        s.settings.eps_abs = tol2
        s.settings.eps_duality_gap_abs = tol2
        s.setup(d2["P"], d2["q"], d2["G"], d2["h"], d2["penalties"])
        return s

    cold2 = overshoot_solver().solve()
    check(
        "cold reference converges",
        cold2.converged == 1 and cold2.iters <= 20,
        f"iters={cold2.iters}",
    )
    ws2 = overshoot_solver()
    ws2.set_warm_start(d2["warm_x"], d2["warm_y"], d2["warm_z_ineq"])
    warm2 = ws2.solve()
    dx2 = np.abs(warm2.x - d2["expected_x"]).max()
    check(
        "warm solve converges accurately",
        warm2.converged == 1 and dx2 < 10 * tol2,
        f"|dx|={dx2:.1e}",
    )
    check(
        "warm solve stays within 20 iters (open spike)",
        warm2.iters <= 20,
        f"iters={warm2.iters} (logged 15, cold {cold2.iters})",
    )
    # Informational: the release jump helps here but does not close the gap.
    off2 = overshoot_solver()
    off2.settings.bcl_release_jump = False
    off2.set_warm_start(d2["warm_x"], d2["warm_y"], d2["warm_z_ineq"])
    noj2 = off2.solve()
    print(f"  [info] bcl_release_jump=False: iters={noj2.iters} (logged 15)")

    n_fail = RESULTS.count(False)
    print(f"\n{'All binding tests passed.' if n_fail == 0 else f'{n_fail} FAILURES'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
