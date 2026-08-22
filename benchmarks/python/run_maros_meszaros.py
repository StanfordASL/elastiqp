#!/usr/bin/env python
"""Maros-Meszaros, restricted to the robotics-relevant subset, solved by
elastiqp / piqp / proxqp / qpax (hard and elastic).

Rationale (see docs/BENCHMARKS.md): the full set is dominated by large,
very sparse, and badly scaled problems — the opposite of the robot-control
QPs ElastiQP targets (small, dense, reasonably conditioned). Following the
ProxQP paper's dimension-restricted subset (they keep n, rows <= 1000; 62
problems), we restrict harder, to the scales the solver is designed for:

    n <= --nmax (default 100) variables,
    rows <= --rows-max (default 500) constraint rows, and
    combined [P; A] density >= --min-density (default 5%)
    (combined, because many small problems pair a dense Hessian with a
    bounds-heavy, sparse-looking A — a dense solver factors both),

and report every problem's size/density/conditioning in
maros_meszaros_stats.csv so the filter is transparent (problems excluded
only by the density cut are visible there). No conditioning filter is
applied — ill-conditioned problems stay in, and the per-problem cond(P)
column lets the reader slice the results either way.

Hard routes solve the HARD problem cold at eps_abs = 1e-6 (eps_rel = 0):
piqp/proxqp on l <= Ax <= u directly, qpax-hard on the one-sided
conversion. elastiqp solves its elastic relaxation (l == u rows hard,
finite sides elastic, Ruiz on — this badly-scaled set is what the flag
exists for) with penalty = 10x the largest dual of a high-accuracy
reference, so by exact penalty it must recover the hard solution — 'ok'
additionally requires it actually did (objective match + feasibility),
making its column directly comparable. qpax-elastic solves qpax's own
elastic relaxation (explicit backend) at the same penalty; qpax supports
no hard equalities there, so l == u rows are folded as paired elastic
inequalities — by the same exact-penalty argument the solution must still
be the hard one, and 'ok' is checked against the hard constraints.

Success ('ok') = solver-reported success AND primal violation <= 1e-4 AND
(when the reference converged) relative objective error <= 1e-4.

Data: the .mat files ship inside the proxsuite release tarball fetched by
the -DELASTIQP_BENCH_EXTERNAL_SOLVERS=ON build (see --data-dir default).

Run from the repo's .venv. Outputs:
results/maros_meszaros_{stats,py_results,py_summary}.csv
(py_ prefixed so the C++ bench_maros_meszaros CSV is not clobbered)
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import scipy.io as sio

import bench_common as bc

EPS = 1e-6
MAX_ITER = 250
VIOL_TOL = 1e-4
OBJ_TOL = 1e-4


def default_data_dir() -> str:
    """maros_meszaros_data inside the newest fetched proxsuite tarball."""
    root = bc.BENCH_DIR.parent
    hits = sorted(
        glob.glob(str(root / "build*" / "_deps" / "proxsuite_src-src" /
                      "test" / "data" / "maros_meszaros_data")),
        key=os.path.getmtime)
    return hits[-1] if hits else ""


def load_shape(path):
    """Sizes and sparsity without densifying (some problems are 90k x 90k)."""
    d = sio.loadmat(path)
    P, A = d["P"], d["A"]
    dens_P = P.nnz / max(1, P.shape[0] * P.shape[1])
    dens_A = A.nnz / max(1, A.shape[0] * A.shape[1])
    # Combined density of the stacked [P; A] data: many small problems have
    # a dense Hessian but a bounds-heavy (hence sparse-looking) A, and it is
    # the overall problem data a dense solver factors.
    dens = (P.nnz + A.nnz) / max(1, (P.shape[0] + A.shape[0]) * P.shape[1])
    return d, P.shape[0], A.shape[0], dens_P, dens_A, dens


def load_problem(d):
    P = np.asarray(d["P"].todense(), dtype=np.float64)
    A = np.asarray(d["A"].todense(), dtype=np.float64)
    q = np.asarray(d["q"], dtype=np.float64).ravel()
    l = np.asarray(d["l"], dtype=np.float64).ravel()
    u = np.asarray(d["u"], dtype=np.float64).ravel()
    l[l <= -1e19] = -np.inf
    u[u >= 1e19] = np.inf
    return P, q, A, l, u


def split_forms(P, q, A, l, u):
    """eq/ineq split and the one-sided conversion shared by qpax/elastiqp."""
    eq = l == u
    A_eq, b_eq = A[eq], u[eq]
    A_in, l_in, u_in = A[~eq], l[~eq], u[~eq]
    rows = []
    rhs = []
    for i in range(A_in.shape[0]):
        if np.isfinite(u_in[i]):
            rows.append(A_in[i])
            rhs.append(u_in[i])
        if np.isfinite(l_in[i]):
            rows.append(-A_in[i])
            rhs.append(-l_in[i])
    G = np.array(rows) if rows else np.zeros((0, P.shape[0]))
    h = np.array(rhs) if rhs else np.zeros(0)
    return A_eq, b_eq, A_in, l_in, u_in, G, h


def objective(P, q, x):
    return float(0.5 * x @ P @ x + q @ x)


# ------------------------------------------------------------------ piqp

def solve_piqp(P, q, forms, eps=EPS, max_iter=MAX_ITER):
    import piqp

    A_eq, b_eq, A_in, l_in, u_in, _, _ = forms
    solver = piqp.DenseSolver()
    solver.settings.eps_abs = eps
    solver.settings.eps_rel = 0
    solver.settings.eps_duality_gap_abs = eps
    solver.settings.eps_duality_gap_rel = 0
    solver.settings.max_iter = max_iter
    solver.setup(P, q, A_eq if A_eq.size else None,
                 b_eq if b_eq.size else None,
                 A_in if A_in.size else None,
                 l_in if A_in.size else None,
                 u_in if A_in.size else None, None, None)
    status = solver.solve()
    ok = status == piqp.PIQP_SOLVED
    res = solver.result
    duals = [np.abs(res.y)] if res.y.size else []
    for z in (res.z_l, res.z_u):
        if z.size:
            duals.append(np.abs(z))
    max_dual = max((d.max() for d in duals if d.size), default=0.0)
    return dict(status=str(status).split(".")[-1], reported_ok=ok,
                x=np.asarray(res.x), iters=int(res.info.iter),
                max_dual=float(max_dual))


def time_piqp(P, q, forms):
    return bc.time_solve(lambda: solve_piqp(P, q, forms))


# ---------------------------------------------------------------- proxqp

def solve_proxqp(P, q, forms):
    import proxsuite

    A_eq, b_eq, A_in, l_in, u_in, _, _ = forms
    n = P.shape[0]
    qp = proxsuite.proxqp.dense.QP(n, A_eq.shape[0], A_in.shape[0])
    qp.settings.eps_abs = EPS
    qp.settings.eps_rel = 0
    qp.settings.check_duality_gap = True
    qp.settings.eps_duality_gap_abs = EPS
    qp.settings.eps_duality_gap_rel = 0
    qp.settings.max_iter = 10 * MAX_ITER  # inner iterations
    qp.settings.verbose = False
    qp.init(P, q, A_eq if A_eq.size else None, b_eq if b_eq.size else None,
            A_in if A_in.size else None, l_in if A_in.size else None,
            u_in if A_in.size else None)
    qp.solve()
    ok = (qp.results.info.status ==
          proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED)
    return dict(status=str(qp.results.info.status).split(".")[-1],
                reported_ok=ok, x=np.asarray(qp.results.x),
                iters=int(qp.results.info.iter))


def time_proxqp(P, q, forms):
    return bc.time_solve(lambda: solve_proxqp(P, q, forms))


# ------------------------------------------------------------------ qpax

_QPAX_CACHE: dict = {}


def qpax_fns():
    if "hard" not in _QPAX_CACHE:
        import jax

        jax.config.update("jax_enable_x64", True)
        import qpax

        # Same iteration budget as the other solvers (qpax's default is 30,
        # which truncates several otherwise-correct solves). Explicit ("e")
        # backend for both routes.
        _QPAX_CACHE["hard"] = jax.jit(
            lambda Q, q, A, b, G, h: qpax.solve_qp(
                Q, q, A, b, G, h, backend="e",
                solver_tol=EPS, max_iter=MAX_ITER))
        _QPAX_CACHE["elastic"] = jax.jit(
            lambda Q, q, G, h, penalty: qpax.solve_qp_elastic(
                Q, q, G, h, penalty, backend="e",
                solver_tol=EPS, max_iter=MAX_ITER),
            static_argnames=("penalty",))
    return _QPAX_CACHE


def solve_qpax(P, q, forms):
    A_eq, b_eq, _, _, _, G, h = forms
    fn = qpax_fns()["hard"]
    out = fn(P, q, A_eq, b_eq, G, h)
    x = np.asarray(out[0])
    converged = bool(np.asarray(out[4]))
    iters = int(np.asarray(out[5]))
    return dict(status="converged" if converged else "not_converged",
                reported_ok=converged, x=x, iters=iters)


def time_qpax(P, q, forms):
    fn = qpax_fns()["hard"]
    A_eq, b_eq, _, _, _, G, h = forms

    def run():
        out = fn(P, q, A_eq, b_eq, G, h)
        out[0].block_until_ready()

    run()  # compile, excluded from timing
    return bc.time_solve(run)


def qpax_elastic_form(forms):
    """qpax's elastic solve has no hard equalities, so fold l == u rows as
    paired elastic inequalities alongside the one-sided conversion."""
    A_eq, b_eq, _, _, _, G, h = forms
    if not A_eq.size:
        return G, h
    G_el = np.vstack([G, A_eq, -A_eq])
    h_el = np.concatenate([h, b_eq, -b_eq])
    return G_el, h_el


def solve_qpax_elastic(P, q, forms, penalty_mag):
    G_el, h_el = qpax_elastic_form(forms)
    fn = qpax_fns()["elastic"]
    out = fn(P, q, G_el, h_el, penalty_mag)
    x = np.asarray(out[0])
    converged = bool(np.asarray(out[6]))
    iters = int(np.asarray(out[7]))
    return dict(status="converged" if converged else "not_converged",
                reported_ok=converged, x=x, iters=iters)


def time_qpax_elastic(P, q, forms, penalty_mag):
    G_el, h_el = qpax_elastic_form(forms)
    fn = qpax_fns()["elastic"]

    def run():
        out = fn(P, q, G_el, h_el, penalty_mag)
        out[0].block_until_ready()

    run()  # compile, excluded from timing
    return bc.time_solve(run)


# -------------------------------------------------------------- elastiqp

def solve_elastiqp(P, q, forms, penalty_mag):
    import elastiqp

    A_eq, b_eq, _, _, _, G, h = forms
    kwargs = dict(eps_abs=EPS, max_iter=MAX_ITER, ruiz=True)
    if A_eq.size:
        sol = elastiqp.solve(P, q, G, h, penalty_mag, A=A_eq, b=b_eq,
                             **kwargs)
    else:
        sol = elastiqp.solve(P, q, G, h, penalty_mag, **kwargs)
    return dict(status="converged" if sol.converged == 1 else "not_converged",
                reported_ok=sol.converged == 1, x=np.asarray(sol.x),
                iters=int(sol.iters))


def time_elastiqp(P, q, forms, penalty_mag):
    return bc.time_solve(lambda: solve_elastiqp(P, q, forms, penalty_mag))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nmax", type=int, default=100)
    ap.add_argument("--rows-max", type=int, default=500)
    ap.add_argument("--min-density", type=float, default=0.05)
    ap.add_argument("--data-dir", default=default_data_dir())
    args = ap.parse_args()
    if not args.data_dir or not os.path.isdir(args.data_dir):
        raise SystemExit(
            "no Maros-Meszaros data dir found; configure a build with "
            "-DELASTIQP_BENCH_EXTERNAL_SOLVERS=ON (which fetches the "
            "proxsuite tarball) or pass --data-dir")

    stats_rows = []
    results_rows = []
    included = []
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.mat"))):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            d, n, rows, dens_P, dens_A, dens = load_shape(path)
        except Exception as e:  # noqa: BLE001
            print(f"skipping {name}: {e}")
            continue
        reasons = []
        if n > args.nmax:
            reasons.append(f"n>{args.nmax}")
        if rows > args.rows_max:
            reasons.append(f"rows>{args.rows_max}")
        if not reasons and dens < args.min_density:
            reasons.append(f"density<{args.min_density}")
        inc = not reasons
        eq_rows = ineq_rows = -1
        cond_P = float("nan")
        if n <= 2000 and rows <= 5000:  # only densify what is cheap
            P, q, A, l, u = load_problem(d)
            eq_rows = int((l == u).sum())
            ineq_rows = int((l != u).sum())
            try:
                cond_P = float(np.linalg.cond(P))
            except Exception:  # noqa: BLE001
                cond_P = float("inf")
            if inc:
                included.append((name, P, q, A, l, u, cond_P))
        stats_rows.append([name, n, rows, eq_rows, ineq_rows, dens_A,
                           dens_P, dens, cond_P, int(inc),
                           "+".join(reasons) or "included"])

    print(f"{len(included)} problems in the subset "
          f"(n<={args.nmax}, rows<={args.rows_max}, "
          f"density>={args.min_density})")

    for name, P, q, A, l, u, cond_P in included:
        forms = split_forms(P, q, A, l, u)
        t0 = time.time()

        # High-accuracy reference (piqp @ 1e-11) for objective + duals.
        ref = solve_piqp(P, q, forms, eps=1e-11, max_iter=1000)
        ref_ok = ref["reported_ok"]
        ref_obj = objective(P, q, ref["x"])
        penalty_mag = 10.0 * max(1.0, ref["max_dual"])

        A_eq, b_eq, _, _, _, G, h = forms

        def record(solver, run, secs):
            viol = bc.primal_violation(A_eq, b_eq, G, h, run["x"])
            obj = objective(P, q, run["x"])
            rel = abs(obj - ref_obj) / max(1.0, abs(ref_obj))
            ok = (run["reported_ok"] and viol <= VIOL_TOL and
                  (not ref_ok or rel <= OBJ_TOL))
            results_rows.append([name, P.shape[0], A.shape[0], solver,
                                 run["status"], int(ok), 1e3 * secs,
                                 run["iters"], viol, obj,
                                 rel if ref_ok else float("nan")])

        for solver, solve_fn, time_fn in [
            ("elastiqp",
             lambda: solve_elastiqp(P, q, forms, penalty_mag),
             lambda: time_elastiqp(P, q, forms, penalty_mag)),
            ("piqp", lambda: solve_piqp(P, q, forms),
             lambda: time_piqp(P, q, forms)),
            ("proxqp", lambda: solve_proxqp(P, q, forms),
             lambda: time_proxqp(P, q, forms)),
            ("qpax-hard", lambda: solve_qpax(P, q, forms),
             lambda: time_qpax(P, q, forms)),
            ("qpax-elastic",
             lambda: solve_qpax_elastic(P, q, forms, penalty_mag),
             lambda: time_qpax_elastic(P, q, forms, penalty_mag)),
        ]:
            try:
                run = solve_fn()
                secs = time_fn()
                record(solver, run, secs)
            except Exception as e:  # noqa: BLE001
                results_rows.append([name, P.shape[0], A.shape[0], solver,
                                     f"error:{type(e).__name__}", 0,
                                     float("nan"), 0, float("nan"),
                                     float("nan"), float("nan")])
        print(f"  {name}: done in {time.time() - t0:.1f}s "
              f"(ref {'ok' if ref_ok else 'FAILED'}, cond(P) {cond_P:.1e})")

    bc.write_csv(bc.RESULTS_DIR / "maros_meszaros_stats.csv",
                 ["name", "n", "rows", "eq_rows", "ineq_rows", "density_A",
                  "density_P", "density_PA", "cond_P", "included", "reason"],
                 stats_rows)
    bc.write_csv(bc.RESULTS_DIR / "maros_meszaros_py_results.csv",
                 ["name", "n", "rows", "solver", "status", "ok", "time_ms",
                  "iters", "primal_viol", "objective", "obj_rel_err"],
                 results_rows)

    # Aggregate: failure rate + shifted geometric mean over the subset,
    # failures assigned the max observed time (piqp_benchmarks convention).
    solvers = sorted({r[3] for r in results_rows})
    times = {s: [] for s in solvers}
    fails = {s: 0 for s in solvers}
    max_time = np.nanmax([r[6] for r in results_rows])
    for r in results_rows:
        s, ok, t = r[3], r[5], r[6]
        if ok and np.isfinite(t):
            times[s].append(t)
        else:
            fails[s] += 1
            times[s].append(max_time)
    summary = []
    sgms = {s: bc.shifted_geom_mean(np.array(times[s]), shift=1.0)
            for s in solvers}
    best = min(sgms.values())
    n_prob = len(included)
    for s in solvers:
        summary.append([s, n_prob, n_prob - fails[s], fails[s] / n_prob,
                        sgms[s], sgms[s] / best])
    bc.write_csv(bc.RESULTS_DIR / "maros_meszaros_py_summary.csv",
                 ["solver", "problems", "solved", "failure_rate",
                  "sgm_time_ms_shift1", "sgm_normalized"], summary)
    for row in summary:
        print(f"{row[0]:>14}: {row[2]}/{row[1]} solved, "
              f"sgm {row[4]:.3f} ms ({row[5]:.2f}x best)")


if __name__ == "__main__":
    main()
