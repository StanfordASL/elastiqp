#!/usr/bin/env python
"""qpax (hard + elastic, explicit backend) on the robot_solver_comparison
problems.

Python-side companion to examples/experiments/robot_solver_comparison.cc:
same recorded control-loop sequences, same feasible/conflict variants, same
per-route statistics — but only the qpax routes, which have no C++ bindings.
The other solvers' numbers come from the C++ binary; this script does not
rerun them.

The conflict variant is regenerated here exactly as in the C++ binary (same
auxiliary PIQP LP, same settings, same row-choice rule), so the tightened
sequences match the C++ run.

Routes:
  qpax-hard     solve_qp on the hard problem (A x = b, G x <= h). On the
                conflict variant the hard problem is infeasible; qpax has no
                infeasibility certificate, so those ticks land in 'other'.
  qpax-elastic  solve_qp_elastic on the elastic problem with the sequence's
                per-row penalty. qpax's elastic form has no hard equalities,
                so A x = b is folded as paired elastic rows [A; -A] with a
                penalty above the largest equality dual (10x the max |y| of a
                PIQP reference on the slack-expanded QP over the sequence,
                untimed) — by exact penalty they are satisfied at the optimum.

Timing: one jax.jit per (sequence, route); the first call (compilation) is
excluded, then each tick is timed once around jax.block_until_ready with
time.perf_counter — mirroring the single-pass per-tick timing of the C++
binary.

Run from the repo's .venv. Output: results/robot_solver_comparison_qpax.csv
(same columns as the C++ robot_solver_comparison.csv, so the rows can be
merged into the same table).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import bench_common as bc

EPS = 1e-5  # overridable via --eps (module-level so the jitted fns see it)
MAX_ITER = 250  # qpax's default 30 truncates otherwise-correct solves
VIOL_THRESH = 1e-4
GAP_FRAC = 0.05
INF = float("inf")


# ----------------------------------------------------------- conflict gen
# Mirrors robot_solver_comparison.cc: one row (fixed per scenario) is
# tightened at every tick to v_min - gap, where v_min is the smallest value
# of that row the remaining constraints admit (an auxiliary LP).

def tightest_value(qp: bc.RobotQP, a: int) -> tuple[float, int]:
    """(v_min, holders) of row a over the other constraints; same PIQP LP
    (tiny Tikhonov term, eps_abs 1e-8) as the C++ TightestValue."""
    import piqp

    n = qp.q.size
    G = np.delete(qp.G, a, axis=0)
    h = np.delete(qp.h, a)
    lp = piqp.DenseSolver()
    lp.settings.verbose = False
    lp.settings.eps_abs = 1e-8
    lp.settings.eps_rel = 0
    lp.settings.max_iter = 2000
    lp.setup(1e-6 * np.eye(n), qp.G[a],
             qp.A if qp.A.size else None, qp.b if qp.b.size else None,
             G, None, h, None, None)
    if lp.solve() != piqp.PIQP_SOLVED:
        return -INF, 0
    z = np.asarray(lp.result.z_u)
    holders = int((z > 0.01 * z.max()).sum())
    return float(qp.G[a] @ lp.result.x), holders


def choose_conflict_row(qp: bc.RobotQP) -> int:
    soft = qp.penalty.min()
    fallback = -1
    for a in range(qp.h.size):
        if qp.penalty[a] > soft:
            continue
        if fallback < 0:
            fallback = a
        if tightest_value(qp, a)[1] >= 2:
            return a
    return fallback


def make_conflict_sequence(seq: list[bc.RobotQP], a: int) -> list[bc.RobotQP]:
    out = []
    for qp in seq:
        gap = GAP_FRAC * max(1.0, abs(qp.h[a]))
        h = qp.h.copy()
        h[a] = tightest_value(qp, a)[0] - gap
        out.append(bc.RobotQP(Q=qp.Q, q=qp.q, A=qp.A, b=qp.b, G=qp.G, h=h,
                              penalty=qp.penalty))
    return out


# ------------------------------------------------- eq penalty calibration

def calibrate_eq_penalty(seq: list[bc.RobotQP]) -> float:
    """10x the largest equality dual of a PIQP reference on the l1-slack
    expanded QP (feasible by construction) over the sequence. Above the
    largest multiplier the exact-penalty argument applies, so qpax-elastic's
    folded equality rows must be satisfied at the optimum. Untimed setup."""
    import piqp

    f = seq[0]
    if f.b.size == 0:
        return 0.0
    n, m, p = f.q.size, f.b.size, f.h.size
    C = np.hstack([np.zeros((p, n)), -np.eye(p)])
    lb = np.concatenate([np.full(n, -INF), np.zeros(p)])
    y_max = 0.0
    for qp in seq:
        H = np.zeros((n + p, n + p))
        H[:n, :n] = qp.Q
        g = np.concatenate([qp.q, qp.penalty])
        A = np.hstack([qp.A, np.zeros((m, p))])
        C[:, :n] = qp.G
        solver = piqp.DenseSolver()
        solver.settings.verbose = False
        solver.settings.eps_abs = EPS
        solver.settings.eps_rel = 0
        solver.setup(H, g, A, qp.b, C, None, qp.h, lb, None)
        if solver.solve() == piqp.PIQP_SOLVED:
            y_max = max(y_max, float(np.abs(solver.result.y).max()))
    return 10.0 * max(1.0, y_max)


# ------------------------------------------------------------------ stats
# Mirrors RouteStats in robot_solver_comparison.cc.

class RouteStats:
    def __init__(self):
        self.us: list[float] = []
        self.iters = 0
        self.solved = self.infeasible = self.other = 0
        self.primal_ticks = 0
        self.viol_rows = self.spurious_rows = self.viol_l1 = 0.0
        self.worst_eq = 0.0

    def add(self, secs: float, iters: int):
        self.us.append(1e6 * secs)
        self.iters += iters

    def add_primal(self, qp: bc.RobotQP, x: np.ndarray, conflict_row: int):
        self.primal_ticks += 1
        v = np.maximum(qp.G @ x - qp.h, 0.0)
        for i in np.flatnonzero(v > VIOL_THRESH):
            self.viol_rows += 1
            if i != conflict_row:
                self.spurious_rows += 1
        self.viol_l1 += v.sum()
        if qp.b.size:
            self.worst_eq = max(self.worst_eq,
                                float(np.abs(qp.A @ x - qp.b).max()))

    def percentile(self, f: float) -> float:
        v = sorted(self.us)
        return v[int(f * (len(v) - 1) + 0.5)]

    def mean(self) -> float:
        return float(np.mean(self.us))

    def mean_iters(self) -> float:
        return self.iters / len(self.us)

    def finalize(self):
        if self.primal_ticks:
            self.viol_rows /= self.primal_ticks
            self.spurious_rows /= self.primal_ticks
            self.viol_l1 /= self.primal_ticks


# ----------------------------------------------------------------- routes

_JAX_FNS: dict = {}


def qpax_fns():
    if "hard" not in _JAX_FNS:
        import jax

        jax.config.update("jax_enable_x64", True)
        import qpax

        _JAX_FNS["hard"] = jax.jit(
            lambda Q, q, A, b, G, h: qpax.solve_qp(
                Q, q, A, b, G, h, backend="e",
                solver_tol=EPS, max_iter=MAX_ITER))
        # penalty is a per-row array here (qpax broadcasts it), so it is a
        # traced argument, not static.
        _JAX_FNS["elastic"] = jax.jit(
            lambda Q, q, G, h, penalty: qpax.solve_qp_elastic(
                Q, q, G, h, penalty, backend="e",
                solver_tol=EPS, max_iter=MAX_ITER))
    return _JAX_FNS


def timed_call(fn, *args) -> tuple[tuple, float]:
    import jax

    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return out, time.perf_counter() - t0


def run_qpax_hard(seq: list[bc.RobotQP], conflict_row: int) -> RouteStats:
    fn = qpax_fns()["hard"]
    f = seq[0]
    fn(f.Q, f.q, f.A, f.b, f.G, f.h)  # compile, excluded from timing
    st = RouteStats()
    for qp in seq:
        out, secs = timed_call(fn, qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h)
        st.add(secs, int(np.asarray(out[5])))
        if bool(np.asarray(out[4])):
            st.solved += 1
            st.add_primal(qp, np.asarray(out[0]), conflict_row)
        else:
            st.other += 1  # no infeasibility certificate in qpax
    st.finalize()
    return st


def elastic_form(qp: bc.RobotQP, eq_penalty: float):
    """Fold A x = b as paired elastic rows (qpax's elastic form has no hard
    equalities); inequality rows keep the sequence's per-row penalty."""
    if qp.b.size == 0:
        return qp.G, qp.h, qp.penalty
    G = np.vstack([qp.G, qp.A, -qp.A])
    h = np.concatenate([qp.h, qp.b, -qp.b])
    penalty = np.concatenate([qp.penalty,
                              np.full(2 * qp.b.size, eq_penalty)])
    return G, h, penalty


def run_qpax_elastic(seq: list[bc.RobotQP], conflict_row: int,
                     eq_penalty: float) -> RouteStats:
    fn = qpax_fns()["elastic"]
    G0, h0, pen0 = elastic_form(seq[0], eq_penalty)
    fn(seq[0].Q, seq[0].q, G0, h0, pen0)  # compile, excluded from timing
    st = RouteStats()
    for qp in seq:
        G, h, pen = elastic_form(qp, eq_penalty)
        out, secs = timed_call(fn, qp.Q, qp.q, G, h, pen)
        st.add(secs, int(np.asarray(out[7])))
        if bool(np.asarray(out[6])):
            st.solved += 1
            st.add_primal(qp, np.asarray(out[0]), conflict_row)
        else:
            st.other += 1
    st.finalize()
    return st


# ----------------------------------------------------------------- report

def print_row(scenario: str, variant: str, route: str, st: RouteStats):
    print(f"{scenario:<8} {variant:<8} {route:<18} | "
          f"{st.mean():9.1f} us (p95 {st.percentile(0.95):9.1f}) "
          f"{st.mean_iters():5.1f} it | "
          f"ok {st.solved:3d} inf {st.infeasible:3d} oth {st.other:3d} | "
          f"viol {st.viol_rows:5.1f} spur {st.spurious_rows:5.1f} "
          f"l1 {st.viol_l1:8.3f} | eq {st.worst_eq:.0e}")


def main():
    global EPS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sequence_file", nargs="?", default=str(bc.SEQUENCE_FILE))
    ap.add_argument("--csv", default=str(bc.RESULTS_DIR),
                    help="output directory for the CSV")
    ap.add_argument("--eq-penalty", type=float, default=None,
                    help="override the calibrated folded-equality penalty")
    ap.add_argument("--eps", type=float, default=EPS,
                    help="qpax solver_tol (unscaled full-KKT inf-norm)")
    args = ap.parse_args()
    EPS = args.eps

    seqs = bc.load_sequences(args.sequence_file)
    print(f"eps={EPS:g}, conflict gap = {100 * GAP_FRAC:.0f}% of the row "
          f"bound\n")

    rows = []
    for name, qps in seqs.items():
        conflict_row = choose_conflict_row(qps[0])
        print(f"{name}: tightened row {conflict_row}")
        conflict_seq = make_conflict_sequence(qps, conflict_row)

        for variant, s, row_c in [("feasible", qps, -1),
                                  ("conflict", conflict_seq, conflict_row)]:
            eq_pen = (args.eq_penalty if args.eq_penalty is not None
                      else calibrate_eq_penalty(s))
            if s[0].b.size:
                print(f"  {variant}: folded-eq penalty {eq_pen:g}")
            for route, st in [
                ("qpax-hard", run_qpax_hard(s, row_c)),
                ("qpax-elastic", run_qpax_elastic(s, row_c, eq_pen)),
            ]:
                rows.append([name, variant, route, len(st.us), st.mean(),
                             st.percentile(0.5), st.percentile(0.95),
                             st.percentile(1.0), st.mean_iters(), st.solved,
                             st.infeasible, st.other, st.viol_rows,
                             st.spurious_rows, st.viol_l1, st.worst_eq])
                print_row(name, variant, route, st)
        print()

    bc.write_csv(Path(args.csv) / "robot_solver_comparison_qpax.csv",
                 ["scenario", "variant", "route", "ticks", "mean_us",
                  "p50_us", "p95_us", "max_us", "mean_iters", "solved",
                  "infeasible", "other", "viol_rows", "spurious_rows",
                  "viol_l1", "worst_eq"],
                 rows)


if __name__ == "__main__":
    main()
