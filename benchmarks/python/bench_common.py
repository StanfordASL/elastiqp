"""Shared helpers for the Python-side ElastiQP benchmarks.

The Python layer hosts the experiments that have to happen in Python —
qpax is JAX-only, so the differentiability comparison lives here — plus
the plotting of the CSVs the C++ benchmarks emit. Precise C++-vs-C++
timing lives in the C++ benchmarks (one harness per comparison table);
here every solver is called through JAX, so dispatch overhead is shared.

Run from the repo's ``.venv`` (``pip install -e ".[bench]"``).
"""

from __future__ import annotations

import dataclasses
import struct
import time
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BENCH_DIR / "results"
SEQUENCE_FILE = BENCH_DIR / "data" / "robot_sequences.bin"


# --------------------------------------------------------------- robot i/o

@dataclasses.dataclass
class RobotQP:
    """One elastic QP: min 0.5 x'Qx + q'x + penalty't
    s.t. Ax = b (hard), Gx - t <= h, t >= 0."""

    Q: np.ndarray
    q: np.ndarray
    A: np.ndarray
    b: np.ndarray
    G: np.ndarray
    h: np.ndarray
    penalty: np.ndarray


def load_sequences(path: Path | str = SEQUENCE_FILE) -> dict[str, list[RobotQP]]:
    """Reads the EQPS binary written by gen_robot_sequences (v1 float64 or
    v2 with a dtype field; always returns float64 arrays)."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"EQPS":
        raise ValueError(f"bad magic in {path}")
    off = 4

    def i32() -> int:
        nonlocal off
        (v,) = struct.unpack_from("<i", raw, off)
        off += 4
        return v

    version = i32()
    if version == 1:
        dtype = np.dtype("<f8")
    elif version == 2:
        code = i32()
        if code not in (0, 1):
            raise ValueError(f"unsupported EQPS dtype code {code}")
        dtype = np.dtype("<f4") if code == 1 else np.dtype("<f8")
    else:
        raise ValueError(f"unsupported EQPS version {version}")

    def mat(rows: int, cols: int) -> np.ndarray:
        nonlocal off
        count = rows * cols
        a = np.frombuffer(raw, dtype=dtype, count=count, offset=off)
        off += dtype.itemsize * count
        # File stores column-major (Eigen default).
        return a.astype(np.float64).reshape((cols, rows)).T.copy()

    def vec(size: int) -> np.ndarray:
        nonlocal off
        a = np.frombuffer(raw, dtype=dtype, count=size, offset=off)
        off += dtype.itemsize * size
        return a.astype(np.float64)

    seqs: dict[str, list[RobotQP]] = {}
    for _ in range(i32()):
        name_len = i32()
        name = raw[off:off + name_len].decode()
        off += name_len
        qps = []
        for _ in range(i32()):
            n, m, p = i32(), i32(), i32()
            qps.append(RobotQP(Q=mat(n, n), q=vec(n), A=mat(m, n), b=vec(m),
                               G=mat(p, n), h=vec(p), penalty=vec(p)))
        seqs[name] = qps
    return seqs


# ------------------------------------------------------------------ timing

def time_solve(fn, min_reps: int = 3, min_time_s: float = 0.05) -> float:
    """Median wall-clock seconds of fn() over adaptively chosen repeats."""
    times = []
    total = 0.0
    while len(times) < min_reps or total < min_time_s:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        total += dt
        if len(times) >= 200:
            break
    return float(np.median(times))


# ----------------------------------------------------------------- metrics

def hard_kkt_residual(P, c, A_eq, b_eq, G, h, x, y, z) -> float:
    """Unscaled l_inf KKT residual of a hard QP
    (min 0.5x'Px + c'x s.t. A_eq x = b_eq, Gx <= h) at (x, y, z)."""
    r = P @ x + c
    if A_eq.shape[0]:
        r = r + A_eq.T @ y
    if G.shape[0]:
        r = r + G.T @ z
    res = np.abs(r).max() if r.size else 0.0
    if A_eq.shape[0]:
        res = max(res, np.abs(A_eq @ x - b_eq).max())
    if G.shape[0]:
        v = G @ x - h
        res = max(res, max(v.max(), 0.0))
        res = max(res, np.abs(z * v).max())
        res = max(res, max((-z).max(), 0.0))
    return float(res)


def elastic_kkt_residual(Q, q, A, b, G, h, penalty, x, t, y, z1, z2) -> float:
    """Unscaled l_inf KKT residual of the elastic QP at (x, t, y, z1, z2)
    (same definition as the C++ benchmarks' problem_gen helper)."""
    stat = Q @ x + q + G.T @ z2
    if A.shape[0]:
        stat = stat + A.T @ y
    res = float(np.abs(stat).max())
    if A.shape[0]:
        res = max(res, float(np.abs(A @ x - b).max()))
    res = max(res, float(np.abs(penalty - z1 - z2).max()))
    viol = G @ x - t - h
    res = max(res, float(np.maximum(viol, 0.0).max()))
    for v in (-t, -z1, -z2):
        res = max(res, float(np.maximum(v, 0.0).max()))
    res = max(res, float(np.abs(z2 * viol).max()))
    res = max(res, float(np.abs(z1 * t).max()))
    return res


def primal_violation(A_eq, b_eq, G, h, x) -> float:
    """Worst primal constraint violation (l_inf)."""
    res = 0.0
    if A_eq.shape[0]:
        res = max(res, float(np.abs(A_eq @ x - b_eq).max()))
    if G.shape[0]:
        res = max(res, float(max((G @ x - h).max(), 0.0)))
    return res


def shifted_geom_mean(values: np.ndarray, shift: float) -> float:
    """Shifted geometric mean, the aggregate used by the OSQP/PIQP
    benchmark suites: exp(mean(log(v + shift))) - shift."""
    v = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(v + shift))) - shift)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(_fmt(v) for v in row) + "\n")
    print(f"wrote {path}")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)
