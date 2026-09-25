"""ElastiQP JAX FFI wrapper

Supported:
- Different methods (das, pdal, ipm)
- jit, vmap, scan
- grad (currently, only for ipm and pdal)

Notes:
- vmap is sequential mode
- Requires float64
- Warm-starting is not differentiable
- Warm-starting requires explicit passing of the previous result
- Recommended default kappa for differentiability: 1e-3
"""

import ctypes
import os
from functools import partial
from pathlib import Path
from typing import NamedTuple

try:
    import jax
except ImportError as e:
    import sys

    if sys.version_info < (3, 11):
        raise ImportError(
            "elastiqp.jax requires Python >= 3.11 (for jax>=0.9.1)"
        ) from e
    raise ImportError("elastiqp.jax requires jax to be installed") from e

from jax import Array
import jax.numpy as jnp
import numpy as np

__all__ = ["solve", "Result", "METHODS"]

METHODS = ("das", "pdal", "ipm")
_METHOD_ID = {"das": 0, "pdal": 1, "ipm": 2}
# Default outer budget for das/pdal/ipm,
# (active-set iterations / BCL rounds / interior-point iterations)
_DEFAULT_MAX_ITER = {"das": 10000, "pdal": 250, "ipm": 250}
_DEFAULT_EPS_ABS = {"das": 1e-6, "pdal": 1e-5, "ipm": 1e-5}
_DEFAULT_RUIZ = {"das": True, "pdal": False, "ipm": False}
# TODO (dan): get these defaults in alignment across backends


def _find_library():
    # Search order mirrors elastiqp._core (see _import_core): an explicit
    # ELASTIQP_BUILD_DIR first, so a source checkout tests the build it just
    # made rather than an installed copy; then next to this file (installed
    # wheel); then <repo>/build for an un-configured checkout.
    candidates = []
    env_dir = os.environ.get("ELASTIQP_BUILD_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path(__file__).resolve().parents[2] / "build")
    for directory in candidates:
        for name in (
            "libelastiqp_jax_ffi.so",
            "elastiqp_jax_ffi.so",
            "elastiqp_jax_ffi.dll",
            "libelastiqp_jax_ffi.dylib",
        ):
            path = directory / name
            if path.exists():
                return path
    raise ImportError(
        "elastiqp_jax_ffi shared library not found (searched "
        f"{[str(c) for c in candidates]}). Reinstall the package, or for "
        "source checkouts build it first (cmake --build build), optionally "
        "pointing ELASTIQP_BUILD_DIR at the build directory."
    )


_lib = ctypes.cdll.LoadLibrary(str(_find_library()))
jax.ffi.register_ffi_target(
    "elastiqp_solve", jax.ffi.pycapsule(_lib.ElastiqpSolve), platform="cpu"
)
jax.ffi.register_ffi_target(
    "elastiqp_solve_warm",
    jax.ffi.pycapsule(_lib.ElastiqpSolveWarm),
    platform="cpu",
)


class Result(NamedTuple):
    x: jax.Array
    t: jax.Array
    y: jax.Array
    z_t: jax.Array
    z: jax.Array
    converged: jax.Array
    iters: jax.Array


def _ffi_solve(
    Q,
    q,
    A,
    b,
    G,
    h,
    penalty,
    eps_abs,
    max_iter,
    ruiz,
    target_kappa,
    method="pdal",
):
    n = Q.shape[-1]
    m = b.shape[-1]
    p = h.shape[-1]
    batch = Q.shape[:-2]
    vec = lambda d: jax.ShapeDtypeStruct(batch + (d,), jnp.float64)
    # (x, t, y, z_t, z) tight solution, then the kappa-relaxed central
    # point (identical to the tight block when target_kappa <= 0), then
    # info = [converged, iters, relax_converged].
    out_types = [
        vec(n),
        vec(p),
        vec(m),
        vec(p),
        vec(p),
        vec(n),
        vec(p),
        vec(m),
        vec(p),
        vec(p),
        vec(3),
    ]
    call = jax.ffi.ffi_call("elastiqp_solve", out_types, vmap_method="sequential")
    return call(
        Q,
        q,
        A,
        b,
        G,
        h,
        penalty,
        eps_abs=np.float64(eps_abs),
        max_iter=np.int64(max_iter),
        ruiz=np.int64(bool(ruiz)),
        target_kappa=np.float64(target_kappa),
        method=np.int64(_METHOD_ID[method]),
    )


def _ffi_solve_warm(
    Q,
    q,
    A,
    b,
    G,
    h,
    penalty,
    x0,
    y0,
    z0,
    eps_abs,
    max_iter,
    ruiz,
    method,
):
    n = Q.shape[-1]
    m = b.shape[-1]
    p = h.shape[-1]
    batch = Q.shape[:-2]
    vec = lambda d: jax.ShapeDtypeStruct(batch + (d,), jnp.float64)
    # (x, t, y, z_t, z), then info = [converged, iters].
    out_types = [vec(n), vec(p), vec(m), vec(p), vec(p), vec(2)]
    call = jax.ffi.ffi_call("elastiqp_solve_warm", out_types, vmap_method="sequential")
    return call(
        Q,
        q,
        A,
        b,
        G,
        h,
        penalty,
        x0,
        y0,
        z0,
        eps_abs=np.float64(eps_abs),
        max_iter=np.int64(max_iter),
        ruiz=np.int64(bool(ruiz)),
        method=np.int64(_METHOD_ID[method]),
    )


@partial(jax.custom_vjp, nondiff_argnums=(10, 11, 12, 13))
def _solve_warm(
    Q,
    q,
    A,
    b,
    G,
    h,
    penalty,
    x0,
    y0,
    z0,
    eps_abs,
    max_iter,
    ruiz,
    method,
):
    return _ffi_solve_warm(
        Q,
        q,
        A,
        b,
        G,
        h,
        penalty,
        x0,
        y0,
        z0,
        eps_abs,
        max_iter,
        ruiz,
        method,
    )


def _solve_warm_fwd(*args):
    raise TypeError("elastiqp.jax.solve is not differentiable with warm_start")


def _solve_warm_bwd(*args):
    raise TypeError("elastiqp.jax.solve is not differentiable with warm_start")


_solve_warm.defvjp(_solve_warm_fwd, _solve_warm_bwd)


def _outer(a, b):
    return a[..., :, None] * b[..., None, :]


# Note: this funciton is a port of equivalent logic in qpax
# See also: kkt_vjp.hpp (c++ version) and _vjp_torch in torch.py
def _kkt_bwd(res, ct):
    """Implicit differentiation of the elastic KKT conditions."""
    Q, A, G, h, x, t, y, z_t, z = res
    xb, tb, yb, z_tb, zb, _ = ct
    n = Q.shape[-1]
    m = A.shape[-2]

    Qs = 0.5 * (Q + jnp.swapaxes(Q, -1, -2))
    Gt = jnp.swapaxes(G, -1, -2)
    At = jnp.swapaxes(A, -1, -2)
    mv = lambda M, v: jnp.einsum("...ij,...j->...i", M, v)

    D = mv(G, x) - t - h
    E = D - z * t / z_t

    # Eliminating the t, z_t, z rows reduces the adjoint system to (x, y).
    rt = -z_t * z_tb + t * tb
    rh = z * zb + z * rt / z_t
    rhs_x = xb - mv(Gt, rh / E)

    H = Qs + jnp.einsum("...ji,...j,...jk->...ik", G, -z / E, G)
    KKT = jnp.concatenate(
        [
            jnp.concatenate([H, At], axis=-1),
            jnp.concatenate([A, jnp.zeros(A.shape[:-2] + (m, m), Q.dtype)], axis=-1),
        ],
        axis=-2,
    )
    vxy = jnp.linalg.solve(KKT, jnp.concatenate([rhs_x, yb], axis=-1))
    vx = vxy[..., :n]
    vy = vxy[..., n:]

    vh = (rh - z * mv(G, vx)) / E
    vpen = (rt + t * vh) / z_t

    qb = -vx
    penalty_b = -vpen
    bb = vy
    hb = vh
    Qb = -0.5 * (_outer(vx, x) + _outer(x, vx))
    Ab = -(_outer(y, vx) + _outer(vy, x))
    Gb = -(_outer(z, vx) + _outer(vh, x))
    return Qb, qb, Ab, bb, Gb, hb, penalty_b


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11))
def _solve(
    Q,
    q,
    A,
    b,
    G,
    h,
    penalty,
    eps_abs,
    max_iter,
    ruiz,
    target_kappa,
    method,
):
    # Tight solution when not differentiating
    out = _ffi_solve(
        Q,
        q,
        A,
        b,
        G,
        h,
        penalty,
        eps_abs,
        max_iter,
        ruiz,
        0.0,
        method,
    )
    return tuple(out[:5]) + (out[10],)  # solution + info


def _solve_fwd(
    Q,
    q,
    A,
    b,
    G,
    h,
    penalty,
    eps_abs,
    max_iter,
    ruiz,
    target_kappa,
    method,
):
    if method == "das":
        raise TypeError(
            "elastiqp.jax.solve is not differentiable with method='das': "
            "See method='ipm' or 'pdal' instead"
        )
    if not target_kappa > 0:
        raise TypeError(
            "elastiqp.jax.solve is not differentiable with target_kappa=0: "
            "Set target_kappa > 0  (e.g. 1e-3) for smoothed gradients"
        )
    out = _ffi_solve(
        Q,
        q,
        A,
        b,
        G,
        h,
        penalty,
        eps_abs,
        max_iter,
        ruiz,
        target_kappa,
        method,
    )
    x, t, y, z_t, z, xr, tr, yr, z_t_r, z_r, info = out
    return (x, t, y, z_t, z, info), (Q, A, G, h, xr, tr, yr, z_t_r, z_r)


def _solve_bwd(eps_abs, max_iter, ruiz, target_kappa, method, res, ct):
    return _kkt_bwd(res, ct)


_solve.defvjp(_solve_fwd, _solve_bwd)


def solve(
    Q: Array,
    q: Array,
    G: Array,
    h: Array,
    penalty: Array | float,
    *,
    A: Array | None = None,
    b: Array | None = None,
    method: str = "das",
    eps_abs: float | None = None,
    max_iter: int | None = None,
    ruiz: bool | None = None,
    target_kappa: float = 1e-3,
    warm_start: tuple[Array, Array, Array] | Result | None = None,
) -> Result:
    """Solve the elastic QP

    min 0.5 x'Qx + q'x + penalty' t
    s.t. A x == b, G x - t <= h, t >= 0

    Args:
        Q (Array): Quadratic cost matrix, shape (n, n)
        q (Array): Quadratic cost vector, shape (n,)
        G (Array): Linear inequality constraint matrix, shape (p, n)
        h (Array): Linear inequality constraint vector, shape (p,)
        penalty (Array | float): L1 penalty, shape (p,) if vector
        A (Array, optional): Linear equality constraint matrix, shape (m, n).
            Defaults to None.
        b (Array, optional): Linear equality constraint vector, shape (m,).
            Defaults to None.
        method (str, optional): Backend (das/pdal/ipm). Defaults to "das".
        eps_abs (float, optional): Solve tolerance.
            Defaults to None (use default for backend)
        max_iter (int, optional): Max solver iterations (backend-dependent).
            Defaults to None (use default for backend).
        ruiz (bool, optional): Whether to use ruiz equilibration.
            Defaults to None (use default for backend).
        target_kappa (float, optional): Kappa-relaxation parameter for smooth
            derivatives. Defaults to 1e-3.
        warm_start (tuple | Result, optional): Explicit warm start, either a
            previous Result or a (x, y, z) tuple. Defaults to None.

    Returns:
        Result: Solution to the elastic QP
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "elastiqp.jax.solve requires double precision. Enable it via "
            "jax.config.update('jax_enable_x64', True) "
            "or set JAX_ENABLE_X64=1 in your environment."
        )
    if eps_abs is None:
        eps_abs = _DEFAULT_EPS_ABS[method]
    if max_iter is None:
        max_iter = _DEFAULT_MAX_ITER[method]
    if ruiz is None:
        ruiz = _DEFAULT_RUIZ[method]
    Q = jnp.asarray(Q)
    q = jnp.asarray(q)
    G = jnp.asarray(G)
    h = jnp.asarray(h)
    if Q.ndim != 2 or q.ndim != 1:
        raise ValueError("solve takes unbatched problems; use jax.vmap for batching")
    n = Q.shape[-1]
    if (A is None) != (b is None):
        raise ValueError("A and b must be provided together")
    if A is None:
        A = jnp.zeros(Q.shape[:-2] + (0, n), dtype=jnp.float64)
        b = jnp.zeros(Q.shape[:-2] + (0,), dtype=jnp.float64)
    else:
        A = jnp.asarray(A)
        b = jnp.asarray(b)
    # penalty may be a scalar or a per-constraint vector of length p
    penalty = jnp.broadcast_to(jnp.asarray(penalty, dtype=jnp.float64), h.shape)

    if warm_start is not None:
        if isinstance(warm_start, Result):
            x0, y0, z0 = warm_start.x, warm_start.y, warm_start.z
        else:
            x0, y0, z0 = warm_start
        x0 = jnp.asarray(x0, dtype=jnp.float64)
        y0 = jnp.asarray(y0, dtype=jnp.float64)
        z0 = jnp.asarray(z0, dtype=jnp.float64)
        if x0.shape != q.shape or y0.shape != b.shape or z0.shape != h.shape:
            raise ValueError(
                "warm_start (x, y, z) must have shapes "
                f"{q.shape}, {b.shape}, {h.shape}; got "
                f"{x0.shape}, {y0.shape}, {z0.shape}"
            )
        x, t, y, z_t, z, info = _solve_warm(
            Q,
            q,
            A,
            b,
            G,
            h,
            penalty,
            x0,
            y0,
            z0,
            float(eps_abs),
            int(max_iter),
            bool(ruiz),
            method,
        )
        return Result(
            x=x,
            t=t,
            y=y,
            z_t=z_t,
            z=z,
            converged=info[..., 0].astype(jnp.int32),
            iters=info[..., 1].astype(jnp.int32),
        )

    x, t, y, z_t, z, info = _solve(
        Q,
        q,
        A,
        b,
        G,
        h,
        penalty,
        float(eps_abs),
        int(max_iter),
        bool(ruiz),
        float(target_kappa),
        method,
    )
    # Recall: info = [converged, iters, relax_converged]. The converged
    # flag here considers both forward and backward (relax)
    converged = jnp.minimum(info[..., 0], info[..., 2])
    return Result(
        x=x,
        t=t,
        y=y,
        z_t=z_t,
        z=z,
        converged=converged.astype(jnp.int32),
        iters=info[..., 1].astype(jnp.int32),
    )
