"""ElastiQP JAX FFI wrapper

Supported:
- Different methods (das, pdal, ipm)
- jit, vmap, scan
- grad

Notes:
- vmap batches into a single FFI call, which solves the problems one by one
- Requires float64
- Warm-starting is differentiable (the warm start itself gets a zero gradient)
- Warm-starting requires explicit passing of the previous result
- Recommended default kappa for differentiability: 1e-3
"""

import ctypes
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

from elastiqp import _DEFAULT_MAX_ITER, _core

__all__ = ["solve", "Result", "METHODS"]

METHODS = ("das", "pdal", "ipm")
_METHOD_ID = {"das": 0, "pdal": 1, "ipm": 2}


def _find_library():
    # The FFI library is built and installed next to the compiled core, so
    # look wherever elastiqp._core was found (installed wheel, editable
    # install, ELASTIQP_BUILD_DIR, or <repo>/build).
    directory = Path(_core.__file__).resolve().parent
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
        f"elastiqp_jax_ffi shared library not found in {directory} (next to "
        "elastiqp._core). Reinstall the package, or for source checkouts "
        "build it first (cmake --build build)."
    )


_lib = ctypes.cdll.LoadLibrary(str(_find_library()))
jax.ffi.register_ffi_target(
    "elastiqp_solve", jax.ffi.pycapsule(_lib.ElastiqpSolve), platform="cpu"
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
    x0,
    y0,
    z0,
    eps_abs,
    max_iter,
    ruiz,
    target_kappa,
    warm,
    method,
):
    n = Q.shape[-1]
    m = b.shape[-1]
    p = h.shape[-1]
    batch = Q.shape[:-2]
    vec = lambda d: jax.ShapeDtypeStruct(batch + (d,), jnp.float64)
    # (x, t, y, z_t, z) tight solution, then the kappa-relaxed central
    # point (identical to the tight block when target_kappa <= 0), then
    # info = [converged, iters, relax_converged].
    sol = [vec(n), vec(p), vec(m), vec(p), vec(p)]
    out_types = sol + sol + [vec(3)]
    call = jax.ffi.ffi_call("elastiqp_solve", out_types, vmap_method="broadcast_all")
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
        target_kappa=np.float64(target_kappa),
        warm=np.int64(bool(warm)),
        method=np.int64(_METHOD_ID[method]),
    )


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
    # Hard rows (penalty = inf: t = 0, z_t = inf) have no t block.
    rt = jnp.where(jnp.isfinite(z_t), -z_t * z_tb + t * tb, 0.0)
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


# target_kappa is only used by _solve_fwd; custom_vjp needs the primal and
# fwd signatures to match. (x0, y0, z0) are ignored unless warm.
@partial(jax.custom_vjp, nondiff_argnums=(10, 11, 12, 13, 14, 15))
def _solve(
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
    target_kappa,
    warm,
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
        x0,
        y0,
        z0,
        eps_abs,
        max_iter,
        ruiz,
        0.0,
        warm,
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
    x0,
    y0,
    z0,
    eps_abs,
    max_iter,
    ruiz,
    target_kappa,
    warm,
    method,
):
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
        x0,
        y0,
        z0,
        eps_abs,
        max_iter,
        ruiz,
        target_kappa,
        warm,
        method,
    )
    x, t, y, z_t, z, xr, tr, yr, z_t_r, z_r, info = out
    return (x, t, y, z_t, z, info), (Q, A, G, h, xr, tr, yr, z_t_r, z_r)


def _solve_bwd(eps_abs, max_iter, ruiz, target_kappa, warm, method, res, ct):
    # Zero cotangent for the warm start: a converged solve of a QP with a
    # unique solution does not depend on where the solver starts. The
    # (x, y, z) cotangents share the (x0, y0, z0) shapes.
    xb, _, yb, _, zb, _ = ct
    zeros = (jnp.zeros_like(xb), jnp.zeros_like(yb), jnp.zeros_like(zb))
    return _kkt_bwd(res, ct) + zeros


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
    eps_abs: float = 1e-6,
    max_iter: int | None = None,
    ruiz: bool = True,
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
        eps_abs (float, optional): Absolute KKT tolerance (same meaning for
            every backend). Defaults to 1e-6.
        max_iter (int, optional): Outer iteration budget; what it counts
            depends on the backend (das: active-set iterations, pdal: BCL
            rounds, ipm: interior-point iterations). Defaults to None (the
            backend's Settings default).
        ruiz (bool, optional): Whether to use ruiz equilibration.
            Defaults to True.
        target_kappa (float, optional): Kappa-relaxation parameter for smooth
            derivatives. Defaults to 1e-3.
        warm_start (tuple | Result, optional): Explicit warm start, either a
            previous Result or a (x, y, z) tuple, (das and pdal only).
            Differentiable: the gradients are the cold solve's, and the warm
            start itself gets a zero gradient (exact when the QP has a unique
            solution and the solve converges). Defaults to None.

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
    if max_iter is None:
        max_iter = _DEFAULT_MAX_ITER[method]
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
        if method == "ipm":
            raise ValueError(
                "warm_start is not supported with method='ipm'; use 'das' or 'pdal'"
            )
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
    else:
        # Ignored by the solver (warm=False)
        x0 = jnp.zeros(q.shape, jnp.float64)
        y0 = jnp.zeros(b.shape, jnp.float64)
        z0 = jnp.zeros(h.shape, jnp.float64)
    x, t, y, z_t, z, info = _solve(
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
        float(target_kappa),
        warm_start is not None,
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
