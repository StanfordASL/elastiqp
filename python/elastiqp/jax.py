"""ElastiQP JAX FFI wrapper

``method`` selects the backend: "das" (dual active set, the default),
"pdal" (primal-dual augmented Lagrangian) or "ipm" (interior point).
Forward solves work with all three. Gradients need the kappa relaxation,
which only the PDAL and IPM backends have: with ``method="pdal"`` or
``"ipm"`` and ``target_kappa > 0`` (log-barrier smoothed gradients,
evaluated at the kappa-relaxed central point with complementarity
s.z = kappa; the default 1e-3 is qpax's) jax.grad works out of the box.
Differentiating an ``method="das"`` solve raises at trace time.

Set ruiz=True for badly-scaled data (the active-set backend has it on by
default).

Supports JIT and vmap (under "sequential" mode, which performs one solve
per entry in the batch). Requires float64 (JAX_ENABLE_X64).

Warm starting is explicit, to keep the call pure: pass the previous
``Result`` (or an ``(x, y, z)`` tuple) as ``warm_start=`` and the fresh
solver is seeded from it. Carry the Result through your loop (a
``lax.scan`` carry, a Python loop variable) like any other state. The
warm-started path is not differentiable.
"""

import ctypes
import os
from functools import partial
from pathlib import Path
from typing import NamedTuple

try:
    import jax
except ImportError as e:
    raise ImportError("elastiqp.jax requires jax to be installed") from e

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
    vmap_method,
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
    call = jax.ffi.ffi_call("elastiqp_solve", out_types, vmap_method=vmap_method)
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
    vmap_method,
    method,
):
    n = Q.shape[-1]
    m = b.shape[-1]
    p = h.shape[-1]
    batch = Q.shape[:-2]
    vec = lambda d: jax.ShapeDtypeStruct(batch + (d,), jnp.float64)
    # (x, t, y, z_t, z), then info = [converged, iters].
    out_types = [vec(n), vec(p), vec(m), vec(p), vec(p), vec(2)]
    call = jax.ffi.ffi_call("elastiqp_solve_warm", out_types, vmap_method=vmap_method)
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


@partial(jax.custom_vjp, nondiff_argnums=(10, 11, 12, 13, 14))
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
    vmap_method,
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
        vmap_method,
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


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12))
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
    vmap_method,
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
        vmap_method,
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
    vmap_method,
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
        vmap_method,
        target_kappa,
        method,
    )
    x, t, y, z_t, z, xr, tr, yr, z_t_r, z_r, info = out
    return (x, t, y, z_t, z, info), (Q, A, G, h, xr, tr, yr, z_t_r, z_r)


def _solve_bwd(eps_abs, max_iter, ruiz, vmap_method, target_kappa, method, res, ct):
    return _kkt_bwd(res, ct)


_solve.defvjp(_solve_fwd, _solve_bwd)


def solve(
    Q,
    q,
    G,
    h,
    penalty,
    *,
    A=None,
    b=None,
    method="das",
    eps_abs=None,
    max_iter=None,
    ruiz=None,
    target_kappa=1e-3,
    vmap_method="sequential",
    warm_start=None,
):
    """Solve the elastic QP

        min 0.5 x'Qx + q'x + penalty' t
        s.t. A x == b, G x - t <= h, t >= 0

    `penalty` may be a scalar or a per-constraint vector of length p.
    `method`, `eps_abs`, `max_iter`, `ruiz` and `target_kappa` are static
    (compile-time) options.

    `method` selects the backend: "das" (dual active set, the default),
    "pdal" (primal-dual augmented Lagrangian) or "ipm" (interior point).
    `max_iter` is the backend's outer budget (active-set iterations, BCL
    rounds, interior-point iterations); None uses the backend default
    (10000 / 250 / 250). `ruiz=None` likewise uses the backend default
    (on for "das", off for "pdal" / "ipm"); set it explicitly for
    badly-scaled data.

    `warm_start` seeds the solve from a previous point: a `Result` from an
    earlier call (of any method) or an `(x, y, z)` tuple with shapes (n,),
    (m,), (p,). The C++ solver is still constructed fresh (the call stays
    pure, so it composes with jit, vmap and lax.scan; carry the Result as
    loop state), but starts from that point: the active-set backend reads
    its working set off z, PDAL / IPM start their iterates there. This
    keeps the iteration savings of a persistent solver but not its cached
    factorization. A warm-started solve is never differentiable (jax.grad
    raises at trace time). `y` must be empty when there are no equalities.

    Without `warm_start` every call is a cold solve. With method "pdal" or
    "ipm" it is differentiable in reverse mode w.r.t. all array arguments when
    target_kappa > 0 (the default); jax.grad with method="das" or with an
    explicit target_kappa=0 raises at trace time (the active-set backend
    has no relaxation; the certificate sits exactly on the constraint
    boundary, where the exact KKT derivative is undefined).
    `ruiz=True` enables Ruiz equilibration for badly-scaled data; the
    solver terminates on and returns unscaled quantities, so it does not
    affect gradients.

    `target_kappa` controls gradient smoothing (qpax-style): kappa > 0
    (the default 1e-3 is also qpax's) differentiates at a kappa-relaxed
    central point with complementarity s.z = kappa, giving smoothed,
    well-conditioned gradients near active-set changes at the cost of an
    O(kappa) bias. With large penalty weights (>= ~1e4) the corrector's
    roundoff amplification grows as penalty^2 / kappa and stalls it at
    small kappa (check `converged`); enable ruiz=True in that regime,
    which rescales the penalties to O(1) and removes the amplification.
    The relaxed point is reached by a Newton corrector in the log-barrier
    retraction coordinates z = b_kappa(v), s = b_kappa(-v). The relaxation
    only runs on the differentiation path: a plain (undifferentiated)
    solve never pays for it, and the returned solution is always the
    tight (unrelaxed) optimum. `converged` covers everything the call
    computed: when differentiating with kappa > 0 it is 0 if either the
    solve or the relaxation failed, so a bad gradient evaluation point is
    never silent.
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
            vmap_method,
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
        vmap_method,
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
