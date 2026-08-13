"""ElastiQP JAX FFI wrapper

Solves are differentiable with ``target_kappa > 0`` (log-barrier smoothed
gradients, evaluated at the kappa-relaxed central point with
complementarity s.z = kappa). The default target_kappa=1e-3 (qpax's
default) makes jax.grad work out of the box; set it to 0 to forbid
differentiation.

Set ruiz=True for badly-scaled data.

Supports JIT and vmap (under "sequential" mode, which performs one solve
per entry in the batch). Requires float64 (JAX_ENABLE_X64).

Currently, does not support warm-starting for functional purity
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

__all__ = ["solve", "Result"]


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
    "elastiqp_pdal_solve", jax.ffi.pycapsule(_lib.ElastiqpPdalSolve), platform="cpu"
)


class Result(NamedTuple):
    x: jax.Array
    t: jax.Array  # elastic slacks (per-row constraint violations)
    y: jax.Array  # equality duals (empty if no equalities)
    z_t: jax.Array  # duals of t >= 0
    z_ineq: jax.Array  # duals of G x - t <= h
    converged: jax.Array  # 0/1; includes the kappa relaxation when it runs
    iters: jax.Array


def _ffi_pdal_solve(
    Q, q, A, b, G, h, penalty, eps_abs, max_iter, ruiz, vmap_method, target_kappa
):
    n = Q.shape[-1]
    m = b.shape[-1]
    p = h.shape[-1]
    batch = Q.shape[:-2]
    vec = lambda d: jax.ShapeDtypeStruct(batch + (d,), jnp.float64)
    # (x, t, y, z_t, z_ineq) tight solution, then the kappa-relaxed central
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
    call = jax.ffi.ffi_call("elastiqp_pdal_solve", out_types, vmap_method=vmap_method)
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
    )


def _outer(a, b):
    return a[..., :, None] * b[..., None, :]


def _kkt_bwd(res, ct):
    """Implicit differentiation of the elastic KKT conditions.

    MIRROR: include/elastiqp/kkt_vjp.hpp is the C++ port of this function
    (for gradients without JAX); keep the two in sync. tests/test_pdal.cc
    pins the C++ side against finite differences of the relaxed map,
    tests/test_jax_ffi.py pins this one.

    Below, z1 and z2 are the duals Result calls z_t and z_ineq; the numeric
    subscripts keep the block algebra (g4/g5, v2/v5, rb1/rb2) readable.

    F(w, theta) = 0 at the solution, with w = (x, t, y, z1, z2) and
    theta = (Q, q, A, b, G, h, penalty):
        g1: Q x + q + A' y + G' z2      = 0
        g2: penalty - z1 - z2           = 0
        g3: A x - b                     = 0
        g4: z1 * t                      = 0
        g5: z2 * (G x - t - h)          = 0
    Solve K' u = wbar with K = dF/dw, then theta_bar = -(dF/dtheta)' u.

    The solve is CONDENSED, mirroring the forward solver: K is never formed.
    Scaling rows g4 by -1/z1 and g5 by 1/z2 makes K symmetric (S = D_r K),
    so the transpose solve K'u = wbar becomes the ordinary solve
    K v = D_r^{-1} wbar = (xb, tb, yb, -z1*z1b, z2*z2b) with u = D_r v.
    Eliminating the diagonal t/z1/z2 blocks of K v = r:
        v4 = -r2 - v5
        v2 = (r4 + t*r2 + t*v5) / z1
        E  = D - z2*t/z1  (diagonal, < 0 at any interior point)
        v5 = (r5 + z2*(r4 + t*r2)/z1 - z2*(G v1)) / E
    leaves the (n+m) saddle system
        [Q + G' diag(-z2/E) G,  A'] [v1]   [r1 - G'(r5_tilde/E)]
        [A,                     0 ] [v3] = [r3]
    -- O(p n^2 + (n+m)^3) instead of O((n+3p+m)^3) for the dense Jacobian.

    With kappa relaxation, g4 and g5 carry constant offsets (z1*t = kappa,
    z2*(Gx - t - h) = -kappa); constants drop out of both dF/dw and
    dF/dtheta, so the formulas are unchanged -- only the evaluation point
    (the relaxed solution stored in res) moves. At that point every
    complementarity pair has margin ~kappa, which bounds the conditioning
    near degenerate active sets.
    """
    Q, A, G, h, x, t, y, z1, z2 = res  # (x..z2) = relaxed point if kappa > 0
    xb, tb, yb, z1b, z2b, _ = ct
    n = Q.shape[-1]
    m = A.shape[-2]

    Qs = 0.5 * (Q + jnp.swapaxes(Q, -1, -2))
    Gt = jnp.swapaxes(G, -1, -2)
    At = jnp.swapaxes(A, -1, -2)
    mv = lambda M, v: jnp.einsum("...ij,...j->...i", M, v)

    D = mv(G, x) - t - h  # = -s2 <= 0
    E = D - z2 * t / z1  # < 0 at any interior point

    # Rescaled rhs for the symmetrized transpose solve.
    r1, r2, r3 = xb, tb, yb
    r4 = -z1 * z1b
    r5 = z2 * z2b

    r5t = r5 + z2 * (r4 + t * r2) / z1
    rhs_x = r1 - mv(Gt, r5t / E)

    # (n+m) saddle system for (v1, v3).
    H = Qs + jnp.einsum("...ji,...j,...jk->...ik", G, -z2 / E, G)
    KKT = jnp.concatenate(
        [
            jnp.concatenate([H, At], axis=-1),
            jnp.concatenate([A, jnp.zeros(A.shape[:-2] + (m, m), Q.dtype)], axis=-1),
        ],
        axis=-2,
    )
    sol = jnp.linalg.solve(KKT, jnp.concatenate([rhs_x, r3], axis=-1))
    v1 = sol[..., :n]
    v3 = sol[..., n:]

    v5 = (r5t - z2 * mv(G, v1)) / E
    # u = D_r v: u1 = v1, u2 = v2, u3 = v3, u5 = v5/z2 (u4 unused).
    v2 = (r4 + t * r2 + t * v5) / z1

    qb = -v1
    penalty_b = -v2
    bb = v3
    hb = v5  # = z2 * u5
    Qb = -0.5 * (_outer(v1, x) + _outer(x, v1))
    Ab = -(_outer(y, v1) + _outer(v3, x))
    Gb = -(_outer(z2, v1) + _outer(v5, x))
    return Qb, qb, Ab, bb, Gb, hb, penalty_b


# custom_vjp-wrapped solve.
#
# The primal path always runs the plain (tight) solve; only the
# differentiation path (fwd) pays for the kappa relaxation, and the bwd
# differentiates the KKT system at the relaxed point via _kkt_bwd. fwd
# raises unless target_kappa > 0: the solver's certificate sits exactly on
# the constraint boundary, where the exact KKT derivative is undefined
# (division by zero complementarity margins).
@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11))
def _solve(
    Q, q, A, b, G, h, penalty, eps_abs, max_iter, ruiz, vmap_method, target_kappa
):
    # Tight solution when not differentiating: no relaxation runs.
    out = _ffi_pdal_solve(
        Q, q, A, b, G, h, penalty, eps_abs, max_iter, ruiz, vmap_method, 0.0
    )
    return tuple(out[:5]) + (out[10],)  # solution + info


def _solve_fwd(
    Q, q, A, b, G, h, penalty, eps_abs, max_iter, ruiz, vmap_method, target_kappa
):
    if not target_kappa > 0:
        raise TypeError(
            "elastiqp.jax.solve is not differentiable with target_kappa=0: "
            "the solution sits exactly on the constraint boundary, where "
            "the exact KKT derivative is undefined. Set target_kappa > 0 "
            "(e.g. 1e-3) for log-barrier smoothed gradients"
        )
    out = _ffi_pdal_solve(
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
    )
    x, t, y, z1, z2, xr, tr, yr, z1r, z2r, info = out
    # Differentiate at the kappa-relaxed point; the returned VALUE is
    # always the tight solution.
    return (x, t, y, z1, z2, info), (Q, A, G, h, xr, tr, yr, z1r, z2r)


def _solve_bwd(eps_abs, max_iter, ruiz, vmap_method, target_kappa, res, ct):
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
    eps_abs=1e-5,
    max_iter=250,
    ruiz=False,
    target_kappa=1e-3,
    vmap_method="sequential",
):
    """Solve the elastic QP

        min 0.5 x'Qx + q'x + penalty' t
        s.t. A x == b (hard, optional), G x - t <= h, t >= 0

    `penalty` may be a scalar or a per-constraint vector of length p.
    `eps_abs`, `max_iter`, `ruiz` and `target_kappa` are static
    (compile-time) options.

    Every call here is a cold solve. It is differentiable in reverse mode
    w.r.t. all array arguments when target_kappa > 0 (the default);
    jax.grad with an explicit target_kappa=0 raises at trace time (the
    certificate sits exactly on the constraint boundary, where the exact
    KKT derivative is undefined).
    `ruiz=True` enables Ruiz equilibration for badly-scaled data; the
    solver terminates on and returns unscaled quantities, so it does not
    affect gradients.

    `target_kappa` controls gradient smoothing (qpax-style): kappa > 0
    (the default 1e-3 is also qpax's) differentiates at a kappa-relaxed
    central point with complementarity s.z = kappa, giving smoothed,
    well-conditioned gradients near active-set changes at the cost of an
    O(kappa) bias. Values below ~1e-3 get increasingly hard for the
    relaxation Newton corrector on ill-conditioned problems (it stalls on
    the humanoid-WBC benchmark at kappa <= 1e-4); check `converged`.
    The relaxed point is reached by a Newton corrector in the log-barrier
    retraction coordinates z = b_kappa(v), s = b_kappa(-v). The relaxation
    only runs on the differentiation path: a plain (undifferentiated)
    solve never pays for it, and the returned solution is always the
    tight (unrelaxed) optimum. `converged` covers everything the call
    computed: when differentiating with kappa > 0 it is 0 if either the
    solve or the relaxation failed, so a bad gradient evaluation point is
    never silent.
    """
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

    x, t, y, z_t, z_ineq, info = _solve(
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
    )
    # Recall: info = [converged, iters, relax_converged]. The converged
    # flag here considers both forward and backward (relax); the two agree
    # whenever no relaxation runs.
    converged = jnp.minimum(info[..., 0], info[..., 2])
    return Result(
        x=x,
        t=t,
        y=y,
        z_t=z_t,
        z_ineq=z_ineq,
        converged=converged.astype(jnp.int32),
        iters=info[..., 1].astype(jnp.int32),
    )
