"""ElastiQP PyTorch interface

The torch analogue of the JAX FFI (elastiqp.jax): the solve is registered as
a custom operator (``torch.ops.elastiqp.solve``), so it composes with
autograd and torch.compile as an opaque primitive rather than as unrolled
solver iterations, and the forward and backward passes each cross into C++
exactly once per problem.

``method`` selects the backend: "das" (dual active set, the default),
"pdal" (primal-dual augmented Lagrangian) or "ipm" (interior point).
Forward solves work with all three. Gradients need the kappa relaxation,
which only the PDAL and IPM backends have: with ``method="pdal"`` or
``"ipm"`` and ``target_kappa > 0`` (log-barrier smoothed gradients,
evaluated at the kappa-relaxed central point with complementarity
s.z = kappa; the default 1e-3 is qpax's) loss.backward() works out of the
box. Differentiating a ``method="das"`` solve raises.

Set ruiz=True for badly-scaled data (the active-set backend has it on by
default).

Leading batch dimensions are supported directly (one solve per entry;
batch shapes of the arguments broadcast) and via torch.vmap. Gradients
come from autograd (loss.backward(), torch.autograd.grad) or the
functional torch.func.grad / jacrev transforms. The solve runs on the CPU
in float64: inputs on other devices / in float32 are moved and upcast,
outputs are float64 on the input device, and gradients flow back through
the casts.

Currently, does not support warm-starting: every call is a cold solve.
"""

from typing import NamedTuple, Tuple

try:
    import torch
except ImportError as e:
    raise ImportError("elastiqp.torch requires torch to be installed") from e

import numpy as np

from elastiqp import _core

__all__ = ["solve", "Result", "METHODS"]

METHODS = ("das", "pdal", "ipm")
# Default outer budget per backend (active-set iterations / BCL rounds /
# interior-point iterations), used when max_iter is None.
_DEFAULT_MAX_ITER = {"das": 10000, "pdal": 250, "ipm": 250}
# Backend eps_abs defaults (see Settings in each header).
_DEFAULT_EPS_ABS = {"das": 1e-6, "pdal": 1e-5, "ipm": 1e-5}
# Ruiz equilibration default per backend (on for the active set, whose LDP
# conditioning depends on it; off for PDAL / IPM), used when ruiz is None.
_DEFAULT_RUIZ = {"das": True, "pdal": False, "ipm": False}

if not hasattr(torch.library, "custom_op"):
    raise ImportError("elastiqp.torch requires torch >= 2.4 (torch.library.custom_op)")


class Result(NamedTuple):
    x: torch.Tensor
    t: torch.Tensor  # elastic slacks (per-row constraint violations)
    y: torch.Tensor  # equality duals (empty if no equalities)
    z_t: torch.Tensor  # duals of t >= 0
    z: torch.Tensor  # duals of G x - t <= h, in [0, penalty]
    converged: torch.Tensor  # 0/1; includes the kappa relaxation when it runs
    iters: torch.Tensor


_NOT_DIFFERENTIABLE_MSG = (
    "elastiqp.torch.solve is not differentiable with target_kappa=0: "
    "the solution sits exactly on the constraint boundary, where "
    "the exact KKT derivative is undefined. Set target_kappa > 0 "
    "(e.g. 1e-3) for log-barrier smoothed gradients"
)
_DAS_NOT_DIFFERENTIABLE_MSG = (
    "elastiqp.torch.solve is not differentiable with method='das': the "
    "active-set backend has no kappa relaxation. Use method='pdal' or "
    "method='ipm' (with target_kappa > 0) for gradients"
)


def _np(x: torch.Tensor) -> np.ndarray:
    # CPU float64 tensor -> numpy view (no copy unless non-contiguous).
    x = x.detach()
    return x.numpy() if x.is_contiguous() else x.contiguous().numpy()


def _requires_grad(x: torch.Tensor) -> bool:
    # Under torch.vmap the wrapper sees BatchedTensors, which report
    # requires_grad=False regardless of the tensor they wrap; look through.
    try:
        from torch._C._functorch import get_unwrapped, is_batchedtensor

        while is_batchedtensor(x):
            x = get_unwrapped(x)
    except ImportError:  # pragma: no cover
        pass
    return x.requires_grad


# --- the primitive -----------------------------------------------------------
#
# (Q, q, A, b, G, h, penalty) -> (x, t, y, z_t, z,
#                                  xr, tr, yr, z_t_r, z_r, info)
#
# Same contract as the JAX FFI handler: the first block is the tight
# solution, the second the kappa-relaxed central point used as the
# differentiation point (a copy of the tight block when target_kappa <= 0),
# and info = [converged, iters, relax_converged]. All float64 on the CPU;
# arbitrary leading batch dimensions, one C++ solve per entry.
#
# The same two implementation functions (_solve_impl, _vjp_impl) are exposed
# twice: as a functorch-style autograd.Function for eager execution (cheap
# dispatch; supports torch.vmap and torch.func.grad), and as custom
# operators for torch.compile, where the solve must be an opaque op with a
# shape function. solve() routes between them.

_SOLVE_OUT = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _solve_impl(
    Q: torch.Tensor,
    q: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    G: torch.Tensor,
    h: torch.Tensor,
    penalty: torch.Tensor,
    eps_abs: float,
    max_iter: int,
    ruiz: bool,
    target_kappa: float,
    method: str,
) -> _SOLVE_OUT:
    batch = tuple(Q.shape[:-2])
    n, m, p = Q.shape[-1], b.shape[-1], h.shape[-1]
    if not batch:  # fast path: no reshapes, C++ outputs handed to torch as-is
        res = _core._solve_relaxed(
            _np(Q),
            _np(q),
            _np(A),
            _np(b),
            _np(G),
            _np(h),
            _np(penalty),
            eps_abs,
            max_iter,
            ruiz,
            target_kappa,
            method,
        )
        return tuple(torch.from_numpy(r) for r in res)
    nb = int(np.prod(batch))
    flat = lambda x, d: _np(x).reshape((nb,) + d)
    Qf, qf = flat(Q, (n, n)), flat(q, (n,))
    Af, bf = flat(A, (m, n)), flat(b, (m,))
    Gf, hf = flat(G, (p, n)), flat(h, (p,))
    pf = flat(penalty, (p,))
    dims = (n, p, m, p, p, n, p, m, p, p, 3)
    out = [np.empty((nb, d)) for d in dims]
    for i in range(nb):
        res = _core._solve_relaxed(
            Qf[i],
            qf[i],
            Af[i],
            bf[i],
            Gf[i],
            hf[i],
            pf[i],
            eps_abs,
            max_iter,
            ruiz,
            target_kappa,
            method,
        )
        for o, r in zip(out, res):
            o[i] = r
    return tuple(torch.from_numpy(o).reshape(batch + (d,)) for o, d in zip(out, dims))


_solve_op = torch.library.custom_op(
    "elastiqp::solve", _solve_impl, mutates_args=(), device_types="cpu"
)


@_solve_op.register_fake
def _(Q, q, A, b, G, h, penalty, eps_abs, max_iter, ruiz, target_kappa, method):
    batch = Q.shape[:-2]
    n, m, p = Q.shape[-1], b.shape[-1], h.shape[-1]
    dims = (n, p, m, p, p, n, p, m, p, p, 3)
    return tuple(Q.new_empty(batch + (d,)) for d in dims)


def _setup_context(ctx, inputs, output):
    Q, q, A, b, G, h, penalty, eps_abs, max_iter, ruiz, target_kappa, method = inputs
    xr, tr, yr, z1r, z2r = output[5:10]
    # Differentiate at the kappa-relaxed point (the tight block is the value).
    ctx.save_for_backward(Q, A, G, h, xr, tr, yr, z1r, z2r)
    ctx.target_kappa = target_kappa
    ctx.method = method


# The backward is itself a custom op so that torch.compile / AOTAutograd can
# trace the backward graph (with fake tensors) without executing it. Zero
# cotangents are passed as size-0 vectors, which the C++ treats as zero.
def _vjp_impl(
    Q: torch.Tensor,
    A: torch.Tensor,
    G: torch.Tensor,
    h: torch.Tensor,
    xr: torch.Tensor,
    tr: torch.Tensor,
    yr: torch.Tensor,
    z1r: torch.Tensor,
    z2r: torch.Tensor,
    ct_x: torch.Tensor,
    ct_t: torch.Tensor,
    ct_y: torch.Tensor,
    ct_z_t: torch.Tensor,
    ct_z: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch = tuple(Q.shape[:-2])
    if not batch:
        res = _core._kkt_vjp(
            _np(Q),
            _np(A),
            _np(G),
            _np(h),
            _np(xr),
            _np(tr),
            _np(yr),
            _np(z1r),
            _np(z2r),
            _np(ct_x),
            _np(ct_t),
            _np(ct_y),
            _np(ct_z_t),
            _np(ct_z),
        )
        return tuple(torch.from_numpy(r) for r in res)
    n, m, p = Q.shape[-1], A.shape[-2], G.shape[-2]
    nb = int(np.prod(batch))
    flat = lambda x, d: _np(x).reshape((nb,) + d)
    ctf = [flat(c, (c.shape[-1],)) for c in (ct_x, ct_t, ct_y, ct_z_t, ct_z)]
    Qf, Af, Gf, hf = flat(Q, (n, n)), flat(A, (m, n)), flat(G, (p, n)), flat(h, (p,))
    xf, tf, yf = flat(xr, (n,)), flat(tr, (p,)), flat(yr, (m,))
    z1f, z2f = flat(z1r, (p,)), flat(z2r, (p,))
    dims = ((n, n), (n,), (m, n), (m,), (p, n), (p,), (p,))
    out = [np.empty((nb,) + d) for d in dims]
    for i in range(nb):
        res = _core._kkt_vjp(
            Qf[i],
            Af[i],
            Gf[i],
            hf[i],
            xf[i],
            tf[i],
            yf[i],
            z1f[i],
            z2f[i],
            ctf[0][i],
            ctf[1][i],
            ctf[2][i],
            ctf[3][i],
            ctf[4][i],
        )
        for o, r in zip(out, res):
            o[i] = r
    return tuple(torch.from_numpy(o).reshape(batch + d) for o, d in zip(out, dims))


_kkt_vjp = torch.library.custom_op(
    "elastiqp::kkt_vjp", _vjp_impl, mutates_args=(), device_types="cpu"
)


@_kkt_vjp.register_fake
def _(Q, A, G, h, xr, tr, yr, z1r, z2r, ct_x, ct_t, ct_y, ct_z_t, ct_z):
    batch = Q.shape[:-2]
    n, m, p = Q.shape[-1], A.shape[-2], G.shape[-2]
    dims = ((n, n), (n,), (m, n), (m,), (p, n), (p,), (p,))
    return tuple(Q.new_empty(batch + d) for d in dims)


def _backward(ctx, grads, vjp):
    if ctx.method == "das":
        raise RuntimeError(_DAS_NOT_DIFFERENTIABLE_MSG)
    if not ctx.target_kappa > 0:
        raise RuntimeError(_NOT_DIFFERENTIABLE_MSG)
    saved = ctx.saved_tensors
    Q = saved[0]
    # Cotangents of the tight solution (x, t, y, z_t, z). The relaxed
    # block and info are internal and never carry a gradient. Missing
    # cotangents travel as size-0 vectors (zero).
    zero = Q.new_zeros(Q.shape[:-2] + (0,))
    ct = tuple(zero if c is None else c for c in grads[:5])
    return tuple(vjp(*saved, *ct)) + (None, None, None, None, None)


def _vmap_rule(call, info, in_dims, Q, q, A, b, G, h, penalty, *opts):
    # The primitive takes leading batch dims natively, so the vmap rule just
    # moves the mapped dim to the front (broadcasting unmapped arguments) and
    # calls it once. Nested vmaps stack batch dims the same way.
    B = info.batch_size

    def bat(x, d):
        if d is None:
            return x.unsqueeze(0).expand((B,) + tuple(x.shape))
        return x.movedim(d, 0)

    args = [bat(x, d) for x, d in zip((Q, q, A, b, G, h, penalty), in_dims[:7])]
    out = call(*args, *opts)
    return out, tuple(0 for _ in out)


_solve_op.register_autograd(
    lambda ctx, *grads: _backward(ctx, grads, _kkt_vjp), setup_context=_setup_context
)
_solve_op.register_vmap(
    lambda info, in_dims, *args: _vmap_rule(_solve_op, info, in_dims, *args)
)


class _Solve(torch.autograd.Function):
    """Eager path: the same primitive without the custom-op dispatch cost."""

    @staticmethod
    def forward(
        Q, q, A, b, G, h, penalty, eps_abs, max_iter, ruiz, target_kappa, method
    ):
        return _solve_impl(
            Q, q, A, b, G, h, penalty, eps_abs, max_iter, ruiz, target_kappa, method
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        _setup_context(ctx, inputs, output)

    @staticmethod
    def backward(ctx, *grads):
        # Inside torch.func transforms (grad, jacrev, vmap(grad)) the
        # cotangents and saved tensors arrive as functorch wrappers that
        # cannot cross into C++, and the backward itself must be built from
        # transformable ops: use the torch-native VJP there. Plain autograd
        # takes the C++ call.
        tensors = ctx.saved_tensors + tuple(g for g in grads if g is not None)
        wrapped = any(map(torch._C._functorch.is_functorch_wrapped_tensor, tensors))
        return _backward(ctx, grads, _vjp_torch if wrapped else _vjp_impl)

    @staticmethod
    def vmap(info, in_dims, *args):
        return _vmap_rule(_Solve.apply, info, in_dims, *args)


def _vjp_torch(Q, A, G, h, x, t, y, z1, z2, ct_x, ct_t, ct_y, ct_z_t, ct_z):
    """KKT VJP in torch ops, for the torch.func transforms.

    Port of _kkt_bwd in python/elastiqp/jax.py (see its docstring for the
    derivation; include/elastiqp/kkt_vjp.hpp is the C++ mirror used on the
    plain autograd path). Same argument convention as _vjp_impl: size-0
    cotangents mean zero. tests/test_torch.py pins this against the C++
    path.
    """
    n, m, p = Q.shape[-1], A.shape[-2], G.shape[-2]
    fill = lambda c, d: c if c.shape[-1] == d else Q.new_zeros(Q.shape[:-2] + (d,))
    xb, tb, yb = fill(ct_x, n), fill(ct_t, p), fill(ct_y, m)
    z1b, z2b = fill(ct_z_t, p), fill(ct_z, p)

    Qs = 0.5 * (Q + Q.transpose(-1, -2))
    Gt = G.transpose(-1, -2)
    At = A.transpose(-1, -2)
    mv = lambda M, v: (M @ v.unsqueeze(-1)).squeeze(-1)
    outer = lambda a, c: a.unsqueeze(-1) * c.unsqueeze(-2)

    D = mv(G, x) - t - h  # = -s2 <= 0
    E = D - z2 * t / z1  # < 0 at any interior point

    r1, r2, r3 = xb, tb, yb
    r4 = -z1 * z1b
    r5 = z2 * z2b
    r5t = r5 + z2 * (r4 + t * r2) / z1
    rhs_x = r1 - mv(Gt, r5t / E)

    H = Qs + Gt @ ((-z2 / E).unsqueeze(-1) * G)
    KKT = torch.cat(
        [
            torch.cat([H, At], dim=-1),
            torch.cat([A, A.new_zeros(A.shape[:-2] + (m, m))], dim=-1),
        ],
        dim=-2,
    )
    sol = torch.linalg.solve(KKT, torch.cat([rhs_x, r3], dim=-1))
    v1 = sol[..., :n]
    v3 = sol[..., n:]

    v5 = (r5t - z2 * mv(G, v1)) / E
    v2 = (r4 + t * r2 + t * v5) / z1

    Qb = -0.5 * (outer(v1, x) + outer(x, v1))
    Ab = -(outer(y, v1) + outer(v3, x))
    Gb = -(outer(z2, v1) + outer(v5, x))
    return Qb, -v1, Ab, v3, Gb, v5, -v2


# --- public API --------------------------------------------------------------


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
):
    """Solve the elastic QP

        min 0.5 x'Qx + q'x + penalty' t
        s.t. A x == b (hard, optional), G x - t <= h, t >= 0

    `penalty` may be a scalar or a per-constraint vector of length p.
    `method`, `eps_abs`, `max_iter`, `ruiz` and `target_kappa` are Python
    scalars (constants under torch.compile).

    `method` selects the backend: "das" (dual active set, the default),
    "pdal" (primal-dual augmented Lagrangian) or "ipm" (interior point).
    `max_iter` is the backend's outer budget (active-set iterations, BCL
    rounds, interior-point iterations); None uses the backend default
    (10000 / 250 / 250). `ruiz=None` likewise uses the backend default
    (on for "das", off for "pdal" / "ipm"); set it explicitly for
    badly-scaled data.

    Leading batch dimensions (Q: (..., n, n), q: (..., n), ...) solve one
    problem per entry; the batch shapes of the arguments broadcast against
    each other, so a single Q may be shared across a batch of q.

    Every call here is a cold solve. With method "pdal" or "ipm" it is
    differentiable in reverse mode w.r.t. all tensor arguments when
    target_kappa > 0 (the default); a solve with method="das" or with an
    explicit target_kappa=0 raises when some input requires grad (the
    active-set backend has no relaxation; the certificate sits exactly on
    the constraint boundary, where the exact KKT derivative is undefined). `ruiz=True` enables Ruiz equilibration for badly-scaled
    data; the solver terminates on and returns unscaled quantities, so it
    does not affect gradients.

    `target_kappa` controls gradient smoothing (qpax-style): kappa > 0
    (the default 1e-3 is also qpax's) differentiates at a kappa-relaxed
    central point with complementarity s.z = kappa, giving smoothed,
    well-conditioned gradients near active-set changes at the cost of an
    O(kappa) bias. With large penalty weights (>= ~1e4) the corrector's
    roundoff amplification grows as penalty^2 / kappa and stalls it at
    small kappa (check `converged`); enable ruiz=True in that regime,
    which rescales the penalties to O(1) and removes the amplification.
    The relaxation only runs when some input requires grad (and grad mode
    is on): a plain solve never pays for it, and the returned solution is
    always the tight (unrelaxed) optimum. `converged` covers everything
    the call computed: when differentiating with kappa > 0 it is 0 if
    either the solve or the relaxation failed, so a bad gradient
    evaluation point is never silent.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if eps_abs is None:
        eps_abs = _DEFAULT_EPS_ABS[method]
    if max_iter is None:
        max_iter = _DEFAULT_MAX_ITER[method]
    if ruiz is None:
        ruiz = _DEFAULT_RUIZ[method]
    Q = torch.as_tensor(Q)
    device = Q.device
    to = lambda x: torch.as_tensor(x, device=device).to(torch.float64)
    Q, q, G, h = to(Q), to(q), to(G), to(h)
    if Q.ndim < 2 or q.ndim < 1 or G.ndim < 2 or h.ndim < 1:
        raise ValueError("Q and G must be matrices, q and h vectors")
    n = Q.shape[-1]
    p = h.shape[-1]
    if (A is None) != (b is None):
        raise ValueError("A and b must be provided together")
    if A is None:
        A = Q.new_zeros((0, n))
        b = Q.new_zeros((0,))
    else:
        A, b = to(A), to(b)
    m = b.shape[-1]
    # penalty may be a scalar or a per-constraint vector of length p
    penalty = to(penalty)
    if penalty.ndim == 0:
        penalty = penalty.expand(p)

    # Broadcast the batch prefixes so a shared Q can pair with a batch of q
    # (skipped on the common path where every argument has the same prefix).
    core = ((Q, 2), (q, 1), (A, 2), (b, 1), (G, 2), (h, 1), (penalty, 1))
    prefixes = [tuple(x.shape[: x.ndim - k]) for x, k in core]
    batch = prefixes[0]
    if any(pre != batch for pre in prefixes):
        batch = tuple(torch.broadcast_shapes(*prefixes))
        Q, q, A, b, G, h, penalty = (
            x.expand(batch + tuple(x.shape[x.ndim - k :])) for x, k in core
        )
    if Q.shape[-2:] != (n, n) or q.shape[-1] != n or G.shape[-1] != n:
        raise ValueError("inconsistent dimension n across Q, q, G")
    if G.shape[-2] != p or penalty.shape[-1] != p:
        raise ValueError("inconsistent dimension p across G, h, penalty")
    if A.shape[-2:] != (m, n):
        raise ValueError("inconsistent equality dimensions across A, b")

    args = (Q, q, A, b, G, h, penalty)
    differentiating = torch.is_grad_enabled() and any(_requires_grad(x) for x in args)
    if differentiating and method == "das":
        raise TypeError(_DAS_NOT_DIFFERENTIABLE_MSG)
    if differentiating and not target_kappa > 0:
        raise TypeError(_NOT_DIFFERENTIABLE_MSG)
    # Tight solution only when not differentiating: no relaxation runs.
    kappa = float(target_kappa) if differentiating else 0.0

    # Under torch.compile the solve must be an opaque custom op (with a
    # shape function); eagerly, the autograd.Function is much cheaper.
    call = _solve_op if torch.compiler.is_compiling() else _Solve.apply
    if device.type != "cpu":
        args = tuple(x.cpu() for x in args)
    out = call(*args, float(eps_abs), int(max_iter), bool(ruiz), kappa, method)
    x, t, y, z_t, z = (o.to(device) for o in out[:5])
    info = out[10]
    # info = [converged, iters, relax_converged]: converged folds in the
    # relaxation, which agrees with the tight flag whenever it does not run.
    converged = torch.minimum(info[..., 0], info[..., 2])
    return Result(
        x=x,
        t=t,
        y=y,
        z_t=z_t,
        z=z,
        converged=converged.to(torch.int32).to(device),
        iters=info[..., 1].to(torch.int32).to(device),
    )
