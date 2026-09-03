"""ElastiQP PyTorch interface

The torch analogue of the JAX FFI (elastiqp.jax): the solve is registered as
a custom operator (``torch.ops.elastiqp.solve``), so it composes with
autograd and torch.compile as an opaque primitive rather than as unrolled
solver iterations, and the forward and backward passes each cross into C++
exactly once per problem.

Solves are differentiable with ``target_kappa > 0`` (log-barrier smoothed
gradients, evaluated at the kappa-relaxed central point with complementarity
s.z = kappa). The default target_kappa=1e-3 (qpax's default) makes
loss.backward() work out of the box; set it to 0 to forbid differentiation.

Set ruiz=True for badly-scaled data.

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

__all__ = ["solve", "Result"]

if not hasattr(torch.library, "custom_op"):
    raise ImportError("elastiqp.torch requires torch >= 2.4 (torch.library.custom_op)")


class Result(NamedTuple):
    x: torch.Tensor
    t: torch.Tensor  # elastic slacks (per-row constraint violations)
    y: torch.Tensor  # equality duals (empty if no equalities)
    z_t: torch.Tensor  # duals of t >= 0
    z_ineq: torch.Tensor  # duals of G x - t <= h
    converged: torch.Tensor  # 0/1; includes the kappa relaxation when it runs
    iters: torch.Tensor


_NOT_DIFFERENTIABLE_MSG = (
    "elastiqp.torch.solve is not differentiable with target_kappa=0: "
    "the solution sits exactly on the constraint boundary, where "
    "the exact KKT derivative is undefined. Set target_kappa > 0 "
    "(e.g. 1e-3) for log-barrier smoothed gradients"
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
# (Q, q, A, b, G, h, penalty) -> (x, t, y, z_t, z_ineq,
#                                  xr, tr, yr, z_t_r, z_ineq_r, info)
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
) -> _SOLVE_OUT:
    batch = tuple(Q.shape[:-2])
    n, m, p = Q.shape[-1], b.shape[-1], h.shape[-1]
    if not batch:  # fast path: no reshapes, C++ outputs handed to torch as-is
        res = _core._solve_relaxed(
            _np(Q), _np(q), _np(A), _np(b), _np(G), _np(h), _np(penalty),
            eps_abs, max_iter, ruiz, target_kappa,
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
            Qf[i], qf[i], Af[i], bf[i], Gf[i], hf[i], pf[i],
            eps_abs, max_iter, ruiz, target_kappa,
        )
        for o, r in zip(out, res):
            o[i] = r
    return tuple(torch.from_numpy(o).reshape(batch + (d,)) for o, d in zip(out, dims))


_solve_op = torch.library.custom_op(
    "elastiqp::solve", _solve_impl, mutates_args=(), device_types="cpu"
)


@_solve_op.register_fake
def _(Q, q, A, b, G, h, penalty, eps_abs, max_iter, ruiz, target_kappa):
    batch = Q.shape[:-2]
    n, m, p = Q.shape[-1], b.shape[-1], h.shape[-1]
    dims = (n, p, m, p, p, n, p, m, p, p, 3)
    return tuple(Q.new_empty(batch + (d,)) for d in dims)


def _setup_context(ctx, inputs, output):
    Q, q, A, b, G, h, penalty, eps_abs, max_iter, ruiz, target_kappa = inputs
    xr, tr, yr, z1r, z2r = output[5:10]
    # Differentiate at the kappa-relaxed point (the tight block is the value).
    ctx.save_for_backward(Q, A, G, h, xr, tr, yr, z1r, z2r)
    ctx.target_kappa = target_kappa


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
    ct_z_ineq: torch.Tensor,
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
            _np(Q), _np(A), _np(G), _np(h), _np(xr), _np(tr), _np(yr), _np(z1r),
            _np(z2r), _np(ct_x), _np(ct_t), _np(ct_y), _np(ct_z_t), _np(ct_z_ineq),
        )
        return tuple(torch.from_numpy(r) for r in res)
    n, m, p = Q.shape[-1], A.shape[-2], G.shape[-2]
    nb = int(np.prod(batch))
    flat = lambda x, d: _np(x).reshape((nb,) + d)
    ctf = [flat(c, (c.shape[-1],)) for c in (ct_x, ct_t, ct_y, ct_z_t, ct_z_ineq)]
    Qf, Af, Gf, hf = flat(Q, (n, n)), flat(A, (m, n)), flat(G, (p, n)), flat(h, (p,))
    xf, tf, yf = flat(xr, (n,)), flat(tr, (p,)), flat(yr, (m,))
    z1f, z2f = flat(z1r, (p,)), flat(z2r, (p,))
    dims = ((n, n), (n,), (m, n), (m,), (p, n), (p,), (p,))
    out = [np.empty((nb,) + d) for d in dims]
    for i in range(nb):
        res = _core._kkt_vjp(
            Qf[i], Af[i], Gf[i], hf[i], xf[i], tf[i], yf[i], z1f[i], z2f[i],
            ctf[0][i], ctf[1][i], ctf[2][i], ctf[3][i], ctf[4][i],
        )
        for o, r in zip(out, res):
            o[i] = r
    return tuple(torch.from_numpy(o).reshape(batch + d) for o, d in zip(out, dims))


_kkt_vjp = torch.library.custom_op(
    "elastiqp::kkt_vjp", _vjp_impl, mutates_args=(), device_types="cpu"
)


@_kkt_vjp.register_fake
def _(Q, A, G, h, xr, tr, yr, z1r, z2r, ct_x, ct_t, ct_y, ct_z_t, ct_z_ineq):
    batch = Q.shape[:-2]
    n, m, p = Q.shape[-1], A.shape[-2], G.shape[-2]
    dims = ((n, n), (n,), (m, n), (m,), (p, n), (p,), (p,))
    return tuple(Q.new_empty(batch + d) for d in dims)


def _backward(ctx, grads, vjp):
    if not ctx.target_kappa > 0:
        raise RuntimeError(_NOT_DIFFERENTIABLE_MSG)
    saved = ctx.saved_tensors
    Q = saved[0]
    # Cotangents of the tight solution (x, t, y, z_t, z_ineq). The relaxed
    # block and info are internal and never carry a gradient. Missing
    # cotangents travel as size-0 vectors (zero).
    zero = Q.new_zeros(Q.shape[:-2] + (0,))
    ct = tuple(zero if c is None else c for c in grads[:5])
    return tuple(vjp(*saved, *ct)) + (None, None, None, None)


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
    def forward(Q, q, A, b, G, h, penalty, eps_abs, max_iter, ruiz, target_kappa):
        return _solve_impl(Q, q, A, b, G, h, penalty, eps_abs, max_iter, ruiz, target_kappa)

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


def _vjp_torch(Q, A, G, h, x, t, y, z1, z2, ct_x, ct_t, ct_y, ct_z_t, ct_z_ineq):
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
    z1b, z2b = fill(ct_z_t, p), fill(ct_z_ineq, p)

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
    eps_abs=1e-5,
    max_iter=250,
    ruiz=False,
    target_kappa=1e-3,
):
    """Solve the elastic QP

        min 0.5 x'Qx + q'x + penalty' t
        s.t. A x == b (hard, optional), G x - t <= h, t >= 0

    `penalty` may be a scalar or a per-constraint vector of length p.
    `eps_abs`, `max_iter`, `ruiz` and `target_kappa` are Python scalars
    (constants under torch.compile).

    Leading batch dimensions (Q: (..., n, n), q: (..., n), ...) solve one
    problem per entry; the batch shapes of the arguments broadcast against
    each other, so a single Q may be shared across a batch of q.

    Every call here is a cold solve. It is differentiable in reverse mode
    w.r.t. all tensor arguments when target_kappa > 0 (the default);
    backward() with an explicit target_kappa=0 raises (the certificate sits
    exactly on the constraint boundary, where the exact KKT derivative is
    undefined). `ruiz=True` enables Ruiz equilibration for badly-scaled
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
    if differentiating and not target_kappa > 0:
        raise TypeError(_NOT_DIFFERENTIABLE_MSG)
    # Tight solution only when not differentiating: no relaxation runs.
    kappa = float(target_kappa) if differentiating else 0.0

    # Under torch.compile the solve must be an opaque custom op (with a
    # shape function); eagerly, the autograd.Function is much cheaper.
    call = _solve_op if torch.compiler.is_compiling() else _Solve.apply
    if device.type != "cpu":
        args = tuple(x.cpu() for x in args)
    out = call(*args, float(eps_abs), int(max_iter), bool(ruiz), kappa)
    x, t, y, z_t, z_ineq = (o.to(device) for o in out[:5])
    info = out[10]
    # info = [converged, iters, relax_converged]: converged folds in the
    # relaxation, which agrees with the tight flag whenever it does not run.
    converged = torch.minimum(info[..., 0], info[..., 2])
    return Result(
        x=x,
        t=t,
        y=y,
        z_t=z_t,
        z_ineq=z_ineq,
        converged=converged.to(torch.int32).to(device),
        iters=info[..., 1].to(torch.int32).to(device),
    )
