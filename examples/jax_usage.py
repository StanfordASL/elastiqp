"""ElastiQP JAX FFI example: jit, vmap, and grad"""

import jax

jax.config.update("jax_enable_x64", True)

import elastiqp.jax
import jax.numpy as jnp
import numpy as np


def main():
    np.set_printoptions(suppress=True)

    print("\n--- ElastiQP: JAX Usage ---")

    print("\nElastiQP provides JAX support via the FFI, so solves can sit inside")
    print("jit-compiled, vmapped, and differentiated code like any JAX function")

    print("\nConsider a 4D problem where x is pulled towards a goal x_des")
    n = 4
    x_des = jnp.array([2.0, -0.3, 0.1, 0.4])
    Q = jnp.eye(n)
    q = -x_des

    print("\nWe'll add the following constraints:")
    print("1) A hard equality constraint, such that sum(x) == 0.5")
    A = jnp.ones((1, n))
    b = jnp.array([0.5])

    print("2) An elastic box, |x_i| <= 1")
    G = jnp.vstack([jnp.eye(n), -jnp.eye(n)])
    h = jnp.ones(2 * n)

    print("\nWe'll use a penalty of 10 on the box, which is high enough for")
    print("exactness: the box behaves as hard constraints (see basic_usage.py)")
    penalty = 10.0

    print("\n--- JIT Compilation ---")

    print("\nFirst, let's solve the problem inside a jit-compiled function")

    @jax.jit
    def solve(q):
        return elastiqp.jax.solve(Q, q, G, h, penalty, A=A, b=b)

    sol = solve(q)
    print(f"x = {sol.x}")
    print("Note that x0 is clipped at its box bound: active, but not violated")
    print(f"t = {sol.t}")
    print("and the hard equality holds")
    print(f"|Ax - b| = {jnp.abs(A @ sol.x - b).max():.2e}")

    print("\n--- Batching with vmap ---")

    print("\nWith vmap, we can solve a batch of QPs in a single call")
    print("For example, sweeping the goal position over 8 scales of x_des")
    targets = jnp.stack([x_des * s for s in jnp.linspace(0.2, 2.0, 8)])
    batched = jax.vmap(lambda xd: elastiqp.jax.solve(Q, -xd, G, h, penalty, A=A, b=b).x)
    xs = batched(targets)
    print(f"\nAfter solving, we get a batch of solutions with shape {xs.shape}")
    print(
        f"where x0 spans [{xs[:, 0].min():.3f}, {xs[:, 0].max():.3f}] as the goal moves"
    )

    print("\n--- Differentiation ---")

    print("\nWith jax.grad, we can differentiate scalar losses through the solver")
    print("This uses implicit differentiation of the KKT system (not unrolling)")
    print("\nDifferentiation requires target_kappa > 0: the solution sits exactly")
    print("on the constraint boundary, so gradients are evaluated at a")
    print("kappa-relaxed central point (as in qpax), trading an O(kappa) bias")
    print("for smooth, well-conditioned gradients")

    def loss(q_, kappa=1e-6):
        sol = elastiqp.jax.solve(
            Q, q_, G, h, penalty, A=A, b=b, target_kappa=kappa
        )
        return jnp.sum(sol.x**2)

    print("\nDifferentiating loss = sum(x^2) w.r.t. the cost vector q:")
    g_q = jax.grad(loss)(q)
    print(f"dloss/dq = {np.asarray(g_q)}")

    print("\nWe can check the gradient against finite differences")
    print("(with a small kappa = 1e-6, so the smoothing bias stays below the")
    print("finite-differencing error)")
    eps = 1e-6
    fd_q = np.zeros(n)
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps
        fd_q[i] = (loss(q + dq) - loss(q - dq)) / (2 * eps)
    print(f"Finite differencing gives dloss/dq = {fd_q}")
    print(f"which matches to a max error of {np.abs(np.asarray(g_q) - fd_q).max():.2e}")

    print("\n--- Smoothed Gradients ---")

    print("\nExactly at an active-set change, the true gradient is discontinuous")
    print("Increasing target_kappa smooths that transition")

    def x0(h0, kappa):
        h_mod = h.at[0].set(h0)
        return elastiqp.jax.solve(
            Q, q, G, h_mod, penalty, A=A, b=b, target_kappa=kappa
        ).x[0]

    print("\nTo see this, let's compare d(x0)/d(h0): the sensitivity of x0 to its")
    print("upper bound, as that bound sweeps across the point where the constraint")
    print("deactivates, for a small and a larger kappa")
    for h0 in [0.8, 1.0, 1.2, 1.4, 1.6]:
        g_sharp = jax.grad(x0)(h0, 1e-9)
        g_smooth = jax.grad(x0)(h0, 1e-3)
        print(
            f"h0 = {h0:.1f}:  kappa=1e-9: {g_sharp:+.4f}   "
            f"kappa=1e-3: {g_smooth:+.4f}"
        )

    print("\nThe near-exact (kappa=1e-9) gradient snaps from 1 to 0 at the")
    print("activation point, while the smoothed gradient transitions continuously")
    print("between the two regimes")


if __name__ == "__main__":
    main()
