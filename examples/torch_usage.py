"""ElastiQP PyTorch example: autograd, batching, and torch.compile"""

import elastiqp.torch
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def main():
    torch.set_printoptions(sci_mode=False)

    print("\n--- ElastiQP: PyTorch Usage ---")

    print("\nElastiQP registers its solve as a torch custom operator, so it sits")
    print("inside autograd, torch.compile, and torch.vmap like any torch function")

    print("\nConsider a 4D problem where x is pulled towards a goal x_des")
    n = 4
    x_des = torch.tensor([2.0, -0.3, 0.1, 0.4])
    Q = torch.eye(n)
    q = -x_des

    print("\nWe'll add the following constraints:")
    print("1) A hard equality constraint, such that sum(x) == 0.5")
    A = torch.ones((1, n))
    b = torch.tensor([0.5])

    print("2) An elastic box, |x_i| <= 1")
    G = torch.vstack([torch.eye(n), -torch.eye(n)])
    h = torch.ones(2 * n)

    print("\nWe'll use a penalty of 10 on the box, which is high enough for")
    print("exactness: the box behaves as hard constraints (see basic_usage.py)")
    penalty = 10.0

    print("\n--- Solving ---")
    sol = elastiqp.torch.solve(Q, q, G, h, penalty, A=A, b=b)
    print(f"x = {sol.x}")
    print("Note that x0 is clipped at its box bound: active, but not violated")
    print(f"t = {sol.t}")
    print("and the hard equality holds")
    print(f"|Ax - b| = {torch.abs(A @ sol.x - b).max():.2e}")

    print("\n--- Batching ---")

    print("\nLeading batch dimensions solve one problem per entry (with the")
    print("other arguments broadcast). For example, sweeping the goal over 8")
    print("scales of x_des:")
    targets = torch.stack([x_des * s for s in torch.linspace(0.2, 2.0, 8)])
    xs = elastiqp.torch.solve(Q, -targets, G, h, penalty, A=A, b=b).x
    print(f"\nWe get a batch of solutions with shape {tuple(xs.shape)}")
    print(
        f"where x0 spans [{xs[:, 0].min():.3f}, {xs[:, 0].max():.3f}] as the goal moves"
    )
    print("(torch.vmap over a per-problem function does the same thing)")

    print("\n--- Differentiation ---")

    print("\nWith autograd, we can differentiate scalar losses through the solver")
    print("This uses implicit differentiation of the KKT system (not unrolling)")
    print("\nDifferentiation requires target_kappa > 0: the solution sits exactly")
    print("on the constraint boundary, so gradients are evaluated at a")
    print("kappa-relaxed central point (as in qpax), trading an O(kappa) bias")
    print("for smooth, well-conditioned gradients")

    def loss(q_, kappa=1e-6):
        sol = elastiqp.torch.solve(
            Q, q_, G, h, penalty, A=A, b=b, target_kappa=kappa, eps_abs=1e-10
        )
        return torch.sum(sol.x**2)

    print("\nDifferentiating loss = sum(x^2) w.r.t. the cost vector q:")
    q_ = q.clone().requires_grad_(True)
    loss(q_).backward()
    g_q = q_.grad
    print(f"dloss/dq = {g_q}")

    print("\nWe can check the gradient against finite differences")
    print("(with a small kappa = 1e-6, so the smoothing bias stays below the")
    print("finite-differencing error)")
    eps = 1e-6
    fd_q = torch.zeros(n)
    with torch.no_grad():
        for i in range(n):
            dq = torch.zeros(n)
            dq[i] = eps
            fd_q[i] = (loss(q + dq) - loss(q - dq)) / (2 * eps)
    print(f"Finite differencing gives dloss/dq = {fd_q}")
    print(f"which matches to a max error of {torch.abs(g_q - fd_q).max():.2e}")

    print("\n--- torch.compile ---")

    print("\nThe solve is an opaque custom op with a registered shape function,")
    print("so it compiles without graph breaks, forward and backward:")

    @torch.compile(fullgraph=True)
    def compiled_loss(q_):
        return loss(q_, kappa=1e-3)

    q_c = q.clone().requires_grad_(True)
    compiled_loss(q_c).backward()
    q_e = q.clone().requires_grad_(True)
    loss(q_e, kappa=1e-3).backward()
    print(f"compiled vs eager gradient: max error {torch.abs(q_c.grad - q_e.grad).max():.1e}")

    print("\n--- Smoothed Gradients ---")

    print("\nExactly at an active-set change, the true gradient is discontinuous")
    print("Increasing target_kappa smooths that transition")

    def dx0_dh0(h0, kappa):
        h0 = torch.tensor(h0, requires_grad=True)
        h_mod = torch.cat([h0.reshape(1), h[1:]])
        x0 = elastiqp.torch.solve(Q, q, G, h_mod, penalty, A=A, b=b, target_kappa=kappa).x[0]
        x0.backward()
        return float(h0.grad)

    print("\nTo see this, let's compare d(x0)/d(h0): the sensitivity of x0 to its")
    print("upper bound, as that bound sweeps across the point where the constraint")
    print("deactivates, for a small and a larger kappa")
    for h0 in [0.8, 1.0, 1.2, 1.4, 1.6]:
        print(
            f"h0 = {h0:.1f}:  kappa=1e-9: {dx0_dh0(h0, 1e-9):+.4f}   "
            f"kappa=1e-3: {dx0_dh0(h0, 1e-3):+.4f}"
        )

    print("\nThe near-exact (kappa=1e-9) gradient snaps from 1 to 0 at the")
    print("activation point, while the smoothed gradient transitions continuously")
    print("between the two regimes")


if __name__ == "__main__":
    main()
