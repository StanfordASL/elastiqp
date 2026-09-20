"""ElastiQP Python bindings example"""

import elastiqp
import numpy as np


def main():
    np.set_printoptions(suppress=True)

    print("\n--- ElastiQP: Basic Usage ---")

    print("\nConsider a 2D toy problem where x is 'pulled towards' a goal (2, 0)")
    x_des = np.array([2.0, 0.0])
    Q = np.eye(2)
    q = -x_des

    print("\nWe'll add the following constraints:")
    print("1) Hard equality constraints, such that x0 + x1 == 1")
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])

    print("2) Elastic inequality constraints")
    print("    2i) |x_i| <= 1.5")
    print("    2ii) x0 <= 0.5")
    G = np.array(
        [
            [1.0, 0.0],  # x0 <=  1.5
            [-1.0, 0.0],  # x0 >= -1.5
            [0.0, 1.0],  # x1 <=  1.5
            [0.0, -1.0],  # x1 >= -1.5
            [1.0, 0.0],  # x0 <=  0.5
        ]
    )
    h = np.array([1.5, 1.5, 1.5, 1.5, 0.5])

    print("\nEach elastic constraint gets a penalty weight")
    print("We'll start by assigning all constraints the same, high penalty")
    penalty = 1e4 * np.ones_like(h)

    sol = elastiqp.solve(Q, q, G, h, penalty, A=A, b=b)
    print("\nAfter solving, we recover the hard-constrained solution exactly")
    print(f"x = {sol.x}")
    print("with zero slack on every inequality")
    print(f"t = {sol.t}")
    print("and equalities hold exactly")
    print(f"|Ax - b| = {abs(A @ sol.x - b).max():.2e}")

    print(
        "\nThis is the exactness property of the L1 relaxation. As long as each penalty"
    )
    print("exceeds the constraint's optimal Lagrange multiplier, the elastic solution")
    print("is exactly the hard-constrained solution. Here, the multipliers are")
    print(f"z = {sol.z}")
    print(
        f"so the active constraint (2ii) stays exact for any penalty above {np.max(sol.z)}"
    )

    print("\nAs a special case, if we set a penalty below the multiplier, that")
    print("constraint turns soft: paying the penalty is cheaper than satisfying it")
    penalty_soft = penalty.copy()
    penalty_soft[4] = 1.0
    sol = elastiqp.solve(Q, q, G, h, penalty_soft, A=A, b=b)
    print(f"x = {sol.x}")

    print("\nInspecting the slacks, we see which constraints were relaxed")
    print(f"t = {sol.t}")

    print("\nNote: the above problem is feasible by construction.")
    print("Let's now consider an infeasible problem (where we really need ElastiQP)")

    print("\n--- Handling Infeasibility ---")

    print("\nNow, add a new (infeasible) inequality constraint")
    print(
        "We'll set x0 >= 0.75 with a high penalty, which conflicts with x0 <= 0.5 (low penalty)"
    )
    G2 = np.vstack([G, [-1.0, 0.0]])
    h2 = np.append(h, -0.75)
    penalty2 = np.array([1e4, 1e4, 1e4, 1e4, 1e2, 1e4])
    sol = elastiqp.solve(Q, q, G2, h2, penalty2, A=A, b=b)

    print(
        "\nSolving the problem, we see that the low-penalty x0 <= 0.5 constraint is violated first"
    )
    print(f"x = {sol.x}")
    print(f"t = {sol.t}")

    print("\nEven with infeasible inequalities, the hard equalities always hold")
    print(f"|Ax - b| = {abs(A @ sol.x - b).max():.2e}")

    print(
        "\nNote that a hard-constrained solver would return 'infeasible' on this problem"
    )

    print("\n--- Warm-Starting ---")

    print("\nTo warm-start, we first need to construct a Solver object")
    solver = elastiqp.Solver()
    print("Then, we call setup() with initial problem data, and update() in the loop")
    solver.setup(Q, q, G, h, penalty, A=A, b=b)

    cold_iters, warm_iters = 0, 0
    num_solves = 200
    for k in range(num_solves):
        # Example: Slowly drifting target along a circle
        theta = 0.001 * k
        q_k = -np.array([2 * np.cos(theta), 2 * np.sin(theta)])
        solver.update(q=q_k)
        sol = solver.solve()
        assert sol.converged
        warm_iters += sol.iters
        cold = elastiqp.solve(Q, q_k, G, h, penalty, A=A, b=b)
        cold_iters += cold.iters

    print(
        "\nAfter running a loop, we see that warm-starting led to fewer iterations on average"
    )
    print(
        f"Average iterations: Warm: {warm_iters / num_solves:.1f}, Cold: {cold_iters / num_solves:.1f}"
    )
    print("\nNote that this speedup can be significant on robot control problems!")

    print("\n--- Differentiability ---")
    print("\nThe 'pdal' and 'ipm' backends can relax() a solution to the")
    print("kappa-smoothed differentiation point; see the JAX and PyTorch examples")


if __name__ == "__main__":
    main()
