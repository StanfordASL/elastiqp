# ElastiQP

An always-feasible QP solver for constrained robot control.

ElastiQP solves the following problem:

$$
\begin{align*}
\underset{x, t}{\text{minimize}} & \quad \frac{1}{2}x^TQx + q^Tx + w^T t \\
\text{s.t.} & \quad Ax = b \\
& \quad Gx - t \leq h \\ 
& \quad t \geq 0
\end{align*}
$$

i.e, a QP with hard equality constraints $Ax = b$ and *elastic* inequality constraints $Gx \leq h$, where every inequality constraint is relaxed with an $\ell_1$ penalty term defined by $w > 0$.

Notably, the elastic slacks $t$ are eliminated *analytically*, so the condensed system stays $n \times n$ regardless of the number of constraints. This enables fast compute times, even with large numbers of inequality constraints $p$.


### Why ElastiQP?

For robot control, we are typically interested in solving small-scale, dense QPs, where the number of inequality constraints $p$ may be much larger than the number of decision variables $n$. In this setting, inequality constraints can often be *momentarily infeasible*, but in a control loop, we always need to return a reasonable solution. 

Likewise, in the control setting, we often encode dynamics via strict equality constraints. For typical systems, dynamics constraints are feasible by construction, and relaxing these would lead to unrealistic behavior. 

Given this, in the case of infeasibility, ElastiQP naturally relaxes the inequality constraints in an $\ell_1$ manner, relaxing *only* the constraints that strictly need to be adjusted for a feasible solution.


### Feature Overview

ElastiQP is a C++/Eigen header-only library with Python bindings and a JAX foreign function interface (FFI). 

The solver is an elastic Primal-Dual Augmented Lagrangian (PDAL) method based on [ProxQP](https://github.com/Simple-Robotics/proxsuite), with a condensed formulation of the elastic slacks.

The solver is fast (particularly with warm-starting), and it is *differentiable* with kappa-smoothed gradients: `relax(kappa)` walks the solution to the kappa-relaxed central point (complementarity $s \odot z = \kappa$) through a log-barrier retraction, for smooth implicit differentiation (see `docs/log_barrier_admm_note.tex` and `docs/pdal_differentiability.md`). Ruiz equilibration is available for poorly-conditioned problems (off by default).

For a rough sense of numbers, on a laptop with an Intel i7 CPU, ElastiQP can solve humanoid-scale whole-body control problems at approximately 32 us.


## Installation

### C++

#### From source:
```
git clone https://github.com/StanfordASL/elastiqp
cd elastiqp
cmake -B build . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# Optional: cmake --install build --config Release
```

For best performance (on your own device), you can also build with `-march=native`
```
cmake -B build-native . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
cmake --build build-native --config Release
```

To build the `benchmarks`, add the following flag: `-DELASTIQP_BUILD_BENCHMARKS=ON`

If you've installed with CMake, you can also `find_package(elastiqp)`

### Python

#### From PyPI
```
pip install elastiqp
```

#### From source

If installing the Python bindings from source, first make sure that you've run the build described above in the C++ section. You'll need to have `nanobind` (and `jax`, if you want to use the FFI) installed in your current python venv when building. Then, from the top-level of the repo,
```
pip install .
```

### JAX

Same as above, but specify the `[jax]` option to pull in the the JAX dependencies and FFI, i.e. `pip install "elastiqp[jax]"`. The JAX interface can then be imported via `elastiqp.jax` 

Note: if using UV, you can directly replace the above `pip` commands with `uv pip`

## Usage


### C++

```cpp
#include "elastiqp/elastiqp.hpp"

// If you just need to solve a single problem:
elastiqp::Solution sol = elastiqp::Solve(Q, q, A, b, G, h, penalty);
// or without equalities: elastiqp::Solve(Q, q, G, h, penalty)

// If you are solving multiple times in a control loop:
elastiqp::Solver solver;
solver.setup(Q, q, A, b, G, h, penalty);
while (running) {
  // Set your updated problem data (example below) and warm-start
  solver.set_q(q_k); solver.set_h(h_k); solver.set_b(b_k);
  const elastiqp::Solution& sol = solver.solve();
}

// When differentiating: relax to the kappa-smoothed differentiation point
// (does not disturb the solver's warm-start state)
const elastiqp::Solution relaxed = solver.relax(kappa);
```

### Python

```python
import elastiqp

# If you just need to solve a single problem:
sol = elastiqp.solve(Q, q, G, h, penalty, A=A, b=b)
# A, b equality terms are optional kwargs in the python API

# If you are solving multiple times in a control loop:
solver = elastiqp.Solver()
solver.setup(Q, q, G, h, penalty, A=A, b=b)
sol = solver.solve()
solver.update(q=q_k, h=h_k, b=b_k)
sol = solver.solve()
```

### JAX

```python
import elastiqp.jax

# Same API as Python, just with elastiqp.jax
# No warm starting, for now
sol = elastiqp.jax.solve(Q, q, G, h, penalty, A=A, b=b)

# Compatible with jax.grad and vjp
def loss(q_):
    # Specify a target_kappa >0 when solving for smoothed gradients
    sol = elastiqp.jax.solve(Q, q_, G, h, penalty, A=A, b=b, target_kappa=1e-3)
    return jnp.sum(sol.x**2)

grad_q = jax.grad(loss)(q)

# Compatible with jit
jit_loss = jax.jit(loss)
l = jit_loss(q)

# Compatible with vmap
vmap_loss = jax.vmap(loss)
batch_q = jnp.tile(q, (10, 1))
batch_ls = vmap_loss(batch_q)
```

For runnable Python/JAX examples, see the `examples` folder

## Acknowledgments

ElastiQP builds on the following excellent projects:

- [qpax](https://github.com/qpax-solver/qpax): ElastiQP is inspired by the condensation strategy from their elastic primal-dual interior point method, and builds on their kappa-smoothed derivatives. Apache-2.0
- [ProxQP](https://github.com/Simple-Robotics/proxsuite): ElastiQP considers an elastic formulation of their primal-dual augmented Lagrangian method. BSD 2-Clause.
- [PIQP](https://github.com/PREDICT-EPFL/piqp): used (vendored, test-only) as an independent reference solver in ElastiQP's test suite, alongside a test-only elastic interior-point reference implementation based on it. BSD 2-Clause.

ElastiQP is licensed under Apache 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party notices.
