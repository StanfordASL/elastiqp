# ElastiQP

[![Paper](http://img.shields.io/badge/arXiv-2609.19080-B31B1B.svg)](https://arxiv.org/abs/2609.19080)

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


### Overview

ElastiQP is a C++/Eigen header-only library with Python bindings and a JAX foreign function interface (FFI). 

ElastiQP's primary backend (`das.hpp`) is is a dual active-set method, based on [DAQP](https://github.com/darnstrom/daqp). We also have two additional backends, which were primarily used as a point of comparison for the paper: `pdal.hpp` is a primal-dual augmented Lagrangian method, based on [ProxQP](https://github.com/Simple-Robotics/proxsuite), and `ipm.hpp` is a proximal interior point method, based on [PIQP](https://github.com/PREDICT-EPFL/piqp) and inspired by [qpax](https://github.com/qpax-solver/qpax)

ElastiQP is *fast*, particularly when warm-started. For a rough sense of numbers, on a laptop with an Intel i7 CPU, ElastiQP can solve humanoid-scale whole-body control problems at approximately 50 us.

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

If you've installed with CMake, you can also `find_package(elastiqp)`

### Python

#### From PyPI

> [!WARNING]  
> The code is not yet available on PyPI but I will upload a copy shortly

```
pip install elastiqp
```

#### From source

If installing the Python bindings from source, first make sure that you've run the build described above in the C++ section. You'll need to have `nanobind` (and `jax`, if you want to use the FFI) installed in your current python venv when building. Then, from the top-level of the repo,
```
pip install .
```

JAX/PyTorch dependencies can be installed with `pip install "elastiqp[jax]"` or `pip install "elastiqp[torch]"`, respectively.

Note: if using UV, you can directly replace the above `pip` commands with `uv pip`


## Usage

### C++

```cpp
#include "elastiqp/elastiqp.hpp"
// or one backend only: "elastiqp/das.hpp", "pdal.hpp", "ipm.hpp"

// If you just need to solve a single problem
elastiqp::Solution sol = elastiqp::Solve(Q, q, A, b, G, h, penalty);
// or without equalities: elastiqp::Solve(Q, q, G, h, penalty)

// If you are solving multiple times in a control loop:
elastiqp::Solver solver;
solver.setup(Q, q, A, b, G, h, penalty);
while (running) {
  // Set your updated problem data and warm-start
  solver.set_q(q_k); solver.set_h(h_k); solver.set_b(b_k);
  const elastiqp::Solution& sol = solver.solve();
}
```

### Python

```python
import elastiqp

# If you just need to solve a single problem:
sol = elastiqp.solve(Q, q, G, h, penalty, A=A, b=b)
# Specify the backend (default "das") via the method kwarg
sol = elastiqp.solve(Q, q, G, h, penalty, A=A, b=b, method="das")

# If you are solving multiple times in a control loop:
solver = elastiqp.Solver()
solver.setup(Q, q, G, h, penalty, A=A, b=b)
sol = solver.solve()
solver.update(q=q_k, h=h_k, b=b_k)
sol = solver.solve()
```

### JAX / PyTorch

> [!WARNING]  
> Differentiability support is still in beta, and is not implemented in the default (active set) backend

```python
import jax
jax.config.update("jax_enable_x64", True)
import elastiqp.jax

# Same API as Python, just with elastiqp.jax
sol = elastiqp.jax.solve(Q, q, G, h, penalty, A=A, b=b)

# Compatible with jax.grad and vjp on the pdal / ipm backends
def loss(q_):
    sol = elastiqp.jax.solve(Q, q_, G, h, penalty, A=A, b=b, method="ipm")
    return jnp.sum(sol.x**2)

grad_q = jax.grad(loss)(q)

# Compatible with jit
jit_loss = jax.jit(loss)
l = jit_loss(q)

# Compatible with vmap
vmap_loss = jax.vmap(loss)
batch_q = jnp.tile(q, (10, 1))
batch_ls = vmap_loss(batch_q)

# Explicit warm starting
def step(state, q_k):
    sol = elastiqp.jax.solve(Q, q_k, G, h, penalty, A=A, b=b, warm_start=state)
    return sol, sol.x
init = elastiqp.jax.solve(Q, qs[0], G, h, penalty, A=A, b=b)
_, xs = jax.lax.scan(step, init, qs)
```

For runnable Python/JAX/PyTorch examples, see the `examples` folder

## Benchmarks

> [!WARNING]  
> Under development

See [StanfordASL/elastiqp_benchmarks](https://github.com/StanfordASL/elastiqp_benchmarks)

## Acknowledgments

ElastiQP builds on the following excellent projects:

- [qpax](https://github.com/qpax-solver/qpax)
- [DAQP](https://github.com/darnstrom/daqp)
- [ProxQP](https://github.com/Simple-Robotics/proxsuite)
- [PIQP](https://github.com/PREDICT-EPFL/piqp)

ElastiQP is licensed under Apache 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party notices.
