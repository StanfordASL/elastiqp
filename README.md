<div align="center">
<img src="https://github.com/user-attachments/assets/12b43719-2b23-478d-9aaa-5481911741b4" alt="logo" width="400" height="73"></img>
</div>

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
cmake --build build --config Release -j
# Optional: cmake --install build --config Release
```

For best performance (on your own device), you can also build with `-march=native`
```
cmake -B build-native . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
cmake --build build-native --config Release -j
```

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

// Repeated solves
elastiqp::Solver solver;
solver.setup(Q, q, A, b, G, h, penalty);
while (running) {
  // Set your updated problem data and warm-start
  solver.set_q(q_k); solver.set_h(h_k); solver.set_b(b_k);
  const elastiqp::Solution& sol = solver.solve();
}
```

### C

The C interface uses the active-set backend. Build `elastiqp_c` with `-DELASTIQP_BUILD_C=ON` (default for top-level builds); add `-DBUILD_SHARED_LIBS=ON` for a shared library. Matrices are dense row-major; only the upper triangle of `Q` is read. Status codes: `> 0` success, `0` unsolved, `< 0` failure. See [`elastiqp_c.h`](include/elastiqp/elastiqp_c.h) for the API.

```c
#include "elastiqp/elastiqp_c.h"

// One-shot solve; penalty >= ELASTIQP_INF makes a row hard.
ElastiQPProblem qp = {.n = n, .m = m, .p = p, .Q = Q, .q = q, .A = A, .b = b,
                      .G = G, .h = h, .penalty = penalty};
ElastiQPSolution res = {.x = x, .z = z};  // caller-owned outputs; NULL ones are skipped
elastiqp_solve_once(&res, &qp, NULL);     // NULL: default settings

// Repeated solves
ElastiQPWorkspace* work;
elastiqp_setup(&work, &qp, NULL);
while (running) {
  elastiqp_set_q(work, q_k); elastiqp_set_h(work, h_k); elastiqp_set_b(work, b_k);
  elastiqp_set_G_rows(work, first, count, rows_k);  // changed rows
  if (elastiqp_solve(&res, work) < 0) { /* ... */ }
}
elastiqp_free(work);
```

Pass settings as an `ElastiQPSettings` struct or set them by name: `elastiqp_set_option(work, "eps_abs", 1e-8)`. Use `elastiqp_num_options` and `elastiqp_option_info` to list names, types, defaults, ranges, and descriptions.

With CMake: `find_package(elastiqp)` and link `elastiqp::elastiqp_c`.

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
> Differentiability support is still in beta

```python
import jax
jax.config.update("jax_enable_x64", True)
import elastiqp.jax

# Same API as Python, just with elastiqp.jax
sol = elastiqp.jax.solve(Q, q, G, h, penalty, A=A, b=b)

# Compatible with jax.grad and vjp
def loss(q_):
    sol = elastiqp.jax.solve(Q, q_, G, h, penalty, A=A, b=b)
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


## Citation

```
@article{morton2026elastiqp,
  author={Morton, Daniel and Arrizabalaga, Jon and Manchester, Zachary and Pavone, Marco},
  title={Elasti{QP}: An Always-Feasible QP Solver for Constrained Robot Control},
  journal={arXiv preprint arXiv:2609.19080},
  year={2026},
}
```
