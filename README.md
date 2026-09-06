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

ElastiQP ships three backends that solve the same elastic QP with the same condensed formulation of the slacks, and return the same `Solution`:

- `das` (default): an elastic dual active-set method based on [DAQP](https://github.com/darnstrom/daqp). The multiplier box $[0, w_i]$ gives the working set a third, saturated state; exact termination, no penalty schedule, and the fastest choice for small dense robot QPs, warm or cold.
- `pdal`: an elastic Primal-Dual Augmented Lagrangian method based on [ProxQP](https://github.com/Simple-Robotics/proxsuite), with a modified BCL outer loop for the elastic residuals.
- `ipm`: an elastic proximal interior-point method based on [PIQP](https://github.com/PREDICT-EPFL/piqp), with the qpax elastic condensation.

In C++ they are `elastiqp::das::Solver`, `elastiqp::pdal::Solver` and `elastiqp::ipm::Solver` (one self-contained header each; `elastiqp::Solver` is the active-set default). In Python, `elastiqp.Solver(method)` / `elastiqp.solve(..., method=)` with `method` in `"das"`, `"pdal"`, `"ipm"`.

All three are fast (particularly with warm-starting). The `pdal` and `ipm` backends are additionally *differentiable* with kappa-smoothed gradients: `relax(kappa)` walks the solution to the kappa-relaxed central point (complementarity $s \odot z = \kappa$) through a log-barrier retraction, for smooth implicit differentiation (see `docs/log_barrier_admm_note.tex` and `docs/pdal_differentiability.md`). The active-set backend is forward-only. Ruiz equilibration is available for poorly-conditioned problems (on by default for `das`, off for `pdal` / `ipm`).

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

The tests are self-contained (Eigen only) and run with `ctest --test-dir build`. Benchmarks, cross-solver comparisons, and tests against external solvers live in the separate [`benchmarks/`](benchmarks/README.md) project (`elastiqp_benchmarks`), which pulls in PIQP / ProxQP / Pinocchio so that this repo does not have to.

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

### PyTorch

Same as above with the `[torch]` option, i.e. `pip install "elastiqp[torch]"`. The PyTorch interface can then be imported via `elastiqp.torch`. It needs no extra compiled code beyond the standard bindings (the solve is registered as a torch custom operator on top of them), so it also works from a source build that was configured without torch installed.

Note: if using UV, you can directly replace the above `pip` commands with `uv pip`

## Usage


### C++

```cpp
#include "elastiqp/elastiqp.hpp"  // umbrella: all three backends
// or one backend only: "elastiqp/elastiqp_das.hpp", "elastiqp_pdal.hpp", "elastiqp_ipm.hpp"

// If you just need to solve a single problem (elastiqp::Solve = the
// active-set default; elastiqp::pdal::Solve / elastiqp::ipm::Solve likewise):
elastiqp::Solution sol = elastiqp::Solve(Q, q, A, b, G, h, penalty);
// or without equalities: elastiqp::Solve(Q, q, G, h, penalty)

// If you are solving multiple times in a control loop:
elastiqp::Solver solver;  // == elastiqp::das::Solver; pdal::Solver, ipm::Solver same API
solver.setup(Q, q, A, b, G, h, penalty);
while (running) {
  // Set your updated problem data (example below) and warm-start
  solver.set_q(q_k); solver.set_h(h_k); solver.set_b(b_k);
  const elastiqp::Solution& sol = solver.solve();
}

// When differentiating (pdal / ipm backends): relax to the kappa-smoothed
// differentiation point (does not disturb the solver's warm-start state).
// Repeated calls on a persistent PDAL solver warm-start from the previous
// relaxed point (~2 Newton steps per call in a control loop); pass
// warm=false to force a restart from the tight solution.
elastiqp::pdal::Solver dsolver;
dsolver.setup(Q, q, A, b, G, h, penalty);
dsolver.solve();
const elastiqp::Solution relaxed = dsolver.relax(kappa);
```

### Python

```python
import elastiqp

# If you just need to solve a single problem:
sol = elastiqp.solve(Q, q, G, h, penalty, A=A, b=b)
# A, b equality terms are optional kwargs in the python API
# method="das" (default) / "pdal" / "ipm" picks the backend:
sol = elastiqp.solve(Q, q, G, h, penalty, A=A, b=b, method="pdal")

# If you are solving multiple times in a control loop:
solver = elastiqp.Solver()  # == elastiqp.das.Solver(); Solver("pdal"), Solver("ipm")
solver.setup(Q, q, G, h, penalty, A=A, b=b)
sol = solver.solve()
solver.update(q=q_k, h=h_k, b=b_k)
sol = solver.solve()

# Backend settings: solver.settings is a das.Settings / pdal.Settings /
# ipm.Settings; elastiqp.Settings(method) builds a default one.
solver.settings.eps_abs = 1e-8
```

### JAX

```python
import elastiqp.jax

# Same API as Python, just with elastiqp.jax (method= picks the backend)
# No warm starting, for now
sol = elastiqp.jax.solve(Q, q, G, h, penalty, A=A, b=b)

# Compatible with jax.grad and vjp on the pdal / ipm backends
def loss(q_):
    # Smoothed gradients are on by default (target_kappa=1e-3, qpax's
    # default); pass target_kappa=0 to forbid differentiation. The default
    # method="das" is forward-only: grad through it raises.
    sol = elastiqp.jax.solve(Q, q_, G, h, penalty, A=A, b=b, method="pdal")
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

### PyTorch

```python
import elastiqp.torch

# Same API again; tensors in, float64 tensors out (solved on the CPU)
sol = elastiqp.torch.solve(Q, q, G, h, penalty, A=A, b=b)

# Compatible with autograd on the pdal / ipm backends (implicit
# differentiation of the KKT system, smoothed at target_kappa=1e-3 by
# default; target_kappa=0 forbids it; the default method="das" is forward-only)
q_ = q.clone().requires_grad_(True)
loss = torch.sum(elastiqp.torch.solve(Q, q_, G, h, penalty, A=A, b=b, method="pdal").x ** 2)
loss.backward()
grad_q = q_.grad

# Compatible with torch.compile: the solve is an opaque custom op
compiled = torch.compile(lambda q_: elastiqp.torch.solve(Q, q_, G, h, penalty, A=A, b=b).x)

# Batched: leading batch dims solve one problem per entry (or use torch.vmap)
batch_x = elastiqp.torch.solve(Q, q.expand(10, -1), G, h, penalty, A=A, b=b).x

# torch.func transforms work too (grad, jacrev, vmap(grad))
J = torch.func.jacrev(lambda q_: elastiqp.torch.solve(Q, q_, G, h, penalty, A=A, b=b, method="pdal").x)(q)
```

For runnable Python/JAX/PyTorch examples, see the `examples` folder

## Benchmarks

See [StanfordASL/elastiqp_benchmarks](https://github.com/StanfordASL/elastiqp_benchmarks)


## Assorted Tips

- If differentiating through problems with large penalty weights (roughly >= 1e4), or for badly row-scaled constraints (mixed units), consider turning on Ruiz equilibration. In Python/JAX/PyTorch: `ruiz=True`; in C++: `settings.ruiz = true` (the active-set backend has it on by default). The scaling is computed at `setup()` and carried through the `set_*` updates (still exact, only the conditioning drifts); the `das` and `pdal` backends re-equilibrate automatically in `solve()` once a matrix update has drifted a scaled row/column norm past `settings.ruiz_refresh_ratio` (4x by default), keeping the warm start. `scaling_drift()` reports the current drift and (PDAL) `reequilibrate()` refreshes on demand, so a control loop never needs a second `setup()` for this.
- `eps_abs` defaults to 1e-5 on every backend but means slightly different things: `das` terminates when no row violates its bound by more than `eps_abs` (user units; dual feasibility is exact, complementarity is bounded by `penalty * eps_abs`), while `pdal` and `ipm` terminate on the inf-norm of the elastic KKT residuals and the duality gap. The one-shot `solve(..., eps_abs=)` sets whichever applies.


## Acknowledgments

ElastiQP builds on the following excellent projects:

- [qpax](https://github.com/qpax-solver/qpax): ElastiQP is inspired by the condensation strategy from their elastic primal-dual interior point method, and builds on their kappa-smoothed derivatives. Apache-2.0
- [DAQP](https://github.com/darnstrom/daqp): the default `das` backend is an elastic formulation of their dual active-set method with recursive LDL' updates (`include/elastiqp/elastiqp_das.hpp`). MIT.
- [ProxQP](https://github.com/Simple-Robotics/proxsuite): the `pdal` backend is an elastic formulation of their primal-dual augmented Lagrangian method (`include/elastiqp/elastiqp_pdal.hpp`). BSD 2-Clause.
- [PIQP](https://github.com/PREDICT-EPFL/piqp): the `ipm` backend (`include/elastiqp/elastiqp_ipm.hpp`), which the test suite also uses as the oracle for the other two, is based on it, and the benchmarks project cross-validates all three against vanilla PIQP. BSD 2-Clause.

ElastiQP is licensed under Apache 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party notices.
