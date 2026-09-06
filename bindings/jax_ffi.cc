// XLA FFI for ElastiQP
//
// Signature:
//
//   (Q, q, A, b, G, h, penalty)
//       -> (x, t, y, z_t, z, xr, tr, yr, z_t_r, z_r, info)
//
// `method` selects the backend: 0 = active set (elastiqp::das), 1 = PDAL,
// 2 = interior point. `ruiz` (0/1) enables equilibration. `max_iter` is the
// backend's outer budget (active-set iterations / BCL rounds / IPM
// iterations).
//
// info = [converged, iters, relax_converged]; converged/iters describe the
// tight solve, relax_converged the kappa relaxation (== converged when no
// relaxation runs).
//
// All buffers are float64. Equality constraints are optional: pass A
// with 0 rows and b with 0 entries for the inequality-only form.
//
// The (xr, ...) block is the kappa-relaxed central point used for smoothed
// implicit differentiation (see qpax): when target_kappa > 0, we re-solve
// from the optimum to the point satisfying the same KKT conditions with
// complementarity s.z = kappa (via the log-barrier retraction), and the
// Python wrapper differentiates there while still returning the tight
// solution as the value. When target_kappa <= 0 the relaxed block is a
// copy of the tight solution. Only the PDAL and IPM backends have relax();
// with method = 0 and target_kappa > 0 the handler returns an error.
//
// Every call constructs a fresh solver, since JAX/XLA expect functional
// purity. Warm starting is therefore explicit: the second handler,
//
//   (Q, q, A, b, G, h, penalty, x0, y0, z0) -> (x, t, y, z_t, z, info)
//
// seeds the fresh solver from a user-frame point (x0, y0, z0), typically
// the previous tick's solution carried through the JAX program as state.
// This keeps the iterate warm start of each backend (DAS reads its working
// set off z0; PDAL / IPM start their iterates there) but not the
// factorization cache a persistent solver would reuse. No relaxation, no
// gradients.

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "elastiqp/elastiqp.hpp"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapMatrix = Eigen::Map<const RowMajorMatrix>;
using MapVector = Eigen::Map<const Eigen::VectorXd>;
using Eigen::VectorXd;

template <typename SolverT>
void apply_options(SolverT& solver, double eps_abs, int64_t max_iter,
                   int64_t ruiz);

template <>
void apply_options(elastiqp::das::Solver& solver, double eps_abs,
                   int64_t max_iter, int64_t ruiz) {
  solver.settings.eps_abs = eps_abs;
  solver.settings.max_iter = static_cast<int>(max_iter);
  solver.settings.ruiz = ruiz != 0;
}
template <>
void apply_options(elastiqp::pdal::Solver& solver, double eps_abs,
                   int64_t max_iter, int64_t ruiz) {
  solver.settings.eps_abs = eps_abs;
  solver.settings.eps_duality_gap_abs = eps_abs;
  solver.settings.max_outer_iter = static_cast<int>(max_iter);
  solver.settings.ruiz = ruiz != 0;
}
template <>
void apply_options(elastiqp::ipm::Solver& solver, double eps_abs,
                   int64_t max_iter, int64_t ruiz) {
  solver.settings.eps_abs = eps_abs;
  solver.settings.eps_duality_gap_abs = eps_abs;
  solver.settings.max_iter = static_cast<int>(max_iter);
  solver.settings.ruiz = ruiz != 0;
}

// Seed a freshly set-up solver from a user-frame (x0, y0, z0).
template <typename SolverT>
void apply_warm_start(SolverT& solver, const VectorXd& x0, const VectorXd& y0,
                      const VectorXd& z0) {
  if constexpr (std::is_same_v<SolverT, elastiqp::ipm::Solver>) {
    solver.warm_start_from(x0, y0, z0);  // reconstructs + floors the slacks
  } else {
    solver.set_warm_start(x0, y0, z0);
  }
}

// Fresh solver per call: XLA assumes FFI calls are pure. Returns the tight
// solution and, when kappa > 0, the relaxed point. With x0 != nullptr the
// solve is seeded from (x0, y0, z0).
template <typename SolverT>
void run_solver(double eps_abs, int64_t max_iter, int64_t ruiz,
                double target_kappa, int64_t n, int64_t m, int64_t p,
                const double* Q, const double* q, const double* A,
                const double* b, const double* G, const double* h,
                const double* penalty, const double* x0, const double* y0,
                const double* z0, elastiqp::Solution& sol,
                elastiqp::Solution& rsol) {
  SolverT solver;
  apply_options(solver, eps_abs, max_iter, ruiz);
  solver.setup(MapMatrix(Q, n, n), MapVector(q, n), MapMatrix(A, m, n),
               MapVector(b, m), MapMatrix(G, p, n), MapVector(h, p),
               MapVector(penalty, p));
  if (x0 != nullptr) {
    apply_warm_start(solver, VectorXd(MapVector(x0, n)),
                     VectorXd(MapVector(y0, m)), VectorXd(MapVector(z0, p)));
  }
  sol = solver.solve();
  // The relax tolerance is decoupled from eps_abs: the VJP linearizes at
  // the relaxed point, so gradient accuracy is set by THIS residual, and
  // the Newton corrector buys digits cheaply. min() keeps an explicitly
  // tight eps_abs tightening the gradients too, without a loose forward
  // tolerance loosening them.
  if constexpr (std::is_same_v<SolverT, elastiqp::das::Solver>) {
    rsol = sol;
  } else {
    rsol = (target_kappa > 0 && p > 0)
               ? solver.relax(target_kappa, std::min(eps_abs, 1e-6), 50)
               : sol;
  }
}

// Backend dispatch shared by both handlers.
ffi::Error dispatch(int64_t method, double eps_abs, int64_t max_iter,
                    int64_t ruiz, double target_kappa, int64_t n, int64_t m,
                    int64_t p, const double* Q, const double* q,
                    const double* A, const double* b, const double* G,
                    const double* h, const double* penalty, const double* x0,
                    const double* y0, const double* z0,
                    elastiqp::Solution& sol, elastiqp::Solution& rsol) {
  switch (method) {
    case 0:
      if (target_kappa > 0 && p > 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "method='das' has no kappa relaxation (not "
                          "differentiable); use method='pdal' or 'ipm'");
      }
      run_solver<elastiqp::das::Solver>(eps_abs, max_iter, ruiz, target_kappa,
                                        n, m, p, Q, q, A, b, G, h, penalty,
                                        x0, y0, z0, sol, rsol);
      return ffi::Error::Success();
    case 1:
      run_solver<elastiqp::pdal::Solver>(eps_abs, max_iter, ruiz,
                                         target_kappa, n, m, p, Q, q, A, b, G,
                                         h, penalty, x0, y0, z0, sol, rsol);
      return ffi::Error::Success();
    case 2:
      run_solver<elastiqp::ipm::Solver>(eps_abs, max_iter, ruiz, target_kappa,
                                        n, m, p, Q, q, A, b, G, h, penalty,
                                        x0, y0, z0, sol, rsol);
      return ffi::Error::Success();
    default:
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "method must be 0 (das), 1 (pdal) or 2 (ipm)");
  }
}

void write_solution(const elastiqp::Solution& sol, int64_t n, int64_t m,
                    int64_t p, ffi::ResultBuffer<ffi::F64>& x,
                    ffi::ResultBuffer<ffi::F64>& t,
                    ffi::ResultBuffer<ffi::F64>& y,
                    ffi::ResultBuffer<ffi::F64>& z_t,
                    ffi::ResultBuffer<ffi::F64>& z) {
  Eigen::Map<VectorXd>(x->typed_data(), n) = sol.x;
  Eigen::Map<VectorXd>(t->typed_data(), p) = sol.t;
  Eigen::Map<VectorXd>(y->typed_data(), m) = sol.y;
  Eigen::Map<VectorXd>(z_t->typed_data(), p) = sol.z_t;
  Eigen::Map<VectorXd>(z->typed_data(), p) = sol.z;
}

ffi::Error ElastiqpSolveImpl(
    double eps_abs, int64_t max_iter, int64_t ruiz, double target_kappa,
    int64_t method, ffi::Buffer<ffi::F64> Q, ffi::Buffer<ffi::F64> q, ffi::Buffer<ffi::F64> A,
    ffi::Buffer<ffi::F64> b, ffi::Buffer<ffi::F64> G, ffi::Buffer<ffi::F64> h,
    ffi::Buffer<ffi::F64> penalty, ffi::ResultBuffer<ffi::F64> x,
    ffi::ResultBuffer<ffi::F64> t, ffi::ResultBuffer<ffi::F64> y,
    ffi::ResultBuffer<ffi::F64> z_t, ffi::ResultBuffer<ffi::F64> z,
    ffi::ResultBuffer<ffi::F64> xr, ffi::ResultBuffer<ffi::F64> tr,
    ffi::ResultBuffer<ffi::F64> yr, ffi::ResultBuffer<ffi::F64> z_t_r,
    ffi::ResultBuffer<ffi::F64> z_r, ffi::ResultBuffer<ffi::F64> info) {
  const int64_t n = q.dimensions()[0];
  const int64_t m = b.dimensions()[0];
  const int64_t p = h.dimensions()[0];

  elastiqp::Solution sol, rsol;
  ffi::Error err = dispatch(method, eps_abs, max_iter, ruiz, target_kappa, n,
                            m, p, Q.typed_data(), q.typed_data(),
                            A.typed_data(), b.typed_data(), G.typed_data(),
                            h.typed_data(), penalty.typed_data(), nullptr,
                            nullptr, nullptr, sol, rsol);
  if (err.failure()) return err;

  write_solution(sol, n, m, p, x, t, y, z_t, z);
  write_solution(rsol, n, m, p, xr, tr, yr, z_t_r, z_r);
  info->typed_data()[0] = static_cast<double>(sol.converged);
  info->typed_data()[1] = static_cast<double>(sol.iters);
  info->typed_data()[2] = static_cast<double>(rsol.converged);
  return ffi::Error::Success();
}

// Warm-started solve: no relaxation block, info = [converged, iters].
ffi::Error ElastiqpSolveWarmImpl(
    double eps_abs, int64_t max_iter, int64_t ruiz, int64_t method,
    ffi::Buffer<ffi::F64> Q, ffi::Buffer<ffi::F64> q, ffi::Buffer<ffi::F64> A,
    ffi::Buffer<ffi::F64> b, ffi::Buffer<ffi::F64> G, ffi::Buffer<ffi::F64> h,
    ffi::Buffer<ffi::F64> penalty, ffi::Buffer<ffi::F64> x0,
    ffi::Buffer<ffi::F64> y0, ffi::Buffer<ffi::F64> z0,
    ffi::ResultBuffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> t,
    ffi::ResultBuffer<ffi::F64> y, ffi::ResultBuffer<ffi::F64> z_t,
    ffi::ResultBuffer<ffi::F64> z, ffi::ResultBuffer<ffi::F64> info) {
  const int64_t n = q.dimensions()[0];
  const int64_t m = b.dimensions()[0];
  const int64_t p = h.dimensions()[0];
  if (x0.dimensions()[0] != n || y0.dimensions()[0] != m ||
      z0.dimensions()[0] != p) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "warm start (x0, y0, z0) must have shapes (n,), (m,), "
                      "(p,) matching (q, b, h)");
  }

  elastiqp::Solution sol, rsol;
  ffi::Error err = dispatch(method, eps_abs, max_iter, ruiz, 0.0, n, m, p,
                            Q.typed_data(), q.typed_data(), A.typed_data(),
                            b.typed_data(), G.typed_data(), h.typed_data(),
                            penalty.typed_data(), x0.typed_data(),
                            y0.typed_data(), z0.typed_data(), sol, rsol);
  if (err.failure()) return err;

  write_solution(sol, n, m, p, x, t, y, z_t, z);
  info->typed_data()[0] = static_cast<double>(sol.converged);
  info->typed_data()[1] = static_cast<double>(sol.iters);
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(ElastiqpSolve, ElastiqpSolveImpl,
                              ffi::Ffi::Bind()
                                  .Attr<double>("eps_abs")
                                  .Attr<int64_t>("max_iter")
                                  .Attr<int64_t>("ruiz")
                                  .Attr<double>("target_kappa")
                                  .Attr<int64_t>("method")
                                  .Arg<ffi::Buffer<ffi::F64>>()  // Q
                                  .Arg<ffi::Buffer<ffi::F64>>()  // q
                                  .Arg<ffi::Buffer<ffi::F64>>()  // A
                                  .Arg<ffi::Buffer<ffi::F64>>()  // b
                                  .Arg<ffi::Buffer<ffi::F64>>()  // G
                                  .Arg<ffi::Buffer<ffi::F64>>()  // h
                                  .Arg<ffi::Buffer<ffi::F64>>()  // penalty
                                  .Ret<ffi::Buffer<ffi::F64>>()  // x
                                  .Ret<ffi::Buffer<ffi::F64>>()  // t
                                  .Ret<ffi::Buffer<ffi::F64>>()  // y
                                  .Ret<ffi::Buffer<ffi::F64>>()  // z_t
                                  .Ret<ffi::Buffer<ffi::F64>>()  // z
                                  .Ret<ffi::Buffer<ffi::F64>>()  // xr
                                  .Ret<ffi::Buffer<ffi::F64>>()  // tr
                                  .Ret<ffi::Buffer<ffi::F64>>()  // yr
                                  .Ret<ffi::Buffer<ffi::F64>>()  // z_t_r
                                  .Ret<ffi::Buffer<ffi::F64>>()  // z_r
                                  .Ret<ffi::Buffer<ffi::F64>>()  // info
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(ElastiqpSolveWarm, ElastiqpSolveWarmImpl,
                              ffi::Ffi::Bind()
                                  .Attr<double>("eps_abs")
                                  .Attr<int64_t>("max_iter")
                                  .Attr<int64_t>("ruiz")
                                  .Attr<int64_t>("method")
                                  .Arg<ffi::Buffer<ffi::F64>>()  // Q
                                  .Arg<ffi::Buffer<ffi::F64>>()  // q
                                  .Arg<ffi::Buffer<ffi::F64>>()  // A
                                  .Arg<ffi::Buffer<ffi::F64>>()  // b
                                  .Arg<ffi::Buffer<ffi::F64>>()  // G
                                  .Arg<ffi::Buffer<ffi::F64>>()  // h
                                  .Arg<ffi::Buffer<ffi::F64>>()  // penalty
                                  .Arg<ffi::Buffer<ffi::F64>>()  // x0
                                  .Arg<ffi::Buffer<ffi::F64>>()  // y0
                                  .Arg<ffi::Buffer<ffi::F64>>()  // z0
                                  .Ret<ffi::Buffer<ffi::F64>>()  // x
                                  .Ret<ffi::Buffer<ffi::F64>>()  // t
                                  .Ret<ffi::Buffer<ffi::F64>>()  // y
                                  .Ret<ffi::Buffer<ffi::F64>>()  // z_t
                                  .Ret<ffi::Buffer<ffi::F64>>()  // z
                                  .Ret<ffi::Buffer<ffi::F64>>()  // info
);
