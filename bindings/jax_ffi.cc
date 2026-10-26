// XLA FFI handlers for ElastiQP. Every call builds a fresh solver (XLA
// assumes purity), so warm starting is an explicit argument.
//
//   ElastiqpSolve: (Q, q, A, b, G, h, penalty, x0, y0, z0)
//                    -> (x, t, y, z_t, z, xr, tr, yr, z_t_r, z_r, info)
//
// method: 0 = das, 1 = pdal, 2 = ipm. max_iter is the backend's outer budget.
// warm != 0 seeds the solve from (x0, y0, z0) (das and pdal only); otherwise
// they are ignored.
// The *r block is the kappa-relaxed point the Python wrappers differentiate
// at; it copies the tight solution when target_kappa <= 0. info = [converged,
// iters, relax_converged]. A has 0 rows and b 0 entries when there are no
// equalities.
//
// Batched: every buffer may carry the same leading batch dimensions (vmap
// with vmap_method="broadcast_all"); the problems are solved one by one.

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
  solver.settings.max_outer_iter = static_cast<int>(max_iter);
  solver.settings.ruiz = ruiz != 0;
}
template <>
void apply_options(elastiqp::ipm::Solver& solver, double eps_abs,
                   int64_t max_iter, int64_t ruiz) {
  solver.settings.eps_abs = eps_abs;
  solver.settings.max_iter = static_cast<int>(max_iter);
  solver.settings.ruiz = ruiz != 0;
}

// x0 != nullptr seeds the solve from (x0, y0, z0); the IPM has no warm
// start (dispatch rejects it).
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
  if constexpr (!std::is_same_v<SolverT, elastiqp::ipm::Solver>) {
    if (x0 != nullptr) {
      solver.set_warm_start(VectorXd(MapVector(x0, n)),
                            VectorXd(MapVector(y0, m)),
                            VectorXd(MapVector(z0, p)));
    }
  }
  sol = solver.solve();
  // Gradient accuracy is set by the relax residual, so cap it at 1e-6.
  rsol = (target_kappa > 0 && p > 0)
             ? solver.relax(target_kappa, std::min(eps_abs, 1e-6), 50)
             : sol;
}

ffi::Error dispatch(int64_t method, double eps_abs, int64_t max_iter,
                    int64_t ruiz, double target_kappa, int64_t n, int64_t m,
                    int64_t p, const double* Q, const double* q,
                    const double* A, const double* b, const double* G,
                    const double* h, const double* penalty, const double* x0,
                    const double* y0, const double* z0, elastiqp::Solution& sol,
                    elastiqp::Solution& rsol) {
  switch (method) {
    case 0:
      run_solver<elastiqp::das::Solver>(eps_abs, max_iter, ruiz, target_kappa,
                                        n, m, p, Q, q, A, b, G, h, penalty, x0,
                                        y0, z0, sol, rsol);
      return ffi::Error::Success();
    case 1:
      run_solver<elastiqp::pdal::Solver>(eps_abs, max_iter, ruiz, target_kappa,
                                         n, m, p, Q, q, A, b, G, h, penalty, x0,
                                         y0, z0, sol, rsol);
      return ffi::Error::Success();
    case 2:
      if (x0 != nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "method='ipm' has no warm start; use method='das' "
                          "or 'pdal'");
      }
      run_solver<elastiqp::ipm::Solver>(eps_abs, max_iter, ruiz, target_kappa,
                                        n, m, p, Q, q, A, b, G, h, penalty, x0,
                                        y0, z0, sol, rsol);
      return ffi::Error::Success();
    default:
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "method must be 0 (das), 1 (pdal) or 2 (ipm)");
  }
}

void write_solution(const elastiqp::Solution& sol, int64_t n, int64_t m,
                    int64_t p, double* x, double* t, double* y, double* z_t,
                    double* z) {
  Eigen::Map<VectorXd>(x, n) = sol.x;
  Eigen::Map<VectorXd>(t, p) = sol.t;
  Eigen::Map<VectorXd>(y, m) = sol.y;
  Eigen::Map<VectorXd>(z_t, p) = sol.z_t;
  Eigen::Map<VectorXd>(z, p) = sol.z;
}

// Product of the leading (batch) dimensions of a buffer whose trailing
// `trailing` dimensions are per-problem.
int64_t batch_size(ffi::Buffer<ffi::F64>& buf, size_t trailing) {
  const auto dims = buf.dimensions();
  int64_t count = 1;
  for (size_t i = 0; i + trailing < dims.size(); ++i) count *= dims[i];
  return count;
}

ffi::Error ElastiqpSolveImpl(
    double eps_abs, int64_t max_iter, int64_t ruiz, double target_kappa,
    int64_t warm, int64_t method, ffi::Buffer<ffi::F64> Q,
    ffi::Buffer<ffi::F64> q, ffi::Buffer<ffi::F64> A, ffi::Buffer<ffi::F64> b,
    ffi::Buffer<ffi::F64> G, ffi::Buffer<ffi::F64> h,
    ffi::Buffer<ffi::F64> penalty, ffi::Buffer<ffi::F64> x0,
    ffi::Buffer<ffi::F64> y0, ffi::Buffer<ffi::F64> z0,
    ffi::ResultBuffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> t,
    ffi::ResultBuffer<ffi::F64> y, ffi::ResultBuffer<ffi::F64> z_t,
    ffi::ResultBuffer<ffi::F64> z, ffi::ResultBuffer<ffi::F64> xr,
    ffi::ResultBuffer<ffi::F64> tr, ffi::ResultBuffer<ffi::F64> yr,
    ffi::ResultBuffer<ffi::F64> z_t_r, ffi::ResultBuffer<ffi::F64> z_r,
    ffi::ResultBuffer<ffi::F64> info) {
  const auto last = [](ffi::Buffer<ffi::F64>& buf) {
    return buf.dimensions().back();
  };
  const int64_t n = last(q);
  const int64_t m = last(b);
  const int64_t p = last(h);
  const int64_t batch = batch_size(q, 1);
  for (ffi::Buffer<ffi::F64>* buf : {&q, &b, &h, &penalty, &x0, &y0, &z0}) {
    if (batch_size(*buf, 1) != batch) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "all arguments must share the same batch dimensions");
    }
  }
  for (ffi::Buffer<ffi::F64>* buf : {&Q, &A, &G}) {
    if (batch_size(*buf, 2) != batch) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "all arguments must share the same batch dimensions");
    }
  }
  const bool w = warm != 0;
  if (w && (last(x0) != n || last(y0) != m || last(z0) != p)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "warm start (x0, y0, z0) must have shapes (n,), (m,), "
                      "(p,) matching (q, b, h)");
  }

  for (int64_t i = 0; i < batch; ++i) {
    elastiqp::Solution sol, rsol;
    ffi::Error err = dispatch(
        method, eps_abs, max_iter, ruiz, target_kappa, n, m, p,
        Q.typed_data() + i * n * n, q.typed_data() + i * n,
        A.typed_data() + i * m * n, b.typed_data() + i * m,
        G.typed_data() + i * p * n, h.typed_data() + i * p,
        penalty.typed_data() + i * p, w ? x0.typed_data() + i * n : nullptr,
        w ? y0.typed_data() + i * m : nullptr,
        w ? z0.typed_data() + i * p : nullptr, sol, rsol);
    if (err.failure()) return err;
    write_solution(sol, n, m, p, x->typed_data() + i * n,
                   t->typed_data() + i * p, y->typed_data() + i * m,
                   z_t->typed_data() + i * p, z->typed_data() + i * p);
    write_solution(rsol, n, m, p, xr->typed_data() + i * n,
                   tr->typed_data() + i * p, yr->typed_data() + i * m,
                   z_t_r->typed_data() + i * p, z_r->typed_data() + i * p);
    double* inf = info->typed_data() + i * 3;
    inf[0] = static_cast<double>(sol.converged);
    inf[1] = static_cast<double>(sol.iters);
    inf[2] = static_cast<double>(rsol.converged);
  }
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(ElastiqpSolve, ElastiqpSolveImpl,
                              ffi::Ffi::Bind()
                                  .Attr<double>("eps_abs")
                                  .Attr<int64_t>("max_iter")
                                  .Attr<int64_t>("ruiz")
                                  .Attr<double>("target_kappa")
                                  .Attr<int64_t>("warm")
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
                                  .Ret<ffi::Buffer<ffi::F64>>()  // xr
                                  .Ret<ffi::Buffer<ffi::F64>>()  // tr
                                  .Ret<ffi::Buffer<ffi::F64>>()  // yr
                                  .Ret<ffi::Buffer<ffi::F64>>()  // z_t_r
                                  .Ret<ffi::Buffer<ffi::F64>>()  // z_r
                                  .Ret<ffi::Buffer<ffi::F64>>()  // info
);
