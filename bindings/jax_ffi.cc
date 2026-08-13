// XLA FFI for ElastiQP
//
// Signature:
//
//   (Q, q, A, b, G, h, penalty)
//       -> (x, t, y, z_t, z_ineq, xr, tr, yr, z_t_r, z_ineq_r, info)
//
// `ruiz` (0/1) enables equilibration
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
// copy of the tight solution.
//
// Cold-start only: the ffi constructs a fresh solver per call since
// JAX/XLA expect functional purity.

#include <algorithm>
#include <cstdint>

#include "elastiqp/elastiqp.hpp"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapMatrix = Eigen::Map<const RowMajorMatrix>;
using MapVector = Eigen::Map<const Eigen::VectorXd>;

ffi::Error ElastiqpPdalImpl(
    double eps_abs, int64_t max_iter, int64_t ruiz, double target_kappa,
    ffi::Buffer<ffi::F64> Q, ffi::Buffer<ffi::F64> q, ffi::Buffer<ffi::F64> A,
    ffi::Buffer<ffi::F64> b, ffi::Buffer<ffi::F64> G, ffi::Buffer<ffi::F64> h,
    ffi::Buffer<ffi::F64> penalty, ffi::ResultBuffer<ffi::F64> x,
    ffi::ResultBuffer<ffi::F64> t, ffi::ResultBuffer<ffi::F64> y,
    ffi::ResultBuffer<ffi::F64> z_t, ffi::ResultBuffer<ffi::F64> z_ineq,
    ffi::ResultBuffer<ffi::F64> xr, ffi::ResultBuffer<ffi::F64> tr,
    ffi::ResultBuffer<ffi::F64> yr, ffi::ResultBuffer<ffi::F64> z_t_r,
    ffi::ResultBuffer<ffi::F64> z_ineq_r, ffi::ResultBuffer<ffi::F64> info) {
  const int64_t n = q.dimensions()[0];
  const int64_t m = b.dimensions()[0];
  const int64_t p = h.dimensions()[0];

  // Fresh solver per call (cold start): XLA assumes FFI calls are pure.
  elastiqp::Solver solver;
  solver.settings.eps_abs = eps_abs;
  solver.settings.eps_duality_gap_abs = eps_abs;
  solver.settings.max_outer_iter = static_cast<int>(max_iter);
  solver.settings.ruiz = ruiz != 0;
  solver.setup(MapMatrix(Q.typed_data(), n, n), MapVector(q.typed_data(), n),
               MapMatrix(A.typed_data(), m, n), MapVector(b.typed_data(), m),
               MapMatrix(G.typed_data(), p, n), MapVector(h.typed_data(), p),
               MapVector(penalty.typed_data(), p));
  const elastiqp::Solution sol = solver.solve();

  Eigen::Map<Eigen::VectorXd>(x->typed_data(), n) = sol.x;
  Eigen::Map<Eigen::VectorXd>(t->typed_data(), p) = sol.t;
  Eigen::Map<Eigen::VectorXd>(y->typed_data(), m) = sol.y;
  Eigen::Map<Eigen::VectorXd>(z_t->typed_data(), p) = sol.z_t;
  Eigen::Map<Eigen::VectorXd>(z_ineq->typed_data(), p) = sol.z_ineq;
  info->typed_data()[0] = static_cast<double>(sol.converged);
  info->typed_data()[1] = static_cast<double>(sol.iters);

  // The relax tolerance is decoupled from eps_abs: the VJP linearizes at
  // the relaxed point, so gradient accuracy is set by THIS residual, and
  // the Newton corrector buys digits cheaply. min() keeps an explicitly
  // tight eps_abs tightening the gradients too, without a loose forward
  // tolerance loosening them.
  const elastiqp::Solution& rsol =
      (target_kappa > 0 && p > 0)
          ? solver.relax(target_kappa, std::min(eps_abs, 1e-6), 50)
          : sol;
  Eigen::Map<Eigen::VectorXd>(xr->typed_data(), n) = rsol.x;
  Eigen::Map<Eigen::VectorXd>(tr->typed_data(), p) = rsol.t;
  Eigen::Map<Eigen::VectorXd>(yr->typed_data(), m) = rsol.y;
  Eigen::Map<Eigen::VectorXd>(z_t_r->typed_data(), p) = rsol.z_t;
  Eigen::Map<Eigen::VectorXd>(z_ineq_r->typed_data(), p) = rsol.z_ineq;
  info->typed_data()[2] = static_cast<double>(rsol.converged);
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(ElastiqpPdalSolve, ElastiqpPdalImpl,
                              ffi::Ffi::Bind()
                                  .Attr<double>("eps_abs")
                                  .Attr<int64_t>("max_iter")
                                  .Attr<int64_t>("ruiz")
                                  .Attr<double>("target_kappa")
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
                                  .Ret<ffi::Buffer<ffi::F64>>()  // z_ineq
                                  .Ret<ffi::Buffer<ffi::F64>>()  // xr
                                  .Ret<ffi::Buffer<ffi::F64>>()  // tr
                                  .Ret<ffi::Buffer<ffi::F64>>()  // yr
                                  .Ret<ffi::Buffer<ffi::F64>>()  // z_t_r
                                  .Ret<ffi::Buffer<ffi::F64>>()  // z_ineq_r
                                  .Ret<ffi::Buffer<ffi::F64>>()  // info
);
