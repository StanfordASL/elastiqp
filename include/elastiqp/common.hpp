// ElastiQP: shared types and helpers for the three elastic QP solvers
//
//   minimize    0.5 x^T Q x + q^T x + penalty^T t
//   subject to  A x == b          (hard, dual y)
//               G x - t <= h      (soft, slack s_ineq, dual z)
//               t >= 0            (slack s_t, dual z_t)
//
// Every inequality row is l1-elastic with weight penalty_i > 0 (inf = hard
// row), so the problem is feasible iff A x = b is consistent. Eliminating
// the slacks analytically leaves the strict QP's KKT system with every
// inequality multiplier boxed to [0, penalty_i], which each method
// (elastiqp_das.hpp, elastiqp_pdal.hpp, elastiqp_ipm.hpp) exploits in its
// own way. This header holds what the three share: the status codes, the
// Solution certificate, the Ruiz equilibration primitives, and the
// equality-consistency certificate.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <limits>

namespace elastiqp {

using Eigen::MatrixXd;
using Eigen::VectorXd;

enum class Status {
  kUnsolved = 0,
  kSolved = 1,
  kMaxIter = 2,
  kNumerics = 3,
  kInfeasible = 4,  // inconsistent equalities, or conflicting hard rows
};

inline const char* status_name(Status s) {
  switch (s) {
    case Status::kUnsolved: return "unsolved";
    case Status::kSolved: return "solved";
    case Status::kMaxIter: return "max_iter";
    case Status::kNumerics: return "numerics";
    case Status::kInfeasible: return "infeasible";
  }
  return "?";
}

// Solution certificate, always in the user's frame (unscaled).
struct Solution {
  VectorXd x;
  VectorXd t;  // elastic slacks max(G x - h, 0) at a solution
  VectorXd y;  // equality duals
  VectorXd z;  // inequality duals, in [0, penalty]
  // Dual of t >= 0 in the expanded form. At a solution z_t = penalty - z;
  // only the kappa-relaxed point of relax() (PDAL, IPM) moves it off that
  // identity, which is why the backward pass (kkt_vjp.hpp) reads it rather
  // than reconstructing it. The expanded-form slacks are not reported: at a
  // solution they are t and [t - (G x - h)]_+, and at a relaxed point they
  // equal those to within primal_res.
  VectorXd z_t;
  Status status = Status::kUnsolved;
  int converged = 0;    // 1 iff status == kSolved
  int iters = 0;        // method's inner iterations (Newton steps, IPM
                        // iterations, working-set changes)
  int outer_iters = 0;  // BCL rounds (PDAL), proximal rounds (DAS), 0 (IPM)
  int n_active = 0, n_saturated = 0;  // three-state row counts
  double primal_obj = 0.0;
  double primal_res = 0.0;
  double dual_res = 0.0;
  double duality_gap = 0.0;
};

// ---- Ruiz equilibration primitives -----------------------------------------
// All three solvers equilibrate the stacked symmetric system
// [Q A' G'; A 0 0; G 0 0] (proxqp's sweep, PIQP's limit_scaling) plus a cost
// normalization, with the penalties scaled along with their rows. Each
// solver keeps its own sweep loop because it stores the data differently;
// these are the pieces the loops share.

// PIQP's guard: a norm below 1e-4 is treated as 1 (the row/column is left
// unscaled, rather than amplified without bound), a norm above 1e4 is
// capped so one sweep's factor stays in [1e-2, 100].
inline double ruiz_limit_scaling(double nrm) {
  return nrm < 1e-4 ? 1.0 : std::min(nrm, 1e4);
}

// Column-wise max-abs of M folded into colmax (max with the existing
// entries) and row-wise max-abs folded into rowmax, one contiguous column
// at a time (both reductions vectorize; a strided rowwise() pass would
// not). For a transposed constraint block C' (n x k) call it with the
// roles swapped: colmax is then per constraint row, rowmax per variable.
inline void fold_max_abs(const MatrixXd& M, VectorXd& colmax,
                         VectorXd& rowmax) {
  for (Eigen::Index k = 0; k < M.cols(); ++k) {
    const auto col = M.col(k).cwiseAbs();
    rowmax = rowmax.cwiseMax(col);
    colmax[k] = std::max(colmax[k], col.maxCoeff());
  }
}

// Turn accumulated max-norms into scale factors 1/sqrt(limit(nrm)) in
// place; returns the largest |1 - factor| (0 = already equilibrated).
inline double ruiz_factors(VectorXd& f) {
  double dev = 0.0;
  for (Eigen::Index k = 0; k < f.size(); ++k) {
    f[k] = 1.0 / std::sqrt(ruiz_limit_scaling(f[k]));
    dev = std::max(dev, std::abs(1.0 - f[k]));
  }
  return dev;
}

// Largest factor by which any max-norm behind the factors f (from
// ruiz_factors) is off from 1: max(r, 1/r) with r = 1/f^2 = the norm.
inline double ruiz_drift(const VectorXd& f, double worst = 1.0) {
  for (Eigen::Index k = 0; k < f.size(); ++k) {
    const double r = f[k] * f[k];
    worst = std::max(worst, std::max(r, 1.0 / r));
  }
  return worst;
}

// Per-sweep cost normalization gamma = 1 / max(1, mean column max-norm of
// the (already column-scaled) Q).
inline double ruiz_cost_gamma(const MatrixXd& Q) {
  double mean = 0.0;
  for (Eigen::Index k = 0; k < Q.cols(); ++k) {
    mean += Q.col(k).cwiseAbs().maxCoeff();
  }
  mean /= std::max<double>(1.0, static_cast<double>(Q.cols()));
  return 1.0 / std::max(1.0, mean);
}

// ---- Equality consistency certificate --------------------------------------
// A x = b is the only way the elastic problem can be infeasible. set_A()
// certifies full row rank (an LDLT of the m x m Gram matrix A A', falling
// back to a column-pivoted QR when the Gram matrix is not clearly positive
// definite); when A is rank deficient, bound(b) is the RMS residual of the
// least-squares solution, a lower bound on the reachable ||A x - b||, so
// bound(b) > eps means eps is unreachable. Work is only done on set_A / a
// rank-deficient A; a full-rank certificate costs one m x m LDLT.
class EqCertificate {
 public:
  void set_A(const MatrixXd& A) {
    m_ = A.rows();
    n_ = A.cols();
    rank_deficient_ = false;
    if (m_ == 0) return;
    if (m_ <= n_) {
      gram_.resize(m_, m_);
      gram_.setZero();
      gram_.selfadjointView<Eigen::Lower>().rankUpdate(A);
      ldlt_.compute(gram_);
      if (ldlt_.info() == Eigen::Success) {
        const auto& d = ldlt_.vectorD();
        const double dmax = d.maxCoeff();
        if (std::isfinite(dmax) && d.minCoeff() > 1e-10 * dmax) {
          return;  // certified full row rank
        }
      }
    }
    qr_.compute(A);
    rank_deficient_ = qr_.rank() < m_;
    if (rank_deficient_) A_ = A;
  }
  bool rank_deficient() const { return rank_deficient_; }
  // Lower bound on the reachable equality residual (0 if full rank).
  double bound(const VectorXd& b) const {
    if (!rank_deficient_ || m_ == 0) return 0.0;
    const VectorXd xls = qr_.solve(b);
    return (A_ * xls - b).norm() / std::sqrt(static_cast<double>(m_));
  }

 private:
  Eigen::Index m_ = 0, n_ = 0;
  bool rank_deficient_ = false;
  MatrixXd gram_, A_;
  Eigen::LDLT<MatrixXd> ldlt_;
  Eigen::ColPivHouseholderQR<MatrixXd> qr_;
};

// ---- Equality-only QP -------------------------------------------------------
// min 0.5 x'Qx + q'x  s.t.  A x = b -- the p == 0 case of every backend --
// solved in the range space the way the DAS working set does it: K = Q +
// rho I (LLT), Y = K^-1 A', S = A Y (+ delta I, LLT), then iterative
// refinement against the UNregularized KKT residual until the proximal bias
// of rho is gone (one or two steps for a PD Q; a few more for a singular Q
// whose reduced Hessian is PD). delta is only needed when A has dependent
// rows (S singular) and only biases y along null(A'), which x never sees;
// it is taken from the equality certificate when there is one, else from
// a failed LLT of S. Returns false (x, y untouched) when K has no Cholesky
// factor (indefinite Q) or S cannot be regularized -- the caller keeps the
// pivoted QR of the full KKT matrix as the fallback. A colPivHouseholderQr
// of the (n+m) KKT matrix, the previous method, is unblocked O((n+m)^3) and
// took 2.5 s at n+m = 2263 against ~60 ms here.
inline bool SolveEqualityQP(const MatrixXd& Q, const VectorXd& q,
                            const MatrixXd& A, const VectorXd& b, double rho,
                            bool rank_deficient, VectorXd& x, VectorXd& y,
                            int max_refine = 20) {
  const Eigen::Index m = b.size();
  MatrixXd K = Q;
  K.diagonal().array() += rho;
  Eigen::LLT<MatrixXd, Eigen::Lower> lltK(K);
  if (lltK.info() != Eigen::Success) return false;
  MatrixXd Y, S;
  Eigen::LLT<MatrixXd, Eigen::Lower> lltS;
  if (m > 0) {
    Y = lltK.solve(A.transpose());
    S.noalias() = A * Y;
    S = 0.5 * (S + S.transpose()).eval();
    const double smax = std::max(1.0, S.diagonal().maxCoeff());
    double delta = rank_deficient ? 1e-8 * smax : 0.0;
    for (int attempt = 0;; ++attempt) {
      MatrixXd Sd = S;
      Sd.diagonal().array() += delta;
      lltS.compute(Sd);
      if (lltS.info() == Eigen::Success) break;
      if (attempt >= 6) return false;
      delta = delta == 0.0 ? 1e-10 * smax : 100.0 * delta;
    }
  }
  VectorXd xs = lltK.solve(-q), ys(m);
  if (m > 0) {
    ys = lltS.solve(A * xs - b);
    xs -= Y * ys;
  }
  // Refinement: solve the same system for the correction of the true
  // residual. Stop at roundoff, on stall, or on a non-finite iterate.
  VectorXd rd(q.size()), rp(m), dx, dy;
  double prev = std::numeric_limits<double>::infinity();
  for (int it = 0; it < max_refine; ++it) {
    rd.noalias() = Q * xs;
    rd += q;
    double scale = std::max({1.0, rd.lpNorm<Eigen::Infinity>(),
                             q.lpNorm<Eigen::Infinity>()});
    if (m > 0) {
      rd.noalias() += A.transpose() * ys;
      rp.noalias() = A * xs;
      scale = std::max({scale, rp.lpNorm<Eigen::Infinity>(),
                        b.lpNorm<Eigen::Infinity>()});
      rp -= b;
    }
    const double res = std::max(rd.lpNorm<Eigen::Infinity>(),
                                m > 0 ? rp.lpNorm<Eigen::Infinity>() : 0.0);
    if (!std::isfinite(res)) return false;
    if (res <= 1e-14 * scale || res > 0.5 * prev) break;
    prev = res;
    dx = lltK.solve(-rd);
    if (m > 0) {
      dy = lltS.solve(A * dx + rp);
      dx -= Y * dy;
      ys += dy;
    }
    xs += dx;
  }
  if (!xs.allFinite() || !ys.allFinite()) return false;
  x = xs;
  y = ys;
  return true;
}

// ---- Row classification of a certificate ----------------------------------
// Three-state counts from the duals (z in [0, penalty]): saturated when the
// row is violated (t > tol), active when its dual is strictly positive and
// the row is not violated, inactive otherwise.
inline void count_row_states(const VectorXd& t, const VectorXd& z, double tol,
                             int& n_active, int& n_saturated) {
  n_active = 0;
  n_saturated = 0;
  for (Eigen::Index i = 0; i < t.size(); ++i) {
    if (t[i] > tol) {
      ++n_saturated;
    } else if (z[i] > tol) {
      ++n_active;
    }
  }
}

}  // namespace elastiqp
