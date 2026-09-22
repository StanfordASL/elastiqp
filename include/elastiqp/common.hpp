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
    case Status::kUnsolved:
      return "unsolved";
    case Status::kSolved:
      return "solved";
    case Status::kMaxIter:
      return "max_iter";
    case Status::kNumerics:
      return "numerics";
    case Status::kInfeasible:
      return "infeasible";
  }
  return "?";
}

struct Solution {
  VectorXd x;
  VectorXd t;  // elastic slacks, max(G x - h, 0)
  VectorXd y;
  VectorXd z;  // inequality duals, in [0, penalty]
  // Dual of t >= 0; equals penalty - z except at a kappa-relaxed point
  // (read by kkt_vjp.hpp).
  VectorXd z_t;
  Status status = Status::kUnsolved;
  int converged = 0;
  int iters = 0;        // inner steps
  int outer_iters = 0;  // BCL rounds (PDAL), proximal rounds (DAS), 0 (IPM)
  int n_active = 0, n_saturated = 0;
  double primal_obj = 0.0;
  double primal_res = 0.0;
  double dual_res = 0.0;
  double duality_gap = 0.0;
};

// PIQP's limit_scaling: leave tiny norms alone, cap the rest.
inline double ruiz_limit_scaling(double nrm) {
  return nrm < 1e-4 ? 1.0 : std::min(nrm, 1e4);
}

inline void fold_max_abs(const MatrixXd& M, VectorXd& colmax,
                         VectorXd& rowmax) {
  for (Eigen::Index k = 0; k < M.cols(); ++k) {
    const auto col = M.col(k).cwiseAbs();
    rowmax = rowmax.cwiseMax(col);
    colmax[k] = std::max(colmax[k], col.maxCoeff());
  }
}

// In place norm -> 1/sqrt(norm); returns the max deviation from 1
// (sweep convergence).
inline double ruiz_factors(VectorXd& f) {
  double dev = 0.0;
  for (Eigen::Index k = 0; k < f.size(); ++k) {
    f[k] = 1.0 / std::sqrt(ruiz_limit_scaling(f[k]));
    dev = std::max(dev, std::abs(1.0 - f[k]));
  }
  return dev;
}

// Worst squared factor or its inverse; drives the ruiz_refresh gate.
inline double ruiz_drift(const VectorXd& f, double worst = 1.0) {
  for (Eigen::Index k = 0; k < f.size(); ++k) {
    const double r = f[k] * f[k];
    worst = std::max(worst, std::max(r, 1.0 / r));
  }
  return worst;
}

inline double ruiz_cost_gamma(const MatrixXd& Q) {
  double mean = 0.0;
  for (Eigen::Index k = 0; k < Q.cols(); ++k) {
    mean += Q.col(k).cwiseAbs().maxCoeff();
  }
  mean /= std::max<double>(1.0, static_cast<double>(Q.cols()));
  return 1.0 / std::max(1.0, mean);
}

// Ingestion-time rank check of A: Gram LDLT fast path, ColPivQR fallback.
// bound() is the RMS least-squares residual of A x = b, a lower bound on
// the reachable residual; nonzero only when A is rank-deficient.
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
          return;  // full row rank
        }
      }
    }
    qr_.compute(A);
    rank_deficient_ = qr_.rank() < m_;
    if (rank_deficient_) A_ = A;
  }
  bool rank_deficient() const { return rank_deficient_; }

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

// Equality QP via the Schur complement of Q + rho I, then iterative
// refinement against the true KKT residual removes the rho bias; false on
// factorization failure.
inline bool RangeSpaceEqualityQP(const MatrixXd& Q, const VectorXd& q,
                                 const MatrixXd& A, const VectorXd& b,
                                 double rho, bool rank_deficient, VectorXd& x,
                                 VectorXd& y, int max_refine = 20) {
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
    // Regularize S when A is rank-deficient.
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
  VectorXd rd(q.size()), rp(m), dx, dy;
  double prev = std::numeric_limits<double>::infinity();
  for (int it = 0; it < max_refine; ++it) {
    rd.noalias() = Q * xs;
    rd += q;
    double scale = std::max(
        {1.0, rd.lpNorm<Eigen::Infinity>(), q.lpNorm<Eigen::Infinity>()});
    if (m > 0) {
      rd.noalias() += A.transpose() * ys;
      rp.noalias() = A * xs;
      scale = std::max(
          {scale, rp.lpNorm<Eigen::Infinity>(), b.lpNorm<Eigen::Infinity>()});
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

// Range-space solve, falling back to a dense KKT ColPivQR.
inline void SolveEqualityQP(const MatrixXd& Q, const VectorXd& q,
                            const MatrixXd& A, const VectorXd& b, double rho,
                            bool rank_deficient, VectorXd& x, VectorXd& y) {
  if (RangeSpaceEqualityQP(Q, q, A, b, rho, rank_deficient, x, y)) return;
  const Eigen::Index n = q.size(), m = b.size();
  if (m == 0) {
    x = Eigen::LDLT<MatrixXd>(Q).solve(-q);
    y.resize(0);
    return;
  }
  MatrixXd Kf = MatrixXd::Zero(n + m, n + m);
  Kf.topLeftCorner(n, n) = Q;
  Kf.topRightCorner(n, m) = A.transpose();
  Kf.bottomLeftCorner(m, n) = A;
  VectorXd rhs(n + m);
  rhs.head(n) = -q;
  rhs.tail(m) = b;
  const VectorXd xy = Kf.colPivHouseholderQr().solve(rhs);
  x = xy.head(n);
  y = xy.tail(m);
}

struct EqualityKKTStats {
  double primal_res = 0.0, primal_res_rel = 0.0;
  double dual_res = 0.0, dual_res_rel = 0.0;
  double primal_obj = 0.0, duality_gap = 0.0, duality_gap_rel = 0.0;

  bool converged(double eps_abs, double eps_rel, bool check_gap,
                 double eps_gap_abs, double eps_gap_rel) const {
    return (primal_res < eps_abs || primal_res_rel < eps_rel) &&
           (dual_res < eps_abs || dual_res_rel < eps_rel) &&
           (!check_gap || duality_gap < eps_gap_abs ||
            duality_gap_rel < eps_gap_rel);
  }
};

// Residuals for the no-inequality shortcut.
inline EqualityKKTStats ComputeEqualityKKT(const MatrixXd& Q, const VectorXd& q,
                                           const MatrixXd& A, const VectorXd& b,
                                           const VectorXd& x,
                                           const VectorXd& y) {
  EqualityKKTStats st;
  const VectorXd Qx = Q * x;
  VectorXd dual = Qx + q;
  double dual_rel_norm =
      std::max(Qx.lpNorm<Eigen::Infinity>(), q.lpNorm<Eigen::Infinity>());
  double by = 0.0;
  if (b.size() > 0) {
    const VectorXd Aty = A.transpose() * y;
    dual += Aty;
    dual_rel_norm = std::max(dual_rel_norm, Aty.lpNorm<Eigen::Infinity>());
    const VectorXd Ax = A * x;
    st.primal_res = (Ax - b).lpNorm<Eigen::Infinity>();
    st.primal_res_rel =
        st.primal_res / std::max(1.0, std::max(Ax.lpNorm<Eigen::Infinity>(),
                                               b.lpNorm<Eigen::Infinity>()));
    by = b.dot(y);
  }
  st.dual_res = dual.lpNorm<Eigen::Infinity>();
  st.dual_res_rel = st.dual_res / std::max(1.0, dual_rel_norm);
  const double xQx = x.dot(Qx), qx = q.dot(x);
  st.primal_obj = 0.5 * xQx + qx;
  st.duality_gap = std::abs(st.primal_obj - (-0.5 * xQx - by));
  st.duality_gap_rel =
      st.duality_gap /
      std::max(1.0, std::max({std::abs(xQx), std::abs(qx), std::abs(by)}));
  return st;
}

// Three-state row counts: saturated if t > tol, else active if z > tol.
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
