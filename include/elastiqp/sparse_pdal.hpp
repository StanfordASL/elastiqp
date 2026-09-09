// ElastiQP-PDAL, sparse backend: the elastic primal-dual augmented Lagrangian
// method of pdal.hpp on sparse data (Eigen::SparseMatrix), for the
// multiple-shooting / MPC problems whose Markovian structure makes dense
// factorizations wasteful.
//
//   minimize    0.5 x^T Q x + q^T x + penalty^T t
//   subject to  A x == b          (hard, dual y)
//               G x - t <= h      (soft, slack s_ineq, dual z)
//               t >= 0            (slack s_t, dual z_t)
//
// The method (three-state row classification on the unclamped multiplier
// estimate z~, semismooth Newton with an exact line search, elastic BCL
// outer loop) is identical to the dense backend; see pdal.hpp and
// docs/elastic_bcl.md. What differs is the linear algebra:
//
//   * Instead of condensing onto the n x n SPD matrix K, each Newton step
//     solves the quasidefinite augmented system (proxsuite's sparse form)
//
//       [ Q + rho I     A'        G_J'    ] [dx]   [rhs_x]
//       [ A          -mu_eq I      0      ] [dy] = [rhs_y]
//       [ G_J           0      -mu_in I   ] [dz]   [rhs_z]
//
//     with a sparse LDL' (Eigen::SimplicialLDLT, AMD ordering). The matrix
//     is quasidefinite for any active set, so the factorization needs no
//     pivoting and the fill-reducing ordering is computed ONCE, at setup,
//     for the pattern with every inequality row present; a row that is not
//     active keeps its column (coupling zeroed, diagonal -1), so the
//     symbolic analysis never has to be redone. Nothing is squared: the
//     dense backend's K = Q + A'A/mu_eq + G_J'G_J/mu_in has condition
//     number ~1/mu, this system does not.
//
//   * Active-set changes between refactorizations are folded in as a
//     Woodbury correction of the cached factor (the sparse counterpart of
//     the dense backend's rank-one LLT updates): with F the rows whose
//     state differs from the factored one, the current Newton matrix is
//     K0 + U S U' with U = [G_F'; 0; 0] and S = diag(+-1/mu_in), and every
//     solve costs one solve with K0 plus |F| cached solves. Refactorization
//     happens when mu or the data change, or F outgrows its budget.
//
//   * Solves are iteratively refined against the current Newton matrix
//     (the LDL' of a matrix with diagonals spanning rho .. -mu_in loses
//     digits; two or three refinement steps restore them).
//
// The Ruiz equilibration, the equality-consistency certificate and the
// p == 0 path have sparse implementations here; relax() and the KKT VJP
// (differentiation) are not implemented for this backend.

#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "elastiqp/common.hpp"
#include "elastiqp/pdal.hpp"

namespace elastiqp::sparse_pdal {

using SpMat = Eigen::SparseMatrix<double>;  // column-major, int indices
using SpMatRow = Eigen::SparseMatrix<double, Eigen::RowMajor>;

// The dense backend's settings (termination, BCL schedule, warm start,
// Ruiz, incremental-update budget) plus the sparse linear algebra knobs.
// incremental_updates / incremental_update_budget /
// incremental_update_max_flips govern the Woodbury correction here (0
// max_flips = auto: the mean column count of L, the point where |F| cached
// solves cost about as much as a refactorization). The relax_* settings
// are unused.
struct Settings : pdal::Settings {
  Settings() {
    // Sparse defaults that differ from the dense backend (validated on the
    // MPC scenarios of the benchmarks repo, docs/sparse_pdal.md):
    //  * mu_eq_init 1e-5: the augmented system is not squared, so a small
    //    equality penalty costs nothing in conditioning and saves the BCL
    //    rounds a dynamics-dominated problem spends walking mu_eq down;
    //  * warm_keep_mu: a quiet control tick then reuses the previous
    //    factorization outright (0.1 factorizations per tick instead of
    //    2-3), gated so a tick that struggled resets the penalties.
    mu_eq_init = 1e-5;
    warm_keep_mu = true;
    warm_keep_mu_max_iters = 25;
  }
  // Iterative refinement of each Newton solve against the Newton matrix
  // (at most refine_iters correction steps, 0 = none). The residual check
  // is one product with the KKT matrix; a correction is one more solve with
  // the cached factor. Stops as soon as the residual is below refine_tol
  // times (1 + |rhs|_inf) -- the common case with a well-scaled factor,
  // so the check is all that is paid -- or when it no longer halves.
  int refine_iters = 3;
  double refine_tol = 1e-11;
  // Equality certificate fallback: when the Gram matrix A A' is not clearly
  // positive definite, a sparse QR of A decides rank deficiency. It is
  // skipped (the check reports "consistent") above this many nonzeros in
  // A, where the QR would dominate setup.
  Eigen::Index eq_check_qr_max_nnz = 2000000;
  // Print one line per BCL round (stderr): residuals, mu, inner iterations
  bool trace = false;
};

// ---- sparse helpers --------------------------------------------------------
namespace detail {

// M <- diag(r) M diag(c), in place.
inline void ScaleRowsCols(SpMat& M, const VectorXd& r, const VectorXd& c) {
  for (Eigen::Index k = 0; k < M.outerSize(); ++k) {
    for (SpMat::InnerIterator it(M, k); it; ++it) {
      it.valueRef() *= r[it.row()] * c[k];
    }
  }
}

// Fold the column max-abs of M into colmax and its row max-abs into rowmax.
inline void FoldMaxAbs(const SpMat& M, VectorXd& colmax, VectorXd& rowmax) {
  for (Eigen::Index k = 0; k < M.outerSize(); ++k) {
    double cm = colmax[k];
    for (SpMat::InnerIterator it(M, k); it; ++it) {
      const double v = std::abs(it.value());
      cm = std::max(cm, v);
      rowmax[it.row()] = std::max(rowmax[it.row()], v);
    }
    colmax[k] = cm;
  }
}

inline double CostGamma(const SpMat& Q) {
  double mean = 0.0;
  for (Eigen::Index k = 0; k < Q.outerSize(); ++k) {
    double cm = 0.0;
    for (SpMat::InnerIterator it(Q, k); it; ++it) {
      cm = std::max(cm, std::abs(it.value()));
    }
    mean += cm;
  }
  mean /= std::max<double>(1.0, static_cast<double>(Q.cols()));
  return 1.0 / std::max(1.0, mean);
}

inline SpMat Symmetrize(const SpMat& Q) {
  SpMat Qt = Q.transpose();
  SpMat S = 0.5 * (Q + Qt);
  S.makeCompressed();
  return S;
}

inline bool SamePattern(const SpMat& a, const SpMat& b) {
  if (a.rows() != b.rows() || a.cols() != b.cols() ||
      a.nonZeros() != b.nonZeros()) {
    return false;
  }
  for (Eigen::Index k = 0; k <= a.outerSize(); ++k) {
    if (a.outerIndexPtr()[k] != b.outerIndexPtr()[k]) return false;
  }
  for (Eigen::Index k = 0; k < a.nonZeros(); ++k) {
    if (a.innerIndexPtr()[k] != b.innerIndexPtr()[k]) return false;
  }
  return true;
}

}  // namespace detail

// Sparse equality-consistency certificate (common.hpp EqCertificate): full
// row rank certified by an LDL' of the Gram matrix A A', else a sparse QR
// decides the rank and gives the least-squares residual bound.
class SparseEqCertificate {
 public:
  void set_A(const SpMat& A, Eigen::Index qr_max_nnz) {
    m_ = A.rows();
    rank_deficient_ = false;
    have_qr_ = false;
    zero_rows_.clear();
    if (m_ == 0) return;
    // All-zero rows (a fixed-pattern schedule leaves them) are consistent
    // iff their right-hand side is 0; they are stripped before the rank
    // test, which would otherwise always report a deficiency.
    std::vector<int> count(static_cast<size_t>(m_), 0);
    for (Eigen::Index k = 0; k < A.outerSize(); ++k) {
      for (SpMat::InnerIterator it(A, k); it; ++it) {
        if (it.value() != 0.0) ++count[static_cast<size_t>(it.row())];
      }
    }
    std::vector<Eigen::Index> keep;
    for (Eigen::Index i = 0; i < m_; ++i) {
      if (count[static_cast<size_t>(i)] == 0) {
        zero_rows_.push_back(i);
      } else {
        keep.push_back(i);
      }
    }
    const Eigen::Index mr = static_cast<Eigen::Index>(keep.size());
    if (mr == 0) return;
    SpMat Ar;
    if (zero_rows_.empty()) {
      Ar = A;
    } else {
      std::vector<Eigen::Index> row_of(static_cast<size_t>(m_), -1);
      for (Eigen::Index r = 0; r < mr; ++r) row_of[static_cast<size_t>(keep[static_cast<size_t>(r)])] = r;
      std::vector<Eigen::Triplet<double>> t;
      for (Eigen::Index k = 0; k < A.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A, k); it; ++it) {
          const Eigen::Index r = row_of[static_cast<size_t>(it.row())];
          if (r >= 0) t.emplace_back(r, k, it.value());
        }
      }
      Ar.resize(mr, A.cols());
      Ar.setFromTriplets(t.begin(), t.end());
    }
    Ar.makeCompressed();
    keep_ = keep;
    if (mr <= A.cols()) {
      SpMat At = Ar.transpose();
      SpMat gram = Ar * At;
      Eigen::SimplicialLDLT<SpMat, Eigen::Lower> ldlt(gram);
      if (ldlt.info() == Eigen::Success) {
        const VectorXd d = ldlt.vectorD();
        const double dmax = d.maxCoeff();
        if (std::isfinite(dmax) && d.minCoeff() > 1e-10 * dmax) {
          return;  // certified full row rank
        }
      }
    }
    if (Ar.nonZeros() > qr_max_nnz) return;  // too big to decide; assume ok
    A_ = Ar;
    qr_.compute(A_);
    have_qr_ = qr_.info() == Eigen::Success;
    rank_deficient_ = have_qr_ && qr_.rank() < mr;
  }
  bool rank_deficient() const { return rank_deficient_; }
  // Lower bound on the reachable ||A x - b||_inf-like residual: the largest
  // |b_i| over all-zero rows, and the RMS least-squares residual of the
  // remaining rows when they are rank deficient.
  double bound(const VectorXd& b) const {
    double lb = 0.0;
    for (Eigen::Index i : zero_rows_) lb = std::max(lb, std::abs(b[i]));
    if (rank_deficient_ && have_qr_ && !keep_.empty()) {
      VectorXd br(static_cast<Eigen::Index>(keep_.size()));
      for (size_t r = 0; r < keep_.size(); ++r) br[static_cast<Eigen::Index>(r)] = b[keep_[r]];
      const VectorXd xls = qr_.solve(br);
      lb = std::max(lb, (A_ * xls - br).norm() /
                            std::sqrt(static_cast<double>(br.size())));
    }
    return lb;
  }

 private:
  Eigen::Index m_ = 0;
  bool rank_deficient_ = false, have_qr_ = false;
  std::vector<Eigen::Index> zero_rows_, keep_;
  SpMat A_;
  Eigen::SparseQR<SpMat, Eigen::COLAMDOrdering<int>> qr_;
};

class Solver {
 public:
  Settings settings;

  enum class RowState : unsigned char { kInactive, kActive, kSaturated };

  void setup(const SpMat& Q, const VectorXd& q, const SpMat& A,
             const VectorXd& b, const SpMat& G, const VectorXd& h,
             const VectorXd& penalty) {
    n_ = q.size();
    m_ = b.size();
    p_ = h.size();
    Q_ = detail::Symmetrize(Q);
    q_ = q;
    A_ = A;
    A_.makeCompressed();
    b_ = b;
    G_ = G;
    G_.makeCompressed();
    h_ = h;
    penalty_ = penalty;
    N_ = n_ + m_ + p_;

    have_warm_ = false;
    explicit_warm_ = false;
    matrix_dirty_ = true;
    pattern_dirty_ = true;
    factored_ = false;
    rho_ = mu_eq_ = mu_in_ = 0.0;

    x_.resize(n_);
    y_.resize(m_);
    z_.resize(p_);
    xk_.resize(n_);
    yk_.resize(m_);
    zk_.resize(p_);

    ztilde_.resize(p_);
    t_.resize(p_);
    s2_.resize(p_);
    r_.resize(p_);
    dzs_.resize(p_);
    dzs_mask_.resize(p_);
    jump_res_prev_.resize(p_);
    jump_res_cur_.resize(p_);
    verr_.resize(n_);
    dyrhs_.resize(m_);
    rhs_.resize(N_);
    sol_w_.resize(N_);
    res_w_.resize(N_);
    cor_w_.resize(N_);
    dx_.resize(n_);
    dy_.resize(m_);
    dz_.resize(p_);
    Qdx_.resize(n_);
    Adx_.resize(m_);
    Gdx_.resize(p_);
    wQx_.resize(n_);
    wGtz_.resize(n_);
    wGtd_.resize(n_);
    wAty_.resize(n_);
    wAx_.resize(m_);
    wGx_.resize(p_);

    state_.assign(static_cast<size_t>(p_), RowState::kInactive);
    f_active_.assign(static_cast<size_t>(p_), false);
    wb_col_of_.assign(static_cast<size_t>(p_), -1);
    wb_rows_.clear();
    wb_sign_.clear();
    bp_.clear();
    bp_.reserve(static_cast<size_t>(2 * p_));
    updates_since_factor_ = 0;

    check_eq_A(A_);
    check_eq_b(b_);

    ruiz_ = settings.ruiz && p_ > 0;
    dxw_.resize(n_);
    dew_.resize(m_);
    diw_.resize(p_);
    dx_s_ = VectorXd::Ones(n_);
    de_s_ = VectorXd::Ones(m_);
    di_s_ = VectorXd::Ones(p_);
    c_s_ = 1.0;
    if (ruiz_) equilibrate();
    update_unscale_vectors();
    Gr_ = G_;  // row-major copy for row dots / scatters
  }

  void setup(const SpMat& Q, const VectorXd& q, const SpMat& A,
             const VectorXd& b, const SpMat& G, const VectorXd& h,
             double penalty) {
    setup(Q, q, A, b, G, h, VectorXd::Constant(h.size(), penalty));
  }
  void setup(const SpMat& Q, const VectorXd& q, const SpMat& G,
             const VectorXd& h, const VectorXd& penalty) {
    setup(Q, q, SpMat(0, q.size()), VectorXd(0), G, h, penalty);
  }
  void setup(const SpMat& Q, const VectorXd& q, const SpMat& G,
             const VectorXd& h, double penalty) {
    setup(Q, q, G, h, VectorXd::Constant(h.size(), penalty));
  }

  Eigen::Index n() const { return n_; }
  Eigen::Index m() const { return m_; }
  Eigen::Index p() const { return p_; }

  // Data updates between solves. A matrix with the same sparsity pattern
  // keeps the symbolic analysis (and the factorization is refreshed at the
  // next solve); a new pattern redoes the analysis.
  void set_Q(const SpMat& Q) {
    SpMat Qs = detail::Symmetrize(Q);
    if (ruiz_) {
      detail::ScaleRowsCols(Qs, dx_s_, dx_s_);
      Qs *= c_s_;
    }
    if (!detail::SamePattern(Qs, Q_)) pattern_dirty_ = true;
    Q_ = Qs;
    matrix_dirty_ = true;
  }
  void set_q(const VectorXd& q) {
    q_ = ruiz_ ? VectorXd(c_s_ * q.cwiseProduct(dx_s_)) : q;
  }
  void set_A(const SpMat& A) {
    check_eq_A(A);
    check_eq_b(ruiz_ ? VectorXd(b_.cwiseProduct(inv_de_)) : b_);
    SpMat As = A;
    As.makeCompressed();
    if (ruiz_) detail::ScaleRowsCols(As, de_s_, dx_s_);
    if (!detail::SamePattern(As, A_)) pattern_dirty_ = true;
    A_ = As;
    matrix_dirty_ = true;
  }
  void set_b(const VectorXd& b) {
    check_eq_b(b);
    b_ = ruiz_ ? VectorXd(b.cwiseProduct(de_s_)) : b;
  }
  void set_G(const SpMat& G) {
    SpMat Gs = G;
    Gs.makeCompressed();
    if (ruiz_) detail::ScaleRowsCols(Gs, di_s_, dx_s_);
    if (!detail::SamePattern(Gs, G_)) pattern_dirty_ = true;
    G_ = Gs;
    Gr_ = G_;
    matrix_dirty_ = true;
  }
  void set_h(const VectorXd& h) {
    h_ = ruiz_ ? VectorXd(h.cwiseProduct(di_s_)) : h;
  }
  void set_penalty(const VectorXd& penalty) {
    penalty_ = ruiz_ ? VectorXd(c_s_ * penalty.cwiseQuotient(di_s_)) : penalty;
  }

  // Recompute the Ruiz scaling from the user frame (pdal.hpp explains why
  // not incrementally) and remap the warm-start iterates.
  void reequilibrate() {
    if (!ruiz_) return;
    if (scaling_pass(dxw_, dew_, diw_) <= settings.ruiz_tol) return;
    const VectorXd dx0 = dx_s_, de0 = de_s_, di0 = di_s_;
    const double c0 = c_s_;
    const VectorXd ix = dx0.cwiseInverse(), ie = de0.cwiseInverse(),
                   ii = di0.cwiseInverse();
    detail::ScaleRowsCols(Q_, ix, ix);
    Q_ *= 1.0 / c0;
    q_ = q_.cwiseProduct(ix) / c0;
    if (m_ > 0) {
      detail::ScaleRowsCols(A_, ie, ix);
      b_ = b_.cwiseProduct(ie);
    }
    detail::ScaleRowsCols(G_, ii, ix);
    h_ = h_.cwiseProduct(ii);
    penalty_ = penalty_.cwiseProduct(di0) / c0;
    dx_s_.setOnes();
    de_s_.setOnes();
    di_s_.setOnes();
    c_s_ = 1.0;
    equilibrate();
    const VectorXd dx = dx_s_.cwiseQuotient(dx0);
    const VectorXd de = de_s_.cwiseQuotient(de0);
    const VectorXd di = di_s_.cwiseQuotient(di0);
    const double gamma = c_s_ / c0;
    const VectorXd yf = gamma * de.cwiseInverse();
    const VectorXd zf = gamma * di.cwiseInverse();
    x_ = x_.cwiseQuotient(dx);
    xk_ = xk_.cwiseQuotient(dx);
    if (m_ > 0) {
      y_ = y_.cwiseProduct(yf);
      yk_ = yk_.cwiseProduct(yf);
    }
    z_ = z_.cwiseProduct(zf);
    zk_ = zk_.cwiseProduct(zf);
    update_unscale_vectors();
    Gr_ = G_;
    matrix_dirty_ = true;
  }

  double scaling_drift() const {
    if (!ruiz_) return 1.0;
    scaling_pass(dxw_, dew_, diw_);
    return ruiz_drift(diw_, ruiz_drift(dew_, ruiz_drift(dxw_)));
  }

  void set_warm_start(const VectorXd& x, const VectorXd& y,
                      const VectorXd& z, double rho = 0.0,
                      double mu_eq = 0.0, double mu_in = 0.0) {
    x_ = ruiz_ ? VectorXd(x.cwiseQuotient(dx_s_)) : x;
    if (m_ > 0) y_ = ruiz_ ? VectorXd(c_s_ * y.cwiseQuotient(de_s_)) : y;
    z_ = ruiz_ ? VectorXd(c_s_ * z.cwiseQuotient(di_s_)) : z;
    rho_ = rho > 0 ? rho : settings.rho;
    // mu <= 0: the settings' initial values, or (warm_keep_mu) the values
    // the previous solve of this solver ended at
    const bool keep = settings.warm_keep_mu && have_warm_ && !matrix_dirty_ &&
                      mu_eq_ > 0 && mu_in_ > 0 &&
                      mu_eq_ >= settings.warm_mu_eq_min &&
                      mu_in_ >= settings.warm_mu_in_min &&
                      (settings.warm_keep_mu_max_iters <= 0 ||
                       last_iters_ <= settings.warm_keep_mu_max_iters);
    mu_eq_ = mu_eq > 0 ? mu_eq : (keep ? mu_eq_ : settings.mu_eq_init);
    mu_in_ = mu_in > 0 ? mu_in : (keep ? mu_in_ : settings.mu_in_init);
    explicit_warm_ = true;
  }
  // AL penalties of the current / last solve
  double mu_eq() const { return mu_eq_; }
  double mu_in() const { return mu_in_; }

  const Solution& solution() const { return sol_; }
  int factorizations() const { return factor_count_; }
  int cold_resets() const { return cold_resets_; }
  double eq_infeasibility() const { return eq_infeas_lb_; }
  // Nonzeros of the cached L factor (0 before the first factorization).
  Eigen::Index factor_nnz() const { return nnz_L_; }
  // Newton solves that went through the Woodbury correction in the last
  // solve() (i.e. active-set changes absorbed without refactoring).
  int woodbury_solves() const { return wb_solve_count_; }

  const Solution& solve() {
    const bool explicit_ws = explicit_warm_;
    explicit_warm_ = false;
    factor_count_ = 0;
    iters_total_ = 0;
    outer_iters_ = 0;
    factor_retries_ = 0;
    cold_resets_ = 0;
    wb_solve_count_ = 0;

    if (ruiz_ && matrix_dirty_ && settings.ruiz_refresh_ratio > 0 &&
        scaling_drift() > settings.ruiz_refresh_ratio) {
      reequilibrate();
    }
    if (pattern_dirty_) analyze_pattern();

    if (settings.check_eq_consistency && settings.eps_rel <= 0 &&
        eq_infeas_lb_ > settings.eps_abs) {
      if (!have_warm_ && !explicit_ws) {
        rho_ = settings.rho;
        mu_eq_ = settings.mu_eq_init;
        mu_in_ = settings.mu_in_init;
        x_.setZero();
        y_.setZero();
        z_.setZero();
      }
      clamp_z();
      update_residuals();
      return finish(Status::kInfeasible);
    }

    if (p_ == 0) return solve_no_inequalities();

    bool warm_path = true;
    if (explicit_ws) {
      clamp_z();
    } else if (settings.warm_start && have_warm_) {
      // The penalties are kept only together with the factorization they
      // belong to: after a matrix update a refactorization is due anyway
      // and deep inherited penalties only cost inner iterations.
      if (!settings.warm_keep_mu || matrix_dirty_ ||
          mu_eq_ < settings.warm_mu_eq_min ||
          mu_in_ < settings.warm_mu_in_min ||
          (settings.warm_keep_mu_max_iters > 0 &&
           last_iters_ > settings.warm_keep_mu_max_iters)) {
        mu_eq_ = settings.mu_eq_init;
        mu_in_ = settings.mu_in_init;
      }
      rho_ = settings.rho;
      clamp_z();
    } else {
      warm_path = false;
      if (!cold_init()) {
        update_residuals();
        return finish(Status::kNumerics);
      }
    }

    eta_ext_ = eta_ext_init();
    eta_in_ = 1.0;

    update_residuals();
    if (converged()) return finish(Status::kSolved);

    const double eta_warm = 0.5 * primal_res_;
    if (warm_path && settings.bcl_warm_eta && eta_warm < 0.1 * eta_ext_) {
      eta_ext_ = eta_warm;
    }

    const bool track_jump_res =
        settings.bcl_split && settings.bcl_saturation_jump;
    if (track_jump_res) jump_res_prev_ = (r_ - t_).cwiseProduct(inv_di_);
    double gap_prev = duality_gap_;

    for (int oiter = 0; oiter < settings.max_outer_iter; ++oiter) {
      outer_iters_ = oiter + 1;
      const double pri_old = primal_res_;
      const double dua_old = dual_res_;

      xk_ = x_;
      if (m_ > 0) yk_ = y_;
      zk_ = z_;
      wGx_.noalias() = G_ * x_;
      ztilde_ = zk_ + (wGx_ - h_) / mu_in_;

      const int it0 = iters_total_;
      if (!inner_loop(eta_in_)) return finish(Status::kNumerics);

      update_residuals();
      if (settings.trace) {
        int na = 0, ns = 0;
        for (Eigen::Index i = 0; i < p_; ++i) {
          na += is_active(i);
          ns += state(i) == RowState::kSaturated;
        }
        std::fprintf(stderr,
                     "  bcl %3d: inner %3d fac %2d wb %2d | pri %.2e (in %.2e eq %.2e)"
                     " dua %.2e gap %.2e | eta_ext %.1e mu_in %.1e mu_eq %.1e"
                     " | act %d sat %d\n",
                     oiter, iters_total_ - it0, factor_count_, wb_solve_count_,
                     primal_res_, in_res_, eq_res_, dual_res_, duality_gap_,
                     eta_ext_, mu_in_, mu_eq_, na, ns);
      }
      if (converged()) return finish(Status::kSolved);

      const double pri_new = primal_res_;
      const double dua_new = dual_res_;
      if (track_jump_res) jump_res_cur_ = (r_ - t_).cwiseProduct(inv_di_);

      if (pri_new <= eta_ext_ || iters_total_ > settings.safe_guard) {
        eta_ext_ *= std::pow(mu_in_, settings.beta_bcl);
        eta_in_ = std::max(eta_in_ * mu_in_, eps_in_min());
        if (settings.bcl_release_jump && residuals_ok() &&
            settings.check_duality_gap && !gap_ok() &&
            gap_decay_too_slow(gap_prev)) {
          double mu_new = mu_in_ * settings.mu_update_factor;
          const double shallowest = release_jump_mu();
          if (shallowest > 0.0) {
            mu_new = std::min(mu_new, settings.mu_update_factor * shallowest);
          }
          shrink_mu(mu_new);
        }
      } else if (settings.bcl_split) {
        if (m_ > 0 && eq_res_ > eta_ext_) y_ = yk_;
        double mu_new = mu_in_ * settings.mu_update_factor;
        if (settings.bcl_saturation_jump && in_res_ > eta_ext_ &&
            pri_new > 0.8 * pri_old) {
          const double shallowest = saturation_jump_mu();
          if (shallowest > 0.0) mu_new = std::min(mu_new, shallowest);
        }
        shrink_mu(mu_new);
      } else {
        if (m_ > 0) y_ = yk_;
        z_ = zk_;
        set_mu(mu_in_ * settings.mu_update_factor,
               mu_eq_ * settings.mu_update_factor);
      }

      if (track_jump_res) jump_res_prev_.swap(jump_res_cur_);
      gap_prev = duality_gap_;

      if (pri_new >= pri_old && dua_new >= dua_old &&
          mu_in_ <= settings.cold_reset_threshold &&
          std::max(pri_new, dua_new) > settings.cold_reset_residual &&
          cold_resets_ < settings.cold_reset_limit) {
        mu_in_ = settings.cold_reset_mu;
        mu_eq_ = settings.cold_reset_mu;
        cold_resets_++;
      }
    }
    return finish(Status::kMaxIter);
  }

 private:
  // ---- equality certificate, Ruiz ----
  void check_eq_A(const SpMat& A) {
    eq_infeas_lb_ = 0.0;
    if (m_ == 0 || !settings.check_eq_consistency) return;
    eq_cert_.set_A(A, settings.eq_check_qr_max_nnz);
  }
  void check_eq_b(const VectorXd& b) {
    eq_infeas_lb_ = 0.0;
    if (m_ == 0 || !settings.check_eq_consistency) return;
    eq_infeas_lb_ = eq_cert_.bound(b);
  }

  double scaling_pass(VectorXd& dx, VectorXd& de, VectorXd& di) const {
    dx.setZero();
    for (Eigen::Index k = 0; k < n_; ++k) {
      double cm = 0.0;
      for (SpMat::InnerIterator it(Q_, k); it; ++it) {
        cm = std::max(cm, std::abs(it.value()));
      }
      dx[k] = cm;
    }
    de.setZero();
    di.setZero();
    if (m_ > 0) detail::FoldMaxAbs(A_, dx, de);
    detail::FoldMaxAbs(G_, dx, di);
    return std::max({ruiz_factors(dx), ruiz_factors(de), ruiz_factors(di)});
  }

  void equilibrate() {
    VectorXd &dx = dxw_, &de = dew_, &di = diw_;
    for (int iter = 0; iter < settings.ruiz_max_iter; ++iter) {
      if (scaling_pass(dx, de, di) <= settings.ruiz_tol) break;
      detail::ScaleRowsCols(Q_, dx, dx);
      q_ = q_.cwiseProduct(dx);
      if (m_ > 0) {
        detail::ScaleRowsCols(A_, de, dx);
        b_ = b_.cwiseProduct(de);
        de_s_ = de_s_.cwiseProduct(de);
      }
      detail::ScaleRowsCols(G_, di, dx);
      h_ = h_.cwiseProduct(di);
      penalty_ = penalty_.cwiseQuotient(di);
      dx_s_ = dx_s_.cwiseProduct(dx);
      di_s_ = di_s_.cwiseProduct(di);
      const double gamma = detail::CostGamma(Q_);
      Q_ *= gamma;
      q_ *= gamma;
      penalty_ *= gamma;
      c_s_ *= gamma;
    }
  }

  void update_unscale_vectors() {
    inv_cdx_ = (c_s_ * dx_s_).cwiseInverse();
    inv_de_ = de_s_.cwiseInverse();
    inv_di_ = di_s_.cwiseInverse();
    y_us_ = de_s_ / c_s_;
    z_us_ = di_s_ / c_s_;
  }

  // ---- KKT pattern and numeric refill ----

  // Assemble the upper triangle of the augmented matrix with every row
  // present, and record where each nonzero of (Q, A, G) and each diagonal
  // lands in its value array, so refills are O(nnz) index gathers.
  void analyze_pattern() {
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(static_cast<size_t>(Q_.nonZeros() + A_.nonZeros() +
                                     G_.nonZeros() + N_));
    for (Eigen::Index k = 0; k < n_; ++k) {
      for (SpMat::InnerIterator it(Q_, k); it; ++it) {
        if (it.row() <= k) trip.emplace_back(it.row(), k, 0.0);
      }
    }
    for (Eigen::Index i = 0; i < N_; ++i) trip.emplace_back(i, i, 0.0);
    for (Eigen::Index k = 0; k < A_.outerSize(); ++k) {
      for (SpMat::InnerIterator it(A_, k); it; ++it) {
        trip.emplace_back(k, n_ + it.row(), 0.0);  // (x_k, y_j)
      }
    }
    for (Eigen::Index k = 0; k < G_.outerSize(); ++k) {
      for (SpMat::InnerIterator it(G_, k); it; ++it) {
        trip.emplace_back(k, n_ + m_ + it.row(), 0.0);  // (x_k, z_i)
      }
    }
    kkt_.resize(N_, N_);
    kkt_.setFromTriplets(trip.begin(), trip.end());
    kkt_.makeCompressed();

    auto pos = [&](Eigen::Index row, Eigen::Index col) -> Eigen::Index {
      const int* inner = kkt_.innerIndexPtr();
      const Eigen::Index lo = kkt_.outerIndexPtr()[col];
      const Eigen::Index hi = kkt_.outerIndexPtr()[col + 1];
      const int* it = std::lower_bound(inner + lo, inner + hi,
                                       static_cast<int>(row));
      return it - inner;
    };
    qmap_.assign(static_cast<size_t>(Q_.nonZeros()), -1);
    Eigen::Index idx = 0;
    for (Eigen::Index k = 0; k < n_; ++k) {
      for (SpMat::InnerIterator it(Q_, k); it; ++it, ++idx) {
        if (it.row() <= k) qmap_[static_cast<size_t>(idx)] = pos(it.row(), k);
      }
    }
    amap_.resize(static_cast<size_t>(A_.nonZeros()));
    idx = 0;
    for (Eigen::Index k = 0; k < A_.outerSize(); ++k) {
      for (SpMat::InnerIterator it(A_, k); it; ++it, ++idx) {
        amap_[static_cast<size_t>(idx)] = pos(k, n_ + it.row());
      }
    }
    gmap_.resize(static_cast<size_t>(G_.nonZeros()));
    idx = 0;
    for (Eigen::Index k = 0; k < G_.outerSize(); ++k) {
      for (SpMat::InnerIterator it(G_, k); it; ++it, ++idx) {
        gmap_[static_cast<size_t>(idx)] = pos(k, n_ + m_ + it.row());
      }
    }
    diag_pos_.resize(static_cast<size_t>(N_));
    for (Eigen::Index i = 0; i < N_; ++i) {
      diag_pos_[static_cast<size_t>(i)] = pos(i, i);
    }
    ldlt_.analyzePattern(kkt_);
    pattern_dirty_ = false;
    factored_ = false;
    nnz_L_ = 0;
  }

  void refill_kkt() {
    double* v = kkt_.valuePtr();
    std::fill(v, v + kkt_.nonZeros(), 0.0);
    const double* qv = Q_.valuePtr();
    for (size_t k = 0; k < qmap_.size(); ++k) {
      if (qmap_[k] >= 0) v[qmap_[k]] += qv[k];
    }
    for (Eigen::Index i = 0; i < n_; ++i) {
      v[diag_pos_[static_cast<size_t>(i)]] += rho_;
    }
    const double* av = A_.valuePtr();
    for (size_t k = 0; k < amap_.size(); ++k) v[amap_[k]] = av[k];
    for (Eigen::Index j = 0; j < m_; ++j) {
      v[diag_pos_[static_cast<size_t>(n_ + j)]] = -mu_eq_;
    }
    for (Eigen::Index i = 0; i < p_; ++i) {
      v[diag_pos_[static_cast<size_t>(n_ + m_ + i)]] =
          is_active(i) ? -mu_in_ : -1.0;
    }
    const double* gv = G_.valuePtr();
    const int* gi = G_.innerIndexPtr();
    for (size_t k = 0; k < gmap_.size(); ++k) {
      v[gmap_[k]] = is_active(gi[k]) ? gv[k] : 0.0;
    }
  }

  bool factor_kkt() {
    refill_kkt();
    ldlt_.factorize(kkt_);
    ++factor_count_;
    if (ldlt_.info() != Eigen::Success) return false;
    const VectorXd& d = ldlt_.vectorD();
    if (!d.allFinite()) return false;
    for (Eigen::Index i = 0; i < d.size(); ++i) {
      if (d[i] == 0.0) return false;
    }
    nnz_L_ = ldlt_.matrixL().nestedExpression().nonZeros();
    return true;
  }

  Eigen::Index max_flips() const {
    if (settings.incremental_update_max_flips > 0) {
      return settings.incremental_update_max_flips;
    }
    // Auto: the mean column count of L. |F| cached solves cost about
    // |F| * nnz(L); a refactorization about sum of squared column counts.
    const Eigen::Index mean_col = N_ > 0 ? nnz_L_ / N_ : 0;
    return std::max<Eigen::Index>(8, std::min<Eigen::Index>(mean_col, 512));
  }

  // Cached factor of the current (rho, mu, data, active set), or the
  // Woodbury correction for the rows that flipped since it was computed.
  bool ensure_factor() {
    const bool data_changed = !factored_ || matrix_dirty_ || f_rho_ != rho_ ||
                              f_mu_eq_ != mu_eq_ || f_mu_in_ != mu_in_;
    if (!data_changed) {
      wb_add_.clear();
      wb_drop_.clear();
      for (Eigen::Index i = 0; i < p_; ++i) {
        const bool flipped = is_active(i) != f_active(i);
        const int col = wb_col_of_[static_cast<size_t>(i)];
        if (flipped && col < 0) wb_add_.push_back(i);
        if (!flipped && col >= 0) wb_drop_.push_back(i);
      }
      if (wb_add_.empty() && wb_drop_.empty()) return true;
      const Eigen::Index n_after =
          static_cast<Eigen::Index>(wb_rows_.size() + wb_add_.size() -
                                    wb_drop_.size());
      if (settings.incremental_updates && n_after <= max_flips() &&
          updates_since_factor_ + static_cast<int>(wb_add_.size()) <=
              settings.incremental_update_budget) {
        if (wb_update()) return true;
      }
    }
    while (!factor_kkt()) {
      if (factor_retries_ < settings.max_factor_retries) {
        rho_ *= 100;
        factor_retries_++;
      } else {
        return false;
      }
    }
    factor_retries_ = 0;
    factored_ = true;
    matrix_dirty_ = false;
    updates_since_factor_ = 0;
    f_rho_ = rho_;
    f_mu_eq_ = mu_eq_;
    f_mu_in_ = mu_in_;
    for (Eigen::Index i = 0; i < p_; ++i) set_f_active(i, is_active(i));
    wb_clear();
    return true;
  }

  // ---- Woodbury correction ----
  void wb_clear() {
    for (Eigen::Index i : wb_rows_) wb_col_of_[static_cast<size_t>(i)] = -1;
    wb_rows_.clear();
    wb_sign_.clear();
  }

  // u_i = [G_i'; 0; 0] as a dense N-vector.
  void scatter_row(Eigen::Index i, VectorXd& u) const {
    u.setZero();
    for (SpMatRow::InnerIterator it(Gr_, i); it; ++it) u[it.col()] = it.value();
  }
  double row_dot(Eigen::Index i, const VectorXd& v) const {
    double s = 0.0;
    for (SpMatRow::InnerIterator it(Gr_, i); it; ++it) {
      s += it.value() * v[it.col()];
    }
    return s;
  }

  bool wb_update() {
    // Drop rows that flipped back (swap with the last column).
    for (Eigen::Index i : wb_drop_) {
      const int col = wb_col_of_[static_cast<size_t>(i)];
      const int last = static_cast<int>(wb_rows_.size()) - 1;
      if (col != last) {
        wb_V_.col(col) = wb_V_.col(last);
        wb_rows_[static_cast<size_t>(col)] = wb_rows_[static_cast<size_t>(last)];
        wb_sign_[static_cast<size_t>(col)] = wb_sign_[static_cast<size_t>(last)];
        wb_col_of_[static_cast<size_t>(wb_rows_[static_cast<size_t>(col)])] =
            col;
      }
      wb_rows_.pop_back();
      wb_sign_.pop_back();
      wb_col_of_[static_cast<size_t>(i)] = -1;
    }
    const Eigen::Index cap =
        static_cast<Eigen::Index>(wb_rows_.size() + wb_add_.size());
    if (wb_V_.rows() != N_ || wb_V_.cols() < cap) {
      MatrixXd grown(N_, std::max<Eigen::Index>(cap, 2 * wb_V_.cols()));
      if (!wb_rows_.empty()) {
        grown.leftCols(static_cast<Eigen::Index>(wb_rows_.size())) =
            wb_V_.leftCols(static_cast<Eigen::Index>(wb_rows_.size()));
      }
      wb_V_.swap(grown);
    }
    for (Eigen::Index i : wb_add_) {
      scatter_row(i, rhs_);
      sol_w_ = ldlt_.solve(rhs_);
      if (!sol_w_.allFinite()) return false;
      const int col = static_cast<int>(wb_rows_.size());
      wb_V_.col(col) = sol_w_;
      wb_rows_.push_back(i);
      // Activating a row adds +G_i'G_i/mu_in to the x block; deactivating
      // one that is in the factor removes it.
      wb_sign_.push_back(is_active(i) ? 1.0 / mu_in_ : -1.0 / mu_in_);
      wb_col_of_[static_cast<size_t>(i)] = col;
      ++updates_since_factor_;
    }
    // C = S^-1 + U' K0^-1 U (small, dense, nonsingular by the determinant
    // lemma since both K0 and K0 + U S U' are quasidefinite).
    const Eigen::Index k = static_cast<Eigen::Index>(wb_rows_.size());
    if (k == 0) return true;  // every flipped row flipped back
    wb_C_.resize(k, k);
    for (Eigen::Index a = 0; a < k; ++a) {
      for (Eigen::Index c = 0; c < k; ++c) {
        wb_C_(a, c) = row_dot(wb_rows_[static_cast<size_t>(a)], wb_V_.col(c));
      }
      wb_C_(a, a) += 1.0 / wb_sign_[static_cast<size_t>(a)];
    }
    wb_lu_.compute(wb_C_);
    const double dmin = wb_lu_.matrixLU().diagonal().cwiseAbs().minCoeff();
    const double dmax = wb_lu_.matrixLU().diagonal().cwiseAbs().maxCoeff();
    return std::isfinite(dmax) && dmin > 1e-14 * dmax;
  }

  // Solve the current Newton matrix (K0 + U S U') with one pass: factor
  // solve plus the Woodbury correction.
  void kkt_solve(const VectorXd& rhs, VectorXd& sol) {
    sol = ldlt_.solve(rhs);
    const Eigen::Index k = static_cast<Eigen::Index>(wb_rows_.size());
    if (k == 0) return;
    wb_c_.resize(k);
    for (Eigen::Index a = 0; a < k; ++a) {
      wb_c_[a] = row_dot(wb_rows_[static_cast<size_t>(a)], sol);
    }
    wb_s_ = wb_lu_.solve(wb_c_);
    sol.noalias() -= wb_V_.leftCols(k) * wb_s_;
  }

  // out = (K0 + U S U') v
  void apply_kkt(const VectorXd& v, VectorXd& out) const {
    out.noalias() = kkt_.selfadjointView<Eigen::Upper>() * v;
    for (size_t a = 0; a < wb_rows_.size(); ++a) {
      const Eigen::Index i = wb_rows_[a];
      const double g = wb_sign_[a] * row_dot(i, v);
      for (SpMatRow::InnerIterator it(Gr_, i); it; ++it) {
        out[it.col()] += g * it.value();
      }
    }
  }

  bool linear_solve(const VectorXd& rhs, VectorXd& sol) {
    kkt_solve(rhs, sol);
    if (!sol.allFinite()) return false;
    if (!wb_rows_.empty()) ++wb_solve_count_;
    const double rhs_norm = rhs.lpNorm<Eigen::Infinity>();
    double prev = std::numeric_limits<double>::infinity();
    for (int it = 0; it < settings.refine_iters; ++it) {
      apply_kkt(sol, res_w_);
      res_w_ -= rhs;
      const double nrm = res_w_.lpNorm<Eigen::Infinity>();
      if (!(nrm > settings.refine_tol * (1.0 + rhs_norm)) || nrm > 0.5 * prev) {
        break;
      }
      prev = nrm;
      kkt_solve(res_w_, cor_w_);
      if (!cor_w_.allFinite()) break;
      sol -= cor_w_;
    }
    return true;
  }

  // ---- solve() entry paths ----

  // p == 0: the equality-only KKT system through the same factor
  // (rho and the equality diagonal delta as regularization) with refinement
  // against the UNregularized system so no proximal bias remains.
  const Solution& solve_no_inequalities() {
    rho_ = settings.rho;
    mu_eq_ = std::max(settings.mu_min_eq, 1e-9);
    mu_in_ = settings.mu_in_init;
    if (!ensure_factor()) {
      x_.setZero();
      y_.setZero();
      update_residuals();
      return finish(Status::kNumerics);
    }
    rhs_.head(n_) = -q_;
    rhs_.segment(n_, m_) = b_;
    sol_w_ = ldlt_.solve(rhs_);
    double prev = std::numeric_limits<double>::infinity();
    for (int it = 0; it < 20 && sol_w_.allFinite(); ++it) {
      // Unregularized residual: [Q x + q + A'y; A x - b]
      res_w_.head(n_).noalias() = Q_ * sol_w_.head(n_);
      res_w_.head(n_) += q_;
      if (m_ > 0) {
        res_w_.head(n_).noalias() += A_.transpose() * sol_w_.segment(n_, m_);
        res_w_.segment(n_, m_).noalias() = A_ * sol_w_.head(n_);
        res_w_.segment(n_, m_) -= b_;
      }
      const double nrm = res_w_.lpNorm<Eigen::Infinity>();
      const double scale =
          std::max({1.0, q_.lpNorm<Eigen::Infinity>(),
                    m_ > 0 ? b_.lpNorm<Eigen::Infinity>() : 0.0});
      if (!std::isfinite(nrm) || nrm <= 1e-14 * scale || nrm > 0.5 * prev) {
        break;
      }
      prev = nrm;
      cor_w_ = ldlt_.solve(res_w_);
      sol_w_ -= cor_w_;
    }
    if (!sol_w_.allFinite()) {
      x_.setZero();
      y_.setZero();
      update_residuals();
      return finish(Status::kNumerics);
    }
    x_ = sol_w_.head(n_);
    y_ = sol_w_.segment(n_, m_);
    update_residuals();
    return finish(converged() ? Status::kSolved : Status::kNumerics);
  }

  bool cold_init() {
    rho_ = settings.rho;
    mu_eq_ = settings.mu_eq_init;
    mu_in_ = settings.mu_in_init;
    x_.setZero();
    y_.setZero();
    z_.setZero();
    xk_.setZero();
    yk_.setZero();
    zk_.setZero();
    std::fill(state_.begin(), state_.end(), RowState::kInactive);
    if (!ensure_factor()) return false;
    rhs_.head(n_) = -q_;
    rhs_.segment(n_, m_) = b_;
    rhs_.tail(p_).setZero();
    if (!linear_solve(rhs_, sol_w_)) return false;
    x_ = sol_w_.head(n_);
    if (m_ > 0) y_ = sol_w_.segment(n_, m_);
    return x_.allFinite() && (m_ == 0 || y_.allFinite());
  }

  // ---- BCL helpers (pdal.hpp) ----
  double eta_ext_init() const { return std::pow(0.1, settings.alpha_bcl); }
  double eps_in_min() const { return std::min(settings.eps_abs, 1e-9); }

  void set_mu(double mu_in_new, double mu_eq_new) {
    mu_in_ = std::max(mu_in_new, settings.mu_min_in);
    mu_eq_ = std::max(mu_eq_new, settings.mu_min_eq);
    eta_ext_ = eta_ext_init() * std::pow(mu_in_, settings.alpha_bcl);
    eta_in_ = std::max(mu_in_, eps_in_min());
    update_residuals();
  }
  void shrink_mu(double mu_new) { set_mu(mu_new, mu_eq_ * (mu_new / mu_in_)); }

  double release_jump_mu() const {
    double shallowest = 0.0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      if (r_[i] < 0.0 && z_[i] > 0.0) {
        shallowest = std::max(shallowest, -r_[i] / z_[i]);
      }
    }
    return shallowest;
  }
  double saturation_jump_mu() const {
    double shallowest = 0.0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      const double res_i = jump_res_cur_[i];
      if (res_i <= settings.eps_abs || res_i <= 0.8 * jump_res_prev_[i]) {
        continue;
      }
      const double den = penalty_[i] - z_[i];
      if (!(den > 0.0) || r_[i] <= 0.0) continue;
      shallowest = std::max(shallowest, r_[i] / den);
    }
    return shallowest;
  }
  bool gap_decay_too_slow(double gap_prev) const {
    if (duality_gap_ >= gap_prev || !(gap_prev > 0.0)) return true;
    const double decay =
        std::pow(duality_gap_ / gap_prev, settings.bcl_release_jump_horizon);
    return duality_gap_ * decay >= settings.eps_duality_gap_abs &&
           duality_gap_rel_ * decay >= settings.eps_duality_gap_rel;
  }

  // ---- inner semismooth Newton ----
  bool inner_loop(double eps_int) {
    for (int it = 0; it < settings.max_iter_in; ++it) {
      const double err = compute_inner_terms();
      if (err <= eps_int && it > 0) return true;

      for (Eigen::Index i = 0; i < p_; ++i) {
        if (ztilde_[i] >= penalty_[i]) {
          state(i) = RowState::kSaturated;
        } else if (ztilde_[i] >= 0.0) {
          state(i) = RowState::kActive;
        } else {
          state(i) = RowState::kInactive;
        }
      }
      if (!ensure_factor()) return false;

      // Dual shift toward each row's target; rows outside the factored
      // active set (or flipped since) carry their shift on the x block.
      for (Eigen::Index i = 0; i < p_; ++i) {
        switch (state(i)) {
          case RowState::kActive: dzs_[i] = ztilde_[i] - z_[i]; break;
          case RowState::kSaturated: dzs_[i] = penalty_[i] - z_[i]; break;
          case RowState::kInactive: dzs_[i] = -z_[i]; break;
        }
        const bool in_factor = f_active(i), active = is_active(i);
        if (in_factor && active) {
          dzs_mask_[i] = 0.0;
          rhs_[n_ + m_ + i] = -mu_in_ * dzs_[i];
        } else if (in_factor != active) {
          dzs_mask_[i] = dzs_[i];
          rhs_[n_ + m_ + i] = 0.0;
        } else {
          dzs_mask_[i] = dzs_[i];
          rhs_[n_ + m_ + i] = -dzs_[i];
        }
      }
      wGtd_.noalias() = G_.transpose() * dzs_mask_;
      rhs_.head(n_) = -verr_ - wGtd_;
      if (m_ > 0) rhs_.segment(n_, m_) = -dyrhs_;
      if (!linear_solve(rhs_, sol_w_)) return false;
      dx_ = sol_w_.head(n_);
      if (m_ > 0) dy_ = sol_w_.segment(n_, m_);
      Gdx_.noalias() = G_ * dx_;
      Qdx_.noalias() = Q_ * dx_;
      if (m_ > 0) Adx_.noalias() = A_ * dx_;
      for (Eigen::Index i = 0; i < p_; ++i) {
        if (!is_active(i)) {
          dz_[i] = dzs_[i];
        } else if (f_active(i)) {
          dz_[i] = sol_w_[n_ + m_ + i];
        } else {
          dz_[i] = Gdx_[i] / mu_in_ + dzs_[i];
        }
      }

      iters_total_++;
      const double alpha = line_search();

      double dwmax = dx_.lpNorm<Eigen::Infinity>();
      if (m_ > 0) dwmax = std::max(dwmax, dy_.lpNorm<Eigen::Infinity>());
      dwmax = std::max(dwmax, dz_.lpNorm<Eigen::Infinity>());
      if (alpha * dwmax < 1e-11 && it > 0) return true;

      x_ += alpha * dx_;
      if (m_ > 0) y_ += alpha * dy_;
      z_ += alpha * dz_;
      clamp_z();
      ztilde_ += (alpha / mu_in_) * Gdx_;
      if (alpha == 0.0) return true;
    }
    return true;
  }

  double compute_inner_terms() {
    wQx_.noalias() = Q_ * x_;
    wGtz_.noalias() = G_.transpose() * z_;
    verr_ = wQx_ + q_ + wGtz_ + rho_ * (x_ - xk_);
    double err = 0.0;
    if (m_ > 0) {
      wAty_.noalias() = A_.transpose() * y_;
      verr_ += wAty_;
      wAx_.noalias() = A_ * x_;
      dyrhs_ = wAx_ - b_ + mu_eq_ * (yk_ - y_);
      err = dyrhs_.lpNorm<Eigen::Infinity>();
    }
    err = std::max(err, verr_.lpNorm<Eigen::Infinity>());
    double inerr = 0.0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      const double zh = std::min(std::max(ztilde_[i], 0.0), penalty_[i]);
      inerr = std::max(inerr, std::abs(zh - z_[i]));
    }
    return std::max(err, mu_in_ * inerr);
  }

  double line_search() {
    double a = dx_.dot(Qdx_) + rho_ * dx_.squaredNorm();
    double b = dx_.dot(wQx_) + dx_.dot(q_) + rho_ * dx_.dot(x_ - xk_);
    if (m_ > 0) {
      a += (Adx_.squaredNorm() + dyrhs_.squaredNorm()) / mu_eq_;
      b += Adx_.dot(dyrhs_) / mu_eq_ + Adx_.dot(y_) -
           dyrhs_.squaredNorm() / mu_eq_;
    }
    const auto grad_at = [&](double alpha) {
      double g = b + a * alpha;
      for (Eigen::Index i = 0; i < p_; ++i) {
        const double c = Gdx_[i];
        const double zta = ztilde_[i] + alpha * c / mu_in_;
        const double za = z_[i] + alpha * dz_[i];
        if (zta >= penalty_[i]) {
          g += penalty_[i] * c - mu_in_ * (penalty_[i] - za) * dz_[i];
        } else if (zta >= 0.0) {
          g += zta * c + (zta - za) * (c - mu_in_ * dz_[i]);
        } else {
          g += mu_in_ * za * dz_[i];
        }
      }
      return g;
    };
    bp_.clear();
    for (Eigen::Index i = 0; i < p_; ++i) {
      const double c = Gdx_[i];
      if (c == 0.0) continue;
      const double a1 = -mu_in_ * ztilde_[i] / c;
      if (a1 > 0.0 && std::isfinite(a1)) bp_.push_back(a1);
      const double a2 = mu_in_ * (penalty_[i] - ztilde_[i]) / c;
      if (a2 > 0.0 && std::isfinite(a2)) bp_.push_back(a2);
    }
    std::sort(bp_.begin(), bp_.end());
    double alpha_prev = 0.0;
    double g_prev = grad_at(0.0);
    if (g_prev >= 0.0) return 0.0;
    for (const double t : bp_) {
      const double gt = grad_at(t);
      if (gt >= 0.0) {
        return alpha_prev + (-g_prev) * (t - alpha_prev) / (gt - g_prev);
      }
      alpha_prev = t;
      g_prev = gt;
    }
    const double g2 = grad_at(alpha_prev + 1.0);
    const double slope = g2 - g_prev;
    if (slope <= 0.0) return alpha_prev + 1.0;
    return alpha_prev + (-g_prev) / slope;
  }

  // ---- residuals, certificate ----
  void clamp_z() { z_ = z_.cwiseMax(0.0).cwiseMin(penalty_); }
  double elastic_slack(Eigen::Index i, double r) const {
    return std::max(r + mu_in_ * (z_[i] - penalty_[i]), 0.0);
  }
  template <class V>
  static double inf_us(const V& v, const VectorXd& s) {
    return v.size() > 0 ? v.cwiseAbs().cwiseProduct(s).maxCoeff() : 0.0;
  }

  void update_residuals() {
    wGx_.noalias() = G_ * x_;
    r_ = wGx_ - h_;
    wQx_.noalias() = Q_ * x_;
    wGtz_.noalias() = G_.transpose() * z_;
    verr_ = wQx_ + q_ + wGtz_;
    double aty_norm = 0.0;
    if (m_ > 0) {
      wAty_.noalias() = A_.transpose() * y_;
      verr_ += wAty_;
      aty_norm = inf_us(wAty_, inv_cdx_);
    }
    dual_res_ = inf_us(verr_, inv_cdx_);
    const double dual_rel_norm =
        std::max({inf_us(wQx_, inv_cdx_), inf_us(q_, inv_cdx_),
                  inf_us(wGtz_, inv_cdx_), aty_norm,
                  inf_us(penalty_, z_us_)});
    dual_res_rel_ = dual_res_ / std::max(1.0, dual_rel_norm);

    double in_res = 0.0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      t_[i] = elastic_slack(i, r_[i]);
      s2_[i] = std::max(t_[i] - r_[i], 0.0);
      in_res = std::max(in_res, (r_[i] - t_[i]) * inv_di_[i]);
    }
    in_res = std::max(in_res, 0.0);
    double primal_rel_norm =
        std::max({inf_us(wGx_ - t_, inv_di_), inf_us(h_, inv_di_),
                  inf_us(s2_, inv_di_), inf_us(t_, inv_di_)});
    double eq_res = 0.0;
    if (m_ > 0) {
      wAx_.noalias() = A_ * x_;
      eq_res = inf_us(wAx_ - b_, inv_de_);
      primal_rel_norm = std::max(
          {primal_rel_norm, inf_us(wAx_, inv_de_), inf_us(b_, inv_de_)});
    }
    in_res_ = in_res;
    eq_res_ = eq_res;
    primal_res_ = std::max(in_res, eq_res);
    primal_res_rel_ = primal_res_ / std::max(1.0, primal_rel_norm);

    const double xQx = x_.dot(wQx_);
    primal_obj_ = (0.5 * xQx + q_.dot(x_) + penalty_.dot(t_)) / c_s_;
    double dual_obj = (-0.5 * xQx - h_.dot(z_)) / c_s_;
    double gap_rel_norm =
        std::max({std::abs(xQx), std::abs(q_.dot(x_)),
                  std::abs(penalty_.dot(t_)), std::abs(h_.dot(z_))}) /
        c_s_;
    if (m_ > 0) {
      const double by = b_.dot(y_);
      dual_obj -= by / c_s_;
      gap_rel_norm = std::max(gap_rel_norm, std::abs(by) / c_s_);
    }
    duality_gap_ = std::abs(primal_obj_ - dual_obj);
    duality_gap_rel_ = duality_gap_ / std::max(1.0, gap_rel_norm);
  }

  bool residuals_ok() const {
    return (primal_res_ < settings.eps_abs ||
            primal_res_rel_ < settings.eps_rel) &&
           (dual_res_ < settings.eps_abs || dual_res_rel_ < settings.eps_rel);
  }
  bool gap_ok() const {
    return duality_gap_ < settings.eps_duality_gap_abs ||
           duality_gap_rel_ < settings.eps_duality_gap_rel;
  }
  bool converged() const {
    return residuals_ok() && (!settings.check_duality_gap || gap_ok());
  }

  const Solution& finish(Status status) {
    sol_.x = x_.cwiseProduct(dx_s_);
    sol_.t = t_.cwiseProduct(inv_di_);
    sol_.y = y_.cwiseProduct(y_us_);
    sol_.z = z_.cwiseProduct(z_us_);
    sol_.z_t = (penalty_ - z_).cwiseProduct(z_us_);
    sol_.status = status;
    sol_.converged = status == Status::kSolved ? 1 : 0;
    sol_.iters = iters_total_;
    sol_.outer_iters = outer_iters_;
    last_iters_ = iters_total_;
    sol_.n_active = sol_.n_saturated = 0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      if (state(i) == RowState::kActive) ++sol_.n_active;
      if (state(i) == RowState::kSaturated) ++sol_.n_saturated;
    }
    sol_.primal_obj = primal_obj_;
    sol_.primal_res = primal_res_;
    sol_.dual_res = dual_res_;
    sol_.duality_gap = duality_gap_;
    if (status != Status::kInfeasible) have_warm_ = status != Status::kNumerics;
    return sol_;
  }

  // ---- state ----
  Eigen::Index n_ = 0, m_ = 0, p_ = 0, N_ = 0;
  SpMat Q_, A_, G_;
  SpMatRow Gr_;
  VectorXd q_, b_, h_, penalty_;

  VectorXd x_, y_, z_, xk_, yk_, zk_;
  bool have_warm_ = false, explicit_warm_ = false;
  double rho_ = 0, mu_eq_ = 0, mu_in_ = 0;
  double eta_ext_ = 0, eta_in_ = 0;

  bool ruiz_ = false;
  double c_s_ = 1.0;
  VectorXd dx_s_, de_s_, di_s_, inv_cdx_, inv_de_, inv_di_, y_us_, z_us_;
  mutable VectorXd dxw_, dew_, diw_;

  SparseEqCertificate eq_cert_;
  double eq_infeas_lb_ = 0.0;

  // KKT pattern, factor, and its Woodbury correction
  SpMat kkt_;
  std::vector<Eigen::Index> qmap_, amap_, gmap_, diag_pos_;
  Eigen::SimplicialLDLT<SpMat, Eigen::Upper, Eigen::AMDOrdering<int>> ldlt_;
  bool pattern_dirty_ = true, matrix_dirty_ = true, factored_ = false;
  double f_rho_ = 0, f_mu_eq_ = 0, f_mu_in_ = 0;
  std::vector<bool> f_active_;
  Eigen::Index nnz_L_ = 0;
  int updates_since_factor_ = 0;
  int factor_retries_ = 0, factor_count_ = 0, iters_total_ = 0;
  int outer_iters_ = 0, cold_resets_ = 0, wb_solve_count_ = 0;
  int last_iters_ = 0;  // inner iterations of the previous solve()
  std::vector<Eigen::Index> wb_rows_, wb_add_, wb_drop_;
  std::vector<double> wb_sign_;
  std::vector<int> wb_col_of_;
  MatrixXd wb_V_, wb_C_;
  Eigen::PartialPivLU<MatrixXd> wb_lu_;
  VectorXd wb_c_, wb_s_;

  std::vector<RowState> state_;
  RowState& state(Eigen::Index i) { return state_[static_cast<size_t>(i)]; }
  RowState state(Eigen::Index i) const {
    return state_[static_cast<size_t>(i)];
  }
  bool is_active(Eigen::Index i) const {
    return state(i) == RowState::kActive;
  }
  bool f_active(Eigen::Index i) const {
    return f_active_[static_cast<size_t>(i)];
  }
  void set_f_active(Eigen::Index i, bool act) {
    f_active_[static_cast<size_t>(i)] = act;
  }

  double primal_res_ = 0, dual_res_ = 0, in_res_ = 0, eq_res_ = 0;
  double primal_res_rel_ = 0, dual_res_rel_ = 0;
  double primal_obj_ = 0, duality_gap_ = 0, duality_gap_rel_ = 0;

  VectorXd ztilde_, t_, s2_, r_, dzs_, dzs_mask_;
  VectorXd jump_res_prev_, jump_res_cur_;
  VectorXd verr_, dyrhs_, rhs_, sol_w_, res_w_, cor_w_;
  VectorXd dx_, dy_, dz_, Qdx_, Adx_, Gdx_;
  VectorXd wQx_, wGtz_, wGtd_, wAty_, wAx_, wGx_;
  std::vector<double> bp_;

  Solution sol_;
};

// One-shot convenience wrappers (cold start).
inline Solution Solve(const SpMat& Q, const VectorXd& q, const SpMat& A,
                      const VectorXd& b, const SpMat& G, const VectorXd& h,
                      const VectorXd& penalty,
                      const Settings& settings = Settings{}) {
  Solver s;
  s.settings = settings;
  s.setup(Q, q, A, b, G, h, penalty);
  return s.solve();
}
inline Solution Solve(const SpMat& Q, const VectorXd& q, const SpMat& A,
                      const VectorXd& b, const SpMat& G, const VectorXd& h,
                      double penalty, const Settings& settings = Settings{}) {
  return Solve(Q, q, A, b, G, h, VectorXd::Constant(h.size(), penalty),
               settings);
}
inline Solution Solve(const SpMat& Q, const VectorXd& q, const SpMat& G,
                      const VectorXd& h, const VectorXd& penalty,
                      const Settings& settings = Settings{}) {
  return Solve(Q, q, SpMat(0, q.size()), VectorXd(0), G, h, penalty,
               settings);
}
inline Solution Solve(const SpMat& Q, const VectorXd& q, const SpMat& G,
                      const VectorXd& h, double penalty,
                      const Settings& settings = Settings{}) {
  return Solve(Q, q, G, h, VectorXd::Constant(h.size(), penalty), settings);
}

}  // namespace elastiqp::sparse_pdal
