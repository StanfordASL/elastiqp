// ElastiQP-PDAL: a primal-dual augmented Lagrangian method for elastic QPs.
//
// This is the default backend of ElastiQP. It is based on ProxQP, adapted
// to the elastic QP form stated in elastiqp.hpp.
//
// Notes:
//
// Eliminating the elastic slacks t analytically turns the problem
// into  min 0.5 x'Qx + q'x + sum_i penalty_i [G_i x - h_i]_+
// s.t. Ax = b, and the AL treatment of each inequality row becomes the
// Moreau envelope of the weighted l1 penalty: quadratic with curvature
// 1/mu_in near the boundary, linear with slope penalty_i past the kink.
// On the dual side the multiplier estimate is projected onto [0, penalty]
// instead of the nonnegative ray (bounded multipliers <=> exact l1
// penalty), so ProxQP's active-set test gains a third state:
//
//   S_i = G_i x - h_i + mu_in * z_prev_i          (shifted row value)
//   inactive   S <= 0                  row out, dual snaps to 0
//   active     0 < S < mu_in*penalty   identical to hard ProxQP
//   saturated  S >= mu_in*penalty      row out, constant gradient
//                                      penalty_i*G_i, dual snaps to penalty_i
//
// Everything else mirrors proxsuite's dense backend (PDAL merit with nu=1,
// exact line search on the piecewise-quadratic merit, BCL outer loop with
// multiplier revert on bad steps, equality-constrained cold start), with the
// linear algebra condensed onto the n x n SPD system
//   K = Q + rho*I + (1/mu_eq) A^T A + (1/mu_in) G_act^T G_act
// (proxsuite's PrimalLDLT shape; identical code shape to elastiqp/ipm.hpp's
// factor()/kkt_solve()). The factorization is cached across Newton
// iterations AND across solve() calls: it is rebuilt only when Q/A/G, the
// active set, or (rho, mu) change -- so re-solves in a control loop where
// only vectors (q, b, h, penalty) drift and the active set is stable cost
// zero factorizations.
//
// Because the penalty is exact (no barrier), an unsaturated solution matches
// the hard-constrained QP exactly, and the reconstructed elastic certificate
// satisfies the t-block dual feasibility penalty - z_t - z_ineq = 0 exactly.
// Warm starting needs no slack/dual flooring: there is no interior to
// protect; the previous (x, y, z_ineq) is reused as-is, with z_ineq clamped to
// [0, penalty], along with mu/rho and the cached factorization.
//
// Omitted from proxsuite: GPDAL merit, incremental LDLT updates + iterative
// refinement, infeasibility detection, box specialization, nonconvex handling.
// Equalities hold to solver tolerance (~eps_abs).

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "elastiqp/types.hpp"  // Status, Solution

namespace elastiqp {

// Primal-dual augmented Lagrangian settings -- every knob this backend has.
// The termination block is field-for-field identical to elastiqp::IpmSettings
// (tests/test_pdal.cc static_asserts the defaults agree); everything below is
// specific to this method. Note that the convergence criteria and defaults
// are ElastiQP's, NOT proxsuite's looser 1e-5.
struct Settings {
  // Termination, on the unscaled elastic-KKT residuals.
  double eps_abs = 1e-8;
  double eps_rel = 1e-9;
  bool check_duality_gap = true;
  double eps_duality_gap_abs = 1e-8;
  double eps_duality_gap_rel = 1e-9;
  int max_factor_retries = 10;

  // Reuse the previous solve's (x, y, z_ineq) and cached factorization from
  // the second solve() on, resetting rho/mu to the values below.
  bool warm_start = true;

  // Iteration budget. The outer BCL loop runs max_outer_iter rounds; each
  // round runs up to max_iter_in semismooth Newton steps, and Solution::iters
  // reports their total (so it is not bounded by max_outer_iter).
  int max_outer_iter = 250;
  int max_iter_in = 1500;

  // Proximal regularization (primal) and AL penalties / dual prox (mu).
  // proxsuite defaults.
  double rho = 1e-6;
  double mu_eq_init = 1e-3;
  double mu_in_init = 1e-1;
  double mu_min_eq = 1e-9;
  double mu_min_in = 1e-8;
  double mu_update_factor = 0.1;

  // BCL outer-loop schedule (proxsuite defaults). eta_ext starts at
  // 0.1^alpha_bcl and tightens by mu_in^beta_bcl on good steps; a bad step
  // reverts the multipliers and shrinks mu by mu_update_factor.
  double alpha_bcl = 0.1;
  double beta_bcl = 0.9;
  // Cold restart of over-tightened mu (proxsuite's escape hatch for stuck
  // problems). It only fires while the residuals are still ABOVE
  // cold_reset_residual: proxsuite's thresholds are tuned for its 1e-5
  // accuracy target, and letting the reset fire in the 1e-8 endgame (where
  // the dual residual sits at machine noise and "not improving" is a coin
  // flip) creates a mu limit cycle that plateaus the primal residual.
  double cold_reset_mu = 1.0 / 1.1;
  double cold_reset_threshold = 1e-5;
  double cold_reset_residual = 1e-5;
  int safe_guard = 10000;  // total-Newton-iteration escape for BCL

  // Ruiz equilibration of the stacked [Q A' G'; A 0 0; G 0 0] structure
  // plus cost normalization. Read at setup() time (ignored when p == 0);
  // set_*() updates are rescaled with the setup()-time scaling (call
  // setup() again to re-equilibrate). The iterates live in the scaled
  // space, but termination and every reported residual stay on the
  // unscaled elastic KKT, and Solution is returned unscaled. The penalty
  // transforms as w_scaled = c * w / delta_row, which keeps the dual box
  // [0, w] and the saturation test consistent in scaled space.
  bool ruiz = false;
  int ruiz_max_iter = 10;
  double ruiz_tol = 1e-3;
};

// Reusable ProxQP-style elastic solver. setup() once, then alternate
// set_*() / solve(). All workspace is allocated in setup(); solve() is
// allocation-free on the warm path (up to std::sort of the preallocated
// line-search breakpoint buffer).
class Solver {
 public:
  Settings settings;

  void setup(const MatrixXd& Q, const VectorXd& q, const MatrixXd& A,
             const VectorXd& b, const MatrixXd& G, const VectorXd& h,
             const VectorXd& penalty) {
    n_ = q.size();
    m_ = b.size();
    p_ = h.size();
    Q_ = 0.5 * (Q + Q.transpose());
    q_ = q;
    A_ = A;
    b_ = b;
    G_ = G;
    h_ = h;
    penalty_ = penalty;
    have_warm_ = false;
    explicit_warm_ = false;
    matrix_dirty_ = true;
    factored_ = false;
    rho_ = 0.0;
    mu_eq_ = 0.0;
    mu_in_ = 0.0;

    x_.resize(n_);
    y_.resize(m_);
    z_.resize(p_);
    xk_.resize(n_);
    yk_.resize(m_);
    zk_.resize(p_);

    S_.resize(p_);
    zhat_.resize(p_);
    t_.resize(p_);
    s2_.resize(p_);
    r_.resize(p_);
    din_.resize(p_);
    pv_.resize(p_);
    tp_.resize(p_);
    verr_.resize(n_);
    dyrhs_.resize(m_);
    rhs_x_.resize(n_);
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

    state_.assign(static_cast<size_t>(p_), 0);
    f_active_.assign(static_cast<size_t>(p_), 0);
    bp_.clear();
    bp_.reserve(static_cast<size_t>(2 * p_));

    GS_.resize(p_, n_);
    K_.resize(n_, n_);
    llt_ = Eigen::LLT<MatrixXd, Eigen::Lower>(n_);

    ruiz_ = settings.ruiz && p_ > 0;
    dx_s_ = VectorXd::Ones(n_);
    de_s_ = VectorXd::Ones(m_);
    di_s_ = VectorXd::Ones(p_);
    c_s_ = 1.0;
    if (ruiz_) equilibrate();
    update_unscale_vectors();
    compute_AtA();
  }

  void setup(const MatrixXd& Q, const VectorXd& q, const MatrixXd& A,
             const VectorXd& b, const MatrixXd& G, const VectorXd& h,
             double penalty) {
    setup(Q, q, A, b, G, h, VectorXd::Constant(h.size(), penalty));
  }

  // Inequality-only overloads (no equality constraints).
  void setup(const MatrixXd& Q, const VectorXd& q, const MatrixXd& G,
             const VectorXd& h, const VectorXd& penalty) {
    setup(Q, q, MatrixXd(0, q.size()), VectorXd(0), G, h, penalty);
  }

  void setup(const MatrixXd& Q, const VectorXd& q, const MatrixXd& G,
             const VectorXd& h, double penalty) {
    setup(Q, q, G, h, VectorXd::Constant(h.size(), penalty));
  }

  // Problem dimensions
  Eigen::Index n() const { return n_; }
  Eigen::Index m() const { return m_; }
  Eigen::Index p() const { return p_; }

  // Data updates between solves (dimensions must not change). Vector
  // updates keep the cached factorization; matrix updates invalidate it.
  // Inputs are unscaled; with Ruiz active they are rescaled into the
  // setup()-time scaled frame on ingestion.
  void set_Q(const MatrixXd& Q) {
    Q_ = 0.5 * (Q + Q.transpose());
    if (ruiz_) Q_ = c_s_ * dx_s_.asDiagonal() * Q_ * dx_s_.asDiagonal();
    matrix_dirty_ = true;
  }
  void set_q(const VectorXd& q) {
    q_ = ruiz_ ? VectorXd(c_s_ * q.cwiseProduct(dx_s_)) : q;
  }
  void set_A(const MatrixXd& A) {
    A_ = ruiz_ ? MatrixXd(de_s_.asDiagonal() * A * dx_s_.asDiagonal()) : A;
    compute_AtA();
    matrix_dirty_ = true;
  }
  void set_b(const VectorXd& b) {
    b_ = ruiz_ ? VectorXd(b.cwiseProduct(de_s_)) : b;
  }
  void set_G(const MatrixXd& G) {
    G_ = ruiz_ ? MatrixXd(di_s_.asDiagonal() * G * dx_s_.asDiagonal()) : G;
    matrix_dirty_ = true;
  }
  void set_h(const VectorXd& h) {
    h_ = ruiz_ ? VectorXd(h.cwiseProduct(di_s_)) : h;
  }
  void set_penalty(const VectorXd& penalty) {
    penalty_ = ruiz_ ? VectorXd(c_s_ * penalty.cwiseQuotient(di_s_)) : penalty;
  }

  // Explicitly seed the next solve()'s starting iterate, replacing the
  // automatic warm start. (x, y, z_ineq) is the whole iterate this method
  // carries -- Solution's t, s_t, s_ineq and z_t are reconstructed from it
  // (see update_residuals), so there is nothing else to seed. Unlike the
  // interior-point solver there are no slacks to keep strictly positive, so
  // any point is valid (z_ineq is clamped into [0, penalty]). rho/mu > 0
  // override the proximal parameters for the next solve; when <= 0 the
  // settings defaults are used (NOT the converged values of the previous
  // solve, which sit at their floors and make a drifted problem's first AL
  // subproblem needlessly stiff). Takes effect once, for the next solve()
  // only, regardless of settings.warm_start.
  void set_warm_start(const VectorXd& x, const VectorXd& y,
                      const VectorXd& z_ineq, double rho = 0.0,
                      double mu_eq = 0.0, double mu_in = 0.0) {
    x_ = ruiz_ ? VectorXd(x.cwiseQuotient(dx_s_)) : x;
    if (m_ > 0) y_ = ruiz_ ? VectorXd(c_s_ * y.cwiseQuotient(de_s_)) : y;
    z_ = ruiz_ ? VectorXd(c_s_ * z_ineq.cwiseQuotient(di_s_)) : z_ineq;
    rho_ = rho > 0 ? rho : settings.rho;
    mu_eq_ = mu_eq > 0 ? mu_eq : settings.mu_eq_init;
    mu_in_ = mu_in > 0 ? mu_in : settings.mu_in_init;
    explicit_warm_ = true;
  }

  const Solution& solution() const { return sol_; }

  // Number of KKT factorizations performed by the last solve() (a warm
  // re-solve with an unchanged active set and unchanged Q/G/A/mu needs 0).
  int factorizations() const { return factor_count_; }

  const Solution& solve() {
    const bool explicit_ws = explicit_warm_;
    explicit_warm_ = false;
    factor_count_ = 0;
    iters_total_ = 0;
    factor_retries_ = 0;

    if (p_ == 0) {
      return solve_no_inequalities();
    }

    if (explicit_ws) {
      z_ = z_.cwiseMax(0.0).cwiseMin(penalty_);
    } else if (settings.warm_start && have_warm_) {
      // Keep (x, y, z) and the cached factorization from the previous
      // solve, but reset the AL penalties to their defaults (proxsuite's
      // warm-start path does the same): the converged mu sit at their
      // floors, and starting a drifted problem with a ~1e8-stiff AL term
      // makes the first subproblems thrash. The BCL loop re-tightens mu
      // cheaply from a near-optimal iterate. z is re-clamped in case the
      // penalty changed.
      mu_eq_ = settings.mu_eq_init;
      mu_in_ = settings.mu_in_init;
      rho_ = settings.rho;
      z_ = z_.cwiseMax(0.0).cwiseMin(penalty_);
    } else {
      if (!cold_init()) {
        update_residuals();
        return finish(Status::kNumerics);
      }
    }

    // BCL state (proxsuite: bcl_eta_ext_init = 0.1^alpha_bcl, eta_in = 1).
    const double eta_ext_init = std::pow(0.1, settings.alpha_bcl);
    const double eps_in_min = std::min(settings.eps_abs, 1e-9);
    double eta_ext = eta_ext_init;
    double eta_in = 1.0;

    update_residuals();
    if (converged()) return finish(Status::kSolved);

    // ------------- outer loop (PMM + BCL, proxsuite qp_solve) -------------
    for (int oiter = 0; oiter < settings.max_outer_iter; ++oiter) {
      const double pri_old = primal_res_;
      const double dua_old = dual_res_;

      // The PMM "multiplier update" is implicit: snapshot the prox centers
      // and let the inner primal-dual Newton move (x, y, z) jointly.
      xk_ = x_;
      if (m_ > 0) yk_ = y_;
      zk_ = z_;
      wGx_.noalias() = G_ * x_;
      S_ = wGx_ - h_ + mu_in_ * zk_;

      if (!inner_loop(eta_in)) {
        return finish(Status::kNumerics);
      }

      update_residuals();
      if (converged()) return finish(Status::kSolved);
      const double pri_new = primal_res_;
      const double dua_new = dual_res_;

      // BCL update (proxsuite bcl_update): the elastic primal feasibility
      // (equality residual + inequality violation beyond the reconstructed
      // slack) decides between a dual update and a mu shrink.
      if (pri_new <= eta_ext || iters_total_ > settings.safe_guard) {
        eta_ext *= std::pow(mu_in_, settings.beta_bcl);
        eta_in = std::max(eta_in * mu_in_, eps_in_min);
      } else {
        // Bad step: revert the multipliers (x is kept) and increase the
        // penalties. The factorization cache invalidates via the mu
        // fingerprint.
        if (m_ > 0) y_ = yk_;
        z_ = zk_;
        mu_in_ = std::max(mu_in_ * settings.mu_update_factor,
                          settings.mu_min_in);
        mu_eq_ = std::max(mu_eq_ * settings.mu_update_factor,
                          settings.mu_min_eq);
        eta_ext = eta_ext_init * std::pow(mu_in_, settings.alpha_bcl);
        eta_in = std::max(mu_in_, eps_in_min);
        update_residuals();
      }

      // Cold restart of stalled, over-tightened penalties (guarded so it
      // cannot fire in the endgame, see Settings::cold_reset_residual).
      if (pri_new >= pri_old && dua_new >= dua_old &&
          mu_in_ <= settings.cold_reset_threshold &&
          std::max(pri_new, dua_new) > settings.cold_reset_residual) {
        mu_in_ = settings.cold_reset_mu;
        mu_eq_ = settings.cold_reset_mu;
      }
    }

    return finish(Status::kMaxIter);
  }

 private:
  void compute_AtA() {
    if (m_ > 0) {
      AtA_.resize(n_, n_);
      AtA_.setZero();
      AtA_.selfadjointView<Eigen::Lower>().rankUpdate(A_.transpose());
    }
  }

  // PIQP's limit_scaling guard (dense/preconditioner.tpp): a norm below
  // 1e-4 is treated as 1, so the row/column is left unscaled -- a row at
  // the numerical noise floor would otherwise be amplified by 1/sqrt(nrm)
  // >= 100x per sweep while its penalty shrinks toward zero; a norm above
  // 1e4 is capped so one sweep's scale factor stays in [1e-2, 100]. The
  // trade-off (accepted by PIQP) is that legitimately tiny rows are not
  // equilibrated; they are at most 1e-4 and barely affect conditioning.
  static double limit_scaling(double nrm) {
    return nrm < 1e-4 ? 1.0 : std::min(nrm, 1e4);
  }

  // Ruiz sweeps on the stacked symmetric structure [Q A' G'; A 0 0; G 0 0]
  // (proxsuite ruiz.hpp, plus PIQP's limit_scaling clamp above), applied in
  // place to the stored data, with the per-sweep cost normalization
  // gamma = 1/max(1, mean |Q| column norm).
  void equilibrate() {
    VectorXd dx(n_), de(m_), di(p_);
    for (int iter = 0; iter < settings.ruiz_max_iter; ++iter) {
      double dev = 0.0;
      for (Eigen::Index k = 0; k < n_; ++k) {
        double nrm = Q_.col(k).cwiseAbs().maxCoeff();
        if (m_ > 0) nrm = std::max(nrm, A_.col(k).cwiseAbs().maxCoeff());
        nrm = std::max(nrm, G_.col(k).cwiseAbs().maxCoeff());
        dx[k] = 1.0 / std::sqrt(limit_scaling(nrm));
        dev = std::max(dev, std::abs(1.0 - dx[k]));
      }
      for (Eigen::Index i = 0; i < m_; ++i) {
        const double nrm = A_.row(i).cwiseAbs().maxCoeff();
        de[i] = 1.0 / std::sqrt(limit_scaling(nrm));
        dev = std::max(dev, std::abs(1.0 - de[i]));
      }
      for (Eigen::Index i = 0; i < p_; ++i) {
        const double nrm = G_.row(i).cwiseAbs().maxCoeff();
        di[i] = 1.0 / std::sqrt(limit_scaling(nrm));
        dev = std::max(dev, std::abs(1.0 - di[i]));
      }
      if (dev <= settings.ruiz_tol) break;
      Q_ = dx.asDiagonal() * Q_ * dx.asDiagonal();
      q_ = q_.cwiseProduct(dx);
      if (m_ > 0) {
        A_ = de.asDiagonal() * A_ * dx.asDiagonal();
        b_ = b_.cwiseProduct(de);
        de_s_ = de_s_.cwiseProduct(de);
      }
      G_ = di.asDiagonal() * G_ * dx.asDiagonal();
      h_ = h_.cwiseProduct(di);
      penalty_ = penalty_.cwiseQuotient(di);
      dx_s_ = dx_s_.cwiseProduct(dx);
      di_s_ = di_s_.cwiseProduct(di);
      double mean = 0.0;
      for (Eigen::Index k = 0; k < n_; ++k) {
        mean += Q_.col(k).cwiseAbs().maxCoeff();
      }
      mean /= static_cast<double>(n_);
      const double gamma = 1.0 / std::max(1.0, mean);
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

  // No inequality constraints: plain (in)equality-constrained QP; solve the
  // KKT system directly and report kSolved/kNumerics from its residuals
  // (identical to elastiqp::IpmSolver::solve_no_inequalities).
  const Solution& solve_no_inequalities() {
    if (m_ == 0) {
      x_ = Eigen::LDLT<MatrixXd>(Q_).solve(-q_);
    } else {
      MatrixXd Kf = MatrixXd::Zero(n_ + m_, n_ + m_);
      Kf.topLeftCorner(n_, n_) = Q_;
      Kf.topRightCorner(n_, m_) = A_.transpose();
      Kf.bottomLeftCorner(m_, n_) = A_;
      VectorXd rhs(n_ + m_);
      rhs.head(n_) = -q_;
      rhs.tail(m_) = b_;
      const VectorXd xy = Kf.colPivHouseholderQr().solve(rhs);
      x_ = xy.head(n_);
      y_ = xy.tail(m_);
    }

    wQx_.noalias() = Q_ * x_;
    verr_ = wQx_ + q_;
    double dual_rel_norm = std::max(wQx_.lpNorm<Eigen::Infinity>(),
                                    q_.lpNorm<Eigen::Infinity>());
    primal_res_ = 0.0;
    primal_res_rel_ = 0.0;
    double by = 0.0;
    if (m_ > 0) {
      wAty_.noalias() = A_.transpose() * y_;
      verr_ += wAty_;
      dual_rel_norm = std::max(dual_rel_norm, wAty_.lpNorm<Eigen::Infinity>());
      wAx_.noalias() = A_ * x_;
      primal_res_ = (wAx_ - b_).lpNorm<Eigen::Infinity>();
      primal_res_rel_ =
          primal_res_ / std::max(1.0, std::max(wAx_.lpNorm<Eigen::Infinity>(),
                                               b_.lpNorm<Eigen::Infinity>()));
      by = b_.dot(y_);
    }
    dual_res_ = verr_.lpNorm<Eigen::Infinity>();
    dual_res_rel_ = dual_res_ / std::max(1.0, dual_rel_norm);

    const double xQx = x_.dot(wQx_);
    primal_obj_ = 0.5 * xQx + q_.dot(x_);
    duality_gap_ = std::abs(primal_obj_ - (-0.5 * xQx - by));
    duality_gap_rel_ =
        duality_gap_ / std::max(1.0, std::max({std::abs(xQx),
                                               std::abs(q_.dot(x_)),
                                               std::abs(by)}));

    return finish(converged() ? Status::kSolved : Status::kNumerics);
  }

  // Cold start (proxsuite EQUALITY_CONSTRAINED_INITIAL_GUESS): reset the
  // proximal state and solve [Q+rho I, A'; A, -mu_eq I][x;y] = [-q; b] via
  // the condensed K with an empty active set; z = 0.
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

    std::fill(state_.begin(), state_.end(), static_cast<signed char>(0));
    if (!ensure_factor()) return false;

    rhs_x_ = -q_;
    if (m_ > 0) {
      rhs_x_.noalias() += (1.0 / mu_eq_) * (A_.transpose() * b_);
    }
    x_ = llt_.solve(rhs_x_);
    if (m_ > 0) {
      wAx_.noalias() = A_ * x_;
      y_ = (wAx_ - b_) / mu_eq_;
    }
    return std::isfinite(x_.sum()) && (m_ == 0 || std::isfinite(y_.sum()));
  }

  // ---- inner semismooth Newton on the PDAL merit (proxsuite
  // primal_dual_newton_semi_smooth). Returns false only on a factorization
  // disaster. Invariant on entry and throughout: S_ = Gx - h + mu_in*zk_.
  bool inner_loop(double eps_int) {
    for (int it = 0; it < settings.max_iter_in; ++it) {
      const double err = compute_inner_terms();
      // The mismatch ||zhat - z|| enters err weighted by mu_in, so at the
      // mu floor a subproblem can pass this test while the multipliers are
      // still far from their clamp targets. The outer loop only runs while
      // the true elastic KKT is unsatisfied, so never accept a subproblem
      // without taking at least one Newton step (which snaps the
      // multipliers) -- otherwise the outer loop can spin to kMaxIter with
      // an unchanged iterate.
      if (err <= eps_int && it > 0) return true;

      // Three-state row classification on the shifted value S (elastic
      // version of proxsuite's active-set test; ties mirror its >=).
      for (Eigen::Index i = 0; i < p_; ++i) {
        if (S_[i] >= mu_in_ * penalty_[i]) {
          state_[static_cast<size_t>(i)] = 2;  // saturated
        } else if (S_[i] >= 0.0) {
          state_[static_cast<size_t>(i)] = 1;  // active
        } else {
          state_[static_cast<size_t>(i)] = 0;  // inactive
        }
      }
      if (!ensure_factor()) return false;

      // Newton system, condensed onto K (see header comment). Non-active
      // rows leave the system: their dual snaps to its known target
      // (0 inactive, penalty saturated) and the target replaces z in the
      // dual residual on the right-hand side.
      for (Eigen::Index i = 0; i < p_; ++i) {
        switch (state_[static_cast<size_t>(i)]) {
          case 1:
            din_[i] = S_[i] - mu_in_ * z_[i];
            pv_[i] = 0.0;
            break;
          case 2:
            din_[i] = 0.0;
            pv_[i] = penalty_[i] - z_[i];
            break;
          default:
            din_[i] = 0.0;
            pv_[i] = -z_[i];
            break;
        }
      }
      tp_ = pv_ + din_ / mu_in_;
      wGtd_.noalias() = G_.transpose() * tp_;
      rhs_x_ = -verr_ - wGtd_;
      if (m_ > 0) {
        rhs_x_.noalias() -= (1.0 / mu_eq_) * (A_.transpose() * dyrhs_);
      }
      dx_ = llt_.solve(rhs_x_);
      if (!std::isfinite(dx_.sum())) return false;
      Gdx_.noalias() = G_ * dx_;
      Qdx_.noalias() = Q_ * dx_;
      if (m_ > 0) {
        Adx_.noalias() = A_ * dx_;
        dy_ = (Adx_ + dyrhs_) / mu_eq_;
      }
      for (Eigen::Index i = 0; i < p_; ++i) {
        dz_[i] = state_[static_cast<size_t>(i)] == 1
                     ? (Gdx_[i] + din_[i]) / mu_in_
                     : pv_[i];
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
      // The exact line search is unclamped (alpha can exceed 1, mirroring
      // proxsuite), so a snap direction dz = -z or w - z can overshoot the
      // dual box. Project back: the merit's z-terms are separable quadratics
      // with minimizers zhat in [0, w], so this never increases the merit,
      // and it preserves the solver invariant z in [0, penalty] that the
      // residuals, duality gap, and reported certificate all rely on.
      z_ = z_.cwiseMax(0.0).cwiseMin(penalty_);
      S_ += alpha * Gdx_;
      if (alpha == 0.0) return true;
    }
    return true;  // out of inner iterations; the outer loop adapts mu
  }

  // Inner stopping quantities (proxsuite compute_inner_loop_saddle_point,
  // with the nonnegative-ray projection replaced by the [0, penalty]
  // clamp). Fills the buffers the Newton step reuses.
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
      const double zh =
          std::min(std::max(S_[i] / mu_in_, 0.0), penalty_[i]);
      zhat_[i] = zh;
      inerr = std::max(inerr, std::abs(zh - z_[i]));
    }
    return std::max(err, mu_in_ * inerr);
  }

  // Exact line search on the piecewise-quadratic PDAL merit along
  // (dx, dy, dz) (proxsuite linesearch::primal_dual_ls). The derivative is
  // piecewise affine in alpha with breakpoints where a row's shifted value
  // S_i(alpha) crosses 0 or mu_in*penalty_i; scan the (nondecreasing in
  // expectation) derivative for its sign change and interpolate.
  double line_search() {
    // alpha-independent scalars of the smooth part. With
    // dy = (Adx + dyrhs)/mu_eq we have Adx - mu_eq*dy = -dyrhs, which
    // collapses the second (dual-coupling) equality term.
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
        const double Sa = S_[i] + alpha * c;
        const double za = z_[i] + alpha * dz_[i];
        const double mw = mu_in_ * penalty_[i];
        if (Sa >= mw) {  // saturated
          g += penalty_[i] * c - mu_in_ * (penalty_[i] - za) * dz_[i];
        } else if (Sa >= 0.0) {  // active
          g += (Sa / mu_in_) * c +
               (Sa / mu_in_ - za) * (c - mu_in_ * dz_[i]);
        } else {  // inactive
          g += mu_in_ * za * dz_[i];
        }
      }
      return g;
    };

    bp_.clear();
    for (Eigen::Index i = 0; i < p_; ++i) {
      const double c = Gdx_[i];
      if (c == 0.0) continue;
      const double a1 = -S_[i] / c;
      if (a1 > 0.0 && std::isfinite(a1)) bp_.push_back(a1);
      const double a2 = (mu_in_ * penalty_[i] - S_[i]) / c;
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
    // Affine tail beyond the last breakpoint.
    const double g2 = grad_at(alpha_prev + 1.0);
    const double slope = g2 - g_prev;
    if (slope <= 0.0) return alpha_prev + 1.0;  // pathological; bounded step
    return alpha_prev + (-g_prev) / slope;
  }

  // ---- factorization cache ----
  bool factor_kkt() {
    Eigen::Index na = 0;
    const double s = std::sqrt(1.0 / mu_in_);
    for (Eigen::Index i = 0; i < p_; ++i) {
      if (state_[static_cast<size_t>(i)] == 1) {
        GS_.row(na) = s * G_.row(i);
        ++na;
      }
    }
    K_.triangularView<Eigen::Lower>() = Q_;
    K_.diagonal().array() += rho_;
    if (m_ > 0) {
      K_.triangularView<Eigen::Lower>() += (1.0 / mu_eq_) * AtA_;
    }
    if (na > 0) {
      K_.selfadjointView<Eigen::Lower>().rankUpdate(
          GS_.topRows(na).transpose());
    }
    llt_.compute(K_);
    ++factor_count_;
    return llt_.info() == Eigen::Success &&
           std::isfinite(K_.diagonal().sum());
  }

  bool ensure_factor() {
    bool need = !factored_ || matrix_dirty_ || f_rho_ != rho_ ||
                f_mu_eq_ != mu_eq_ || f_mu_in_ != mu_in_;
    if (!need) {
      for (Eigen::Index i = 0; i < p_; ++i) {
        const bool act = state_[static_cast<size_t>(i)] == 1;
        if (act != (f_active_[static_cast<size_t>(i)] != 0)) {
          need = true;
          break;
        }
      }
    }
    if (!need) return true;

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
    f_rho_ = rho_;
    f_mu_eq_ = mu_eq_;
    f_mu_in_ = mu_in_;
    for (Eigen::Index i = 0; i < p_; ++i) {
      f_active_[static_cast<size_t>(i)] =
          state_[static_cast<size_t>(i)] == 1 ? 1 : 0;
    }
    return true;
  }

  // Unscaled residuals of the elastic QP at the reconstructed expanded
  // point (mirrors elastiqp::IpmSolver::update_residuals_nr):
  //   t = [Gx - h + mu_in (z - penalty)]_+   (argmin of the folded slack)
  //   z_ineq = z, z_t = penalty - z, s_t = t, s_ineq = [t - (Gx - h)]_+
  // The t-block dual residual penalty - z_t - z_ineq vanishes identically,
  // and z_t, z_ineq >= 0 exactly (z is kept in [0, penalty]).
  // All internal quantities are in the (possibly Ruiz-scaled) frame; every
  // norm below is unscaled componentwise, so the reported residuals and
  // the termination test are on the true elastic KKT regardless of
  // equilibration (with Ruiz off, all unscale vectors are ones).
  void update_residuals() {
    const auto inf_us = [](const VectorXd& v, const VectorXd& s) {
      return v.size() > 0 ? v.cwiseAbs().cwiseProduct(s).maxCoeff() : 0.0;
    };
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
      t_[i] = std::max(r_[i] + mu_in_ * (z_[i] - penalty_[i]), 0.0);
      s2_[i] = std::max(t_[i] - r_[i], 0.0);
      in_res = std::max(in_res, (r_[i] - t_[i]) * inv_di_[i]);
    }
    in_res = std::max(in_res, 0.0);
    zhat_ = wGx_ - t_;  // reuse as scratch: unscaled ||Gx - t|| below
    double primal_rel_norm = std::max(
        {inf_us(zhat_, inv_di_), inf_us(h_, inv_di_), inf_us(s2_, inv_di_),
         inf_us(t_, inv_di_)});
    double eq_res = 0.0;
    if (m_ > 0) {
      wAx_.noalias() = A_ * x_;
      dyrhs_ = wAx_ - b_;  // reuse as scratch
      eq_res = inf_us(dyrhs_, inv_de_);
      primal_rel_norm = std::max(
          {primal_rel_norm, inf_us(wAx_, inv_de_), inf_us(b_, inv_de_)});
    }
    primal_res_ = std::max(in_res, eq_res);
    primal_res_rel_ = primal_res_ / std::max(1.0, primal_rel_norm);

    // Objectives: every scaled term is c_s_ times its unscaled value.
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

  bool converged() const {
    return (primal_res_ < settings.eps_abs ||
            primal_res_rel_ < settings.eps_rel) &&
           (dual_res_ < settings.eps_abs ||
            dual_res_rel_ < settings.eps_rel) &&
           (!settings.check_duality_gap ||
            duality_gap_ < settings.eps_duality_gap_abs ||
            duality_gap_rel_ < settings.eps_duality_gap_rel);
  }

  const Solution& finish(Status status) {
    // Unscale into the user's frame (identity when Ruiz is off).
    sol_.x = x_.cwiseProduct(dx_s_);
    sol_.t = t_.cwiseProduct(inv_di_);
    sol_.y = y_.cwiseProduct(y_us_);
    sol_.s_t = sol_.t;
    sol_.s_ineq = s2_.cwiseProduct(inv_di_);
    sol_.z_t = (penalty_ - z_).cwiseProduct(z_us_);
    sol_.z_ineq = z_.cwiseProduct(z_us_);
    sol_.status = status;
    sol_.converged = status == Status::kSolved ? 1 : 0;
    sol_.iters = iters_total_;
    sol_.primal_obj = primal_obj_;
    sol_.primal_res = primal_res_;
    sol_.dual_res = dual_res_;
    sol_.duality_gap = duality_gap_;
    have_warm_ = status != Status::kNumerics;
    return sol_;
  }

  // Problem data
  Eigen::Index n_ = 0, m_ = 0, p_ = 0;
  MatrixXd Q_, A_, G_;
  VectorXd q_, b_, h_, penalty_;
  MatrixXd AtA_;  // cached A^T A (lower triangle valid), only when m_ > 0

  // Iterates and prox centers (persist across solves for warm starting)
  VectorXd x_, y_, z_;
  VectorXd xk_, yk_, zk_;
  bool have_warm_ = false;
  bool explicit_warm_ = false;

  // Proximal / AL state (persists across solves)
  double rho_ = 0, mu_eq_ = 0, mu_in_ = 0;

  // Ruiz scaling state (identity when ruiz_ is false)
  bool ruiz_ = false;
  double c_s_ = 1.0;
  VectorXd dx_s_, de_s_, di_s_;             // cumulative scale factors
  VectorXd inv_cdx_, inv_de_, inv_di_;      // residual unscaling
  VectorXd y_us_, z_us_;                    // dual unscaling (de/c, di/c)

  // Factorization cache
  bool matrix_dirty_ = true, factored_ = false;
  double f_rho_ = 0, f_mu_eq_ = 0, f_mu_in_ = 0;
  std::vector<signed char> f_active_;
  int factor_retries_ = 0, factor_count_ = 0, iters_total_ = 0;

  // Row states: 0 inactive, 1 active, 2 saturated
  std::vector<signed char> state_;

  // Residual scalars
  double primal_res_ = 0, dual_res_ = 0;
  double primal_res_rel_ = 0, dual_res_rel_ = 0;
  double primal_obj_ = 0, duality_gap_ = 0, duality_gap_rel_ = 0;

  // Workspace (allocated in setup, reused every iteration)
  VectorXd S_, zhat_, t_, s2_, r_, din_, pv_, tp_;
  VectorXd verr_, dyrhs_, rhs_x_, dx_, dy_, dz_, Qdx_, Adx_, Gdx_;
  VectorXd wQx_, wGtz_, wGtd_, wAty_, wAx_, wGx_;
  MatrixXd GS_, K_;
  Eigen::LLT<MatrixXd, Eigen::Lower> llt_;
  std::vector<double> bp_;

  Solution sol_;
};

// One-shot convenience wrappers (cold start).
inline Solution Solve(
    const MatrixXd& Q, const VectorXd& q, const MatrixXd& A, const VectorXd& b,
    const MatrixXd& G, const VectorXd& h, const VectorXd& penalty,
    const Settings& settings = {}) {
  Solver solver;
  solver.settings = settings;
  solver.setup(Q, q, A, b, G, h, penalty);
  return solver.solve();
}

inline Solution Solve(
    const MatrixXd& Q, const VectorXd& q, const MatrixXd& A, const VectorXd& b,
    const MatrixXd& G, const VectorXd& h, double penalty,
    const Settings& settings = {}) {
  return Solve(Q, q, A, b, G, h, VectorXd::Constant(h.size(), penalty),
               settings);
}

// Inequality-only overloads.
inline Solution Solve(
    const MatrixXd& Q, const VectorXd& q, const MatrixXd& G, const VectorXd& h,
    const VectorXd& penalty, const Settings& settings = {}) {
  return Solve(Q, q, MatrixXd(0, q.size()), VectorXd(0), G, h, penalty,
               settings);
}

inline Solution Solve(
    const MatrixXd& Q, const VectorXd& q, const MatrixXd& G, const VectorXd& h,
    double penalty, const Settings& settings = {}) {
  return Solve(Q, q, G, h, VectorXd::Constant(h.size(), penalty), settings);
}

}  // namespace elastiqp
