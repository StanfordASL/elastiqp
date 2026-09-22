// Elastic primal-dual augmented Lagrangian backend, based on ProxQP

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "elastiqp/common.hpp"

namespace elastiqp::pdal {

struct Settings {
  // Convergence.
  double eps_abs = 1e-5;
  double eps_rel = 0;
  bool check_duality_gap = true;
  double eps_duality_gap_abs = 1e-5;
  double eps_duality_gap_rel = 0;
  int max_factor_retries = 10;

  // Factorization reuse via rank-one updates on active-set flips.
  bool incremental_updates = true;
  int incremental_update_budget = 256;
  int incremental_update_max_flips = 0;  // 0: use n/3.

  bool check_eq_consistency = true;
  bool warm_start = true;

  // Outer (BCL) / inner (Newton) iteration limits.
  int max_outer_iter = 250;
  int max_iter_in = 1500;

  // Proximal and penalty parameters.
  double rho = 1e-6;
  double mu_eq_init = 1e-3;
  double mu_in_init = 1e-1;
  double mu_min_eq = 1e-9;
  double mu_min_in = 1e-8;
  double mu_update_factor = 0.1;

  // BCL tolerance schedule and elastic penalty-update rules; see paper.
  double alpha_bcl = 0.1;
  double beta_bcl = 0.9;
  bool bcl_split = true;
  bool bcl_saturation_jump = true;
  bool bcl_release_jump = true;
  int bcl_release_jump_horizon = 4;
  bool bcl_warm_eta = true;

  double cold_reset_mu = 1.0 / 1.1;
  double cold_reset_threshold = 1e-5;
  double cold_reset_residual = 1e-5;
  int cold_reset_limit = 0;  // 0: disabled.
  int safe_guard = 10000;

  // Ruiz equilibration; refresh when scaling drifts by this ratio (0: never).
  bool ruiz = false;
  int ruiz_max_iter = 10;
  double ruiz_tol = 1e-3;
  double ruiz_refresh_ratio = 4.0;

  // relax(): regularization and warm-start policy (flip_tol < 0: always warm).
  double relax_reg = 1e-9;
  int relax_warm_budget = 15;
  int relax_warm_flip_tol = 0;
};

class Solver {
 public:
  Settings settings;

  // Inner-loop row classification by the unclamped dual estimate ztilde.
  enum class RowState : unsigned char { kInactive, kActive, kSaturated };

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
    relax_have_warm_ = false;
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

    ztilde_.resize(p_);
    t_.resize(p_);
    s_in_.resize(p_);
    r_.resize(p_);
    dzs_.resize(p_);
    jump_res_prev_.resize(p_);
    jump_res_cur_.resize(p_);
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

    state_.assign(static_cast<size_t>(p_), RowState::kInactive);
    f_active_.assign(static_cast<size_t>(p_), false);
    bp_.clear();
    bp_.reserve(static_cast<size_t>(2 * p_));

    GS_.resize(p_, n_);
    K_.resize(n_, n_);
    llt_ = Eigen::LLT<MatrixXd, Eigen::Lower>(n_);
    flip_idx_.reserve(static_cast<size_t>(p_));
    upd_vec_.resize(n_);
    updates_since_factor_ = 0;

    relax_ready_ = false;

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
    compute_AtA();
  }

  void setup(const MatrixXd& Q, const VectorXd& q, const MatrixXd& A,
             const VectorXd& b, const MatrixXd& G, const VectorXd& h,
             double penalty) {
    setup(Q, q, A, b, G, h, VectorXd::Constant(h.size(), penalty));
  }

  void setup(const MatrixXd& Q, const VectorXd& q, const MatrixXd& G,
             const VectorXd& h, const VectorXd& penalty) {
    setup(Q, q, MatrixXd(0, q.size()), VectorXd(0), G, h, penalty);
  }

  void setup(const MatrixXd& Q, const VectorXd& q, const MatrixXd& G,
             const VectorXd& h, double penalty) {
    setup(Q, q, G, h, VectorXd::Constant(h.size(), penalty));
  }

  Eigen::Index n() const { return n_; }
  Eigen::Index m() const { return m_; }
  Eigen::Index p() const { return p_; }

  // Setters store data in the current Ruiz frame; matrix changes invalidate
  // the factorization.
  void set_Q(const MatrixXd& Q) {
    Q_ = 0.5 * (Q + Q.transpose());
    if (ruiz_) Q_ = c_s_ * dx_s_.asDiagonal() * Q_ * dx_s_.asDiagonal();
    matrix_dirty_ = true;
  }
  void set_q(const VectorXd& q) {
    q_ = ruiz_ ? VectorXd(c_s_ * q.cwiseProduct(dx_s_)) : q;
  }
  void set_A(const MatrixXd& A) {
    check_eq_A(A);
    check_eq_b(ruiz_ ? VectorXd(b_.cwiseProduct(inv_de_)) : b_);
    A_ = ruiz_ ? MatrixXd(de_s_.asDiagonal() * A * dx_s_.asDiagonal()) : A;
    compute_AtA();
    matrix_dirty_ = true;
  }
  void set_b(const VectorXd& b) {
    check_eq_b(b);
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

  // Recompute the Ruiz scaling from the user frame and carry the warm-start
  // iterates across.
  void reequilibrate() {
    if (!ruiz_) return;
    if (scaling_pass(dxw_, dew_, diw_) <= settings.ruiz_tol) return;
    const VectorXd dx0 = dx_s_, de0 = de_s_, di0 = di_s_;
    const double c0 = c_s_;
    // Undo the current scaling, then rescale from scratch. Refreshing
    // incrementally leaks row scale into the column factors (test_ruiz.cc).
    const VectorXd ix = dx0.cwiseInverse(), ie = de0.cwiseInverse(),
                   ii = di0.cwiseInverse();
    Q_ = (ix.asDiagonal() * Q_ * ix.asDiagonal()) / c0;
    q_ = q_.cwiseProduct(ix) / c0;
    if (m_ > 0) {
      A_ = ie.asDiagonal() * A_ * ix.asDiagonal();
      b_ = b_.cwiseProduct(ie);
    }
    G_ = ii.asDiagonal() * G_ * ix.asDiagonal();
    h_ = h_.cwiseProduct(ii);
    penalty_ = penalty_.cwiseProduct(di0) / c0;
    dx_s_.setOnes();
    de_s_.setOnes();
    di_s_.setOnes();
    c_s_ = 1.0;
    equilibrate();
    // Map iterates by the ratio of new to old scaling.
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
    if (relax_have_warm_) {
      xr_ = xr_.cwiseQuotient(dx);
      if (m_ > 0) yr_ = yr_.cwiseProduct(yf);
      tr_ = tr_.cwiseProduct(di);
      for (Eigen::Index i = 0; i < p_; ++i) {
        v_t_r_[i] = zf[i] * retraction(v_t_r_[i], relax_kappa_s_) -
                  di[i] * retraction(-v_t_r_[i], relax_kappa_s_);
        v_in_r_[i] = zf[i] * retraction(v_in_r_[i], relax_kappa_s_) -
                  di[i] * retraction(-v_in_r_[i], relax_kappa_s_);
      }
      relax_kappa_s_ *= gamma;
    }
    update_unscale_vectors();
    compute_AtA();
    matrix_dirty_ = true;
  }

  // Largest rescale a fresh Ruiz pass would apply (1 = still balanced).
  double scaling_drift() const {
    if (!ruiz_) return 1.0;
    scaling_pass(dxw_, dew_, diw_);
    return ruiz_drift(diw_, ruiz_drift(dew_, ruiz_drift(dxw_)));
  }

  // Seed the next solve from user-frame (x, y, z); 0 keeps the default rho/mu.
  void set_warm_start(const VectorXd& x, const VectorXd& y,
                      const VectorXd& z, double rho = 0.0,
                      double mu_eq = 0.0, double mu_in = 0.0) {
    x_ = ruiz_ ? VectorXd(x.cwiseQuotient(dx_s_)) : x;
    if (m_ > 0) y_ = ruiz_ ? VectorXd(c_s_ * y.cwiseQuotient(de_s_)) : y;
    z_ = ruiz_ ? VectorXd(c_s_ * z.cwiseQuotient(di_s_)) : z;
    rho_ = rho > 0 ? rho : settings.rho;
    mu_eq_ = mu_eq > 0 ? mu_eq : settings.mu_eq_init;
    mu_in_ = mu_in > 0 ? mu_in : settings.mu_in_init;
    explicit_warm_ = true;
  }

  const Solution& solution() const { return sol_; }
  int factorizations() const { return factor_count_; }
  int cold_resets() const { return cold_resets_; }

  // Certified lower bound on ||Ax - b|| (0 when consistent or unchecked).
  double eq_infeasibility() const { return eq_infeas_lb_; }

  const Solution& solve() {
    const bool explicit_ws = explicit_warm_;
    explicit_warm_ = false;
    factor_count_ = 0;
    iters_total_ = 0;
    outer_iters_ = 0;
    factor_retries_ = 0;
    cold_resets_ = 0;

    if (ruiz_ && matrix_dirty_ && settings.ruiz_refresh_ratio > 0 &&
        scaling_drift() > settings.ruiz_refresh_ratio) {
      reequilibrate();
    }

    // Inconsistent equalities: report the certificate instead of iterating.
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

    if (p_ == 0) {
      return solve_no_inequalities();
    }

    bool warm_path = true;
    if (explicit_ws) {
      clamp_z();
    } else if (settings.warm_start && have_warm_) {
      // Warm start keeps (x, y, z) but restarts the penalty schedule.
      mu_eq_ = settings.mu_eq_init;
      mu_in_ = settings.mu_in_init;
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

    // Warm start: begin with a tolerance below the current residual so the
    // first outer step makes progress.
    const double eta_warm = 0.5 * primal_res_;
    if (warm_path && settings.bcl_warm_eta && eta_warm < 0.1 * eta_ext_) {
      eta_ext_ = eta_warm;
    }

    const bool track_jump_res =
        settings.bcl_split && settings.bcl_saturation_jump;
    if (track_jump_res) {
      jump_res_prev_ = (r_ - t_).cwiseProduct(inv_di_);
    }
    double gap_prev = duality_gap_;

    // Outer loop: proximal ALM with BCL updates. Inner solves at fixed
    // (xk, yk, zk, mu).
    for (int oiter = 0; oiter < settings.max_outer_iter; ++oiter) {
      outer_iters_ = oiter + 1;
      const double pri_old = primal_res_;
      const double dua_old = dual_res_;

      xk_ = x_;
      if (m_ > 0) yk_ = y_;
      zk_ = z_;
      wGx_.noalias() = G_ * x_;
      ztilde_ = zk_ + (wGx_ - h_) / mu_in_;

      if (!inner_loop(eta_in_)) {
        return finish(Status::kNumerics);
      }

      update_residuals();
      if (converged()) return finish(Status::kSolved);

      const double pri_new = primal_res_;
      const double dua_new = dual_res_;

      if (track_jump_res) {
        jump_res_cur_ = (r_ - t_).cwiseProduct(inv_di_);
      }

      if (pri_new <= eta_ext_ || iters_total_ > settings.safe_guard) {
        // Primal progress: tighten tolerances, keep mu.
        eta_ext_ *= std::pow(mu_in_, settings.beta_bcl);
        eta_in_ = std::max(eta_in_ * mu_in_, eps_in_min());
        // Residuals met but gap decaying too slowly: jump mu below the
        // shallowest releasing row.
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
        // Stalled: shrink mu; revert y only if the equalities stalled, never z.
        if (m_ > 0 && eq_res_ > eta_ext_) y_ = yk_;
        double mu_new = mu_in_ * settings.mu_update_factor;
        // Rows stuck short of saturation: jump mu to where the shallowest one
        // saturates.
        if (settings.bcl_saturation_jump && in_res_ > eta_ext_ &&
            pri_new > 0.8 * pri_old) {
          const double shallowest = saturation_jump_mu();
          if (shallowest > 0.0) mu_new = std::min(mu_new, shallowest);
        }
        shrink_mu(mu_new);
      } else {
        // Classic BCL: revert both multipliers.
        if (m_ > 0) y_ = yk_;
        z_ = zk_;
        set_mu(mu_in_ * settings.mu_update_factor,
               mu_eq_ * settings.mu_update_factor);
      }

      if (track_jump_res) jump_res_prev_.swap(jump_res_cur_);
      gap_prev = duality_gap_;

      // Cold reset (off by default).
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

  // Re-solve the KKT system with complementarity smoothed at barrier kappa
  // (differentiable); see paper.
  const Solution& relax(double kappa, double tol = 1e-6, int max_iter = 50,
                        bool warm = true) {
    if (p_ == 0 || kappa <= 0.0 || (!have_warm_ && !relax_have_warm_)) {
      return sol_;
    }
    if (!relax_ready_) relax_alloc();
    const double kappa_s = c_s_ * kappa;

    // Reuse the last relax point only if few retraction sign flips are
    // predicted.
    bool use_warm = warm && relax_have_warm_;
    if (use_warm && have_warm_ && settings.relax_warm_flip_tol >= 0) {
      use_warm = relax_predict_flips(kappa_s) <= settings.relax_warm_flip_tol;
    }

    if (use_warm) {
      relax_run(kappa_s, tol, std::min(max_iter, settings.relax_warm_budget));
      // Warm attempt failed within budget: fall back to a cold start from
      // solve().
      if (sol_.converged != 1 && have_warm_) {
        const int warm_iters = sol_.iters;
        relax_init_retraction();
        relax_run(kappa_s, tol, max_iter);
        sol_.iters += warm_iters;
      }
    } else {
      if (!have_warm_) return sol_;
      relax_init_retraction();
      relax_run(kappa_s, tol, max_iter);
    }
    relax_have_warm_ = sol_.converged == 1;
    relax_kappa_s_ = kappa_s;
    return sol_;
  }

 private:
  void compute_AtA() {
    if (m_ > 0) {
      AtA_.resize(n_, n_);
      AtA_.setZero();
      AtA_.selfadjointView<Eigen::Lower>().rankUpdate(A_.transpose());
    }
  }

  // Consistency certificate for Ax = b, refreshed whenever A or b changes.
  void check_eq_A(const MatrixXd& A) {
    eq_infeas_lb_ = 0.0;
    if (m_ == 0 || !settings.check_eq_consistency) return;
    eq_cert_.set_A(A);
  }
  void check_eq_b(const VectorXd& b) {
    eq_infeas_lb_ = 0.0;
    if (m_ == 0 || !settings.check_eq_consistency) return;
    eq_infeas_lb_ = eq_cert_.bound(b);
  }

  // One Ruiz pass: writes column/row scale factors, returns max deviation
  // from 1.
  double scaling_pass(VectorXd& dx, VectorXd& de, VectorXd& di) const {
    for (Eigen::Index k = 0; k < n_; ++k) {
      dx[k] = Q_.col(k).cwiseAbs().maxCoeff();
    }
    de.setZero();
    di.setZero();
    if (m_ > 0) fold_max_abs(A_, dx, de);
    fold_max_abs(G_, dx, di);
    return std::max({ruiz_factors(dx), ruiz_factors(de), ruiz_factors(di)});
  }

  // Ruiz equilibration accumulating dx_s_/de_s_/di_s_/c_s_. penalty scales
  // like z.
  void equilibrate() {
    VectorXd &dx = dxw_, &de = dew_, &di = diw_;
    for (int iter = 0; iter < settings.ruiz_max_iter; ++iter) {
      if (scaling_pass(dx, de, di) <= settings.ruiz_tol) break;
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
      const double gamma = ruiz_cost_gamma(Q_);
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

  const Solution& solve_no_inequalities() {
    SolveEqualityQP(Q_, q_, A_, b_, settings.rho, eq_cert_.rank_deficient(),
                    x_, y_);
    const EqualityKKTStats st = ComputeEqualityKKT(Q_, q_, A_, b_, x_, y_);
    primal_res_ = st.primal_res;
    primal_res_rel_ = st.primal_res_rel;
    dual_res_ = st.dual_res;
    dual_res_rel_ = st.dual_res_rel;
    primal_obj_ = st.primal_obj;
    duality_gap_ = st.duality_gap;
    duality_gap_rel_ = st.duality_gap_rel;
    const bool ok = st.converged(settings.eps_abs, settings.eps_rel,
                                 settings.check_duality_gap,
                                 settings.eps_duality_gap_abs,
                                 settings.eps_duality_gap_rel);
    return finish(ok ? Status::kSolved : Status::kNumerics);
  }

  // Cold start: x from the equality-penalized unconstrained system, y from
  // its residual.
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

  double eta_ext_init() const { return std::pow(0.1, settings.alpha_bcl); }
  double eps_in_min() const { return std::min(settings.eps_abs, 1e-9); }

  // Set penalties and reset the BCL tolerances to the schedule for the new mu.
  void set_mu(double mu_in_new, double mu_eq_new) {
    mu_in_ = std::max(mu_in_new, settings.mu_min_in);
    mu_eq_ = std::max(mu_eq_new, settings.mu_min_eq);
    eta_ext_ = eta_ext_init() * std::pow(mu_in_, settings.alpha_bcl);
    eta_in_ = std::max(mu_in_, eps_in_min());
    update_residuals();
  }

  // Shrink mu_in to mu_new and mu_eq by the same ratio.
  void shrink_mu(double mu_new) {
    set_mu(mu_new, mu_eq_ * (mu_new / mu_in_));
  }

  // mu at which the shallowest satisfied row with an oversized dual
  // (r < 0, z > 0) releases to 0.
  double release_jump_mu() const {
    double shallowest = 0.0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      if (r_[i] < 0.0 && z_[i] > 0.0) {
        shallowest = std::max(shallowest, -r_[i] / z_[i]);
      }
    }
    return shallowest;
  }

  // mu at which the shallowest stalled row (residual not shrinking) saturates.
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

  // True if the gap, extrapolated over the horizon at its current decay rate,
  // misses tolerance.
  bool gap_decay_too_slow(double gap_prev) const {
    if (duality_gap_ >= gap_prev || !(gap_prev > 0.0)) return true;
    const double decay =
        std::pow(duality_gap_ / gap_prev, settings.bcl_release_jump_horizon);
    return duality_gap_ * decay >= settings.eps_duality_gap_abs &&
           duality_gap_rel_ * decay >= settings.eps_duality_gap_rel;
  }

  // Semismooth Newton on the augmented Lagrangian at fixed prox center and mu.
  bool inner_loop(double eps_int) {
    for (int it = 0; it < settings.max_iter_in; ++it) {
      const double err = compute_inner_terms();
      // Always take at least one step.
      if (err <= eps_int && it > 0) return true;

      // Classify rows, then the dual step is the clamp of ztilde onto
      // [0, penalty].
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

      for (Eigen::Index i = 0; i < p_; ++i) {
        switch (state(i)) {
          case RowState::kActive:
            dzs_[i] = ztilde_[i] - z_[i];
            break;
          case RowState::kSaturated:
            dzs_[i] = penalty_[i] - z_[i];
            break;
          case RowState::kInactive:
            dzs_[i] = -z_[i];
            break;
        }
      }
      wGtd_.noalias() = G_.transpose() * dzs_;
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
        dz_[i] = is_active(i) ? Gdx_[i] / mu_in_ + dzs_[i] : dzs_[i];
      }

      iters_total_++;
      const double alpha = line_search();

      // Step too small to change anything.
      double dwmax = dx_.lpNorm<Eigen::Infinity>();
      if (m_ > 0) dwmax = std::max(dwmax, dy_.lpNorm<Eigen::Infinity>());
      dwmax = std::max(dwmax, dz_.lpNorm<Eigen::Infinity>());
      if (alpha * dwmax < 1e-11 && it > 0) return true;

      x_ += alpha * dx_;
      if (m_ > 0) y_ += alpha * dy_;
      z_ += alpha * dz_;
      // ztilde tracks Gx exactly along the step; z is clamped.
      clamp_z();
      ztilde_ += (alpha / mu_in_) * Gdx_;
      if (alpha == 0.0) return true;
    }
    return true;
  }

  // Stationarity residuals at the inner iterate; returns their inf-norm.
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

  // Exact line search: the merit is piecewise quadratic in alpha, so walk its
  // breakpoints.
  double line_search() {
    // Smooth part: derivative is b + a*alpha.
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
        if (zta >= penalty_[i]) {  // saturated
          g += penalty_[i] * c - mu_in_ * (penalty_[i] - za) * dz_[i];
        } else if (zta >= 0.0) {  // active
          g += zta * c + (zta - za) * (c - mu_in_ * dz_[i]);
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
    // Past the last breakpoint the derivative is linear.
    const double g2 = grad_at(alpha_prev + 1.0);
    const double slope = g2 - g_prev;
    if (slope <= 0.0) return alpha_prev + 1.0;
    return alpha_prev + (-g_prev) / slope;
  }

  // K = Q + rho I + A'A / mu_eq + G_act' G_act / mu_in.
  bool factor_kkt() {
    Eigen::Index na = 0;
    const double s = std::sqrt(1.0 / mu_in_);
    for (Eigen::Index i = 0; i < p_; ++i) {
      if (state(i) == RowState::kActive) {
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

  // Rank-one updates when few rows flipped; otherwise refactor, escalating rho
  // on failure.
  bool ensure_factor() {
    const bool data_changed = !factored_ || matrix_dirty_ || f_rho_ != rho_ ||
                              f_mu_eq_ != mu_eq_ || f_mu_in_ != mu_in_;
    if (!data_changed) {
      flip_idx_.clear();
      for (Eigen::Index i = 0; i < p_; ++i) {
        if (is_active(i) != f_active(i)) flip_idx_.push_back(i);
      }
      if (flip_idx_.empty()) return true;

      const Eigen::Index max_flips =
          settings.incremental_update_max_flips > 0
              ? settings.incremental_update_max_flips
              : std::max<Eigen::Index>(1, n_ / 3);
      if (settings.incremental_updates &&
          static_cast<Eigen::Index>(flip_idx_.size()) <= max_flips &&
          updates_since_factor_ + static_cast<int>(flip_idx_.size()) <=
              settings.incremental_update_budget) {
        bool ok = true;
        for (Eigen::Index i : flip_idx_) {
          const bool act = is_active(i);
          upd_vec_ = G_.row(i).transpose();
          llt_.rankUpdate(upd_vec_, (act ? 1.0 : -1.0) / mu_in_);
          ++updates_since_factor_;
          if (llt_.info() != Eigen::Success) {
            ok = false;
            break;
          }
          set_f_active(i, act);
        }
        if (ok) return true;
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
    return true;
  }

  void relax_alloc() {
    xr_.resize(n_);
    tr_.resize(p_);
    yr_.resize(m_);
    v_t_r_.resize(p_);
    v_in_r_.resize(p_);
    z_t_r_.resize(p_);
    z_in_r_.resize(p_);
    s_t_r_.resize(p_);
    s_in_r_.resize(p_);
    rf1_.resize(n_);
    rf2_.resize(p_);
    rf3_.resize(m_);
    rf4_.resize(p_);
    rf5_.resize(p_);
    d_t_r_.resize(p_);
    d_in_r_.resize(p_);
    einvr_.resize(p_);
    lamr_.resize(p_);
    wr_.resize(p_);
    pvr_.resize(p_);
    dxr_.resize(n_);
    dtr_.resize(p_);
    dyr_.resize(m_);
    dv_t_r_.resize(p_);
    dv_in_r_.resize(p_);
    llt_r_ = Eigen::LLT<MatrixXd, Eigen::Lower>(n_);
    relax_ready_ = true;
  }

  // Newton on the smoothed KKT system with a merit-backtracking line search.
  void relax_run(double kappa_s, double tol, int max_iter) {
    double rho = settings.relax_reg;
    double delta = settings.relax_reg;
    int retries = 0;
    int iter = 0;
    Status status = Status::kMaxIter;
    double res = relax_residual(kappa_s);
    while (iter < max_iter) {
      if (!std::isfinite(res)) {
        status = Status::kNumerics;
        break;
      }
      if (res < tol) {
        status = Status::kSolved;
        break;
      }
      iter++;

      // Eliminate (t, v_t, v_in) onto x; see relax_factor.
      d_t_r_ = z_t_r_.cwiseQuotient(s_t_r_);
      d_in_r_ = z_in_r_.cwiseQuotient(s_in_r_);
      bool ok = relax_factor(rho, delta);
      while (!ok && retries < settings.max_factor_retries) {
        rho *= 100;
        delta *= 100;
        retries++;
        ok = relax_factor(rho, delta);
      }
      if (!ok) {
        status = Status::kNumerics;
        break;
      }
      retries = 0;

      wr_ = d_t_r_.cwiseProduct(rf4_) + d_in_r_.cwiseProduct(rf5_) - rf2_;
      pvr_ = d_in_r_.cwiseProduct(rf5_ - einvr_.cwiseProduct(wr_));
      rhs_x_ = -rf1_;
      rhs_x_.noalias() -= G_.transpose() * pvr_;
      if (m_ > 0) {
        rhs_x_.noalias() -= (1.0 / delta) * (A_.transpose() * rf3_);
      }
      dxr_ = llt_r_.solve(rhs_x_);
      Gdx_.noalias() = G_ * dxr_;
      dtr_ = einvr_.cwiseProduct(d_in_r_.cwiseProduct(Gdx_) + wr_);
      if (m_ > 0) {
        dyr_.noalias() = A_ * dxr_;
        dyr_ += rf3_;
        dyr_ /= delta;
      }
      for (Eigen::Index i = 0; i < p_; ++i) {
        dv_t_r_[i] = (rf4_[i] - dtr_[i]) / retraction_dcomp(v_t_r_[i], kappa_s);
        dv_in_r_[i] = (rf5_[i] + Gdx_[i] - dtr_[i]) /
                   retraction_dcomp(v_in_r_[i], kappa_s);
      }

      const double merit_prev = relax_merit_;
      xr_ += dxr_;
      tr_ += dtr_;
      if (m_ > 0) yr_ += dyr_;
      v_t_r_ += dv_t_r_;
      v_in_r_ += dv_in_r_;
      double alpha = 1.0;
      double res_new = relax_residual(kappa_s);
      for (int bt = 0;
           bt < 12 && !(std::isfinite(relax_merit_) &&
                        relax_merit_ <= merit_prev);
           ++bt) {
        alpha *= 0.5;
        xr_ -= alpha * dxr_;
        tr_ -= alpha * dtr_;
        if (m_ > 0) yr_ -= alpha * dyr_;
        v_t_r_ -= alpha * dv_t_r_;
        v_in_r_ -= alpha * dv_in_r_;
        res_new = relax_residual(kappa_s);
      }
      res = res_new;
    }
    if (status == Status::kMaxIter) {
      if (!std::isfinite(res)) {
        status = Status::kNumerics;
      } else if (res < tol) {
        status = Status::kSolved;
      }
    }
    relax_finish(status, iter);
  }

  // Smoothed complementarity: retraction(v) * retraction(-v) = kappa. Branch
  // avoids cancellation.
  static double retraction(double v, double kappa) {
    const double r = std::sqrt(v * v + 4.0 * kappa);
    return v >= 0.0 ? 0.5 * (v + r) : 2.0 * kappa / (r - v);
  }

  // 1 - retraction'(v), computed without cancellation.
  static double retraction_dcomp(double v, double kappa) {
    const double r = std::sqrt(v * v + 4.0 * kappa);
    const double small = 2.0 * kappa / (r * (r + std::abs(v)));
    return v >= 0.0 ? small : 1.0 - small;
  }

  // Smoothed-KKT residual at the relax iterate; fills rf1..rf5
  // (x-stationarity, t-stationarity, eq, t >= 0, Gx - h <= t).
  double relax_residual(double kappa_s) {
    for (Eigen::Index i = 0; i < p_; ++i) {
      z_t_r_[i] = retraction(v_t_r_[i], kappa_s);
      s_t_r_[i] = retraction(-v_t_r_[i], kappa_s);
      z_in_r_[i] = retraction(v_in_r_[i], kappa_s);
      s_in_r_[i] = retraction(-v_in_r_[i], kappa_s);
    }
    wQx_.noalias() = Q_ * xr_;
    wGtz_.noalias() = G_.transpose() * z_in_r_;
    rf1_ = wQx_ + q_ + wGtz_;
    if (m_ > 0) {
      wAty_.noalias() = A_.transpose() * yr_;
      rf1_ += wAty_;
      wAx_.noalias() = A_ * xr_;
      rf3_ = wAx_ - b_;
    }
    rf2_ = penalty_ - z_t_r_ - z_in_r_;
    wGx_.noalias() = G_ * xr_;
    rf4_ = s_t_r_ - tr_;
    rf5_ = s_in_r_ + wGx_ - h_ - tr_;
    relax_dual_res_ =
        std::max(inf_us(rf1_, inv_cdx_), inf_us(rf2_, z_us_));
    relax_primal_res_ =
        std::max({inf_us(rf4_, inv_di_), inf_us(rf5_, inv_di_),
                  m_ > 0 ? inf_us(rf3_, inv_de_) : 0.0});
    relax_merit_ = ssq_us(rf1_, inv_cdx_) + ssq_us(rf2_, z_us_) +
                   ssq_us(rf4_, inv_di_) + ssq_us(rf5_, inv_di_) +
                   (m_ > 0 ? ssq_us(rf3_, inv_de_) : 0.0);
    return std::max(relax_dual_res_, relax_primal_res_);
  }

  // Retraction variables whose sign differs between the solve() point and the
  // last relax() point.
  int relax_predict_flips(double kappa_s) {
    const double corner2 = 100.0 * kappa_s;
    wGx_.noalias() = G_ * x_;
    int flips = 0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      const RowRetraction rr = row_retraction(i, wGx_[i] - h_[i]);
      if ((rr.v_t > 0) != (v_t_r_[i] > 0) &&
          std::abs(rr.v_t * v_t_r_[i]) > corner2) {
        flips++;
      }
      if ((rr.v_in > 0) != (v_in_r_[i] > 0) &&
          std::abs(rr.v_in * v_in_r_[i]) > corner2) {
        flips++;
      }
    }
    return flips;
  }

  // Seed relax from the solve() point.
  void relax_init_retraction() {
    xr_ = x_;
    if (m_ > 0) yr_ = y_;
    wGx_.noalias() = G_ * x_;
    for (Eigen::Index i = 0; i < p_; ++i) {
      const RowRetraction rr = row_retraction(i, wGx_[i] - h_[i]);
      tr_[i] = rr.t;
      v_t_r_[i] = rr.v_t;
      v_in_r_[i] = rr.v_in;
    }
  }

  // Reduced Newton matrix after eliminating (t, v_t, v_in).
  bool relax_factor(double rho, double delta) {
    einvr_ = ((d_t_r_ + d_in_r_).array() + rho).cwiseInverse();
    lamr_ = d_in_r_.array() * (d_t_r_.array() + rho) * einvr_.array();
    GS_.noalias() = lamr_.cwiseSqrt().asDiagonal() * G_;
    K_.triangularView<Eigen::Lower>() = Q_;
    K_.diagonal().array() += rho;
    if (m_ > 0) {
      K_.triangularView<Eigen::Lower>() += (1.0 / delta) * AtA_;
    }
    K_.selfadjointView<Eigen::Lower>().rankUpdate(GS_.transpose());
    llt_r_.compute(K_);
    return llt_r_.info() == Eigen::Success &&
           std::isfinite(K_.diagonal().sum());
  }

  void relax_finish(Status status, int iters) {
    sol_.x = xr_.cwiseProduct(dx_s_);
    sol_.t = tr_.cwiseProduct(inv_di_);
    sol_.y = yr_.cwiseProduct(y_us_);
    sol_.z = z_in_r_.cwiseProduct(z_us_);
    sol_.z_t = z_t_r_.cwiseProduct(z_us_);
    sol_.status = status;
    sol_.converged = status == Status::kSolved ? 1 : 0;
    sol_.iters = iters;
    sol_.outer_iters = 0;
    count_row_states(sol_.t, sol_.z, settings.eps_abs, sol_.n_active,
                     sol_.n_saturated);
    wQx_.noalias() = Q_ * xr_;
    const double xQx = xr_.dot(wQx_);
    sol_.primal_obj = (0.5 * xQx + q_.dot(xr_) + penalty_.dot(tr_)) / c_s_;
    double dual_obj = (-0.5 * xQx - h_.dot(z_in_r_)) / c_s_;
    if (m_ > 0) dual_obj -= b_.dot(yr_) / c_s_;
    sol_.primal_res = relax_primal_res_;
    sol_.dual_res = relax_dual_res_;
    sol_.duality_gap = std::abs(sol_.primal_obj - dual_obj);
  }

  void clamp_z() { z_ = z_.cwiseMax(0.0).cwiseMin(penalty_); }

  // t_i = max(r_i + mu_in (z_i - penalty_i), 0): elastic slack implied by the
  // current dual.
  double elastic_slack(Eigen::Index i, double r) const {
    return std::max(r + mu_in_ * (z_[i] - penalty_[i]), 0.0);
  }

  // Retraction coordinates (t, v_t, v_in) of row i from the solve() iterate
  // (v = z - s).
  struct RowRetraction {
    double t, v_t, v_in;
  };
  RowRetraction row_retraction(Eigen::Index i, double r) const {
    const double t = elastic_slack(i, r);
    return {t, (penalty_[i] - z_[i]) - t, z_[i] - std::max(t - r, 0.0)};
  }

  // Norms in the user frame; s holds the unscaling factors.
  template <typename V>
  static double inf_us(const V& v, const VectorXd& s) {
    return v.size() > 0 ? v.cwiseAbs().cwiseProduct(s).maxCoeff() : 0.0;
  }
  template <typename V>
  static double ssq_us(const V& v, const VectorXd& s) {
    return v.size() > 0 ? v.cwiseProduct(s).squaredNorm() : 0.0;
  }

  // Outer residuals, objectives and duality gap in the user frame.
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
      s_in_[i] = std::max(t_[i] - r_[i], 0.0);
      in_res = std::max(in_res, (r_[i] - t_[i]) * inv_di_[i]);
    }
    in_res = std::max(in_res, 0.0);
    double primal_rel_norm = std::max(
        {inf_us(wGx_ - t_, inv_di_), inf_us(h_, inv_di_),
         inf_us(s_in_, inv_di_), inf_us(t_, inv_di_)});
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
           (dual_res_ < settings.eps_abs ||
            dual_res_rel_ < settings.eps_rel);
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
    sol_.n_active = sol_.n_saturated = 0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      if (state(i) == RowState::kActive) ++sol_.n_active;
      if (state(i) == RowState::kSaturated) ++sol_.n_saturated;
    }
    sol_.primal_obj = primal_obj_;
    sol_.primal_res = primal_res_;
    sol_.dual_res = dual_res_;
    sol_.duality_gap = duality_gap_;
    // A Numerics failure discards the warm start.
    if (status != Status::kInfeasible) {
      have_warm_ = status != Status::kNumerics;
    }
    return sol_;
  }

  // Problem data (Ruiz frame).
  Eigen::Index n_ = 0, m_ = 0, p_ = 0;
  MatrixXd Q_, A_, G_;
  VectorXd q_, b_, h_, penalty_;
  MatrixXd AtA_;

  // Iterates and prox centers.
  VectorXd x_, y_, z_;
  VectorXd xk_, yk_, zk_;
  bool have_warm_ = false;
  bool explicit_warm_ = false;

  // Penalties and BCL tolerances.
  double rho_ = 0, mu_eq_ = 0, mu_in_ = 0;
  double eta_ext_ = 0, eta_in_ = 0;

  // Ruiz scaling (x = dx_s * x_s, etc.) and cached unscaling vectors.
  bool ruiz_ = false;
  double c_s_ = 1.0;
  VectorXd dx_s_, de_s_, di_s_;
  VectorXd inv_cdx_, inv_de_, inv_di_;
  VectorXd y_us_, z_us_;
  mutable VectorXd dxw_, dew_, diw_;

  EqCertificate eq_cert_;
  double eq_infeas_lb_ = 0.0;

  // Factorization cache: parameters and active set at last factor.
  bool matrix_dirty_ = true, factored_ = false;
  double f_rho_ = 0, f_mu_eq_ = 0, f_mu_in_ = 0;
  std::vector<bool> f_active_;
  std::vector<Eigen::Index> flip_idx_;
  VectorXd upd_vec_;
  int updates_since_factor_ = 0;
  int factor_retries_ = 0, factor_count_ = 0, iters_total_ = 0;
  int outer_iters_ = 0;
  int cold_resets_ = 0;

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

  // Outer residuals.
  double primal_res_ = 0, dual_res_ = 0;
  double in_res_ = 0, eq_res_ = 0;
  double primal_res_rel_ = 0, dual_res_rel_ = 0;
  double primal_obj_ = 0, duality_gap_ = 0, duality_gap_rel_ = 0;

  // Inner-loop workspace.
  VectorXd ztilde_;
  VectorXd t_, s_in_, r_, dzs_;
  VectorXd jump_res_prev_, jump_res_cur_;
  VectorXd verr_, dyrhs_, rhs_x_, dx_, dy_, dz_, Qdx_, Adx_, Gdx_;
  VectorXd wQx_, wGtz_, wGtd_, wAty_, wAx_, wGx_;
  MatrixXd GS_, K_;
  Eigen::LLT<MatrixXd, Eigen::Lower> llt_;
  std::vector<double> bp_;

  // relax() workspace and warm-start state.
  VectorXd xr_, tr_, yr_, v_t_r_, v_in_r_;
  VectorXd z_t_r_, z_in_r_, s_t_r_, s_in_r_;
  VectorXd rf1_, rf2_, rf3_, rf4_, rf5_;
  VectorXd d_t_r_, d_in_r_, einvr_, lamr_, wr_, pvr_;
  VectorXd dxr_, dtr_, dyr_, dv_t_r_, dv_in_r_;
  Eigen::LLT<MatrixXd, Eigen::Lower> llt_r_;
  double relax_primal_res_ = 0, relax_dual_res_ = 0;
  double relax_merit_ = 0;
  bool relax_have_warm_ = false;
  double relax_kappa_s_ = 0;
  bool relax_ready_ = false;

  Solution sol_;
};

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

}  // namespace elastiqp::pdal
