// Elastic proximal interior-point backend, based on PIQP

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef ELASTIQP_IPM_DEBUG
#include <cstdio>
#endif

#include "elastiqp/common.hpp"

namespace elastiqp::ipm {

struct Settings {
  // Convergence.
  double eps_abs = 1e-5;
  double eps_rel = 0;
  bool check_duality_gap = true;
  double eps_duality_gap_abs = 1e-5;
  double eps_duality_gap_rel = 0;
  int max_factor_retries = 10;

  bool check_eq_consistency = true;
  int max_iter = 250;

  // Proximal regularization: initial values, floors, and stall thresholds for
  // lowering the floor.
  double rho_init = 1e-6;
  double delta_init = 1e-4;
  double infeasibility_threshold = 0.9;
  double reg_lower_limit = 1e-10;
  double reg_finetune_lower_limit = 1e-13;
  int reg_finetune_primal_update_threshold = 7;
  int reg_finetune_dual_update_threshold = 7;

  double tau = 0.99;  // Fraction-to-boundary.

  // Ruiz equilibration; recomputed by solve() after any matrix update.
  bool ruiz = true;
  int ruiz_max_iter = 10;
  double ruiz_tol = 1e-3;
};

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
    has_iterate_ = false;

    x_.resize(n_);
    t_.resize(p_);
    y_.resize(m_);
    s_t_.resize(p_);
    s_in_.resize(p_);
    z_t_.resize(p_);
    z_in_.resize(p_);
    xi_x_.resize(n_);
    xi_t_.resize(p_);
    nu_y_.resize(m_);
    nu_t_.resize(p_);
    nu_in_.resize(p_);

    rnr_x_.resize(n_);
    rnr_t_.resize(p_);
    rnr_y_.resize(m_);
    rnr_z_t_.resize(p_);
    rnr_z_in_.resize(p_);
    res_x_.resize(n_);
    res_t_.resize(p_);
    res_y_.resize(m_);
    res_z_t_.resize(p_);
    res_z_in_.resize(p_);
    res_s_t_.resize(p_);
    res_s_in_.resize(p_);

    w_t_.resize(p_);
    w_in_.resize(p_);
    w_t_inv_.resize(p_);
    w_in_inv_.resize(p_);
    d_inv_.resize(p_);
    lambda_.resize(p_);
    GS_.resize(p_, n_);
    K_.resize(n_, n_);
    llt_ = Eigen::LLT<MatrixXd, Eigen::Lower>(n_);
    rb_t_.resize(p_);
    rb_in_.resize(p_);
    wv_.resize(p_);
    pv_.resize(p_);
    rhs_x_.resize(n_);
    Gdx_.resize(p_);
    dx_.resize(n_);
    dt_.resize(p_);
    dy_.resize(m_);
    ds_t_.resize(p_);
    ds_in_.resize(p_);
    dz_t_.resize(p_);
    dz_in_.resize(p_);
    wQx_.resize(n_);
    wGtz_in_.resize(n_);
    wGxt_.resize(p_);
    wAty_.resize(n_);
    wAx_.resize(m_);
    zero_p_ = VectorXd::Zero(p_);

    check_eq_A(A_);
    check_eq_b(b_);

    ruiz_ = settings.ruiz && p_ > 0;
    dx_s_ = VectorXd::Ones(n_);
    de_s_ = VectorXd::Ones(m_);
    di_s_ = VectorXd::Ones(p_);
    c_s_ = 1.0;
    if (ruiz_) equilibrate();
    update_unscale_vectors();
    compute_AtA();
    matrix_dirty_ = false;
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

  // Setters store data in the current Ruiz frame; solve() re-equilibrates
  // after a matrix change (every solve is cold, so there is no iterate to
  // carry across the new scaling).
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

  const Solution& solution() const { return sol_; }

  // Certified lower bound on ||Ax - b|| (0 when consistent or unchecked).
  double eq_infeasibility() const { return eq_infeas_lb_; }

  const Solution& solve() {
    if (ruiz_ && matrix_dirty_) reequilibrate();
    matrix_dirty_ = false;

    // Inconsistent equalities: report the certificate instead of iterating.
    if (settings.check_eq_consistency && settings.eps_rel <= 0 &&
        eq_infeas_lb_ > settings.eps_abs) {
      x_.setZero();
      t_.setZero();
      y_.setZero();
      s_t_.setOnes();
      s_in_.setOnes();
      z_t_.setOnes();
      z_in_.setOnes();
      if (p_ > 0) {
        update_residuals_nr();
      } else {
        wQx_.noalias() = Q_ * x_;
        wAty_.noalias() = A_.transpose() * y_;
        wAx_.noalias() = A_ * x_;
        dual_res_ = (wQx_ + q_ + wAty_)
                        .cwiseProduct(inv_cdx_)
                        .lpNorm<Eigen::Infinity>();
        primal_res_ =
            (wAx_ - b_).cwiseProduct(inv_de_).lpNorm<Eigen::Infinity>();
        primal_obj_ = (0.5 * x_.dot(wQx_) + q_.dot(x_)) / c_s_;
        duality_gap_ =
            std::abs(primal_obj_ - (-0.5 * x_.dot(wQx_) - b_.dot(y_)) / c_s_);
      }
      return finish(Status::kInfeasible, 0);
    }

    if (p_ == 0) {
      return solve_no_inequalities();
    }

    factor_retries_ = 0;
    no_primal_update_ = 0;
    no_dual_update_ = 0;

    rho_ = settings.rho_init;
    delta_ = settings.delta_init;
    reg_limit_ = settings.reg_lower_limit;
    if (!init_cold()) {
      return finish(Status::kNumerics, 0);
    }
    mu_ = calculate_mu();

    xi_x_ = x_;
    xi_t_ = t_;
    nu_t_ = z_t_;
    nu_in_ = z_in_;
    if (m_ > 0) nu_y_ = y_;

    update_residuals_nr();
    prev_primal_res_ = primal_res_;
    prev_dual_res_ = dual_res_;

    int iter = 0;
    while (iter < settings.max_iter) {
      if ((primal_res_ < settings.eps_abs ||
           primal_res_rel_ < settings.eps_rel) &&
          (dual_res_ < settings.eps_abs || dual_res_rel_ < settings.eps_rel) &&
          (!settings.check_duality_gap ||
           duality_gap_ < settings.eps_duality_gap_abs ||
           duality_gap_rel_ < settings.eps_duality_gap_rel)) {
        return finish(Status::kSolved, iter);
      }

      update_residuals_r();

      iter++;

      // Keep duals strictly positive.
      {
        bool boundary_shifted = false;
        const double eps = std::numeric_limits<double>::epsilon();
        for (Eigen::Index i = 0; i < p_; ++i) {
          if (z_t_[i] < eps) {
            z_t_[i] += eps;
            boundary_shifted = true;
          }
          if (z_in_[i] < eps) {
            z_in_[i] += eps;
            boundary_shifted = true;
          }
        }
        if (boundary_shifted) mu_ = calculate_mu();
      }

      // Stalled at the regularization floor with small prox terms: lower the
      // floor.
      if ((no_primal_update_ > settings.reg_finetune_primal_update_threshold &&
           rho_ == reg_limit_ &&
           reg_limit_ != settings.reg_finetune_lower_limit) ||
          (no_dual_update_ > settings.reg_finetune_dual_update_threshold &&
           delta_ == reg_limit_ &&
           reg_limit_ != settings.reg_finetune_lower_limit)) {
        if (dual_prox_inf_ < settings.infeasibility_threshold &&
            primal_prox_inf_ < settings.infeasibility_threshold) {
          reg_limit_ = settings.reg_finetune_lower_limit;
          no_primal_update_ = 0;
          no_dual_update_ = 0;
        }
      }

      // Factorization failed: inflate regularization and retry.
      update_scalings();
      bool regularization_changed = false;
      while (!factor()) {
        if (factor_retries_ < settings.max_factor_retries) {
          delta_ *= 100;
          rho_ *= 100;
          factor_retries_++;
          reg_limit_ = std::min(10 * reg_limit_, settings.eps_abs);
          regularization_changed = true;
          update_scalings();
          continue;
        }
        return finish(Status::kNumerics, iter);
      }
      factor_retries_ = 0;

      if (regularization_changed) update_residuals_r();

      // Predictor step.
      res_s_t_ = -s_t_.cwiseProduct(z_t_);
      res_s_in_ = -s_in_.cwiseProduct(z_in_);
      kkt_solve(res_x_, res_y_, res_t_, res_z_t_, res_z_in_, res_s_t_,
                res_s_in_);

      double alpha_s, alpha_z;
      calculate_step(alpha_s, alpha_z);
      alpha_s *= settings.tau;
      alpha_z *= settings.tau;

      double sigma =
          ((s_t_ + alpha_s * ds_t_).dot(z_t_ + alpha_z * dz_t_) +
           (s_in_ + alpha_s * ds_in_).dot(z_in_ + alpha_z * dz_in_)) /
          (mu_ * static_cast<double>(2 * p_));
      sigma = std::max(0.0, std::min(1.0, sigma));
      sigma = sigma * sigma * sigma;

      // Mehrotra corrector with centering sigma.
      res_s_t_.array() += -ds_t_.array() * dz_t_.array() + sigma * mu_;
      res_s_in_.array() += -ds_in_.array() * dz_in_.array() + sigma * mu_;
      kkt_solve(res_x_, res_y_, res_t_, res_z_t_, res_z_in_, res_s_t_,
                res_s_in_);

      calculate_step(alpha_s, alpha_z);
      const double primal_step = alpha_s * settings.tau;
      const double dual_step = alpha_z * settings.tau;

      x_ += primal_step * dx_;
      t_ += primal_step * dt_;
      s_t_ += primal_step * ds_t_;
      s_in_ += primal_step * ds_in_;
      if (m_ > 0) y_ += dual_step * dy_;
      z_t_ += dual_step * dz_t_;
      z_in_ += dual_step * dz_in_;

      const double mu_prev = mu_;
      mu_ = calculate_mu();
      const double mu_rate = std::max(0.0, (mu_prev - mu_) / mu_prev);

      update_residuals_nr();
#ifdef ELASTIQP_IPM_DEBUG
      std::printf(
          "it %3d pres %.2e dres %.2e gap %.2e mu %.2e rho %.1e delta %.1e lim "
          "%.1e "
          "a_p %.2e a_d %.2e sigma %.2e |x| %.2e |t| %.2e |z_in| %.2e |z_t| "
          "%.2e nopu %d nodu %d\n",
          iter, primal_res_, dual_res_, duality_gap_, mu_, rho_, delta_,
          reg_limit_, primal_step, dual_step, sigma,
          x_.lpNorm<Eigen::Infinity>(), t_.lpNorm<Eigen::Infinity>(),
          z_in_.lpNorm<Eigen::Infinity>(), z_t_.lpNorm<Eigen::Infinity>(),
          no_primal_update_, no_dual_update_);
#endif

      // Move each prox center only when its residual improved; otherwise
      // count a stall.
      if (dual_res_ < 0.95 * prev_dual_res_ ||
          (dual_res_ < settings.eps_abs || dual_res_rel_ < settings.eps_rel) ||
          (rho_ == settings.reg_finetune_lower_limit &&
           dual_prox_inf_ < settings.infeasibility_threshold)) {
        xi_x_ = x_;
        xi_t_ = t_;
        rho_ = std::max(reg_limit_, (1.0 - mu_rate) * rho_);
      } else {
        no_primal_update_++;
        if (iter < 5 || dual_prox_inf_ < settings.infeasibility_threshold) {
          rho_ = std::max(reg_limit_, (1.0 - 0.666 * mu_rate) * rho_);
        }
      }

      if (primal_res_ < 0.95 * prev_primal_res_ ||
          (primal_res_ < settings.eps_abs ||
           primal_res_rel_ < settings.eps_rel) ||
          (delta_ == settings.reg_finetune_lower_limit &&
           primal_prox_inf_ < settings.infeasibility_threshold)) {
        if (m_ > 0) nu_y_ = y_;
        nu_t_ = z_t_;
        nu_in_ = z_in_;
        delta_ = std::max(reg_limit_, (1.0 - mu_rate) * delta_);
      } else {
        no_dual_update_++;
        if (iter < 5 || primal_prox_inf_ < settings.infeasibility_threshold) {
          delta_ = std::max(reg_limit_, (1.0 - 0.666 * mu_rate) * delta_);
        }
      }
    }

    return finish(Status::kMaxIter, iter);
  }

  // Re-solve the KKT system with complementarity fixed at kappa
  // (differentiable); see paper.
  const Solution& relax(double kappa, double tol = 1e-8, int max_iter = 30) {
    if (p_ == 0 || !has_iterate_) return sol_;

    // Row scaling cancels in each s.z pair; only the cost scale remains.
    const double kappa_s = c_s_ * kappa;
    int iter = 0;
    Status status = Status::kMaxIter;
    while (iter < max_iter) {
      update_residuals_nr();
      double comp_res = 0;
      for (Eigen::Index i = 0; i < p_; ++i) {
        comp_res = std::max(comp_res, std::abs(s_t_[i] * z_t_[i] - kappa_s));
        comp_res = std::max(comp_res, std::abs(s_in_[i] * z_in_[i] - kappa_s));
      }
      comp_res /= c_s_;
      if (std::max({primal_res_, dual_res_, comp_res}) < tol) {
        status = Status::kSolved;
        break;
      }
      iter++;

      // Prox center at the current iterate: regularization without bias.
      xi_x_ = x_;
      xi_t_ = t_;
      nu_t_ = z_t_;
      nu_in_ = z_in_;
      if (m_ > 0) nu_y_ = y_;
      update_residuals_r();

      update_scalings();
      bool ok = true;
      while (!factor()) {
        if (factor_retries_ < settings.max_factor_retries) {
          delta_ *= 100;
          rho_ *= 100;
          factor_retries_++;
          update_scalings();
          continue;
        }
        ok = false;
        break;
      }
      factor_retries_ = 0;
      if (!ok) {
        status = Status::kNumerics;
        break;
      }

      res_s_t_.array() = kappa_s - (s_t_.array() * z_t_.array());
      res_s_in_.array() = kappa_s - (s_in_.array() * z_in_.array());
      kkt_solve(res_x_, res_y_, res_t_, res_z_t_, res_z_in_, res_s_t_,
                res_s_in_);

      double alpha_s, alpha_z;
      calculate_step(alpha_s, alpha_z);
      const double primal_step = alpha_s * settings.tau;
      const double dual_step = alpha_z * settings.tau;

      x_ += primal_step * dx_;
      t_ += primal_step * dt_;
      s_t_ += primal_step * ds_t_;
      s_in_ += primal_step * ds_in_;
      if (m_ > 0) y_ += dual_step * dy_;
      z_t_ += dual_step * dz_t_;
      z_in_ += dual_step * dz_in_;
    }

    update_residuals_nr();
    return finish(status, iter);
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

  // Undo the current Ruiz scaling and recompute it from the user frame
  // (refreshing incrementally leaks row scale into the column factors).
  void reequilibrate() {
    const VectorXd ix = dx_s_.cwiseInverse(), ie = de_s_.cwiseInverse(),
                   ii = di_s_.cwiseInverse();
    Q_ = (ix.asDiagonal() * Q_ * ix.asDiagonal()) / c_s_;
    q_ = q_.cwiseProduct(ix) / c_s_;
    if (m_ > 0) {
      A_ = ie.asDiagonal() * A_ * ix.asDiagonal();
      b_ = b_.cwiseProduct(ie);
    }
    G_ = ii.asDiagonal() * G_ * ix.asDiagonal();
    h_ = h_.cwiseProduct(ii);
    penalty_ = penalty_.cwiseProduct(di_s_) / c_s_;
    dx_s_.setOnes();
    de_s_.setOnes();
    di_s_.setOnes();
    c_s_ = 1.0;
    equilibrate();
    update_unscale_vectors();
    compute_AtA();
  }

  // Ruiz equilibration accumulating dx_s_/de_s_/di_s_/c_s_; penalty scales
  // like z.
  void equilibrate() {
    VectorXd dx(n_), de(m_), di(p_);
    for (int iter = 0; iter < settings.ruiz_max_iter; ++iter) {
      for (Eigen::Index k = 0; k < n_; ++k) {
        dx[k] = Q_.col(k).cwiseAbs().maxCoeff();
      }
      de.setZero();
      di.setZero();
      if (m_ > 0) fold_max_abs(A_, dx, de);
      fold_max_abs(G_, dx, di);
      const double dev =
          std::max({ruiz_factors(dx), ruiz_factors(de), ruiz_factors(di)});
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
    SolveEqualityQP(Q_, q_, A_, b_, settings.rho_init,
                    eq_cert_.rank_deficient(), x_, y_);
    const EqualityKKTStats st = ComputeEqualityKKT(Q_, q_, A_, b_, x_, y_);
    primal_res_ = st.primal_res;
    primal_res_rel_ = st.primal_res_rel;
    dual_res_ = st.dual_res;
    dual_res_rel_ = st.dual_res_rel;
    primal_obj_ = st.primal_obj;
    duality_gap_ = st.duality_gap;
    duality_gap_rel_ = st.duality_gap_rel;
    const bool ok = st.converged(
        settings.eps_abs, settings.eps_rel, settings.check_duality_gap,
        settings.eps_duality_gap_abs, settings.eps_duality_gap_rel);
    return finish(ok ? Status::kSolved : Status::kNumerics, 0);
  }

  // Diagonal scalings for eliminating the t-block (see kkt_solve).
  void update_scalings() {
    w_t_ = s_t_.cwiseQuotient(z_t_).array() + delta_;
    w_in_ = s_in_.cwiseQuotient(z_in_).array() + delta_;
    w_t_inv_ = w_t_.cwiseInverse();
    w_in_inv_ = w_in_.cwiseInverse();
    d_inv_ = (w_t_inv_ + w_in_inv_).array() + rho_;
    d_inv_ = d_inv_.cwiseInverse();
    lambda_ = (w_t_inv_.array() + rho_) * w_in_inv_.array() * d_inv_.array();
  }

  // Condensed KKT: K = Q + rho I + A'A/delta + G' diag(lambda) G.
  bool factor() {
    GS_.noalias() = lambda_.cwiseSqrt().asDiagonal() * G_;
    K_.triangularView<Eigen::Lower>() = Q_;
    K_.diagonal().array() += rho_;
    if (m_ > 0) {
      K_.triangularView<Eigen::Lower>() += (1.0 / delta_) * AtA_;
    }
    K_.selfadjointView<Eigen::Lower>().rankUpdate(GS_.transpose());
    llt_.compute(K_);
    return llt_.info() == Eigen::Success && std::isfinite(K_.diagonal().sum());
  }

  // Solve the regularized KKT system, eliminating (s, z, t, y) onto x.
  void kkt_solve(const VectorXd& v_x, const VectorXd& v_y, const VectorXd& v_t,
                 const VectorXd& v_z1, const VectorXd& v_z2,
                 const VectorXd& v_s1, const VectorXd& v_s2) {
    rb_t_ = v_z1 - v_s1.cwiseQuotient(z_t_);
    rb_in_ = v_z2 - v_s2.cwiseQuotient(z_in_);
    wv_ = v_t - w_t_inv_.cwiseProduct(rb_t_) - w_in_inv_.cwiseProduct(rb_in_);
    pv_ = w_in_inv_.cwiseProduct(d_inv_.cwiseProduct(wv_) + rb_in_);
    rhs_x_.noalias() = G_.transpose() * pv_;
    rhs_x_ += v_x;
    if (m_ > 0) {
      rhs_x_.noalias() += (1.0 / delta_) * (A_.transpose() * v_y);
    }
    dx_ = llt_.solve(rhs_x_);
    if (m_ > 0) {
      dy_.noalias() = A_ * dx_;
      dy_ -= v_y;
      dy_ /= delta_;
    }
    Gdx_.noalias() = G_ * dx_;
    dt_ = d_inv_.cwiseProduct(wv_ + w_in_inv_.cwiseProduct(Gdx_));
    dz_in_ = w_in_inv_.cwiseProduct(Gdx_ - dt_ - rb_in_);
    dz_t_ = -w_t_inv_.cwiseProduct(rb_t_ + dt_);
    ds_t_ = (v_s1 - s_t_.cwiseProduct(dz_t_)).cwiseQuotient(z_t_);
    ds_in_ = (v_s2 - s_in_.cwiseProduct(dz_in_)).cwiseQuotient(z_in_);
  }

  double calculate_mu() const {
    return (s_t_.dot(z_t_) + s_in_.dot(z_in_)) / static_cast<double>(2 * p_);
  }

  // Largest step keeping slacks and duals nonnegative.
  void calculate_step(double& alpha_s, double& alpha_z) const {
    alpha_s = 1.0;
    alpha_z = 1.0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      if (ds_t_[i] < 0) alpha_s = std::min(alpha_s, -s_t_[i] / ds_t_[i]);
      if (ds_in_[i] < 0) alpha_s = std::min(alpha_s, -s_in_[i] / ds_in_[i]);
      if (dz_t_[i] < 0) alpha_z = std::min(alpha_z, -z_t_[i] / dz_t_[i]);
      if (dz_in_[i] < 0) alpha_z = std::min(alpha_z, -z_in_[i] / dz_in_[i]);
    }
  }

  // Residuals without prox terms (Ruiz frame); their norms, the objectives
  // and the gap are measured in the user frame.
  void update_residuals_nr() {
    prev_primal_res_ = primal_res_;
    prev_dual_res_ = dual_res_;

    const auto inf_us = [](const VectorXd& v, const VectorXd& s) {
      return v.size() > 0 ? v.cwiseAbs().cwiseProduct(s).maxCoeff() : 0.0;
    };

    wQx_.noalias() = Q_ * x_;
    wGtz_in_.noalias() = G_.transpose() * z_in_;
    rnr_x_ = -wQx_ - q_ - wGtz_in_;
    double aty_norm = 0;
    if (m_ > 0) {
      wAty_.noalias() = A_.transpose() * y_;
      rnr_x_ -= wAty_;
      aty_norm = inf_us(wAty_, inv_cdx_);
    }
    rnr_t_ = z_t_ + z_in_;
    const double z12_norm = inf_us(rnr_t_, z_us_);
    rnr_t_ -= penalty_;
    double dual_rel_norm =
        std::max({inf_us(wQx_, inv_cdx_), inf_us(q_, inv_cdx_),
                  inf_us(wGtz_in_, inv_cdx_), aty_norm, z12_norm,
                  inf_us(penalty_, z_us_)});

    wGxt_.noalias() = G_ * x_;
    wGxt_ -= t_;
    rnr_z_t_ = t_ - s_t_;
    rnr_z_in_ = h_ - wGxt_ - s_in_;
    double primal_rel_norm = std::max(
        {inf_us(wGxt_, inv_di_), inf_us(h_, inv_di_), inf_us(s_in_, inv_di_),
         inf_us(t_, inv_di_), inf_us(s_t_, inv_di_)});
    double eq_res_norm = 0;
    if (m_ > 0) {
      wAx_.noalias() = A_ * x_;
      rnr_y_ = b_ - wAx_;
      eq_res_norm = inf_us(rnr_y_, inv_de_);
      primal_rel_norm = std::max(
          {primal_rel_norm, inf_us(wAx_, inv_de_), inf_us(b_, inv_de_)});
    }

    const double xQx = x_.dot(wQx_);
    primal_obj_ = (0.5 * xQx + q_.dot(x_) + penalty_.dot(t_)) / c_s_;
    double dual_obj = (-0.5 * xQx - h_.dot(z_in_)) / c_s_;
    double gap_rel_norm =
        std::max({std::abs(xQx), std::abs(q_.dot(x_)),
                  std::abs(penalty_.dot(t_)), std::abs(h_.dot(z_in_))}) /
        c_s_;
    if (m_ > 0) {
      const double by = b_.dot(y_);
      dual_obj -= by / c_s_;
      gap_rel_norm = std::max(gap_rel_norm, std::abs(by) / c_s_);
    }
    duality_gap_ = std::abs(primal_obj_ - dual_obj);
    duality_gap_rel_ = duality_gap_ / std::max(1.0, gap_rel_norm);

    primal_res_ = std::max(
        {inf_us(rnr_z_t_, inv_di_), inf_us(rnr_z_in_, inv_di_), eq_res_norm});
    primal_res_rel_ = primal_res_ / std::max(1.0, primal_rel_norm);
    dual_res_ = std::max(inf_us(rnr_x_, inv_cdx_), inf_us(rnr_t_, z_us_));
    dual_res_rel_ = dual_res_ / std::max(1.0, dual_rel_norm);
  }

  // Add the prox terms; record their size for the stall tests.
  void update_residuals_r() {
    res_x_ = rnr_x_ - rho_ * (x_ - xi_x_);
    res_t_ = rnr_t_ - rho_ * (t_ - xi_t_);
    res_z_t_ = rnr_z_t_ - delta_ * (nu_t_ - z_t_);
    res_z_in_ = rnr_z_in_ - delta_ * (nu_in_ - z_in_);
    primal_prox_inf_ =
        delta_ * std::max((nu_t_ - z_t_).lpNorm<Eigen::Infinity>(),
                          (nu_in_ - z_in_).lpNorm<Eigen::Infinity>());
    if (m_ > 0) {
      res_y_ = rnr_y_ - delta_ * (nu_y_ - y_);
      primal_prox_inf_ = std::max(
          primal_prox_inf_, delta_ * (nu_y_ - y_).lpNorm<Eigen::Infinity>());
    }
    dual_prox_inf_ = rho_ * std::max((x_ - xi_x_).lpNorm<Eigen::Infinity>(),
                                     (t_ - xi_t_).lpNorm<Eigen::Infinity>());
  }

  // Cold start: one KKT solve from the origin, then shift slacks and duals
  // interior.
  bool init_cold() {
    x_.setZero();
    t_.setZero();
    y_.setZero();
    s_t_.setOnes();
    s_in_.setOnes();
    z_t_.setOnes();
    z_in_.setOnes();

    update_scalings();
    while (!factor()) {
      if (factor_retries_ < settings.max_factor_retries) {
        delta_ *= 100;
        rho_ *= 100;
        factor_retries_++;
        reg_limit_ = std::min(10 * reg_limit_, settings.eps_abs);
        update_scalings();
      } else {
        return false;
      }
    }
    factor_retries_ = 0;

    kkt_solve(-q_, b_, -penalty_, zero_p_, h_, zero_p_, zero_p_);
    x_ = dx_;
    t_ = dt_;
    if (m_ > 0) y_ = dy_;
    z_t_ = dz_t_;
    z_in_ = dz_in_;
    s_t_ = ds_t_;
    s_in_ = ds_in_;

    const double delta_s =
        std::max(0.0, std::max(-s_t_.minCoeff(), -s_in_.minCoeff()));
    const double delta_z =
        std::max(0.0, std::max(-z_t_.minCoeff(), -z_in_.minCoeff()));
    s_t_.array() += delta_s;
    s_in_.array() += delta_s;
    z_t_.array() += delta_z;
    z_in_.array() += delta_z;

    // Project each (s, z) pair onto the central path at mu_init.
    const double mu_init = std::max(calculate_mu(), 1e-10);
    for (Eigen::Index i = 0; i < p_; ++i) {
      double c = z_t_[i] - delta_z;
      z_t_[i] = 0.5 * (c + std::sqrt(c * c + 4 * mu_init));
      s_t_[i] = z_t_[i] - c;
      c = z_in_[i] - delta_z;
      z_in_[i] = 0.5 * (c + std::sqrt(c * c + 4 * mu_init));
      s_in_[i] = z_in_[i] - c;
    }
    return true;
  }

  // Unscale to the user frame. relax() needs an iterate from a solve that
  // did not fail numerically; a gated (kInfeasible) tick keeps the last one.
  const Solution& finish(Status status, int iters) {
    sol_.x = x_.cwiseProduct(dx_s_);
    sol_.t = t_.cwiseProduct(inv_di_);
    sol_.y = y_.cwiseProduct(y_us_);
    sol_.z = z_in_.cwiseProduct(z_us_);
    sol_.z_t = z_t_.cwiseProduct(z_us_);
    sol_.status = status;
    sol_.converged = status == Status::kSolved ? 1 : 0;
    sol_.iters = iters;
    sol_.outer_iters = 0;
    count_row_states(sol_.t, sol_.z, settings.eps_abs, sol_.n_active,
                     sol_.n_saturated);
    sol_.primal_obj = primal_obj_;
    sol_.primal_res = primal_res_;
    sol_.dual_res = dual_res_;
    sol_.duality_gap = duality_gap_;
    if (status != Status::kInfeasible) {
      has_iterate_ = status != Status::kNumerics;
    }
    return sol_;
  }

  // Problem data (Ruiz frame).
  Eigen::Index n_ = 0, m_ = 0, p_ = 0;
  MatrixXd Q_, A_, G_;
  VectorXd q_, b_, h_, penalty_;
  MatrixXd AtA_;  // Lower triangle only.

  // Iterates.
  VectorXd x_, t_, y_, s_t_, s_in_, z_t_, z_in_;
  bool has_iterate_ = false;

  EqCertificate eq_cert_;
  double eq_infeas_lb_ = 0.0;

  // Ruiz scaling, inverses, and dual unscale factors.
  bool ruiz_ = false;
  double c_s_ = 1.0;
  VectorXd dx_s_, de_s_, di_s_;
  VectorXd inv_cdx_, inv_de_, inv_di_;
  VectorXd y_us_, z_us_;
  bool matrix_dirty_ = false;  // a matrix setter ran since the last solve

  // Prox centers.
  VectorXd xi_x_, xi_t_, nu_y_, nu_t_, nu_in_;

  double rho_ = 0, delta_ = 0, reg_limit_ = 0, mu_ = 0;
  int factor_retries_ = 0, no_primal_update_ = 0, no_dual_update_ = 0;

  // Negated KKT residuals: rnr_ without prox terms, res_ with.
  VectorXd rnr_x_, rnr_t_, rnr_y_, rnr_z_t_, rnr_z_in_;
  VectorXd res_x_, res_t_, res_y_, res_z_t_, res_z_in_;
  VectorXd res_s_t_, res_s_in_;
  double primal_res_ = 0, dual_res_ = 0;
  double prev_primal_res_ = 0, prev_dual_res_ = 0;
  double primal_res_rel_ = 0, dual_res_rel_ = 0;
  double primal_obj_ = 0, duality_gap_ = 0, duality_gap_rel_ = 0;
  double primal_prox_inf_ = 0, dual_prox_inf_ = 0;

  // KKT workspace.
  VectorXd w_t_, w_in_, w_t_inv_, w_in_inv_, d_inv_, lambda_;
  MatrixXd GS_, K_;
  Eigen::LLT<MatrixXd, Eigen::Lower> llt_;
  VectorXd rb_t_, rb_in_, wv_, pv_, rhs_x_, Gdx_;
  VectorXd dx_, dt_, dy_, ds_t_, ds_in_, dz_t_, dz_in_;
  VectorXd wQx_, wGtz_in_, wGxt_, wAty_, wAx_;
  VectorXd zero_p_;

  Solution sol_;
};

inline Solution Solve(const MatrixXd& Q, const VectorXd& q, const MatrixXd& A,
                      const VectorXd& b, const MatrixXd& G, const VectorXd& h,
                      const VectorXd& penalty, const Settings& settings = {}) {
  Solver solver;
  solver.settings = settings;
  solver.setup(Q, q, A, b, G, h, penalty);
  return solver.solve();
}

inline Solution Solve(const MatrixXd& Q, const VectorXd& q, const MatrixXd& A,
                      const VectorXd& b, const MatrixXd& G, const VectorXd& h,
                      double penalty, const Settings& settings = {}) {
  return Solve(Q, q, A, b, G, h, VectorXd::Constant(h.size(), penalty),
               settings);
}

inline Solution Solve(const MatrixXd& Q, const VectorXd& q, const MatrixXd& G,
                      const VectorXd& h, const VectorXd& penalty,
                      const Settings& settings = {}) {
  return Solve(Q, q, MatrixXd(0, q.size()), VectorXd(0), G, h, penalty,
               settings);
}

inline Solution Solve(const MatrixXd& Q, const VectorXd& q, const MatrixXd& G,
                      const VectorXd& h, double penalty,
                      const Settings& settings = {}) {
  return Solve(Q, q, G, h, VectorXd::Constant(h.size(), penalty), settings);
}

}  // namespace elastiqp::ipm
