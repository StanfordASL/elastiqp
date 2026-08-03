// ElastiQP-IPM: a proximal interior-point method for elastic QPs
//
// This is the secondary backend. In general, elastiqp/pdal.hpp's Solver
// will be faster on most robotics tasks. However, the IPM solver adds:
//
//   * differentiability -- relax(kappa) walks back to a kappa-relaxed
//     central point, used for smooth derivatives
//   * equality residuals near machine precision (tighter than PDAL)
//
// You can also combine the two backends. For instance, if you require
// differentiability, a fast forward-backward path is (1) solve the forward
// path with PDAL, and (2) differentiate with IPM (see warm_start_from()).
//
// This backend is based on PIQP, with the elastic condensation tricks of
// qpax, and a few other changes (see notes below). The elastic QP form is
// stated in elastiqp.hpp.
//
// Notes:
//
// PIQP regularizes its Newton system with a primal proximal term rho (on the
// primal iterate distance to a prox center xi) and a dual proximal term delta
// (on the dual distance to a prox center nu). Its dense backend condenses the
// hard-constrained KKT onto
//   Q + rho*I + (1/delta) A^T A + G^T diag(1/(s/z + delta)) G.
// For the elastic form, eliminating (s1, s2, z1, z2, t) gives the same
// structure with a generalized diagonal weight:
//
//   W1 = s1/z1 + delta,  W2 = s2/z2 + delta,  D = rho + 1/W1 + 1/W2
//   K  = Q + rho*I + (1/delta) A^T A + G^T diag(Lambda) G,
//   Lambda = (rho + 1/W1) / (W2 .* D)
//
// Lambda -> 1/W2 as W1 -> 0 (hard constraint limit, PIQP's weight) and
// Lambda -> 1/(s1/z1 + s2/z2) as rho, delta -> 0 (qpax's elastic weight).
// The equality dual step is recovered as dy = (A dx - v_y) / delta, exactly
// as in PIQP's dense backend.
//
// Faithful to PIQP: initialization (unit slacks KKT solve + positivity shift),
// Mehrotra predictor-corrector with separate primal/dual step sizes,
// regularization update rules driven by residual progress and mu decrease,
// factorization retries, and absolute/relative + duality-gap convergence
// criteria on the unregularized residuals.
// Omitted from PIQP: iterative refinement and infeasibility detection.
//
// Beyond vanilla PIQP (which re-initializes its iterates on every solve),
// IpmSolver supports warm starting across repeated solves of slowly-changing
// problems (e.g. a control loop); see init_warm() for the mechanism.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <limits>

#include "elastiqp/types.hpp"  // Status, Solution

namespace elastiqp {

// Interior-point settings -- every knob this backend has. The termination
// block is field-for-field identical to elastiqp::Settings (tests/test_pdal.cc
// static_asserts the defaults agree); everything below is specific to this
// method.
struct IpmSettings {
  // Termination, on the unregularized elastic-KKT residuals.
  double eps_abs = 1e-8;
  double eps_rel = 1e-9;
  bool check_duality_gap = true;
  double eps_duality_gap_abs = 1e-8;
  double eps_duality_gap_rel = 1e-9;
  int max_factor_retries = 10;

  // Reuse the previous solve's iterate from the second solve() on, after
  // flooring slacks and duals off the boundary (see init_warm()).
  bool warm_start = true;

  // Interior-point iterations, which is also what Solution::iters reports.
  int max_iter = 250;

  double rho_init = 1e-6;
  double delta_init = 1e-4;

  double infeasibility_threshold = 0.9;

  double reg_lower_limit = 1e-10;
  double reg_finetune_lower_limit = 1e-13;
  int reg_finetune_primal_update_threshold = 7;
  int reg_finetune_dual_update_threshold = 7;

  double tau = 0.99;

  // Warm-start flooring (takes effect from the second solve() on, and for
  // warm_start_from()). The previous solution seeds the iterate; slacks and
  // duals are floored at warm_start_fraction times the residual of the
  // previous iterate under the new data (clamped to [min_floor, max_floor])
  // so step lengths stay healthy when the problem has shifted.
  double warm_start_fraction = 0.1;
  double warm_start_min_floor = 1e-8;
  double warm_start_max_floor = 1.0;

  // Ruiz equilibration of the stacked [Q A' G'; A 0 0; G 0 0] structure
  // plus cost normalization -- field-for-field the same block as
  // elastiqp::Settings, with the same semantics. Read at setup() time
  // (ignored when p == 0); set_*() updates are rescaled with the
  // setup()-time scaling (call setup() again to re-equilibrate). The
  // iterates live in the scaled space, but termination and every reported
  // residual stay on the unscaled elastic KKT, and Solution is returned
  // unscaled. The penalty transforms as w_scaled = c * w / delta_row; the
  // slacks scale with the row and the duals against it, so each s.z
  // product only picks up the cost factor c.
  bool ruiz = false;
  int ruiz_max_iter = 10;
  double ruiz_tol = 1e-3;
};

// Reusable PIQP-style elastic solver. setup() once, then alternate
// set_*() / solve() to exploit warm starting across a sequence of related
// problems. All workspace is allocated in setup(); solve() is allocation-free
// on the warm path.
class IpmSolver {
 public:
  IpmSettings settings;

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

    x_.resize(n_);
    t_.resize(p_);
    y_.resize(m_);
    s1_.resize(p_);
    s2_.resize(p_);
    z1_.resize(p_);
    z2_.resize(p_);
    xi_x_.resize(n_);
    xi_t_.resize(p_);
    nu_y_.resize(m_);
    nu1_.resize(p_);
    nu2_.resize(p_);

    rnr_x_.resize(n_);
    rnr_t_.resize(p_);
    rnr_y_.resize(m_);
    rnr_z1_.resize(p_);
    rnr_z2_.resize(p_);
    res_x_.resize(n_);
    res_t_.resize(p_);
    res_y_.resize(m_);
    res_z1_.resize(p_);
    res_z2_.resize(p_);
    res_s1_.resize(p_);
    res_s2_.resize(p_);

    w1_.resize(p_);
    w2_.resize(p_);
    w1_inv_.resize(p_);
    w2_inv_.resize(p_);
    d_inv_.resize(p_);
    lambda_.resize(p_);
    GS_.resize(p_, n_);
    K_.resize(n_, n_);
    llt_ = Eigen::LLT<MatrixXd, Eigen::Lower>(n_);
    rb1_.resize(p_);
    rb2_.resize(p_);
    wv_.resize(p_);
    pv_.resize(p_);
    rhs_x_.resize(n_);
    Gdx_.resize(p_);
    dx_.resize(n_);
    dt_.resize(p_);
    dy_.resize(m_);
    ds1_.resize(p_);
    ds2_.resize(p_);
    dz1_.resize(p_);
    dz2_.resize(p_);
    wQx_.resize(n_);
    wGtz2_.resize(n_);
    wGxt_.resize(p_);
    wAty_.resize(n_);
    wAx_.resize(m_);
    zero_p_ = VectorXd::Zero(p_);

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

  // Data updates between solves (dimensions must not change). The previous
  // solution is kept as the warm-start point. Inputs are unscaled; with Ruiz
  // active they are rescaled into the setup()-time scaled frame on ingestion.
  void set_Q(const MatrixXd& Q) {
    Q_ = 0.5 * (Q + Q.transpose());
    if (ruiz_) Q_ = c_s_ * dx_s_.asDiagonal() * Q_ * dx_s_.asDiagonal();
  }
  void set_q(const VectorXd& q) {
    q_ = ruiz_ ? VectorXd(c_s_ * q.cwiseProduct(dx_s_)) : q;
  }
  void set_A(const MatrixXd& A) {
    A_ = ruiz_ ? MatrixXd(de_s_.asDiagonal() * A * dx_s_.asDiagonal()) : A;
    compute_AtA();
  }
  void set_b(const VectorXd& b) {
    b_ = ruiz_ ? VectorXd(b.cwiseProduct(de_s_)) : b;
  }
  void set_G(const MatrixXd& G) {
    G_ = ruiz_ ? MatrixXd(di_s_.asDiagonal() * G * dx_s_.asDiagonal()) : G;
  }
  void set_h(const VectorXd& h) {
    h_ = ruiz_ ? VectorXd(h.cwiseProduct(di_s_)) : h;
  }
  void set_penalty(const VectorXd& penalty) {
    penalty_ = ruiz_ ? VectorXd(c_s_ * penalty.cwiseQuotient(di_s_)) : penalty;
  }

  // Explicitly seed the next solve()'s starting iterate, replacing the
  // automatic warm start (see init_warm). The point is used exactly as
  // given, in the user's unscaled frame: the slacks and duals must be
  // strictly positive, and lifting the iterate off the boundary is the
  // caller's responsibility. rho/delta > 0 reset the proximal
  // regularization state; when <= 0 the state carried over from the
  // previous solve (or rho_init/delta_init if there is none) is kept.
  // Takes effect once, for the next solve() only, regardless of
  // settings.warm_start -- this is the hook for building external
  // warm-start strategies on top of the solver.
  void set_warm_start(const VectorXd& x, const VectorXd& t, const VectorXd& y,
                      const VectorXd& s_t, const VectorXd& s_ineq,
                      const VectorXd& z_t, const VectorXd& z_ineq,
                      double rho = 0.0, double delta = 0.0) {
    scale_iterate(x, t, y, s_t, s_ineq, z_t, z_ineq);
    rho_ = rho > 0 ? rho : (rho_ > 0 ? rho_ : settings.rho_init);
    delta_ = delta > 0 ? delta : (delta_ > 0 ? delta_ : settings.delta_init);
    explicit_warm_ = true;
  }

  // Seed the next solve() from a Solution produced by either backend --
  // the handoff for "solve fast with elastiqp::Solver, then differentiate
  // here". Unlike set_warm_start(), this applies the interior-point
  // boundary floor for you, which a foreign solution needs: a converged
  // PDAL certificate sits exactly on the boundary, and an unfloored
  // interior-point start there collapses the fraction-to-boundary step
  // length. Dimensions must match setup(). Takes effect once, for the
  // next solve() only.
  void warm_start_from(const Solution& sol) {
    scale_iterate(sol.x, sol.t, sol.y, sol.s_t, sol.s_ineq, sol.z_t,
                  sol.z_ineq);
    if (rho_ <= 0) rho_ = settings.rho_init;
    if (delta_ <= 0) delta_ = settings.delta_init;
    if (p_ > 0) init_warm();  // floor slacks/duals off the boundary
    explicit_warm_ = true;
  }

  const Solution& solution() const { return sol_; }

  const Solution& solve() {
    const bool explicit_ws = explicit_warm_;
    explicit_warm_ = false;
    if (p_ == 0) {
      return solve_no_inequalities();
    }

    factor_retries_ = 0;
    no_primal_update_ = 0;
    no_dual_update_ = 0;

    if (explicit_ws) {
      // Iterate and regularization state were provided via set_warm_start();
      // use them as-is.
      reg_limit_ = settings.reg_lower_limit;
    } else if (settings.warm_start && have_warm_) {
      // Keep rho_, delta_, reg_limit_ from the previous solve. PIQP's
      // regularization schedule shrinks with the relative decrease of mu;
      // a warm start begins with mu already tiny, so restarting from
      // rho_init/delta_init would leave the schedule with no fuel and
      // dominate the iteration count. The previous (converged) values are
      // the right scale, and factorization retries bump them back up if the
      // perturbed problem needs it.
      init_warm();
    } else {
      rho_ = settings.rho_init;
      delta_ = settings.delta_init;
      reg_limit_ = settings.reg_lower_limit;
      if (!init_cold()) {
        return finish(Status::kNumerics, 0);
      }
    }
    mu_ = calculate_mu();

    xi_x_ = x_;
    xi_t_ = t_;
    nu1_ = z1_;
    nu2_ = z2_;
    if (m_ > 0) nu_y_ = y_;

    update_residuals_nr();
    prev_primal_res_ = primal_res_;
    prev_dual_res_ = dual_res_;

    // ---------------- main loop (PIQP::solve_impl) ------------------------
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

      // Avoid getting too close to the boundary (division by zero guard).
      {
        bool boundary_shifted = false;
        const double eps = std::numeric_limits<double>::epsilon();
        for (Eigen::Index i = 0; i < p_; ++i) {
          if (z1_[i] < eps) { z1_[i] += eps; boundary_shifted = true; }
          if (z2_[i] < eps) { z2_[i] += eps; boundary_shifted = true; }
        }
        if (boundary_shifted) mu_ = calculate_mu();
      }

      // Avoid converging to a local minimum: lower the regularization floor
      // once progress stalls at the current limit (PIQP's finetune logic).
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

      // ------------------ predictor step ------------------
      res_s1_ = -s1_.cwiseProduct(z1_);
      res_s2_ = -s2_.cwiseProduct(z2_);
      kkt_solve(res_x_, res_y_, res_t_, res_z1_, res_z2_, res_s1_, res_s2_);

      double alpha_s, alpha_z;
      calculate_step(alpha_s, alpha_z);
      alpha_s *= settings.tau;
      alpha_z *= settings.tau;

      double sigma = ((s1_ + alpha_s * ds1_).dot(z1_ + alpha_z * dz1_) +
                      (s2_ + alpha_s * ds2_).dot(z2_ + alpha_z * dz2_)) /
                     (mu_ * static_cast<double>(2 * p_));
      sigma = std::max(0.0, std::min(1.0, sigma));
      sigma = sigma * sigma * sigma;

      // ------------------ corrector step ------------------
      res_s1_.array() += -ds1_.array() * dz1_.array() + sigma * mu_;
      res_s2_.array() += -ds2_.array() * dz2_.array() + sigma * mu_;
      kkt_solve(res_x_, res_y_, res_t_, res_z1_, res_z2_, res_s1_, res_s2_);

      calculate_step(alpha_s, alpha_z);
      const double primal_step = alpha_s * settings.tau;
      const double dual_step = alpha_z * settings.tau;

      // ------------------ update ------------------
      x_ += primal_step * dx_;
      t_ += primal_step * dt_;
      s1_ += primal_step * ds1_;
      s2_ += primal_step * ds2_;
      if (m_ > 0) y_ += dual_step * dy_;
      z1_ += dual_step * dz1_;
      z2_ += dual_step * dz2_;

      const double mu_prev = mu_;
      mu_ = calculate_mu();
      const double mu_rate = std::max(0.0, (mu_prev - mu_) / mu_prev);

      // ------------------ update regularization ------------------
      update_residuals_nr();

      if (dual_res_ < 0.95 * prev_dual_res_ ||
          (dual_res_ < settings.eps_abs ||
           dual_res_rel_ < settings.eps_rel) ||
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
        nu1_ = z1_;
        nu2_ = z2_;
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

  // Re-solve from the current iterate to a kappa-relaxed central point:
  // the same KKT conditions but with relaxed complementarity
  // s1.z1 = s2.z2 = kappa (qpax's "relaxation"). Differentiating the KKT
  // system at this point instead of the exact solution yields smoothed
  // gradients whose backward solve stays well-conditioned near degenerate
  // (weakly-active) constraints -- the complementarity margins are bounded
  // below by ~kappa. Call after solve(); solution() afterwards returns the
  // RELAXED point, not the optimum.
  const Solution& relax(double kappa, double tol = 1e-8, int max_iter = 30) {
    if (p_ == 0 || !have_warm_) return sol_;

    // kappa is in the user's frame. Row scaling cancels in each s.z pair
    // (s scales with the row, z against it), so the scaled-space target is
    // just c_s_ * kappa, and the residual unscales by 1/c_s_.
    const double kappa_s = c_s_ * kappa;
    int iter = 0;
    Status status = Status::kMaxIter;
    while (iter < max_iter) {
      update_residuals_nr();
      double comp_res = 0;
      for (Eigen::Index i = 0; i < p_; ++i) {
        comp_res = std::max(comp_res, std::abs(s1_[i] * z1_[i] - kappa_s));
        comp_res = std::max(comp_res, std::abs(s2_[i] * z2_[i] - kappa_s));
      }
      comp_res /= c_s_;
      if (std::max({primal_res_, dual_res_, comp_res}) < tol) {
        status = Status::kSolved;
        break;
      }
      iter++;

      // Proximal centers at the current iterate: the regularized system then
      // agrees with the true KKT at this point, and rho/delta (already at
      // their converged floors) only stabilize the factorization.
      xi_x_ = x_;
      xi_t_ = t_;
      nu1_ = z1_;
      nu2_ = z2_;
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

      // Newton step toward the kappa-hyperbola (no Mehrotra correction).
      res_s1_.array() = kappa_s - (s1_.array() * z1_.array());
      res_s2_.array() = kappa_s - (s2_.array() * z2_.array());
      kkt_solve(res_x_, res_y_, res_t_, res_z1_, res_z2_, res_s1_, res_s2_);

      double alpha_s, alpha_z;
      calculate_step(alpha_s, alpha_z);
      const double primal_step = alpha_s * settings.tau;
      const double dual_step = alpha_z * settings.tau;

      x_ += primal_step * dx_;
      t_ += primal_step * dt_;
      s1_ += primal_step * ds1_;
      s2_ += primal_step * ds2_;
      if (m_ > 0) y_ += dual_step * dy_;
      z1_ += dual_step * dz1_;
      z2_ += dual_step * dz2_;
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

  // PIQP's limit_scaling guard (dense/preconditioner.tpp): a norm below
  // 1e-4 is treated as 1, so the row/column is left unscaled -- a row at
  // the numerical noise floor would otherwise be amplified by 1/sqrt(nrm)
  // >= 100x per sweep while its penalty shrinks toward zero (this stalls
  // the interior point method: tests/test_ipm.cc's noise-row test hits
  // max_iter without the guard); a norm above 1e4 is capped so one sweep's
  // scale factor stays in [1e-2, 100].
  static double limit_scaling(double nrm) {
    return nrm < 1e-4 ? 1.0 : std::min(nrm, 1e4);
  }

  // Ruiz sweeps on the stacked symmetric structure [Q A' G'; A 0 0; G 0 0],
  // applied in place to the stored data, with the per-sweep cost
  // normalization gamma = 1/max(1, mean |Q| column norm) -- identical to
  // elastiqp::Solver::equilibrate() (pdal.hpp) so the two backends scale a
  // given problem the same way.
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

  // Ingest an unscaled iterate into the setup()-time scaled frame
  // (see IpmSettings::ruiz). Identity when Ruiz is off.
  void scale_iterate(const VectorXd& x, const VectorXd& t, const VectorXd& y,
                     const VectorXd& s_t, const VectorXd& s_ineq,
                     const VectorXd& z_t, const VectorXd& z_ineq) {
    if (!ruiz_) {
      x_ = x;
      t_ = t;
      if (m_ > 0) y_ = y;
      s1_ = s_t;
      s2_ = s_ineq;
      z1_ = z_t;
      z2_ = z_ineq;
      return;
    }
    x_ = x.cwiseQuotient(dx_s_);
    t_ = t.cwiseProduct(di_s_);
    if (m_ > 0) y_ = c_s_ * y.cwiseQuotient(de_s_);
    s1_ = s_t.cwiseProduct(di_s_);
    s2_ = s_ineq.cwiseProduct(di_s_);
    z1_ = c_s_ * z_t.cwiseQuotient(di_s_);
    z2_ = c_s_ * z_ineq.cwiseQuotient(di_s_);
  }

  // No inequality constraints: with equalities the problem is a plain
  // equality-constrained QP; solve its (indefinite) KKT system directly.
  // The solve is not iterative, so failures (singular KKT, inconsistent
  // A x = b) are caught by checking the KKT residuals against the usual
  // convergence criteria: kSolved or kNumerics.
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

    // KKT residuals and duality gap (update_residuals_nr minus the
    // inequality terms, which would be norms of empty vectors).
    wQx_.noalias() = Q_ * x_;
    rnr_x_ = -wQx_ - q_;
    double dual_rel_norm = std::max(wQx_.lpNorm<Eigen::Infinity>(),
                                    q_.lpNorm<Eigen::Infinity>());
    primal_res_ = 0.0;
    primal_res_rel_ = 0.0;
    double by = 0.0;
    if (m_ > 0) {
      wAty_.noalias() = A_.transpose() * y_;
      rnr_x_ -= wAty_;
      dual_rel_norm = std::max(dual_rel_norm, wAty_.lpNorm<Eigen::Infinity>());
      wAx_.noalias() = A_ * x_;
      rnr_y_ = b_ - wAx_;
      primal_res_ = rnr_y_.lpNorm<Eigen::Infinity>();
      primal_res_rel_ =
          primal_res_ / std::max(1.0, std::max(wAx_.lpNorm<Eigen::Infinity>(),
                                               b_.lpNorm<Eigen::Infinity>()));
      by = b_.dot(y_);
    }
    dual_res_ = rnr_x_.lpNorm<Eigen::Infinity>();
    dual_res_rel_ = dual_res_ / std::max(1.0, dual_rel_norm);

    const double xQx = x_.dot(wQx_);
    primal_obj_ = 0.5 * xQx + q_.dot(x_);
    duality_gap_ = std::abs(primal_obj_ - (-0.5 * xQx - by));
    duality_gap_rel_ =
        duality_gap_ / std::max(1.0, std::max({std::abs(xQx),
                                               std::abs(q_.dot(x_)),
                                               std::abs(by)}));

    const bool ok =
        (primal_res_ < settings.eps_abs ||
         primal_res_rel_ < settings.eps_rel) &&
        (dual_res_ < settings.eps_abs || dual_res_rel_ < settings.eps_rel) &&
        (!settings.check_duality_gap ||
         duality_gap_ < settings.eps_duality_gap_abs ||
         duality_gap_rel_ < settings.eps_duality_gap_rel);
    return finish(ok ? Status::kSolved : Status::kNumerics, 0);
  }

  // Elastic KKT scalings for the current (s, z, rho, delta).
  void update_scalings() {
    w1_ = s1_.cwiseQuotient(z1_).array() + delta_;
    w2_ = s2_.cwiseQuotient(z2_).array() + delta_;
    w1_inv_ = w1_.cwiseInverse();
    w2_inv_ = w2_.cwiseInverse();
    d_inv_ = (w1_inv_ + w2_inv_).array() + rho_;
    d_inv_ = d_inv_.cwiseInverse();
    lambda_ = (w1_inv_.array() + rho_) * w2_inv_.array() * d_inv_.array();
  }

  // Factor K = Q + rho*I + (1/delta) A^T A + G^T diag(lambda) G via a
  // symmetric rank update on the row-scaled G.
  bool factor() {
    GS_.noalias() = lambda_.cwiseSqrt().asDiagonal() * G_;
    K_.triangularView<Eigen::Lower>() = Q_;
    K_.diagonal().array() += rho_;
    if (m_ > 0) {
      K_.triangularView<Eigen::Lower>() += (1.0 / delta_) * AtA_;
    }
    K_.selfadjointView<Eigen::Lower>().rankUpdate(GS_.transpose());
    llt_.compute(K_);
    return llt_.info() == Eigen::Success &&
           std::isfinite(K_.diagonal().sum());
  }

  // Solve the regularized elastic Newton system
  //   (Q + rho I) dx + A^T dy   + G^T dz2           = v_x
  //   A dx - delta dy                               = v_y
  //   rho dt - dz1 - dz2                            = v_t
  //   -dt + ds1 - delta dz1                         = v_z1
  //   G dx - dt + ds2 - delta dz2                   = v_z2
  //   S1 dz1 + Z1 ds1 = v_s1,  S2 dz2 + Z2 ds2 = v_s2
  // by condensation onto K (factored above). Results in dx_, dy_, dt_,
  // ds*_, dz*_.
  void kkt_solve(const VectorXd& v_x, const VectorXd& v_y, const VectorXd& v_t,
                 const VectorXd& v_z1, const VectorXd& v_z2,
                 const VectorXd& v_s1, const VectorXd& v_s2) {
    rb1_ = v_z1 - v_s1.cwiseQuotient(z1_);
    rb2_ = v_z2 - v_s2.cwiseQuotient(z2_);
    wv_ = v_t - w1_inv_.cwiseProduct(rb1_) - w2_inv_.cwiseProduct(rb2_);
    pv_ = w2_inv_.cwiseProduct(d_inv_.cwiseProduct(wv_) + rb2_);
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
    dt_ = d_inv_.cwiseProduct(wv_ + w2_inv_.cwiseProduct(Gdx_));
    dz2_ = w2_inv_.cwiseProduct(Gdx_ - dt_ - rb2_);
    dz1_ = -w1_inv_.cwiseProduct(rb1_ + dt_);
    ds1_ = (v_s1 - s1_.cwiseProduct(dz1_)).cwiseQuotient(z1_);
    ds2_ = (v_s2 - s2_.cwiseProduct(dz2_)).cwiseQuotient(z2_);
  }

  double calculate_mu() const {
    return (s1_.dot(z1_) + s2_.dot(z2_)) / static_cast<double>(2 * p_);
  }

  // Largest step in [0, 1] keeping slacks (alpha_s) and duals (alpha_z)
  // nonnegative (PIQP::calculate_step). The equality dual y is free.
  void calculate_step(double& alpha_s, double& alpha_z) const {
    alpha_s = 1.0;
    alpha_z = 1.0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      if (ds1_[i] < 0) alpha_s = std::min(alpha_s, -s1_[i] / ds1_[i]);
      if (ds2_[i] < 0) alpha_s = std::min(alpha_s, -s2_[i] / ds2_[i]);
      if (dz1_[i] < 0) alpha_z = std::min(alpha_z, -z1_[i] / dz1_[i]);
      if (dz2_[i] < 0) alpha_z = std::min(alpha_z, -z2_[i] / dz2_[i]);
    }
  }

  // Unregularized residuals, objectives, and convergence norms
  // (PIQP::update_residuals_nr). The rnr_* VECTORS stay in the (possibly
  // Ruiz-scaled) frame -- they seed the Newton right-hand sides via
  // update_residuals_r -- but every NORM below is unscaled componentwise
  // (PIQP unscales through its preconditioner the same way), so the
  // reported residuals and the termination test are on the true elastic
  // KKT regardless of equilibration (with Ruiz off, all unscale vectors
  // are ones).
  void update_residuals_nr() {
    prev_primal_res_ = primal_res_;
    prev_dual_res_ = dual_res_;

    const auto inf_us = [](const VectorXd& v, const VectorXd& s) {
      return v.size() > 0 ? v.cwiseAbs().cwiseProduct(s).maxCoeff() : 0.0;
    };

    // Dual residual: [Q x + q + A^T y + G^T z2; penalty - z1 - z2]
    wQx_.noalias() = Q_ * x_;
    wGtz2_.noalias() = G_.transpose() * z2_;
    rnr_x_ = -wQx_ - q_ - wGtz2_;
    double aty_norm = 0;
    if (m_ > 0) {
      wAty_.noalias() = A_.transpose() * y_;
      rnr_x_ -= wAty_;
      aty_norm = inf_us(wAty_, inv_cdx_);
    }
    rnr_t_ = z1_ + z2_;
    const double z12_norm = inf_us(rnr_t_, z_us_);
    rnr_t_ -= penalty_;
    double dual_rel_norm = std::max(
        {inf_us(wQx_, inv_cdx_), inf_us(q_, inv_cdx_),
         inf_us(wGtz2_, inv_cdx_), aty_norm, z12_norm,
         inf_us(penalty_, z_us_)});

    // Primal residual: [b - A x; t - s1; G x - t + s2 - h]
    // (negated, PIQP convention)
    wGxt_.noalias() = G_ * x_;
    wGxt_ -= t_;
    rnr_z1_ = t_ - s1_;
    rnr_z2_ = h_ - wGxt_ - s2_;
    double primal_rel_norm = std::max(
        {inf_us(wGxt_, inv_di_), inf_us(h_, inv_di_), inf_us(s2_, inv_di_),
         inf_us(t_, inv_di_), inf_us(s1_, inv_di_)});
    double eq_res_norm = 0;
    if (m_ > 0) {
      wAx_.noalias() = A_ * x_;
      rnr_y_ = b_ - wAx_;
      eq_res_norm = inf_us(rnr_y_, inv_de_);
      primal_rel_norm = std::max(
          {primal_rel_norm, inf_us(wAx_, inv_de_), inf_us(b_, inv_de_)});
    }

    // Objectives and duality gap. Dual objective of the elastic QP:
    // -0.5 x^T Q x - b^T y - h^T z2 (the t >= 0 bound has zero rhs).
    // Every scaled term is c_s_ times its unscaled value.
    const double xQx = x_.dot(wQx_);
    primal_obj_ = (0.5 * xQx + q_.dot(x_) + penalty_.dot(t_)) / c_s_;
    double dual_obj = (-0.5 * xQx - h_.dot(z2_)) / c_s_;
    double gap_rel_norm =
        std::max({std::abs(xQx), std::abs(q_.dot(x_)),
                  std::abs(penalty_.dot(t_)), std::abs(h_.dot(z2_))}) /
        c_s_;
    if (m_ > 0) {
      const double by = b_.dot(y_);
      dual_obj -= by / c_s_;
      gap_rel_norm = std::max(gap_rel_norm, std::abs(by) / c_s_);
    }
    duality_gap_ = std::abs(primal_obj_ - dual_obj);
    duality_gap_rel_ = duality_gap_ / std::max(1.0, gap_rel_norm);

    primal_res_ = std::max({inf_us(rnr_z1_, inv_di_),
                            inf_us(rnr_z2_, inv_di_), eq_res_norm});
    primal_res_rel_ = primal_res_ / std::max(1.0, primal_rel_norm);
    dual_res_ = std::max(inf_us(rnr_x_, inv_cdx_), inf_us(rnr_t_, z_us_));
    dual_res_rel_ = dual_res_ / std::max(1.0, dual_rel_norm);
  }

  // Regularized residuals (PIQP::update_residuals_r) and proximal
  // infeasibility measures.
  void update_residuals_r() {
    res_x_ = rnr_x_ - rho_ * (x_ - xi_x_);
    res_t_ = rnr_t_ - rho_ * (t_ - xi_t_);
    res_z1_ = rnr_z1_ - delta_ * (nu1_ - z1_);
    res_z2_ = rnr_z2_ - delta_ * (nu2_ - z2_);
    primal_prox_inf_ =
        delta_ * std::max((nu1_ - z1_).lpNorm<Eigen::Infinity>(),
                          (nu2_ - z2_).lpNorm<Eigen::Infinity>());
    if (m_ > 0) {
      res_y_ = rnr_y_ - delta_ * (nu_y_ - y_);
      primal_prox_inf_ =
          std::max(primal_prox_inf_,
                   delta_ * (nu_y_ - y_).lpNorm<Eigen::Infinity>());
    }
    dual_prox_inf_ = rho_ * std::max((x_ - xi_x_).lpNorm<Eigen::Infinity>(),
                                     (t_ - xi_t_).lpNorm<Eigen::Infinity>());
  }

  // Cold start (PIQP::solve_impl preamble): unit slacks, KKT solve with rhs
  // (-q, b, -penalty, -x_l, h_u) = (-q, b, -penalty, 0, h), then shift (s, z)
  // into the positive orthant.
  bool init_cold() {
    x_.setZero();
    t_.setZero();
    y_.setZero();
    s1_.setOnes();
    s2_.setOnes();
    z1_.setOnes();
    z2_.setOnes();

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
    z1_ = dz1_;
    z2_ = dz2_;
    s1_ = ds1_;
    s2_ = ds2_;

    const double delta_s =
        std::max(0.0, std::max(-s1_.minCoeff(), -s2_.minCoeff()));
    const double delta_z =
        std::max(0.0, std::max(-z1_.minCoeff(), -z2_.minCoeff()));
    s1_.array() += delta_s;
    s2_.array() += delta_s;
    z1_.array() += delta_z;
    z2_.array() += delta_z;

    const double mu_init = std::max(calculate_mu(), 1e-10);
    // Per-element shift onto the s*z = mu hyperbola (PIQP's sqrt correction).
    for (Eigen::Index i = 0; i < p_; ++i) {
      double c = z1_[i] - delta_z;
      z1_[i] = 0.5 * (c + std::sqrt(c * c + 4 * mu_init));
      s1_[i] = z1_[i] - c;
      c = z2_[i] - delta_z;
      z2_[i] = 0.5 * (c + std::sqrt(c * c + 4 * mu_init));
      s2_[i] = z2_[i] - c;
    }
    return true;
  }

  // Warm start: keep the previous iterate, but lift slacks and duals off the
  // boundary in proportion to how infeasible the old iterate is for the NEW
  // data. A converged iterate has s (or z) ~ 0 at active (or inactive)
  // constraints; if the problem then shifts by r, the Newton step needs
  // slack/dual moves of size ~ r, and any component sitting at ~0 truncates
  // the step length to ~0 -- the solver would stall re-centering itself.
  // Flooring at f ~ r keeps step lengths healthy while preserving the
  // active-set information; as r grows this degrades gracefully toward a
  // centered (cold-like) point. The equality dual y is free and carries
  // over unchanged. With Ruiz active the floor mixes frames (unscaled
  // residual, scaled slacks); equilibrated rows are O(1) so the mismatch
  // is bounded by the [min_floor, max_floor] clamp, and the floor is a
  // heuristic either way.
  void init_warm() {
    update_residuals_nr();  // residuals of the previous iterate, new data
    const double r = std::max(primal_res_, dual_res_);
    const double f =
        std::min(std::max(settings.warm_start_fraction * r,
                          settings.warm_start_min_floor),
                 settings.warm_start_max_floor);
    s1_ = s1_.cwiseMax(f);
    s2_ = s2_.cwiseMax(f);
    z1_ = z1_.cwiseMax(f);
    z2_ = z2_.cwiseMax(f);
  }

  // Internals index the two inequality blocks 1/2 to match PIQP's derivation
  // (W1, W2, Lambda); the returned certificate names them by block instead.
  const Solution& finish(Status status, int iters) {
    // Unscale into the user's frame (identity when Ruiz is off).
    sol_.x = x_.cwiseProduct(dx_s_);
    sol_.t = t_.cwiseProduct(inv_di_);
    sol_.y = y_.cwiseProduct(y_us_);
    sol_.s_t = s1_.cwiseProduct(inv_di_);
    sol_.s_ineq = s2_.cwiseProduct(inv_di_);
    sol_.z_t = z1_.cwiseProduct(z_us_);
    sol_.z_ineq = z2_.cwiseProduct(z_us_);
    sol_.status = status;
    sol_.converged = status == Status::kSolved ? 1 : 0;
    sol_.iters = iters;
    sol_.primal_obj = primal_obj_;
    sol_.primal_res = primal_res_;
    sol_.dual_res = dual_res_;
    sol_.duality_gap = duality_gap_;
    // A converged (or at least finite) iterate seeds the next warm start.
    have_warm_ = status != Status::kNumerics;
    return sol_;
  }

  // Problem data
  Eigen::Index n_ = 0, m_ = 0, p_ = 0;
  MatrixXd Q_, A_, G_;
  VectorXd q_, b_, h_, penalty_;
  MatrixXd AtA_;  // cached A^T A (lower triangle valid), only when m_ > 0

  // Iterates (persist across solves for warm starting)
  VectorXd x_, t_, y_, s1_, s2_, z1_, z2_;
  bool have_warm_ = false;
  bool explicit_warm_ = false;  // next solve seeded via set_warm_start()

  // Ruiz scaling state (identity when ruiz_ is false)
  bool ruiz_ = false;
  double c_s_ = 1.0;
  VectorXd dx_s_, de_s_, di_s_;         // cumulative scale factors
  VectorXd inv_cdx_, inv_de_, inv_di_;  // residual unscaling
  VectorXd y_us_, z_us_;                // dual unscaling (de/c, di/c)

  // Proximal centers (PIQP: xi for primal, nu for dual)
  VectorXd xi_x_, xi_t_, nu_y_, nu1_, nu2_;

  // Regularization state
  double rho_ = 0, delta_ = 0, reg_limit_ = 0, mu_ = 0;
  int factor_retries_ = 0, no_primal_update_ = 0, no_dual_update_ = 0;

  // Residuals (PIQP convention: negated KKT residuals)
  VectorXd rnr_x_, rnr_t_, rnr_y_, rnr_z1_, rnr_z2_;  // non-regularized
  VectorXd res_x_, res_t_, res_y_, res_z1_, res_z2_;  // regularized
  VectorXd res_s1_, res_s2_;                          // complementarity rhs
  double primal_res_ = 0, dual_res_ = 0;
  double prev_primal_res_ = 0, prev_dual_res_ = 0;
  double primal_res_rel_ = 0, dual_res_rel_ = 0;
  double primal_obj_ = 0, duality_gap_ = 0, duality_gap_rel_ = 0;
  double primal_prox_inf_ = 0, dual_prox_inf_ = 0;

  // KKT workspace (allocated in setup, reused every iteration)
  VectorXd w1_, w2_, w1_inv_, w2_inv_, d_inv_, lambda_;
  MatrixXd GS_, K_;
  Eigen::LLT<MatrixXd, Eigen::Lower> llt_;
  VectorXd rb1_, rb2_, wv_, pv_, rhs_x_, Gdx_;
  VectorXd dx_, dt_, dy_, ds1_, ds2_, dz1_, dz2_;
  VectorXd wQx_, wGtz2_, wGxt_, wAty_, wAx_;  // residual-evaluation buffers
  VectorXd zero_p_;

  Solution sol_;
};

// One-shot convenience wrappers (cold start).
inline Solution IpmSolve(
    const MatrixXd& Q, const VectorXd& q, const MatrixXd& A, const VectorXd& b,
    const MatrixXd& G, const VectorXd& h, const VectorXd& penalty,
    const IpmSettings& settings = {}) {
  IpmSolver solver;
  solver.settings = settings;
  solver.setup(Q, q, A, b, G, h, penalty);
  return solver.solve();
}

inline Solution IpmSolve(
    const MatrixXd& Q, const VectorXd& q, const MatrixXd& A, const VectorXd& b,
    const MatrixXd& G, const VectorXd& h, double penalty,
    const IpmSettings& settings = {}) {
  return IpmSolve(Q, q, A, b, G, h, VectorXd::Constant(h.size(), penalty),
                  settings);
}

// Inequality-only overloads.
inline Solution IpmSolve(
    const MatrixXd& Q, const VectorXd& q, const MatrixXd& G, const VectorXd& h,
    const VectorXd& penalty, const IpmSettings& settings = {}) {
  return IpmSolve(Q, q, MatrixXd(0, q.size()), VectorXd(0), G, h, penalty,
                  settings);
}

inline Solution IpmSolve(
    const MatrixXd& Q, const VectorXd& q, const MatrixXd& G, const VectorXd& h,
    double penalty, const IpmSettings& settings = {}) {
  return IpmSolve(Q, q, G, h, VectorXd::Constant(h.size(), penalty),
                  settings);
}

}  // namespace elastiqp
