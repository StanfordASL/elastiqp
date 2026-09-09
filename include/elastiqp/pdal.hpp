// ElastiQP-PDAL: a primal-dual augmented Lagrangian method for the elastic QP
//
//   minimize    0.5 x^T Q x + q^T x + penalty^T t
//   subject to  A x == b          (hard, dual y)
//               G x - t <= h      (soft, slack s_ineq, dual z)
//               t >= 0            (slack s_t, dual z_t)
//
// Every inequality row is L1-elastic, so the problem is feasible iff
// Ax = b is consistent. The slacks are eliminated analytically: each
// multiplier lives in [0, penalty], and proxsuite's active-set test on the
// unclamped multiplier estimate gains a third state,
//
//   z~_i = z_prev_i + (G_i x - h_i) / mu_in
//   inactive    z~ < 0                 row out, dual = 0
//   active      0 <= z~ < penalty      identical to hard ProxQP
//   saturated   z~ >= penalty          row out, dual = penalty
//
// The method is proxsuite's dense PDAL (BCL outer loop, semismooth Newton
// with exact line search) condensed onto the n x n SPD system
//   K = Q + rho I + (1/mu_eq) A^T A + (1/mu_in) G_act^T G_act,
// whose factorization is cached across iterations and solve() calls.
// Elastic BCL changes: docs/elastic_bcl.md. Differentiation via relax():
// docs/pdal_differentiability.md.

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
  // Termination criteria
  // Note: recommend leaving check_duality_gap=true for ensuring
  // complementarity holds for the elastic problem
  double eps_abs = 1e-5;
  double eps_rel = 0;
  bool check_duality_gap = true;
  double eps_duality_gap_abs = 1e-5;
  double eps_duality_gap_rel = 0;
  int max_factor_retries = 10;

  // Incremental factorization updates on active set changes
  // rather than a full refactorization.
  bool incremental_updates = true;
  // Cap on incremental updates between refactorizations
  int incremental_update_budget = 256;
  // Number of active set changes allowable for incremental updates
  // More than this = just refactorize anyways. 0 means "auto" (=n/3)
  int incremental_update_max_flips = 0;

  // Check if equalities are inconsistent (the only infeasibility case).
  // Runs when A/b data is set; only applies if eps_rel = 0
  bool check_eq_consistency = true;

  // Warm-start from a previous solution + cached factorization
  bool warm_start = true;
  // Keep the AL penalties (mu_eq, mu_in) where the previous solve left
  // them instead of resetting them to *_init on a warm start, when the
  // cached factorization is still valid (vector-only drift): with an
  // unchanged active set the previous solve's factorization is then
  // reusable as is (no refactorization at all on a quiet tick), and the
  // first Newton step already enforces the hard equalities to ~mu_eq.
  // After a matrix update the penalties reset (a refactorization is due
  // anyway). Off = ProxQP behaviour.
  bool warm_keep_mu = false;
  // Thresholds on the kept penalties (warm_keep_mu): when the previous
  // solve ended with mu_eq or mu_in below these, both are reset to their
  // *_init values instead of being kept -- a solve that had to drive mu to
  // its floor (a conflict tick, a degenerate active set) must not hand a
  // floor-level penalty to the next tick, whose first inner loop would
  // then re-identify the active set at that stiffness one row at a time
  // and end at the floor again (a self-sustaining cascade). 0 = keep always.
  double warm_mu_eq_min = 0.0;
  double warm_mu_in_min = 0.0;
  // The same reset when the previous solve needed more than this many
  // inner iterations (a tick that struggled does not hand its penalties
  // on; a floor-level mu after a cheap tick is fine). 0 = no limit.
  int warm_keep_mu_max_iters = 0;

  // Iteration budget
  int max_outer_iter = 250;  // BCL rounds
  int max_iter_in = 1500;  // semismooth Newton steps

  // Proximal regularization and AL penalties (proxsuite defaults)
  double rho = 1e-6;
  double mu_eq_init = 1e-3;
  double mu_in_init = 1e-1;
  double mu_min_eq = 1e-9;
  double mu_min_in = 1e-8;
  double mu_update_factor = 0.1;

  // BCL outer-loop schedule (proxsuite defaults)
  double alpha_bcl = 0.1;
  double beta_bcl = 0.9;

  // ElastiQP BCL strategy (modified from proxsuite)
  // On a BCL bad step, revert the equality duals (only if their residual
  // is at fault); never revert the inequality duals
  bool bcl_split = true;
  // Saturation jump: on a stalled bad step, drop mu to the shallowest
  // saturation point among the stalled inequality rows
  bool bcl_saturation_jump = true;
  // Release jump: mirror for gap stalls, drop mu so the oversized dual on
  // the shallowest satisfied row releases to 0
  bool bcl_release_jump = true;
  // Fire the release jump when the gap's geometric decay would still need more
  // than this many rounds to pass the gap tolerance
  int bcl_release_jump_horizon = 4;
  // Warm-start eta seeding
  bool bcl_warm_eta = true;

  // Cold restart of over-tightened mu (proxsuite logic, mainly)
  // Disabled by default for ElastiQP (reset limit 0)
  double cold_reset_mu = 1.0 / 1.1;
  double cold_reset_threshold = 1e-5;
  double cold_reset_residual = 1e-5;
  int cold_reset_limit = 0;
  int safe_guard = 10000;  // total-Newton-iteration escape for BCL

  // Ruiz equilibration (at problem setup)
  bool ruiz = false;
  int ruiz_max_iter = 10;
  double ruiz_tol = 1e-3;
  // Automatic re-equilibration: the scaling computed at setup() is kept by
  // the set_* updates (it stays exact, only the conditioning drifts). When a
  // matrix update leaves some column/row max-norm of the scaled (Q, A, G)
  // more than this factor away from 1, solve() re-equilibrates first (see
  // reequilibrate(); the refactorization is already forced by the update).
  // 0 = never (manual reequilibrate() only).
  double ruiz_refresh_ratio = 4.0;

  // Regularization added to the relax() Newton system
  double relax_reg = 1e-9;

  // relax() / backward-pass warm-starting
  // Attempts for warm-starting relax() before restarting from the tight sol
  int relax_warm_budget = 15;
  // Skip warm relax() if more than these rows flip sides of the s.z = kappa
  // hyperbola between solves. Set negative to disable
  int relax_warm_flip_tol = 0;
};

class Solver {
 public:
  Settings settings;

  // Inequality row classification on the unclamped estimate z~ (see header)
  enum class RowState : unsigned char { kInactive, kActive, kSaturated };

  void setup(const MatrixXd& Q, const VectorXd& q, const MatrixXd& A,
             const VectorXd& b, const MatrixXd& G, const VectorXd& h,
             const VectorXd& penalty) {
    // Dimensions and problem data
    n_ = q.size();
    m_ = b.size();
    p_ = h.size();
    Q_ = 0.5 * (Q + Q.transpose());  // symmetrize
    q_ = q;
    A_ = A;
    b_ = b;
    G_ = G;
    h_ = h;
    penalty_ = penalty;

    // Reset warm-start, factorization-cache, and proximal state
    have_warm_ = false;
    relax_have_warm_ = false;
    explicit_warm_ = false;
    matrix_dirty_ = true;
    factored_ = false;
    rho_ = 0.0;
    mu_eq_ = 0.0;
    mu_in_ = 0.0;

    // Iterates and prox centers
    x_.resize(n_);
    y_.resize(m_);
    z_.resize(p_);
    xk_.resize(n_);
    yk_.resize(m_);
    zk_.resize(p_);

    // solve() workspace, preallocated
    ztilde_.resize(p_);
    t_.resize(p_);
    s2_.resize(p_);
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

    // Active-set state and line-search breakpoint buffer
    state_.assign(static_cast<size_t>(p_), RowState::kInactive);
    f_active_.assign(static_cast<size_t>(p_), false);
    bp_.clear();
    bp_.reserve(static_cast<size_t>(2 * p_));

    // Condensed KKT staging and its factorization
    GS_.resize(p_, n_);
    K_.resize(n_, n_);
    llt_ = Eigen::LLT<MatrixXd, Eigen::Lower>(n_);
    flip_idx_.reserve(static_cast<size_t>(p_));
    upd_vec_.resize(n_);
    updates_since_factor_ = 0;

    relax_ready_ = false;  // relax() workspace is allocated on first use

    // Equality-consistency certificate, on the still-unscaled (A_, b_)
    check_eq_A(A_);
    check_eq_b(b_);

    // Scaling state (identity unless Ruiz runs) and the A^T A cache
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

  // Data updates between solves
  // Vector-only updates keep the cached factorization
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

  // Recompute the Ruiz scaling for the current (Q, A, G) and rescale the
  // stored data and every warm-start iterate in place. A setup() would do
  // the same but discards the warm start (and redoes allocations, the A'A
  // cache, and the equality-consistency check). No-op when Ruiz is off, or
  // the data is still equilibrated. solve() calls this automatically when
  // the drift exceeds settings.ruiz_refresh_ratio.
  void reequilibrate() {
    if (!ruiz_) return;
    if (scaling_pass(dxw_, dew_, diw_) <= settings.ruiz_tol) return;
    const VectorXd dx0 = dx_s_, de0 = de_s_, di0 = di_s_;
    const double c0 = c_s_;
    // Undo the current scaling and equilibrate from identity, exactly as
    // setup() would. Continuing incrementally from the scaled data is NOT
    // equivalent: the max-norm fixed point of the constraint blocks is only
    // determined up to (E, D) -> (a E, D / a), and only Q pins the split. A
    // row that grew dominates its columns and pushes half its scale into
    // dx; when it shrinks back other rows dominate and nothing pushes it
    // out, so repeated refreshes leak the row scale into the column factors
    // without bound (scaled Q -> 0, tests/test_ruiz.cc).
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
    // Old scaled frame -> new scaled frame. Primal x scales like 1/dx,
    // t (and s) like di, duals like c/de and c/di.
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
    // relax() warm iterate: (z, s) pairs live in v = z - s with z.s = kappa_s
    if (relax_have_warm_) {
      xr_ = xr_.cwiseQuotient(dx);
      if (m_ > 0) yr_ = yr_.cwiseProduct(yf);
      tr_ = tr_.cwiseProduct(di);
      for (Eigen::Index i = 0; i < p_; ++i) {
        v1r_[i] = zf[i] * retraction(v1r_[i], relax_kappa_s_) -
                  di[i] * retraction(-v1r_[i], relax_kappa_s_);
        v2r_[i] = zf[i] * retraction(v2r_[i], relax_kappa_s_) -
                  di[i] * retraction(-v2r_[i], relax_kappa_s_);
      }
      relax_kappa_s_ *= gamma;
    }
    update_unscale_vectors();
    compute_AtA();
    matrix_dirty_ = true;
  }

  // How far the scaled (Q, A, G) has drifted from equilibrated: the largest
  // factor by which any column/row max-norm is off from 1 (1 = still
  // equilibrated, also when Ruiz is off).
  double scaling_drift() const {
    if (!ruiz_) return 1.0;
    scaling_pass(dxw_, dew_, diw_);
    return ruiz_drift(diw_, ruiz_drift(dew_, ruiz_drift(dxw_)));
  }

  // Explicitly set the warm-start for the next solve
  // Set rho/mu <= 0 to use the default settings
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

  // Number of KKT factorizations performed by the last solve()
  int factorizations() const { return factor_count_; }

  // Number of BCL cold resets performed by the last solve()
  int cold_resets() const { return cold_resets_; }

  // Lower bound on the reachable equality residual (0 if consistent or the
  // check is disabled)
  double eq_infeasibility() const { return eq_infeas_lb_; }

  const Solution& solve() {
    const bool explicit_ws = explicit_warm_;
    explicit_warm_ = false;
    factor_count_ = 0;
    iters_total_ = 0;
    outer_iters_ = 0;
    factor_retries_ = 0;
    cold_resets_ = 0;

    // A matrix update may have drifted the Ruiz scaling; the refactorization
    // it forces makes this the cheap moment to refresh
    if (ruiz_ && matrix_dirty_ && settings.ruiz_refresh_ratio > 0 &&
        scaling_drift() > settings.ruiz_refresh_ratio) {
      reequilibrate();
    }

    // Inconsistent equalities: eps_abs is unreachable, report and exit
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
      // Keep (x, y, z) and the cached factorization, reset AL penalties
      // (unless warm_keep_mu); z is re-clamped in case the penalty changed
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

    // BCL state (proxsuite defaults)
    eta_ext_ = eta_ext_init();
    eta_in_ = 1.0;

    // Initial termination check
    update_residuals();
    if (converged()) return finish(Status::kSolved);

    // If warm-starting, start eta_ext around the current residual level
    // But, only do so if it results in a 10x skip. Smaller skips might lead
    // to a bad step / mu-shrink that makes it no longer worth it
    const double eta_warm = 0.5 * primal_res_;
    if (warm_path && settings.bcl_warm_eta && eta_warm < 0.1 * eta_ext_) {
      eta_ext_ = eta_warm;
    }

    // Store initial residuals to track slow convergence across outer rounds
    const bool track_jump_res =
        settings.bcl_split && settings.bcl_saturation_jump;
    if (track_jump_res) {
      jump_res_prev_ = (r_ - t_).cwiseProduct(inv_di_);
    }
    double gap_prev = duality_gap_;

    // Outer loop (PMM + BCL)
    for (int oiter = 0; oiter < settings.max_outer_iter; ++oiter) {
      outer_iters_ = oiter + 1;
      const double pri_old = primal_res_;
      const double dua_old = dual_res_;

      // PMM: center the proximal terms on the current iterate and refresh
      // the unclamped multiplier estimate for the new center
      xk_ = x_;
      if (m_ > 0) yk_ = y_;
      zk_ = z_;
      wGx_.noalias() = G_ * x_;
      ztilde_ = zk_ + (wGx_ - h_) / mu_in_;

      // PMM: newton solve for subproblem
      if (!inner_loop(eta_in_)) {
        return finish(Status::kNumerics);
      }

      // Termination check after newton solve
      update_residuals();
      if (converged()) return finish(Status::kSolved);

      const double pri_new = primal_res_;
      const double dua_new = dual_res_;

      // Store residuals for convergence tracking for this round
      if (track_jump_res) {
        jump_res_cur_ = (r_ - t_).cwiseProduct(inv_di_);
      }

      // BCL: accept the step and tighten tolerances, or shrink mu on a bad step
      if (pri_new <= eta_ext_ || iters_total_ > settings.safe_guard) {
        eta_ext_ *= std::pow(mu_in_, settings.beta_bcl);
        eta_in_ = std::max(eta_in_ * mu_in_, eps_in_min());
        // Release jump: residuals pass but the gap stalls because oversized
        // duals on satisfied rows must come down, and every round counts as
        // good so mu never shrinks (docs/elastic_bcl.md). Jump one factor
        // past the shallowest such row so its dual snaps to 0 next step
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
        // Bad step (elastic): revert y only if the equalities are at fault,
        // never z; saturation jump: drop mu just far enough to saturate one
        // stalled row
        if (m_ > 0 && eq_res_ > eta_ext_) y_ = yk_;
        double mu_new = mu_in_ * settings.mu_update_factor;
        if (settings.bcl_saturation_jump && in_res_ > eta_ext_ &&
            pri_new > 0.8 * pri_old) {
          const double shallowest = saturation_jump_mu();
          if (shallowest > 0.0) mu_new = std::min(mu_new, shallowest);
        }
        shrink_mu(mu_new);
      } else {
        // Bad step (proxqp): revert both duals, shrink both mu
        if (m_ > 0) y_ = yk_;
        z_ = zk_;
        set_mu(mu_in_ * settings.mu_update_factor,
               mu_eq_ * settings.mu_update_factor);
      }

      // Keep track of residuals for the next round
      if (track_jump_res) jump_res_prev_.swap(jump_res_cur_);
      gap_prev = duality_gap_;

      // Cold restart of stalled, over-tightened penalties
      // Proxqp has this on, we keep this normally off (limit=0)
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

  // Backward pass: walk the tight solution to a kappa-relaxed central point
  // for smooth implicit differentiation (docs/pdal_differentiability.md).
  // The returned Solution is at the relaxed point
  const Solution& relax(double kappa, double tol = 1e-6, int max_iter = 50,
                        bool warm = true) {
    if (p_ == 0 || kappa <= 0.0 || (!have_warm_ && !relax_have_warm_)) {
      return sol_;
    }
    if (!relax_ready_) relax_alloc();
    // Ruiz-scaled kappa (the di factors cancel)
    const double kappa_s = c_s_ * kappa;

    // Warm-start from the previous relaxed iterate, unless too many rows are
    // predicted to flip sides of the s.z = kappa hyperbola (the prediction
    // needs a tight solve to compare against)
    bool use_warm = warm && relax_have_warm_;
    if (use_warm && have_warm_ && settings.relax_warm_flip_tol >= 0) {
      use_warm = relax_predict_flips(kappa_s) <= settings.relax_warm_flip_tol;
    }

    if (use_warm) {
      relax_run(kappa_s, tol, std::min(max_iter, settings.relax_warm_budget));
      // Stalled warm attempt: retry from the tight solution
      if (sol_.converged != 1 && have_warm_) {
        const int warm_iters = sol_.iters;
        relax_init_retraction();
        relax_run(kappa_s, tol, max_iter);
        sol_.iters += warm_iters;
      }
    } else {
      if (!have_warm_) return sol_;  // no tight solve to start from
      relax_init_retraction();
      relax_run(kappa_s, tol, max_iter);
    }
    relax_have_warm_ = sol_.converged == 1;
    relax_kappa_s_ = kappa_s;
    return sol_;
  }

 private:
  // ---- problem data: A'A cache, equality certificate, Ruiz scaling ----
  void compute_AtA() {
    if (m_ > 0) {
      AtA_.resize(n_, n_);
      AtA_.setZero();
      AtA_.selfadjointView<Eigen::Lower>().rankUpdate(A_.transpose());
    }
  }

  // Equality consistency (common.hpp EqCertificate) on the unscaled (A, b):
  // a lower bound on the reachable equality residual, tested in solve()
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

  // One Ruiz pass over the current (scaled) data: per-column and per-row
  // factors 1/sqrt(max-norm) of the stacked (Q, A, G) system, and the
  // largest deviation |1 - factor| from equilibrated (common.hpp)
  double scaling_pass(VectorXd& dx, VectorXd& de, VectorXd& di) const {
    for (Eigen::Index k = 0; k < n_; ++k) {
      dx[k] = Q_.col(k).cwiseAbs().maxCoeff();  // symmetric: col max = row max
    }
    de.setZero();
    di.setZero();
    if (m_ > 0) fold_max_abs(A_, dx, de);
    fold_max_abs(G_, dx, di);
    return std::max({ruiz_factors(dx), ruiz_factors(de), ruiz_factors(di)});
  }

  // Ruiz equilibration of the stacked (Q, A, G) system
  // proxqp ruiz logic + piqp limit_scaling + elastic penalty scaling.
  // Runs on the currently stored data and accumulates into the cumulative
  // factors; both setup() and reequilibrate() call it on unscaled data with
  // identity factors (see reequilibrate() for why not incrementally).
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
      penalty_ = penalty_.cwiseQuotient(di);  // elastiqp addition
      dx_s_ = dx_s_.cwiseProduct(dx);
      di_s_ = di_s_.cwiseProduct(di);
      const double gamma = ruiz_cost_gamma(Q_);
      Q_ *= gamma;
      q_ *= gamma;
      penalty_ *= gamma;  // elastiqp addition
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

  // ---- solve() entry paths ----

  // No inequality constraints: plain equality-constrained (or unconstrained)
  // QP, just solve the KKT system directly + report status from residuals
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

  // Cold start (proxsuite EQUALITY_CONSTRAINED_INITIAL_GUESS): solve
  // [Q+rho I, A'; A, -mu_eq I][x;y] = [-q; b] via K with an empty active
  // set; z = 0
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

  // ---- BCL outer-loop helpers ----
  double eta_ext_init() const { return std::pow(0.1, settings.alpha_bcl); }
  double eps_in_min() const { return std::min(settings.eps_abs, 1e-9); }

  // Set both AL penalties (floored), reset the BCL tolerances to the new
  // mu_in, and refresh the residuals (t depends on mu_in)
  void set_mu(double mu_in_new, double mu_eq_new) {
    mu_in_ = std::max(mu_in_new, settings.mu_min_in);
    mu_eq_ = std::max(mu_eq_new, settings.mu_min_eq);
    eta_ext_ = eta_ext_init() * std::pow(mu_in_, settings.alpha_bcl);
    eta_in_ = std::max(mu_in_, eps_in_min());
    update_residuals();
  }

  // Shrink mu_in to mu_new and mu_eq by the same ratio
  void shrink_mu(double mu_new) {
    set_mu(mu_new, mu_eq_ * (mu_new / mu_in_));
  }

  // Release-jump target: the mu_in at which the shallowest oversized dual on a
  // satisfied row (r < 0, z > 0) snaps to 0. Returns 0 if there is none
  double release_jump_mu() const {
    double shallowest = 0.0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      if (r_[i] < 0.0 && z_[i] > 0.0) {
        shallowest = std::max(shallowest, -r_[i] / z_[i]);
      }
    }
    return shallowest;
  }

  // Saturation-jump target: a violated row improving < 20% per round needs
  // mu_in <= r / (penalty - z) for its dual to reach the penalty cap.
  // Returns the shallowest such mu_in, or 0 if no row is stalled
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

  // True if the gap's per-round geometric decay cannot reach the gap
  // tolerance within bcl_release_jump_horizon more rounds
  bool gap_decay_too_slow(double gap_prev) const {
    if (duality_gap_ >= gap_prev || !(gap_prev > 0.0)) return true;
    const double decay =
        std::pow(duality_gap_ / gap_prev, settings.bcl_release_jump_horizon);
    return duality_gap_ * decay >= settings.eps_duality_gap_abs &&
           duality_gap_rel_ * decay >= settings.eps_duality_gap_rel;
  }

  // ---- inner semismooth Newton ----

  // Newton on the PDAL merit (proxsuite primal_dual_newton_semi_smooth).
  // Returns false only on a factorization disaster. Invariant on entry and
  // throughout: ztilde_ = zk_ + (Gx - h) / mu_in
  bool inner_loop(double eps_int) {
    for (int it = 0; it < settings.max_iter_in; ++it) {
      const double err = compute_inner_terms();
      // The dual mismatch in err is scaled by mu_in, so always take at least
      // one step; otherwise tiny mu_in can stall the outer loop
      if (err <= eps_int && it > 0) return true;

      // Three-state row classification on z~ (elastic version of
      // proxsuite's active-set test; ties mirror its >=)
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

      // Newton system condensed onto K. Each row's dual shift is toward its
      // target: z~ (active), penalty (saturated), 0 (inactive); non-active
      // rows leave the system and their shift is exact
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

      double dwmax = dx_.lpNorm<Eigen::Infinity>();
      if (m_ > 0) dwmax = std::max(dwmax, dy_.lpNorm<Eigen::Infinity>());
      dwmax = std::max(dwmax, dz_.lpNorm<Eigen::Infinity>());
      if (alpha * dwmax < 1e-11 && it > 0) return true;

      x_ += alpha * dx_;
      if (m_ > 0) y_ += alpha * dy_;
      z_ += alpha * dz_;
      // alpha is unclamped, so the step can push z outside [0, penalty].
      // So, project it back. Note: this never increases the merit
      // (merit z terms are separable quadratics with minimizers in range)
      clamp_z();
      ztilde_ += (alpha / mu_in_) * Gdx_;
      if (alpha == 0.0) return true;
    }
    return true;  // out of inner iterations; the outer loop adapts mu
  }

  // Inner stopping criterion (proxsuite compute_inner_loop_saddle_point
  // with the [0, penalty] clamp). Fills the buffers the Newton step reuses
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

  // Exact line search on the piecewise-quadratic PDAL merit (proxsuite
  // linesearch::primal_dual_ls): the derivative is piecewise affine in
  // alpha with breakpoints where z~_i(alpha) crosses 0 or penalty_i; scan
  // for its sign change and interpolate
  double line_search() {
    // Smooth part g(alpha) = b + a*alpha. The dual-coupling equality term
    // collapses via Adx - mu_eq*dy = -dyrhs
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

  bool ensure_factor() {
    const bool data_changed = !factored_ || matrix_dirty_ || f_rho_ != rho_ ||
                              f_mu_eq_ != mu_eq_ || f_mu_in_ != mu_in_;
    if (!data_changed) {
      flip_idx_.clear();
      for (Eigen::Index i = 0; i < p_; ++i) {
        if (is_active(i) != f_active(i)) flip_idx_.push_back(i);
      }
      if (flip_idx_.empty()) return true;

      // Only the active set changed: fold the flipped rows into the cached
      // factor as rank-one up/downdates when that beats a refactorization.
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
            ok = false;  // factor corrupted; fall through to refactorize
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

  // ---- relax() ----

  // Workspace for relax(), allocated on the first call after setup() so
  // forward-only users pay nothing for the backward pass
  void relax_alloc() {
    xr_.resize(n_);
    tr_.resize(p_);
    yr_.resize(m_);
    v1r_.resize(p_);
    v2r_.resize(p_);
    z1r_.resize(p_);
    z2r_.resize(p_);
    s1r_.resize(p_);
    s2r_.resize(p_);
    rf1_.resize(n_);
    rf2_.resize(p_);
    rf3_.resize(m_);
    rf4_.resize(p_);
    rf5_.resize(p_);
    d1r_.resize(p_);
    d2r_.resize(p_);
    einvr_.resize(p_);
    lamr_.resize(p_);
    wr_.resize(p_);
    pvr_.resize(p_);
    dxr_.resize(n_);
    dtr_.resize(p_);
    dyr_.resize(m_);
    dv1r_.resize(p_);
    dv2r_.resize(p_);
    llt_r_ = Eigen::LLT<MatrixXd, Eigen::Lower>(n_);
    relax_ready_ = true;
  }

  // Newton on the kappa-relaxed KKT from the current (xr, tr, yr, v1r, v2r)
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

      // Condensed Newton system. Eliminating dv1, dv2 (with b' in (0, 1)
      // and D = b'(v)/b'(-v) = z/s) and dt gives, per row,
      //   E = rho + D1 + D2,  Lambda = D2 (rho + D1) / E,
      //   E dt = D2 G dx - F2 + D1 F4 + D2 F5,
      // and the n x n SPD system
      //   [Q + rho I + (1/delta) A'A + G' diag(Lambda) G] dx = rhs.
      d1r_ = z1r_.cwiseQuotient(s1r_);
      d2r_ = z2r_.cwiseQuotient(s2r_);
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

      wr_ = d1r_.cwiseProduct(rf4_) + d2r_.cwiseProduct(rf5_) - rf2_;
      pvr_ = d2r_.cwiseProduct(rf5_ - einvr_.cwiseProduct(wr_));
      rhs_x_ = -rf1_;
      rhs_x_.noalias() -= G_.transpose() * pvr_;
      if (m_ > 0) {
        rhs_x_.noalias() -= (1.0 / delta) * (A_.transpose() * rf3_);
      }
      dxr_ = llt_r_.solve(rhs_x_);
      Gdx_.noalias() = G_ * dxr_;
      dtr_ = einvr_.cwiseProduct(d2r_.cwiseProduct(Gdx_) + wr_);
      if (m_ > 0) {
        dyr_.noalias() = A_ * dxr_;
        dyr_ += rf3_;
        dyr_ /= delta;
      }
      for (Eigen::Index i = 0; i < p_; ++i) {
        dv1r_[i] = (rf4_[i] - dtr_[i]) / retraction_dcomp(v1r_[i], kappa_s);
        dv2r_[i] = (rf5_[i] + Gdx_[i] - dtr_[i]) /
                   retraction_dcomp(v2r_[i], kappa_s);
      }

      // Backtrack on the 2-norm merit; terminate on max norm
      const double merit_prev = relax_merit_;
      xr_ += dxr_;
      tr_ += dtr_;
      if (m_ > 0) yr_ += dyr_;
      v1r_ += dv1r_;
      v2r_ += dv2r_;
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
        v1r_ -= alpha * dv1r_;
        v2r_ -= alpha * dv2r_;
        res_new = relax_residual(kappa_s);
      }
      res = res_new;
    }
    // The loop tests res before stepping, so classify the final step too
    if (status == Status::kMaxIter) {
      if (!std::isfinite(res)) {
        status = Status::kNumerics;
      } else if (res < tol) {
        status = Status::kSolved;
      }
    }
    relax_finish(status, iter);
  }


  // Closed-form prox of kappa*(-log): the positive root of
  // s^2 - v s - kappa = 0, i.e. b_k(v) = (v + sqrt(v^2 + 4 kappa))/2, with
  // the cancellation-free branch b_k(v) = 2 kappa / (sqrt(..) - v) for
  // v < 0 (docs/log_barrier_admm_note.tex).
  static double retraction(double v, double kappa) {
    const double r = std::sqrt(v * v + 4.0 * kappa);
    return v >= 0.0 ? 0.5 * (v + r) : 2.0 * kappa / (r - v);
  }

  // 1 - b_k'(v) = b_k'(-v) in (0, 1). b_k'(v) = (1 + v/r)/2 cancels for
  // |v| >> sqrt(kappa); the stable small branch is
  // (1 - |v|/r)/2 = 2 kappa / (r (r + |v|)).
  static double retraction_dcomp(double v, double kappa) {
    const double r = std::sqrt(v * v + 4.0 * kappa);
    const double small = 2.0 * kappa / (r * (r + std::abs(v)));
    return v >= 0.0 ? small : 1.0 - small;
  }

  // Relaxed-KKT residuals at (xr, tr, yr, v1r, v2r), with (z, s) pairs
  // materialized through the retraction so z.s = kappa holds identically:
  //   F1 = Q x + q + A'y + G'z2      F2 = penalty - z1 - z2
  //   F3 = A x - b                   F4 = s1 - t     F5 = s2 - (h + t - Gx)
  // Returns the unscaled max norm (termination) and fills relax_merit_,
  // the squared 2-norm (line search: Newton is descent for the 2-norm only)
  double relax_residual(double kappa_s) {
    for (Eigen::Index i = 0; i < p_; ++i) {
      z1r_[i] = retraction(v1r_[i], kappa_s);
      s1r_[i] = retraction(-v1r_[i], kappa_s);
      z2r_[i] = retraction(v2r_[i], kappa_s);
      s2r_[i] = retraction(-v2r_[i], kappa_s);
    }
    wQx_.noalias() = Q_ * xr_;
    wGtz_.noalias() = G_.transpose() * z2r_;
    rf1_ = wQx_ + q_ + wGtz_;
    if (m_ > 0) {
      wAty_.noalias() = A_.transpose() * yr_;
      rf1_ += wAty_;
      wAx_.noalias() = A_ * xr_;
      rf3_ = wAx_ - b_;
    }
    rf2_ = penalty_ - z1r_ - z2r_;
    wGx_.noalias() = G_ * xr_;
    rf4_ = s1r_ - tr_;
    rf5_ = s2r_ + wGx_ - h_ - tr_;
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

  // Rows whose retraction v-sign at the tight iterate disagrees with the
  // stored relaxed iterate's, ignoring pairs near the hyperbola corner
  // (|v_old v_new| <= 100 kappa_s). Predicts the cost of a warm relax()
  int relax_predict_flips(double kappa_s) {
    const double corner2 = 100.0 * kappa_s;
    wGx_.noalias() = G_ * x_;
    int flips = 0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      const RowRetraction rr = row_retraction(i, wGx_[i] - h_[i]);
      if ((rr.v1 > 0) != (v1r_[i] > 0) &&
          std::abs(rr.v1 * v1r_[i]) > corner2) {
        flips++;
      }
      if ((rr.v2 > 0) != (v2r_[i] > 0) &&
          std::abs(rr.v2 * v2r_[i]) > corner2) {
        flips++;
      }
    }
    return flips;
  }

  // Cold start for relax(): the tight iterate's elastic certificate mapped
  // through v = z - s
  void relax_init_retraction() {
    xr_ = x_;
    if (m_ > 0) yr_ = y_;
    wGx_.noalias() = G_ * x_;
    for (Eigen::Index i = 0; i < p_; ++i) {
      const RowRetraction rr = row_retraction(i, wGx_[i] - h_[i]);
      tr_[i] = rr.t;
      v1r_[i] = rr.v1;
      v2r_[i] = rr.v2;
    }
  }

  // Condensation scalings E^-1, Lambda for the current (D1, D2, rho), then
  // factor K = Q + rho I + (1/delta) A'A + G' diag(Lambda) G into llt_r_.
  // Same shape as factor_kkt(), but into a separate factorization (and
  // reusing the K_/GS_ staging buffers) so the solve() cache stays valid.
  bool relax_factor(double rho, double delta) {
    einvr_ = ((d1r_ + d2r_).array() + rho).cwiseInverse();
    lamr_ = d2r_.array() * (d1r_.array() + rho) * einvr_.array();
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

  // Unscaled certificate at the relaxed point. Only sol_ is written; the
  // solver iterate and warm-start state are untouched. The duality gap
  // converges to ~2 p kappa, not 0
  void relax_finish(Status status, int iters) {
    sol_.x = xr_.cwiseProduct(dx_s_);
    sol_.t = tr_.cwiseProduct(inv_di_);
    sol_.y = yr_.cwiseProduct(y_us_);
    sol_.z = z2r_.cwiseProduct(z_us_);
    sol_.z_t = z1r_.cwiseProduct(z_us_);
    sol_.status = status;
    sol_.converged = status == Status::kSolved ? 1 : 0;
    sol_.iters = iters;
    sol_.outer_iters = 0;
    count_row_states(sol_.t, sol_.z, settings.eps_abs, sol_.n_active,
                     sol_.n_saturated);
    wQx_.noalias() = Q_ * xr_;
    const double xQx = xr_.dot(wQx_);
    sol_.primal_obj = (0.5 * xQx + q_.dot(xr_) + penalty_.dot(tr_)) / c_s_;
    double dual_obj = (-0.5 * xQx - h_.dot(z2r_)) / c_s_;
    if (m_ > 0) dual_obj -= b_.dot(yr_) / c_s_;
    sol_.primal_res = relax_primal_res_;
    sol_.dual_res = relax_dual_res_;
    sol_.duality_gap = std::abs(sol_.primal_obj - dual_obj);
  }

  // ---- shared row/vector helpers ----

  // Keep the inequality duals in the bounded multiplier set [0, penalty]
  void clamp_z() { z_ = z_.cwiseMax(0.0).cwiseMin(penalty_); }

  // Elastic slack of row i with violation r = (Gx - h)_i at the current
  // mu_in: t = [r + mu_in (z - penalty)]_+ (argmin of the folded slack)
  double elastic_slack(Eigen::Index i, double r) const {
    return std::max(r + mu_in_ * (z_[i] - penalty_[i]), 0.0);
  }

  // Row i of the tight iterate mapped to the relax() retraction
  // coordinates v = z - s: v1 = z_t - s_t, v2 = z - s_ineq
  struct RowRetraction {
    double t, v1, v2;
  };
  RowRetraction row_retraction(Eigen::Index i, double r) const {
    const double t = elastic_slack(i, r);
    return {t, (penalty_[i] - z_[i]) - t, z_[i] - std::max(t - r, 0.0)};
  }

  // Unscaled inf-norm and squared 2-norm of a scaled-frame vector (or
  // expression) v, with componentwise unscaling factors s
  template <typename V>
  static double inf_us(const V& v, const VectorXd& s) {
    return v.size() > 0 ? v.cwiseAbs().cwiseProduct(s).maxCoeff() : 0.0;
  }
  template <typename V>
  static double ssq_us(const V& v, const VectorXd& s) {
    return v.size() > 0 ? v.cwiseProduct(s).squaredNorm() : 0.0;
  }

  // ---- residuals and termination ----

  // Residuals of the elastic QP at the reconstructed expanded point
  //   t = [Gx - h + mu_in (z - penalty)]_+
  //   z_t = penalty - z, s_t = t, s_ineq = [t - (Gx - h)]_+
  // The t-block dual residual vanishes identically. All norms are unscaled
  // componentwise, so termination is tested on the true elastic KKT
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
    double primal_rel_norm = std::max(
        {inf_us(wGx_ - t_, inv_di_), inf_us(h_, inv_di_),
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
    // Unscale into the user's frame (identity when Ruiz is off).
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
    // kInfeasible exits early without touching the iterate: keep the
    // warm-start state as it was.
    if (status != Status::kInfeasible) {
      have_warm_ = status != Status::kNumerics;
    }
    return sol_;
  }

  // ---- state ----

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
  // BCL tolerances for the current solve
  double eta_ext_ = 0, eta_in_ = 0;

  // Ruiz scaling state (identity when ruiz_ is false)
  bool ruiz_ = false;
  double c_s_ = 1.0;
  VectorXd dx_s_, de_s_, di_s_;             // cumulative scale factors
  VectorXd inv_cdx_, inv_de_, inv_di_;      // residual unscaling
  VectorXd y_us_, z_us_;                    // dual unscaling (de/c, di/c)
  mutable VectorXd dxw_, dew_, diw_;        // scaling_pass() workspace

  // Equality-consistency certificate (check_eq_A / check_eq_b)
  EqCertificate eq_cert_;
  double eq_infeas_lb_ = 0.0;

  // Factorization cache
  bool matrix_dirty_ = true, factored_ = false;
  double f_rho_ = 0, f_mu_eq_ = 0, f_mu_in_ = 0;
  std::vector<bool> f_active_;  // active set of the cached factor
  std::vector<Eigen::Index> flip_idx_;  // rows flipped since the cached factor
  VectorXd upd_vec_;                    // rank-one update staging
  int updates_since_factor_ = 0;
  int factor_retries_ = 0, factor_count_ = 0, iters_total_ = 0;
  int outer_iters_ = 0;
  int cold_resets_ = 0;
  int last_iters_ = 0;  // inner iterations of the previous solve()

  // Per-row active-set state and its accessors
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

  // Residual scalars
  double primal_res_ = 0, dual_res_ = 0;
  double in_res_ = 0, eq_res_ = 0;  // primal_res_ = max of these
  double primal_res_rel_ = 0, dual_res_rel_ = 0;
  double primal_obj_ = 0, duality_gap_ = 0, duality_gap_rel_ = 0;

  // Workspace (allocated in setup, reused every iteration)
  VectorXd ztilde_;  // unclamped multiplier estimate (proxsuite's S / mu_in)
  VectorXd t_, s2_, r_, dzs_;  // dzs_: per-row dual shift in the Newton step
  // Per-row inequality residuals across outer rounds, for the saturation
  // jump's stall gate (maintained only while the jump is enabled).
  VectorXd jump_res_prev_, jump_res_cur_;
  VectorXd verr_, dyrhs_, rhs_x_, dx_, dy_, dz_, Qdx_, Adx_, Gdx_;
  VectorXd wQx_, wGtz_, wGtd_, wAty_, wAx_, wGx_;
  MatrixXd GS_, K_;
  Eigen::LLT<MatrixXd, Eigen::Lower> llt_;
  std::vector<double> bp_;

  // relax() iterate and workspace (allocated in setup). Kept separate from
  // the solve() state so the relaxation never disturbs warm starting or the
  // factorization cache.
  VectorXd xr_, tr_, yr_, v1r_, v2r_;      // iterate (v parametrizes z, s)
  VectorXd z1r_, z2r_, s1r_, s2r_;         // retraction images of v
  VectorXd rf1_, rf2_, rf3_, rf4_, rf5_;   // relaxed-KKT residuals
  VectorXd d1r_, d2r_, einvr_, lamr_, wr_, pvr_;  // condensation scalings
  VectorXd dxr_, dtr_, dyr_, dv1r_, dv2r_;        // Newton step
  Eigen::LLT<MatrixXd, Eigen::Lower> llt_r_;
  double relax_primal_res_ = 0, relax_dual_res_ = 0;
  double relax_merit_ = 0;  // squared 2-norm of the relaxed-KKT residual
  // True while (xr_, tr_, yr_, v1r_, v2r_) holds a converged relaxed
  // iterate usable as the next relax() warm start; cleared by setup().
  bool relax_have_warm_ = false;
  double relax_kappa_s_ = 0;  // scaled kappa the warm iterate was solved at
  bool relax_ready_ = false;  // workspace sized for the current problem

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

}  // namespace elastiqp::pdal
