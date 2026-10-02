// Elastic dual active-set backend, based on DAQP

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#ifdef ELASTIQP_DAS_DEBUG
#include <cstdio>
#endif

#include "elastiqp/common.hpp"

namespace elastiqp::das {

struct Settings {
  double eps_abs = 1e-6;
  double eps_rel = 0.0;
  // Working-set LDL^T pivot below this = dependent row.
  double sing_tol = 3.7e-11;
  // Min/max Cholesky pivot ratio accepted for Q_s before adding a prox shift.
  double zero_tol = 1e-11;
  // Initial prox shift (x max|diag Q_s|) when Q_s is not PD; 0 disables.
  double eps_prox = 1e-6;
  // Outer-loop tolerance on eps*|x - xc|_inf; 0 -> eps_abs.
  double eta_prox = 0.0;
  // Over-relaxation of the prox center when the working set stopped changing;
  // <=1 disables.
  double prox_relaxation = 1.5;
  // Max 100x prox-shift increases after an inner numerical failure.
  int prox_escalations = 3;
  // Inner active-set iterations per outer iteration.
  int max_iter = 10000;
  int max_outer = 1000;
  bool warm_start = true;
  // Keep the Cholesky of Q_s across solves when only rows/vectors changed.
  bool reuse_factorization = true;

  bool ruiz = true;
  int ruiz_max_iter = 10;
  double ruiz_tol = 1e-3;
  // Re-equilibrate when scaling drift exceeds this; 0 = never.
  double ruiz_refresh_ratio = 4.0;

  // Return kInfeasible early on inconsistent equalities.
  bool check_eq_consistency = true;

  // Absolute dual-objective increase counted as progress by the cycle guard.
  double progress_tol = 1e-14;
  // Stalled iterations after a removal before repair / kNumerics.
  int cycle_tol = 10;
  // At optimality, refactor once if the smallest pivot is below this.
  double refactor_tol = 1e-9;

  // relax(): Newton regularization, and x100 retries on a failed factorization.
  double relax_reg = 1e-9;
  int relax_factor_retries = 10;

  double prox_tol() const { return eta_prox > 0 ? eta_prox : eps_abs; }
};

class Solver {
 public:
  Settings settings;

  // kSaturated: multiplier fixed at its penalty cap. kDropped: dependent
  // equality left out of the working set.
  enum class RowState : unsigned char {
    kInactive,
    kActive,
    kSaturated,
    kEquality,
    kDropped
  };

  void setup(const MatrixXd& Q, const VectorXd& q, const MatrixXd& A,
             const VectorXd& b, const MatrixXd& G, const VectorXd& h,
             const VectorXd& penalty) {
    n_ = static_cast<int>(q.size());
    m_ = static_cast<int>(b.size());
    p_ = static_cast<int>(h.size());
    mp_ = m_ + p_;
    Q_ = 0.5 * (Q + Q.transpose());
    q_ = q;
    Ct_.resize(n_, mp_);
    Ct_.leftCols(m_) = A.transpose();
    Ct_.rightCols(p_) = G.transpose();
    rhs_.resize(mp_);
    rhs_.head(m_) = b;
    rhs_.tail(p_) = h;
    penalty_ = penalty;
    dx_ = VectorXd::Ones(n_);
    dr_ = VectorXd::Ones(mp_);
    c_ = 1.0;
    scaled_valid_ = false;
    eq_infeas_ = 0.0;

    // Preallocate. Note: Eigen's blocked triangular solve may still allocate
    // on large problems, roughly, once n*(m+p)>16384.
    Qs_.resize(n_, n_);
    Cts_.resize(n_, mp_);
    llt_ = Eigen::LLT<MatrixXd, Eigen::Upper>(n_);
    qs_.resize(n_);
    rhss_.resize(mp_);
    ws_.resize(p_);
    fx_.resize(n_);
    fr_.resize(mp_);
    xu_.resize(n_);
    changed_.reserve(static_cast<size_t>(mp_));
    changed_cols_.resize(n_, mp_);
    warm_x_.setZero(n_);
    warm_y_.setZero(m_);
    warm_z_.setZero(p_);
    Mt_.resize(n_, mp_);
    scale_.resize(mp_);
    hi_.resize(mp_);
    d_.resize(mp_);
    v_.resize(n_);
    u_.resize(n_);
    uS_.setZero(n_);
    mu_.resize(mp_);
    x_.setZero(n_);
    xc_.setZero(n_);
    state_.assign(static_cast<size_t>(mp_), RowState::kInactive);
    for (int i = 0; i < m_; ++i) state_[i] = RowState::kEquality;
    lam_full_.setZero(mp_);

    const int kmax = n_ + 1;
    W_.clear();
    W_.reserve(static_cast<size_t>(kmax));
    rows_.reserve(static_cast<size_t>(kmax));
    L_.setZero(kmax, kmax);
    D_.setZero(kmax);
    Gram_.setZero(kmax, kmax);
    lam_.setZero(kmax);
    lam_star_.setZero(kmax);
    dir_.setZero(kmax);
    work_.setZero(kmax);

    Q_dirty_ = true;
    penalty_dirty_ = false;
    col_dirty_.assign(static_cast<size_t>(mp_), 1);
    rhs_dirty_ = true;
    have_solution_ = false;
    explicit_warm_ = false;
    eps_ = 0.0;
    tol_.resize(mp_);
    res_.resize(mp_);
    wQx_.resize(n_);
    sol_.x.setZero(n_);
    sol_.y.setZero(m_);
    sol_.z.setZero(p_);
    sol_.t.setZero(p_);
    sol_.z_t.setZero(p_);
    sol_.status = Status::kUnsolved;
    rw_.ready = false;
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

  int n() const { return n_; }
  int m() const { return m_; }
  int p() const { return p_; }

  void set_Q(const MatrixXd& Q) {
    Q_ = 0.5 * (Q + Q.transpose());
    Q_dirty_ = true;
  }
  void set_q(const VectorXd& q) {
    q_ = q;
    rhs_dirty_ = true;
  }
  void set_A(const MatrixXd& A) { set_rows(A, 0); }
  void set_b(const VectorXd& b) {
    rhs_.head(m_) = b;
    rhs_dirty_ = true;
  }
  void set_G(const MatrixXd& G) { set_rows(G, m_); }
  void set_h(const VectorXd& h) {
    rhs_.tail(p_) = h;
    rhs_dirty_ = true;
  }
  void set_penalty(const VectorXd& penalty) {
    penalty_ = penalty;
    penalty_dirty_ = true;
  }

  const Solution& solution() const { return sol_; }

  // Seeds states and multipliers from (x, y, z) on the next solve(), replacing
  // the internal warm start.
  void set_warm_start(const VectorXd& x, const VectorXd& y, const VectorXd& z) {
    warm_x_ = x;
    warm_y_ = y;
    warm_z_ = z;
    explicit_warm_ = true;
  }

  RowState row_state(int i) const {
    return state_[static_cast<size_t>(m_ + i)];
  }
  bool proximal() const { return eps_ > 0; }
  double prox_eps() const { return eps_; }
  bool prox_escalated() const { return prox_escalated_; }
  // Worst dropped-equality residual when the last solve returned kInfeasible.
  double eq_infeasibility() const { return eq_infeas_; }
  double scaling_drift() const { return drift_; }
  bool rescaled() const { return rescaled_; }
  int rows_updated() const { return rows_updated_; }
  bool refactored() const { return refactored_; }
  int refactors() const { return refactors_; }

  const Solution& solve() {
    sol_.iters = 0;
    sol_.outer_iters = 0;
    refactors_ = 0;
    eq_infeas_ = 0.0;
    rows_updated_ = 0;
    refactored_ = false;
    rescaled_ = false;
    bool rebuild = !settings.warm_start || !have_solution_;
    bool any_col = false;
    for (char c : col_dirty_) any_col |= c != 0;
    if (Q_dirty_ || !scaled_valid_ || !settings.reuse_factorization) {
      rescale_matrices();
      if (!factor()) return finish(Status::kNumerics);
      rebuild = true;
    } else if (any_col) {
      if (!update_rows(rebuild)) {
        rescale_matrices();
        if (!factor()) return finish(Status::kNumerics);
        rebuild = true;
      }
    }
    Q_dirty_ = false;
    std::fill(col_dirty_.begin(), col_dirty_.end(), 0);
    if (penalty_dirty_) {
      for (int i = 0; i < p_; ++i) {
        ws_[i] = c_ * penalty_[i] / dr_[m_ + i];
        hi_[m_ + i] = std::isfinite(ws_[i])
                          ? ws_[i] / scale_[m_ + i]
                          : std::numeric_limits<double>::infinity();
        // A saturated row turned hard has no cap left: active at its old one.
        RowState& s = state_[static_cast<size_t>(m_ + i)];
        if (s == RowState::kSaturated && !std::isfinite(ws_[i]))
          s = RowState::kActive;
      }
      penalty_dirty_ = false;
      rebuild = true;
    }
    if (rhs_dirty_) rescale_vectors();
    for (int i = 0; i < mp_; ++i)
      tol_[i] = (settings.eps_abs + settings.eps_rel * std::abs(rhs_[i])) *
                scale_[i] * dr_[i];
    if (explicit_warm_) {
      explicit_warm_ = false;
      seed_working_set();
      rebuild = true;
    } else if (!settings.warm_start || !have_solution_) {
      for (int i = m_; i < mp_; ++i)
        state_[static_cast<size_t>(i)] = RowState::kInactive;
      lam_full_.setZero();
      xc_.setZero();
    } else {
      xc_ = x_;
    }

    Status st = Status::kSolved;
    bool center_relaxed = false;
    int escalations = 0;
    prox_escalated_ = false;
    if (rebuild) rebuild_working_set();
    const bool rhs_changed = rhs_dirty_;
    for (int outer = 0; outer < settings.max_outer; ++outer) {
      sol_.outer_iters++;
      form_rhs();
      if (outer == 0 && n_dropped_ > 0 && (rebuild || rhs_changed) &&
          settings.check_eq_consistency && !equalities_consistent())
        return finish(Status::kInfeasible);
      int inner = 0;
      st = ldp(inner);
      sol_.iters += inner;
      if (st == Status::kNumerics && eps_ > 0 &&
          escalations < settings.prox_escalations) {
        // Inner loop broke down: recenter at the current x, grow the prox shift
        // 100x, restart.
        escalations++;
        prox_escalated_ = true;
        for (int i = 0; i < static_cast<int>(W_.size()); ++i)
          lam_full_[W_[static_cast<size_t>(i)]] = lam_[i];
        xc_ = llt_.matrixU().solve(u_ - v_);
        if (!factor(100.0 * eps_)) return finish(Status::kNumerics);
        for (int i = 0; i < mp_; ++i)
          tol_[i] = (settings.eps_abs + settings.eps_rel * std::abs(rhs_[i])) *
                    scale_[i] * dr_[i];
        rebuild_working_set();
        center_relaxed = false;
        continue;
      }
      if (st != Status::kSolved) return finish(st);
      x_ = llt_.matrixU().solve(u_ - v_);
      if (eps_ <= 0) break;
      // Outer loop converges when the prox center stops moving (user frame).
      const double diff =
          (x_ - xc_).cwiseQuotient(dx_).lpNorm<Eigen::Infinity>() / c_;
      if (eps_ * diff <= settings.prox_tol()) {
        if (center_relaxed) {
          center_relaxed = false;
          xc_ = x_;
          continue;
        }
        break;
      }
      // Working set unchanged: over-relax the center, then confirm with one
      // exact step.
      if (inner == 1 && settings.prox_relaxation > 1.0) {
        xc_ += settings.prox_relaxation * (x_ - xc_);
        center_relaxed = true;
      } else {
        xc_ = x_;
        center_relaxed = false;
      }
      if (outer + 1 == settings.max_outer) st = Status::kMaxIter;
    }
    return finish(st);
  }

  // Moves the solve() result to the kappa-relaxed central point (s.z = kappa
  // on both slack blocks; hard rows keep t = 0), the differentiation point read
  // by kkt_vjp.hpp. Newton on the smoothed KKT system, seeded from the working
  // set. Needs a solve() on the current data; leaves the working set and warm
  // start untouched. The first call after setup() allocates its workspace.
  const Solution& relax(double kappa, double tol = 1e-6, int max_iter = 50) {
    if (p_ == 0 || kappa <= 0.0) return sol_;
    bool stale = Q_dirty_ || penalty_dirty_ || rhs_dirty_;
    for (char c : col_dirty_) stale |= c != 0;
    if (stale && have_solution_) {
      sol_.status = Status::kUnsolved;
      sol_.converged = 0;
    }
    if (stale || !have_solution_) return sol_;
    if (!rw_.ready) relax_alloc();
    RelaxWork& w = rw_;
    const double kappa_s = c_ * kappa;
    tol = relax_tolerance(tol, kappa);
    const auto Gt = Cts_.rightCols(p_);
    const auto At = Cts_.leftCols(m_);
    relax_seed();
    if (m_ > 0) {
      w.AtA.setZero();
      w.AtA.selfadjointView<Eigen::Lower>().rankUpdate(At);
    }

    double rho = settings.relax_reg;
    double delta = settings.relax_reg;
    int iter = 0;
    Status status = Status::kMaxIter;
    double res = relax_residual(kappa_s);
    while (true) {
      if (!std::isfinite(res)) {
        status = Status::kNumerics;
        break;
      }
      if (res < tol) {
        status = Status::kSolved;
        break;
      }
      if (iter >= max_iter) break;
      iter++;

      // Eliminate (t, v_t, v_in) onto x; see relax_factor.
      for (int i = 0; i < p_; ++i) {
        w.d_in[i] = w.z_in[i] / w.s_in[i];
        w.d_t[i] = w.hard[static_cast<size_t>(i)] ? 0.0 : w.z_t[i] / w.s_t[i];
      }
      bool ok = relax_factor(rho, delta);
      for (int retries = 0; !ok && retries < settings.relax_factor_retries;
           ++retries) {
        rho *= 100;
        delta *= 100;
        ok = relax_factor(rho, delta);
      }
      if (!ok) {
        status = Status::kNumerics;
        break;
      }

      for (int i = 0; i < p_; ++i) {
        if (w.hard[static_cast<size_t>(i)]) {
          w.w[i] = 0.0;
          w.pv[i] = w.d_in[i] * w.f5[i];
        } else {
          w.w[i] = w.d_t[i] * w.f4[i] + w.d_in[i] * w.f5[i] - w.f2[i];
          w.pv[i] = w.d_in[i] * (w.f5[i] - w.einv[i] * w.w[i]);
        }
      }
      w.dx = -w.f1;
      w.dx.noalias() -= Gt * w.pv;
      if (m_ > 0) w.dx.noalias() -= (1.0 / delta) * (At * w.f3);
      w.llt.solveInPlace(w.dx);
      w.Gdx.noalias() = Gt.transpose() * w.dx;
      for (int i = 0; i < p_; ++i) {
        const bool hard = w.hard[static_cast<size_t>(i)];
        w.dt[i] = w.einv[i] * (w.d_in[i] * w.Gdx[i] + w.w[i]);
        w.dv_t[i] =
            hard ? 0.0
                 : (w.f4[i] - w.dt[i]) / retraction_dcomp(w.v_t[i], kappa_s);
        w.dv_in[i] = (w.f5[i] + w.Gdx[i] - w.dt[i]) /
                     retraction_dcomp(w.v_in[i], kappa_s);
      }
      if (m_ > 0) {
        w.dy.noalias() = At.transpose() * w.dx;
        w.dy += w.f3;
        w.dy /= delta;
      }

      // Backtrack on the 2-norm merit.
      const double merit_prev = w.merit;
      double alpha = 1.0;
      relax_step(alpha);
      res = relax_residual(kappa_s);
      for (int bt = 0;
           bt < 12 && !(std::isfinite(w.merit) && w.merit <= merit_prev);
           ++bt) {
        alpha *= 0.5;
        relax_step(-alpha);
        res = relax_residual(kappa_s);
      }
    }
    relax_finish(status, iter);
    return sol_;
  }

 private:
  // relax() workspace, in the scaled frame.
  struct RelaxWork {
    bool ready = false;
    std::vector<char> hard;  // penalty = inf: no elastic slack
    VectorXd x, t, y, v_t, v_in;
    VectorXd z_t, z_in, s_t, s_in;
    VectorXd f1, f2, f3, f4, f5;
    VectorXd d_t, d_in, einv, lam, w, pv, Gx, Gdx;
    VectorXd dx, dt, dy, dv_t, dv_in;
    MatrixXd K, GS, AtA;
    Eigen::LLT<MatrixXd, Eigen::Lower> llt;
    double primal_res = 0.0, dual_res = 0.0, merit = 0.0;
  };

  void relax_alloc() {
    RelaxWork& w = rw_;
    w.hard.resize(static_cast<size_t>(p_));
    w.x.resize(n_);
    w.y.resize(m_);
    for (VectorXd* v :
         {&w.t,  &w.v_t, &w.v_in, &w.z_t, &w.z_in, &w.s_t,  &w.s_in,
          &w.f2, &w.f4,  &w.f5,   &w.d_t, &w.d_in, &w.einv, &w.lam,
          &w.w,  &w.pv,  &w.Gx,   &w.Gdx, &w.dt,   &w.dv_t, &w.dv_in})
      v->setZero(p_);
    w.f1.resize(n_);
    w.f3.resize(m_);
    w.dx.resize(n_);
    w.dy.resize(m_);
    w.K.resize(n_, n_);
    w.GS.resize(n_, p_);
    w.AtA.resize(n_, n_);
    w.llt = Eigen::LLT<MatrixXd, Eigen::Lower>(n_);
    w.ready = true;
  }

  // Retraction coordinates v = z - s of the solve() point, duals read off the
  // row states.
  void relax_seed() {
    RelaxWork& w = rw_;
    w.x = x_;
    for (int i = 0; i < m_; ++i)
      w.y[i] = state_[static_cast<size_t>(i)] == RowState::kEquality
                   ? lam_full_[i] * scale_[i]
                   : 0.0;
    w.Gx.noalias() = Cts_.rightCols(p_).transpose() * x_;
    for (int i = 0; i < p_; ++i) {
      const int row = m_ + i;
      const double r = w.Gx[i] - rhss_[row];
      const double pen = ws_[i];
      const bool hard = !std::isfinite(pen);
      w.hard[static_cast<size_t>(i)] = hard;
      double z = 0.0;
      if (state_[static_cast<size_t>(row)] == RowState::kSaturated)
        z = pen;
      else if (state_[static_cast<size_t>(row)] == RowState::kActive)
        z = std::min(std::max(lam_full_[row] * scale_[row], 0.0), pen);
      const double t = hard ? 0.0 : std::max(r, 0.0);
      w.t[i] = t;
      w.v_t[i] = hard ? 0.0 : (pen - z) - t;
      w.v_in[i] = z - std::max(t - r, 0.0);
    }
  }

  // Smoothed-KKT residual in user units; fills f1..f5 (x-stationarity,
  // t-stationarity, equalities, t >= 0, G x - h <= t) in the scaled frame.
  double relax_residual(double kappa_s) {
    RelaxWork& w = rw_;
    const auto Gt = Cts_.rightCols(p_);
    for (int i = 0; i < p_; ++i) {
      w.z_in[i] = retraction(w.v_in[i], kappa_s);
      w.s_in[i] = retraction(-w.v_in[i], kappa_s);
      if (w.hard[static_cast<size_t>(i)]) continue;
      w.z_t[i] = retraction(w.v_t[i], kappa_s);
      w.s_t[i] = retraction(-w.v_t[i], kappa_s);
    }
    double dual = 0.0, primal = 0.0, merit = 0.0;
    const auto fold = [&merit](double& worst, double v) {
      worst = std::max(worst, std::abs(v));
      merit += v * v;
    };
    w.f1.noalias() = Qs_ * w.x;
    w.f1 += qs_;
    w.f1.noalias() += Gt * w.z_in;
    if (m_ > 0) {
      const auto At = Cts_.leftCols(m_);
      w.f1.noalias() += At * w.y;
      w.f3.noalias() = At.transpose() * w.x;
      w.f3 -= rhss_.head(m_);
      for (int i = 0; i < m_; ++i) fold(primal, w.f3[i] / dr_[i]);
    }
    for (int k = 0; k < n_; ++k) fold(dual, w.f1[k] / (c_ * dx_[k]));
    w.Gx.noalias() = Gt.transpose() * w.x;
    for (int i = 0; i < p_; ++i) {
      const double d = dr_[m_ + i];
      if (w.hard[static_cast<size_t>(i)]) {
        w.f2[i] = 0.0;
        w.f4[i] = 0.0;
        w.f5[i] = w.s_in[i] + w.Gx[i] - rhss_[m_ + i];
      } else {
        w.f2[i] = ws_[i] - w.z_t[i] - w.z_in[i];
        w.f4[i] = w.s_t[i] - w.t[i];
        w.f5[i] = w.s_in[i] + w.Gx[i] - rhss_[m_ + i] - w.t[i];
        fold(dual, w.f2[i] * d / c_);
        fold(primal, w.f4[i] / d);
      }
      fold(primal, w.f5[i] / d);
    }
    w.dual_res = dual;
    w.primal_res = primal;
    w.merit = merit;
    // std::max drops NaN; the sum of squares does not.
    return std::isfinite(merit) ? std::max(dual, primal) : merit;
  }

  // Reduced Newton matrix Q + rho I + G' Lam G + A'A / delta after eliminating
  // (t, v_t, v_in).
  bool relax_factor(double rho, double delta) {
    RelaxWork& w = rw_;
    for (int i = 0; i < p_; ++i) {
      if (w.hard[static_cast<size_t>(i)]) {
        w.einv[i] = 0.0;
        w.lam[i] = w.d_in[i];
      } else {
        w.einv[i] = 1.0 / (w.d_t[i] + w.d_in[i] + rho);
        w.lam[i] = w.d_in[i] * (w.d_t[i] + rho) * w.einv[i];
      }
    }
    w.GS.noalias() = Cts_.rightCols(p_) * w.lam.cwiseSqrt().asDiagonal();
    w.K.triangularView<Eigen::Lower>() = Qs_;
    w.K.diagonal().array() += rho;
    if (m_ > 0) w.K.triangularView<Eigen::Lower>() += (1.0 / delta) * w.AtA;
    w.K.selfadjointView<Eigen::Lower>().rankUpdate(w.GS);
    w.llt.compute(w.K);
    return w.llt.info() == Eigen::Success &&
           std::isfinite(w.K.diagonal().sum());
  }

  void relax_step(double alpha) {
    RelaxWork& w = rw_;
    w.x += alpha * w.dx;
    w.t += alpha * w.dt;
    if (m_ > 0) w.y += alpha * w.dy;
    w.v_t += alpha * w.dv_t;
    w.v_in += alpha * w.dv_in;
  }

  // Unscales the relaxed point into sol_; row-state counts stay those of the
  // solve().
  void relax_finish(Status status, int iters) {
    const RelaxWork& w = rw_;
    const double inf = std::numeric_limits<double>::infinity();
    sol_.x = dx_.cwiseProduct(w.x);
    for (int i = 0; i < m_; ++i) sol_.y[i] = w.y[i] * dr_[i] / c_;
    double pen_t = 0.0;
    for (int i = 0; i < p_; ++i) {
      const double d = dr_[m_ + i];
      const bool hard = w.hard[static_cast<size_t>(i)];
      sol_.z[i] = w.z_in[i] * d / c_;
      sol_.t[i] = hard ? 0.0 : w.t[i] / d;
      sol_.z_t[i] = hard ? inf : w.z_t[i] * d / c_;
      if (!hard) pen_t += penalty_[i] * sol_.t[i];
    }
    sol_.status = status;
    sol_.converged = status == Status::kSolved ? 1 : 0;
    sol_.iters = iters;
    sol_.outer_iters = 0;
    wQx_.noalias() = Q_ * sol_.x;
    const double xQx = sol_.x.dot(wQx_);
    sol_.primal_obj = 0.5 * xQx + q_.dot(sol_.x) + pen_t;
    double dual_obj = -0.5 * xQx - rhs_.tail(p_).dot(sol_.z);
    if (m_ > 0) dual_obj -= rhs_.head(m_).dot(sol_.y);
    sol_.duality_gap = std::abs(sol_.primal_obj - dual_obj);
    sol_.primal_res = w.primal_res;
    sol_.dual_res = w.dual_res;
  }

  // Cholesky of Q_s, doubling the prox shift until pivots are acceptable; then
  // M = L^{-1} C_s^T.
  bool factor(double eps_start = 0.0) {
    double scale = 0.0;
    for (int i = 0; i < n_; ++i) scale = std::max(scale, std::abs(Qs_(i, i)));
    eps_ = eps_start;
    for (int tries = 0; tries < 18; ++tries) {
      if (eps_ > 0)
        llt_.compute(Qs_ + eps_ * MatrixXd::Identity(n_, n_));
      else
        llt_.compute(Qs_);
      bool ok = llt_.info() == Eigen::Success;
      if (ok) {
        double pmin = std::numeric_limits<double>::infinity(), pmax = 0.0;
        for (int i = 0; i < n_; ++i) {
          const double piv = llt_.matrixLLT()(i, i);
          pmin = std::min(pmin, piv * piv);
          pmax = std::max(pmax, piv * piv);
        }
        ok = pmin > settings.zero_tol * pmax && pmin > 0;
      }
      if (ok) break;
      if (settings.eps_prox <= 0) return false;
      eps_ = eps_ > 0 ? 2.0 * eps_ : settings.eps_prox * std::max(1.0, scale);
      if (tries == 17) return false;
    }
    Mt_ = Cts_;
    llt_.matrixL().solveInPlace(Mt_);
    for (int i = 0; i < mp_; ++i) normalize_row(i);
    rows_updated_ = mp_;
    refactored_ = true;
    rhs_dirty_ = true;
    return true;
  }
  void normalize_row(int i) {
    const double nrm = Mt_.col(i).norm();
    scale_[i] = nrm > 1e-300 ? 1.0 / nrm : 1.0;
    Mt_.col(i) *= scale_[i];
    if (i < m_) {
      hi_[i] = std::numeric_limits<double>::infinity();
    } else {
      const double w = ws_[i - m_];
      hi_[i] =
          std::isfinite(w) ? w * nrm : std::numeric_limits<double>::infinity();
    }
  }

  // Marks only rows that actually changed.
  void set_rows(const MatrixXd& R, int offset) {
    for (int i = 0; i < R.rows(); ++i) {
      bool same = true;
      for (int k = 0; k < n_ && same; ++k) same = Ct_(k, offset + i) == R(i, k);
      if (same) continue;
      Ct_.col(offset + i) = R.row(i).transpose();
      col_dirty_[static_cast<size_t>(offset + i)] = 1;
    }
  }

  // Refreshes M columns of changed rows without refactoring Q_s; false if Ruiz
  // drift demands a full rescale.
  bool update_rows(bool& rebuild) {
    for (int i = 0; i < mp_; ++i) {
      if (!col_dirty_[static_cast<size_t>(i)]) continue;
      Cts_.col(i) = dr_[i] * dx_.cwiseProduct(Ct_.col(i));
    }
    if (settings.ruiz && settings.ruiz_refresh_ratio > 0) {
      scaling_pass(fx_, fr_);
      drift_ = ruiz_drift(fr_, ruiz_drift(fx_));
      if (drift_ > settings.ruiz_refresh_ratio) return false;
    }
    changed_.clear();
    for (int i = 0; i < mp_; ++i)
      if (col_dirty_[static_cast<size_t>(i)]) changed_.push_back(i);
    auto rhs =
        changed_cols_.leftCols(static_cast<Eigen::Index>(changed_.size()));
    for (size_t j = 0; j < changed_.size(); ++j)
      rhs.col(static_cast<Eigen::Index>(j)) = Cts_.col(changed_[j]);
    llt_.matrixL().solveInPlace(rhs);
    bool sat_changed = false;
    for (size_t j = 0; j < changed_.size(); ++j) {
      const int i = changed_[j];
      Mt_.col(i) = rhs.col(static_cast<Eigen::Index>(j));
      normalize_row(i);
      rows_updated_++;
      const RowState st = state_[static_cast<size_t>(i)];
      if (st == RowState::kActive || st == RowState::kEquality ||
          st == RowState::kDropped)
        rebuild = true;
      if (st == RowState::kSaturated) sat_changed = true;
    }
    if (sat_changed && !rebuild) {
      uS_.setZero();
      for (int i = m_; i < mp_; ++i)
        if (state_[static_cast<size_t>(i)] == RowState::kSaturated)
          uS_ -= hi_[i] * Mt_.col(i);
    }
    return true;
  }

  // Recomputes v and d after q, rhs, or the prox center change.
  void form_rhs() {
    v_ = qs_;
    if (eps_ > 0) v_ -= eps_ * xc_;
    llt_.matrixL().solveInPlace(v_);
    d_.noalias() = Mt_.transpose() * v_;
    d_ += scale_.cwiseProduct(rhss_);
    rhs_dirty_ = false;
  }

  // One Ruiz pass over the scaled matrices; returns how far they are from
  // equilibrated.
  double scaling_pass(VectorXd& fx, VectorXd& fr) const {
    for (int k = 0; k < n_; ++k) fx[k] = Qs_.col(k).cwiseAbs().maxCoeff();
    fr.setZero();
    fold_max_abs(Cts_, fr, fx);
    return std::max(ruiz_factors(fx), ruiz_factors(fr));
  }

  // Applies the current scaling; re-equilibrates from the user frame
  // (preserving x) when invalid or drifted.
  void rescale_matrices() {
    apply_matrix_scaling();
    if (!settings.ruiz) return;
    bool refresh = !scaled_valid_;
    if (!refresh && settings.ruiz_refresh_ratio > 0) {
      scaling_pass(fx_, fr_);
      drift_ = ruiz_drift(fr_, ruiz_drift(fx_));
      refresh = drift_ > settings.ruiz_refresh_ratio;
    }
    if (refresh) {
      // x in the user frame, kept across the rescale.
      xu_ = dx_.cwiseProduct(x_);
      dx_.setOnes();
      dr_.setOnes();
      c_ = 1.0;
      apply_matrix_scaling();
      for (int it = 0; it < settings.ruiz_max_iter; ++it) {
        if (scaling_pass(fx_, fr_) <= settings.ruiz_tol) break;
        Qs_ = fx_.asDiagonal() * Qs_ * fx_.asDiagonal();
        Cts_ = fx_.asDiagonal() * Cts_ * fr_.asDiagonal();
        dx_ = dx_.cwiseProduct(fx_);
        dr_ = dr_.cwiseProduct(fr_);
        const double gamma = ruiz_cost_gamma(Qs_);
        Qs_ *= gamma;
        c_ *= gamma;
      }
      drift_ = 1.0;
      scaled_valid_ = true;
      rescaled_ = true;
      x_ = xu_.cwiseQuotient(dx_);
      for (int i = 0; i < p_; ++i) ws_[i] = c_ * penalty_[i] / dr_[m_ + i];
    }
    rhs_dirty_ = true;
  }
  void apply_matrix_scaling() {
    Qs_ = c_ * dx_.asDiagonal() * Q_ * dx_.asDiagonal();
    Cts_ = dx_.asDiagonal() * Ct_ * dr_.asDiagonal();
    for (int i = 0; i < p_; ++i) ws_[i] = c_ * penalty_[i] / dr_[m_ + i];
  }
  void rescale_vectors() {
    qs_ = c_ * dx_.cwiseProduct(q_);
    rhss_ = dr_.cwiseProduct(rhs_);
  }

  // Solves the kept equalities and checks the dropped ones agree. Certifies
  // only if the kept rows fit to tolerance.
  bool equalities_consistent() {
    const int k = n_eq_;
    for (int i = 0; i < k; ++i) work_[i] = -d_[W_[static_cast<size_t>(i)]];
    ldl_solve(k, work_, dir_);
    xu_.setZero();  // u of the kept equalities alone
    for (int i = 0; i < k; ++i)
      xu_ -= dir_[i] * Mt_.col(W_[static_cast<size_t>(i)]);
    double worst = 0.0, worst_kept = 0.0;
    for (int i = 0; i < m_; ++i) {
      const double r =
          std::abs((Mt_.col(i).dot(xu_) - d_[i]) / (scale_[i] * dr_[i]));
      if (state_[static_cast<size_t>(i)] == RowState::kDropped)
        worst = std::max(worst, r);
      else
        worst_kept = std::max(worst_kept, r);
    }
    const double tol =
        settings.eps_abs +
        settings.eps_rel * rhs_.head(m_).lpNorm<Eigen::Infinity>();
    if (worst_kept > tol || worst <= tol + worst_kept) {
      eq_infeas_ = 0.0;
      return true;
    }
    eq_infeas_ = worst;
    return false;
  }

  // Multiplier lower bound: free for equalities, 0 for inequalities.
  double lo(int row) const {
    return row < m_ ? -std::numeric_limits<double>::infinity() : 0.0;
  }

  // Appends a row and extends the LDL^T of the working-set Gram matrix; false
  // if dependent.
  bool add_row(int row, double lam) {
    const int k = static_cast<int>(W_.size());
    for (int j = 0; j < k; ++j) {
      const double g = Mt_.col(W_[static_cast<size_t>(j)]).dot(Mt_.col(row));
      Gram_(k, j) = g;
      Gram_(j, k) = g;
    }
    Gram_(k, k) = Mt_.col(row).squaredNorm();
    W_.push_back(row);
    lam_[k] = lam;
    return refactor_from(k);
  }

  // Recomputes rows r.. of L D L^T = Gram from scratch.
  bool refactor_from(int r) {
    const int k = static_cast<int>(W_.size());
    bool ok = true;
    for (int i = r; i < k; ++i) {
      double dd = Gram_(i, i);
      if (i > 0) {
        work_.head(i) = Gram_.col(i).head(i);
        L_.topLeftCorner(i, i)
            .template triangularView<Eigen::UnitLower>()
            .solveInPlace(work_.head(i));
        L_.row(i).head(i) = work_.head(i).cwiseQuotient(D_.head(i)).transpose();
        dd -= L_.row(i).head(i).dot(work_.head(i));
      }
      D_[i] = dd;
      if (dd <= settings.sing_tol) ok = false;
    }
    return ok;
  }

  // Deletes working-set row r; rank-one update of the trailing block, falling
  // back to refactor_from.
  void remove_row(int r) {
    removed_ = true;
    const int k = static_cast<int>(W_.size());
    const int t = k - 1 - r;
    double alpha = D_[r];
    for (int i = 0; i < t; ++i) work_[i] = L_(r + 1 + i, r);
    for (int i = r; i + 1 < k; ++i) {
      W_[static_cast<size_t>(i)] = W_[static_cast<size_t>(i + 1)];
      lam_[i] = lam_[i + 1];
      D_[i] = D_[i + 1];
    }
    for (int i = r; i + 1 < k; ++i) {
      for (int j = 0; j < r; ++j) L_(i, j) = L_(i + 1, j);
      for (int j = r; j < i; ++j) L_(i, j) = L_(i + 1, j + 1);
    }
    for (int i = r; i + 1 < k; ++i)
      for (int j = 0; j < k; ++j) Gram_(i, j) = Gram_(i + 1, j);
    for (int j = r; j + 1 < k; ++j)
      for (int i = 0; i + 1 < k; ++i) Gram_(i, j) = Gram_(i, j + 1);
    W_.pop_back();
    bool ok = alpha > 0.0;
    for (int j = 0; ok && j < t; ++j) {
      const int jj = r + j;
      const double p = work_[j];
      const double dnew = D_[jj] + alpha * p * p;
      if (!(dnew > settings.sing_tol)) {
        ok = false;
        break;
      }
      const double beta = alpha * p / dnew;
      alpha *= D_[jj] / dnew;
      D_[jj] = dnew;
      for (int i = j + 1; i < t; ++i) {
        work_[i] -= p * L_(r + i, jj);
        L_(r + i, jj) += beta * work_[i];
      }
    }
    if (!ok) refactor_from(r);
  }

  // Keeps uS_ (saturated rows' contribution to u) in sync.
  void set_state(int row, RowState s) {
    RowState& cur = state_[static_cast<size_t>(row)];
    if (cur == RowState::kSaturated && s != RowState::kSaturated)
      uS_ += hi_[row] * Mt_.col(row);
    if (s == RowState::kSaturated && cur != RowState::kSaturated)
      uS_ -= hi_[row] * Mt_.col(row);
    cur = s;
  }

  // Classifies explicit warm-start duals into row states.
  void seed_working_set() {
    x_ = warm_x_.cwiseQuotient(dx_);
    xc_ = x_;
    const double tol = settings.eps_abs;
    for (int i = 0; i < mp_; ++i) {
      RowState& s = state_[static_cast<size_t>(i)];
      const double to_lam = c_ / (scale_[i] * dr_[i]);
      if (i < m_) {
        s = RowState::kEquality;
        lam_full_[i] = warm_y_[i] * to_lam;
        continue;
      }
      const double z = warm_z_[i - m_];
      if (std::isfinite(hi_[i]) && z >= penalty_[i - m_] - tol) {
        s = RowState::kSaturated;
        lam_full_[i] = hi_[i];
      } else if (z > tol) {
        s = RowState::kActive;
        lam_full_[i] = z * to_lam;
      } else {
        s = RowState::kInactive;
        lam_full_[i] = 0.0;
      }
    }
  }

  // Rebuilds W_ from row states: equalities first (dependent ones dropped),
  // then active rows (dependent or excess ones demoted).
  void rebuild_working_set() {
    W_.clear();
    uS_.setZero();
    n_dropped_ = 0;
    for (int i = 0; i < m_; ++i) {
      if (state_[static_cast<size_t>(i)] == RowState::kDropped)
        state_[static_cast<size_t>(i)] = RowState::kEquality;
      if (!add_row(i, lam_full_[i])) {
        W_.pop_back();
        state_[static_cast<size_t>(i)] = RowState::kDropped;
        n_dropped_++;
      }
    }
    n_eq_ = static_cast<int>(W_.size());
    for (int i = m_; i < mp_; ++i) {
      RowState& s = state_[static_cast<size_t>(i)];
      if (s == RowState::kSaturated) {
        uS_ -= hi_[i] * Mt_.col(i);
      } else if (s == RowState::kActive) {
        const double lam = std::min(std::max(lam_full_[i], 0.0), hi_[i]);
        if (!add_row(i, lam) || static_cast<int>(W_.size()) > n_) {
          W_.pop_back();
          s = RowState::kInactive;
          if (std::isfinite(hi_[i]) && lam > 0.5 * hi_[i]) {
            s = RowState::kSaturated;
            uS_ -= hi_[i] * Mt_.col(i);
          }
        }
      }
    }
  }

  // Repair: re-adds the current working set from scratch in row order.
  void refactor_working_set() {
    refactors_++;
    for (int i = 0; i < static_cast<int>(W_.size()); ++i)
      lam_full_[W_[static_cast<size_t>(i)]] = lam_[i];
    rows_.assign(W_.begin(), W_.end());
    std::sort(rows_.begin(), rows_.end());
    W_.clear();
    n_eq_ = 0;
    for (int row : rows_) {
      const bool eq = row < m_;
      if (!add_row(row, lam_full_[row])) {
        W_.pop_back();
        if (eq) {
          state_[static_cast<size_t>(row)] = RowState::kDropped;
          n_dropped_++;
        } else {
          const bool sat =
              std::isfinite(hi_[row]) && lam_full_[row] > 0.5 * hi_[row];
          state_[static_cast<size_t>(row)] =
              sat ? RowState::kSaturated : RowState::kInactive;
          lam_full_[row] = sat ? hi_[row] : 0.0;
        }
      } else if (eq) {
        n_eq_++;
      }
    }
    uS_.setZero();
    for (int i = m_; i < mp_; ++i)
      if (state_[static_cast<size_t>(i)] == RowState::kSaturated)
        uS_ -= hi_[i] * Mt_.col(i);
  }

  // Multipliers that make every working-set row tight.
  void compute_csp() {
    const int k = static_cast<int>(W_.size());
    for (int i = 0; i < k; ++i) {
      const int row = W_[static_cast<size_t>(i)];
      work_[i] = Mt_.col(row).dot(uS_) - d_[row];
    }
    ldl_solve(k, work_, lam_star_);
  }

  void ldl_solve(int k, VectorXd& rhs, VectorXd& x) {
    if (k == 0) return;
    const auto Lk = L_.topLeftCorner(k, k);
    Lk.template triangularView<Eigen::UnitLower>().solveInPlace(rhs.head(k));
    x.head(k) = rhs.head(k).cwiseQuotient(D_.head(k));
    Lk.transpose().template triangularView<Eigen::UnitUpper>().solveInPlace(
        x.head(k));
  }

  // Steps lam toward lam_star until a multiplier hits a bound; removes that row
  // and returns its index, -1 if none.
  int blocking_step() {
    const int k = static_cast<int>(W_.size());
    double alpha = 1.0;
    int block = -1;
    bool block_hi = false;
    for (int i = 0; i < k; ++i) {
      const int row = W_[static_cast<size_t>(i)];
      const double li = lam_[i], ls = lam_star_[i];
      if (ls < lo(row)) {
        const double a = (li - lo(row)) / (li - ls);
        if (a < alpha) {
          alpha = a;
          block = i;
          block_hi = false;
        }
      } else if (ls > hi_[row]) {
        const double a = (hi_[row] - li) / (ls - li);
        if (a < alpha) {
          alpha = a;
          block = i;
          block_hi = true;
        }
      }
    }
    if (block < 0) {
      lam_.head(k) = lam_star_.head(k);
      return -1;
    }
    for (int i = 0; i < k; ++i) lam_[i] += alpha * (lam_star_[i] - lam_[i]);
    const int row = W_[static_cast<size_t>(block)];
    set_state(row, block_hi ? RowState::kSaturated : RowState::kInactive);
    lam_full_[row] = block_hi ? hi_[row] : 0.0;
    remove_row(block);
    return block;
  }

  // Dependent working set: move multipliers along the null direction until one
  // hits a bound. No blocker = infeasible LDP.
  Status singular_step(int sign) {
    const int k = static_cast<int>(W_.size()) - 1;
    if (k > 0) {
      dir_.head(k) = -L_.row(k).head(k).transpose();
      L_.topLeftCorner(k, k)
          .transpose()
          .template triangularView<Eigen::UnitUpper>()
          .solveInPlace(dir_.head(k));
    }
    dir_[k] = 1.0;
    double alpha = std::numeric_limits<double>::infinity();
    int block = -1;
    bool block_hi = false;
    for (int i = 0; i <= k; ++i) {
      const int row = W_[static_cast<size_t>(i)];
      const double pi = sign * dir_[i];
      if (pi > 0 && std::isfinite(hi_[row])) {
        const double a = (hi_[row] - lam_[i]) / pi;
        if (a < alpha) {
          alpha = a;
          block = i;
          block_hi = true;
        }
      } else if (pi < 0 && std::isfinite(lo(row))) {
        const double a = (lam_[i] - lo(row)) / (-pi);
        if (a < alpha) {
          alpha = a;
          block = i;
          block_hi = false;
        }
      }
    }
    if (block < 0) return Status::kInfeasible;
    for (int i = 0; i <= k; ++i) lam_[i] += alpha * sign * dir_[i];
    const int row = W_[static_cast<size_t>(block)];
    set_state(row, block_hi ? RowState::kSaturated : RowState::kInactive);
    lam_full_[row] = block_hi ? hi_[row] : 0.0;
    remove_row(block);
    return Status::kSolved;
  }

  Status ldp(int& iters) {
    // +1: new row came from inactive (lam rising from 0); -1: from saturated
    // (falling from hi).
    int singular_sign = 0;
    const int k0 = static_cast<int>(W_.size());
    if (k0 > 0 && D_[k0 - 1] <= settings.sing_tol) singular_sign = 1;
    double best_dual = -std::numeric_limits<double>::infinity();
    int stalled = 0;
    bool tried_repair = false;
    removed_ = false;
    for (iters = 1; iters < settings.max_iter; ++iters) {
      if (singular_sign != 0) {
        const Status st = singular_step(singular_sign);
        if (st != Status::kSolved) return st;
        const int k = static_cast<int>(W_.size());
        singular_sign =
            (k > 0 && D_[k - 1] <= settings.sing_tol) ? singular_sign : 0;
        continue;
      }
      compute_csp();
      if (blocking_step() >= 0) continue;
      // u and scaled residuals mu at the current multipliers.
      u_ = uS_;
      for (int i = 0; i < static_cast<int>(W_.size()); ++i)
        u_ -= lam_[i] * Mt_.col(W_[static_cast<size_t>(i)]);
      mu_.noalias() = Mt_.transpose() * u_;
      mu_ -= d_;
      // Up to two refinement solves so working-set rows are tight to 0.1 tol.
      for (int round = 0; round < 2; ++round) {
        const int k = static_cast<int>(W_.size());
        double worst_w = 0.0;
        for (int i = 0; i < k; ++i) {
          const int row = W_[static_cast<size_t>(i)];
          worst_w = std::max(worst_w, std::abs(mu_[row]) / tol_[row]);
        }
        if (worst_w <= 0.1) break;
        for (int i = 0; i < k; ++i) work_[i] = mu_[W_[static_cast<size_t>(i)]];
        ldl_solve(k, work_, dir_);
        for (int i = 0; i < k; ++i) {
          lam_[i] += dir_[i];
          u_ -= dir_[i] * Mt_.col(W_[static_cast<size_t>(i)]);
        }
        mu_.noalias() = Mt_.transpose() * u_;
        mu_ -= d_;
      }
      {
        double dual = -0.5 * u_.squaredNorm();
        for (int i = 0; i < static_cast<int>(W_.size()); ++i)
          dual -= d_[W_[static_cast<size_t>(i)]] * lam_[i];
        for (int i = m_; i < mp_; ++i)
          if (state_[static_cast<size_t>(i)] == RowState::kSaturated)
            dual -= d_[i] * hi_[i];
#ifdef ELASTIQP_DAS_DEBUG
        std::printf("it %d k %d dual %.17g best %.17g stalled %d\n", iters,
                    static_cast<int>(W_.size()), dual, best_dual, stalled);
#endif
        // Cycle guard: no dual progress since a removal -> refactor once, then
        // give up.
        const bool progressed = !std::isfinite(best_dual) ||
                                dual - best_dual > settings.progress_tol;
        const bool removed = removed_;
        removed_ = false;
        if (!progressed && removed) {
          if (++stalled > settings.cycle_tol) {
            if (tried_repair) {
#ifdef ELASTIQP_DAS_DEBUG
              std::printf("numerics: second stall (cycle) at it %d k %d\n",
                          iters, static_cast<int>(W_.size()));
#endif
              return Status::kNumerics;
            }
            tried_repair = true;
            refactor_working_set();
            stalled = 0;
            best_dual = -std::numeric_limits<double>::infinity();
            const int k = static_cast<int>(W_.size());
            singular_sign = (k > 0 && D_[k - 1] <= settings.sing_tol) ? 1 : 0;
            continue;
          }
        } else if (progressed) {
          best_dual = dual;
          stalled = 0;
        }
      }
      // Most violated inactive row, or saturated row whose constraint has gone
      // slack.
      int add = -1;
      double worst = 1.0;
      for (int i = m_; i < mp_; ++i) {
        const RowState s = state_[static_cast<size_t>(i)];
        if (s == RowState::kInactive) {
          const double v = mu_[i] / tol_[i];
          if (v > worst) {
            worst = v;
            add = i;
          }
        } else if (s == RowState::kSaturated) {
          const double v = -mu_[i] / tol_[i];
          if (v > worst) {
            worst = v;
            add = i;
          }
        }
      }
      if (add < 0) {
        // Optimal. Refactor once first if the factorization looks degraded.
        const int k = static_cast<int>(W_.size());
        double min_d = std::numeric_limits<double>::infinity();
        for (int i = 0; i < k; ++i) min_d = std::min(min_d, D_[i]);
        if (k > 2 && !tried_repair && min_d < settings.refactor_tol) {
          tried_repair = true;
          refactor_working_set();
          singular_sign = (static_cast<int>(W_.size()) > 0 &&
                           D_[W_.size() - 1] <= settings.sing_tol)
                              ? 1
                              : 0;
          continue;
        }
        for (int i = 0; i < k; ++i)
          lam_full_[W_[static_cast<size_t>(i)]] = lam_[i];
        return Status::kSolved;
      }
      if (static_cast<int>(W_.size()) > n_) {
#ifdef ELASTIQP_DAS_DEBUG
        std::printf(
            "numerics: working set larger than n at it %d (worst %.3g row "
            "%d)\n",
            iters, worst, add);
#endif
        return Status::kNumerics;
      }
      const bool from_sat =
          state_[static_cast<size_t>(add)] == RowState::kSaturated;
      const double lam0 = from_sat ? hi_[add] : 0.0;
      set_state(add, RowState::kActive);
      if (!add_row(add, lam0)) singular_sign = from_sat ? -1 : 1;
    }
    return Status::kMaxIter;
  }

  // Unscales the solution and computes residuals and objectives in the user
  // frame.
  const Solution& finish(Status st) {
    sol_.status = st;
    sol_.x = dx_.cwiseProduct(x_);
    sol_.n_active = sol_.n_saturated = 0;
    for (int i = 0; i < mp_; ++i) {
      double lam = 0.0;
      switch (state_[static_cast<size_t>(i)]) {
        case RowState::kActive:
        case RowState::kEquality:
          lam = lam_full_[i] * scale_[i] * dr_[i] / c_;
          break;
        case RowState::kSaturated:
          lam = penalty_[i - m_];
          break;
        default:
          break;
      }
      if (i < m_)
        sol_.y[i] = lam;
      else
        sol_.z[i - m_] = lam;
      if (i >= m_ && state_[static_cast<size_t>(i)] == RowState::kActive)
        sol_.n_active++;
      if (i >= m_ && state_[static_cast<size_t>(i)] == RowState::kSaturated)
        sol_.n_saturated++;
    }
    sol_.converged = st == Status::kSolved ? 1 : 0;
    have_solution_ = st == Status::kSolved;
    res_.noalias() = Ct_.transpose() * sol_.x;
    res_ -= rhs_;
    if (p_ > 0) {
      const auto r = res_.tail(p_);
      sol_.t = r.cwiseMax(0.0);
      sol_.z_t = penalty_ - sol_.z;
    }
    wQx_.noalias() = Q_ * sol_.x;
    const double xQx = sol_.x.dot(wQx_);
    wQx_ += q_;
    if (m_ > 0) wQx_.noalias() += Ct_.leftCols(m_) * sol_.y;
    if (p_ > 0) wQx_.noalias() += Ct_.rightCols(p_) * sol_.z;
    sol_.dual_res = wQx_.lpNorm<Eigen::Infinity>();
    sol_.primal_res = m_ > 0 ? res_.head(m_).lpNorm<Eigen::Infinity>() : 0.0;
    sol_.primal_obj =
        0.5 * xQx + q_.dot(sol_.x) + (p_ > 0 ? penalty_.dot(sol_.t) : 0.0);
    double dual_obj = -0.5 * xQx;
    if (m_ > 0) dual_obj -= rhs_.head(m_).dot(sol_.y);
    if (p_ > 0) dual_obj -= rhs_.tail(p_).dot(sol_.z);
    sol_.duality_gap = std::abs(sol_.primal_obj - dual_obj);
    return sol_;
  }

  // Problem data (user frame) and Ruiz scaling: x = dx .* x_s, rows scaled by
  // dr, cost by c.
  int n_ = 0, m_ = 0, p_ = 0, mp_ = 0;
  MatrixXd Q_, Ct_, Qs_, Cts_;
  VectorXd q_, rhs_, penalty_, qs_, rhss_, ws_, dx_, dr_;
  double c_ = 1.0, drift_ = 1.0, eq_infeas_ = 0.0;
  bool scaled_valid_ = false;
  int n_eq_ = 0, n_dropped_ = 0;
  // Cholesky of Q_s and LDP data in the whitened frame; Mt_ = M.
  Eigen::LLT<MatrixXd, Eigen::Upper> llt_;
  double eps_ = 0.0;
  MatrixXd Mt_;
  VectorXd scale_, hi_, d_, v_, u_, uS_, mu_, x_, xc_, lam_full_;
  // Scratch: Ruiz factors, a user-frame / unscaled n-vector, the columns
  // update_rows re-solves, and the rows refactor_working_set re-adds.
  VectorXd fx_, fr_, xu_;
  MatrixXd changed_cols_;
  std::vector<int> changed_, rows_;
  std::vector<RowState> state_;
  // Working set and LDL^T of its Gram matrix.
  std::vector<int> W_;
  MatrixXd L_, Gram_;
  VectorXd D_, lam_, lam_star_, dir_, work_;
  // Dirty flags and per-solve diagnostics.
  bool Q_dirty_ = true, penalty_dirty_ = false, rhs_dirty_ = true,
       have_solution_ = false;
  bool removed_ = false;
  bool prox_escalated_ = false;
  bool explicit_warm_ = false;
  VectorXd warm_x_, warm_y_, warm_z_;
  std::vector<char> col_dirty_;
  VectorXd tol_;
  int rows_updated_ = 0, refactors_ = 0;
  bool refactored_ = false, rescaled_ = false;
  VectorXd res_, wQx_;
  RelaxWork rw_;
  Solution sol_;
};

inline Solution Solve(const MatrixXd& Q, const VectorXd& q, const MatrixXd& A,
                      const VectorXd& b, const MatrixXd& G, const VectorXd& h,
                      const VectorXd& penalty, const Settings& settings = {}) {
  Solver s;
  s.settings = settings;
  s.setup(Q, q, A, b, G, h, penalty);
  return s.solve();
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

}  // namespace elastiqp::das
