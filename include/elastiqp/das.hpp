// ElastiQP-DAS: a dual active-set method for the elastic QP
//
//   minimize    0.5 x^T Q x + q^T x + penalty^T max(G x - h, 0)
//   subject to  A x == b                          (hard, dual y)
//
// i.e. the elastic QP of elastiqp/common.hpp, solved the way DAQP (Arnstrom,
// Bemporad, Axehill: "A dual active-set solver for embedded quadratic
// programming using recursive LDL' updates", 2022) solves the strict QP.
// Single header, Eigen only; DAQP's core logic without its MIQP /
// hierarchical / AVI / equality-elimination / codegen machinery. This is
// the default ElastiQP backend (see the paper for why).
//
// Why the elastic form fits a dual active-set method. With R'R = Q the
// strict QP is a least-distance problem (LDP) in u = R x + R^-T q,
//   minimize 0.5 |u|^2   s.t.  M u <= d,   M = C R^-1,  d = [b; h] + M v,
// whose dual is  max_{lam >= 0} -0.5 |M' lam|^2 - d' lam.  DAQP is a primal
// active-set method on that dual: the working set W holds the rows whose
// multipliers are free, the rest sit at 0. Relaxing row i with an l1 penalty
// w_i only adds the upper bound lam_i <= w_i to the dual, so the dual becomes
// a box-constrained QP and the active-set method gains a third state,
//
//   inactive    lam_i = 0          row out of W, exerts no force
//   active      0 < lam_i < w_i    row in W, multiplier from the LDL solve
//   saturated   lam_i = w_i        row out of W, exerts the fixed force w_i M_i'
//
// which is exactly ElastiQP's three-state classification. The saturated
// rows enter the working-set solve only through a fixed shift of u,
//   u = -M_W' lam_W - M_S' w_S,     (M_W M_W') lam_W = M_W u_S - d_W,
// so DAQP's recursive LDL' of the Gram matrix M_W M_W' and its add/remove
// updates are unchanged. A row enters W when its KKT condition fails: an
// inactive row that is violated (M_i u > d_i) or a saturated row that is
// strictly satisfied (M_i u < d_i, the multiplier wants to drop below w_i).
// The line search toward the working-set stationary point is blocked by
// either bound, and the blocking row leaves W to whichever bound it hit.
// Because every elastic multiplier is bounded, the dual can only be
// unbounded (primal infeasible) through the hard rows; with all rows
// elastic the method terminates with a solution for any data, like
// ElastiQP. Hard rows are penalty = inf; equality rows are free rows that
// never leave W.
//
// Outer proximal-point loop (DAQP's daqp_prox). The LDP needs Q > 0. When the
// Cholesky of Q finds a pivot below zero_tol times the largest one, Q is
// shifted to Q + eps I and the problem is solved as a proximal-point
// iteration, x_{k+1} = argmin{ elastic QP with Q + eps I, q - eps x_k }, warm
// started on the working set (only the LDP right-hand side changes between
// rounds, so the LDL' is reused), until eps |x_{k+1} - x_k|_inf <= eps_abs
// (or eta_prox when set), which bounds the stationarity error of the
// unshifted problem by that tolerance.
// Positive definite Q takes a single round. An LP (Q = 0) is handled by the
// same loop.
//
// Warm start across solve() calls keeps the working set, the saturated set
// and the multipliers. Factorization reuse: the Cholesky of Q survives any
// update that leaves Q alone; set_G / set_A diff their rows against the
// stored ones and only the changed rows are re-solved against R (one
// triangular solve each); the working-set LDL' is kept unless one of its
// rows changed; a q / h / b update touches only the LDP right-hand side.
// Rows of M are normalized to unit norm (DAQP's scaling) and the multiplier
// box becomes [0, w_i |M_i|], but tolerances are stated in USER units: a row
// enters the working set when its violation G_i x - h_i (or, for a saturated
// row, its slack) exceeds eps_abs + eps_rel |h_i|, and the proximal loop
// stops when the stationarity residual of the unshifted problem is below
// eps_abs too (eta_prox overrides it). Complementarity is then bounded by
// penalty_i * eps, which is what a user-unit constraint tolerance implies
// for an l1-elastic row.
//
// Robustness additions over the bare method (all on by default):
//  * Ruiz equilibration of the stacked (Q, A, G) system, ElastiQP's pass
//    (column and row factors 1/sqrt(max-norm), cost scaling, penalty scaled
//    with the rows), computed at setup and refreshed when a matrix update
//    leaves the scaled norms more than ruiz_refresh_ratio from 1. The LDP row
//    normalization sits on top of it; the column and cost scaling is what
//    conditions R and hence M = C R^-1.
//  * Equality consistency certificate: dependent equality rows are found when
//    the working set is built (singular pivot) and checked against the
//    least-norm solution of the independent ones before any iteration;
//    inconsistent data returns kInfeasible with eq_infeasibility() > 0.
//  * DAQP's cycle guard: the dual objective must increase (by progress_tol,
//    ABSOLUTE: the dual carries an offset of order |R^-T q|^2 that makes a
//    relative test call genuine ulp-sized gains stalls); stalls are counted
//    only when a row has left the working set since the last check (a cycle
//    needs a removal). After cycle_tol stalls the working set is refactored
//    from scratch in index order (a different pivot order); a second stall is
//    kNumerics. A pivot below refactor_tol at optimality triggers the same
//    refactor before the solution is accepted.
//  * Proximal escalation: when the inner method fails with kNumerics inside
//    the proximal loop (a degenerate vertex the guard cannot pass), the shift
//    is raised x100 -- up to prox_escalations times -- R, M and the working
//    set are rebuilt and the loop continues from the last dual-feasible
//    point. A larger shift changes the dual geometry and breaks the ties; the
//    price is more proximal rounds, paid only on the failure path.
//  * The equality certificate is withheld when its own least-norm solve is
//    not accurate to the tolerance on the KEPT rows (M = C R^-1 amplifies the
//    null-space components of the equality rows by 1/sqrt(eps), so a
//    well-posed A can give a Gram matrix with pivots near sing_tol); the
//    solve then proceeds with the dependent rows dropped.

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
  // Termination: a row may violate (or a saturated row fall short of) its
  // bound by eps_abs + eps_rel |h_i| in user units. Complementarity is then
  // bounded by penalty_i * eps, which is what a user-unit row tolerance
  // implies for an l1-elastic row.
  double eps_abs = 1e-6;
  double eps_rel = 0.0;
  double sing_tol = 3.7e-11; // LDL' pivot below which the working set is dependent
  double zero_tol = 1e-11;   // Cholesky pivot ratio below which Q is treated as singular
  double eps_prox = 1e-6;    // proximal shift, relative to max|Q_ii| (0: refuse singular Q)
  double eta_prox = 0.0;     // stationarity tolerance of the proximal fixed point, user
                             // units; <= 0 (default) follows eps_abs
  double prox_relaxation = 1.5;  // over-relaxation of the prox center when the working set is stable
  int prox_escalations = 3;  // x100 shifts tried after an inner numerics failure (0: fail at once)
  int max_iter = 10000;      // total active-set iterations per solve()
  int max_outer = 1000;      // proximal-point rounds per solve()
  bool warm_start = true;
  bool reuse_factorization = true;  // keep R, M columns and the working-set LDL' across updates
  // Ruiz equilibration (ElastiQP's pass) and its drift-gated refresh
  bool ruiz = true;
  int ruiz_max_iter = 10;
  double ruiz_tol = 1e-3;
  double ruiz_refresh_ratio = 4.0;  // 0: never refresh after setup
  // Equality consistency certificate before iterating
  bool check_eq_consistency = true;
  // Cycle guard (DAQP: progress_tol, cycle_tol, refactor_tol)
  double progress_tol = 1e-14;  // absolute dual increase that counts as progress
  int cycle_tol = 10;
  double refactor_tol = 1e-9;
};

class Solver {
 public:
  Settings settings;

  enum class RowState : unsigned char { kInactive, kActive, kSaturated, kEquality, kDropped };

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
    sol_.status = Status::kUnsolved;
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
  // set_A / set_G mark only the rows that actually changed.
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
    penalty_dirty_ = true;  // the multiplier box moves; rebuild the working set
  }

  const Solution& solution() const { return sol_; }

  // Seed the next solve() from a user-frame point (x, y, z), e.g. the
  // Solution of a previous solve of a nearby problem (by any backend) or
  // state carried through a functional (JAX) loop. The working set is read
  // off the three-state classification of z: 0 inactive, (0, penalty)
  // active with multiplier z, penalty saturated; rows within
  // settings.eps_abs of a bound are put on it. y seeds the equality
  // multipliers and x the proximal center. Dimensions must match setup().
  // Takes effect once, for the next solve() only, regardless of
  // settings.warm_start; unlike the implicit warm start it has no
  // factorization to reuse (the working-set LDL' is rebuilt).
  void set_warm_start(const VectorXd& x, const VectorXd& y, const VectorXd& z) {
    warm_x_ = x;
    warm_y_ = y;
    warm_z_ = z;
    explicit_warm_ = true;
  }

  RowState row_state(int i) const { return state_[static_cast<size_t>(m_ + i)]; }
  bool proximal() const { return eps_ > 0; }
  double prox_eps() const { return eps_; }
  // The last solve() had to escalate the proximal shift after an inner
  // numerics failure (see Settings::prox_escalations).
  bool prox_escalated() const { return prox_escalated_; }
  // Lower bound on the reachable equality residual (0 when consistent);
  // > 0 iff the last solve() returned kInfeasible for inconsistent equalities.
  double eq_infeasibility() const { return eq_infeas_; }
  // Largest deviation of the scaled system's column/row max-norms from 1.
  double scaling_drift() const { return drift_; }
  // Ruiz factors were recomputed during the last solve() (setup, or the
  // drift gate fired)
  bool rescaled() const { return rescaled_; }
  // Rows of [A; G] re-solved against R in the last solve() (m+p on a full
  // refactorization, 0 when only q/h/b changed).
  int rows_updated() const { return rows_updated_; }
  bool refactored() const { return refactored_; }
  // Cycle-guard / pivot refactorizations of the working set in the last solve()
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
      if (!update_rows(rebuild)) {  // Ruiz drift forced a full refresh
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
        hi_[m_ + i] = std::isfinite(ws_[i]) ? ws_[i] / scale_[m_ + i]
                                           : std::numeric_limits<double>::infinity();
      }
      penalty_dirty_ = false;
      rebuild = true;
    }
    if (rhs_dirty_) rescale_vectors();
    // Row tolerances in normalized units: mu_i = scale_i dr_i (C_i x - rhs_i).
    for (int i = 0; i < mp_; ++i)
      tol_[i] = (settings.eps_abs + settings.eps_rel * std::abs(rhs_[i])) * scale_[i] * dr_[i];
    if (explicit_warm_) {
      explicit_warm_ = false;
      seed_working_set();
      rebuild = true;
    } else if (!settings.warm_start || !have_solution_) {
      // Cold start: equalities only, nothing saturated, prox center at 0.
      for (int i = m_; i < mp_; ++i) state_[static_cast<size_t>(i)] = RowState::kInactive;
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
      if (st == Status::kNumerics && eps_ > 0 && escalations < settings.prox_escalations) {
        // The inner active-set method stalled (a degenerate vertex where
        // the dual no longer moves, e.g. QSHIP04S with 1454 of 1458 rows
        // active). A larger proximal shift changes the dual geometry and
        // breaks the ties: refactor Q + eps I with eps x 100, re-solve the
        // rows against the new R, rebuild the working set from the current
        // states and continue from the last dual-feasible point as the new
        // prox center. Only ever runs on a failure path.
        escalations++;
        prox_escalated_ = true;
        for (int i = 0; i < static_cast<int>(W_.size()); ++i)
          lam_full_[W_[static_cast<size_t>(i)]] = lam_[i];
        xc_ = llt_.matrixU().solve(u_ - v_);
        if (!factor(100.0 * eps_)) return finish(Status::kNumerics);
        for (int i = 0; i < mp_; ++i)  // row normalization changed with R
          tol_[i] = (settings.eps_abs + settings.eps_rel * std::abs(rhs_[i])) * scale_[i] * dr_[i];
        rebuild_working_set();
        center_relaxed = false;
        continue;
      }
      if (st != Status::kSolved) return finish(st);
      // x = R^-1 (u - v)
      x_ = llt_.matrixU().solve(u_ - v_);
      if (eps_ <= 0) break;
      // Stationarity residual of the unshifted problem, user units:
      // eps (xs - xc) / (c dx)
      const double diff = (x_ - xc_).cwiseQuotient(dx_).lpNorm<Eigen::Infinity>() / c_;
      const double eta = settings.eta_prox > 0 ? settings.eta_prox : settings.eps_abs;
      if (eps_ * diff <= eta) {
        if (center_relaxed) {  // confirm from the unrelaxed center
          center_relaxed = false;
          xc_ = x_;
          continue;
        }
        break;
      }
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

 private:
  // ---------------------------------------------------------------- setup
  // Cholesky of Q (shifted by eps I when Q is not numerically positive
  // definite), M' = R^-T C', row normalization and the multiplier boxes.
  bool factor(double eps_start = 0.0) {
    double scale = 0.0;
    for (int i = 0; i < n_; ++i) scale = std::max(scale, std::abs(Qs_(i, i)));
    eps_ = eps_start;
    for (int tries = 0; tries < 18; ++tries) {
      MatrixXd Qe = Qs_;
      if (eps_ > 0) Qe.diagonal().array() += eps_;
      llt_.compute(Qe);
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
    // M' = L^-1 C'  (column i is row i of M)
    Mt_ = llt_.matrixL().solve(Cts_);
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
      hi_[i] = std::isfinite(w) ? w * nrm : std::numeric_limits<double>::infinity();
    }
  }

  // set_A / set_G: store the rows that differ and mark them dirty.
  void set_rows(const MatrixXd& R, int offset) {
    for (int i = 0; i < R.rows(); ++i) {
      bool same = true;
      for (int k = 0; k < n_ && same; ++k) same = Ct_(k, offset + i) == R(i, k);
      if (same) continue;
      Ct_.col(offset + i) = R.row(i).transpose();
      col_dirty_[static_cast<size_t>(offset + i)] = 1;
    }
  }

  // Partial update: re-solve only the changed rows against the kept R.
  // Returns false if the Ruiz drift check demands a full refresh; sets
  // `rebuild` when a changed row sits in the working set.
  bool update_rows(bool& rebuild) {
    for (int i = 0; i < mp_; ++i) {
      if (!col_dirty_[static_cast<size_t>(i)]) continue;
      Cts_.col(i) = dr_[i] * dx_.cwiseProduct(Ct_.col(i));
    }
    if (settings.ruiz && settings.ruiz_refresh_ratio > 0) {
      VectorXd fx(n_), fr(mp_);
      scaling_pass(fx, fr);
      drift_ = ruiz_drift(fr, ruiz_drift(fx));
      if (drift_ > settings.ruiz_refresh_ratio) return false;
    }
    // One batched triangular solve for the changed rows (a per-column solve
    // is several times slower than Eigen's blocked multi-rhs solve).
    std::vector<int> changed;
    for (int i = 0; i < mp_; ++i)
      if (col_dirty_[static_cast<size_t>(i)]) changed.push_back(i);
    MatrixXd rhs(n_, static_cast<Eigen::Index>(changed.size()));
    for (size_t j = 0; j < changed.size(); ++j) rhs.col(static_cast<Eigen::Index>(j)) = Cts_.col(changed[j]);
    llt_.matrixL().solveInPlace(rhs);
    bool sat_changed = false;
    for (size_t j = 0; j < changed.size(); ++j) {
      const int i = changed[j];
      Mt_.col(i) = rhs.col(static_cast<Eigen::Index>(j));
      normalize_row(i);
      rows_updated_++;
      const RowState st = state_[static_cast<size_t>(i)];
      if (st == RowState::kActive || st == RowState::kEquality || st == RowState::kDropped) rebuild = true;
      if (st == RowState::kSaturated) sat_changed = true;
    }
    if (sat_changed && !rebuild) {
      uS_.setZero();
      for (int i = m_; i < mp_; ++i)
        if (state_[static_cast<size_t>(i)] == RowState::kSaturated) uS_ -= hi_[i] * Mt_.col(i);
    }
    return true;
  }

  // v = R^-T (q - eps x_center),  d = scale .* (rhs + M v)   (scaled frame)
  void form_rhs() {
    VectorXd qe = qs_;
    if (eps_ > 0) qe -= eps_ * xc_;
    v_ = llt_.matrixL().solve(qe);
    d_ = scale_.cwiseProduct(rhss_) + Mt_.transpose() * v_;
    rhs_dirty_ = false;
  }

  // ------------------------------------------------------------- scaling
  // Scaled frame: x = dx .* xs, rows scaled by dr, cost by c:
  //   Qs = c Dx Q Dx, qs = c Dx q, Cs = Dr C Dx, rhss = Dr rhs, ws = c w ./ dr
  // Multipliers unscale as lam = dr .* lam_s / c.
  // One Ruiz pass on the current scaled data (common.hpp primitives on the
  // transposed constraint block Cts_ = [A; G]' scaled); returns the largest
  // |1 - factor|.
  double scaling_pass(VectorXd& fx, VectorXd& fr) const {
    for (int k = 0; k < n_; ++k) fx[k] = Qs_.col(k).cwiseAbs().maxCoeff();
    fr.setZero();
    fold_max_abs(Cts_, fr, fx);
    return std::max(ruiz_factors(fx), ruiz_factors(fr));
  }

  // Apply the current factors to the user-frame matrices; recompute the
  // factors (Ruiz from the user frame) at setup or when the drift is large.
  void rescale_matrices() {
    apply_matrix_scaling();
    if (!settings.ruiz) return;
    bool refresh = !scaled_valid_;
    if (!refresh && settings.ruiz_refresh_ratio > 0) {
      VectorXd fx(n_), fr(mp_);
      scaling_pass(fx, fr);
      drift_ = ruiz_drift(fr, ruiz_drift(fx));
      refresh = drift_ > settings.ruiz_refresh_ratio;
    }
    if (refresh) {
      const VectorXd x_user = dx_.cwiseProduct(x_);  // keep the warm x across a rescale
      dx_.setOnes();
      dr_.setOnes();
      c_ = 1.0;
      apply_matrix_scaling();
      VectorXd fx(n_), fr(mp_);
      for (int it = 0; it < settings.ruiz_max_iter; ++it) {
        if (scaling_pass(fx, fr) <= settings.ruiz_tol) break;
        Qs_ = fx.asDiagonal() * Qs_ * fx.asDiagonal();
        Cts_ = fx.asDiagonal() * Cts_ * fr.asDiagonal();
        dx_ = dx_.cwiseProduct(fx);
        dr_ = dr_.cwiseProduct(fr);
        const double gamma = ruiz_cost_gamma(Qs_);
        Qs_ *= gamma;
        c_ *= gamma;
      }
      drift_ = 1.0;
      scaled_valid_ = true;
      rescaled_ = true;
      x_ = x_user.cwiseQuotient(dx_);
      for (int i = 0; i < p_; ++i) ws_[i] = c_ * penalty_[i] / dr_[m_ + i];
    }
    rhs_dirty_ = true;
  }
  void apply_matrix_scaling() {
    Qs_ = c_ * dx_.asDiagonal() * Q_ * dx_.asDiagonal();
    Cts_ = dx_.asDiagonal() * Ct_ * dr_.asDiagonal();
    ws_.resize(p_);
    for (int i = 0; i < p_; ++i) ws_[i] = c_ * penalty_[i] / dr_[m_ + i];
  }
  void rescale_vectors() {
    qs_ = c_ * dx_.cwiseProduct(q_);
    rhss_ = dr_.cwiseProduct(rhs_);
  }

  // ------------------------------------------------- equality certificate
  // The independent equality rows are the leading n_eq_ rows of W. Their
  // least-norm solution u_E = -M_E' lam, (M_E M_E') lam = -d_E, is the point
  // every equality-feasible u projects to; a dropped (dependent) row is
  // consistent iff it holds there.
  bool equalities_consistent() {
    const int k = n_eq_;
    for (int i = 0; i < k; ++i) work_[i] = -d_[W_[static_cast<size_t>(i)]];
    ldl_solve(k, work_, dir_);
    VectorXd uE = VectorXd::Zero(n_);
    for (int i = 0; i < k; ++i) uE -= dir_[i] * Mt_.col(W_[static_cast<size_t>(i)]);
    // Residuals in the user frame (rows were scaled by dr_i and by scale_i).
    // The kept rows hold at u_E up to the roundoff of the LDL' solve; when
    // that roundoff alone is above the tolerance the certificate cannot
    // tell inconsistency from conditioning and is not issued. This happens
    // under a small proximal shift: M = C R^-1 amplifies the null-space
    // components of the equality rows by 1/sqrt(eps), so a well-posed A can
    // have a Gram matrix with pivots near sing_tol (QSHELL: the kept rows'
    // residual is 1e-5 at eps = 1e-6, 1e-9 at eps = 1e-2).
    double worst = 0.0, worst_kept = 0.0;
    for (int i = 0; i < m_; ++i) {
      const double r = std::abs((Mt_.col(i).dot(uE) - d_[i]) / (scale_[i] * dr_[i]));
      if (state_[static_cast<size_t>(i)] == RowState::kDropped) worst = std::max(worst, r);
      else worst_kept = std::max(worst_kept, r);
    }
    const double tol = settings.eps_abs + settings.eps_rel * rhs_.head(m_).lpNorm<Eigen::Infinity>();
    if (worst_kept > tol || worst <= tol + worst_kept) {
      eq_infeas_ = 0.0;  // not certified
      return true;
    }
    eq_infeas_ = worst;
    return false;
  }

  // ---------------------------------------------------------- working set
  double lo(int row) const { return row < m_ ? -std::numeric_limits<double>::infinity() : 0.0; }

  // Append row `row` to W with multiplier `lam`, extending the LDL' of the
  // Gram matrix. Returns false if the new pivot is (numerically) zero, in
  // which case the row is in W with a singular last pivot and the caller
  // must resolve it (singular direction step or removal).
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

  // Recompute rows r.. of the LDL' from the Gram matrix; returns false if a
  // pivot is singular (only the last row can be, by construction).
  bool refactor_from(int r) {
    const int k = static_cast<int>(W_.size());
    bool ok = true;
    for (int i = r; i < k; ++i) {
      // Row i of L: solve L_{:i,:i} w = Gram_{i,:i}' (Gram is symmetric,
      // so read the contiguous column), L_{i,:i} = w ./ D.
      double dd = Gram_(i, i);
      if (i > 0) {
        work_.head(i) = Gram_.col(i).head(i);
        L_.topLeftCorner(i, i)
            .template triangularView<Eigen::UnitLower>()
            .solveInPlace(work_.head(i));
        L_.row(i).head(i) =
            work_.head(i).cwiseQuotient(D_.head(i)).transpose();
        dd -= L_.row(i).head(i).dot(work_.head(i));
      }
      D_[i] = dd;
      if (dd <= settings.sing_tol) ok = false;
    }
    return ok;
  }

  // Delete row r of W and its row/column of the LDL'. Dropping row r of
  // the Gram matrix leaves the trailing block's factor as
  // L33 D3 L33' + d_r l32 l32', a POSITIVE rank-one update of the (k-r-1)
  // trailing rows (Gill-Golub-Murray-Saunders 1974 method C1, DAQP's
  // remove_constraint): O((k-r)^2) against the O(k^3) of recomputing rows
  // r.. from the Gram matrix, which dominated cold solves with hundreds of
  // active rows (OSQP-suite SVM n=2020: 258 s vs DAQP's 10 s at the same
  // iteration count). The Gram matrix is still kept shifted so the full
  // refactorization remains the fallback when the update leaves a pivot
  // at or below sing_tol (dependent rows; the last-pivot-only invariant
  // the LDP relies on is then re-established by refactor_from).
  void remove_row(int r) {
    removed_ = true;
    const int k = static_cast<int>(W_.size());
    const int t = k - 1 - r;  // trailing rows after the deletion
    double alpha = D_[r];
    for (int i = 0; i < t; ++i) work_[i] = L_(r + 1 + i, r);  // l32
    for (int i = r; i + 1 < k; ++i) {
      W_[static_cast<size_t>(i)] = W_[static_cast<size_t>(i + 1)];
      lam_[i] = lam_[i + 1];
      D_[i] = D_[i + 1];
    }
    // Rows r+1.. of L move up, columns r+1.. move left (row-major order
    // of the writes never overwrites a source entry before it is read).
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

  void set_state(int row, RowState s) {
    RowState& cur = state_[static_cast<size_t>(row)];
    if (cur == RowState::kSaturated && s != RowState::kSaturated)
      uS_ += hi_[row] * Mt_.col(row);
    if (s == RowState::kSaturated && cur != RowState::kSaturated)
      uS_ -= hi_[row] * Mt_.col(row);
    cur = s;
  }

  // set_warm_start(): map the user-frame (x, y, z) onto the scaled iterate,
  // the working-set states and the normalized-row multipliers (the inverse
  // of finish()'s unscaling). Needs the current scaling and hi_, so it runs
  // in solve() after factor() / the penalty update.
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

  // Rebuild the LDL' for the warm (or cold) working set on the current M.
  void rebuild_working_set() {
    W_.clear();
    uS_.setZero();
    n_dropped_ = 0;
    // Equalities first, so a singular pivot among them means dependence.
    for (int i = 0; i < m_; ++i) {
      if (state_[static_cast<size_t>(i)] == RowState::kDropped) state_[static_cast<size_t>(i)] = RowState::kEquality;
      if (!add_row(i, lam_full_[i])) {
        // Dependent equality: consistent iff it holds at the least-norm
        // solution of the independent ones (equalities_consistent()).
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

  // Rebuild the LDL' of the current working set from scratch in row-index
  // order (a different pivot order than the entry order) and recompute the
  // saturated shift; DAQP's reset_daqp_workspace + daqp_activate_constraints.
  void refactor_working_set() {
    refactors_++;
    for (int i = 0; i < static_cast<int>(W_.size()); ++i)
      lam_full_[W_[static_cast<size_t>(i)]] = lam_[i];
    std::vector<int> rows(W_.begin(), W_.end());
    std::sort(rows.begin(), rows.end());
    W_.clear();
    n_eq_ = 0;
    for (int row : rows) {
      const bool eq = row < m_;
      if (!add_row(row, lam_full_[row])) {
        // Dependent in this order too: park it on the nearest bound.
        W_.pop_back();
        if (eq) {
          state_[static_cast<size_t>(row)] = RowState::kDropped;
          n_dropped_++;
        } else {
          const bool sat = std::isfinite(hi_[row]) && lam_full_[row] > 0.5 * hi_[row];
          state_[static_cast<size_t>(row)] = sat ? RowState::kSaturated : RowState::kInactive;
          lam_full_[row] = sat ? hi_[row] : 0.0;
        }
      } else if (eq) {
        n_eq_++;
      }
    }
    uS_.setZero();
    for (int i = m_; i < mp_; ++i)
      if (state_[static_cast<size_t>(i)] == RowState::kSaturated) uS_ -= hi_[i] * Mt_.col(i);
  }

  // ------------------------------------------------------------ inner LDP
  // (M_W M_W') lam* = M_W u_S - d_W
  void compute_csp() {
    const int k = static_cast<int>(W_.size());
    for (int i = 0; i < k; ++i) {
      const int row = W_[static_cast<size_t>(i)];
      work_[i] = Mt_.col(row).dot(uS_) - d_[row];
    }
    ldl_solve(k, work_, lam_star_);
  }

  // Solve (L D L') x = rhs on the leading k rows of the working-set factor
  // (Eigen's blocked triangular solves; L is column-major so both sweeps
  // are contiguous). rhs is overwritten.
  void ldl_solve(int k, VectorXd& rhs, VectorXd& x) {
    if (k == 0) return;
    const auto Lk = L_.topLeftCorner(k, k);
    Lk.template triangularView<Eigen::UnitLower>().solveInPlace(rhs.head(k));
    x.head(k) = rhs.head(k).cwiseQuotient(D_.head(k));
    Lk.transpose().template triangularView<Eigen::UnitUpper>().solveInPlace(
        x.head(k));
  }

  // Step from lam toward lam*, stopping at the first bound. Returns the
  // blocking index in W or -1 for a full step.
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
        if (a < alpha) { alpha = a; block = i; block_hi = false; }
      } else if (ls > hi_[row]) {
        const double a = (hi_[row] - li) / (ls - li);
        if (a < alpha) { alpha = a; block = i; block_hi = true; }
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

  // Working set is dependent through its last row (entered with sign
  // `sign`: +1 from inactive, -1 from saturated). Move along the null
  // direction of M_W' (dual increases linearly) until a bound blocks.
  Status singular_step(int sign) {
    const int k = static_cast<int>(W_.size()) - 1;
    // p_{:k} = -L_{:k,:k}^-T l,  p_k = 1
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
        if (a < alpha) { alpha = a; block = i; block_hi = true; }
      } else if (pi < 0 && std::isfinite(lo(row))) {
        const double a = (lam_[i] - lo(row)) / (-pi);
        if (a < alpha) { alpha = a; block = i; block_hi = false; }
      }
    }
    if (block < 0) return Status::kInfeasible;  // dual unbounded: hard rows conflict
    for (int i = 0; i <= k; ++i) lam_[i] += alpha * sign * dir_[i];
    const int row = W_[static_cast<size_t>(block)];
    set_state(row, block_hi ? RowState::kSaturated : RowState::kInactive);
    lam_full_[row] = block_hi ? hi_[row] : 0.0;
    remove_row(block);
    return Status::kSolved;
  }

  Status ldp(int& iters) {
    int singular_sign = 0;  // nonzero while the last row of W has a singular pivot
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
        singular_sign = (k > 0 && D_[k - 1] <= settings.sing_tol) ? singular_sign : 0;
        continue;
      }
      compute_csp();
      if (blocking_step() >= 0) continue;  // a row left W; re-solve
      // Dual feasible: u and the KKT check on the rows outside W.
      u_ = uS_;
      for (int i = 0; i < static_cast<int>(W_.size()); ++i)
        u_ -= lam_[i] * Mt_.col(W_[static_cast<size_t>(i)]);
      mu_ = Mt_.transpose() * u_ - d_;
      // Iterative refinement of the working-set solve: the residual of
      // (M_W M_W') lam = M_W u_S - d_W at the current lam is mu_W, which an
      // ill-conditioned Gram matrix (near-dependent active rows) leaves
      // well above the tolerance (DAQP: daqp_refine_active).
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
        mu_ = Mt_.transpose() * u_ - d_;
      }
      // Cycle guard: the dual objective -0.5|u|^2 - d'lam must increase.
      {
        double dual = -0.5 * u_.squaredNorm();
        for (int i = 0; i < static_cast<int>(W_.size()); ++i)
          dual -= d_[W_[static_cast<size_t>(i)]] * lam_[i];
        for (int i = m_; i < mp_; ++i)
          if (state_[static_cast<size_t>(i)] == RowState::kSaturated) dual -= d_[i] * hi_[i];
#ifdef ELASTIQP_DAS_DEBUG
        std::printf("it %d k %d dual %.17g best %.17g stalled %d\n", iters,
                    static_cast<int>(W_.size()), dual, best_dual, stalled);
#endif
        // Progress is measured in ABSOLUTE terms (DAQP's convention). The
        // dual carries an offset of order |R^-T q|^2 -- 1e10 on a
        // badly-scaled LP with a 1e-6 proximal shift -- against which a
        // relative tolerance turns genuine per-iteration gains (a few ulps,
        // e.g. QSTANDAT, QSCSD1, QSCTAP1 of Maros-Meszaros) into "stalls"
        // and aborts a converging solve as kNumerics. And a cycle needs a
        // removal: iterations that only grew the working set since the last
        // check cannot be part of one and are not counted.
        const bool progressed = !std::isfinite(best_dual) ||
            dual - best_dual > settings.progress_tol;
        const bool removed = removed_;
        removed_ = false;
        if (!progressed && removed) {
          if (++stalled > settings.cycle_tol) {
            if (tried_repair) {
#ifdef ELASTIQP_DAS_DEBUG
              std::printf("numerics: second stall (cycle) at it %d k %d\n", iters,
                          static_cast<int>(W_.size()));
#endif
              return Status::kNumerics;  // DAQP_EXIT_CYCLE
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
      // Most violated KKT condition in user units (mu_i / tol_i > 1).
      int add = -1;
      double worst = 1.0;
      for (int i = m_; i < mp_; ++i) {
        const RowState s = state_[static_cast<size_t>(i)];
        if (s == RowState::kInactive) {
          const double v = mu_[i] / tol_[i];
          if (v > worst) { worst = v; add = i; }
        } else if (s == RowState::kSaturated) {
          const double v = -mu_[i] / tol_[i];
          if (v > worst) { worst = v; add = i; }
        }
      }
      if (add < 0) {
        // Ill-conditioned LDL' at optimality: refactor once in a different
        // pivot order and re-verify (DAQP's refactor_tol repair).
        const int k = static_cast<int>(W_.size());
        double min_d = std::numeric_limits<double>::infinity();
        for (int i = 0; i < k; ++i) min_d = std::min(min_d, D_[i]);
        if (k > 2 && !tried_repair && min_d < settings.refactor_tol) {
          tried_repair = true;
          refactor_working_set();
          singular_sign = (static_cast<int>(W_.size()) > 0 &&
                           D_[W_.size() - 1] <= settings.sing_tol) ? 1 : 0;
          continue;
        }
        for (int i = 0; i < k; ++i)
          lam_full_[W_[static_cast<size_t>(i)]] = lam_[i];
        return Status::kSolved;
      }
      if (static_cast<int>(W_.size()) > n_) {
#ifdef ELASTIQP_DAS_DEBUG
        std::printf("numerics: working set larger than n at it %d (worst %.3g row %d)\n",
                    iters, worst, add);
#endif
        return Status::kNumerics;
      }
      const bool from_sat = state_[static_cast<size_t>(add)] == RowState::kSaturated;
      const double lam0 = from_sat ? hi_[add] : 0.0;
      set_state(add, RowState::kActive);
      if (!add_row(add, lam0)) singular_sign = from_sat ? -1 : 1;
    }
    return Status::kMaxIter;
  }

  // --------------------------------------------------------------- output
  const Solution& finish(Status st) {
    sol_.status = st;
    sol_.x = dx_.cwiseProduct(x_);  // user frame
    sol_.y.resize(m_);
    sol_.z.resize(p_);
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
      if (i < m_) sol_.y[i] = lam; else sol_.z[i - m_] = lam;
      if (i >= m_ && state_[static_cast<size_t>(i)] == RowState::kActive) sol_.n_active++;
      if (i >= m_ && state_[static_cast<size_t>(i)] == RowState::kSaturated) sol_.n_saturated++;
    }
    sol_.converged = st == Status::kSolved ? 1 : 0;
    have_solution_ = st == Status::kSolved;
    // Certificate in the user frame: t, z_t and the elastic KKT residuals
    // (one Q x, one C' [y; z], one C x).
    res_.noalias() = Ct_.transpose() * sol_.x - rhs_;  // [A x - b; G x - h]
    if (p_ > 0) {
      const auto r = res_.tail(p_);
      sol_.t = r.cwiseMax(0.0);
      sol_.z_t = penalty_ - sol_.z;
    } else {
      sol_.t.resize(0);
      sol_.z_t.resize(0);
    }
    wQx_.noalias() = Q_ * sol_.x;
    const double xQx = sol_.x.dot(wQx_);
    wQx_ += q_;
    if (m_ > 0) wQx_.noalias() += Ct_.leftCols(m_) * sol_.y;
    if (p_ > 0) wQx_.noalias() += Ct_.rightCols(p_) * sol_.z;
    sol_.dual_res = wQx_.lpNorm<Eigen::Infinity>();
    sol_.primal_res = m_ > 0 ? res_.head(m_).lpNorm<Eigen::Infinity>() : 0.0;
    sol_.primal_obj = 0.5 * xQx + q_.dot(sol_.x) +
                      (p_ > 0 ? penalty_.dot(sol_.t) : 0.0);
    double dual_obj = -0.5 * xQx;
    if (m_ > 0) dual_obj -= rhs_.head(m_).dot(sol_.y);
    if (p_ > 0) dual_obj -= rhs_.tail(p_).dot(sol_.z);
    sol_.duality_gap = std::abs(sol_.primal_obj - dual_obj);
    return sol_;
  }

  // Problem (user frame) and its Ruiz-scaled copy
  int n_ = 0, m_ = 0, p_ = 0, mp_ = 0;
  MatrixXd Q_, Ct_, Qs_, Cts_;
  VectorXd q_, rhs_, penalty_, qs_, rhss_, ws_, dx_, dr_;
  double c_ = 1.0, drift_ = 1.0, eq_infeas_ = 0.0;
  bool scaled_valid_ = false;
  int n_eq_ = 0, n_dropped_ = 0;
  // LDP
  Eigen::LLT<MatrixXd, Eigen::Upper> llt_;
  double eps_ = 0.0;
  MatrixXd Mt_;              // n x (m+p), normalized rows of M as columns
  VectorXd scale_, hi_, d_, v_, u_, uS_, mu_, x_, xc_, lam_full_;
  std::vector<RowState> state_;
  // Working set and LDL' of its Gram matrix
  std::vector<int> W_;
  MatrixXd L_, Gram_;
  VectorXd D_, lam_, lam_star_, dir_, work_;
  bool Q_dirty_ = true, penalty_dirty_ = false, rhs_dirty_ = true, have_solution_ = false;
  bool removed_ = false;  // a row left W since the last cycle-guard check
  bool prox_escalated_ = false;
  // set_warm_start() point (user frame), consumed by the next solve()
  bool explicit_warm_ = false;
  VectorXd warm_x_, warm_y_, warm_z_;
  std::vector<char> col_dirty_;
  VectorXd tol_;
  int rows_updated_ = 0, refactors_ = 0;
  bool refactored_ = false, rescaled_ = false;
  VectorXd res_, wQx_;  // finish() buffers
  Solution sol_;
};

// One-shot convenience wrappers (cold start).
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
// Inequality-only overloads.
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
