// ElastiQP: an elastic QP solver for robot control.
//
//   minimize    0.5 x^T Q x + q^T x + penalty^T t
//   subject to  A x == b          (hard, dual y)
//               G x - t <= h      (soft, slack s_ineq, dual z_ineq)
//               t >= 0            (slack s_t, dual z_t)
//
// Every inequality gets its own L1-penalized slack t_i. As such, the
// inequalities cannot cause infeasibility, and the problem is feasible
// iff the equality constraints are consistent. The slacks are handled
// analytically rather than added as explicit decision variables.
//
// The solver is a primal-dual augmented Lagrangian (PDAL) method based on
// ProxQP, adapted to the elastic form above.
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
// (proxsuite's PrimalLDLT shape). The factorization is cached across Newton
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
//
// Differentiability: relax(kappa) walks the converged solution to a
// kappa-relaxed central point (complementarity s.z = kappa) for smooth
// implicit differentiation, using a log-barrier retraction

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace elastiqp {

using Eigen::MatrixXd;
using Eigen::VectorXd;

enum class Status {
  kUnsolved = 0,
  kSolved = 1,
  kMaxIter = 2,
  kNumerics = 3,
};

// The elastic KKT certificate. One slack/dual pair per constraint block:
// _t for the bound t >= 0, _ineq for the elastic rows Gx - t <= h. Both
// pairs have length p; y (length m) is the equality dual. At a solution
// z_t = penalty - z_ineq, so z_ineq in [0, penalty] -- the bounded
// multiplier that makes the L1 penalty exact.
struct Solution {
  VectorXd x;
  VectorXd t;               // per-constraint elastic slacks (= violations)
  VectorXd y;               // equality duals
  VectorXd s_t, s_ineq;     // slacks for t >= 0 and Gx - t <= h
  VectorXd z_t, z_ineq;     // duals for t >= 0 and Gx - t <= h
  Status status = Status::kUnsolved;
  int converged = 0;  // 1 iff status == kSolved
  int iters = 0;      // total inner semismooth Newton steps
  double primal_obj = 0.0;
  double primal_res = 0.0;
  double dual_res = 0.0;
  double duality_gap = 0.0;
};

// Primal-dual augmented Lagrangian settings -- every knob the solver has.
// The 1e-5 accuracy target matches proxsuite's default and is sized for
// control: on the robot-control benchmarks, warm-started solves deliver
// ~1e-6 KKT residuals at this setting, while asking for 1e-8 costs 2-3x
// the iterations. High-accuracy use can tighten eps_abs; 1e-8 converges
// fine, just slower. (Unlike proxsuite we keep the duality-gap check on
// by default -- it is a stricter stop and cheap to evaluate.)
struct Settings {
  // Termination, on the unscaled elastic-KKT residuals.
  double eps_abs = 1e-5;
  double eps_rel = 0;
  bool check_duality_gap = true;
  double eps_duality_gap_abs = 1e-5;
  double eps_duality_gap_rel = 0;
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
  // cold_reset_limit caps how many times the reset may fire per solve(),
  // and defaults to 0: DISABLED. Measured 2026-08: the reset never fired
  // on any cold solve (720 random instances across every structure /
  // penalty / accuracy tier, plus the 144 Maros-Meszaros small-dense
  // elastic solves) -- it fired only on warm starts, where it is
  // actively harmful. A warm start whose drift pushes weakly-active rows
  // toward elastic saturation resolves them by multiplier creep (~r/mu
  // per outer round), which looks "non-improving" to the reset test
  // exactly when the mu ladder finally reaches creep-resolving depth --
  // the reset then re-slows the creep by 5 orders of magnitude, and
  // uncapped it limit-cycles to kMaxIter (13-22% of ticks on small-drift
  // high-penalty trajectories; capped at 1 it still wastes an eta_ext
  // ladder cycle on ~a third of such ticks, ~1.8x mean iterations).
  // With the reset off, a stuck solve rides the mu ladder to its floor
  // (mu_min_in / mu_min_eq), where the creep becomes a jump and
  // saturation resolves; a genuinely stiff problem at the floor either
  // finishes or fails fast through the factorization-retry path instead
  // of cycling. Set a positive limit to restore the escape hatch.
  double cold_reset_mu = 1.0 / 1.1;
  double cold_reset_threshold = 1e-5;
  double cold_reset_residual = 1e-5;
  int cold_reset_limit = 0;
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

  // Proximal regularization of the relax() Newton system (primal diagonal
  // and equality dual). The prox centers sit
  // at the current iterate, so the value only damps the step -- it does not
  // perturb the relaxed point -- and it is escalated x100 on factorization
  // failure like the main loop's rho.
  double relax_reg = 1e-9;

  // Iteration budget for a relax() WARM attempt before it gives up and
  // restarts from the tight retraction (see relax()). A healthy warm
  // start converges in 1-10 iterations (bench_relax_warm, bench_diff_
  // robot); a warm start whose smoothed configuration flipped on
  // high-penalty rows stalls instead of converging, and without this cap
  // it burns the full max_iter before the fallback. The effective budget
  // is min(relax_warm_budget, max_iter).
  int relax_warm_budget = 15;

  // Predicted-flip gate for a relax() warm attempt. Before continuing
  // from the stored chain iterate, count the rows whose smoothed-
  // configuration sign (which side of the s.z = kappa hyperbola, per
  // block) under the retraction of the CURRENT tight certificate
  // disagrees with the stored iterate; if more than this many disagree,
  // skip the warm attempt and start from the retraction directly --
  // each such flip re-pays the barrier curvature walk, and at high
  // penalty a flipped warm start stalls to the budget fallback anyway
  // (bench_relax_warm). Only well-separated rows count: a near-corner
  // row (|v_old * v_new| <= 100 kappa) flips cheaply, and its converged
  // relaxed sign can chronically disagree with its retraction sign
  // without any tick-to-tick flip. The gate needs a tight certificate
  // from solve(), so gradient-only chains (set_*() + relax() with no
  // solve()) are not gated. Set negative to disable.
  int relax_warm_flip_tol = 0;
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
  // (see update_residuals), so there is nothing else to seed. Unlike an
  // interior-point method there are no slacks to keep strictly positive, so
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

  // Number of BCL cold resets the last solve() fired (capped by
  // Settings::cold_reset_limit).
  int cold_resets() const { return cold_resets_; }

  const Solution& solve() {
    const bool explicit_ws = explicit_warm_;
    explicit_warm_ = false;
    factor_count_ = 0;
    iters_total_ = 0;
    factor_retries_ = 0;
    cold_resets_ = 0;

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
      // cannot fire in the endgame, see Settings::cold_reset_residual, and
      // capped per solve, see Settings::cold_reset_limit).
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

  // Re-solve from the converged solution to a kappa-relaxed central point of
  // the elastic QP: the same KKT conditions, but with the complementarity
  // pairs relaxed to s_t.z_t = s_ineq.z_ineq = kappa (the point an interior
  // point method would reach by stopping its central path at kappa).
  // Differentiating the KKT system
  // there instead of at the exact solution yields gradients whose backward
  // solve stays well-conditioned near degenerate (weakly-active)
  // constraints: the complementarity margins are bounded below by ~kappa.
  //
  // Method (see docs/log_barrier_admm_note.tex): replacing the slack indicator
  // with the barrier -kappa*sum log(s) turns the slack update into the
  // smooth retraction b_k(v) = (v + sqrt(v^2 + 4 kappa))/2, and the paired
  // update z = b_k(v), s = b_k(-v) satisfies z.s = kappa and z, s > 0
  // EXACTLY for any v (the note's ADMM produces exactly this pair with
  // v = z - s). relax() therefore parametrizes each slack/dual pair by its
  // v and Newton-iterates the remaining smooth conditions -- stationarity
  // in x and t, Ax = b, s_t = t, s_ineq = h + t - Gx -- in
  // (x, t, y, v_t, v_ineq), started from the tight solution through the
  // same retraction. Complementarity and positivity hold by construction
  // at every iterate, so there is no fraction-to-boundary safeguard, just
  // a residual backtracking line search; from the retraction start this
  // typically converges in 2-9 Newton steps (one n x n factorization
  // each, reusing the solve() condensation shape with weights
  // Lambda = z1 z2 / (z1 s2 + z2 s1), qpax's elastic weight).
  //
  // Call after solve(); the returned Solution (and solution()) is the
  // RELAXED point, not the optimum, with tol on the unscaled relaxed-KKT
  // residuals. The default tol is tighter than the solver's eps_abs on
  // purpose: gradient accuracy is governed by this residual (the implicit
  // differentiation linearizes here), and extra digits are nearly free in
  // this quadratically convergent corrector, whereas the forward solve
  // pays linear-tail iterations for them. The solver's own iterate and
  // factorization cache are untouched: a subsequent warm solve() still
  // starts from the tight solution. No-op when p == 0 or kappa <= 0.
  // Rows with penalty_i = 0 have no interior (z_t + z_ineq = 0 cannot
  // hold with z > 0), so the relaxation cannot converge for them -- drop
  // such rows instead.
  //
  // Warm starting (warm = true, the default): after a converged relax(),
  // the relaxed iterate is kept, and the next call continues from it
  // instead of the tight retraction. In a control loop this chains the
  // relaxed points tick to tick, which is much cheaper than restarting
  // from the tight certificate: tick drift lives in the LINEAR residual
  // blocks, which Newton removes quadratically (~2 steps), whereas the
  // retraction start pays ~log2(res0/kappa) linear-rate halvings
  // traversing the barrier curvature from the boundary. (Do not gate this
  // on the warm iterate's residual size -- drift makes it large, but it
  // is not what governs the iteration count.) The first call, and any
  // call after setup(), starts from the retraction; if a warm-started run
  // fails to converge, it silently redoes the solve from the retraction
  // (iters then counts both runs). When a tight certificate is
  // available, a predicted-flip gate (Settings::relax_warm_flip_tol)
  // skips the warm attempt outright when the data step flipped the
  // smoothed configuration of well-separated rows -- the regime where
  // the warm start pays the curvature walk per flipped row and loses to
  // the retraction. The stored chain also survives data
  // updates without an intervening solve(), so a gradient-only loop can
  // run set_*() + relax() alone. Changing kappa between calls is allowed
  // (v re-materializes on the new manifold); expect a few extra
  // iterations. The win is regime-dependent: it needs a mostly-stable
  // smoothed row configuration across calls, and a data step that flips
  // many weakly-active rows makes the warm start re-pay the curvature on
  // each flipped row (costing MORE than the retraction, though it still
  // converges to the same point). For sequences of essentially unrelated
  // problems, pass warm = false to force the retraction start.
  const Solution& relax(double kappa, double tol = 1e-6, int max_iter = 50,
                        bool warm = true) {
    if (p_ == 0 || kappa <= 0.0 || (!have_warm_ && !relax_have_warm_)) {
      return sol_;
    }
    // kappa is in the user's frame; each s.z pair picks up only the cost
    // factor under Ruiz (s scales with the row, z against it).
    const double kappa_s = c_s_ * kappa;

    if (warm && relax_have_warm_ &&
        !(have_warm_ && settings.relax_warm_flip_tol >= 0 &&
          relax_predict_flips(kappa_s) > settings.relax_warm_flip_tol)) {
      // Continue from the previous relaxed iterate, still in the
      // workspace; on non-convergence redo from the retraction (needs a
      // tight certificate to reconstruct from). The attempt is capped at
      // relax_warm_budget: a healthy warm start converges well inside it,
      // and a doomed one (smoothed configuration flipped on high-penalty
      // rows) stalls rather than converges, so extra iterations before
      // the fallback are pure waste (measured in bench_relax_warm). The
      // predicted-flip gate above catches most doomed starts before they
      // spend the budget (see Settings::relax_warm_flip_tol).
      relax_run(kappa_s, tol,
                std::min(max_iter, settings.relax_warm_budget));
      if (sol_.converged != 1 && have_warm_) {
        const int warm_iters = sol_.iters;
        relax_init_retraction();
        relax_run(kappa_s, tol, max_iter);
        sol_.iters += warm_iters;
      }
    } else {
      if (!have_warm_) return sol_;
      // Initialize from the retraction of the tight certificate: the
      // starting point already sits on the z.s = kappa manifold.
      relax_init_retraction();
      relax_run(kappa_s, tol, max_iter);
    }
    relax_have_warm_ = sol_.converged == 1;
    return sol_;
  }

 private:
  const Solution& relax_run(double kappa_s, double tol, int max_iter) {
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
      einvr_ = ((d1r_ + d2r_).array() + rho).cwiseInverse();
      lamr_ = d2r_.array() * (d1r_.array() + rho) * einvr_.array();
      bool ok = true;
      while (!relax_factor(rho, delta)) {
        if (retries < settings.max_factor_retries) {
          rho *= 100;
          delta *= 100;
          retries++;
          einvr_ = ((d1r_ + d2r_).array() + rho).cwiseInverse();
          lamr_ = d2r_.array() * (d1r_.array() + rho) * einvr_.array();
        } else {
          ok = false;
          break;
        }
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

      // Full Newton step, then halve until the merit 0.5||F||^2 stops
      // increasing (the retraction keeps every trial point feasible, so
      // plain backtracking is the only safeguard needed). The accept test
      // MUST use the 2-norm merit, for which the Newton step is a descent
      // direction -- see relax_residual(); termination stays on the max
      // norm.
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
    // The loop tests res BEFORE each step; credit a final step that landed
    // inside tol.
    if (status == Status::kMaxIter && std::isfinite(res) && res < tol) {
      status = Status::kSolved;
    }
    if (!std::isfinite(res)) status = Status::kNumerics;
    return relax_finish(status, iter);
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
  // KKT system directly and report kSolved/kNumerics from its residuals.
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

  // ---- relax() helpers ----

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

  // Residuals of the kappa-relaxed KKT at (xr, tr, yr, v1r, v2r), with the
  // slack/dual pairs materialized through the retraction (so z.s = kappa
  // identically and the complementarity rows never appear):
  //   F1 = Q x + q + A'y + G'z2      F2 = penalty - z1 - z2
  //   F3 = A x - b                   F4 = s1 - t     F5 = s2 - (h + t - Gx)
  // Returns the max unscaled norm (termination metric); norms are unscaled
  // componentwise exactly as in update_residuals(). Also fills
  // relax_merit_, the squared 2-norm of the same unscaled residual stack:
  // the Newton step is a guaranteed descent direction for 0.5||F||_2^2
  // (grad = J'F, step = -J^{-1}F, slope = -F'F < 0) but NOT for the max
  // norm, so the line search must accept on the 2-norm merit or it stalls
  // whenever a full step trades residual between rows.
  double relax_residual(double kappa_s) {
    const auto inf_us = [](const VectorXd& v, const VectorXd& s) {
      return v.size() > 0 ? v.cwiseAbs().cwiseProduct(s).maxCoeff() : 0.0;
    };
    const auto ssq_us = [](const VectorXd& v, const VectorXd& s) {
      return v.size() > 0 ? v.cwiseProduct(s).squaredNorm() : 0.0;
    };
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

  // Predicted expensive flips for a warm relax() attempt: rows whose
  // retraction v-sign (from the current tight certificate, same map as
  // relax_init_retraction()) disagrees with the stored chain iterate's,
  // counting only pairs well clear of the hyperbola corner
  // (|v_old * v_new| > 100 kappa_s, i.e. a traversal of >~ 7 halvings).
  // O(p n); does not touch the chain iterate (wGx_ is recomputed by
  // both relax entry paths).
  int relax_predict_flips(double kappa_s) {
    const double corner2 = 100.0 * kappa_s;
    wGx_.noalias() = G_ * x_;
    int flips = 0;
    for (Eigen::Index i = 0; i < p_; ++i) {
      const double r = wGx_[i] - h_[i];
      const double ti = std::max(r + mu_in_ * (z_[i] - penalty_[i]), 0.0);
      const double v1 = (penalty_[i] - z_[i]) - ti;
      const double v2 = z_[i] - std::max(ti - r, 0.0);
      if ((v1 > 0) != (v1r_[i] > 0) && std::abs(v1 * v1r_[i]) > corner2) {
        flips++;
      }
      if ((v2 > 0) != (v2r_[i] > 0) && std::abs(v2 * v2r_[i]) > corner2) {
        flips++;
      }
    }
    return flips;
  }

  // Retraction start for relax(): the reconstructed elastic certificate of
  // the tight iterate mapped through v = z - s. Its x, t, y satisfy the
  // tight KKT to solver tolerance, so the residual is concentrated in the
  // rows the smoothing shifts -- and its v sign pattern encodes the
  // current smoothed configuration exactly.
  void relax_init_retraction() {
    xr_ = x_;
    if (m_ > 0) yr_ = y_;
    wGx_.noalias() = G_ * x_;
    for (Eigen::Index i = 0; i < p_; ++i) {
      const double r = wGx_[i] - h_[i];
      const double ti = std::max(r + mu_in_ * (z_[i] - penalty_[i]), 0.0);
      tr_[i] = ti;
      v1r_[i] = (penalty_[i] - z_[i]) - ti;         // z_t - s_t
      v2r_[i] = z_[i] - std::max(ti - r, 0.0);      // z_ineq - s_ineq
    }
  }

  // Factor K = Q + rho I + (1/delta) A'A + G' diag(Lambda) G into llt_r_.
  // Same shape as factor_kkt(), but into a separate factorization (and
  // reusing the K_/GS_ staging buffers) so the solve() cache stays valid.
  bool relax_factor(double rho, double delta) {
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

  // Certificate at the relaxed point (unscaled). Deliberately does NOT
  // touch the solver iterate, residual scalars, or have_warm_: only sol_
  // reflects the relaxation. The duality gap converges to ~2 p kappa (each
  // relaxed pair contributes kappa), not 0.
  const Solution& relax_finish(Status status, int iters) {
    sol_.x = xr_.cwiseProduct(dx_s_);
    sol_.t = tr_.cwiseProduct(inv_di_);
    sol_.y = yr_.cwiseProduct(y_us_);
    sol_.s_t = s1r_.cwiseProduct(inv_di_);
    sol_.s_ineq = s2r_.cwiseProduct(inv_di_);
    sol_.z_t = z1r_.cwiseProduct(z_us_);
    sol_.z_ineq = z2r_.cwiseProduct(z_us_);
    sol_.status = status;
    sol_.converged = status == Status::kSolved ? 1 : 0;
    sol_.iters = iters;
    wQx_.noalias() = Q_ * xr_;
    const double xQx = xr_.dot(wQx_);
    sol_.primal_obj = (0.5 * xQx + q_.dot(xr_) + penalty_.dot(tr_)) / c_s_;
    double dual_obj = (-0.5 * xQx - h_.dot(z2r_)) / c_s_;
    if (m_ > 0) dual_obj -= b_.dot(yr_) / c_s_;
    sol_.primal_res = relax_primal_res_;
    sol_.dual_res = relax_dual_res_;
    sol_.duality_gap = std::abs(sol_.primal_obj - dual_obj);
    return sol_;
  }

  // Unscaled residuals of the elastic QP at the reconstructed expanded
  // point:
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
  int cold_resets_ = 0;

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
