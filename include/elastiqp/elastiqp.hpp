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
// The solver is a LOG-BARRIER primal-dual proximal augmented Lagrangian
// (barrier PDAL) method: the proximal-method-of-multipliers / BCL outer
// structure of ProxQP, with the slack indicators replaced by the barrier
// -kappa*sum log(s) so that every inner subproblem is smooth
// (docs/log_barrier_pdal.tex, adapted to the elastic form in
// docs/log_barrier_pdal_implementation.md).
//
// Notes:
//
// Each elastic row carries two slack/dual pairs: (s_t, z_t) for t >= 0 and
// (s_ineq, z_ineq) for Gx - t <= h, with the t-stationarity tying
// z_t + z_ineq = penalty (the bounded multiplier that makes the l1 penalty
// exact). Every pair is parametrized by a single implicit-complementarity
// coordinate v through the log-barrier retraction
//
//   z = b_kappa(v),  s = b_kappa(-v),  b_kappa(v) = (v + sqrt(v^2+4kappa))/2,
//
// which satisfies z.s = kappa and z, s > 0 identically for ANY v. The inner
// subproblem at a fixed central-path parameter kappa is therefore an
// unconstrained smooth root-finding problem in (x, t, y, v_t, v_ineq)
// (residuals F1..F5, see barrier_kkt_fill), solved by damped Newton with a
// backtracking line search on ||F||_2^2. The outer BCL loop anneals kappa
// (kappa_init -> kappa_min, one shrink per good step) alongside the usual
// mu schedule, warm-starting each barrier subproblem with the previous v --
// inexact path following with BCL-controlled subproblem accuracy.
//
// The dual proximal terms bound the condensed row weights by 1/mu_in
// independently of kappa (beta = b' / (b'(-v) + mu_in b') <= 1/mu_in), so
// the Newton systems stay conditioned like ProxQP's as kappa -> 0. In that
// limit b_kappa becomes the orthant projection and the smooth solver
// degenerates exactly into the ordinary active-set PDAL: rows far from
// their kinks have weights snapped to 0 (inactive/saturated) or their
// active value, which is what lets the factorization cache (keyed on the
// weight vector Lambda, see ensure_factor) recover the zero/low-cost
// warm re-solves of an active-set method.
//
// The linear algebra is condensed onto the n x n SPD system
//   K = Q + rho*I + (1/mu_eq) A^T A + G^T diag(Lambda) G,
//   Lambda_i = beta2_i (rho + beta1_i) / (rho + beta1_i + beta2_i),
// (the elastic pair-elimination weight; with beta = z/s it is qpax's
// elastic weight). One LLT per Newton step when the weights move, zero
// when they are frozen within kkt_cache_tol.
//
// Termination and every reported quantity stay on the TIGHT elastic KKT:
// the certificate is reconstructed by hard projection from (x, z_ineq)
// exactly as in the active-set solver (update_residuals), so an
// unsaturated converged solution matches the hard-constrained QP to
// tolerance, reported slacks of feasible rows are identically zero, and
// penalty - z_t - z_ineq = 0 holds exactly. The barrier bias at
// kappa_min (duality gap ~ 2 p kappa) sits below the termination
// tolerances. Warm starting reuses the previous (x, y, z_ineq) with
// z_ineq clamped to [0, penalty], and starts directly at kappa_min.
//
// Omitted from proxsuite: GPDAL merit, incremental LDLT updates + iterative
// refinement, infeasibility detection, box specialization, nonconvex
// handling. Equalities hold to solver tolerance (~eps_abs).
//
// Differentiability: relax(kappa) runs the SAME smooth Newton corrector at
// a fixed kappa (without the proximal terms -- the centers sit at the
// iterate) to walk the converged solution to the kappa-relaxed central
// point (s.z = kappa) for smooth implicit differentiation. Forward and
// backward pass are thus two stopping points of one barrier method:
// solve() follows kappa down to kappa_min, relax() holds it at the
// differentiation target.

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
  int iters = 0;      // total inner barrier-Newton steps
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
  // round runs up to max_iter_in barrier-Newton steps, and Solution::iters
  // reports their total (so it is not bounded by max_outer_iter).
  int max_outer_iter = 250;
  int max_iter_in = 1500;

  // Proximal regularization (primal) and AL penalties / dual prox (mu).
  // rho and the mu_init values are proxsuite defaults. The mu floors are
  // HIGHER than proxsuite's (1e-9 / 1e-8): the barrier solver's duals are
  // closed-form maps of x (y = yk + (Ax - b)/mu_eq, z through the row
  // solve), so evaluation round-off in the primal enters the dual
  // residual amplified by 1/mu -- the floor must keep machine-eps / mu
  // below the termination tolerance. mu only sets the outer contraction
  // RATE (any fixed mu > 0 converges), so the cost of the higher floor is
  // negligible.
  double rho = 1e-6;
  double mu_eq_init = 1e-3;
  double mu_in_init = 1e-1;
  double mu_min_eq = 1e-6;
  double mu_min_in = 1e-6;
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

  // Log-barrier central-path schedule of the forward pass
  // (docs/log_barrier_pdal.tex). Every BCL round solves the smooth barrier
  // PDAL subproblem at the current kappa (complementarity target
  // s.z = kappa on both elastic pairs); kappa shrinks by
  // kappa_update_factor after each good outer step, down to kappa_min,
  // where the barrier bias (duality gap ~ 2 p kappa, residual bias
  // O(kappa/margin)) sits below any practical termination tolerance.
  // Warm-started solves match the smoothing to the distance from the
  // solution instead of starting the full schedule: the starting kappa is
  // kappa_warm_scale * max(primal_res, dual_res) at entry, clamped into
  // [kappa_min, kappa_init] -- a near-optimal start re-solves almost
  // tight, while a badly drifted one gets the same globalization as a
  // cold start. The default is deliberately conservative (structural
  // drift on badly scaled data stays robust). Control loops whose active
  // set is stable across ticks can set kappa_warm_scale = 0: warm solves
  // then start at kappa_min and typically converge in ~2 Newton steps
  // (measured ~4x faster warm ticks on the humanoid WBC benchmark). All
  // kappas must be > 0: the retraction derivative is undefined at
  // (v, kappa) = (0, 0).
  double kappa_init = 1e-2;
  double kappa_warm_scale = 1e-1;
  double kappa_update_factor = 1e-2;
  double kappa_min = 1e-13;

  // Factorization reuse for the barrier Newton systems: the cached KKT
  // factor is reused while every row weight Lambda_i satisfies
  //   |dLambda_i| ||G_i||^2 <= tol * (Lambda_i ||G_i||^2 + min diag K),
  // i.e. each row is frozen relative to itself, with an absolute
  // allowance anchored to the smallest curvature in K. Steps taken on a
  // reused factor are inexact-Newton steps safeguarded by the line
  // search; at this tolerance the direction error is negligible. Decided
  // rows' weights freeze as kappa -> 0, so warm re-solves with a stable
  // configuration skip refactorization the way an active-set cache does.
  double kkt_cache_tol = 1e-9;

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

    zhat_.resize(p_);
    t_.resize(p_);
    s2_.resize(p_);
    r_.resize(p_);
    verr_.resize(n_);
    dyrhs_.resize(m_);
    rhs_x_.resize(n_);
    Gdx_.resize(p_);
    wQx_.resize(n_);
    wGtz_.resize(n_);
    wAty_.resize(n_);
    wAx_.resize(m_);
    wGx_.resize(p_);

    // Barrier iterate of the forward pass (re-derived from (x, z) at the
    // start of every solve) and its prox centers.
    tb_.resize(p_);
    tk_.resize(p_);
    v1_.resize(p_);
    v2_.resize(p_);
    v1k_.resize(p_);
    v2k_.resize(p_);
    z2k_.resize(p_);
    zk_.resize(p_);
    beta1_.resize(p_);
    beta2_.resize(p_);
    dhat1_.resize(p_);
    dhat2_.resize(p_);
    zc_.resize(p_);
    g2_.resize(p_);
    f_lam_.resize(p_);

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

    // Central-path schedule: cold starts anneal kappa_init -> kappa_min
    // (one shrink per good BCL step); a (near-optimal) warm start goes
    // straight to kappa_min. The barrier iterate (t, v1, v2) is derived
    // from the tight certificate of (x, z) through the retraction, so it
    // starts exactly on the s.z = kappa manifold.
    const bool cold =
        !explicit_ws && !(settings.warm_start && have_warm_);
    kappa_ = cold ? settings.kappa_init
                  : std::min(settings.kappa_init,
                             std::max(settings.kappa_min,
                                      settings.kappa_warm_scale *
                                          std::max(primal_res_, dual_res_)));

    barrier_init();

    // ------------- outer loop (PMM + BCL, proxsuite qp_solve) -------------
    for (int oiter = 0; oiter < settings.max_outer_iter; ++oiter) {
      const double pri_old = primal_res_;
      const double dua_old = dual_res_;
      const double kappa_s = c_s_ * kappa_;

      // The PMM "multiplier update" is implicit: snapshot the prox centers
      // and let the inner Newton move (x, t, y, v1, v2) jointly. The dual
      // centers are the retraction images of the v snapshot at the
      // CURRENT kappa (after a shrink this is the path-following
      // re-centering of the previous round's pairs).
      const int iters_before = iters_total_;
      xk_ = x_;
      tk_ = tb_;
      if (m_ > 0) yk_ = y_;
      v1k_ = v1_;
      v2k_ = v2_;
      zk_ = z_;  // certificate snapshot for the bad-step revert
      for (Eigen::Index i = 0; i < p_; ++i) {
        z2k_[i] = retraction(v2k_[i], kappa_s);
      }

      if (!inner_loop(eta_in, kappa_s)) {
        return finish(Status::kNumerics);
      }

      // Certificate dual for the tight reconstruction: decided rows snap
      // to their exact bound, active rows keep the inner loop's
      // linearized dual iterate, clamped into the box [0, penalty] that
      // the reported certificate, the BCL test, and warm starting all
      // rely on. "Decided" is read off the barrier pair itself: a row is
      // tight-saturated iff its t-bound dual z1 = kappa/t vanishes as
      // kappa -> 0 (slack bounded away from zero), and tight-inactive
      // iff z2 does -- degenerate ties keep z1, z2 = O(1) and are
      // robustly excluded (snapping a tie stalls convergence; leaving a
      // truly saturated dual epsilon below penalty turns into a
      // penalty-scaled duality-gap bias). The kappa_s * 1e6 threshold
      // means "slack > 1e-6"; the absolute cap keeps early large-kappa
      // rounds from snapping.
      for (Eigen::Index i = 0; i < p_; ++i) {
        const double zsnap =
            std::min(kappa_s * 1e6, 1e-10 * (1.0 + penalty_[i]));
        double zi = std::min(std::max(zc_[i], 0.0), penalty_[i]);
        if (z1r_[i] < zsnap) {
          zi = penalty_[i];
        } else if (z2r_[i] < zsnap) {
          zi = 0.0;
        }
        z_[i] = zi;
      }

      update_residuals();
      if (converged()) return finish(Status::kSolved);
      const double pri_new = primal_res_;
      const double dua_new = dual_res_;

      // BCL update (proxsuite bcl_update): the elastic primal feasibility
      // (equality residual + inequality violation beyond the reconstructed
      // slack) decides between a dual update and a mu shrink. Good steps
      // also advance the central path.
      if (pri_new <= eta_ext || iters_total_ > settings.safe_guard) {
        // Floor eta_ext at the termination tolerance: the BCL test judges
        // whether the subproblem made adequate PRIMAL progress, and
        // demanding progress below eps_abs is meaningless. Without the
        // floor eta_ext decays to denormals, every endgame step reads
        // "bad", and mu collapses to its floor -- raising the 1/mu noise
        // amplification of the dual maps exactly when the last digits of
        // the dual residual and duality gap are being ground out.
        eta_ext = std::max(eta_ext * std::pow(mu_in_, settings.beta_bcl),
                           0.1 * settings.eps_abs);
        eta_in = std::max(eta_in * mu_in_, eps_in_min);
        // Anneal the central path; rounds the inner Newton dispatches in
        // a step or two are tracking the path easily, so take a double
        // kappa step (self-regulating: near-optimal warm starts and easy
        // endgames skip levels, hard stretches keep the fine schedule).
        double kfac = settings.kappa_update_factor;
        if (iters_total_ - iters_before <= 2) kfac *= kfac;
        kappa_ = std::max(kappa_ * kfac, settings.kappa_min);
        // Endgame dual push: the primal is at tolerance but the dual
        // residual has stopped improving -- the dual prox contraction is
        // too slow at the current mu (weakly curved directions contract
        // like sigma * mu per round). Tighten mu WITHOUT the bad-step
        // revert: the multipliers are the best estimates available, and
        // the eta_ext floor above would otherwise pin mu forever.
        // (Gated to dua clearly above tolerance: pushing mu once the dual
        // is within a decade of eps_abs trades the last digits for 1/mu
        // evaluation noise.)
        if (pri_new <= 0.1 * settings.eps_abs &&
            dua_new > 10.0 * settings.eps_abs && dua_new > 0.9 * dua_old) {
          mu_in_ = std::max(mu_in_ * settings.mu_update_factor,
                            settings.mu_min_in);
          mu_eq_ = std::max(mu_eq_ * settings.mu_update_factor,
                            settings.mu_min_eq);
        }
      } else {
        // Bad step: revert the multipliers (x and t are kept) and increase
        // the penalties. The certificate dual reverts to its round-start
        // SNAPSHOT (re-deriving it from the retraction of v2k would
        // re-inject map round-off amplified by 1/mu_in). The factorization
        // cache invalidates via the Lambda fingerprint once the weights
        // move.
        if (m_ > 0) y_ = yk_;
        v1_ = v1k_;
        v2_ = v2k_;
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

  // Re-solve from the converged solution to a kappa-relaxed central point of
  // the elastic QP: the same KKT conditions, but with the complementarity
  // pairs relaxed to s_t.z_t = s_ineq.z_ineq = kappa (the point an interior
  // point method would reach by stopping its central path at kappa).
  // Differentiating the KKT system
  // there instead of at the exact solution yields gradients whose backward
  // solve stays well-conditioned near degenerate (weakly-active)
  // constraints: the complementarity margins are bounded below by ~kappa.
  //
  // Method (docs/log_barrier_pdal.tex; historically
  // docs/log_barrier_admm_note.tex): replacing the slack indicator
  // with the barrier -kappa*sum log(s) turns the slack update into the
  // smooth retraction b_k(v) = (v + sqrt(v^2 + 4 kappa))/2, and the paired
  // update z = b_k(v), s = b_k(-v) satisfies z.s = kappa and z, s > 0
  // EXACTLY for any v. relax() therefore parametrizes each slack/dual pair by its
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
  // starts from the tight solution. No-op when p == 0, kappa <= 0, or the
  // previous solve ended in kNumerics. Rows with penalty_i = 0 have no
  // interior (z_t + z_ineq = 0 cannot hold with z > 0), so the relaxation
  // cannot converge for them -- drop such rows instead.
  const Solution& relax(double kappa, double tol = 1e-6, int max_iter = 50) {
    if (p_ == 0 || !have_warm_ || kappa <= 0.0) return sol_;
    // kappa is in the user's frame; each s.z pair picks up only the cost
    // factor under Ruiz (s scales with the row, z against it).
    const double kappa_s = c_s_ * kappa;

    // Initialize from the retraction of the tight certificate: the
    // starting point already sits on the z.s = kappa manifold.
    relax_init_retraction();

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

      // Condensed Newton system, shared with the forward pass
      // (condense_weights / barrier_direction). relax() is the mu_in = 0,
      // centers-at-the-iterate instance of the barrier machinery: the
      // pair weights are beta = b'(v)/b'(-v) = z/s (unbounded near the
      // boundary as kappa -> 0, which is why Ruiz matters for large
      // penalties here), Dhat = b'(-v), and per row
      //   E = rho + beta1 + beta2,  Lambda = beta2 (rho + beta1) / E,
      // giving the n x n SPD system
      //   [Q + rho I + (1/delta) A'A + G' diag(Lambda) G] dx = rhs.
      beta1_ = z1r_.cwiseQuotient(s1r_);
      beta2_ = z2r_.cwiseQuotient(s2r_);
      for (Eigen::Index i = 0; i < p_; ++i) {
        dhat1_[i] = retraction_dcomp(v1r_[i], kappa_s);
        dhat2_[i] = retraction_dcomp(v2r_[i], kappa_s);
      }
      condense_weights(rho);
      bool ok = true;
      while (!relax_factor(rho, delta)) {
        if (retries < settings.max_factor_retries) {
          rho *= 100;
          delta *= 100;
          retries++;
          condense_weights(rho);
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

      barrier_direction(delta, llt_r_);

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

    lamr_.setZero();  // no inequality rows in the cold-start system
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

  // ---- inner loop ----
  // With the row blocks and the equality dual eliminated in closed form
  // at every evaluation point (row_update; y*(x) = yk + (Ax - b)/mu_eq),
  // the barrier subproblem is an unconstrained smooth CONVEX minimization
  // in x alone,
  //   Phi(x) = f(x) + rho/2||x - xk||^2
  //          + yk'(Ax-b) + ||Ax-b||^2/(2 mu_eq) + sum_i rowval_i(G_i x),
  // (partial max over duals / min over primal slacks preserves convexity;
  // Danskin gives grad Phi = the reduced F1 and hess Phi = the condensed
  // K). Each iteration takes the condensed Newton step and an exact line
  // search: the slice derivative g(alpha) = Phi'(x + alpha dx) . dx is
  // strictly increasing, so its root is found by bracketed regula falsi
  // -- the smooth analogue of proxsuite's exact piecewise line search,
  // and like it, alpha may exceed 1. A g evaluation costs p scalar row
  // solves plus O(p) dot products (no matvecs). Returns false only on a
  // factorization disaster / non-finite iterates.
  bool inner_loop(double eps_int, double kappa_s) {
    // Closed-form block updates at the incoming centers (this is where
    // the PMM multiplier update actually happens).
    // The equality dual stays an ADDITIVE iterate (y += alpha dy): the
    // first step folds y onto its closed form y* = yk + (Ax - b)/mu_eq
    // exactly (the map is linear), and accumulating instead of
    // re-deriving keeps fresh A*x round-off from entering y amplified by
    // 1/mu_eq -- the same reason the certificate dual zc_ is linearized.
    wGx_.noalias() = G_ * x_;
    row_update(kappa_s);
    if (m_ > 0) wAx_.noalias() = A_ * x_;
    zc_ = z2r_;
    double err = inner_residual(kappa_s);
    for (int it = 0; it < settings.max_iter_in; ++it) {
      if (!std::isfinite(err)) return false;
      // The outer loop only runs while the true elastic KKT is
      // unsatisfied, so never accept a subproblem without taking at least
      // one Newton step -- otherwise the outer loop can spin to kMaxIter
      // with an unchanged iterate.
      if (err <= eps_int && it > 0) return true;

      // Pair weights (both derivative branches evaluated stably). The
      // proxed row pair has beta2 = b'/(b'(-v) + mu_in b') <= 1/mu_in
      // uniformly in kappa; the barrier-slaved t pair has beta1 = z1/s1
      // (unbounded near the boundary, but it only enters through
      // Lambda = beta2 (rho + beta1)/(rho + beta1 + beta2) <= beta2 and
      // through beta1 * F4 products whose F4 is zero to relative
      // round-off after row_update).
      for (Eigen::Index i = 0; i < p_; ++i) {
        // B = b'(v) and C = b'(-v) are the same {small, 1 - small} pair
        // swapped by the sign of v: one sqrt serves both, stably.
        const double u1 = v1_[i];
        const double r1 = std::sqrt(u1 * u1 + 4.0 * kappa_s);
        const double s1 = 2.0 * kappa_s / (r1 * (r1 + std::abs(u1)));
        const double c1 = u1 >= 0.0 ? s1 : 1.0 - s1;
        const double b1 = u1 >= 0.0 ? 1.0 - s1 : s1;
        dhat1_[i] = c1;
        beta1_[i] = b1 / c1;
        const double u2 = v2_[i];
        const double r2 = std::sqrt(u2 * u2 + 4.0 * kappa_s);
        const double s2 = 2.0 * kappa_s / (r2 * (r2 + std::abs(u2)));
        const double c2 = u2 >= 0.0 ? s2 : 1.0 - s2;
        const double b2 = u2 >= 0.0 ? 1.0 - s2 : s2;
        dhat2_[i] = c2 + mu_in_ * b2;
        beta2_[i] = b2 / dhat2_[i];
      }
      condense_weights(rho_);
      if (!ensure_factor()) return false;
      barrier_direction(mu_eq_, llt_);
      if (!std::isfinite(dxr_.sum())) return false;

      iters_total_++;
      // Certificate dual: the LINEARIZED update from the pre-step point.
      // The Newton step zeroes the measured F1 (which contained this z)
      // to second order, so the pair (x + alpha dx, zc) reaches the
      // linear-solve round-off floor -- unlike the row-map z, which
      // re-derives from x every evaluation and re-injects primal
      // round-off amplified by 1/mu_in (visible at penalty ~ 1e5 against
      // tight tolerances). This is the additive dual iterate of ordinary
      // PDAL, recovered from the same step data.
      zc_ = z2r_;  // pre-step row duals (the line search overwrites z2r_)
      const double alpha = line_search(kappa_s);
      zc_ += alpha * (beta2_.cwiseProduct(dhat2_).cwiseProduct(dv2r_));
      const double dwmax = dxr_.lpNorm<Eigen::Infinity>();
      if (alpha * dwmax < 1e-11 && it > 0) return true;

      // Accept: move x and the additive y along, then re-solve the rows
      // at full precision at the accepted point (zhat_ still holds the
      // pre-step G x from the line search).
      x_ += alpha * dxr_;
      wGx_ = zhat_ + alpha * Gdx_;
      if (m_ > 0) y_ += alpha * dyr_;
      row_update(kappa_s);
      err = inner_residual(kappa_s);
      if (alpha == 0.0) return true;
    }
    return true;  // out of inner iterations; the outer loop adapts mu
  }

  // Exact line search on the convex slice: finds the root of the strictly
  // increasing g(alpha) = Phi'(x + alpha dx) . dx
  //   = c0 + c1 alpha + Gdx . z2*(alpha) + Adx . y*(alpha),
  // where the rows and equality dual are re-eliminated at every trial
  // point. Leaves wGx_ and the row blocks at the returned alpha. dyrhs_
  // holds A dx on exit (reused by the caller's y update).
  double line_search(double kappa_s) {
    wQx_.noalias() = Q_ * x_;
    verr_ = wQx_ + q_ + rho_ * (x_ - xk_);
    double c0 = dxr_.dot(verr_);
    rhs_x_.noalias() = Q_ * dxr_;  // scratch: Q dx
    double c1 = dxr_.dot(rhs_x_) + rho_ * dxr_.squaredNorm();
    double cy0 = 0.0, cy1 = 0.0;
    if (m_ > 0) {
      dyrhs_.noalias() = A_ * dxr_;
      // Adx . y*(alpha) with y*(x) = y + F3/mu_eq (rf3_ is current):
      cy0 = dyrhs_.dot(y_) + dyrhs_.dot(rf3_) / mu_eq_;
      cy1 = dyrhs_.squaredNorm() / mu_eq_;
    }
    zhat_ = wGx_;  // base G x (zhat_ is free scratch during the inner loop)

    // Loose row tolerance for trial evaluations: the search only needs
    // bracketing accuracy on g. The caller's residual evaluation redoes
    // the accepted point's rows at full precision.
    const auto g_at = [&](double alpha) {
      wGx_ = zhat_ + alpha * Gdx_;
      row_update(kappa_s, 1e-9);
      return c0 + cy0 + alpha * (c1 + cy1) + Gdx_.dot(z2r_);
    };

    // g(0) is free: the rows are already at their optima for the current
    // x, so no row solve is needed. g(0) = -dx'K dx < 0 up to round-off;
    // a nonnegative value means the direction carries no descent left
    // (round-off floor) -- report a zero step so the caller's stagnation
    // guard can finish the loop.
    const double g0 = c0 + cy0 + Gdx_.dot(z2r_);
    if (g0 >= 0.0) return 0.0;
    // Wolfe-style curvature acceptance: |g(alpha)| <= c2 |g(0)| holds in
    // a neighborhood of the 1D minimizer of the smooth convex slice and
    // keeps the evaluation count low (often a single g(1)); c2 = 0.5 is
    // well inside the standard Wolfe range for Newton directions.
    const double gtol = 0.5 * (-g0);

    double alpha = 1.0;
    double g = g_at(1.0);
    if (std::abs(g) <= gtol) return alpha;

    double alo = 0.0, glo = g0, ahi = 1.0, ghi = g;
    if (g < 0.0) {
      // Still descending at alpha = 1: expand (strong convexity
      // guarantees g -> +infinity eventually).
      alo = 1.0;
      glo = g;
      for (int k = 0; k < 12; ++k) {
        alpha *= 2.0;
        g = g_at(alpha);
        if (std::abs(g) <= gtol) return alpha;
        if (g > 0.0) break;
        alo = alpha;
        glo = g;
      }
      if (g < 0.0) return alpha;  // deep flat stretch; bounded step
      ahi = alpha;
      ghi = g;
    }

    // Regula falsi with bisection safeguard on the increasing g.
    for (int k = 0; k < 20; ++k) {
      double amid = alo + (-glo) * (ahi - alo) / (ghi - glo);
      const double span = ahi - alo;
      if (!(amid > alo + 0.02 * span && amid < ahi - 0.02 * span) ||
          !std::isfinite(amid)) {
        amid = 0.5 * (alo + ahi);
      }
      g = g_at(amid);
      alpha = amid;
      if (std::abs(g) <= gtol) return alpha;
      if (g >= 0.0) {
        ahi = amid;
        ghi = g;
      } else {
        alo = amid;
        glo = g;
      }
    }
    // Iteration cap: land on the bracket's descent side (g < 0 there, so
    // the merit strictly decreased on [0, alo]). The caller re-solves the
    // rows at the returned alpha.
    return alo;
  }

  // Exact per-row minimization of the barrier subproblem for fixed x --
  // the elastic analogue of the note's closed-form smooth multiplier
  // update (docs/log_barrier_pdal.tex sec. 3), and the smooth counterpart
  // of the active-set solver's dual snap. For fixed x (and centers), each
  // row's (t, s1, z1, s2, z2) block is determined by F2 = F4 = F5 = 0 with
  // z.s = kappa. The t-bound pair carries NO dual prox (mirroring the
  // tight solver, where z_t is slaved to the fold, never proximally
  // anchored -- anchoring it freezes z_t near penalty and kills the row's
  // feasibility pull), so F4 gives s1 = t, z1 = kappa/t exactly. The row
  // pair eliminates through its proximal equation F5: with
  // d2 = (t - r) - mu z2k and mk = mu_in * kappa,
  //   s2 = b_mk(d2),  mu z2 = b_mk(-d2)
  // (the paired retraction, so z2.s2 = kappa to round-off), leaving one
  // strictly increasing scalar equation per row,
  //   F2(t) = penalty - kappa/t - z2(t) + rho (t - tk) = 0,   t > 0,
  // solved by bracketed Newton whose out-of-bracket fallback is the
  // frozen-z2 quadratic model rho t^2 + (penalty - z2 - rho tk) t - kappa
  // = 0 (closed-form positive root; it solves the barrier + linear part
  // exactly, so it jumps between the t ~ kappa/penalty inactive regime
  // and the t ~ violation regime in one step). This absorbs
  // arbitrarily large multiplier updates in closed form, so the coupled
  // Newton only moves (x, y). With the rows at their exact optima, the
  // condensed n x n step IS the Newton step of the reduced smooth system
  // in (x, y) (implicit function theorem through the same elimination).
  //
  // Reads the row values from wGx_ (the caller keeps it at G x for the
  // point being evaluated -- during the line search that is one axpy, not
  // a matvec) and writes (tb_, v1_, v2_) plus the dual images z1r_, z2r_.
  // ftol is the relative residual tolerance of the scalar solves: the
  // line search passes a loose one (it only needs g(alpha) bracketing
  // accuracy), every state-defining call uses round-off level -- a looser
  // final tolerance leaves a systematic z2 offset of ftol * penalty that
  // floors the dual residual (visible at penalty ~ 1e5 against
  // eps_abs = 1e-8).
  void row_update(double kappa_s, double ftol = 4e-16) {
    const double mk = mu_in_ * kappa_s;
    for (Eigen::Index i = 0; i < p_; ++i) {
      const double r = wGx_[i] - h_[i];
      const double w = penalty_[i];
      const double a2 = r + mu_in_ * z2k_[i];
      const double tc = tk_[i];
      double z2 = 0.0;
      // One sqrt serves both b_mk(-d) and its derivative (the stable
      // branches of retraction / retraction_dcomp, inlined).
      const auto fval = [&](double t, double* deriv) {
        const double d = t - a2;
        const double rr = std::sqrt(d * d + 4.0 * mk);
        const double bm = d <= 0.0 ? 0.5 * (rr - d) : 2.0 * mk / (rr + d);
        z2 = bm / mu_in_;
        if (deriv != nullptr) {
          const double small = 2.0 * mk / (rr * (rr + std::abs(d)));
          const double bp = d >= 0.0 ? small : 1.0 - small;
          *deriv = kappa_s / (t * t) + bp / mu_in_ + rho_;
        }
        return w - kappa_s / t - z2 + rho_ * (t - tc);
      };
      // Start from the previous t, floored into the barrier's natural
      // scale so cold zeros are valid.
      double t = std::max(tb_[i], kappa_s / (w + 1.0));
      double lo = 0.0;
      double hi = std::numeric_limits<double>::infinity();
      bool fresh = false;  // z2 corresponds to the current t
      for (int k = 0; k < 60; ++k) {
        double ft;
        const double f = fval(t, &ft);
        fresh = true;
        if (f > 0.0) {
          hi = t;
        } else {
          lo = t;
        }
        const double fscale =
            w + kappa_s / t + z2 + std::abs(rho_ * (t - tc));
        if (std::abs(f) <= ftol * fscale) break;
        double tn = t - f / ft;
        if (!(tn > lo && tn < hi) || !std::isfinite(tn)) {
          // Frozen-z2 quadratic model (stable positive-root formula): it
          // solves the barrier + linear part exactly, jumping between the
          // inactive and violated regimes in one step.
          const double bq = w - z2 - rho_ * tc;
          const double disc = std::sqrt(bq * bq + 4.0 * rho_ * kappa_s);
          tn = bq >= 0.0 ? (2.0 * kappa_s) / (bq + disc)
                         : (disc - bq) / (2.0 * rho_);
          if (!(tn > lo && tn < hi) || !std::isfinite(tn)) {
            tn = std::isfinite(hi) ? 0.5 * (lo + hi) : 2.0 * t + 1.0;
          }
        }
        const bool done = std::abs(tn - t) <= 1e-14 * (1.0 + std::abs(tn));
        t = tn;
        fresh = false;
        if (done) break;
      }
      if (!fresh) fval(t, nullptr);  // final pair at the accepted t
      tb_[i] = t;
      z1r_[i] = kappa_s / t;
      z2r_[i] = z2;
      v1_[i] = kappa_s / t - t;              // z1 - s1, exact manifold pair
      v2_[i] = z2 - retraction(t - a2, mk);  // z2 - s2
    }
  }

  // Inner stopping quantity: the max norm of the barrier-PDAL residuals
  // at the current iterate (scaled frame, matching the absolute eta_in
  double inner_residual(double kappa_s) {
    barrier_kkt_fill(x_, tb_, y_, v1_, v2_, kappa_s, true);
    double err = std::max(rf1_.lpNorm<Eigen::Infinity>(),
                          rf2_.lpNorm<Eigen::Infinity>());
    err = std::max({err, rf4_.lpNorm<Eigen::Infinity>(),
                    rf5_.lpNorm<Eigen::Infinity>()});
    if (m_ > 0) err = std::max(err, rf3_.lpNorm<Eigen::Infinity>());
    return err;
  }

  // Barrier iterate (t, v1, v2) from the tight certificate of the current
  // (x, z): the same reconstruction update_residuals reports, mapped
  // through v = z - s per pair. Places the iterate exactly on the
  // s.z = kappa manifold for any kappa.
  void barrier_init() {
    wGx_.noalias() = G_ * x_;
    for (Eigen::Index i = 0; i < p_; ++i) {
      const double r = wGx_[i] - h_[i];
      const double ti = std::max(r + mu_in_ * (z_[i] - penalty_[i]), 0.0);
      tb_[i] = ti;
      v1_[i] = (penalty_[i] - z_[i]) - ti;      // z_t - s_t
      v2_[i] = z_[i] - std::max(ti - r, 0.0);   // z_ineq - s_ineq
    }
  }

  // ---- factorization cache ----
  // K = Q + rho*I + (1/mu_eq) A'A + G' diag(lamr_) G into llt_.
  bool factor_kkt() {
    K_.triangularView<Eigen::Lower>() = Q_;
    K_.diagonal().array() += rho_;
    if (m_ > 0) {
      K_.triangularView<Eigen::Lower>() += (1.0 / mu_eq_) * AtA_;
    }
    if (p_ > 0 && lamr_.maxCoeff() > 0.0) {
      GS_.noalias() = lamr_.cwiseSqrt().asDiagonal() * G_;
      K_.selfadjointView<Eigen::Lower>().rankUpdate(GS_.transpose());
    }
    llt_.compute(K_);
    ++factor_count_;
    return llt_.info() == Eigen::Success &&
           std::isfinite(K_.diagonal().sum());
  }

  // Refactor only when (rho, mu_eq), the matrices, or the row weights
  // Lambda moved: reuse is accepted while every row's weight change
  // perturbs K by less than kkt_cache_tol relative to its diagonal scale
  // (|dLambda_i| ||G_i||^2 <= tol * max|diag K|, a bound on the spectral
  // perturbation). Weights of decided rows freeze as kappa -> 0, so a
  // settled configuration -- notably warm re-solves -- skips
  // factorizations entirely; steps on a reused factor are inexact-Newton
  // steps safeguarded by the merit line search. On LLT failure rho is
  // escalated x100 (the slightly stale Lambda then only adds to the
  // Jacobian inexactness, which the same safeguard covers).
  bool ensure_factor() {
    if (matrix_dirty_) {
      for (Eigen::Index i = 0; i < p_; ++i) {
        g2_[i] = G_.row(i).squaredNorm();
      }
    }
    bool need = !factored_ || matrix_dirty_ || f_rho_ != rho_ ||
                f_mu_eq_ != mu_eq_;
    if (!need) {
      // Per-row acceptance: relative freeze of the row's own weight plus
      // an absolute term anchored to the SMALLEST curvature in K (the
      // max-diagonal alone lets mid-weight rows drift by amounts that are
      // large against weakly-curved directions, capping the attainable
      // accuracy on ill-conditioned endgames).
      const double tol = settings.kkt_cache_tol;
      for (Eigen::Index i = 0; i < p_; ++i) {
        if (std::abs(lamr_[i] - f_lam_[i]) * g2_[i] >
            tol * (f_lam_[i] * g2_[i] + f_kdiag_min_)) {
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
    f_lam_ = lamr_;
    f_kdiag_min_ = K_.diagonal().minCoeff();
    return true;
  }

  // ---- relax() helpers ----

  // Closed-form prox of kappa*(-log): the positive root of
  // s^2 - v s - kappa = 0, i.e. b_k(v) = (v + sqrt(v^2 + 4 kappa))/2, with
  // the cancellation-free branch b_k(v) = 2 kappa / (sqrt(..) - v) for
  // v < 0 (docs/log_barrier_pdal.tex eq. 24).
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

  // ---- shared barrier machinery (forward inner loop AND relax()) ----

  // Materialize the retraction pairs (z1r_, s1r_, z2r_, s2r_) and the
  // barrier-PDAL residuals rf1_..rf5_ at the given point:
  //   F1 = Q x + q + A'y + G'z2 [+ rho (x - xk)]
  //   F2 = penalty - z1 - z2    [+ rho (t - tk)]
  //   F3 = A x - b              [+ mu_eq (yk - y)]
  //   F4 = s1 - t               (no dual prox: z_t is barrier-slaved)
  //   F5 = G x + s2 - h - t     [+ mu_in (z2k - z2)]
  // with_prox adds the bracketed proximal terms against the current
  // centers (xk_, tk_, yk_, z2k_); relax() calls without them (its
  // centers sit at the iterate, where they vanish identically).
  void barrier_kkt_fill(const VectorXd& x, const VectorXd& t,
                        const VectorXd& y, const VectorXd& v1,
                        const VectorXd& v2, double kappa_s, bool with_prox) {
    for (Eigen::Index i = 0; i < p_; ++i) {
      // Each pair shares one sqrt (stable branches of retraction).
      const double u1 = v1[i];
      const double r1 = std::sqrt(u1 * u1 + 4.0 * kappa_s);
      z1r_[i] = u1 >= 0.0 ? 0.5 * (u1 + r1) : 2.0 * kappa_s / (r1 - u1);
      s1r_[i] = u1 >= 0.0 ? 2.0 * kappa_s / (r1 + u1) : 0.5 * (r1 - u1);
      const double u2 = v2[i];
      const double r2 = std::sqrt(u2 * u2 + 4.0 * kappa_s);
      z2r_[i] = u2 >= 0.0 ? 0.5 * (u2 + r2) : 2.0 * kappa_s / (r2 - u2);
      s2r_[i] = u2 >= 0.0 ? 2.0 * kappa_s / (r2 + u2) : 0.5 * (r2 - u2);
    }
    wQx_.noalias() = Q_ * x;
    wGtz_.noalias() = G_.transpose() * z2r_;
    rf1_ = wQx_ + q_ + wGtz_;
    if (m_ > 0) {
      wAty_.noalias() = A_.transpose() * y;
      rf1_ += wAty_;
      wAx_.noalias() = A_ * x;
      rf3_ = wAx_ - b_;
    }
    rf2_ = penalty_ - z1r_ - z2r_;
    wGx_.noalias() = G_ * x;
    rf4_ = s1r_ - t;
    rf5_ = s2r_ + wGx_ - h_ - t;
    if (with_prox) {
      rf1_ += rho_ * (x - xk_);
      rf2_ += rho_ * (t - tk_);
      if (m_ > 0) rf3_ += mu_eq_ * (yk_ - y);
      rf5_ += mu_in_ * (z2k_ - z2r_);
    }
  }

  // E^{-1} and the condensed row weights from the pair weights beta1/2:
  //   E = rho + beta1 + beta2,  Lambda = beta2 (rho + beta1) / E.
  // With beta = z/s (relax) these are qpax's elastic weights; with the
  // dual-proximal beta = b'/(b'(-v) + mu_in b') (forward pass) they are
  // bounded by 1/mu_in uniformly in kappa.
  void condense_weights(double rho) {
    einvr_ = ((beta1_ + beta2_).array() + rho).cwiseInverse();
    lamr_ = beta2_.array() * (beta1_.array() + rho) * einvr_.array();
  }

  // Newton direction of the condensed barrier system: reads the residuals
  // rf1_..rf5_ and the weights beta1_/beta2_/dhat1_/dhat2_/einvr_ (which
  // must match the factorization in llt), writes (dxr_, dtr_, dyr_,
  // dv1r_, dv2r_). Eliminating (dv1, dv2, dt) row-wise:
  //   Dhat dv1 = F4 - dt,   Dhat dv2 = F5 + G dx - dt,
  //   E dt = beta2 G dx + g,   g = beta1 F4 + beta2 F5 - F2,
  // leaves K dx = -F1 - (1/mu_eq) A'F3 - G'(beta2 (F5 - g/E)) and
  // dy = (A dx + F3)/mu_eq.
  void barrier_direction(double mu_eq,
                         const Eigen::LLT<MatrixXd, Eigen::Lower>& llt) {
    wr_ = beta1_.cwiseProduct(rf4_) + beta2_.cwiseProduct(rf5_) - rf2_;
    pvr_ = beta2_.cwiseProduct(rf5_ - einvr_.cwiseProduct(wr_));
    rhs_x_ = -rf1_;
    rhs_x_.noalias() -= G_.transpose() * pvr_;
    if (m_ > 0) {
      rhs_x_.noalias() -= (1.0 / mu_eq) * (A_.transpose() * rf3_);
    }
    dxr_ = llt.solve(rhs_x_);
    // One step of iterative refinement against the staged K_ (which
    // matches llt by construction): the condensed matrix carries
    // curvatures from rho up to max(1/mu_eq, 1/mu_in) row weights, and at
    // cond(K) ~ 1e10+ the plain LLT solve's cond * eps error becomes the
    // dual-residual floor in weakly-curved directions (e.g. barely
    // regularized force variables outside every inequality row).
    verr_.noalias() = K_.selfadjointView<Eigen::Lower>() * dxr_;
    verr_ -= rhs_x_;
    dxr_ -= llt.solve(verr_);
    Gdx_.noalias() = G_ * dxr_;
    dtr_ = einvr_.cwiseProduct(beta2_.cwiseProduct(Gdx_) + wr_);
    if (m_ > 0) {
      dyr_.noalias() = A_ * dxr_;
      dyr_ += rf3_;
      dyr_ /= mu_eq;
    }
    dv1r_ = (rf4_ - dtr_).cwiseQuotient(dhat1_);
    dv2r_ = (rf5_ + Gdx_ - dtr_).cwiseQuotient(dhat2_);
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
    barrier_kkt_fill(xr_, tr_, yr_, v1r_, v2r_, kappa_s, false);
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

  // Iterates and prox centers. (x, y, z) persist across solves for warm
  // starting; the barrier coordinates (tb, v1, v2) are re-derived from
  // them at the start of every solve (barrier_init) and live only within
  // it, together with their centers (tk, v1k, v2k, z1k, z2k).
  VectorXd x_, y_, z_;
  VectorXd xk_, yk_;
  VectorXd tb_, v1_, v2_;
  VectorXd tk_, v1k_, v2k_, z2k_, zk_;
  bool have_warm_ = false;
  bool explicit_warm_ = false;

  // Proximal / AL / central-path state (persists across solves)
  double rho_ = 0, mu_eq_ = 0, mu_in_ = 0;
  double kappa_ = 0;

  // Ruiz scaling state (identity when ruiz_ is false)
  bool ruiz_ = false;
  double c_s_ = 1.0;
  VectorXd dx_s_, de_s_, di_s_;             // cumulative scale factors
  VectorXd inv_cdx_, inv_de_, inv_di_;      // residual unscaling
  VectorXd y_us_, z_us_;                    // dual unscaling (de/c, di/c)

  // Factorization cache (Lambda fingerprint, see ensure_factor)
  bool matrix_dirty_ = true, factored_ = false;
  double f_rho_ = 0, f_mu_eq_ = 0, f_kdiag_min_ = 0;
  VectorXd f_lam_, g2_;
  int factor_retries_ = 0, factor_count_ = 0, iters_total_ = 0;

  // Residual scalars
  double primal_res_ = 0, dual_res_ = 0;
  double primal_res_rel_ = 0, dual_res_rel_ = 0;
  double primal_obj_ = 0, duality_gap_ = 0, duality_gap_rel_ = 0;

  // Workspace (allocated in setup, reused every iteration)
  VectorXd zhat_, t_, s2_, r_;
  VectorXd verr_, dyrhs_, rhs_x_, Gdx_;
  VectorXd wQx_, wGtz_, wAty_, wAx_, wGx_;
  VectorXd beta1_, beta2_, dhat1_, dhat2_;  // barrier pair weights
  VectorXd zc_;  // linearized certificate dual of the inner loop
  MatrixXd GS_, K_;
  Eigen::LLT<MatrixXd, Eigen::Lower> llt_;

  // relax() iterate and workspace (allocated in setup). The iterate and
  // factorization are kept separate from the solve() state so the
  // relaxation never disturbs warm starting or the factorization cache;
  // the residual/direction scratch (rf*, einvr_, lamr_, d*r_) is shared
  // with the forward inner loop, which recomputes it every iteration.
  VectorXd xr_, tr_, yr_, v1r_, v2r_;      // iterate (v parametrizes z, s)
  VectorXd z1r_, z2r_, s1r_, s2r_;         // retraction images of v
  VectorXd rf1_, rf2_, rf3_, rf4_, rf5_;   // barrier-KKT residuals
  VectorXd einvr_, lamr_, wr_, pvr_;       // condensation scalings
  VectorXd dxr_, dtr_, dyr_, dv1r_, dv2r_;        // Newton step
  Eigen::LLT<MatrixXd, Eigen::Lower> llt_r_;
  double relax_primal_res_ = 0, relax_dual_res_ = 0;
  double relax_merit_ = 0;  // squared 2-norm of the relaxed-KKT residual

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
