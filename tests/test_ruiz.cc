// Ruiz equilibration and automatic re-equilibration under drift.
//
// The row-scaling paths (setup-time invariance, vector setters, auto refresh
// on drifting G/A rows) are covered for every backend in test_solvers.cc;
// the exact remap of the PDAL's warm iterates is in test_pdal.cc. This suite
// covers the paths that only column and objective drift exercise, and the
// long-horizon behaviour of the drift-gated refresh, for the two backends
// that refresh (PDAL: reequilibrate() from the user frame; active set:
// rescale_matrices() on the next solve), plus the user-frame reporting for
// all three:
//
// (1) column substitution x = D x' (Q' = DQD, q' = Dq, G' = GD, A' = AD)
//     drives the dx (column) factors through the refresh, so the x remap
//     is checked rather than incidentally exercised;
// (2) objective scaling (alpha Q, alpha q, alpha penalty) leaves x unchanged
//     and scales every dual by alpha; the refresh must change the cost scale
//     (gamma != 1), which for the PDAL also remaps the relax() kappa;
// (3) 200 ticks of oscillating row AND column scales: cumulative factors and
//     the cost scale (gamma <= 1 per pass) must not degrade iteration counts
//     or accuracy, and the refresh must stay sparse;
// (4) reported Solution fields (objective, residuals, gap) are in the user's
//     frame: they must match an independent recomputation on the unscaled
//     data, and relative termination (eps_abs = 0) must work under Ruiz;
// (5) a drifting matrix update before any solve (no warm iterate) and on top
//     of a pending set_warm_start() must not corrupt the remapped state (PDAL);
// (6) limit_scaling through the update path: a row exploding past the 1e4
//     clamp in one tick, and a row decaying to the noise floor after setup;
// (7) the certified equality-infeasibility gate across a refresh, and the
//     frame independence of eq_infeasibility();
// (8) semantics pins (PDAL): ruiz_refresh_ratio <= 1 refreshes on any drift
//     past ruiz_tol, and settings.ruiz is latched at setup().

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <type_traits>

#include "elastiqp/elastiqp.hpp"
#include "problem_gen.hpp"
#include "test_util.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;
using test_util::Backend;
using test_util::Check;
using test_util::InfNorm;
using test_util::Label;
using test_util::RelDiff;
using test_util::SolveWith;
namespace pdal = elastiqp::pdal;

namespace {

template <class Solver>
typename Backend<Solver>::Settings Tight() {
  auto s = Backend<Solver>::Tight();
  s.ruiz = true;
  return s;
}

// eps_rel-only termination: the gap clause exists for the PDAL and IPM only
template <class S>
void RelaxGapTolerances(S& s) {
  if constexpr (!std::is_same_v<S, elastiqp::das::Settings>) {
    s.eps_duality_gap_abs = 0.0;
    s.eps_duality_gap_rel = 1e-9;
  }
}

// Whether the pending matrix update will trigger (PDAL: scaling_drift() is
// live; active set: only known after the solve, see Refreshed) / did
// trigger a refresh on the solve that just ran.
template <class Solver>
bool DriftedBefore(const Solver& s) {
  if constexpr (std::is_same_v<Solver, pdal::Solver>) {
    return s.scaling_drift() > s.settings.ruiz_refresh_ratio;
  } else {
    (void)s;
    return false;
  }
}
template <class Solver>
bool Refreshed(const Solver& s, bool drifted_before) {
  if constexpr (std::is_same_v<Solver, pdal::Solver>) {
    return drifted_before;
  } else {
    return s.rescaled();
  }
}

// The user-frame residuals and objectives the solver reports, recomputed
// from the returned Solution on the unscaled data
struct Recomputed {
  double primal_obj, dual_obj, primal_res, dual_res;
};

Recomputed Recompute(const MatrixXd& Q, const VectorXd& q, const MatrixXd& A,
                     const VectorXd& b, const MatrixXd& G, const VectorXd& h,
                     const VectorXd& penalty, const elastiqp::Solution& s) {
  Recomputed r;
  const VectorXd Qx = Q * s.x;
  r.primal_obj = 0.5 * s.x.dot(Qx) + q.dot(s.x) + penalty.dot(s.t);
  r.dual_obj = -0.5 * s.x.dot(Qx) - h.dot(s.z) -
               (b.size() > 0 ? b.dot(s.y) : 0.0);
  VectorXd stat = Qx + q + G.transpose() * s.z;
  if (b.size() > 0) stat += A.transpose() * s.y;
  r.dual_res = InfNorm(stat);
  r.primal_res = std::max(0.0, (G * s.x - h - s.t).maxCoeff());
  if (b.size() > 0) r.primal_res = std::max(r.primal_res, InfNorm(A * s.x - b));
  return r;
}

// Column-scaled copy of a problem: the substitution x = D x'
QPData ColumnScaled(const QPData& qp, const VectorXd& d) {
  QPData s = qp;
  s.Q = d.asDiagonal() * qp.Q * d.asDiagonal();
  s.q = qp.q.cwiseProduct(d);
  s.G = qp.G * d.asDiagonal();
  s.A = qp.A * d.asDiagonal();
  return s;
}

// Cells (1), (2), (3), (6), (7): the two refreshing backends
template <class Solver>
void DriftSuite() {
  using B = Backend<Solver>;
  std::mt19937 rng(7);
  const double kappa = 1e-4;
  char name[96];
  const auto L = [&](const char* cell) { return Label<Solver>(name, sizeof name, cell); };

  std::printf("[%s] Ruiz: column drift (x = D x') through auto refresh\n", B::name);
  {
    // D grows geometrically to 10^[-2, 2] over the ticks, so the column
    // factors drift past the refresh ratio several times. The invariance
    // D_k x'_k = x_0 is exact (q fixed), so the reference is tick 0.
    const int n = 30, m = 8, p = 200, ticks = 16;
    const QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    std::uniform_real_distribution<double> unif(-2.0, 2.0);
    VectorXd logd(n);
    for (int j = 0; j < n; ++j) logd[j] = unif(rng);

    Solver warm;
    warm.settings = Tight<Solver>();
    warm.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, penalty);
    const auto s0 = warm.solve();
    VectorXd r0x;
    if constexpr (B::has_relax) r0x = warm.relax(kappa).x;
    int refreshes = 0, warm_iters = 0, cold_iters = 0;
    double worst_dx = 0, worst_rdx = 0, worst_drift = 0, worst_kkt = 0;
    bool all_conv = s0.converged == 1;
    for (int k = 1; k <= ticks; ++k) {
      const VectorXd dk =
          Eigen::pow(10.0, (logd * (static_cast<double>(k) / ticks)).array())
              .matrix();
      const QPData qs = ColumnScaled(qp0, dk);
      warm.set_Q(qs.Q);
      warm.set_q(qs.q);
      warm.set_G(qs.G);
      warm.set_A(qs.A);
      const bool pre = DriftedBefore(warm);
      const auto ws = warm.solve();
      refreshes += Refreshed(warm, pre);
      worst_drift = std::max(worst_drift, warm.scaling_drift());
      Solver cold;
      cold.settings = Tight<Solver>();
      cold.setup(qs.Q, qs.q, qs.A, qs.b, qs.G, qs.h, penalty);
      const auto cs = cold.solve();
      all_conv &= ws.converged == 1 && cs.converged == 1;
      if constexpr (B::has_relax) {
        const auto wr = warm.relax(kappa);
        const auto cr = cold.relax(kappa);
        all_conv &= wr.converged == 1 && cr.converged == 1;
        worst_rdx = std::max(worst_rdx, RelDiff(cr.x, wr.x));
      }
      warm_iters += ws.iters;
      cold_iters += cs.iters;
      // Invariance in the base frame: D x' = x_0, duals unchanged
      worst_dx = std::max(
          {worst_dx, RelDiff(s0.x, dk.cwiseProduct(ws.x)),
           RelDiff(s0.z, ws.z), RelDiff(s0.y, ws.y)});
      worst_kkt = std::max(
          worst_kkt, problem_gen::ElasticKKTResidual(
                         qs.Q, qs.q, qs.A, qs.b, qs.G, qs.h, penalty, ws.x,
                         ws.t, ws.y, ws.z_t, ws.z));
    }
    std::printf("  refreshes=%d/%d worst_drift_after=%.2f | iters warm=%d "
                "cold=%d | worst_kkt=%9.2e\n",
                refreshes, ticks, worst_drift, warm_iters, cold_iters,
                worst_kkt);
    Check(L("column refresh fires and settles"),
          refreshes >= 2 && refreshes < ticks &&
              worst_drift <= warm.settings.ruiz_refresh_ratio,
          worst_drift, "drift");
    Check(L("D x' = x_0, duals invariant"),
          all_conv && worst_dx < 1e-5 && worst_kkt < 1e-5, worst_dx, "|dx|");
    if constexpr (B::has_relax) {
      Check(L("relax matches fresh setup"), worst_rdx < 1e-4, worst_rdx, "|dx|");
    }
  }

  std::printf("[%s] Ruiz: objective scale (alpha Q, alpha q, alpha w)\n", B::name);
  {
    // Scaling the whole objective by alpha leaves x unchanged and scales
    // every dual by alpha. alpha = 1e5 pushes the scaled Q columns past the
    // 1e4 limit_scaling clamp, the only way the cost scale (gamma < 1)
    // engages, so the refresh must remap the duals and the relax() kappa by
    // gamma as well as by dx. The 1e-6 step afterwards leaves Q below the
    // G/A column max, so the drift stays under the ratio and no refresh
    // fires: the stale scaling must still solve correctly.
    const int n = 24, m = 6, p = 120;
    const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    Solver warm;
    warm.settings = Tight<Solver>();
    warm.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    const auto s0 = warm.solve();
    bool ok = s0.converged == 1;
    double worst = 0, worst_r = 0;
    int refreshes = 0;
    double alpha_total = 1.0;
    for (const double alpha : {1e5, 1e-6, 30.0}) {
      alpha_total *= alpha;
      warm.set_Q(alpha_total * qp.Q);
      warm.set_q(alpha_total * qp.q);
      warm.set_penalty(alpha_total * penalty);
      const bool pre = DriftedBefore(warm);
      const auto ws = warm.solve();
      refreshes += Refreshed(warm, pre);
      Solver cold;
      cold.settings = Tight<Solver>();
      cold.setup(alpha_total * qp.Q, alpha_total * qp.q, qp.A, qp.b, qp.G,
                 qp.h, alpha_total * penalty);
      const auto cs = cold.solve();
      ok &= ws.converged == 1 && cs.converged == 1 &&
            warm.scaling_drift() <= warm.settings.ruiz_refresh_ratio;
      int wr_iters = 0, cr_iters = 0;
      if constexpr (B::has_relax) {
        // relax() at the same kappa: the relaxed point is NOT invariant to
        // alpha (s.z = kappa), so compare against a fresh solver instead
        const auto wr = warm.relax(kappa);
        const auto cr = cold.relax(kappa);
        ok &= wr.converged == 1 && cr.converged == 1;
        worst_r = std::max(worst_r, RelDiff(cr.x, wr.x));
        wr_iters = wr.iters;
        cr_iters = cr.iters;
      }
      worst = std::max({worst, RelDiff(s0.x, ws.x),
                        RelDiff(alpha_total * s0.z, ws.z),
                        RelDiff(alpha_total * s0.y, ws.y),
                        std::abs(ws.primal_obj - alpha_total * s0.primal_obj) /
                            std::max(1.0, std::abs(alpha_total * s0.primal_obj))});
      std::printf("  alpha=%8.1e drift_after=%.2f iters=%d (cold %d) relax "
                  "iters=%d (cold %d)\n",
                  alpha_total, warm.scaling_drift(), ws.iters, cs.iters,
                  wr_iters, cr_iters);
    }
    Check(L("refresh fires on objective scale"), refreshes >= 1, refreshes,
          "refreshes");
    Check(L("x invariant, duals scale by alpha"), ok && worst < 1e-5, worst,
          "rel");
    if constexpr (B::has_relax) {
      Check(L("relax matches fresh setup"), worst_r < 1e-4, worst_r, "|dx|");
    }
  }

  std::printf("[%s] Ruiz: 200 ticks of oscillating row + column scales\n", B::name);
  {
    // Rows (a fifth of G and A) swing over 3 decades with period 40 and
    // columns (a third) over 2 decades with period 60, plus q noise. The
    // refresh runs incrementally on top of the previous scaling, so the
    // cumulative factors and cost scale see every swing; the last period
    // must cost no more than the first.
    const int n = 30, m = 8, p = 200, ticks = 200, period = 40;
    const QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty0 = VectorXd::Constant(p, 10.0);
    Solver warm;
    warm.settings = Tight<Solver>();
    warm.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, penalty0);
    std::normal_distribution<double> dist;
    VectorXd q = qp0.q;
    int refreshes = 0, first_period = 0, last_period = 0, worst_tick = 0;
    int cold_checks = 0;
    double worst_dx = 0, worst_kkt = 0, worst_drift = 0, worst_obj = 0;
    bool all_conv = true;
    for (int k = 0; k < ticks; ++k) {
      const double phase = 2.0 * 3.14159265358979323846 * k;
      VectorXd rs = VectorXd::Ones(p), es = VectorXd::Ones(m),
               cs = VectorXd::Ones(n);
      for (int i = 0; i < p; i += 5) rs[i] = std::pow(10.0, 1.5 * std::sin(phase / period));
      for (int i = 0; i < m; i += 4) es[i] = std::pow(10.0, 1.5 * std::sin(phase / period));
      for (int j = 0; j < n; j += 3) cs[j] = std::pow(10.0, 1.0 * std::sin(phase / (1.5 * period)));
      for (int i = 0; i < n; ++i) q[i] += 0.01 * dist(rng);
      const MatrixXd Q = cs.asDiagonal() * qp0.Q * cs.asDiagonal();
      const VectorXd qs = q.cwiseProduct(cs);
      const MatrixXd G = rs.asDiagonal() * qp0.G * cs.asDiagonal();
      const VectorXd h = qp0.h.cwiseProduct(rs);
      const VectorXd penalty = penalty0.cwiseQuotient(rs);
      const MatrixXd A = es.asDiagonal() * qp0.A * cs.asDiagonal();
      const VectorXd b = qp0.b.cwiseProduct(es);
      warm.set_Q(Q);
      warm.set_q(qs);
      warm.set_G(G);
      warm.set_h(h);
      warm.set_penalty(penalty);
      warm.set_A(A);
      warm.set_b(b);
      const bool pre = DriftedBefore(warm);
      const auto ws = warm.solve();
      refreshes += Refreshed(warm, pre);
      worst_drift = std::max(worst_drift, warm.scaling_drift());
      all_conv &= ws.converged == 1;
      if (k > 0 && k < period) first_period += ws.iters;
      if (k >= ticks - period) last_period += ws.iters;
      if (k > 0) worst_tick = std::max(worst_tick, ws.iters);
      worst_kkt = std::max(
          worst_kkt, problem_gen::ElasticKKTResidual(
                         Q, qs, A, b, G, h, penalty, ws.x, ws.t, ws.y, ws.z_t,
                         ws.z));
      worst_obj = std::max(
          worst_obj,
          std::abs(ws.primal_obj - problem_gen::ElasticObjective(
                                       Q, qs, G, h, penalty, ws.x)) /
              std::max(1.0, std::abs(ws.primal_obj)));
      if (k % 10 == 9) {
        Solver cold;
        cold.settings = Tight<Solver>();
        cold.setup(Q, qs, A, b, G, h, penalty);
        const auto c = cold.solve();
        all_conv &= c.converged == 1;
        worst_dx = std::max(worst_dx, RelDiff(c.x, ws.x));
        ++cold_checks;
      }
    }
    std::printf("  refreshes=%d/%d worst_drift_after=%.2f | iters first "
                "period=%d last period=%d worst tick=%d | worst_kkt=%9.2e\n",
                refreshes, ticks, worst_drift, first_period, last_period,
                worst_tick, worst_kkt);
    Check(L("refresh stays sparse and settles"),
          refreshes >= 4 && refreshes <= ticks / 3 &&
              worst_drift <= warm.settings.ruiz_refresh_ratio,
          refreshes, "refreshes");
    Check(L("all ticks converge, match cold"),
          all_conv && cold_checks == ticks / 10 && worst_dx < 1e-5 &&
              worst_kkt < 1e-5,
          worst_dx, "|dx|");
    Check(L("no long-horizon degradation"),
          last_period <= 3 * first_period / 2, last_period, "iters");
    Check(L("reported objective in user frame"), worst_obj < 1e-6, worst_obj,
          "rel");
  }

  std::printf("[%s] Ruiz: limit_scaling through the update path\n", B::name);
  {
    // Row scaling by s with penalty / s leaves the elastic QP unchanged, so
    // the base solution is the reference throughout (q fixed).
    const int n = 16, m = 4, p = 60;
    const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    Solver solver;
    solver.settings = Tight<Solver>();
    solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    const auto s0 = solver.solve();

    // One row explodes by 1e7: the per-pass factor is clamped to 0.01, so
    // the refresh needs several passes to bring the row back to O(1)
    MatrixXd G = qp.G;
    VectorXd h = qp.h, pen = penalty;
    const int i_big = 3;
    G.row(i_big) *= 1e7;
    h[i_big] *= 1e7;
    pen[i_big] /= 1e7;
    solver.set_G(G);
    solver.set_h(h);
    solver.set_penalty(pen);
    const bool pre_big = DriftedBefore(solver);
    const auto s1 = solver.solve();
    const bool fired_big = Refreshed(solver, pre_big);
    const double dx1 = RelDiff(s0.x, s1.x);
    std::printf("  row x1e7: refreshed=%d drift after %.2f iters=%d\n",
                fired_big, solver.scaling_drift(), s1.iters);
    Check(L("row x1e7 refresh reaches O(1)"),
          fired_big && solver.scaling_drift() < 1.5 &&
              s0.converged == 1 && s1.converged == 1 && dx1 < 1e-5,
          dx1, "|dx|");

    // A different row decays to the noise floor (constraint and rhs both
    // ~1e-13, penalty unchanged): limit_scaling must leave it alone, the
    // drift must not report it, and the solution must match the problem
    // without that row
    const int i_noise = 7;
    std::normal_distribution<double> dist;
    for (int j = 0; j < n; ++j) G(i_noise, j) = 1e-13 * dist(rng);
    h[i_noise] = 1e-13 * dist(rng);
    solver.set_G(G);
    solver.set_h(h);
    const bool pre_noise = DriftedBefore(solver);
    const auto s2 = solver.solve();
    const bool fired_noise = Refreshed(solver, pre_noise);
    MatrixXd Gr(p - 1, n);
    VectorXd hr(p - 1), pr(p - 1);
    Gr << G.topRows(i_noise), G.bottomRows(p - i_noise - 1);
    hr << h.head(i_noise), h.tail(p - i_noise - 1);
    pr << pen.head(i_noise), pen.tail(p - i_noise - 1);
    const auto ref = SolveWith<Solver>(qp.Q, qp.q, qp.A, qp.b, Gr, hr, pr);
    const double dx2 = RelDiff(ref.x, s2.x);
    std::printf("  row -> noise: refreshed=%d iters=%d\n", fired_noise,
                s2.iters);
    Check(L("noise row ignored by drift, solved"),
          !fired_noise && s2.converged == 1 && ref.converged == 1 &&
              dx2 < 1e-5,
          dx2, "|dx|");
  }

  std::printf("[%s] Ruiz: equality-infeasibility gate across a refresh\n", B::name);
  {
    // The certificate is computed on unscaled (A, b): it must fire through a
    // drift-triggered refresh, report a frame-independent bound, and leave
    // the warm start usable once b is consistent again.
    const int n = 10, p = 20;
    const QPData qp = problem_gen::Feasible(rng, n, p);
    MatrixXd Ai(2, n);
    Ai.row(0) = problem_gen::Randn(rng, 1, n);
    Ai.row(1) = Ai.row(0);  // rank-deficient by construction
    VectorXd bc(2), bi(2);
    bc << 0.5, 0.5;
    bi << 0.0, 1.0;
    // The refresh is tripped through G (row scale s with penalty / s keeps
    // the QP, and so the warm iterate, unchanged); A is re-set unscaled
    const double s = 100.0;
    Solver on, off;  // library defaults (eps_rel = 0 gates)
    on.settings.ruiz = true;
    on.setup(qp.Q, qp.q, Ai, bc, qp.G, qp.h, 10.0);
    off.setup(qp.Q, qp.q, Ai, bi, qp.G, qp.h, 10.0);
    const auto good = on.solve();
    on.set_G(s * qp.G);
    on.set_h(s * qp.h);
    on.set_penalty(VectorXd::Constant(p, 10.0 / s));
    on.set_A(Ai);
    on.set_b(bi);
    const bool pre = DriftedBefore(on);
    const auto fail = on.solve();
    const bool fired = Refreshed(on, pre);
    const auto off_fail = off.solve();
    const double lb_err =
        std::abs(on.eq_infeasibility() - off.eq_infeasibility()) /
        off.eq_infeasibility();
    Check(L("gate fires through refresh"),
          good.status == elastiqp::Status::kSolved && fired &&
              on.scaling_drift() < 1.5 &&
              fail.status == elastiqp::Status::kInfeasible && fail.iters == 0 &&
              off_fail.status == elastiqp::Status::kInfeasible,
          on.eq_infeasibility(), "eq_infeas");
    Check(L("eq_infeasibility() frame-independent"), lb_err < 1e-9, lb_err,
          "rel");
    on.set_b(bc);
    const auto again = on.solve();
    Check(L("warm start survives gated tick"),
          again.status == elastiqp::Status::kSolved &&
              again.iters <= good.iters && on.eq_infeasibility() < 1e-10,
          again.iters, "iters");
  }

}

// Cell (4): every backend
template <class Solver>
void FieldsSuite() {
  using B = Backend<Solver>;
  std::mt19937 rng(11);
  const double kappa = 1e-4;
  char name[96];
  const auto L = [&](const char* cell) { return Label<Solver>(name, sizeof name, cell); };
  std::printf("[%s] Ruiz: reported fields are in the user's frame\n", B::name);
  {
    // Badly row- and column-scaled problem; every reported scalar must
    // match a recomputation from the returned (unscaled) Solution, and
    // agree with the Ruiz-off solve.
    const int n = 20, m = 5, p = 80;
    QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    std::uniform_real_distribution<double> unif(-3.0, 3.0);
    VectorXd penalty(p), d(n);
    for (int i = 0; i < p; ++i) {
      const double s = std::pow(10.0, unif(rng));
      qp.G.row(i) *= s;
      qp.h[i] *= s;
      penalty[i] = 10.0 / s;
    }
    for (int j = 0; j < n; ++j) d[j] = std::pow(10.0, 0.5 * unif(rng));
    qp = ColumnScaled(qp, d);
    Solver on, off;
    on.settings = Tight<Solver>();
    off.settings = Tight<Solver>();
    off.settings.ruiz = false;
    on.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    off.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    const auto son = on.solve();
    const auto soff = off.solve();
    const Recomputed rc =
        Recompute(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty, son);
    const double obj_scale = std::max(1.0, std::abs(rc.primal_obj));
    const double obj_err = std::abs(son.primal_obj - rc.primal_obj) / obj_scale;
    const double gap_err =
        std::abs(son.duality_gap - std::abs(rc.primal_obj - rc.dual_obj)) /
        obj_scale;
    const double res_err = std::max(std::abs(son.primal_res - rc.primal_res),
                                    std::abs(son.dual_res - rc.dual_res));
    std::printf("  ruiz on: iters=%d obj=%.6f pri=%.1e dua=%.1e gap=%.1e | "
                "off: iters=%d conv=%d obj=%.6f\n",
                son.iters, son.primal_obj, son.primal_res, son.dual_res,
                son.duality_gap, soff.iters, soff.converged, soff.primal_obj);
    Check(L("objective and gap match recomputation"),
          son.converged == 1 && obj_err < 1e-9 && gap_err < 1e-7, obj_err,
          "rel");
    Check(L("residuals match recomputation"), res_err < 1e-9, res_err, "abs");
    Check(L("objective matches ruiz off"),
          soff.converged == 1 &&
              std::abs(son.primal_obj - soff.primal_obj) / obj_scale < 1e-6 &&
              RelDiff(soff.x, son.x) < 1e-4,
          std::abs(son.primal_obj - soff.primal_obj) / obj_scale, "rel");

    // relax(): the same fields at the relaxed point
    if constexpr (B::has_relax) {
      const auto ron = on.relax(kappa);
      const Recomputed rr =
          Recompute(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty, ron);
      const double robj_err = std::abs(ron.primal_obj - rr.primal_obj) /
                              std::max(1.0, std::abs(rr.primal_obj));
      Check(L("relax objective in user frame"),
            ron.converged == 1 && robj_err < 1e-9, robj_err, "rel");
    }

    // Relative termination only (eps_abs = 0 is unreachable): the relative
    // norms are unscaled, so this must converge under Ruiz to the same point.
    Solver rel;
    rel.settings = Tight<Solver>();
    rel.settings.eps_abs = 0.0;
    rel.settings.eps_rel = 1e-9;
    RelaxGapTolerances(rel.settings);
    rel.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    const auto srel = rel.solve();
    const double rdx = RelDiff(soff.x, srel.x);
    Check(L("eps_rel-only termination under ruiz"),
          srel.converged == 1 && rdx < 1e-4, rdx, "|dx|");
  }

}

// Cells (5), (8): PDAL only
void PdalSuite() {
  std::mt19937 rng(13);
  std::printf("[pdal] Ruiz: refresh with no warm iterate / pending warm start\n");
  {
    const int n = 16, m = 4, p = 60;
    const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    std::uniform_real_distribution<double> unif(-3.0, 3.0);
    VectorXd rs(p);
    for (int i = 0; i < p; ++i) rs[i] = std::pow(10.0, unif(rng));
    const MatrixXd G = rs.asDiagonal() * qp.G;
    const VectorXd h = qp.h.cwiseProduct(rs);
    const VectorXd pen = penalty.cwiseQuotient(rs);
    const auto ref = SolveWith<pdal::Solver>(qp.Q, qp.q, qp.A, qp.b, qp.G,
                                             qp.h, penalty);

    // Drift before the first solve: the iterates are uninitialized
    pdal::Solver fresh;
    fresh.settings = Tight<pdal::Solver>();
    fresh.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    fresh.set_G(G);
    fresh.set_h(h);
    fresh.set_penalty(pen);
    const bool fired = fresh.scaling_drift() > fresh.settings.ruiz_refresh_ratio;
    const auto s1 = fresh.solve();
    const double dx1 = RelDiff(ref.x, s1.x);
    Check("first solve after drift", fired && s1.converged == 1 && dx1 < 1e-5,
          dx1, "|dx|");

    // Drift on top of an explicit warm start: the seed is remapped and must
    // still converge immediately (the seed IS the solution)
    pdal::Solver seeded;
    seeded.settings = Tight<pdal::Solver>();
    seeded.settings.warm_start = false;
    seeded.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    seeded.solve();
    seeded.set_warm_start(ref.x, ref.y, ref.z.cwiseQuotient(rs));
    seeded.set_G(G);
    seeded.set_h(h);
    seeded.set_penalty(pen);
    const auto s2 = seeded.solve();
    const double dx2 = RelDiff(ref.x, s2.x);
    Check("explicit warm start through refresh",
          s2.converged == 1 && s2.iters <= 2 && dx2 < 1e-5, s2.iters, "iters");
  }

  std::printf("[pdal] Ruiz: settings semantics\n");
  {
    const int n = 12, m = 3, p = 40;
    const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    MatrixXd G = qp.G;
    G.row(0) *= 1.05;  // drift 1.05: under the default ratio, stays

    // ruiz_refresh_ratio <= 1: any drift past ruiz_tol refreshes
    pdal::Solver eager, lazy;
    eager.settings = Tight<pdal::Solver>();
    lazy.settings = Tight<pdal::Solver>();
    eager.settings.ruiz_refresh_ratio = 1.0;
    eager.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, 10.0);
    lazy.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, 10.0);
    eager.set_G(G);
    lazy.set_G(G);
    const double before = eager.scaling_drift();
    eager.solve();
    lazy.solve();
    Check("ratio <= 1 refreshes on any drift",
          before > 1.01 && eager.scaling_drift() < 1.005 &&
              lazy.scaling_drift() > 1.01,
          eager.scaling_drift(), "drift");

    // settings.ruiz is latched at setup(): flipping it afterwards changes
    // nothing until the next setup()
    pdal::Solver late, off, early;
    late.settings = Tight<pdal::Solver>();
    off.settings = Tight<pdal::Solver>();
    early.settings = Tight<pdal::Solver>();
    late.settings.ruiz = false;
    off.settings.ruiz = false;
    late.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, 10.0);
    off.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, 10.0);
    late.settings.ruiz = true;
    late.reequilibrate();
    const auto sl = late.solve();
    const auto so = off.solve();
    Check("ruiz=true after setup() is inert",
          late.scaling_drift() == 1.0 && sl.iters == so.iters &&
              InfNorm(sl.x - so.x) == 0.0,
          sl.iters, "iters");
    early.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, 10.0);
    early.settings.ruiz = false;
    early.set_G(G);
    Check("ruiz=false after setup() keeps scaling",
          early.scaling_drift() > 1.01, early.scaling_drift(), "drift");
  }

}

}  // namespace

int main() {
  DriftSuite<pdal::Solver>();
  DriftSuite<elastiqp::das::Solver>();
  FieldsSuite<elastiqp::das::Solver>();
  FieldsSuite<pdal::Solver>();
  FieldsSuite<elastiqp::ipm::Solver>();
  PdalSuite();
  std::printf("%s\n", test_util::g_all_ok ? "ALL OK" : "FAILURES");
  return test_util::g_all_ok ? 0 : 1;
}
