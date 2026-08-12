// Differentiable collision detection between two 2D squares across
// consecutive timesteps (self-contained: Eigen + ElastiQP +
// elastiqp/kkt_vjp.hpp).
//
// Closest-point formulation (qpax "Collision Detection" section): with
// points p1, p2 constrained to polytopes A_i p_i <= b_i,
//
//   minimize    0.5 ||p1 - p2||^2
//   subject to  A1 p1 <= b1,  A2 p2 <= b2
//
// as an elastic QP in x = (p1, p2) with all-inequality rows and a large
// penalty (the instances are feasible, so the elastic solution matches
// the strict one). Square 1 is static at the origin; square 2 translates
// in a straight line past it at constant speed while rotating at a
// constant rate. Every tick queries distance = ||p1* - p2*|| and its
// gradient w.r.t. the moving square's configuration (cx, cy, th), via
// relax(kappa) + the implicit-KKT vjp: d(dist)/dtheta flows through
// G(th) and h(cx, cy, th) of the moving rows.
//
// The benchmark runs the identical trajectory two ways --
//   cold: a fresh solver per tick (setup + solve + relax from scratch)
//   warm: one persistent solver; per tick set_G/set_h, warm-started
//         solve(), and relax() warm-started from the previous tick's
//         relaxed iterate
// -- and reports per-tick timings for each phase, so cold vs warm and
// forward vs differentiation (relax + vjp) costs are directly
// comparable. The vjp cost is identical in both paths and timed once.
//
// Measured behavior on this problem (n = 4, p = 8): the warm forward
// solve cuts mean iterations ~9 -> ~7.7 (~1.4x on wall time, more at
// finer tick spacing). The relax() warm start is a
// no-op here -- the cold retraction start already converges in ~2 Newton
// steps (its floor), and the per-tick drift exceeds the warm-candidate
// adoption basin (0.02 sqrt(kappa), see Settings::relax_warm_start) even
// at 10x finer ticks -- so warm relax matches cold relax plus the ~0.2 us
// candidate evaluation. It is left on to exercise the mechanism and its
// graceful degradation; expect it to pay off on larger problems where
// the retraction start needs 5-9 steps, not here.
//
// A per-tick trace prints distance and gradient; away from corner
// transitions |d(dist)/d(cx,cy)| ~ 1 (translating a face 1:1 with the
// gap), dipping/smoothing near corner-face handoffs at scale kappa.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "elastiqp/elastiqp.hpp"
#include "elastiqp/kkt_vjp.hpp"

namespace {

using Eigen::Matrix2d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;

using Clock = std::chrono::steady_clock;

double UsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

double Median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double Mean(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}

// Scenario: unit squares (half-width 0.5). Square 2 starts left of the
// static square and sweeps past it with a minimum face gap of ~0.14
// (corner reach 0.5*sqrt(2), centers 1.35 apart at closest approach), so
// the distance stays strictly positive.
constexpr double kHalf = 0.5;
constexpr int kTicks = 601;
constexpr double kDt = 0.01;
constexpr double kX0 = -3.0, kY0 = 1.35, kVx = 1.0, kOmega = 1.0;
constexpr double kPenalty = 1e4;
constexpr double kKappa = 1e-3;
constexpr int kPasses = 6;  // pass 0 is warmup, excluded from stats

// Body-frame square: [I; -I] p_body <= half * ones.
MatrixXd BodySquare() {
  MatrixXd A(4, 2);
  A << 1, 0, -1, 0, 0, 1, 0, -1;
  return A;
}

struct TickProblem {
  MatrixXd G;      // 8 x 4: rows 0-3 static square on p1, 4-7 moving on p2
  VectorXd h;      // 8
  MatrixXd A2;     // moving square's world-frame rows (4 x 2)
  MatrixXd dA2;    // dA2/dth
  Vector2d c;      // moving square center
};

TickProblem MakeTick(int k) {
  const double t = k * kDt;
  const double th = kOmega * t;
  const MatrixXd Ab = BodySquare();
  const VectorXd bb = VectorXd::Constant(4, kHalf);

  // World-frame rows of a square at (c, th): A_body R(th)^T p <= b + A c.
  Matrix2d R, dR;
  R << std::cos(th), -std::sin(th), std::sin(th), std::cos(th);
  dR << -std::sin(th), -std::cos(th), std::cos(th), -std::sin(th);

  TickProblem tp;
  tp.c = Vector2d(kX0 + kVx * t, kY0);
  tp.A2 = Ab * R.transpose();
  tp.dA2 = Ab * dR.transpose();
  tp.G = MatrixXd::Zero(8, 4);
  tp.G.block(0, 0, 4, 2) = Ab;      // static square, identity pose
  tp.G.block(4, 2, 4, 2) = tp.A2;   // moving square
  tp.h.resize(8);
  tp.h << bb, bb + tp.A2 * tp.c;
  return tp;
}

struct TickResult {
  double dist;
  double gx, gy, gth;  // d(dist)/d(cx, cy, th)
};

// Distance value at the tight solution, gradient via the vjp at the
// relaxed point (mirrors the JAX wrapper: value tight, derivative path
// relaxed), chained through dG/dth, dh/dc, dh/dth of the moving rows.
// The KktVjp workspace is set up once by the caller (control-loop
// pattern: compute() does not allocate).
TickResult Gradients(const TickProblem& tp, const MatrixXd& Q,
                     const elastiqp::Solution& tight,
                     const elastiqp::Solution& relaxed,
                     elastiqp::KktVjp& vjp) {
  const Vector2d d =
      tight.x.head<2>() - tight.x.tail<2>();
  TickResult r;
  r.dist = d.norm();

  elastiqp::Cotangents ct;
  ct.x.resize(4);
  ct.x << d / r.dist, -d / r.dist;
  const elastiqp::DataGrads& g =
      vjp.compute(Q, MatrixXd(0, 4), tp.G, tp.h, relaxed, ct);

  const VectorXd hb = g.h.tail(4);
  const Vector2d gc = tp.A2.transpose() * hb;  // dh/dc = A2
  r.gx = gc.x();
  r.gy = gc.y();
  r.gth = (g.G.block(4, 2, 4, 2).array() * tp.dA2.array()).sum() +
          hb.dot(tp.dA2 * tp.c);
  return r;
}

}  // namespace

int main() {
  MatrixXd Q = MatrixXd::Zero(4, 4);
  Q.topLeftCorner(2, 2) = Matrix2d::Identity();
  Q.bottomRightCorner(2, 2) = Matrix2d::Identity();
  Q.topRightCorner(2, 2) = -Matrix2d::Identity();
  Q.bottomLeftCorner(2, 2) = -Matrix2d::Identity();
  const VectorXd q = VectorXd::Zero(4);

  // Warm path: one persistent solver across all ticks (and passes).
  elastiqp::Solver warm;
  {
    const TickProblem tp = MakeTick(0);
    warm.setup(Q, q, tp.G, tp.h, kPenalty);
  }
  elastiqp::KktVjp vjp;
  vjp.setup(4, 0, 8);

  std::vector<double> t_cold_setup, t_cold_solve, t_cold_relax;
  std::vector<double> t_warm_update, t_warm_solve, t_warm_relax, t_vjp;
  std::vector<double> it_cold, it_warm, rxit_cold, rxit_warm, fact_warm;
  std::vector<TickResult> trace(kTicks);
  bool all_ok = true;

  for (int pass = 0; pass < kPasses; ++pass) {
    const bool record = pass > 0;
    for (int k = 0; k < kTicks; ++k) {
      const TickProblem tp = MakeTick(k);

      // ---- cold: fresh solver, setup + solve + relax from scratch ----
      elastiqp::Solver cold;
      auto t0 = Clock::now();
      cold.setup(Q, q, tp.G, tp.h, kPenalty);
      const double us_setup = UsSince(t0);
      t0 = Clock::now();
      const elastiqp::Solution csol = cold.solve();
      const double us_csolve = UsSince(t0);
      t0 = Clock::now();
      const elastiqp::Solution crel = cold.relax(kKappa);
      const double us_crelax = UsSince(t0);
      all_ok = all_ok && csol.converged == 1 && crel.converged == 1;

      // ---- warm: vector+matrix update, warm solve, warm relax ----
      t0 = Clock::now();
      warm.set_G(tp.G);
      warm.set_h(tp.h);
      const double us_update = UsSince(t0);
      t0 = Clock::now();
      const elastiqp::Solution wsol = warm.solve();
      const double us_wsolve = UsSince(t0);
      const int wfacts = warm.factorizations();
      t0 = Clock::now();
      const elastiqp::Solution wrel = warm.relax(kKappa);
      const double us_wrelax = UsSince(t0);
      all_ok = all_ok && wsol.converged == 1 && wrel.converged == 1;

      // ---- gradient (identical for both paths; timed once) ----
      t0 = Clock::now();
      const TickResult tr = Gradients(tp, Q, wsol, wrel, vjp);
      const double us_vjp = UsSince(t0);

      if (record) {
        t_cold_setup.push_back(us_setup);
        t_cold_solve.push_back(us_csolve);
        t_cold_relax.push_back(us_crelax);
        t_warm_update.push_back(us_update);
        t_warm_solve.push_back(us_wsolve);
        t_warm_relax.push_back(us_wrelax);
        t_vjp.push_back(us_vjp);
        it_cold.push_back(csol.iters);
        it_warm.push_back(wsol.iters);
        rxit_cold.push_back(crel.iters);
        rxit_warm.push_back(wrel.iters);
        fact_warm.push_back(wfacts);
      }
      if (pass == kPasses - 1) trace[k] = tr;
    }
  }

  std::printf(
      "bench_collision_2d: two unit squares, one static, one translating\n"
      "at %.1f m/s + rotating at %.1f rad/s past it; %d ticks of %.0f ms,\n"
      "%d passes (first excluded), kappa = %g, times in us\n\n",
      kVx, kOmega, kTicks, kDt * 1e3, kPasses, kKappa);

  std::printf("trace (last pass, every 40th tick):\n");
  std::printf("%5s %6s %8s %9s %9s %9s %9s\n", "tick", "cx", "dist",
              "dd/dcx", "dd/dcy", "dd/dth", "|dd/dc|");
  for (int k = 0; k < kTicks; k += 40) {
    const TickResult& tr = trace[k];
    std::printf("%5d %6.2f %8.4f %9.4f %9.4f %9.4f %9.4f\n", k,
                kX0 + kVx * k * kDt, tr.dist, tr.gx, tr.gy, tr.gth,
                std::hypot(tr.gx, tr.gy));
  }

  const auto row = [](const char* name, const std::vector<double>& v) {
    std::printf("%-22s %9.2f %9.2f\n", name, Median(v), Mean(v));
  };
  std::printf("\nper-tick timings:%28s %9s\n", "median", "mean");
  row("cold setup", t_cold_setup);
  row("cold solve", t_cold_solve);
  row("cold relax", t_cold_relax);
  row("warm set_G/set_h", t_warm_update);
  row("warm solve", t_warm_solve);
  row("warm relax", t_warm_relax);
  row("vjp (backward)", t_vjp);

  std::printf("\nper-tick iterations:%25s %9s\n", "median", "mean");
  row("cold solve iters", it_cold);
  row("warm solve iters", it_warm);
  row("cold relax iters", rxit_cold);
  row("warm relax iters", rxit_warm);
  row("warm factorizations", fact_warm);

  const double cf = Median(t_cold_solve);
  const double wf = Median(t_warm_solve);
  const double cd = Median(t_cold_relax) + Median(t_vjp);
  const double wd = Median(t_warm_relax) + Median(t_vjp);
  std::printf("\nsummary (medians):\n");
  std::printf("  forward, cold -> warm:        %7.2f -> %7.2f us (%.1fx)\n",
              cf, wf, cf / wf);
  std::printf("  diff (relax+vjp), cold->warm: %7.2f -> %7.2f us (%.1fx)\n",
              cd, wd, cd / wd);
  std::printf("  diff cost / forward solve:    cold %.2fx, warm %.2fx\n",
              cd / cf, wd / wf);
  std::printf("  all ticks converged: %s\n", all_ok ? "yes" : "NO");
  return all_ok ? 0 : 1;
}
