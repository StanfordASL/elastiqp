// Active-set-specific behaviour of elastiqp::das::Solver (the elastic QP
// itself is covered for every backend in test_solvers.cc).
//
// 1. Random elastic QPs against the PDAL at tight tolerance: the dual
//    active-set answer must satisfy the elastic KKT conditions to solver
//    precision and match the objective, including rank-deficient Q and an
//    LP (proximal-point rounds) and hard rows (penalty = inf).
// 2. The saturation-creep cells of test_bcl_creep (same generator, same
//    seeds) as a warm chain: rows drifting across the saturation boundary
//    cost one working-set change per crossing, so the per-tick change
//    count is what is bounded; the full KKT residual is reported but not
//    gated (its complementarity products scale the row tolerance by the
//    1e4 penalty).
// 3. Robustness: a badly scaled parametrization with and without Ruiz;
//    dependent / inconsistent equalities (certificate before iterating);
//    every inequality row duplicated (dependent working sets, the cycle
//    guard's regime, refactors()); the drift-gated Ruiz refresh through a
//    row exploding past the limit_scaling clamp and a row decaying to the
//    noise floor.
// 4. Factorization reuse: constant Q, a fifth of G changing per tick,
//    same answers as a full refactorization and only those rows re-solved.
// 5. The pinch/release ticks of test_gap_creep: saturated -> active ->
//    inactive on a warm working set.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "drift_traj.hpp"
#include "elastiqp/elastiqp.hpp"
#include "problem_gen.hpp"
#include "test_util.hpp"

using drift_traj::Drift;
using drift_traj::Size;
using drift_traj::Structure;
using drift_traj::Trajectory;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;
using test_util::Check;
using test_util::InfNorm;
using test_util::SolveWith;
namespace das = elastiqp::das;
namespace pdal = elastiqp::pdal;

namespace {

double Now() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch())
      .count();
}

double Kkt(const QPData& qp, const VectorXd& penalty,
           const elastiqp::Solution& s) {
  return problem_gen::ElasticKKTResidual(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h,
                                         penalty, s.x, s.t, s.y, s.z_t, s.z);
}

elastiqp::Solution Reference(const QPData& qp, const VectorXd& penalty) {
  pdal::Settings st;
  st.eps_abs = 1e-9;
  st.eps_duality_gap_abs = 1e-9;
  return SolveWith<pdal::Solver>(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty,
                                 st);
}

// ---------------------------------------------------------------- part 1

void RandomCase(const char* label, const QPData& qp, const VectorXd& penalty,
                double kkt_tol, double obj_tol) {
  const elastiqp::Solution s =
      das::Solve(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
  const elastiqp::Solution ref = Reference(qp, penalty);
  const double obj =
      problem_gen::ElasticObjective(qp.Q, qp.q, qp.G, qp.h, penalty, s.x);
  const double obj_ref =
      problem_gen::ElasticObjective(qp.Q, qp.q, qp.G, qp.h, penalty, ref.x);
  char name[96];
  std::snprintf(name, sizeof(name), "%s: solved (it=%d, outer=%d, sat=%d)",
                label, s.iters, s.outer_iters, s.n_saturated);
  Check(name, s.converged == 1, static_cast<int>(s.status), "status");
  std::snprintf(name, sizeof(name), "%s: elastic KKT residual", label);
  Check(name, Kkt(qp, penalty, s) <= kkt_tol, Kkt(qp, penalty, s), "kkt");
  std::snprintf(name, sizeof(name), "%s: objective matches pdal", label);
  const double rel = std::abs(obj - obj_ref) / std::max(1.0, std::abs(obj_ref));
  Check(name, rel <= obj_tol && ref.converged == 1, rel, "rel");
}

// Hard rows (penalty = inf): feasible rows are the strict QP (compared
// against a high-penalty elastic solve), a conflicting pair of hard rows
// is certified infeasible (dual unbounded), and a conflict between a hard
// row and an elastic one moves only the elastic row.
void HardRows(std::mt19937& rng) {
  const double inf = std::numeric_limits<double>::infinity();
  const QPData qp = problem_gen::Feasible(rng, 12, 30);
  const VectorXd w_inf = VectorXd::Constant(30, inf);
  const elastiqp::Solution s = das::Solve(qp.Q, qp.q, qp.G, qp.h, w_inf);
  const elastiqp::Solution ref = Reference(qp, VectorXd::Constant(30, 1e6));
  Check("hard rows, feasible: solved", s.converged == 1,
        static_cast<int>(s.status), "status");
  Check("hard rows, feasible: matches strict QP",
        InfNorm(s.x - ref.x) <= 1e-6 && (s.z_t.array() == inf).all() &&
            s.t.maxCoeff() <= 1e-10,
        InfNorm(s.x - ref.x), "err");

  QPData c = problem_gen::Infeasible(rng, 12, 30, 1);  // rows 0/1 conflict
  const elastiqp::Solution bad = das::Solve(c.Q, c.q, c.G, c.h, w_inf);
  Check("conflicting hard rows: kInfeasible",
        bad.status == elastiqp::Status::kInfeasible && bad.converged == 0,
        static_cast<int>(bad.status), "status");

  VectorXd w_mix = VectorXd::Constant(30, inf);
  w_mix[1] = 10.0;  // the conflicting partner is elastic
  const elastiqp::Solution mix = das::Solve(c.Q, c.q, c.G, c.h, w_mix);
  const VectorXd r = c.G * mix.x - c.h;
  Check("hard vs elastic conflict: only the elastic row moves",
        mix.converged == 1 && r[1] > 0.5 && r.head(1).maxCoeff() <= 1e-9 &&
            r.tail(28).maxCoeff() <= 1e-9,
        r[1], "viol");
}

// Same problem in a badly scaled parametrization x = S x', rows scaled by e
// (penalties w/e keep the elastic objective identical), with and without
// Ruiz; the objective is invariant so it is compared to the reference of the
// well-scaled original.
void ScaledCase(std::mt19937& rng) {
  const QPData base = problem_gen::Infeasible(rng, 20, 40, 3);
  const VectorXd w0 = VectorXd::Constant(40, 10.0);
  const elastiqp::Solution ref = Reference(base, w0);
  const double obj_ref =
      problem_gen::ElasticObjective(base.Q, base.q, base.G, base.h, w0, ref.x);
  std::uniform_real_distribution<double> ucol(-3.0, 3.0), urow(-2.0, 2.0);
  VectorXd S(20), e(40);
  for (int i = 0; i < 20; ++i) S[i] = std::pow(10.0, ucol(rng));
  for (int i = 0; i < 40; ++i) e[i] = std::pow(10.0, urow(rng));
  QPData qp = base;
  qp.Q = S.asDiagonal() * base.Q * S.asDiagonal();
  qp.q = S.cwiseProduct(base.q);
  qp.G = e.asDiagonal() * base.G * S.asDiagonal();
  qp.h = e.cwiseProduct(base.h);
  const VectorXd w = w0.cwiseQuotient(e);
  for (int ruiz = 0; ruiz < 2; ++ruiz) {
    das::Settings st;
    st.ruiz = ruiz == 1;
    const elastiqp::Solution s =
        das::Solve(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w, st);
    const double obj =
        problem_gen::ElasticObjective(qp.Q, qp.q, qp.G, qp.h, w, s.x);
    const double rel =
        std::abs(obj - obj_ref) / std::max(1.0, std::abs(obj_ref));
    const double xerr = InfNorm(S.cwiseProduct(s.x) - ref.x);
    if (ruiz == 0) {  // informational: the failure mode Ruiz exists for
      std::printf(
          "  [info] badly scaled, ruiz=0: status=%d it=%d obj rel "
          "err=%.1e x err=%.1e\n",
          static_cast<int>(s.status), s.iters, rel, xerr);
      continue;
    }
    char name[96];
    std::snprintf(name, sizeof(name),
                  "badly scaled (cols 1e+-3, rows 1e+-2) ruiz: solved (it=%d)",
                  s.iters);
    Check(name, s.converged == 1, static_cast<int>(s.status), "status");
    Check("badly scaled ruiz: objective matches", rel <= 1e-6, rel, "rel");
    Check("badly scaled ruiz: x matches (orig frame)", xerr <= 1e-4, xerr,
          "err");
  }
}

// Dependent and inconsistent equalities: a duplicated row of A with the same
// b is consistent (solved, the dependent row dropped), with a different b it
// must be certified infeasible before any iteration.
void EqualityCases(std::mt19937& rng) {
  QPData qp = problem_gen::InfeasibleEq(rng, 20, 4, 30, 2);
  const VectorXd w = VectorXd::Constant(30, 10.0);
  MatrixXd A(5, 20);
  VectorXd b(5);
  A << qp.A, qp.A.row(1);
  b << qp.b, qp.b[1];
  const elastiqp::Solution ref = Reference(qp, w);
  das::Solver s;
  s.setup(qp.Q, qp.q, A, b, qp.G, qp.h, w);
  const elastiqp::Solution ok = s.solve();
  Check("duplicate consistent equality: solved", ok.converged == 1,
        static_cast<int>(ok.status), "status");
  Check("duplicate consistent equality: x matches",
        InfNorm(ok.x - ref.x) <= 1e-6, InfNorm(ok.x - ref.x), "err");
  b[4] += 0.5;
  s.set_b(b);
  const elastiqp::Solution bad = s.solve();
  Check("inconsistent equality: kInfeasible, no iterations",
        bad.status == elastiqp::Status::kInfeasible && bad.iters == 0,
        static_cast<int>(bad.status), "status");
  Check("inconsistent equality: eq_infeasibility ~ 0.5",
        std::abs(s.eq_infeasibility() - 0.5) < 1e-6, s.eq_infeasibility(),
        "lb");
  b[4] -= 0.5;
  s.set_b(b);
  const elastiqp::Solution again = s.solve();
  Check("consistent again: solved", again.converged == 1,
        static_cast<int>(again.status), "status");
}

// Every inequality row duplicated: dependent working sets at every step
// (singular-direction steps, ties in the KKT choice), the cycle guard's regime.
void DegenerateCase(std::mt19937& rng) {
  const QPData base = problem_gen::Infeasible(rng, 15, 30, 3);
  const VectorXd w0 = VectorXd::Constant(30, 10.0);
  const elastiqp::Solution ref = Reference(base, w0);
  QPData qp = base;
  qp.G.resize(60, 15);
  qp.h.resize(60);
  qp.G << base.G, base.G;
  qp.h << base.h, base.h;
  const VectorXd w = VectorXd::Constant(60, 5.0);  // pairs sum to the original
  das::Solver s;
  s.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
  const elastiqp::Solution sol = s.solve();
  char name[96];
  std::snprintf(name, sizeof(name),
                "duplicated rows: solved (it=%d, refactors=%d)", sol.iters,
                s.refactors());
  Check(name, sol.converged == 1, static_cast<int>(sol.status), "status");
  Check("duplicated rows: x matches", InfNorm(sol.x - ref.x) <= 1e-5,
        InfNorm(sol.x - ref.x), "err");
}

// Proximal-point loop: singular Q with no constraints and q in its null
// space is unbounded, so the loop cannot reach a fixed point and must not
// report kSolved; a positive definite Q takes exactly one round.
void ProxLoop(std::mt19937& rng) {
  const QPData qp = problem_gen::Feasible(rng, 10, 20);
  MatrixXd Qs = MatrixXd::Zero(10, 10);
  Qs(0, 0) = 1.0;
  das::Settings capped;
  capped.max_outer = 50;
  const elastiqp::Solution sing =
      das::Solve(Qs, qp.q, MatrixXd(0, 10), VectorXd(0), VectorXd(0), capped);
  Check("singular unconstrained (unbounded) not solved",
        sing.converged == 0 && sing.outer_iters == 50,
        static_cast<double>(sing.outer_iters), "outer");
  das::Solver s;
  s.setup(qp.Q, qp.q, qp.G, qp.h, 10.0);
  const elastiqp::Solution pd = s.solve();
  Check("positive definite Q: one proximal round, eps = 0",
        pd.converged == 1 && pd.outer_iters == 1 && !s.proximal(), s.prox_eps(),
        "eps");
  // Singular Q with constraints, bounded by a stiff box: proximal rounds.
  MatrixXd Gb(20 + 20, 10);
  VectorXd hb(40), wb(40);
  Gb << qp.G, MatrixXd::Identity(10, 10), -MatrixXd::Identity(10, 10);
  hb << qp.h, VectorXd::Constant(20, 5.0);
  wb << VectorXd::Constant(20, 10.0), VectorXd::Constant(20, 1e4);
  const elastiqp::Solution sq = das::Solve(Qs, qp.q, Gb, hb, wb);
  const elastiqp::Solution sref = SolveWith<pdal::Solver>(Qs, qp.q, Gb, hb, wb);
  Check("singular Q with constraints: proximal rounds, solved",
        sq.converged == 1 && sq.outer_iters >= 2 && sref.converged == 1 &&
            InfNorm(sq.x - sref.x) < 1e-5,
        static_cast<double>(sq.outer_iters), "outer");
  das::Settings refuse;
  refuse.eps_prox = 0.0;  // refuse singular Q
  const elastiqp::Solution rf = das::Solve(Qs, qp.q, qp.G, qp.h, 10.0, refuse);
  Check("eps_prox = 0 refuses singular Q (kNumerics)",
        rf.status == elastiqp::Status::kNumerics, static_cast<int>(rf.status),
        "status");
}

// Drift-gated Ruiz refresh through the update path: a row exploding past
// the 1e4 clamp in one tick (rescaled() fires, drift back to O(1)) and a
// row decaying to the noise floor (left alone, solved as if absent).
void RuizRefresh(std::mt19937& rng) {
  const int n = 16, m = 4, p = 60;
  const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
  const VectorXd penalty = VectorXd::Constant(p, 10.0);
  das::Settings tight;
  tight.eps_abs = 1e-8;  // eta_prox follows it
  das::Solver solver;
  solver.settings = tight;
  solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
  const elastiqp::Solution s0 = solver.solve();
  MatrixXd G = qp.G;
  VectorXd h = qp.h, pen = penalty;
  const int i_big = 3;
  G.row(i_big) *= 1e7;
  h[i_big] *= 1e7;
  pen[i_big] /= 1e7;
  solver.set_G(G);
  solver.set_h(h);
  solver.set_penalty(pen);
  const elastiqp::Solution s1 = solver.solve();
  const double dx1 = InfNorm(s0.x - s1.x);
  std::printf("  row x1e7: rescaled=%d drift_after=%.2f iters=%d\n",
              solver.rescaled(), solver.scaling_drift(), s1.iters);
  Check("row x1e7 refresh reaches O(1)",
        solver.rescaled() && solver.scaling_drift() < 1.5 &&
            s0.converged == 1 && s1.converged == 1 && dx1 < 1e-5,
        dx1, "|dx|");
  const int i_noise = 7;
  std::normal_distribution<double> dist;
  for (int j = 0; j < n; ++j) G(i_noise, j) = 1e-13 * dist(rng);
  h[i_noise] = 1e-13 * dist(rng);
  solver.set_G(G);
  solver.set_h(h);
  const elastiqp::Solution s2 = solver.solve();
  MatrixXd Gr(p - 1, n);
  VectorXd hr(p - 1), pr(p - 1);
  Gr << G.topRows(i_noise), G.bottomRows(p - i_noise - 1);
  hr << h.head(i_noise), h.tail(p - i_noise - 1);
  pr << pen.head(i_noise), pen.tail(p - i_noise - 1);
  const elastiqp::Solution ref =
      das::Solve(qp.Q, qp.q, qp.A, qp.b, Gr, hr, pr, tight);
  const double dx2 = InfNorm(ref.x - s2.x);
  std::printf("  row -> noise: rescaled=%d iters=%d\n", solver.rescaled(),
              s2.iters);
  Check("noise row solved, matches problem without it",
        !solver.rescaled() && s2.converged == 1 && ref.converged == 1 &&
            dx2 < 1e-5,
        dx2, "|dx|");
  // ruiz_refresh_ratio <= 1 refreshes on any drift past ruiz_tol
  das::Solver eager, lazy;
  eager.settings.ruiz_refresh_ratio = 1.0;
  eager.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
  lazy.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
  eager.solve();
  lazy.solve();
  MatrixXd G2 = qp.G;
  G2.row(0) *= 1.05;
  eager.set_G(G2);
  lazy.set_G(G2);
  eager.solve();
  lazy.solve();
  Check("ratio <= 1 refreshes on any drift",
        eager.rescaled() && !lazy.rescaled() && lazy.scaling_drift() > 1.01,
        lazy.scaling_drift(), "drift");
}

// Factorization reuse: hum-wbc-sized problem with constant Q and A, q/h
// drifting every tick and a fixed fifth of the G rows changing (the dense
// rows of a control QP), against the same chain with reuse disabled.
void ReuseBench() {
  const int n = 46, m = 18, p = 132, ticks = 200, changing = p / 5;
  std::mt19937 rng(7);
  const QPData base = problem_gen::InfeasibleEq(rng, n, m, p, 4);
  const VectorXd w = VectorXd::Constant(p, 1e3);
  std::normal_distribution<double> nd;
  double us[2] = {0, 0};
  int rows_updated[2] = {0, 0}, refactors[2] = {0, 0}, iters[2] = {0, 0};
  double xdiff = 0;
  std::vector<VectorXd> xs;
  for (int mode = 0; mode < 2; ++mode) {
    std::mt19937 drift(11);
    das::Solver s;
    s.settings.reuse_factorization = mode == 0;
    s.setup(base.Q, base.q, base.A, base.b, base.G, base.h, w);
    QPData qp = base;
    for (int k = 0; k < ticks; ++k) {
      for (int i = 0; i < n; ++i) qp.q[i] += 1e-2 * nd(drift);
      for (int i = 0; i < p; ++i) qp.h[i] += 1e-2 * nd(drift);
      for (int i = 0; i < changing; ++i)
        for (int j = 0; j < n; ++j) qp.G(i, j) += 1e-3 * nd(drift);
      s.set_q(qp.q);
      s.set_h(qp.h);
      s.set_G(qp.G);
      s.set_A(qp.A);  // unchanged: must be detected as such
      const double t0 = Now();
      const elastiqp::Solution& sol = s.solve();
      us[mode] += Now() - t0;
      rows_updated[mode] += s.rows_updated();
      refactors[mode] += s.refactored() ? 1 : 0;
      iters[mode] += sol.iters;
      if (sol.converged != 1) {
        xdiff = 1e9;
        break;
      }
      if (mode == 0) {
        xs.push_back(sol.x);
      } else {
        xdiff = std::max(xdiff, InfNorm(sol.x - xs[static_cast<size_t>(k)]));
      }
    }
  }
  std::printf(
      "  [reuse n=%d m=%d p=%d, %d/%d G rows change] reuse: %.1f "
      "us/tick, %.1f rows re-solved/tick, %d refactors, %.1f it | "
      "full: %.1f us/tick, %d refactors, %.1f it\n",
      n, m, p, changing, p, us[0] / ticks,
      static_cast<double>(rows_updated[0]) / ticks, refactors[0],
      static_cast<double>(iters[0]) / ticks, us[1] / ticks, refactors[1],
      static_cast<double>(iters[1]) / ticks);
  Check("reuse: same x as full refactorization", xdiff <= 1e-7, xdiff, "err");
  Check("reuse: only the changed rows re-solved (after the first tick)",
        rows_updated[0] == (m + p) + changing * (ticks - 1),
        static_cast<double>(rows_updated[0]) / ticks, "rows");
  Check("reuse: no refactorization after the first tick", refactors[0] == 1,
        refactors[0], "refactors");
}

void RandomSuite() {
  std::printf("Random elastic QPs (cold)\n");
  std::mt19937 rng(7);
  RandomCase("feasible n=20 p=40", problem_gen::Feasible(rng, 20, 40),
             VectorXd::Constant(40, 1e3), 1e-6, 1e-7);
  RandomCase("conflicts n=20 p=40 (4 pairs)",
             problem_gen::Infeasible(rng, 20, 40, 4),
             VectorXd::Constant(40, 10.0), 1e-6, 1e-7);
  RandomCase("conflicts+eq n=30 m=8 p=60 (6 pairs)",
             problem_gen::InfeasibleEq(rng, 30, 8, 60, 6),
             VectorXd::Constant(60, 10.0), 1e-6, 1e-7);
  {
    VectorXd w(60);
    for (int i = 0; i < 60; ++i) w[i] = (i % 3 == 0) ? 1e4 : 1.0;
    RandomCase("mixed penalties 1/1e4 (6 pairs)",
               problem_gen::InfeasibleEq(rng, 30, 8, 60, 6), w, 1e-5, 1e-7);
  }
  {
    // Rank-deficient Q (rank 10 of 20): proximal-point rounds.
    QPData qp = problem_gen::Infeasible(rng, 20, 40, 3);
    const MatrixXd B = problem_gen::Randn(rng, 10, 20);
    qp.Q = B.transpose() * B;
    RandomCase("rank-deficient Q (prox)", qp, VectorXd::Constant(40, 10.0),
               1e-5, 1e-6);
  }
  {
    // LP with a bounded feasible set: box rows on every variable.
    QPData qp = problem_gen::Infeasible(rng, 12, 24, 2);
    qp.Q.setZero();
    MatrixXd G(qp.G.rows() + 24, 12);
    VectorXd h(qp.h.size() + 24);
    G << qp.G, MatrixXd::Identity(12, 12), -MatrixXd::Identity(12, 12);
    h << qp.h, VectorXd::Constant(24, 5.0);
    qp.G = G;
    qp.h = h;
    VectorXd w(48);  // box rows stiff enough to bound the elastic LP
    w << VectorXd::Constant(24, 10.0), VectorXd::Constant(24, 1e4);
    // Full-KKT tolerance loosened: the 1e4 box penalty multiplies the row
    // tolerance in the complementarity products.
    RandomCase("LP, Q = 0 (prox)", qp, w, 2e-2, 1e-6);
  }
  HardRows(rng);
  ScaledCase(rng);
  EqualityCases(rng);
  DegenerateCase(rng);
  ProxLoop(rng);
  RuizRefresh(rng);
  ReuseBench();
}

// ---------------------------------------------------------------- part 2

void CreepCell(Structure st, Size sz, const VectorXd& penalty, double sigma,
               const char* label, int max_tick_iters) {
  const int ticks = 100;
  const unsigned seed = 91u * static_cast<unsigned>(sz.n) +
                        static_cast<unsigned>(sz.p) +
                        7u * static_cast<unsigned>(st);
  const Trajectory traj = drift_traj::MakeTrajectory(sz, st, penalty, sigma,
                                                     Drift::kQH, seed, ticks);

  das::Solver d;
  d.setup(traj.base.Q, traj.q[0], traj.base.A, traj.b[0], traj.base.G,
          traj.h[0], traj.penalty);
  pdal::Solver e;
  e.settings.eps_abs = 1e-5;
  e.settings.eps_rel = 0;
  e.settings.ruiz = true;
  e.setup(traj.base.Q, traj.q[0], traj.base.A, traj.b[0], traj.base.G,
          traj.h[0], traj.penalty);

  int fails = 0, total = 0, e_total = 0, e_fails = 0, sat_max = 0;
  double kkt_worst = 0, d_us = 0, e_us = 0, xdiff_worst = 0;
  double res_inactive = 0, res_saturated = 0, res_active = 0, res_stat = 0,
         res_eq = 0;
  int warm_worst = 0, cold_iters = 0;
  QPData qp = traj.base;
  for (int k = 0; k < ticks; ++k) {
    qp.q = traj.q[k];
    qp.h = traj.h[k];
    if (sz.m > 0) qp.b = traj.b[k];
    d.set_q(qp.q);
    d.set_h(qp.h);
    if (sz.m > 0) d.set_b(qp.b);
    double t0 = Now();
    const elastiqp::Solution s = d.solve();
    d_us += Now() - t0;
    e.set_q(qp.q);
    e.set_h(qp.h);
    if (sz.m > 0) e.set_b(qp.b);
    t0 = Now();
    const elastiqp::Solution es = e.solve();
    e_us += Now() - t0;
    if (s.converged != 1) fails++;
    if (es.converged != 1) e_fails++;
    total += s.iters;
    e_total += es.iters;
    if (k == 0) cold_iters = s.iters;
    if (k > 0) warm_worst = std::max(warm_worst, s.iters);
    sat_max = std::max(sat_max, s.n_saturated);
    kkt_worst = std::max(kkt_worst, Kkt(qp, traj.penalty, s));
    // Residual components (the full KKT residual multiplies a 1e-6 row
    // tolerance by the 1e4 penalty in its complementarity products).
    {
      const VectorXd r = qp.G * s.x - qp.h;
      for (int i = 0; i < r.size(); ++i) {
        if (s.z[i] <= 0) {
          res_inactive = std::max(res_inactive, r[i]);
        } else if (s.z[i] >= traj.penalty[i]) {
          res_saturated = std::max(res_saturated, -r[i]);
        } else {
          res_active = std::max(res_active, std::abs(r[i]));
        }
      }
      VectorXd stat = qp.Q * s.x + qp.q + qp.G.transpose() * s.z;
      if (sz.m > 0) stat += qp.A.transpose() * s.y;
      res_stat = std::max(res_stat, InfNorm(stat));
      if (sz.m > 0) res_eq = std::max(res_eq, InfNorm(qp.A * s.x - qp.b));
    }
    xdiff_worst = std::max(xdiff_worst, InfNorm(s.x - es.x));
  }
  std::printf(
      "  [%s] as %.1f us/tick %.1f it/tick (cold tick %d, warm worst "
      "%d, sat<=%d) | pdal %.1f us/tick %.1f it/tick (%d fails) | "
      "max|x_as - x_pdal| %.1e\n",
      label, d_us / ticks, static_cast<double>(total) / ticks, cold_iters,
      warm_worst, sat_max, e_us / ticks, static_cast<double>(e_total) / ticks,
      e_fails, xdiff_worst);
  std::printf(
      "  [%s] worst residuals: stat %.1e eq %.1e active|r| %.1e "
      "inactive viol %.1e saturated slack %.1e (full KKT incl. "
      "complementarity %.1e)\n",
      label, res_stat, res_eq, res_active, res_inactive, res_saturated,
      kkt_worst);
  char name[96];
  std::snprintf(name, sizeof(name), "%s: all ticks solve", label);
  Check(name, fails == 0, fails, "fails");
  std::snprintf(name, sizeof(name), "%s: stationarity + equalities", label);
  Check(name, std::max(res_stat, res_eq) <= 1e-6, std::max(res_stat, res_eq),
        "res");
  std::snprintf(name, sizeof(name), "%s: active rows hold", label);
  Check(name, res_active <= 1e-6, res_active, "res");
  std::snprintf(name, sizeof(name), "%s: inactive/saturated rows within tol",
                label);
  // Rows may violate by settings.eps_abs in user units (2x for roundoff)
  Check(name, std::max(res_inactive, res_saturated) <= 2 * d.settings.eps_abs,
        std::max(res_inactive, res_saturated), "res");
  std::snprintf(name, sizeof(name), "%s: x matches pdal", label);
  Check(name, xdiff_worst <= 1e-3, xdiff_worst, "err");
  std::snprintf(name, sizeof(name),
                "%s: warm ticks' working-set changes bounded", label);
  Check(name, warm_worst <= max_tick_iters, warm_worst, "iters");
}

void CreepSuite() {
  std::printf("BCL creep cells, warm chain (qh drift)\n");
  CreepCell(Structure::kFeas, {14, 0, 100}, VectorXd::Constant(100, 1e4), 1e-4,
            "feas uniform 1e4", 80);
  CreepCell(Structure::kDegen, {14, 0, 100}, VectorXd::Constant(100, 1e4), 1e-4,
            "degen uniform 1e4", 160);
  VectorXd spike(200);
  for (int i = 0; i < 200; ++i) spike[i] = (i % 8 == 0) ? 1e4 : 10.0;
  CreepCell(Structure::kInfeas, {30, 8, 200}, spike, 1e-3,
            "infeas spike 10/1e4", 120);
}

// ---------------------------------------------------------------- part 3

void GapCreep() {
  std::printf("Deactivation creep: pinch-release warm tick\n");
  const MatrixXd Q = MatrixXd::Identity(2, 2);
  const VectorXd penalty = VectorXd::Constant(3, 1000.0);
  VectorXd q463(2), h463(3), q464(2), h464(3);
  MatrixXd G463(3, 2), G464(3, 2);
  q463 << 0.21920301334862091, 4.1139326748986216;
  G463 << 1, 0, 0, 1, 0.048596449461558944, -0.99361503907741344;
  h463 << 2.4088071318512285, -5.7350402445204285, 2.1510357789205634;
  q464 << -1.2682556160930214, 3.9605375674075187;
  G464 << 1, 0, 0, 1, 0.076622662009877995, -0.98842746321542529;
  h464 << 5.3894716545331169, -5.4329140545125938, 5.017615727279078;

  const elastiqp::Solution ref = pdal::Solve(Q, q464, G464, h464, penalty);
  das::Solver s;
  s.setup(Q, q463, G463, h463, penalty);
  const elastiqp::Solution pinch = s.solve();
  Check("pinch tick solves", pinch.converged == 1, pinch.iters, "iters");
  Check("pinch duals ride the cap", pinch.z.maxCoeff() > 0.99e3,
        pinch.z.maxCoeff(), "zmax");
  s.set_q(q464);
  s.set_G(G464);
  s.set_h(h464);
  const elastiqp::Solution rel = s.solve();
  Check("release tick solves warm", rel.converged == 1, rel.iters, "iters");
  Check("release tick iters bounded", rel.iters <= 8, rel.iters, "iters");
  const double xerr = InfNorm(rel.x - ref.x);
  Check("release x matches pdal cold", xerr <= 1e-4, xerr, "err");
  Check("release duals leave the cap", rel.z.maxCoeff() < 0.5e3,
        rel.z.maxCoeff(), "zmax");
}

}  // namespace

int main() {
  RandomSuite();
  CreepSuite();
  GapCreep();
  std::printf(test_util::g_all_ok ? "\nAll active-set tests passed.\n"
                                  : "\nFAILURES\n");
  return test_util::g_all_ok ? 0 : 1;
}
