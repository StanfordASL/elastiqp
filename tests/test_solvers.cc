// Correctness of the three ElastiQP backends on the elastic QP itself.
//
// Self-contained (Eigen only). Every cell below runs for the active-set
// (elastiqp::das), PDAL (elastiqp::pdal) and interior-point (elastiqp::ipm)
// solvers through the same template, on the same random instances, and
// checks against the IPM solver at tight tolerance as the oracle (it
// reaches the same elastic solution by a different method; the benchmarks
// repo cross-validates all three against vanilla PIQP). A closing suite
// solves each instance with all three backends and cross-checks x, the
// objective and the duals directly.
//
// Checks:
// (1) hard-constrained-yet-feasible problems, where the elastic result must
//     coincide with the strict QP (t at the backend's exactness);
// (2) infeasible problems, where the elastic solution must match the
//     reference solved on the identical elastic problem;
// (3) equalities (feasible, inconsistent, rank deficient), the certified
//     infeasibility gate, edge dimensions (p = 0, m = 0);
// (4) warm starts across drifting data, penalties and matrices, Ruiz
//     equilibration, exact-penalty behaviour, and Solution invariants.
// Method-specific behaviour lives in test_das.cc, test_pdal.cc, test_ipm.cc.

#include <cmath>
#include <cstdio>
#include <random>
#include <tuple>

#include "drift_traj.hpp"
#include "elastiqp/elastiqp.hpp"
#include "problem_gen.hpp"
#include "test_util.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;
using test_util::Backend;
using test_util::Check;
using test_util::InfNorm;
using test_util::IpmRef;
using test_util::Label;
using test_util::MaxT;
using test_util::SolveWith;

namespace {

// Reference for the ORIGINAL strict problem (hard A, b and hard G, h):
// the elastic reference with a penalty far above any inequality dual, so
// the slacks vanish and z is the hard problem's inequality dual.
struct StrictSol {
  VectorXd x, z;
  int converged;
};

StrictSol SolveStrictRef(const QPData& qp, double penalty = 1e6) {
  const elastiqp::Solution s =
      IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h,
             VectorXd::Constant(qp.h.size(), penalty));
  StrictSol r;
  r.converged = s.converged == 1 && (s.t.size() == 0 || s.t.maxCoeff() < 1e-8);
  r.x = s.x;
  r.z = s.z;
  return r;
}

double Kkt(const QPData& qp, const VectorXd& w, const elastiqp::Solution& s) {
  return problem_gen::ElasticKKTResidual(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h,
                                         w, s.x, s.t, s.y, s.z_t, s.z);
}
double Kkt(const MatrixXd& Q, const VectorXd& q, const MatrixXd& A,
           const VectorXd& b, const MatrixXd& G, const VectorXd& h,
           const VectorXd& w, const elastiqp::Solution& s) {
  return problem_gen::ElasticKKTResidual(Q, q, A, b, G, h, w, s.x, s.t, s.y,
                                         s.z_t, s.z);
}

template <class Solver>
void GenericSuite() {
  using B = Backend<Solver>;
  std::mt19937 rng(42);  // same instances for every backend
  char name[96];
  const auto L = [&](const char* cell) { return Label<Solver>(name, sizeof name, cell); };
  char cell[64];

  std::printf("[%s] feasible => matches strict QP, t ~ 0\n", B::name);
  for (auto [n, p] : {std::pair{10, 12}, {14, 100}, {58, 500}}) {
    const QPData qp = problem_gen::Feasible(rng, n, p);
    const auto sol = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h,
                                       VectorXd::Constant(p, 1e3));
    const StrictSol ref = SolveStrictRef(qp);
    const double dx = InfNorm(sol.x - ref.x);
    std::snprintf(cell, sizeof cell, "n=%d p=%d", n, p);
    Check(L(cell),
          sol.converged == 1 && ref.converged == 1 && dx < 1e-5 &&
              MaxT(sol) <= B::slack_tol,
          dx, "|dx|");
  }

  std::printf("[%s] KKT residual of the elastic problem\n", B::name);
  for (auto [n, p] : {std::pair{14, 60}, {30, 200}}) {
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    const VectorXd w = VectorXd::Constant(p, 10.0);
    const auto sol = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h, w);
    const double res = Kkt(qp, w, sol);
    std::snprintf(cell, sizeof cell, "n=%d p=%d infeasible", n, p);
    Check(L(cell), sol.converged == 1 && res < 1e-6, res, "kkt");
  }

  std::printf("[%s] infeasible => matches the IPM reference\n", B::name);
  for (auto [n, p] : {std::pair{8, 10}, {14, 100}, {58, 300}}) {
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    const VectorXd w = VectorXd::Constant(p, 10.0);
    const auto sol = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h, w);
    const auto ref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const double dx = InfNorm(sol.x - ref.x);
    std::snprintf(cell, sizeof cell, "n=%d p=%d", n, p);
    Check(L(cell),
          sol.converged == 1 && ref.converged == 1 && dx < 1e-5 &&
              MaxT(sol) > 0.1,
          dx, "|dx|");
  }

  std::printf("[%s] per-constraint penalty weights\n", B::name);
  {
    const int n = 10, p = 40;
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    VectorXd w(p);
    for (int i = 0; i < p; ++i) w[i] = (i % 2) ? 100.0 : 5.0;
    const auto sol = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h, w);
    const auto ref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const double dx = InfNorm(sol.x - ref.x);
    Check(L("n=10 p=40 mixed penalty"),
          sol.converged == 1 && ref.converged == 1 && dx < 1e-5, dx, "|dx|");
  }

  std::printf("[%s] PSD-only Q (rank deficient)\n", B::name);
  {
    const int n = 20, p = 60;
    QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    const MatrixXd R = problem_gen::Randn(rng, n / 2, n);
    qp.Q = R.transpose() * R;  // rank n/2
    const VectorXd w = VectorXd::Constant(p, 10.0);
    const auto sol = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h, w);
    const auto ref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const double dx = InfNorm(sol.x - ref.x);
    Check(L("n=20 (rank 10) p=60"),
          sol.converged == 1 && ref.converged == 1 && dx < 1e-4, dx, "|dx|");
  }

  std::printf("[%s] hard equalities, feasible ineqs => strict reference\n",
              B::name);
  for (auto [n, m, p] : {std::tuple{14, 4, 60}, {30, 8, 200}, {58, 15, 500}}) {
    const QPData qp = problem_gen::RandomFeasible(rng, n, m, p);
    const auto sol = SolveWith<Solver>(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h,
                                       VectorXd::Constant(p, 1e3));
    const StrictSol ref = SolveStrictRef(qp);
    const double dx = InfNorm(sol.x - ref.x);
    const double eq_res = InfNorm(qp.A * sol.x - qp.b);
    std::snprintf(cell, sizeof cell, "n=%d m=%d p=%d", n, m, p);
    Check(L(cell),
          sol.converged == 1 && ref.converged == 1 && dx < 1e-5 &&
              eq_res < 1e-6 && MaxT(sol) <= B::slack_tol,
          std::max(dx, eq_res), "|dx|,eq");
  }

  std::printf("[%s] equalities hold when inequalities are infeasible\n",
              B::name);
  for (auto [n, m, p] :
       {std::tuple{14, 4, 100}, {30, 8, 200}, {58, 15, 400}}) {
    const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd w = VectorXd::Constant(p, 10.0);
    const auto sol = SolveWith<Solver>(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const auto ref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const double dx = InfNorm(sol.x - ref.x);
    const double eq_res = InfNorm(qp.A * sol.x - qp.b);
    const double kkt = Kkt(qp, w, sol);
    std::snprintf(cell, sizeof cell, "n=%d m=%d p=%d eq=%.1e", n, m, p,
                  eq_res);
    Check(L(cell),
          sol.converged == 1 && ref.converged == 1 && dx < 1e-5 &&
              eq_res < 1e-6 && kkt < 1e-6 && MaxT(sol) > 0.1,
          dx, "|dx|");
  }

  std::printf("[%s] equality-only edge case (p=0)\n", B::name);
  {
    const QPData qp = problem_gen::RandomFeasible(rng, 20, 8, 0);
    const MatrixXd G(0, 20);
    const VectorXd h(0), w(0);
    const auto sol = SolveWith<Solver>(qp.Q, qp.q, qp.A, qp.b, G, h, w);
    const int n = 20, m = 8;
    MatrixXd Kf = MatrixXd::Zero(n + m, n + m);
    Kf.topLeftCorner(n, n) = qp.Q;
    Kf.topRightCorner(n, m) = qp.A.transpose();
    Kf.bottomLeftCorner(m, n) = qp.A;
    VectorXd rhs(n + m);
    rhs << -qp.q, qp.b;
    const VectorXd xy = Kf.colPivHouseholderQr().solve(rhs);
    const double dx = InfNorm(sol.x - xy.head(n));
    Check(L("n=20 m=8 p=0"), sol.converged == 1 && dx < 1e-7, dx, "|dx|");

    // Inconsistent equalities at eps_rel > 0 (the certificate can only
    // gate the absolute clause): must not be reported solved.
    MatrixXd Ai(2, n);
    Ai.row(0) = qp.A.row(0);
    Ai.row(1) = qp.A.row(0);
    VectorXd bi(2);
    bi << 0.0, 1.0;
    const auto bad = SolveWith<Solver>(qp.Q, qp.q, Ai, bi, G, h, w);
    Check(L("inconsistent A x = b not solved"),
          bad.converged == 0 && bad.status != elastiqp::Status::kSolved,
          static_cast<double>(bad.status), "status");

    // Singular Q with no constraints: unbounded along the null space
    // (q has a component there); must not be reported solved.
    MatrixXd Qs = MatrixXd::Zero(n, n);
    Qs(0, 0) = 1.0;
    typename B::Settings capped = B::Tight();
    B::CapOuter(capped, 50);
    const auto sing = SolveWith<Solver>(Qs, qp.q, MatrixXd(0, n), VectorXd(0),
                                        G, h, w, capped);
    Check(L("singular unconstrained not solved"), sing.converged == 0,
          static_cast<double>(sing.status), "status");
  }

  std::printf("[%s] inconsistent equalities with p>0 do not converge\n",
              B::name);
  {
    const int n = 10, p = 20;
    const QPData qp = problem_gen::Feasible(rng, n, p);
    MatrixXd Ai(2, n);
    Ai.row(0) = problem_gen::Randn(rng, 1, n);
    Ai.row(1) = Ai.row(0);
    VectorXd bi(2);
    bi << 0.0, 1.0;
    Solver solver;
    solver.settings = B::Tight();
    B::CapOuter(solver.settings, 40);
    solver.setup(qp.Q, qp.q, Ai, bi, qp.G, qp.h, VectorXd::Constant(p, 10.0));
    const auto& sol = solver.solve();
    Check(L("inconsistent eq (p>0) not kSolved"),
          sol.converged == 0 && sol.status != elastiqp::Status::kSolved,
          static_cast<double>(sol.status), "status");
  }

  // The two blocks above run at eps_rel > 0, where the ingestion-time
  // consistency certificate cannot gate. This block exercises the gate
  // itself at the library default eps_rel = 0.
  std::printf("[%s] certified equality-infeasibility gate\n", B::name);
  {
    std::mt19937 gate_rng(2026);
    const int n = 10, p = 20;
    const QPData qp = problem_gen::Feasible(gate_rng, n, p);
    MatrixXd Ai(2, n);
    Ai.row(0) = problem_gen::Randn(gate_rng, 1, n);
    Ai.row(1) = Ai.row(0);  // rank-deficient by construction
    VectorXd bc(2), bi(2);
    bc << 0.5, 0.5;  // consistent (duplicated row, same rhs)
    bi << 0.0, 1.0;  // inconsistent
    const VectorXd w = VectorXd::Constant(p, 10.0);

    Solver solver;  // default settings: eps_rel 0
    solver.setup(qp.Q, qp.q, Ai, bi, qp.G, qp.h, w);
    const auto fail = solver.solve();
    Check(L("inconsistent b fails fast (0 iters)"),
          fail.status == elastiqp::Status::kInfeasible && fail.converged == 0 &&
              fail.iters == 0 && solver.eq_infeasibility() > 0.1,
          solver.eq_infeasibility(), "eq_infeas");

    solver.set_b(bc);
    const auto good = solver.solve();
    Check(L("consistent b, rank-deficient A solves"),
          good.status == elastiqp::Status::kSolved &&
              solver.eq_infeasibility() < 1e-10,
          InfNorm(Ai * good.x - bc), "eq_res");

    solver.set_b(bi);
    const auto fail2 = solver.solve();
    Check(L("set_b to inconsistent fails fast"),
          fail2.status == elastiqp::Status::kInfeasible && fail2.iters == 0,
          solver.eq_infeasibility(), "eq_infeas");

    solver.set_b(bc);
    const auto good2 = solver.solve();
    Check(L("warm start survives transient bad tick"),
          good2.status == elastiqp::Status::kSolved &&
              good2.iters <= good.iters,
          static_cast<double>(good2.iters), "iters");

    // Certificate is computed on unscaled data: gate must fire under Ruiz.
    Solver rz;
    rz.settings.ruiz = true;
    rz.setup(qp.Q, qp.q, Ai, bi, qp.G, qp.h, w);
    const auto rzfail = rz.solve();
    Check(L("gate fires with ruiz on"),
          rzfail.status == elastiqp::Status::kInfeasible &&
              rzfail.iters == 0 && rz.eq_infeasibility() > 0.1,
          rz.eq_infeasibility(), "eq_infeas");

    // p = 0 path shares the gate.
    Solver p0;
    p0.setup(qp.Q, qp.q, Ai, bi, MatrixXd(0, n), VectorXd(0), VectorXd(0));
    const auto p0fail = p0.solve();
    Check(L("p=0 inconsistent eq gated"),
          p0fail.status == elastiqp::Status::kInfeasible && p0fail.iters == 0,
          p0.eq_infeasibility(), "eq_infeas");

    // Opt-out: never kInfeasible. The PDAL and IPM run to failure; the
    // active set drops the dependent row from its working set and solves
    // the consistent subset (reports kSolved with that row violated).
    Solver off;
    off.settings.check_eq_consistency = false;
    B::CapOuter(off.settings, 5);
    off.setup(qp.Q, qp.q, Ai, bi, qp.G, qp.h, w);
    const auto offsol = off.solve();
    bool off_ok = offsol.status != elastiqp::Status::kInfeasible &&
                  off.eq_infeasibility() == 0.0;
    if constexpr (!std::is_same_v<Solver, elastiqp::das::Solver>) {
      off_ok &= offsol.converged == 0;
    }
    Check(L("check_eq_consistency=false not gated"), off_ok,
          InfNorm(Ai * offsol.x - bi), "eq_res");
  }

  std::printf("[%s] warm start with equalities (drift q, h, b)\n", B::name);
  {
    const int n = 30, m = 8, p = 200, ticks = 20;
    const QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd w = VectorXd::Constant(p, 10.0);
    Solver warm;
    warm.settings = B::Tight();
    warm.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, w);
    std::normal_distribution<double> dist;
    VectorXd q = qp0.q, h = qp0.h, b = qp0.b;
    int warm_iters = 0, cold_iters = 0;
    double worst_dx = 0, worst_eq = 0, worst_kkt = 0;
    bool all_conv = true;
    for (int k = 0; k < ticks; ++k) {
      for (int i = 0; i < n; ++i) q[i] += 0.01 * dist(rng);
      for (int i = 0; i < p; ++i) h[i] += 0.01 * dist(rng);
      for (int i = 0; i < m; ++i) b[i] += 0.01 * dist(rng);
      warm.set_q(q);
      warm.set_h(h);
      warm.set_b(b);
      const auto& ws = warm.solve();
      const auto cs = SolveWith<Solver>(qp0.Q, q, qp0.A, b, qp0.G, h, w);
      all_conv &= ws.converged == 1 && cs.converged == 1;
      warm_iters += ws.iters;
      cold_iters += cs.iters;
      worst_dx = std::max(worst_dx, InfNorm(ws.x - cs.x));
      worst_eq = std::max(worst_eq, InfNorm(qp0.A * ws.x - b));
      worst_kkt = std::max(worst_kkt, Kkt(qp0.Q, q, qp0.A, b, qp0.G, h, w, ws));
    }
    std::printf("  cold iters=%d warm iters=%d worst_eq=%9.2e kkt=%9.2e\n",
                cold_iters, warm_iters, worst_eq, worst_kkt);
    Check(L("n=30 m=8 p=200 20 ticks"),
          all_conv && worst_dx < 1e-4 && worst_eq < 1e-6 && worst_kkt < 1e-6 &&
              warm_iters < cold_iters,
          worst_dx, "|dx|");
  }

  std::printf("[%s] warm start across perturbed problems\n", B::name);
  {
    const int n = 30, p = 200, ticks = 20;
    const QPData qp0 = problem_gen::Infeasible(rng, n, p, p / 4);
    const VectorXd w = VectorXd::Constant(p, 10.0);
    Solver warm;
    warm.settings = B::Tight();
    warm.setup(qp0.Q, qp0.q, qp0.G, qp0.h, w);
    std::normal_distribution<double> dist;
    VectorXd q = qp0.q, h = qp0.h;
    int warm_iters = 0, cold_iters = 0;
    double worst_dx = 0, worst_kkt = 0;
    bool all_conv = true;
    for (int k = 0; k < ticks; ++k) {
      for (int i = 0; i < n; ++i) q[i] += 0.01 * dist(rng);
      for (int i = 0; i < p; ++i) h[i] += 0.01 * dist(rng);
      warm.set_q(q);
      warm.set_h(h);
      const auto& ws = warm.solve();
      const auto cs = SolveWith<Solver>(qp0.Q, q, qp0.G, h, w);
      all_conv &= ws.converged == 1 && cs.converged == 1;
      warm_iters += ws.iters;
      cold_iters += cs.iters;
      worst_dx = std::max(worst_dx, InfNorm(ws.x - cs.x));
      worst_kkt = std::max(
          worst_kkt, Kkt(qp0.Q, q, qp0.A, qp0.b, qp0.G, h, w, ws));
    }
    std::printf("  cold iters=%d warm iters=%d worst_kkt=%9.2e\n", cold_iters,
                warm_iters, worst_kkt);
    Check(L("n=30 p=200 20 ticks"),
          all_conv && worst_dx < 1e-4 && worst_kkt < 1e-6 &&
              warm_iters < cold_iters,
          worst_dx, "|dx|");
  }

  std::printf("[%s] warm chain along a drifting trajectory (drift_traj)\n",
              B::name);
  for (const auto st : {drift_traj::Structure::kFeas,
                        drift_traj::Structure::kInfeas,
                        drift_traj::Structure::kDegen}) {
    const drift_traj::Size sz{16, 4, 80};
    const int ticks = 30;
    const drift_traj::Trajectory traj = drift_traj::MakeTrajectory(
        sz, st, 10.0, 1e-3, drift_traj::Drift::kQHG,
        100u + static_cast<unsigned>(st), ticks);
    Solver warm;
    warm.settings = B::Tight();
    warm.setup(traj.base.Q, traj.q[0], traj.base.A, traj.b[0], traj.G[0],
               traj.h[0], traj.penalty);
    int warm_iters = 0, cold_iters = 0;
    double worst_dx = 0, worst_kkt = 0;
    bool all_conv = true;
    for (int k = 0; k < ticks; ++k) {
      warm.set_q(traj.q[k]);
      warm.set_h(traj.h[k]);
      warm.set_b(traj.b[k]);
      warm.set_G(traj.G[k]);
      const auto& ws = warm.solve();
      const auto cs = SolveWith<Solver>(traj.base.Q, traj.q[k], traj.base.A,
                                        traj.b[k], traj.G[k], traj.h[k],
                                        traj.penalty);
      all_conv &= ws.converged == 1 && cs.converged == 1;
      warm_iters += ws.iters;
      cold_iters += cs.iters;
      worst_dx = std::max(worst_dx, InfNorm(ws.x - cs.x));
      worst_kkt = std::max(
          worst_kkt, Kkt(traj.base.Q, traj.q[k], traj.base.A, traj.b[k],
                         traj.G[k], traj.h[k], traj.penalty, ws));
    }
    std::snprintf(cell, sizeof cell, "%s qhG drift: cold=%d warm=%d",
                  drift_traj::Name(st), cold_iters, warm_iters);
    Check(L(cell),
          all_conv && worst_dx < 1e-4 && worst_kkt < 1e-6 &&
              warm_iters < cold_iters,
          worst_dx, "|dx|");
  }

  std::printf("[%s] heavy saturation (every row violated)\n", B::name);
  {
    const int n = 12, p = 40;
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 2);
    const VectorXd w = VectorXd::Constant(p, 10.0);
    const auto sol = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h, w);
    const auto ref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const double dx = InfNorm(sol.x - ref.x);
    Check(L("n=12 p=40 all-conflict"),
          sol.converged == 1 && ref.converged == 1 && dx < 1e-5, dx, "|dx|");
  }

  std::printf("[%s] Ruiz equilibration on badly row-scaled data\n", B::name);
  {
    // Row-scaling (G_i, h_i) by s_i with penalty_i / s_i (and (A_j, b_j)
    // by any s_j) leaves the elastic QP unchanged -- only its conditioning
    // moves. The reference is the well-scaled problem's solution.
    const int n = 20, m = 5, p = 80;
    const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const auto ref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h,
                            VectorXd::Constant(p, 10.0));
    std::uniform_real_distribution<double> unif(-4.0, 4.0);
    MatrixXd Gs = qp.G, As = qp.A;
    VectorXd hs = qp.h, bs = qp.b, ws(p);
    for (int i = 0; i < p; ++i) {
      const double s = std::pow(10.0, unif(rng));
      Gs.row(i) *= s;
      hs[i] *= s;
      ws[i] = 10.0 / s;
    }
    for (int i = 0; i < m; ++i) {
      const double s = std::pow(10.0, unif(rng));
      As.row(i) *= s;
      bs[i] *= s;
    }
    typename B::Settings off = B::Tight(), on = B::Tight();
    off.ruiz = false;
    on.ruiz = true;
    const auto soff = SolveWith<Solver>(qp.Q, qp.q, As, bs, Gs, hs, ws, off);
    const auto son = SolveWith<Solver>(qp.Q, qp.q, As, bs, Gs, hs, ws, on);
    const double dx = InfNorm(son.x - ref.x);
    const double kkt = Kkt(qp.Q, qp.q, As, bs, Gs, hs, ws, son);
    std::printf("  ruiz off: converged=%d iters=%d | ruiz on: iters=%d\n",
                soff.converged, soff.iters, son.iters);
    Check(L("n=20 m=5 p=80 rows 10^[-4,4]"),
          son.converged == 1 && dx < 1e-4 && kkt < 1e-5, dx, "|dx|");
  }

  std::printf("[%s] Ruiz + warm start (drift q, h, b through setters)\n",
              B::name);
  {
    // The setters must rescale updates into the scaled frame; warm
    // starting across drifting data validates the whole chain.
    const int n = 30, m = 8, p = 200, ticks = 20;
    QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    VectorXd w(p);
    std::uniform_real_distribution<double> unif(-3.0, 3.0);
    for (int i = 0; i < p; ++i) {
      const double s = std::pow(10.0, unif(rng));
      qp0.G.row(i) *= s;
      qp0.h[i] *= s;
      w[i] = 10.0 / s;
    }
    typename B::Settings on = B::Tight();
    on.ruiz = true;
    Solver warm;
    warm.settings = on;
    warm.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, w);
    std::normal_distribution<double> dist;
    VectorXd q = qp0.q, h = qp0.h, b = qp0.b;
    int warm_iters = 0, cold_iters = 0;
    double worst_kkt = 0, worst_dx = 0;
    bool all_conv = true;
    for (int k = 0; k < ticks; ++k) {
      for (int i = 0; i < n; ++i) q[i] += 0.01 * dist(rng);
      for (int i = 0; i < p; ++i) h[i] += 0.01 * dist(rng) * std::abs(h[i]);
      for (int i = 0; i < m; ++i) b[i] += 0.01 * dist(rng);
      warm.set_q(q);
      warm.set_h(h);
      warm.set_b(b);
      const auto& ws = warm.solve();
      const auto cs = SolveWith<Solver>(qp0.Q, q, qp0.A, b, qp0.G, h, w, on);
      all_conv &= ws.converged == 1 && cs.converged == 1;
      warm_iters += ws.iters;
      cold_iters += cs.iters;
      worst_dx = std::max(worst_dx, InfNorm(ws.x - cs.x));
      worst_kkt = std::max(worst_kkt, Kkt(qp0.Q, q, qp0.A, b, qp0.G, h, w, ws));
    }
    std::printf("  cold iters=%d warm iters=%d worst_kkt=%9.2e\n", cold_iters,
                warm_iters, worst_kkt);
    // Validates the setter RESCALING chain (accuracy under drift), not
    // warm-start economics: on this badly row-scaled construction at eps
    // 1e-8 the warm/cold iteration ratio is seed-fragile for the PDAL.
    Check(L("n=30 m=8 p=200 ruiz 20 ticks"),
          all_conv && worst_dx < 1e-4 && worst_kkt < 1e-5 &&
              warm_iters < 3 * cold_iters / 2,
          worst_dx, "|dx|");
  }

  std::printf("[%s] exact-penalty threshold (recovery vs saturation)\n",
              B::name);
  {
    // p < n so all rows are linearly independent and the hard dual is
    // unique.
    const int n = 16, p = 12;
    const QPData qp = problem_gen::Feasible(rng, n, p);
    const StrictSol ref = SolveStrictRef(qp);
    const double zmax = ref.z.maxCoeff();
    // Penalty above the hard dual: exact recovery, slack vanishes.
    const auto hi = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h,
                                      VectorXd::Constant(p, 2.0 * zmax + 1.0));
    const double dx_hi = InfNorm(hi.x - ref.x);
    Check(L("penalty > ||z*||: hard recovery, t ~ 0"),
          hi.converged == 1 && ref.converged == 1 && dx_hi < 1e-5 &&
              MaxT(hi) <= B::slack_tol,
          dx_hi, "|dx|");
    // Penalty below the largest dual: that row saturates, genuine
    // violation appears; ground truth is the elastic reference.
    const VectorXd pen_lo = VectorXd::Constant(p, 0.5 * zmax);
    const auto lo = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h, pen_lo);
    const auto eref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, pen_lo);
    const double dx_lo = InfNorm(lo.x - eref.x);
    Check(L("penalty < ||z*||: saturates, matches ref"),
          lo.converged == 1 && eref.converged == 1 && dx_lo < 1e-5 &&
              MaxT(lo) > 1e-6,
          dx_lo, "|dx|");
    // Numerically extreme penalty: same recovery, well conditioned.
    const auto huge = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h,
                                        VectorXd::Constant(p, 1e8));
    const double dx_huge = InfNorm(huge.x - ref.x);
    Check(L("penalty = 1e8: still exact recovery"),
          huge.converged == 1 && dx_huge < 1e-5 && MaxT(huge) <= B::slack_tol,
          dx_huge, "|dx|");
  }

  std::printf("[%s] penalty exactly at the hard dual (degenerate tie)\n",
              B::name);
  {
    // With w_i = z*_i the dual is pinned to the clamp boundary and
    // complementarity is degenerate; x* stays unique (strictly convex Q).
    const int n = 14, p = 40;
    const QPData qp = problem_gen::Feasible(rng, n, p);
    const StrictSol ref = SolveStrictRef(qp);
    Eigen::Index imax;
    const double zmax = ref.z.maxCoeff(&imax);
    VectorXd pen = VectorXd::Constant(p, 2.0 * zmax + 1.0);
    pen[imax] = ref.z[imax];  // exact tie on the most-active row
    const auto tie = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h, pen);
    const double dx = InfNorm(tie.x - ref.x);
    Check(L("tie row still converges to hard x*"),
          tie.converged == 1 && dx < 1e-4, dx, "|dx|");
  }

  std::printf("[%s] warm start under penalty drift (weight scheduling)\n",
              B::name);
  {
    const int n = 20, m = 5, p = 80, ticks = 12;
    const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    Solver warm;
    warm.settings = B::Tight();
    warm.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, VectorXd::Constant(p, 10.0));
    std::uniform_real_distribution<double> unif(-1.0, 1.5);
    double worst_dx = 0;
    bool all_ok = true;
    for (int k = 0; k < ticks; ++k) {
      // Log-uniform weights in [0.1, ~30]: drops below the current duals
      // exercise the warm-start clamp of z into [0, penalty].
      VectorXd pen(p);
      for (int i = 0; i < p; ++i) pen[i] = std::pow(10.0, unif(rng));
      warm.set_penalty(pen);
      const auto& ws = warm.solve();
      const auto cs = SolveWith<Solver>(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, pen);
      all_ok &= ws.converged == 1 && cs.converged == 1 &&
                ws.z.minCoeff() >= 0.0 &&
                (ws.z - pen).maxCoeff() <= B::invariant_tol;
      worst_dx = std::max(worst_dx, InfNorm(ws.x - cs.x));
    }
    Check(L("n=20 m=5 p=80 12 penalty ticks"), all_ok && worst_dx < 1e-4,
          worst_dx, "|dx|");
  }

  std::printf("[%s] warm start under matrix drift (Q, A, G)\n", B::name);
  {
    const int n = 20, m = 5, p = 80, ticks = 10;
    const QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd w = VectorXd::Constant(p, 10.0);
    // PSD Hessian drift keeps Q positive definite for every tick.
    const MatrixXd R = problem_gen::Randn(rng, n, n);
    const MatrixXd dQ = 0.02 * (R.transpose() * R) / n;
    const MatrixXd dA = 0.002 * problem_gen::Randn(rng, m, n);
    const MatrixXd dG = 0.002 * problem_gen::Randn(rng, p, n);
    Solver warm;
    warm.settings = B::Tight();
    warm.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, w);
    double worst_dx = 0, worst_kkt = 0;
    bool all_ok = true;
    for (int k = 1; k <= ticks; ++k) {
      const MatrixXd Q = qp0.Q + k * dQ;
      const MatrixXd A = qp0.A + k * dA;
      const MatrixXd G = qp0.G + k * dG;
      warm.set_Q(Q);
      warm.set_A(A);
      warm.set_G(G);
      const auto& ws = warm.solve();
      const auto cs = SolveWith<Solver>(Q, qp0.q, A, qp0.b, G, qp0.h, w);
      all_ok &= ws.converged == 1 && cs.converged == 1;
      worst_dx = std::max(worst_dx, InfNorm(ws.x - cs.x));
      worst_kkt = std::max(worst_kkt, Kkt(Q, qp0.q, A, qp0.b, G, qp0.h, w, ws));
    }
    Check(L("n=20 m=5 p=80 10 matrix ticks"),
          all_ok && worst_dx < 1e-4 && worst_kkt < 1e-6, worst_dx, "|dx|");
  }

  std::printf("[%s] status reporting and Solution invariants\n", B::name);
  {
    const int n = 12, p = 40;
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    const VectorXd w = VectorXd::Constant(p, 10.0);
    typename B::Settings capped = B::Tight();
    B::CapIters(capped, 1);
    const auto cap = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h, w, capped);
    Check(L("iteration cap reports kMaxIter"),
          cap.status == elastiqp::Status::kMaxIter && cap.converged == 0 &&
              std::isfinite(cap.primal_res) && std::isfinite(cap.dual_res),
          cap.dual_res, "dual_res");

    const auto sol = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h, w);
    const VectorXd r = qp.G * sol.x - qp.h;
    const double e_w = InfNorm(w - sol.z_t - sol.z);
    const double e_t = InfNorm(sol.t - r.cwiseMax(0.0));
    // (the interior point keeps its slacks positive, not t itself, and
    // z_t + z = w only holds to eps, so the box is checked to the
    // backend's exactness)
    const bool boxed = sol.z.minCoeff() >= -B::invariant_tol &&
                       (sol.z - w).maxCoeff() <= B::invariant_tol &&
                       sol.t.minCoeff() >= -B::invariant_tol;
    const double e_obj =
        std::abs(sol.primal_obj -
                 problem_gen::ElasticObjective(qp.Q, qp.q, qp.G, qp.h, w, sol.x));
    const double tol = std::max(B::invariant_tol, 1e-8);
    Check(L("z_t+z==w, t==[r]+, box, obj"),
          sol.converged == 1 && e_w <= B::invariant_tol && e_t < tol &&
              boxed && e_obj < 1e-7,
          std::max({e_w, e_t, e_obj}), "inv");
    int n_act = 0, n_sat = 0;
    elastiqp::count_row_states(sol.t, sol.z, 1e-6, n_act, n_sat);
    Check(L("n_active / n_saturated reported"),
          sol.n_saturated == n_sat && sol.n_active == n_act &&
              sol.n_saturated > 0,
          static_cast<double>(sol.n_active), "n_active");
  }

  std::printf("[%s] w_i = 0 rows == removed rows\n", B::name);
  {
    const int n = 12, p = 30, k_zero = 5;
    const QPData qp = problem_gen::Feasible(rng, n, p);
    VectorXd pen = VectorXd::Constant(p, 10.0);
    pen.head(k_zero).setZero();  // free rows: no cost for violating them
    const auto full = SolveWith<Solver>(qp.Q, qp.q, qp.G, qp.h, pen);
    const MatrixXd Gr = qp.G.bottomRows(p - k_zero);
    const VectorXd hr = qp.h.tail(p - k_zero);
    const auto reduced = IpmRef(qp.Q, qp.q, MatrixXd(0, n), VectorXd(0), Gr,
                                hr, VectorXd::Constant(p - k_zero, 10.0));
    const double dx = InfNorm(full.x - reduced.x);
    Check(L("5 free rows"),
          full.converged == 1 && reduced.converged == 1 && dx < 1e-5, dx,
          "|dx|");
  }

  std::printf("[%s] duplicated rows (rank-deficient active set)\n", B::name);
  {
    // Duplicating a row and splitting its penalty is the same elastic QP;
    // the duplicated active/saturated rows make G_act rank deficient.
    const int n = 12, p = 30, k_dup = 5;
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    MatrixXd G2(p + k_dup, n);
    VectorXd h2(p + k_dup), pen2(p + k_dup);
    G2.topRows(p) = qp.G;
    h2.head(p) = qp.h;
    pen2.head(p).setConstant(10.0);
    G2.bottomRows(k_dup) = qp.G.topRows(k_dup);
    h2.tail(k_dup) = qp.h.head(k_dup);
    pen2.head(k_dup).setConstant(5.0);
    pen2.tail(k_dup).setConstant(5.0);
    const auto dup = SolveWith<Solver>(qp.Q, qp.q, G2, h2, pen2);
    const auto ref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h,
                            VectorXd::Constant(p, 10.0));
    const double dx = InfNorm(dup.x - ref.x);
    Check(L("split-penalty duplicates match original"),
          dup.converged == 1 && ref.converged == 1 && dx < 1e-5, dx, "|dx|");
  }

  std::printf("[%s] Ruiz leaves near-zero noise rows alone (limit_scaling)\n",
              B::name);
  {
    // Rows at the numerical noise floor must not destabilize equilibration:
    // limit_scaling leaves rows with norm < 1e-4 unscaled. The noise rows
    // perturb the objective by O(1e-12), so the solution must match the
    // problem without them.
    const int n = 16, p = 60, k_noise = 6;
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    MatrixXd G2(p + k_noise, n);
    VectorXd h2(p + k_noise);
    G2.topRows(p) = qp.G;
    h2.head(p) = qp.h;
    std::normal_distribution<double> dist;
    for (int i = 0; i < k_noise; ++i) {
      for (int j = 0; j < n; ++j) G2(p + i, j) = 1e-13 * dist(rng);
      h2[p + i] = 1e-13 * dist(rng);
    }
    typename B::Settings on = B::Tight();
    on.ruiz = true;
    const auto sol = SolveWith<Solver>(qp.Q, qp.q, G2, h2,
                                       VectorXd::Constant(p + k_noise, 10.0), on);
    const auto ref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h,
                            VectorXd::Constant(p, 10.0));
    const double dx = InfNorm(sol.x - ref.x);
    std::printf("  iters=%d\n", sol.iters);
    Check(L("6 noise rows at 1e-13"),
          sol.converged == 1 && ref.converged == 1 && dx < 1e-4, dx, "|dx|");
  }

  std::printf("[%s] fuzz vs the IPM oracle over random instances\n", B::name);
  {
    int fails = 0;
    double worst_dx = 0;
    for (int k = 0; k < 20; ++k) {
      const int n = 5 + k, m = k % 4, p = 10 + 3 * k;
      const int conflicts = (k % 3 == 0) ? p / 4 : 0;
      const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, conflicts);
      const VectorXd w = VectorXd::Constant(p, 10.0);
      const auto a = SolveWith<Solver>(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
      const auto b = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
      fails += a.converged != 1 || b.converged != 1;
      worst_dx = std::max(worst_dx, InfNorm(a.x - b.x));
    }
    Check(L("20 random instances agree"), fails == 0 && worst_dx < 1e-4,
          worst_dx, "|dx|");
  }

  std::printf("[%s] p=0, m=0 edge case\n", B::name);
  {
    const QPData qp = problem_gen::Feasible(rng, 15, 4);
    const auto sol = SolveWith<Solver>(qp.Q, qp.q, MatrixXd(0, 15), VectorXd(0),
                                       VectorXd(0));
    const double res = InfNorm(qp.Q * sol.x + qp.q);
    Check(L("n=15 p=0"), sol.converged == 1 && res < 1e-7, res, "res");
  }
}

// All three backends on the same instances, compared directly: x and the
// objective always, the duals where the solution is generically unique
// (fewer active rows than n).
void CrossCheckSuite() {
  std::printf("[cross] as vs pdal vs ipm on the same instances\n");
  std::mt19937 rng(2025);
  int fails = 0;
  double worst_dx = 0, worst_dobj = 0, worst_dz = 0, worst_dy = 0;
  int cells = 0;
  for (int k = 0; k < 24; ++k) {
    const int n = 6 + 2 * k, m = (k % 3 == 0) ? 0 : k % 5, p = 2 * n + k;
    const int conflicts = (k % 2 == 0) ? p / 4 : 0;
    QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, conflicts);
    VectorXd w = VectorXd::Constant(p, 10.0);
    if (k % 4 == 1) {
      for (int i = 0; i < p; ++i) w[i] = (i % 3 == 0) ? 1e3 : 1.0;
    }
    if (k % 6 == 5) {
      // Rank-deficient Q (proximal rounds for the DAS), kept bounded by a
      // stiff box on every variable.
      const MatrixXd R = problem_gen::Randn(rng, n / 2, n);
      qp.Q = R.transpose() * R;
      MatrixXd G(p + 2 * n, n);
      VectorXd h(p + 2 * n), wb(p + 2 * n);
      G << qp.G, MatrixXd::Identity(n, n), -MatrixXd::Identity(n, n);
      h << qp.h, VectorXd::Constant(2 * n, 5.0);
      wb << w, VectorXd::Constant(2 * n, 1e4);
      qp.G = G;
      qp.h = h;
      w = wb;
    }
    const auto a = SolveWith<elastiqp::das::Solver>(qp.Q, qp.q, qp.A, qp.b, qp.G,
                                                   qp.h, w);
    const auto pd = SolveWith<elastiqp::pdal::Solver>(qp.Q, qp.q, qp.A, qp.b,
                                                      qp.G, qp.h, w);
    const auto ip = SolveWith<elastiqp::ipm::Solver>(qp.Q, qp.q, qp.A, qp.b,
                                                     qp.G, qp.h, w);
    fails += a.converged != 1 || pd.converged != 1 || ip.converged != 1;
    const double obj_scale = std::max(1.0, std::abs(ip.primal_obj));
    worst_dx = std::max({worst_dx, InfNorm(a.x - pd.x), InfNorm(a.x - ip.x),
                         InfNorm(pd.x - ip.x)});
    worst_dobj = std::max({worst_dobj,
                           std::abs(a.primal_obj - pd.primal_obj) / obj_scale,
                           std::abs(a.primal_obj - ip.primal_obj) / obj_scale});
    if (a.n_active + m < n) {  // unique duals (independent active rows)
      worst_dz = std::max({worst_dz, InfNorm(a.z - pd.z), InfNorm(a.z - ip.z)});
      if (m > 0) {
        worst_dy = std::max({worst_dy, InfNorm(a.y - pd.y), InfNorm(a.y - ip.y)});
      }
    }
    ++cells;
  }
  std::printf("  %d instances: |dx|=%.1e |dobj|=%.1e |dz|=%.1e |dy|=%.1e\n",
              cells, worst_dx, worst_dobj, worst_dz, worst_dy);
  Check("cross: all converge, x and objective agree",
        fails == 0 && worst_dx < 1e-5 && worst_dobj < 1e-7, worst_dx, "|dx|");
  Check("cross: duals agree (unique cases)", worst_dz < 1e-3 && worst_dy < 1e-3,
        std::max(worst_dz, worst_dy), "|dz|,|dy|");
}

}  // namespace

int main() {
  GenericSuite<elastiqp::das::Solver>();
  GenericSuite<elastiqp::pdal::Solver>();
  GenericSuite<elastiqp::ipm::Solver>();
  CrossCheckSuite();
  std::printf(test_util::g_all_ok ? "\nAll solver tests passed.\n"
                                  : "\nFAILURES\n");
  return test_util::g_all_ok ? 0 : 1;
}
