// Correctness tests for the test-only IPM reference implementation
// (tests/ipm_reference.hpp). The reference cross-validates elastiqp::Solver
// in test_pdal.cc, so it must itself be validated against vanilla PIQP:
// (1) on hard-constrained-yet-feasible problems where the elastic
// result should coincide with the PIQP result;
// (2) on the expanded (n+p) variable form of the elastic problem

#include <cstdio>
#include <random>

#include "ipm_reference.hpp"
#include "piqp/piqp.hpp"
#include "problem_gen.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;

namespace {

bool g_all_ok = true;

void Check(const char* name, bool ok, double val, const char* what) {
  std::printf("  %-38s %s=%9.2e %s\n", name, what, val, ok ? "OK" : "FAIL");
  g_all_ok &= ok;
}

// This suite cross-validates against PIQP solved at eps ~ 1e-10 with
// agreement thresholds around 1e-5..1e-6, which needs more accuracy than
// the control-sized library default (eps_abs = 1e-5). Every IPM solve
// here therefore pins the tight tolerances explicitly.
elastiqp::IpmSettings TightIpmSettings() {
  elastiqp::IpmSettings s;
  s.eps_abs = 1e-8;
  s.eps_rel = 1e-9;
  s.eps_duality_gap_abs = 1e-8;
  s.eps_duality_gap_rel = 1e-9;
  return s;
}

template <typename... Args>
elastiqp::Solution TightIpmSolve(Args&&... args) {
  return elastiqp::IpmSolve(std::forward<Args>(args)..., TightIpmSettings());
}

// Vanilla piqp on the expanded formulation (equalities carried over as hard
// constraints on the x block); returns (x, t, y, z_t, z_ineq).
struct ExpandedSol {
  VectorXd x, t, y, z_t, z_ineq;
  piqp::Status status;
};

ExpandedSol SolveExpanded(const QPData& qp, const VectorXd& penalty,
                          double eps = 1e-10) {
  const Eigen::Index n = qp.q.size();
  const Eigen::Index m = qp.b.size();
  const Eigen::Index p = qp.h.size();
  const problem_gen::ExpandedElastic e =
      problem_gen::MakeExpanded(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
  piqp::DenseSolver<double> solver;
  solver.settings().eps_abs = eps;
  solver.settings().eps_rel = 0;
  solver.settings().verbose = false;
  if (m > 0) {
    solver.setup(e.P, e.c, e.A, e.b, e.Gt, piqp::nullopt, e.h, e.lb,
                 piqp::nullopt);
  } else {
    solver.setup(e.P, e.c, piqp::nullopt, piqp::nullopt, e.Gt, piqp::nullopt,
                 e.h, e.lb, piqp::nullopt);
  }
  ExpandedSol s;
  s.status = solver.solve();
  s.x = solver.result().x.head(n);
  s.t = solver.result().x.tail(p);
  s.y = solver.result().y;
  s.z_t = solver.result().z_bl.tail(p);
  s.z_ineq = solver.result().z_u;
  return s;
}

// Vanilla piqp on the ORIGINAL strict problem (hard A, b and hard G, h).
struct StrictSol {
  VectorXd x;
  piqp::Status status;
};

StrictSol SolveStrictPiqp(const QPData& qp, double eps = 1e-10) {
  piqp::DenseSolver<double> solver;
  solver.settings().eps_abs = eps;
  solver.settings().eps_rel = 0;
  const bool has_eq = qp.b.size() > 0;
  solver.setup(qp.Q, qp.q,
               has_eq ? piqp::optional<piqp::CMatRef<double>>(qp.A)
                      : piqp::nullopt,
               has_eq ? piqp::optional<piqp::CVecRef<double>>(qp.b)
                      : piqp::nullopt,
               qp.G, piqp::nullopt, qp.h, piqp::nullopt, piqp::nullopt);
  StrictSol s;
  s.status = solver.solve();
  s.x = solver.result().x;
  return s;
}

}  // namespace

int main() {
  std::mt19937 rng(42);

  std::printf("IPM: feasible => matches strict QP, t ~ 0\n");
  for (auto [n, p] : {std::pair{10, 12}, {14, 100}, {58, 500}}) {
    const QPData qp = problem_gen::Feasible(rng, n, p);
    const auto esol = TightIpmSolve(qp.Q, qp.q, qp.G, qp.h, 1e3);
    const StrictSol ref = SolveStrictPiqp(qp);
    const double dx = (esol.x - ref.x).lpNorm<Eigen::Infinity>();
    char name[64];
    std::snprintf(name, sizeof(name), "n=%d p=%d", n, p);
    Check(name,
          esol.converged == 1 && ref.status == piqp::PIQP_SOLVED &&
              dx < 1e-5 && esol.t.maxCoeff() < 1e-6,
          dx, "|dx|");
  }

  std::printf("IPM: KKT residual of the elastic problem\n");
  for (auto [n, p] : {std::pair{14, 60}, {30, 200}}) {
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    const auto esol = TightIpmSolve(qp.Q, qp.q, qp.G, qp.h, penalty);
    const double res = problem_gen::ElasticKKTResidual(
        qp.Q, qp.q, qp.G, qp.h, penalty, esol.x, esol.t, esol.z_t, esol.z_ineq);
    char name[64];
    std::snprintf(name, sizeof(name), "n=%d p=%d infeasible", n, p);
    Check(name, esol.converged == 1 && res < 1e-6, res, "kkt");
  }

  std::printf("IPM: infeasible => matches vanilla piqp (expanded)\n");
  for (auto [n, p] : {std::pair{8, 10}, {14, 100}, {58, 300}}) {
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    const auto esol = TightIpmSolve(qp.Q, qp.q, qp.G, qp.h, penalty);
    const ExpandedSol ref = SolveExpanded(qp, penalty);
    const double dx = (esol.x - ref.x).lpNorm<Eigen::Infinity>();
    char name[64];
    std::snprintf(name, sizeof(name), "n=%d p=%d", n, p);
    Check(name,
          esol.converged == 1 && ref.status == piqp::PIQP_SOLVED && dx < 1e-5 &&
              esol.t.maxCoeff() > 0.1,
          dx, "|dx|");
  }

  std::printf("IPM: per-constraint penalty weights\n");
  {
    const int n = 10, p = 40;
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    VectorXd penalty(p);
    for (int i = 0; i < p; ++i) penalty[i] = (i % 2) ? 100.0 : 5.0;
    const auto esol = TightIpmSolve(qp.Q, qp.q, qp.G, qp.h, penalty);
    const ExpandedSol ref = SolveExpanded(qp, penalty);
    const double dx = (esol.x - ref.x).lpNorm<Eigen::Infinity>();
    Check("n=10 p=40 mixed penalty",
          esol.converged == 1 && ref.status == piqp::PIQP_SOLVED && dx < 1e-5,
          dx, "|dx|");
  }

  std::printf("IPM: PSD-only Q (rank deficient)\n");
  {
    // The proximal rho regularization handles semidefinite Q natively.
    const int n = 20, p = 60;
    QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    const MatrixXd R = problem_gen::Randn(rng, n / 2, n);
    qp.Q = R.transpose() * R;  // rank n/2
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    const auto esol = TightIpmSolve(qp.Q, qp.q, qp.G, qp.h, penalty);
    const ExpandedSol ref = SolveExpanded(qp, penalty);
    const double dx = (esol.x - ref.x).lpNorm<Eigen::Infinity>();
    Check("n=20 (rank 10) p=60",
          esol.converged == 1 && ref.status == piqp::PIQP_SOLVED && dx < 1e-4,
          dx, "|dx|");
  }

  std::printf(
      "IPM: hard equalities, feasible ineqs => matches strict piqp\n");
  for (auto [n, m, p] :
       {std::tuple{14, 4, 60}, {30, 8, 200}, {58, 15, 500}}) {
    const QPData qp = problem_gen::RandomFeasible(rng, n, m, p);
    const auto esol =
        TightIpmSolve(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, 1e3);
    const StrictSol ref = SolveStrictPiqp(qp);
    const double dx = (esol.x - ref.x).lpNorm<Eigen::Infinity>();
    const double eq_res = (qp.A * esol.x - qp.b).lpNorm<Eigen::Infinity>();
    char name[64];
    std::snprintf(name, sizeof(name), "n=%d m=%d p=%d", n, m, p);
    Check(name,
          esol.converged == 1 && ref.status == piqp::PIQP_SOLVED &&
              dx < 1e-5 && eq_res < 1e-6 && esol.t.maxCoeff() < 1e-6,
          std::max(dx, eq_res), "|dx|,eq");
  }

  std::printf(
      "IPM: hard equalities HOLD when inequalities are infeasible\n");
  for (auto [n, m, p] :
       {std::tuple{14, 4, 100}, {30, 8, 200}, {58, 15, 400}}) {
    const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    const auto esol =
        TightIpmSolve(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    const ExpandedSol ref = SolveExpanded(qp, penalty);
    const double dx = (esol.x - ref.x).lpNorm<Eigen::Infinity>();
    const double eq_res = (qp.A * esol.x - qp.b).lpNorm<Eigen::Infinity>();
    const double kkt = problem_gen::ElasticKKTResidual(
        qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty, esol.x, esol.t, esol.y,
        esol.z_t, esol.z_ineq);
    char name[64];
    std::snprintf(name, sizeof(name), "n=%d m=%d p=%d eq_res=%.1e", n, m, p,
                  eq_res);
    Check(name,
          esol.converged == 1 && ref.status == piqp::PIQP_SOLVED &&
              dx < 1e-5 && eq_res < 1e-6 && kkt < 1e-6 &&
              esol.t.maxCoeff() > 0.1,
          dx, "|dx|");
  }

  std::printf("IPM: equality-only edge case (p=0)\n");
  {
    const QPData qp = problem_gen::RandomFeasible(rng, 20, 8, 0);
    const MatrixXd G(0, 20);
    const VectorXd h(0);
    const auto esol = TightIpmSolve(qp.Q, qp.q, qp.A, qp.b, G, h, 10.0);
    // Analytic reference: [Q A'; A 0] [x; y] = [-q; b].
    const int n = 20, m = 8;
    MatrixXd Kf = MatrixXd::Zero(n + m, n + m);
    Kf.topLeftCorner(n, n) = qp.Q;
    Kf.topRightCorner(n, m) = qp.A.transpose();
    Kf.bottomLeftCorner(m, n) = qp.A;
    VectorXd rhs(n + m);
    rhs << -qp.q, qp.b;
    const VectorXd xy = Kf.colPivHouseholderQr().solve(rhs);
    const double dx = (esol.x - xy.head(n)).lpNorm<Eigen::Infinity>();
    Check("n=20 m=8 p=0", esol.converged == 1 && dx < 1e-7, dx, "|dx|");

    // Failures must be reported, not silently returned as kSolved:
    // inconsistent equalities and a singular (unbounded) KKT system.
    MatrixXd Ai(2, n);
    Ai.row(0) = qp.A.row(0);
    Ai.row(1) = qp.A.row(0);
    VectorXd bi(2);
    bi << 0.0, 1.0;
    const auto bad = TightIpmSolve(qp.Q, qp.q, Ai, bi, G, h, 10.0);
    Check("inconsistent A x = b detected",
          bad.status == elastiqp::Status::kNumerics && bad.converged == 0 &&
              bad.primal_res > 0.1,
          bad.primal_res, "primal_res");

    MatrixXd Qs = MatrixXd::Zero(n, n);
    Qs(0, 0) = 1.0;
    const auto sing = TightIpmSolve(Qs, qp.q, MatrixXd(0, n), VectorXd(0),
                                      G, h, 10.0);
    Check("singular unconstrained KKT detected",
          sing.status == elastiqp::Status::kNumerics && sing.converged == 0,
          sing.dual_res, "dual_res");
  }

  std::printf("IPM: kappa relaxation (smoothed-gradient point)\n");
  {
    // relax(kappa) must land on the kappa-central path: same stationarity
    // and primal feasibility, but s.z = kappa on every pair.
    const int n = 20, m = 5, p = 80;
    const double kappa = 1e-3;
    const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    elastiqp::IpmSolver solver;
    solver.settings = TightIpmSettings();
    solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    const auto tight = solver.solve();
    const auto& rsol = solver.relax(kappa);

    double comp_err = 0;
    for (int i = 0; i < p; ++i) {
      comp_err = std::max(comp_err,
                          std::abs(rsol.s_t[i] * rsol.z_t[i] - kappa));
      comp_err = std::max(comp_err,
                          std::abs(rsol.s_ineq[i] * rsol.z_ineq[i] - kappa));
    }
    const double stat =
        (qp.Q * rsol.x + qp.q + qp.A.transpose() * rsol.y +
         qp.G.transpose() * rsol.z_ineq)
            .lpNorm<Eigen::Infinity>();
    const double eq = (qp.A * rsol.x - qp.b).lpNorm<Eigen::Infinity>();
    const double dx = (rsol.x - tight.x).lpNorm<Eigen::Infinity>();
    std::printf("  comp_err=%.1e stat=%.1e eq=%.1e |x_r - x|=%.1e\n",
                comp_err, stat, eq, dx);
    Check("n=20 m=5 p=80 kappa=1e-3",
          tight.converged == 1 && rsol.converged == 1 && comp_err < 1e-8 &&
              stat < 1e-7 && eq < 1e-7 && dx < 0.1 && dx > 0,
          comp_err, "comp");
  }

  std::printf("IPM: warm start with equalities (drift q, h, b)\n");
  {
    const int n = 30, m = 8, p = 200, ticks = 20;
    const QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);

    elastiqp::IpmSolver warm;
    warm.settings = TightIpmSettings();
    warm.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, penalty);

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
      const auto cs = TightIpmSolve(qp0.Q, q, qp0.A, b, qp0.G, h, penalty);
      all_conv &= ws.converged == 1 && cs.converged == 1;
      warm_iters += ws.iters;
      cold_iters += cs.iters;
      worst_dx = std::max(worst_dx, (ws.x - cs.x).lpNorm<Eigen::Infinity>());
      worst_eq = std::max(
          worst_eq, (qp0.A * ws.x - b).lpNorm<Eigen::Infinity>());
      worst_kkt = std::max(
          worst_kkt,
          problem_gen::ElasticKKTResidual(qp0.Q, q, qp0.A, b, qp0.G, h,
                                          penalty, ws.x, ws.t, ws.y, ws.z_t,
                                          ws.z_ineq));
    }
    std::printf("  cold iters=%d warm iters=%d worst_eq=%9.2e kkt=%9.2e\n",
                cold_iters, warm_iters, worst_eq, worst_kkt);
    Check("n=30 m=8 p=200 20 ticks",
          all_conv && worst_dx < 1e-4 && worst_eq < 1e-6 && worst_kkt < 1e-6 &&
              warm_iters < cold_iters,
          worst_dx, "|dx|");
  }

  std::printf("IPM: warm start across perturbed problems\n");
  {
    const int n = 30, p = 200, ticks = 20;
    const QPData qp0 = problem_gen::Infeasible(rng, n, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);

    elastiqp::IpmSolver warm;
    warm.settings = TightIpmSettings();
    warm.setup(qp0.Q, qp0.q, qp0.G, qp0.h, penalty);

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
      const auto cs = TightIpmSolve(qp0.Q, q, qp0.G, h, penalty);
      all_conv &= ws.converged == 1 && cs.converged == 1;
      warm_iters += ws.iters;
      cold_iters += cs.iters;
      worst_dx = std::max(worst_dx, (ws.x - cs.x).lpNorm<Eigen::Infinity>());
      worst_kkt = std::max(
          worst_kkt, problem_gen::ElasticKKTResidual(qp0.Q, q, qp0.G, h,
                                                     penalty, ws.x, ws.t,
                                                     ws.z_t, ws.z_ineq));
    }
    std::printf("  cold iters=%d warm iters=%d worst_kkt=%9.2e\n", cold_iters,
                warm_iters, worst_kkt);
    Check("n=30 p=200 20 ticks",
          all_conv && worst_dx < 1e-4 && worst_kkt < 1e-6 &&
              warm_iters < cold_iters,
          worst_dx, "|dx|");
  }

  std::printf("IPM: explicit warm start via set_warm_start\n");
  {
    // Reimplement the built-in floor strategy externally through the
    // set_warm_start hook (with warm_start = false, so nothing happens
    // automatically): the explicit path must behave like a warm start --
    // converge, match independent cold solves, and beat their iteration
    // count -- proving the hook seeds the iterate as given.
    const int n = 30, m = 8, p = 200, ticks = 20;
    const QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);

    elastiqp::IpmSolver solver;
    solver.settings = TightIpmSettings();
    solver.settings.warm_start = false;
    solver.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, penalty);

    std::normal_distribution<double> dist;
    VectorXd q = qp0.q, h = qp0.h, b = qp0.b;
    int explicit_iters = 0, cold_iters = 0;
    double worst_dx = 0, worst_eq = 0, worst_kkt = 0;
    bool all_conv = true;
    for (int k = 0; k < ticks; ++k) {
      for (int i = 0; i < n; ++i) q[i] += 0.01 * dist(rng);
      for (int i = 0; i < p; ++i) h[i] += 0.01 * dist(rng);
      for (int i = 0; i < m; ++i) b[i] += 0.01 * dist(rng);
      solver.set_q(q);
      solver.set_h(h);
      solver.set_b(b);
      if (k > 0) {
        const auto& prev = solver.solution();
        // External floor at 0.1x the previous iterate's KKT residual under
        // the new data (primal residual includes the interior-point slacks).
        VectorXd rd = qp0.Q * prev.x + q + qp0.G.transpose() * prev.z_ineq +
                      qp0.A.transpose() * prev.y;
        const double r = std::max(
            {rd.lpNorm<Eigen::Infinity>(),
             (penalty - prev.z_t - prev.z_ineq).lpNorm<Eigen::Infinity>(),
             (b - qp0.A * prev.x).lpNorm<Eigen::Infinity>(),
             (prev.t - prev.s_t).lpNorm<Eigen::Infinity>(),
             (h - (qp0.G * prev.x - prev.t) - prev.s_ineq)
                 .lpNorm<Eigen::Infinity>()});
        const double f = std::min(std::max(0.1 * r, 1e-8), 1.0);
        solver.set_warm_start(prev.x, prev.t, prev.y, prev.s_t.cwiseMax(f),
                              prev.s_ineq.cwiseMax(f), prev.z_t.cwiseMax(f),
                              prev.z_ineq.cwiseMax(f));
      }
      const auto& ws = solver.solve();
      const auto cs = TightIpmSolve(qp0.Q, q, qp0.A, b, qp0.G, h, penalty);
      all_conv &= ws.converged == 1 && cs.converged == 1;
      explicit_iters += ws.iters;
      cold_iters += cs.iters;
      worst_dx = std::max(worst_dx, (ws.x - cs.x).lpNorm<Eigen::Infinity>());
      worst_eq =
          std::max(worst_eq, (qp0.A * ws.x - b).lpNorm<Eigen::Infinity>());
      worst_kkt = std::max(
          worst_kkt,
          problem_gen::ElasticKKTResidual(qp0.Q, q, qp0.A, b, qp0.G, h,
                                          penalty, ws.x, ws.t, ws.y, ws.z_t,
                                          ws.z_ineq));
    }
    std::printf(
        "  cold iters=%d explicit iters=%d worst_eq=%9.2e kkt=%9.2e\n",
        cold_iters, explicit_iters, worst_eq, worst_kkt);
    Check("n=30 m=8 p=200 20 ticks (explicit)",
          all_conv && worst_dx < 1e-4 && worst_eq < 1e-6 && worst_kkt < 1e-6 &&
              explicit_iters < cold_iters,
          worst_dx, "|dx|");
  }

  std::printf("IPM: Ruiz equilibration on badly row-scaled data\n");
  {
    // Row-scaling (G_i, h_i) by s_i with penalty_i / s_i (and (A_j, b_j)
    // by any s_j) leaves the elastic QP unchanged -- only its conditioning
    // moves. The reference is the well-scaled problem's solution.
    const int n = 20, m = 5, p = 80;
    const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const auto ref =
        TightIpmSolve(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, 10.0);
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
    elastiqp::IpmSolver off, on;
    off.settings = TightIpmSettings();
    on.settings = TightIpmSettings();
    on.settings.ruiz = true;
    off.setup(qp.Q, qp.q, As, bs, Gs, hs, ws);
    on.setup(qp.Q, qp.q, As, bs, Gs, hs, ws);
    const auto soff = off.solve();
    const auto& son = on.solve();
    const double dx = (son.x - ref.x).lpNorm<Eigen::Infinity>();
    const double kkt = problem_gen::ElasticKKTResidual(
        qp.Q, qp.q, As, bs, Gs, hs, ws, son.x, son.t, son.y, son.z_t,
        son.z_ineq);
    std::printf("  ruiz off: converged=%d iters=%d | ruiz on: iters=%d\n",
                soff.converged, soff.iters, son.iters);
    Check("n=20 m=5 p=80 rows 10^[-4,4]",
          son.converged == 1 && dx < 1e-4 && kkt < 1e-5, dx, "|dx|");
  }

  std::printf("IPM: Ruiz + warm start (drift q, h, b through setters)\n");
  {
    // The setters must rescale updates into the setup()-time scaled frame;
    // warm starting across drifting data validates the whole chain.
    const int n = 30, m = 8, p = 200, ticks = 20;
    QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    VectorXd penalty(p);
    std::uniform_real_distribution<double> unif(-3.0, 3.0);
    for (int i = 0; i < p; ++i) {
      const double s = std::pow(10.0, unif(rng));
      qp0.G.row(i) *= s;
      qp0.h[i] *= s;
      penalty[i] = 10.0 / s;
    }
    elastiqp::IpmSolver warm;
    warm.settings = TightIpmSettings();
    warm.settings.ruiz = true;
    warm.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, penalty);

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
      elastiqp::IpmSolver cold;
      cold.settings = TightIpmSettings();
      cold.settings.ruiz = true;
      cold.setup(qp0.Q, q, qp0.A, b, qp0.G, h, penalty);
      const auto& cs = cold.solve();
      all_conv &= ws.converged == 1 && cs.converged == 1;
      warm_iters += ws.iters;
      cold_iters += cs.iters;
      worst_dx = std::max(worst_dx, (ws.x - cs.x).lpNorm<Eigen::Infinity>());
      worst_kkt = std::max(
          worst_kkt,
          problem_gen::ElasticKKTResidual(qp0.Q, q, qp0.A, b, qp0.G, h,
                                          penalty, ws.x, ws.t, ws.y, ws.z_t,
                                          ws.z_ineq));
    }
    std::printf("  cold iters=%d warm iters=%d worst_kkt=%9.2e\n", cold_iters,
                warm_iters, worst_kkt);
    Check("n=30 m=8 p=200 ruiz 20 ticks",
          all_conv && worst_dx < 1e-4 && worst_kkt < 1e-5 &&
              warm_iters < cold_iters,
          worst_dx, "|dx|");
  }

  std::printf("IPM: Ruiz + warm_start_from roundtrip\n");
  {
    const int n = 14, p = 40;
    QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    VectorXd pen(p);
    std::uniform_real_distribution<double> unif(-3.0, 3.0);
    for (int i = 0; i < p; ++i) {
      const double s = std::pow(10.0, unif(rng));
      qp.G.row(i) *= s;
      qp.h[i] *= s;
      pen[i] = 10.0 / s;
    }
    elastiqp::IpmSolver solver;
    solver.settings = TightIpmSettings();
    solver.settings.ruiz = true;
    solver.settings.warm_start = false;
    solver.setup(qp.Q, qp.q, qp.G, qp.h, pen);
    const auto first = solver.solve();
    // Seeding with the (unscaled) solution must roundtrip through the
    // scaling and reconverge in a handful of iterations.
    solver.warm_start_from(first);
    const auto& again = solver.solve();
    Check("seeded resolve converges immediately",
          first.converged == 1 && again.converged == 1 && again.iters <= 5,
          static_cast<double>(again.iters), "iters");
  }

  std::printf("IPM: Ruiz + kappa relaxation\n");
  {
    // relax(kappa) targets s.z = kappa in the USER frame; the row scaling
    // cancels within each s.z pair, leaving only the cost normalization to
    // undo. Same acceptance as the unscaled kappa test above.
    const int n = 20, m = 5, p = 80;
    const double kappa = 1e-3;
    QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    VectorXd penalty(p);
    std::uniform_real_distribution<double> unif(-3.0, 3.0);
    for (int i = 0; i < p; ++i) {
      const double s = std::pow(10.0, unif(rng));
      qp.G.row(i) *= s;
      qp.h[i] *= s;
      penalty[i] = 10.0 / s;
    }
    elastiqp::IpmSolver solver;
    solver.settings = TightIpmSettings();
    solver.settings.ruiz = true;
    solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    const auto tight = solver.solve();
    const auto& rsol = solver.relax(kappa);

    double comp_err = 0;
    for (int i = 0; i < p; ++i) {
      comp_err = std::max(comp_err,
                          std::abs(rsol.s_t[i] * rsol.z_t[i] - kappa));
      comp_err = std::max(comp_err,
                          std::abs(rsol.s_ineq[i] * rsol.z_ineq[i] - kappa));
    }
    const double stat =
        (qp.Q * rsol.x + qp.q + qp.A.transpose() * rsol.y +
         qp.G.transpose() * rsol.z_ineq)
            .lpNorm<Eigen::Infinity>();
    const double eq = (qp.A * rsol.x - qp.b).lpNorm<Eigen::Infinity>();
    std::printf("  comp_err=%.1e stat=%.1e eq=%.1e\n", comp_err, stat, eq);
    Check("n=20 m=5 p=80 ruiz kappa=1e-3",
          tight.converged == 1 && rsol.converged == 1 && comp_err < 1e-7 &&
              stat < 1e-6 && eq < 1e-7,
          comp_err, "comp");
  }

  std::printf("IPM: Ruiz leaves near-zero noise rows alone (limit_scaling)\n");
  {
    // Same guard as the PDAL test: rows at the numerical noise floor must
    // not be amplified by equilibration (PIQP-style limit_scaling treats
    // norms < 1e-4 as unscalable). The noise rows perturb the objective by
    // O(1e-12), so the solution must match the problem without them.
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
    elastiqp::IpmSolver solver;
    solver.settings = TightIpmSettings();
    solver.settings.ruiz = true;
    solver.setup(qp.Q, qp.q, G2, h2, VectorXd::Constant(p + k_noise, 10.0));
    const auto sol = solver.solve();
    const auto ref = TightIpmSolve(qp.Q, qp.q, qp.G, qp.h, 10.0);
    const double dx = (sol.x - ref.x).lpNorm<Eigen::Infinity>();
    std::printf("  iters=%d\n", sol.iters);
    Check("6 noise rows at 1e-13",
          sol.converged == 1 && ref.converged == 1 && dx < 1e-4, dx, "|dx|");
  }

  std::printf("IPM: p=0 edge case\n");
  {
    const QPData qp = problem_gen::Feasible(rng, 15, 4);
    const MatrixXd G(0, 15);
    const VectorXd h(0);
    const auto esol = TightIpmSolve(qp.Q, qp.q, G, h, 10.0);
    const double res = (qp.Q * esol.x + qp.q).lpNorm<Eigen::Infinity>();
    Check("n=15 p=0", esol.converged == 1 && res < 1e-7, res, "res");
  }

  std::printf(g_all_ok ? "\nAll IPM reference tests passed.\n"
                       : "\nFAILURES\n");
  return g_all_ok ? 0 : 1;
}
