// Cross-validation of elastiqp::Solver against vanilla PIQP.
//
// Lives in the benchmarks repo because it needs an external solver. The
// core elastiqp test suite (tests/test_pdal.cc in the elastiqp repo) checks
// the same properties against the in-repo IPM reference; this file is the
// independent, third-party oracle for those checks:
// (1) PIQP on hard-constrained-yet-feasible problems where the elastic
//     result should coincide with the PIQP result;
// (2) PIQP on the expanded (n+p) variable form of the elastic problem.

#include <cmath>
#include <cstdio>
#include <random>

#include "elastiqp/elastiqp.hpp"
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

// PIQP references are solved at eps ~ 1e-10 with agreement thresholds
// around 1e-5..1e-6, which needs more accuracy than the control-sized
// library default (eps_abs = 1e-5); every elastiqp solve here pins the
// tight tolerances explicitly.
elastiqp::Settings TightSettings() {
  elastiqp::Settings s;
  s.eps_abs = 1e-8;
  s.eps_rel = 1e-9;
  s.eps_duality_gap_abs = 1e-8;
  s.eps_duality_gap_rel = 1e-9;
  return s;
}

template <typename... Args>
elastiqp::Solution TightSolve(Args&&... args) {
  return elastiqp::Solve(std::forward<Args>(args)..., TightSettings());
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
// z are the inequality duals (used to place penalties relative to the
// hard problem's dual in the exact-penalty tests).
struct StrictSol {
  VectorXd x, z;
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
  s.z = solver.result().z_u;
  return s;
}

}  // namespace

int main() {
  std::mt19937 rng(42);

  std::printf("PIQP: feasible => matches strict QP, t exactly 0\n");
  for (auto [n, p] : {std::pair{10, 12}, {14, 100}, {58, 500}}) {
    const QPData qp = problem_gen::Feasible(rng, n, p);
    const auto esol = TightSolve(qp.Q, qp.q, qp.G, qp.h, 1e3);
    const StrictSol ref = SolveStrictPiqp(qp);
    const double dx = (esol.x - ref.x).lpNorm<Eigen::Infinity>();
    char name[64];
    std::snprintf(name, sizeof(name), "n=%d p=%d", n, p);
    // The l1 penalty is exact (no barrier): below saturation the
    // reconstructed slack is identically zero, not just ~1e-6.
    Check(name,
          esol.converged == 1 && ref.status == piqp::PIQP_SOLVED &&
              dx < 1e-5 && esol.t.maxCoeff() == 0.0,
          dx, "|dx|");
  }


  std::printf("PIQP: infeasible => matches vanilla piqp (expanded)\n");
  for (auto [n, p] : {std::pair{8, 10}, {14, 100}, {58, 300}}) {
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    const auto esol = TightSolve(qp.Q, qp.q, qp.G, qp.h, penalty);
    const ExpandedSol ref = SolveExpanded(qp, penalty);
    const double dx = (esol.x - ref.x).lpNorm<Eigen::Infinity>();
    char name[64];
    std::snprintf(name, sizeof(name), "n=%d p=%d", n, p);
    Check(name,
          esol.converged == 1 && ref.status == piqp::PIQP_SOLVED && dx < 1e-5 &&
              esol.t.maxCoeff() > 0.1,
          dx, "|dx|");
  }

  std::printf("PIQP: per-constraint penalty weights\n");
  {
    const int n = 10, p = 40;
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    VectorXd penalty(p);
    for (int i = 0; i < p; ++i) penalty[i] = (i % 2) ? 100.0 : 5.0;
    const auto esol = TightSolve(qp.Q, qp.q, qp.G, qp.h, penalty);
    const ExpandedSol ref = SolveExpanded(qp, penalty);
    const double dx = (esol.x - ref.x).lpNorm<Eigen::Infinity>();
    Check("n=10 p=40 mixed penalty",
          esol.converged == 1 && ref.status == piqp::PIQP_SOLVED && dx < 1e-5,
          dx, "|dx|");
  }

  std::printf("PIQP: PSD-only Q (rank deficient)\n");
  {
    const int n = 20, p = 60;
    QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    const MatrixXd R = problem_gen::Randn(rng, n / 2, n);
    qp.Q = R.transpose() * R;  // rank n/2
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    const auto esol = TightSolve(qp.Q, qp.q, qp.G, qp.h, penalty);
    const ExpandedSol ref = SolveExpanded(qp, penalty);
    const double dx = (esol.x - ref.x).lpNorm<Eigen::Infinity>();
    Check("n=20 (rank 10) p=60",
          esol.converged == 1 && ref.status == piqp::PIQP_SOLVED && dx < 1e-4,
          dx, "|dx|");
  }

  std::printf("PIQP: hard equalities, feasible ineqs => matches strict piqp\n");
  for (auto [n, m, p] :
       {std::tuple{14, 4, 60}, {30, 8, 200}, {58, 15, 500}}) {
    const QPData qp = problem_gen::RandomFeasible(rng, n, m, p);
    const auto esol =
        TightSolve(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, 1e3);
    const StrictSol ref = SolveStrictPiqp(qp);
    const double dx = (esol.x - ref.x).lpNorm<Eigen::Infinity>();
    const double eq_res = (qp.A * esol.x - qp.b).lpNorm<Eigen::Infinity>();
    char name[64];
    std::snprintf(name, sizeof(name), "n=%d m=%d p=%d", n, m, p);
    Check(name,
          esol.converged == 1 && ref.status == piqp::PIQP_SOLVED &&
              dx < 1e-5 && eq_res < 1e-6 && esol.t.maxCoeff() == 0.0,
          std::max(dx, eq_res), "|dx|,eq");
  }

  std::printf("PIQP: equalities hold when inequalities are infeasible\n");
  for (auto [n, m, p] :
       {std::tuple{14, 4, 100}, {30, 8, 200}, {58, 15, 400}}) {
    const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    const auto esol =
        TightSolve(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
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

  std::printf("PIQP: heavy saturation (every row violated)\n");
  {
    // All rows conflict pairwise: half the rows saturate at the solution and
    // drop out of the Newton system; the rho*I prox term carries K.
    const int n = 12, p = 40;
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 2);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    const auto esol = TightSolve(qp.Q, qp.q, qp.G, qp.h, penalty);
    const ExpandedSol ref = SolveExpanded(qp, penalty);
    const double dx = (esol.x - ref.x).lpNorm<Eigen::Infinity>();
    Check("n=12 p=40 all-conflict",
          esol.converged == 1 && ref.status == piqp::PIQP_SOLVED && dx < 1e-5,
          dx, "|dx|");
  }


  std::printf("PIQP: exact-penalty threshold (recovery vs saturation)\n");
  {
    // p < n so all rows are linearly independent and the hard dual is
    // unique -- with degenerate active sets (p > n) the dual set is a
    // polytope and "0.5x the reported dual" can still bound another valid
    // dual vector, requiring no violation at all.
    const int n = 16, p = 12;
    const QPData qp = problem_gen::Feasible(rng, n, p);
    const StrictSol ref = SolveStrictPiqp(qp);
    const double zmax = ref.z.maxCoeff();
    // Penalty above the hard dual: exact recovery, slack identically zero.
    const auto hi =
        TightSolve(qp.Q, qp.q, qp.G, qp.h, 2.0 * zmax + 1.0);
    const double dx_hi = (hi.x - ref.x).lpNorm<Eigen::Infinity>();
    Check("penalty > ||z*||: hard recovery, t == 0",
          hi.converged == 1 && ref.status == piqp::PIQP_SOLVED &&
              dx_hi < 1e-5 && hi.t.maxCoeff() == 0.0,
          dx_hi, "|dx|");
    // Penalty below the largest dual: that row saturates, genuine
    // violation appears; ground truth is the expanded formulation.
    const VectorXd pen_lo = VectorXd::Constant(p, 0.5 * zmax);
    const auto lo = TightSolve(qp.Q, qp.q, qp.G, qp.h, pen_lo);
    const ExpandedSol eref = SolveExpanded(qp, pen_lo);
    const double dx_lo = (lo.x - eref.x).lpNorm<Eigen::Infinity>();
    Check("penalty < ||z*||: saturates, matches expanded",
          lo.converged == 1 && eref.status == piqp::PIQP_SOLVED &&
              dx_lo < 1e-5 && lo.t.maxCoeff() > 1e-6,
          dx_lo, "|dx|");
    // Numerically extreme penalty: same recovery, well conditioned.
    const auto huge = TightSolve(qp.Q, qp.q, qp.G, qp.h, 1e8);
    const double dx_huge = (huge.x - ref.x).lpNorm<Eigen::Infinity>();
    Check("penalty = 1e8: still exact recovery",
          huge.converged == 1 && dx_huge < 1e-5 && huge.t.maxCoeff() == 0.0,
          dx_huge, "|dx|");
  }

  std::printf("PIQP: penalty exactly at the hard dual (degenerate tie)\n");
  {
    // The delicate case both frameworks share: with w_i = z*_i the dual is
    // pinned to the clamp boundary and complementarity is degenerate; x*
    // stays unique (strictly convex Q) and must still be found.
    const int n = 14, p = 40;
    const QPData qp = problem_gen::Feasible(rng, n, p);
    const StrictSol ref = SolveStrictPiqp(qp);
    Eigen::Index imax;
    const double zmax = ref.z.maxCoeff(&imax);
    VectorXd pen = VectorXd::Constant(p, 2.0 * zmax + 1.0);
    pen[imax] = ref.z[imax];  // exact tie on the most-active row
    const auto tie = TightSolve(qp.Q, qp.q, qp.G, qp.h, pen);
    const double dx = (tie.x - ref.x).lpNorm<Eigen::Infinity>();
    Check("tie row still converges to hard x*",
          tie.converged == 1 && dx < 1e-4, dx, "|dx|");
  }


  std::printf(g_all_ok ? "\nAll PIQP cross-validation tests passed.\n"
                       : "\nFAILURES\n");
  return g_all_ok ? 0 : 1;
}
