// Benchmark of ElastiQP against the naive alternative:
//
//   elastiqp -- elastiqp::Solve (PDAL, condensed n x n elastic KKT)
//   piqp+    -- vanilla piqp::DenseSolver on the expanded (n+p)-variable
//               formulation (explicit t variables)
//
// The condensed formulation eliminates the p elastic slacks from the KKT
// system, so each factorization costs O(n^3 + p n^2) instead of the
// O((n+p)^3) a generic dense solver pays on the expanded problem -- the
// gap this benchmark measures as p grows past n.
//
// Both are run cold (setup + solve) at a matched absolute tolerance.
// Accuracy is measured solver-agnostically: the elastic KKT residual at the
// returned point and the objective gap against a high-accuracy reference.
// Iteration counts are reported per solver but are not comparable across
// the two (PDAL inner semismooth Newton steps vs PIQP interior-point
// iterations).

#include <chrono>
#include <cstdio>
#include <random>

#include "elastiqp/elastiqp.hpp"
#include "piqp/piqp.hpp"
#include "problem_gen.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;
using std::chrono::duration;
using std::chrono::steady_clock;

namespace {

constexpr double kEps = 1e-8;
constexpr int kMaxIter = 250;

struct Stats {
  double us = 0;      // avg wall-clock per solve
  int iters = 0;
  double kkt = 0;     // elastic KKT residual at solution
  double obj = 0;     // elastic objective at solution
  bool ok = false;    // converged
};

int Repeats(int n, int p) {
  // Keep total runtime bounded as problems grow.
  const long work = static_cast<long>(n + p) * (n + p);
  return work > 100000 ? 20 : 100;
}

Stats RunElastiQP(const QPData& qp, const VectorXd& penalty) {
  Stats st;
  elastiqp::Settings settings;
  settings.eps_abs = kEps;
  settings.eps_rel = 0;
  settings.eps_duality_gap_abs = kEps;
  settings.eps_duality_gap_rel = 0;
  settings.max_outer_iter = kMaxIter;
  const int reps = Repeats(static_cast<int>(qp.q.size()), 0);
  elastiqp::Solution sol;
  const auto t0 = steady_clock::now();
  for (int r = 0; r < reps; ++r)
    sol = elastiqp::Solve(qp.Q, qp.q, qp.G, qp.h, penalty, settings);
  const auto t1 = steady_clock::now();
  st.us = duration<double, std::micro>(t1 - t0).count() / reps;
  st.iters = sol.iters;
  st.ok = sol.converged == 1;
  st.kkt = problem_gen::ElasticKKTResidual(qp.Q, qp.q, qp.G, qp.h, penalty,
                                           sol.x, sol.t, sol.z_t, sol.z_ineq);
  st.obj = problem_gen::ElasticObjective(qp.Q, qp.q, qp.G, qp.h, penalty,
                                         sol.x);
  return st;
}

Stats RunPiqpExpanded(const QPData& qp, const VectorXd& penalty, double eps,
                      int reps_override = 0) {
  Stats st;
  const Eigen::Index n = qp.q.size();
  const Eigen::Index p = qp.h.size();
  const problem_gen::ExpandedElastic e =
      problem_gen::MakeExpanded(qp.Q, qp.q, qp.G, qp.h, penalty);
  const int reps = reps_override > 0
                       ? reps_override
                       : Repeats(static_cast<int>(n), static_cast<int>(p));
  piqp::Status status = piqp::PIQP_UNSOLVED;
  VectorXd x(n), t(p), z1(p), z2(p);
  piqp::isize iters = 0;
  const auto t0 = steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    piqp::DenseSolver<double> solver;
    solver.settings().eps_abs = eps;
    solver.settings().eps_rel = 0;
    solver.settings().eps_duality_gap_abs = eps;
    solver.settings().eps_duality_gap_rel = 0;
    solver.settings().max_iter = kMaxIter;
    solver.setup(e.P, e.c, piqp::nullopt, piqp::nullopt, e.Gt, piqp::nullopt,
                 e.h, e.lb, piqp::nullopt);
    status = solver.solve();
    x = solver.result().x.head(n);
    t = solver.result().x.tail(p);
    z1 = solver.result().z_bl.tail(p);
    z2 = solver.result().z_u;
    iters = solver.result().info.iter;
  }
  const auto t1 = steady_clock::now();
  st.us = duration<double, std::micro>(t1 - t0).count() / reps;
  st.iters = static_cast<int>(iters);
  st.ok = status == piqp::PIQP_SOLVED;
  st.kkt = problem_gen::ElasticKKTResidual(qp.Q, qp.q, qp.G, qp.h, penalty, x,
                                           t, z1, z2);
  st.obj = problem_gen::ElasticObjective(qp.Q, qp.q, qp.G, qp.h, penalty, x);
  return st;
}

void RunSweep(const char* title, bool infeasible) {
  std::printf("%s (penalty = 10, eps = %.0e)\n", title, kEps);
  std::printf("  %4s %4s | %12s %5s %9s | %10s %5s %9s | %8s | %9s\n", "n",
              "p", "elastiqp[us]", "it", "kkt", "piqp+[us]", "it", "kkt",
              "speedup", "d_obj");
  for (int n : {14, 30, 58}) {
    for (int p : {25, 50, 100, 200, 350, 500}) {
      std::mt19937 rng(7 * n + p);
      const QPData qp = infeasible ? problem_gen::Infeasible(rng, n, p, p / 4)
                                   : problem_gen::Feasible(rng, n, p);
      const VectorXd penalty = VectorXd::Constant(p, 10.0);

      // High-accuracy reference objective (vanilla piqp at 1e-11).
      const Stats ref = RunPiqpExpanded(qp, penalty, 1e-11, 1);

      const Stats a = RunElastiQP(qp, penalty);
      const Stats c = RunPiqpExpanded(qp, penalty, kEps);

      const double dobj = std::abs(a.obj - ref.obj);
      std::printf(
          "  %4d %4d | %12.1f %5d %9.1e%s| %10.1f %5d %9.1e%s| %7.1fx | "
          "%9.1e\n",
          n, p, a.us, a.iters, a.kkt, a.ok ? " " : "!", c.us, c.iters, c.kkt,
          c.ok ? " " : "!", c.us / a.us, dobj);
    }
    std::printf("\n");
  }
}

void RunRobustness() {
  std::printf(
      "Robustness: penalty magnitude sweep (infeasible, n=30, p=200)\n");
  std::printf("  %8s | %12s %5s %9s | %9s\n", "penalty", "elastiqp[us]", "it",
              "kkt", "d_obj");
  for (double pen : {1.0, 1e2, 1e4, 1e6}) {
    std::mt19937 rng(123);
    const QPData qp = problem_gen::Infeasible(rng, 30, 200, 50);
    const VectorXd penalty = VectorXd::Constant(200, pen);
    const Stats ref = RunPiqpExpanded(qp, penalty, 1e-11, 1);
    const Stats a = RunElastiQP(qp, penalty);
    std::printf("  %8.0e | %12.1f %5d %9.1e%s| %9.1e\n", pen, a.us, a.iters,
                a.kkt, a.ok ? " " : "!", std::abs(a.obj - ref.obj));
  }

  std::printf(
      "\nRobustness: rank-deficient Q (rank n/2, infeasible, penalty=10)\n");
  std::printf("  %4s %4s | %12s %5s %9s\n", "n", "p", "elastiqp[us]", "it",
              "kkt");
  for (auto [n, p] : {std::pair{14, 100}, {30, 200}, {58, 500}}) {
    std::mt19937 rng(31 * n + p);
    QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    const MatrixXd R = problem_gen::Randn(rng, n / 2, n);
    qp.Q = R.transpose() * R;
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    const Stats a = RunElastiQP(qp, penalty);
    std::printf("  %4d %4d | %12.1f %5d %9.1e%s\n", n, p, a.us, a.iters,
                a.kkt, a.ok ? " " : "!");
  }
  std::printf("\n");
}

}  // namespace

int main() {
  std::printf(
      "ElastiQP benchmark (cold solves, avg wall-clock).\n"
      "  elastiqp = elastiqp::Solve (PDAL, condensed n x n elastic KKT)\n"
      "  piqp+    = vanilla piqp on the expanded n+p formulation\n"
      "  '!' marks non-converged runs; d_obj = elastic-objective gap vs a\n"
      "  high-accuracy reference. Iteration counts are not comparable\n"
      "  across solvers (PDAL Newton steps vs PIQP IPM iterations).\n\n");

  RunSweep("Feasible problems", false);
  RunSweep("Infeasible problems (p/4 conflicting pairs)", true);
  RunRobustness();
  return 0;
}
