// Timing benchmark on random dense elastic QPs (self-contained: only
// Eigen + the ElastiQP header + common/problem_gen.hpp).
//
// For each problem family (feasible / feasible+eq / infeasible /
// infeasible+eq) and size, reports over a set of random instances:
//   * cold solve() wall time, inner iterations, KKT factorizations
//   * relax(kappa) wall time and Newton iterations for several kappa
//     (cold: retraction start from the tight solution, no warm start)
//   * one KKT vjp (backward pass, elastiqp/kkt_vjp.hpp) wall time
// so "cost of differentiability" = relax + vjp, comparable to the
// forward solve on the same instance.
//
// All numbers are medians across instances; per-instance times take the
// best of a few repetitions to shed timer/allocator noise. Penalties
// mirror tests/test_pdal.cc: 1e3 for feasible instances (exact-penalty
// regime) and 10 for the deliberately conflicting rows.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "elastiqp/elastiqp.hpp"
#include "elastiqp/kkt_vjp.hpp"
#include "problem_gen.hpp"

namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;

using Clock = std::chrono::steady_clock;

double UsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

double Median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Family {
  const char* name;
  double penalty;
  // m > 0 adds hard equalities; conflicts > 0 makes the instance
  // instantaneously infeasible (pairs of contradictory rows).
  bool equalities;
  bool infeasible;
};

constexpr Family kFamilies[] = {
    {"feasible", 1e3, false, false},
    {"feasible+eq", 1e3, true, false},
    {"infeasible", 10.0, false, true},
    {"infeasible+eq", 10.0, true, true},
};

constexpr std::pair<int, int> kSizes[] = {
    {10, 20}, {30, 60}, {100, 200}};

constexpr double kKappas[] = {1e-2, 1e-4, 1e-6};
constexpr int kInstances = 5;
constexpr int kReps = 3;  // best-of-kReps per instance for each timing

QPData MakeInstance(std::mt19937& rng, const Family& fam, int n, int p) {
  const int m = fam.equalities ? std::max(1, n / 5) : 0;
  if (fam.infeasible) {
    return fam.equalities
               ? problem_gen::InfeasibleEq(rng, n, m, p, p / 4)
               : problem_gen::Infeasible(rng, n, p, p / 4);
  }
  return problem_gen::RandomFeasible(rng, n, m, p);
}

}  // namespace

int main() {
  std::printf(
      "bench_random_qp: cold solve / relax(kappa) / kkt vjp timings\n"
      "medians over %d instances, best of %d reps each; times in us\n\n",
      kInstances, kReps);
  std::printf(
      "%-14s %5s %4s %5s | %9s %5s %5s | %8s %3s %8s %3s %8s %3s | %8s | %s\n",
      "family", "n", "m", "p", "solve", "iter", "fact", "rx@1e-2", "it",
      "rx@1e-4", "it", "rx@1e-6", "it", "vjp", "ok");

  for (const Family& fam : kFamilies) {
    for (const auto& [n, p] : kSizes) {
      std::mt19937 rng(1234u + static_cast<unsigned>(n));
      const int m = fam.equalities ? std::max(1, n / 5) : 0;
      const VectorXd penalty = VectorXd::Constant(p, fam.penalty);
      elastiqp::KktVjp vjp;
      vjp.setup(n, m, p);  // compute() below is then allocation-free

      std::vector<double> t_solve, t_vjp;
      std::vector<double> t_relax[3];
      std::vector<double> it_relax[3];
      std::vector<double> iters, facts;
      int n_ok = 0;

      for (int inst = 0; inst < kInstances; ++inst) {
        const QPData qp = MakeInstance(rng, fam, n, p);

        // Cold solves: fresh solver per rep so no state carries over.
        double best = 0.0;
        elastiqp::Solver solver;
        bool ok = false;
        for (int rep = 0; rep < kReps; ++rep) {
          solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
          const auto t0 = Clock::now();
          const auto& sol = solver.solve();
          const double us = UsSince(t0);
          if (rep == 0 || us < best) best = us;
          if (rep == 0) {
            iters.push_back(sol.iters);
            facts.push_back(solver.factorizations());
          }
          ok = sol.converged == 1;
        }
        t_solve.push_back(best);

        // Relax per kappa: retraction start each time.
        for (int k = 0; k < 3; ++k) {
          double best_rx = 0.0;
          for (int rep = 0; rep < kReps; ++rep) {
            const auto t0 = Clock::now();
            const auto& rsol = solver.relax(kKappas[k]);
            const double us = UsSince(t0);
            if (rep == 0 || us < best_rx) best_rx = us;
            if (rep == 0) it_relax[k].push_back(rsol.iters);
            ok = ok && rsol.converged == 1;
          }
          t_relax[k].push_back(best_rx);
        }

        // Backward pass at the last relaxed point (cost is
        // kappa-independent), cotangent on x only, as for a loss(x).
        elastiqp::Cotangents ct;
        ct.x = problem_gen::Randn(rng, n, 1);
        const elastiqp::Solution rsol = solver.solution();
        double best_vjp = 0.0;
        for (int rep = 0; rep < kReps; ++rep) {
          const auto t0 = Clock::now();
          const elastiqp::DataGrads& g =
              vjp.compute(qp.Q, qp.A, qp.G, qp.h, rsol, ct);
          const double us = UsSince(t0);
          if (rep == 0 || us < best_vjp) best_vjp = us;
          if (!std::isfinite(g.q.sum())) ok = false;
        }
        t_vjp.push_back(best_vjp);

        if (ok) ++n_ok;
      }

      std::printf(
          "%-14s %5d %4d %5d | %9.1f %5.0f %5.0f | %8.1f %3.0f %8.1f %3.0f "
          "%8.1f %3.0f | %8.1f | %d/%d\n",
          fam.name, n, m, p, Median(t_solve), Median(iters), Median(facts),
          Median(t_relax[0]), Median(it_relax[0]), Median(t_relax[1]),
          Median(it_relax[1]), Median(t_relax[2]), Median(it_relax[2]),
          Median(t_vjp), n_ok, kInstances);
    }
    std::printf("\n");
  }
  return 0;
}
