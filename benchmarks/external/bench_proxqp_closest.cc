// ElastiQP vs ProxQP's closest-feasible-QP mode
// (settings.primal_infeasibility_solving): on a primal-infeasible QP,
// ProxQP converges to the solution of the closest feasible QP, where
// "closest" means the smallest l2-norm shift of the constraint right-hand
// sides. The elastic formulation instead penalizes per-constraint slacks
// in l1, which concentrates the relaxation on the few genuinely
// conflicting constraints (and, above the exact-penalty threshold, finds
// a minimum-l1 shift). This benchmark quantifies both the cost
// (iterations, time -- including what plain infeasibility *detection*
// costs) and the structure of the resulting constraint violations
// (l0/l1/l2/linf) on problems with a known conflict set.
//
// Methodology: cold solves averaged over reps, tolerances matched at
// eps_abs = 1e-6, eps_rel = 0. Iteration counts are reported per solver
// but are not comparable across the two (ElastiQP's PDAL inner semismooth
// Newton steps vs ProxQP inner iterations).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <random>

#include <proxsuite/proxqp/dense/dense.hpp>

#include "elastiqp/elastiqp.hpp"
#include "problem_gen.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;
using std::chrono::duration;
using std::chrono::steady_clock;
namespace pq = proxsuite::proxqp;

namespace {

constexpr double kEps = 1e-6;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kGap = 1.0;  // infeasibility gap per conflicting pair

struct Stats {
  double us = 0;
  double iters = 0;
  bool ok = false;
};

int Repeats(long n, long p) {
  const long work = (n + p) * (n + p);
  if (work > 1000000) return 3;
  if (work > 100000) return 10;
  return 50;
}

template <typename F>
double TimeUs(int reps, F&& f) {
  const auto t0 = steady_clock::now();
  for (int r = 0; r < reps; ++r) f();
  const auto t1 = steady_clock::now();
  return duration<double, std::micro>(t1 - t0).count() / reps;
}

void ProxSettings(pq::Settings<double>& s, bool duality_gap) {
  s.eps_abs = kEps;
  s.eps_rel = 0;
  s.check_duality_gap = duality_gap;
  s.eps_duality_gap_abs = kEps;
  s.eps_duality_gap_rel = 0;
}

// ---------------------------------------------------------------------------
// Closest-feasible QP (l2 shift) vs elastic relaxation (l1 shift)
// ---------------------------------------------------------------------------

struct ShiftStats {
  int nnz = 0;       // constraints violated by more than 1e-4
  int spurious = 0;  // violated constraints OUTSIDE the conflicting pairs
  double l1 = 0, l2 = 0, linf = 0;
};

// The first 2 * n_conflicts rows are the conflicting pairs; everything
// after them is individually satisfiable, so violations there are shift
// "leaking" onto constraints that never needed to move. The count
// threshold (1e-4) sits well above the solver tolerance (1e-6) and well
// below the conflict gap (1), so it counts real shifts, not convergence
// noise.
ShiftStats Violations(const MatrixXd& G, const VectorXd& h, const VectorXd& x,
                      int n_conflicts) {
  const VectorXd v = (G * x - h).cwiseMax(0.0);
  ShiftStats s;
  s.nnz = static_cast<int>((v.array() > 1e-4).count());
  s.spurious = static_cast<int>(
      (v.tail(v.size() - 2 * n_conflicts).array() > 1e-4).count());
  s.l1 = v.lpNorm<1>();
  s.l2 = v.norm();
  s.linf = v.lpNorm<Eigen::Infinity>();
  return s;
}

struct ElasticRun {
  Stats st;
  ShiftStats shift;
};

ElasticRun RunElastic(const QPData& qp, double penalty_mag,
                      int n_conflicts = 0) {
  const Eigen::Index p = qp.h.size();
  const VectorXd penalty = VectorXd::Constant(p, penalty_mag);
  elastiqp::Settings settings;
  settings.eps_abs = kEps;
  settings.eps_rel = 0;
  settings.eps_duality_gap_abs = kEps;
  settings.eps_duality_gap_rel = 0;
  elastiqp::Solution sol;
  const int reps = Repeats(qp.q.size(), 0);
  ElasticRun run;
  run.st.us = TimeUs(reps, [&] {
    sol = elastiqp::Solve(qp.Q, qp.q, qp.G, qp.h, penalty, settings);
  });
  run.st.iters = sol.iters;
  run.st.ok = sol.converged == 1;
  run.shift = Violations(qp.G, qp.h, sol.x, n_conflicts);
  return run;
}

// ProxQP dense on the HARD problem (no slacks). With
// primal_infeasibility_solving off this measures what an infeasibility
// certificate costs; with it on, the closest-feasible (minimal l2 shift)
// solve.
ElasticRun RunProxHard(const QPData& qp, bool closest_feasible,
                       int n_conflicts = 0) {
  const Eigen::Index n = qp.q.size();
  const Eigen::Index p = qp.h.size();
  const VectorXd l = VectorXd::Constant(p, -kInf);
  ElasticRun run;
  const int reps = std::max(1, Repeats(n, p) / 5);
  auto solve_once = [&] {
    pq::dense::QP<double> solver(n, 0, p);
    ProxSettings(solver.settings, /*duality_gap=*/false);
    solver.settings.primal_infeasibility_solving = closest_feasible;
    solver.init(qp.Q, qp.q, proxsuite::nullopt, proxsuite::nullopt, qp.G, l,
                qp.h);
    solver.solve();
    run.st.iters = static_cast<double>(solver.results.info.iter);
    run.st.ok =
        solver.results.info.status ==
        (closest_feasible
             ? pq::QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE
             : pq::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE);
    if (closest_feasible) {
      run.shift = Violations(qp.G, qp.h, solver.results.x, n_conflicts);
    }
  };
  run.st.us = TimeUs(reps, solve_once);
  return run;
}

void RunClosestFeasible() {
  std::printf(
      "Infeasible problems (p/4 conflicting pairs, gap %.0f): elastic l1\n"
      "relaxation vs ProxQP-dense on the hard problem.\n"
      "  detect = primal_infeasibility_solving OFF (time to the\n"
      "           infeasibility certificate; no usable primal returned)\n"
      "  clfeas = primal_infeasibility_solving ON (closest feasible QP,\n"
      "           minimal l2 shift of the constraints)\n"
      "  '!' marks runs not reaching the expected status.\n\n",
      kGap);
  std::printf("  %4s %4s | %12s %5s | %9s %5s | %9s %5s | %11s %7s\n", "n",
              "p", "elastiqp[us]", "it", "detect", "it", "clfeas", "it",
              "cf/elastiqp", "cf/det");
  for (auto [n, p] : {std::pair{14, 100}, {30, 200}, {58, 500}}) {
    std::mt19937 rng(7 * n + p);
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4, kGap);
    const ElasticRun el = RunElastic(qp, 10.0);
    const ElasticRun det = RunProxHard(qp, /*closest_feasible=*/false);
    const ElasticRun cf = RunProxHard(qp, /*closest_feasible=*/true);
    std::printf(
        "  %4d %4d | %12.1f %5.0f%s| %9.1f %5.0f%s| %9.1f %5.0f%s| %10.1fx "
        "%6.1fx\n",
        n, p, el.st.us, el.st.iters, el.st.ok ? " " : "!", det.st.us,
        det.st.iters, det.st.ok ? " " : "!", cf.st.us, cf.st.iters,
        cf.st.ok ? " " : "!", cf.st.us / el.st.us, cf.st.us / det.st.us);
  }

  std::printf(
      "\nStructure of the constraint shift at the returned solution\n"
      "(violations (Gx - h)_+). Each of the p/4 conflicting pairs must\n"
      "absorb its gap somewhere between its two rows (so l1 >= p/4 * gap\n"
      "and up to 2*(p/4) rows can be violated); every other constraint is\n"
      "individually satisfiable. 'spur' counts violated constraints outside\n"
      "the conflict set -- shift leaked onto constraints that never needed\n"
      "to move, which the l1 penalty suppresses and the l2 objective\n"
      "invites.\n\n");
  auto row = [](const ShiftStats& s) {
    static char buf[4][64];
    static int slot = 0;
    slot = (slot + 1) % 4;
    std::snprintf(buf[slot], sizeof(buf[slot]), "%4d %4d %7.1f %6.2f", s.nnz,
                  s.spurious, s.l1, s.linf);
    return buf[slot];
  };
  std::printf("  %4s %4s %5s | %24s | %24s | %24s\n", "n", "p", "confl",
              "elastiqp pen=10", "elastiqp pen=1e4", "proxqp clfeas");
  std::printf("  %4s %4s %5s | %24s | %24s | %24s\n", "", "", "",
              "nnz spur      l1   linf", "nnz spur      l1   linf",
              "nnz spur      l1   linf");
  for (auto [n, p] : {std::pair{14, 100}, {30, 200}, {58, 500}}) {
    std::mt19937 rng(7 * n + p);
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4, kGap);
    const ElasticRun lo = RunElastic(qp, 10.0, p / 4);
    const ElasticRun hi = RunElastic(qp, 1e4, p / 4);
    const ElasticRun cf = RunProxHard(qp, /*closest_feasible=*/true, p / 4);
    std::printf("  %4d %4d %5d | %24s | %24s | %24s\n", n, p, p / 4,
                row(lo.shift), row(hi.shift), row(cf.shift));
  }

  std::printf(
      "\nOverhead check on FEASIBLE problems (does enabling\n"
      "primal_infeasibility_solving cost anything when it never fires?):\n\n");
  std::printf("  %4s %4s | %12s %5s | %9s %5s | %9s %5s\n", "n", "p",
              "elastiqp[us]", "it", "prox[us]", "it", "prox+cf", "it");
  for (auto [n, p] : {std::pair{14, 100}, {30, 200}, {58, 500}}) {
    std::mt19937 rng(7 * n + p);
    const QPData qp = problem_gen::Feasible(rng, n, p);
    const ElasticRun el = RunElastic(qp, 10.0);
    ElasticRun off, on;
    for (bool cf : {false, true}) {
      ElasticRun& r = cf ? on : off;
      const VectorXd l = VectorXd::Constant(p, -kInf);
      const int reps = std::max(1, Repeats(n, p) / 5);
      r.st.us = TimeUs(reps, [&] {
        pq::dense::QP<double> solver(n, 0, p);
        ProxSettings(solver.settings, false);
        solver.settings.primal_infeasibility_solving = cf;
        solver.init(qp.Q, qp.q, proxsuite::nullopt, proxsuite::nullopt, qp.G,
                    l, qp.h);
        solver.solve();
        r.st.iters = static_cast<double>(solver.results.info.iter);
        r.st.ok =
            solver.results.info.status == pq::QPSolverOutput::PROXQP_SOLVED;
      });
    }
    std::printf("  %4d %4d | %12.1f %5.0f%s| %9.1f %5.0f%s| %9.1f %5.0f%s\n",
                n, p, el.st.us, el.st.iters, el.st.ok ? " " : "!", off.st.us,
                off.st.iters, off.st.ok ? " " : "!", on.st.us, on.st.iters,
                on.st.ok ? " " : "!");
  }
}

}  // namespace

int main() {
  std::printf(
      "ElastiQP (elastic l1 relaxation) vs ProxQP closest-feasible mode\n"
      "(eps_abs = %.0e, eps_rel = 0; iteration counts are not comparable\n"
      "across solvers)\n\n",
      kEps);
  RunClosestFeasible();
  return 0;
}
