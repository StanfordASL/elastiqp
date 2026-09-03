// Hard equalities vs. double-sided elastic equalities.
//
// ElastiQP keeps A x = b hard (its own dual block, mu_eq schedule, consistency
// certificate). The alternative is to drop the equality block and fold each
// row into a pair of elastic inequalities  a'x - t1 <= b, -a'x - t2 <= -b
// with a large penalty w. By the exact-penalty property this reproduces the
// hard solution iff w > ||y*||_inf, at the price of 2m extra elastic rows and
// a penalty-driven conditioning of the AL. This benchmark measures that price:
// per-tick solve time / iterations, fails, delivered ||Ax - b||_inf, and the
// deviation of x from the hard-equality solution, across a ladder of w, on
// the committed robot sequences (hum-wbc, biman-ik) and on drifting random
// QPs with equalities. Cold and warm modes.
//
// Usage: bench_eq_elastic [sequence_file] [--csv <dir>]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "elastiqp/elastiqp.hpp"
#include "problem_gen.hpp"
#include "qp_io.hpp"

#ifndef ELASTIQP_ARCH_LABEL
#define ELASTIQP_ARCH_LABEL "default"
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using robot_control::NamedSequence;
using robot_control::RobotQP;
using std::chrono::duration;
using std::chrono::steady_clock;

namespace {

constexpr double kEps = 1e-6;
const double kWeights[] = {1e2, 1e3, 1e4, 1e6};

struct Stats {
  std::vector<double> us;
  double mean_us = 0, p95_us = 0, iters = 0;
  double worst_eq = 0;    // ||Ax - b||_inf on the ORIGINAL equality rows
  double worst_dx = 0;    // ||x - x_hard||_inf
  double worst_dobj = 0;  // max relative objective gap vs the hard solution
  double worst_y = 0;     // ||y||_inf of the hard route (elastic: n/a)
  int fails = 0;
};

double Percentile(std::vector<double> v, double frac) {
  const auto idx = static_cast<std::size_t>(
      frac * static_cast<double>(v.size() - 1) + 0.5);
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(idx),
                   v.end());
  return v[idx];
}

// Elastic folding of the equality block: [G; A; -A], [h; b; -b], w on the
// 2m new rows.
RobotQP Fold(const RobotQP& qp, double w) {
  const auto m = qp.b.size(), p = qp.h.size(), n = qp.q.size();
  RobotQP f;
  f.Q = qp.Q;
  f.q = qp.q;
  f.A.resize(0, n);
  f.b.resize(0);
  f.G.resize(p + 2 * m, n);
  f.G << qp.G, qp.A, -qp.A;
  f.h.resize(p + 2 * m);
  f.h << qp.h, qp.b, -qp.b;
  f.penalty.resize(p + 2 * m);
  f.penalty << qp.penalty, VectorXd::Constant(2 * m, w);
  return f;
}

// Runs a sequence through one persistent solver; x_ref (optional) is the
// per-tick reference solution for the deviation column.
Stats Run(const std::vector<RobotQP>& seq, const std::vector<RobotQP>& orig,
          bool warm, const std::vector<VectorXd>* x_ref,
          std::vector<VectorXd>* x_out) {
  elastiqp::Solver solver;
  solver.settings.eps_abs = kEps;
  solver.settings.eps_rel = 0;
  solver.settings.eps_duality_gap_abs = kEps;
  solver.settings.eps_duality_gap_rel = 0;
  solver.settings.warm_start = warm;
  const RobotQP& first = seq.front();
  solver.setup(first.Q, first.q, first.A, first.b, first.G, first.h,
               first.penalty);
  Stats st;
  long total_iters = 0;
  for (std::size_t k = 0; k < seq.size(); ++k) {
    const RobotQP& qp = seq[k];
    solver.set_Q(qp.Q);
    solver.set_q(qp.q);
    if (qp.b.size() > 0) {
      solver.set_A(qp.A);
      solver.set_b(qp.b);
    }
    solver.set_G(qp.G);
    solver.set_h(qp.h);
    const auto t0 = steady_clock::now();
    const elastiqp::Solution& sol = solver.solve();
    const auto t1 = steady_clock::now();
    st.us.push_back(duration<double, std::micro>(t1 - t0).count());
    total_iters += sol.iters;
    if (sol.converged != 1) st.fails++;
    const RobotQP& o = orig[k];
    st.worst_eq = std::max(st.worst_eq,
                           (o.A * sol.x - o.b).lpNorm<Eigen::Infinity>());
    if (x_ref) {
      st.worst_dx = std::max(
          st.worst_dx, (sol.x - (*x_ref)[k]).lpNorm<Eigen::Infinity>());
      const double f_el = problem_gen::ElasticObjective(
          o.Q, o.q, o.G, o.h, o.penalty, sol.x);
      const double f_hard = problem_gen::ElasticObjective(
          o.Q, o.q, o.G, o.h, o.penalty, (*x_ref)[k]);
      st.worst_dobj = std::max(
          st.worst_dobj, (f_el - f_hard) / std::max(1.0, std::abs(f_hard)));
    }
    if (sol.y.size() > 0)
      st.worst_y = std::max(st.worst_y, sol.y.lpNorm<Eigen::Infinity>());
    if (x_out) x_out->push_back(sol.x);
  }
  double sum = 0;
  for (double u : st.us) sum += u;
  st.mean_us = sum / static_cast<double>(st.us.size());
  st.p95_us = Percentile(st.us, 0.95);
  st.iters = static_cast<double>(total_iters) / static_cast<double>(st.us.size());
  return st;
}

struct Row {
  std::string scenario, route, mode;
  int n, m, p;
  Stats st;
};

void RunScenario(const std::string& name, const std::vector<RobotQP>& seq,
                 std::vector<Row>& rows) {
  const RobotQP& f = seq.front();
  const int n = static_cast<int>(f.q.size()), m = static_cast<int>(f.b.size()),
            p = static_cast<int>(f.h.size());
  std::printf("%-14s n=%3d m=%3d p=%3d\n", name.c_str(), n, m, p);
  for (int mode = 0; mode < 2; ++mode) {
    const bool warm = mode == 1;
    std::vector<VectorXd> x_hard;
    Stats hard = Run(seq, seq, warm, nullptr, &x_hard);
    rows.push_back({name, "hard", warm ? "warm" : "cold", n, m, p, hard});
    std::printf("  %-4s %-12s %8.1f us (p95 %8.1f) %5.1f it | eq %.1e dx %8s "
                "|y|max %.1e %s\n",
                warm ? "warm" : "cold", "hard", hard.mean_us, hard.p95_us,
                hard.iters, hard.worst_eq, "-", hard.worst_y,
                hard.fails ? "(FAILS!)" : "");
    for (double w : kWeights) {
      std::vector<RobotQP> folded;
      folded.reserve(seq.size());
      for (const RobotQP& qp : seq) folded.push_back(Fold(qp, w));
      Stats el = Run(folded, seq, warm, &x_hard, nullptr);
      char route[32];
      std::snprintf(route, sizeof route, "elastic-w%.0e", w);
      rows.push_back({name, route, warm ? "warm" : "cold", n, m, p, el});
      std::printf("  %-4s %-12s %8.1f us (p95 %8.1f) %5.1f it | eq %.1e dx "
                  "%.1e dobj %+.1e | %.2fx %s\n",
                  warm ? "warm" : "cold", route, el.mean_us, el.p95_us,
                  el.iters, el.worst_eq, el.worst_dx, el.worst_dobj,
                  el.mean_us / hard.mean_us,
                  el.fails ? "(FAILS!)" : "");
    }
  }
}

// Drifting random QP with equalities: one RandomFeasible instance whose data
// drifts by a random walk each tick (coherent, robot-like).
std::vector<RobotQP> RandomDrift(int n, int m, int p, int ticks, double drift,
                                 unsigned seed) {
  std::mt19937 rng(seed);
  problem_gen::QPData base = problem_gen::RandomFeasible(rng, n, m, p);
  std::vector<RobotQP> seq;
  RobotQP qp;
  qp.Q = base.Q;
  qp.q = base.q;
  qp.A = base.A;
  qp.b = base.b;
  qp.G = base.G;
  qp.h = base.h;
  qp.penalty = VectorXd::Constant(p, 10.0);
  for (int k = 0; k < ticks; ++k) {
    qp.q += drift * problem_gen::Randn(rng, n, 1);
    qp.b += drift * problem_gen::Randn(rng, m, 1);
    qp.h += drift * problem_gen::Randn(rng, p, 1);
    qp.A += drift * problem_gen::Randn(rng, m, n);
    qp.G += drift * problem_gen::Randn(rng, p, n);
    seq.push_back(qp);
  }
  return seq;
}

void WriteCsv(const std::string& dir, const std::vector<Row>& rows) {
  const std::string path = dir + "/eq_elastic_summary.csv";
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    return;
  }
  std::fprintf(f,
               "scenario,arch,route,mode,n,m,p,ticks,mean_us,p95_us,mean_iters,"
               "worst_eq_residual,worst_dx_vs_hard,worst_dobj_vs_hard,worst_y_hard,fails\n");
  for (const Row& r : rows)
    std::fprintf(f, "%s,%s,%s,%s,%d,%d,%d,%zu,%.3f,%.3f,%.2f,%.3e,%.3e,%.3e,%.3e,%d\n",
                 r.scenario.c_str(), ELASTIQP_ARCH_LABEL, r.route.c_str(),
                 r.mode.c_str(), r.n, r.m, r.p, r.st.us.size(), r.st.mean_us,
                 r.st.p95_us, r.st.iters, r.st.worst_eq, r.st.worst_dx,
                 r.st.worst_dobj, r.st.worst_y, r.st.fails);
  std::fclose(f);
  std::printf("wrote %s\n", path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  std::string path, csv_dir;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--csv" && i + 1 < argc)
      csv_dir = argv[++i];
    else
      path = argv[i];
  }
  if (path.empty()) {
#ifdef ELASTIQP_SEQUENCE_FILE
    path = ELASTIQP_SEQUENCE_FILE;
#else
    path = "robot_sequences.bin";
#endif
  }
  std::printf("ElastiQP hard vs elastic equalities [arch: %s] (eps=%g)\n",
              ELASTIQP_ARCH_LABEL, kEps);
  std::vector<Row> rows;
  for (const NamedSequence& seq : robot_control::LoadSequences(path))
    if (seq.qps.front().b.size() > 0) RunScenario(seq.name, seq.qps, rows);
  RunScenario("rand-n20-m5", RandomDrift(20, 5, 30, 200, 1e-2, 1), rows);
  RunScenario("rand-n60-m20", RandomDrift(60, 20, 90, 200, 1e-2, 2), rows);
  RunScenario("rand-n60-m20-bigdrift",
              RandomDrift(60, 20, 90, 200, 1e-1, 3), rows);
  if (!csv_dir.empty()) WriteCsv(csv_dir, rows);
  return 0;
}
