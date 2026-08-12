// Robotics benchmark for ElastiQP: per-tick solve cost on realistic control
// loops (differential IK, arm OSC, humanoid WBC — see
// robotics/robot_control.hpp), replayed from the sequence file written by
// gen_robot_sequences (the committed benchmarks/data/robot_sequences.bin by
// default). The model evaluation happened at generation time, so the timings
// here measure the solver only.
//
// This target has no Pinocchio dependency, so it can be compiled with any
// -march (e.g. a second build tree configured with -march=native for a
// codegen comparison; the arch column in the CSV carries the provenance).
//
// Every problem matrix drifts every tick (Q, q, A, b, G, h all depend on the
// state), which is the honest robot-control workload for warm starting.
// Modes: cold (persistent solver, warm_start = false: workspace reuse only)
// vs warm (workspace + iterate reuse), at eps = 1e-6.
//
// Each tick is timed individually, so the report carries the distribution
// (mean, std, p50, p95, max), not just the average — worst-case latency is
// what a real-time loop budgets for. Accuracy is measured off the clock:
// the worst elastic KKT residual and the worst hard-equality residual
// ||Ax - b||_inf actually delivered over each trajectory.
//
// Usage: bench_robot_control [sequence_file] [--csv <dir>]
// (default sequence file: the build-time path baked in by CMake, i.e. the
// committed benchmarks/data/robot_sequences.bin). With --csv, writes
// robot_control_summary.csv (one row per scenario x mode) and
// robot_control_ticks.csv (per-tick times) into <dir>.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "elastiqp/elastiqp.hpp"
#include "problem_gen.hpp"
#include "qp_io.hpp"

#ifndef ELASTIQP_ARCH_LABEL
#define ELASTIQP_ARCH_LABEL "default"
#endif

using robot_control::NamedSequence;
using robot_control::RobotQP;
using std::chrono::duration;
using std::chrono::steady_clock;

namespace {

constexpr double kEps = 1e-6;

struct SeqStats {
  std::vector<double> tick_us;   // per-tick solve time
  std::vector<int> tick_iters;   // per-tick iteration count
  double mean_us = 0;
  double std_us = 0;
  double p50_us = 0;
  double p95_us = 0;
  double max_us = 0;
  double iters = 0;
  double worst_kkt = 0;
  double worst_eq = 0;  // ||Ax - b||_inf, 0 for equality-free scenarios
  int fails = 0;
};

using Sequence = std::vector<RobotQP>;

double Percentile(std::vector<double> v, double frac) {
  const auto idx = static_cast<std::size_t>(
      frac * static_cast<double>(v.size() - 1) + 0.5);
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(idx),
                   v.end());
  return v[idx];
}

void Finalize(SeqStats& st) {
  const auto ticks = static_cast<double>(st.tick_us.size());
  double sum = 0, sum2 = 0;
  long total_iters = 0;
  for (double us : st.tick_us) {
    sum += us;
    sum2 += us * us;
  }
  for (int it : st.tick_iters) total_iters += it;
  st.mean_us = sum / ticks;
  st.std_us = std::sqrt(std::max(0.0, sum2 / ticks - st.mean_us * st.mean_us));
  st.p50_us = Percentile(st.tick_us, 0.50);
  st.p95_us = Percentile(st.tick_us, 0.95);
  st.max_us = *std::max_element(st.tick_us.begin(), st.tick_us.end());
  st.iters = static_cast<double>(total_iters) / ticks;
}

SeqStats RunSeq(const Sequence& seq, bool warm) {
  elastiqp::Solver solver;
  solver.settings.eps_abs = kEps;
  solver.settings.eps_rel = 0;
  solver.settings.eps_duality_gap_abs = kEps;
  solver.settings.eps_duality_gap_rel = 0;
  solver.settings.warm_start = warm;
  const RobotQP& first = seq.front();
  solver.setup(first.Q, first.q, first.A, first.b, first.G, first.h,
               first.penalty);

  SeqStats st;
  st.tick_us.reserve(seq.size());
  st.tick_iters.reserve(seq.size());
  for (const RobotQP& qp : seq) {
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
    st.tick_us.push_back(duration<double, std::micro>(t1 - t0).count());
    st.tick_iters.push_back(sol.iters);
    if (sol.converged != 1) st.fails++;
  }
  Finalize(st);
  // Accuracy guard, off the clock: replay the sequence and record the worst
  // elastic KKT and hard-equality residuals actually delivered.
  elastiqp::Solver check;
  check.settings = solver.settings;
  check.setup(first.Q, first.q, first.A, first.b, first.G, first.h,
              first.penalty);
  for (const RobotQP& qp : seq) {
    check.set_Q(qp.Q);
    check.set_q(qp.q);
    if (qp.b.size() > 0) {
      check.set_A(qp.A);
      check.set_b(qp.b);
    }
    check.set_G(qp.G);
    check.set_h(qp.h);
    const elastiqp::Solution& sol = check.solve();
    st.worst_kkt = std::max(
        st.worst_kkt,
        problem_gen::ElasticKKTResidual(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h,
                                        qp.penalty, sol.x, sol.t, sol.y,
                                        sol.z_t, sol.z_ineq));
    if (qp.b.size() > 0) {
      st.worst_eq = std::max(
          st.worst_eq,
          (qp.A * sol.x - qp.b).lpNorm<Eigen::Infinity>());
    }
  }
  return st;
}

void WriteCsv(const std::string& dir,
              const std::vector<NamedSequence>& seqs,
              const std::vector<SeqStats>& cold,
              const std::vector<SeqStats>& warm) {
  const std::string summary_path = dir + "/robot_control_summary.csv";
  std::FILE* f = std::fopen(summary_path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", summary_path.c_str());
    return;
  }
  std::fprintf(f,
               "scenario,arch,solver,mode,n,m,p,ticks,mean_us,std_us,p50_us,"
               "p95_us,max_us,mean_iters,worst_kkt,worst_eq_residual,fails\n");
  for (std::size_t i = 0; i < seqs.size(); ++i) {
    const RobotQP& qp = seqs[i].qps.front();
    for (int mode = 0; mode < 2; ++mode) {
      const SeqStats& st = mode == 0 ? cold[i] : warm[i];
      std::fprintf(
          f, "%s,%s,elastiqp,%s,%d,%d,%d,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,"
          "%.3e,%.3e,%d\n",
          seqs[i].name.c_str(), ELASTIQP_ARCH_LABEL,
          mode == 0 ? "cold" : "warm", static_cast<int>(qp.q.size()),
          static_cast<int>(qp.b.size()), static_cast<int>(qp.h.size()),
          st.tick_us.size(), st.mean_us, st.std_us, st.p50_us, st.p95_us,
          st.max_us, st.iters, st.worst_kkt, st.worst_eq, st.fails);
    }
  }
  std::fclose(f);
  std::printf("wrote %s\n", summary_path.c_str());

  const std::string ticks_path = dir + "/robot_control_ticks.csv";
  f = std::fopen(ticks_path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", ticks_path.c_str());
    return;
  }
  std::fprintf(f, "scenario,arch,solver,mode,tick,us,iters\n");
  for (std::size_t i = 0; i < seqs.size(); ++i) {
    for (int mode = 0; mode < 2; ++mode) {
      const SeqStats& st = mode == 0 ? cold[i] : warm[i];
      for (std::size_t k = 0; k < st.tick_us.size(); ++k) {
        std::fprintf(f, "%s,%s,elastiqp,%s,%zu,%.3f,%d\n", seqs[i].name.c_str(),
                     ELASTIQP_ARCH_LABEL, mode == 0 ? "cold" : "warm", k,
                     st.tick_us[k], st.tick_iters[k]);
      }
    }
  }
  std::fclose(f);
  std::printf("wrote %s\n", ticks_path.c_str());
}

void Report(const NamedSequence& seq, const SeqStats& cold,
            const SeqStats& warm) {
  const int n = static_cast<int>(seq.qps.front().q.size());
  const int m = static_cast<int>(seq.qps.front().b.size());
  const int p = static_cast<int>(seq.qps.front().h.size());
  std::printf(
      "%-16s n=%3d m=%3d p=%3d | cold %8.1f +- %6.1f us (p95 %8.1f) "
      "%5.1f it | warm %8.1f +- %6.1f us (p95 %8.1f) %5.1f it | "
      "speedup %.2fx | kkt %.1e eq %.1e %s\n",
      seq.name.c_str(), n, m, p, cold.mean_us, cold.std_us,
      cold.p95_us, cold.iters, warm.mean_us, warm.std_us, warm.p95_us,
      warm.iters, cold.mean_us / warm.mean_us,
      std::max(cold.worst_kkt, warm.worst_kkt),
      std::max(cold.worst_eq, warm.worst_eq),
      (cold.fails + warm.fails) ? "(FAILS!)" : "");
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  std::string csv_dir;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--csv" && i + 1 < argc) {
      csv_dir = argv[++i];
    } else {
      path = argv[i];
    }
  }
  if (path.empty()) {
#ifdef ELASTIQP_SEQUENCE_FILE
    path = ELASTIQP_SEQUENCE_FILE;
#else
    path = "robot_sequences.bin";
#endif
  }
  std::vector<NamedSequence> seqs;
  try {
    seqs = robot_control::LoadSequences(path);
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "%s\nGenerate the sequence file first: gen_robot_sequences "
                 "%s\n",
                 e.what(), path.c_str());
    return 1;
  }
  std::printf("ElastiQP robotics benchmark [arch: %s] (eps=%g, %s)\n",
              ELASTIQP_ARCH_LABEL, kEps, path.c_str());
  std::vector<SeqStats> cold, warm;
  for (const NamedSequence& seq : seqs) {
    cold.push_back(RunSeq(seq.qps, /*warm=*/false));
    warm.push_back(RunSeq(seq.qps, /*warm=*/true));
    Report(seq, cold.back(), warm.back());
  }
  if (!csv_dir.empty()) WriteCsv(csv_dir, seqs, cold, warm);
  return 0;
}
