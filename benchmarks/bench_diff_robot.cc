// Differentiability cost at robot scale: what does a gradient add to a
// control tick, and does warm-starting relax() across ticks pay?
//
// Replays the robot control-loop sequences (see bench_robot_control.cc) and
// per tick runs the full differentiable-solve pipeline: warm forward solve,
// then relax(kappa) to the kappa-relaxed central point, then one KKT vjp
// (elastiqp/kkt_vjp.hpp) -- the pattern of a differentiable-MPC loop. The
// bench_collision_2d demo is the deliberate small-scale contrast: there
// (n=4, p=8) the relax warm start is a documented no-op because the fresh
// retraction start already converges at its 2-step floor and the per-tick
// drift exceeds the warm-candidate adoption basin (0.02*sqrt(kappa)). The
// robot problems (up to n=46, m=18, p=132, drift ~1e-3..1e-4) sit in the
// regime where docs/pdal_differentiability.md measured 2-2.4x warm relax
// speedups on synthetic instances; this benchmark measures whether that
// carries over to real robot data.
//
// Protocol: two solvers march over the sequence in lockstep with identical
// settings and identical warm forward solves, differing ONLY in
// settings.relax_warm_start (false vs true), so per tick the cold and warm
// relax() calls start from the same tight solve on the same data. The
// forward-solve timings of the two instances are both reported as a sanity
// control (they run the same computation and must agree statistically; the
// timing order alternates per tick to cancel cache-warming bias). The warm
// start must never change the answer, only the cost: each tick asserts both
// relax() calls converged to the same relaxed point.
//
// Verdict criterion (mechanical, agreed up front): keep
// Settings::relax_warm_start if warm relax achieves mean <= 0.9x cold on at
// least one robot scenario at kappa = 1e-3 AND warm p95 <= 1.1x cold on
// every (scenario, kappa) cell (graceful degradation: a fallback tick costs
// three residual evaluations, never a factorization). If it fails both,
// this output is the evidence for removing the feature.
//
// Usage: bench_diff_robot [sequence_file] [--csv <dir>]
// With --csv, writes diff_robot_summary.csv (one row per scenario x kappa)
// and diff_robot_ticks.csv (per-tick times) into <dir>.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "elastiqp/elastiqp.hpp"
#include "elastiqp/kkt_vjp.hpp"
#include "qp_io.hpp"

#ifndef ELASTIQP_ARCH_LABEL
#define ELASTIQP_ARCH_LABEL "default"
#endif

using robot_control::NamedSequence;
using robot_control::RobotQP;
using std::chrono::duration;
using std::chrono::steady_clock;

namespace {

// relax() terminates on the max UNSCALED KKT residual, and the torque-box
// penalty rows are 1e5-scale, where double precision floors the attainable
// residual near 2e-11 -- so the library-default 1e-8 is the honest tight
// tolerance here (1e-10, the setting of the synthetic O(1)-scale experiments
// in docs/pdal_differentiability.md, sits at that floor). max_iter matches
// the jax_ffi default; ticks that exhaust it are counted in the fail
// columns, not hidden.
constexpr double kFwdEps = 1e-8;
constexpr double kRelaxTol = 1e-8;
constexpr int kRelaxMaxIter = 50;
constexpr double kKappas[] = {1e-3, 1e-4, 1e-6};

struct TickRow {
  double fwd_cold_us = 0;   // forward solve on the relax-cold instance
  double fwd_warm_us = 0;   // forward solve on the relax-warm instance
  double relax_cold_us = 0;
  double relax_warm_us = 0;
  int iters_cold = 0;       // relax() Newton steps (incl. any regret budget)
  int iters_warm = 0;
  double vjp_us = 0;
};

struct CellStats {
  std::vector<TickRow> ticks;
  int fails_cold = 0;  // cold relax() did not converge
  int fails_warm = 0;  // warm relax() did not converge
  int mismatch = 0;    // both converged but to different relaxed points
  int vjp_bad = 0;     // non-finite backward pass
  int fails() const { return fails_cold + fails_warm + mismatch + vjp_bad; }
};

double Mean(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

double Percentile(std::vector<double> v, double frac) {
  const auto idx = static_cast<std::size_t>(
      frac * static_cast<double>(v.size() - 1) + 0.5);
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(idx),
                   v.end());
  return v[idx];
}

std::vector<double> Extract(const std::vector<TickRow>& rows,
                            double TickRow::*field) {
  std::vector<double> v;
  v.reserve(rows.size());
  for (const TickRow& r : rows) v.push_back(r.*field);
  return v;
}

void SetData(elastiqp::Solver& solver, const RobotQP& qp) {
  solver.set_Q(qp.Q);
  solver.set_q(qp.q);
  if (qp.b.size() > 0) {
    solver.set_A(qp.A);
    solver.set_b(qp.b);
  }
  solver.set_G(qp.G);
  solver.set_h(qp.h);
}

elastiqp::Solver MakeSolver(const RobotQP& first, bool relax_warm) {
  elastiqp::Solver solver;
  solver.settings.eps_abs = kFwdEps;
  solver.settings.eps_rel = 0;
  solver.settings.warm_start = true;
  solver.settings.relax_warm_start = relax_warm;
  solver.setup(first.Q, first.q, first.A, first.b, first.G, first.h,
               first.penalty);
  return solver;
}

CellStats RunCell(const std::vector<RobotQP>& seq, double kappa) {
  const RobotQP& first = seq.front();
  const int n = static_cast<int>(first.q.size());
  const int m = static_cast<int>(first.b.size());
  const int p = static_cast<int>(first.h.size());

  elastiqp::Solver cold = MakeSolver(first, /*relax_warm=*/false);
  elastiqp::Solver warm = MakeSolver(first, /*relax_warm=*/true);
  elastiqp::KktVjp vjp;
  vjp.setup(n, m, p);  // compute() below is then allocation-free
  elastiqp::Cotangents ct;

  CellStats st;
  st.ticks.reserve(seq.size());
  for (std::size_t k = 0; k < seq.size(); ++k) {
    const RobotQP& qp = seq[k];
    TickRow row;

    // Identical forward solves; timing order alternates to cancel
    // cache-warming bias between the two instances.
    elastiqp::Solver* order[2] = {&cold, &warm};
    double* fwd_us[2] = {&row.fwd_cold_us, &row.fwd_warm_us};
    const int a = static_cast<int>(k % 2), b2 = 1 - a;
    SetData(*order[a], qp);
    auto t0 = steady_clock::now();
    order[a]->solve();
    *fwd_us[a] = duration<double, std::micro>(steady_clock::now() - t0).count();
    SetData(*order[b2], qp);
    t0 = steady_clock::now();
    order[b2]->solve();
    *fwd_us[b2] =
        duration<double, std::micro>(steady_clock::now() - t0).count();

    // relax(): the only difference between the two instances.
    t0 = steady_clock::now();
    const elastiqp::Solution& rc = cold.relax(kappa, kRelaxTol, kRelaxMaxIter);
    row.relax_cold_us =
        duration<double, std::micro>(steady_clock::now() - t0).count();
    row.iters_cold = rc.iters;
    const bool cold_ok = rc.converged == 1;
    const Eigen::VectorXd xc = rc.x;

    t0 = steady_clock::now();
    const elastiqp::Solution& rw = warm.relax(kappa, kRelaxTol, kRelaxMaxIter);
    row.relax_warm_us =
        duration<double, std::micro>(steady_clock::now() - t0).count();
    row.iters_warm = rw.iters;

    // Off the clock: the warm start may only change the cost, not the point.
    if (!cold_ok) st.fails_cold++;
    if (rw.converged != 1) st.fails_warm++;
    if (cold_ok && rw.converged == 1) {
      const double dx = (rw.x - xc).lpNorm<Eigen::Infinity>();
      const double tol_x =
          10 * kRelaxTol * std::max(1.0, xc.lpNorm<Eigen::Infinity>());
      if (dx > tol_x) st.mismatch++;
    }

    // Backward pass at the warm instance's relaxed point, cotangent on x
    // only, as for a loss 0.5*||x||^2.
    ct.x = rw.x;
    t0 = steady_clock::now();
    const elastiqp::DataGrads& g = vjp.compute(qp.Q, qp.A, qp.G, qp.h, rw, ct);
    row.vjp_us =
        duration<double, std::micro>(steady_clock::now() - t0).count();
    if (!std::isfinite(g.q.sum())) st.vjp_bad++;

    st.ticks.push_back(row);
  }
  return st;
}

struct CellSummary {
  double fwd_mean, fwd_p95;
  double rxc_mean, rxc_p50, rxc_p95, rxc_iters;
  double rxw_mean, rxw_p50, rxw_p95, rxw_iters;
  double speedup, adopt_frac, regret_frac;
  double vjp_mean, amort;
};

CellSummary Summarize(const CellStats& st) {
  CellSummary s{};
  // Both forward columns measure the same computation; pool them.
  std::vector<double> fwd = Extract(st.ticks, &TickRow::fwd_cold_us);
  const std::vector<double> fw = Extract(st.ticks, &TickRow::fwd_warm_us);
  fwd.insert(fwd.end(), fw.begin(), fw.end());
  const std::vector<double> rxc = Extract(st.ticks, &TickRow::relax_cold_us);
  const std::vector<double> rxw = Extract(st.ticks, &TickRow::relax_warm_us);
  const std::vector<double> vjp = Extract(st.ticks, &TickRow::vjp_us);
  s.fwd_mean = Mean(fwd);
  s.fwd_p95 = Percentile(fwd, 0.95);
  s.rxc_mean = Mean(rxc);
  s.rxc_p50 = Percentile(rxc, 0.50);
  s.rxc_p95 = Percentile(rxc, 0.95);
  s.rxw_mean = Mean(rxw);
  s.rxw_p50 = Percentile(rxw, 0.50);
  s.rxw_p95 = Percentile(rxw, 0.95);
  s.vjp_mean = Mean(vjp);
  double ic = 0, iw = 0, adopt = 0, regret = 0;
  for (const TickRow& r : st.ticks) {
    ic += r.iters_cold;
    iw += r.iters_warm;
    adopt += r.iters_warm < r.iters_cold;
    regret += r.iters_warm > r.iters_cold;
  }
  const auto ticks = static_cast<double>(st.ticks.size());
  s.rxc_iters = ic / ticks;
  s.rxw_iters = iw / ticks;
  s.adopt_frac = adopt / ticks;
  s.regret_frac = regret / ticks;
  s.speedup = s.rxc_mean / s.rxw_mean;
  s.amort = (s.rxw_mean + s.vjp_mean) / s.fwd_mean;
  return s;
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

  std::printf(
      "ElastiQP differentiability-at-robot-scale benchmark [arch: %s]\n"
      "forward eps=%g, relax tol=%g, %s\n"
      "per tick: warm solve -> relax(kappa) cold vs warm -> kkt vjp\n\n",
      ELASTIQP_ARCH_LABEL, kFwdEps, kRelaxTol, path.c_str());
  std::printf(
      "%-9s %7s | %8s %8s | %8s (%4s it) %8s (%4s it) %6s %5s %5s | %7s | "
      "%5s | %s\n",
      "scenario", "kappa", "fwd", "fwd p95", "rx cold", "", "rx warm", "",
      "spdup", "adopt", "regrt", "vjp", "amort",
      "fails (cold/warm/mismatch/vjp)");

  std::FILE* fsum = nullptr;
  std::FILE* ftick = nullptr;
  if (!csv_dir.empty()) {
    const std::string sp = csv_dir + "/diff_robot_summary.csv";
    const std::string tp = csv_dir + "/diff_robot_ticks.csv";
    fsum = std::fopen(sp.c_str(), "w");
    ftick = std::fopen(tp.c_str(), "w");
    if (!fsum || !ftick) {
      std::fprintf(stderr, "cannot write CSVs into %s\n", csv_dir.c_str());
      return 1;
    }
    std::fprintf(fsum,
                 "scenario,arch,kappa,n,m,p,ticks,fwd_mean_us,fwd_p95_us,"
                 "relax_cold_mean_us,relax_cold_p50_us,relax_cold_p95_us,"
                 "relax_cold_iters,relax_warm_mean_us,relax_warm_p50_us,"
                 "relax_warm_p95_us,relax_warm_iters,speedup,adopt_frac,"
                 "regret_frac,vjp_mean_us,amort_ratio,fails_cold,fails_warm,"
                 "mismatch,vjp_bad\n");
    std::fprintf(ftick,
                 "scenario,arch,kappa,tick,fwd_cold_us,fwd_warm_us,"
                 "relax_cold_us,relax_warm_us,iters_cold,iters_warm,vjp_us\n");
  }

  int total_fails = 0;
  for (const NamedSequence& seq : seqs) {
    const RobotQP& qp = seq.qps.front();
    for (const double kappa : kKappas) {
      const CellStats st = RunCell(seq.qps, kappa);
      const CellSummary s = Summarize(st);
      total_fails += st.fails();
      std::printf(
          "%-9s %7.0e | %8.1f %8.1f | %8.1f (%4.1f it) %8.1f (%4.1f it) "
          "%5.2fx %5.2f %5.2f | %7.1f | %5.2f | %d/%d/%d/%d%s\n",
          seq.name.c_str(), kappa, s.fwd_mean, s.fwd_p95, s.rxc_mean,
          s.rxc_iters, s.rxw_mean, s.rxw_iters, s.speedup, s.adopt_frac,
          s.regret_frac, s.vjp_mean, s.amort, st.fails_cold, st.fails_warm,
          st.mismatch, st.vjp_bad, st.fails() ? " (FAILS!)" : "");
      if (fsum) {
        std::fprintf(
            fsum,
            "%s,%s,%g,%d,%d,%d,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.3f,%.3f,"
            "%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d\n",
            seq.name.c_str(), ELASTIQP_ARCH_LABEL, kappa,
            static_cast<int>(qp.q.size()), static_cast<int>(qp.b.size()),
            static_cast<int>(qp.h.size()), st.ticks.size(), s.fwd_mean,
            s.fwd_p95, s.rxc_mean, s.rxc_p50, s.rxc_p95, s.rxc_iters,
            s.rxw_mean, s.rxw_p50, s.rxw_p95, s.rxw_iters, s.speedup,
            s.adopt_frac, s.regret_frac, s.vjp_mean, s.amort, st.fails_cold,
            st.fails_warm, st.mismatch, st.vjp_bad);
      }
      if (ftick) {
        for (std::size_t k = 0; k < st.ticks.size(); ++k) {
          const TickRow& r = st.ticks[k];
          std::fprintf(ftick, "%s,%s,%g,%zu,%.3f,%.3f,%.3f,%.3f,%d,%d,%.3f\n",
                       seq.name.c_str(), ELASTIQP_ARCH_LABEL, kappa, k,
                       r.fwd_cold_us, r.fwd_warm_us, r.relax_cold_us,
                       r.relax_warm_us, r.iters_cold, r.iters_warm, r.vjp_us);
        }
      }
    }
    std::printf("\n");
  }
  if (fsum) {
    std::fclose(fsum);
    std::printf("wrote %s/diff_robot_summary.csv\n", csv_dir.c_str());
  }
  if (ftick) {
    std::fclose(ftick);
    std::printf("wrote %s/diff_robot_ticks.csv\n", csv_dir.c_str());
  }
  return total_fails ? 1 : 0;
}
