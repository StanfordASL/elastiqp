// Differentiability cost at robot scale: what does a gradient add to a
// control tick?
//
// Replays the robot control-loop sequences (see bench_robot_control.cc) and
// per tick runs the full differentiable-solve pipeline: warm forward solve,
// then relax(kappa) to the kappa-relaxed central point, then one KKT vjp
// (elastiqp/kkt_vjp.hpp) -- the pattern of a differentiable-MPC loop. The
// relax() always starts from the retraction of the tick's tight certificate
// (the standard cold start). The robot problems span n=6..46, m=0..18,
// p=24..132 at drift ~1e-3..1e-4 per tick.
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

// relax() terminates on the max UNSCALED KKT residual. The penalty tiers
// here reach 1e5, and an inactive row with penalty w relaxes to the pair
// (z, s) = (w, kappa/w): the relax() Newton elimination amplifies roundoff
// by z/s = w^2/kappa, which reaches 1e14 at kappa = 1e-6 and stalls the
// corrector well above any usable tolerance. Ruiz equilibration removes
// that amplification at the source (the scaled penalty is O(1)), which is
// why it is enabled below -- with it, the 1e-8 tolerance is attainable at
// every kappa in the sweep. max_iter matches the jax_ffi default; ticks
// that exhaust it are counted in the fail columns, not hidden.
constexpr double kFwdEps = 1e-8;
constexpr double kRelaxTol = 1e-8;
constexpr int kRelaxMaxIter = 50;
constexpr double kKappas[] = {1e-3, 1e-4, 1e-6};

struct TickRow {
  double fwd_us = 0;    // warm forward solve
  double relax_us = 0;
  int relax_iters = 0;  // relax() Newton steps
  double vjp_us = 0;
};

struct CellStats {
  std::vector<TickRow> ticks;
  int fails_relax = 0;  // relax() did not converge
  int vjp_bad = 0;      // non-finite backward pass
  int fails() const { return fails_relax + vjp_bad; }
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

elastiqp::Solver MakeSolver(const RobotQP& first) {
  elastiqp::Solver solver;
  solver.settings.eps_abs = kFwdEps;
  solver.settings.eps_rel = 0;
  solver.settings.warm_start = true;
  solver.settings.ruiz = true;  // see the kRelaxTol comment above
  solver.setup(first.Q, first.q, first.A, first.b, first.G, first.h,
               first.penalty);
  return solver;
}

CellStats RunCell(const std::vector<RobotQP>& seq, double kappa) {
  const RobotQP& first = seq.front();
  const int n = static_cast<int>(first.q.size());
  const int m = static_cast<int>(first.b.size());
  const int p = static_cast<int>(first.h.size());

  elastiqp::Solver solver = MakeSolver(first);
  elastiqp::KktVjp vjp;
  vjp.setup(n, m, p);  // compute() below is then allocation-free
  elastiqp::Cotangents ct;

  CellStats st;
  st.ticks.reserve(seq.size());
  for (std::size_t k = 0; k < seq.size(); ++k) {
    const RobotQP& qp = seq[k];
    TickRow row;

    SetData(solver, qp);
    auto t0 = steady_clock::now();
    solver.solve();
    row.fwd_us = duration<double, std::micro>(steady_clock::now() - t0).count();

    t0 = steady_clock::now();
    const elastiqp::Solution& rel =
        solver.relax(kappa, kRelaxTol, kRelaxMaxIter);
    row.relax_us =
        duration<double, std::micro>(steady_clock::now() - t0).count();
    row.relax_iters = rel.iters;
    if (rel.converged != 1) st.fails_relax++;

    // Backward pass at the relaxed point, cotangent on x only, as for a
    // loss 0.5*||x||^2.
    ct.x = rel.x;
    t0 = steady_clock::now();
    const elastiqp::DataGrads& g =
        vjp.compute(qp.Q, qp.A, qp.G, qp.h, rel, ct);
    row.vjp_us =
        duration<double, std::micro>(steady_clock::now() - t0).count();
    if (!std::isfinite(g.q.sum())) st.vjp_bad++;

    st.ticks.push_back(row);
  }
  return st;
}

struct CellSummary {
  double fwd_mean, fwd_p95;
  double rx_mean, rx_p50, rx_p95, rx_iters;
  double vjp_mean, amort;
};

CellSummary Summarize(const CellStats& st) {
  CellSummary s{};
  const std::vector<double> fwd = Extract(st.ticks, &TickRow::fwd_us);
  const std::vector<double> rx = Extract(st.ticks, &TickRow::relax_us);
  const std::vector<double> vjp = Extract(st.ticks, &TickRow::vjp_us);
  s.fwd_mean = Mean(fwd);
  s.fwd_p95 = Percentile(fwd, 0.95);
  s.rx_mean = Mean(rx);
  s.rx_p50 = Percentile(rx, 0.50);
  s.rx_p95 = Percentile(rx, 0.95);
  s.vjp_mean = Mean(vjp);
  double it = 0;
  for (const TickRow& r : st.ticks) it += r.relax_iters;
  s.rx_iters = it / static_cast<double>(st.ticks.size());
  s.amort = (s.rx_mean + s.vjp_mean) / s.fwd_mean;
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
      "per tick: warm solve -> relax(kappa) -> kkt vjp\n\n",
      ELASTIQP_ARCH_LABEL, kFwdEps, kRelaxTol, path.c_str());
  std::printf(
      "%-9s %7s | %8s %8s | %8s %8s %8s (%4s it) | %7s | %5s | %s\n",
      "scenario", "kappa", "fwd", "fwd p95", "relax", "rx p50", "rx p95", "",
      "vjp", "amort", "fails (relax/vjp)");

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
                 "relax_mean_us,relax_p50_us,relax_p95_us,relax_iters,"
                 "vjp_mean_us,amort_ratio,fails_relax,vjp_bad\n");
    std::fprintf(ftick,
                 "scenario,arch,kappa,tick,fwd_us,relax_us,relax_iters,"
                 "vjp_us\n");
  }

  int total_fails = 0;
  for (const NamedSequence& seq : seqs) {
    const RobotQP& qp = seq.qps.front();
    for (const double kappa : kKappas) {
      const CellStats st = RunCell(seq.qps, kappa);
      const CellSummary s = Summarize(st);
      total_fails += st.fails();
      std::printf(
          "%-9s %7.0e | %8.1f %8.1f | %8.1f %8.1f %8.1f (%4.1f it) | %7.1f "
          "| %5.2f | %d/%d%s\n",
          seq.name.c_str(), kappa, s.fwd_mean, s.fwd_p95, s.rx_mean, s.rx_p50,
          s.rx_p95, s.rx_iters, s.vjp_mean, s.amort, st.fails_relax,
          st.vjp_bad, st.fails() ? " (FAILS!)" : "");
      if (fsum) {
        std::fprintf(
            fsum,
            "%s,%s,%g,%d,%d,%d,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.3f,%.3f,"
            "%d,%d\n",
            seq.name.c_str(), ELASTIQP_ARCH_LABEL, kappa,
            static_cast<int>(qp.q.size()), static_cast<int>(qp.b.size()),
            static_cast<int>(qp.h.size()), st.ticks.size(), s.fwd_mean,
            s.fwd_p95, s.rx_mean, s.rx_p50, s.rx_p95, s.rx_iters, s.vjp_mean,
            s.amort, st.fails_relax, st.vjp_bad);
      }
      if (ftick) {
        for (std::size_t k = 0; k < st.ticks.size(); ++k) {
          const TickRow& r = st.ticks[k];
          std::fprintf(ftick, "%s,%s,%g,%zu,%.3f,%.3f,%d,%.3f\n",
                       seq.name.c_str(), ELASTIQP_ARCH_LABEL, kappa, k,
                       r.fwd_us, r.relax_us, r.relax_iters, r.vjp_us);
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
