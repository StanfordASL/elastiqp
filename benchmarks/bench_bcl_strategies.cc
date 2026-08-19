// Which BCL strategy is correct for the elastic problem?
//
// The forward warm solve showed failures in bench_relax_warm at commit
// b995082 (the last proxqp-parity BCL): at penalty 1e4 and small drift
// (sigma 1e-4), up to 19 of 100 warm ticks hit kMaxIter. The cause is
// the interaction of proxqp's BCL/reset machinery with elastic
// saturation creep: a drift tick pushes weakly-active rows toward
// saturation, the multipliers creep there at ~r/mu per outer round,
// the reset test reads the creep as "no progress" and re-widens mu --
// a limit cycle. Commits ce51511..4a21686 replaced that machinery in
// four steps, each behind a Settings flag.
//
// This benchmark isolates the failure regime and replays it under each
// strategy generation (cumulative, oldest first):
//   proxqp   uncapped cold reset, classic BCL      (b995082 parity)
//   cap1     cold reset capped at 1 per solve      (719cbf8)
//   noreset  cold reset off                        (ce51511)
//   split    + block-split bad step                (0a0025a)
//   jump     + creep-resolving mu jump             (de1c48a)
//   eta      + warm-start eta seeding = SHIPPED    (4a21686, defaults)
//
// Cells: the creep regime (penalty 1e4, sigma 1e-4 / 1e-3, all three
// structures, both eps tiers); the same regime with MIXED per-row
// penalties (w in {10, 1e4}: alt / spike / dip patterns -- the
// bcl_mu_jump target is read off the single worst-residual row, so a
// soft row winning that argmax while stiff rows creep is the suspected
// weak spot); and control cells (penalty 10; large drift sigma 1e-1). Trajectories
// use the same generator and seed formula as bench_relax_warm /
// bench_fwd_warm, so the proxqp rows reproduce the b995082 failures on
// the identical tick sequences. Reported per cell (100 warm ticks):
//   us | mean it | mean fac | worst-tick it | rst (ticks with >= 1
//   cold reset) | fails; plus one cold reference line (shipped
//   defaults, warm_start off) for scale.
//
// Expected shape: proxqp fails in the creep regime; cap1 removes the
// fails but keeps the reset firings and ~1.5-1.8x iterations; noreset
// removes the limit cycle; split/jump/eta each cut the remaining creep
// cost. In the control cells the reset never fires, so proxqp / cap1 /
// noreset must be identical there, and the elastic departures must not
// regress.
//
// Usage: bench_bcl_strategies [--csv <dir>]  (writes bcl_strategy_cells.csv)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>

#include "drift_traj.hpp"
#include "elastiqp/elastiqp.hpp"

using drift_traj::Drift;
using drift_traj::Name;
using drift_traj::Size;
using drift_traj::Structure;
using drift_traj::Trajectory;
using std::chrono::duration;
using std::chrono::steady_clock;

namespace {

constexpr int kTicks = 100;

// One BCL strategy generation. reset_limit = max_outer_iter means
// "uncapped" (proxqp has no cap; the reset can fire every outer round).
struct Strategy {
  const char* name;
  bool split, jump, warm_eta;
  int reset_limit;
};

constexpr Strategy kStrategies[] = {
    {"proxqp", false, false, false, 250},
    {"cap1", false, false, false, 1},
    {"noreset", false, false, false, 0},
    {"split", true, false, false, 0},
    {"jump", true, true, false, 0},
    {"eta", true, true, true, 0},
};

void Configure(elastiqp::Settings& s, const Strategy& st, double eps,
               bool ruiz, bool warm) {
  s.eps_abs = eps;
  s.eps_rel = 0;
  s.ruiz = ruiz;
  s.warm_start = warm;
  s.bcl_split = st.split;
  s.bcl_mu_jump = st.jump;
  s.bcl_warm_eta = st.warm_eta;
  s.cold_reset_limit = st.reset_limit;
}

struct ChainResult {
  double us = 0, it = 0, fac = 0;
  int max_it = 0;
  int resets = 0;  // ticks where >= 1 cold reset fired
  int fails = 0;
};

// Run one warm (or cold-reference) chain over the trajectory.
ChainResult RunChain(const Trajectory& traj, const Strategy& st, double eps,
                     bool ruiz, bool warm) {
  elastiqp::Solver s;
  Configure(s.settings, st, eps, ruiz, warm);
  s.setup(traj.Q.empty() ? traj.base.Q : traj.Q[0], traj.q[0], traj.base.A,
          traj.b[0], traj.G.empty() ? traj.base.G : traj.G[0], traj.h[0],
          traj.penalty);

  ChainResult r;
  for (int k = 0; k < kTicks; ++k) {
    s.set_q(traj.q[k]);
    s.set_h(traj.h[k]);
    if (traj.base.b.size() > 0) s.set_b(traj.b[k]);
    if (!traj.G.empty()) s.set_G(traj.G[k]);
    if (!traj.Q.empty()) s.set_Q(traj.Q[k]);

    const auto t0 = steady_clock::now();
    const elastiqp::Solution& sol = s.solve();
    r.us += duration<double, std::micro>(steady_clock::now() - t0).count();
    r.it += sol.iters;
    r.fac += s.factorizations();
    r.max_it = std::max(r.max_it, sol.iters);
    r.resets += s.cold_resets() > 0;
    if (sol.converged != 1) r.fails++;
  }
  r.us /= kTicks;
  r.it /= kTicks;
  r.fac /= kTicks;
  return r;
}

std::FILE* g_csv = nullptr;

void Header() {
  std::printf(
      "  %-6s %3s %3s %4s %-6s %6s %6s %-8s | %8s %6s %5s %6s %4s %s\n",
      "struct", "n", "m", "p", "pen", "sigma", "eps", "strategy", "us", "it",
      "fac", "max it", "rst", "fails");
}

void ReportRow(const char* group, Size sz, Structure st,
               const char* pen_label, double sigma, double eps,
               const char* strategy, const ChainResult& r) {
  std::printf("  %-6s %3d %3d %4d %-6s %6.0e %6.0e %-8s | %8.1f %6.1f %5.1f "
              "%6d %4d %d%s\n",
              Name(st), sz.n, sz.m, sz.p, pen_label, sigma, eps, strategy,
              r.us, r.it, r.fac, r.max_it, r.resets, r.fails,
              r.fails ? " (FAILS)" : "");
  if (g_csv) {
    std::fprintf(g_csv, "%s,%s,%d,%d,%d,%s,%g,%g,%s,%.3f,%.3f,%.3f,%d,%d,%d\n",
                 group, Name(st), sz.n, sz.m, sz.p, pen_label, sigma, eps,
                 strategy, r.us, r.it, r.fac, r.max_it, r.resets, r.fails);
  }
}

void RunCell(const char* group, Size sz, Structure st,
             const Eigen::VectorXd& penalty, const char* pen_label,
             double sigma, double eps) {
  const bool ruiz = penalty.maxCoeff() >= 1e4;
  const unsigned seed = 91u * static_cast<unsigned>(sz.n) +
                        static_cast<unsigned>(sz.p) +
                        7u * static_cast<unsigned>(st);
  const Trajectory traj = drift_traj::MakeTrajectory(sz, st, penalty, sigma,
                                                     Drift::kQH, seed, kTicks);
  for (const Strategy& strat : kStrategies) {
    ReportRow(group, sz, st, pen_label, sigma, eps, strat.name,
              RunChain(traj, strat, eps, ruiz, /*warm=*/true));
  }
  // Cold reference: shipped defaults, no warm start. The pathologies
  // above are warm-only; this line gives the per-tick cost floor.
  ReportRow(group, sz, st, pen_label, sigma, eps, "cold-ref",
            RunChain(traj, kStrategies[5], eps, ruiz, /*warm=*/false));
  std::printf("\n");
}

void RunCell(const char* group, Size sz, Structure st, double penalty_w,
             double sigma, double eps) {
  char label[16];
  std::snprintf(label, sizeof(label), "%g", penalty_w);
  RunCell(group, sz, st, Eigen::VectorXd::Constant(sz.p, penalty_w), label,
          sigma, eps);
}

// Mixed penalty patterns (period 8). The bcl_mu_jump target is read off
// the single worst-residual row, so the mix decides whether a soft row
// can win that argmax while stiff rows creep:
//   alt    every other row soft (50/50)
//   spike  1 stiff row in 8 (stiff minority carries all the creep)
//   dip    1 soft row in 8 (a soft-row argmax winner degrades the jump
//          to the classic ladder -- the suspected worst case)
Eigen::VectorXd MixedPenalty(int p, const char* pattern, double lo,
                             double hi) {
  Eigen::VectorXd w(p);
  for (int i = 0; i < p; ++i) {
    bool stiff = true;
    if (std::string(pattern) == "alt") stiff = i % 2 == 0;
    if (std::string(pattern) == "spike") stiff = i % 8 == 0;
    if (std::string(pattern) == "dip") stiff = i % 8 != 0;
    w[i] = stiff ? hi : lo;
  }
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  std::string csv_dir;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--csv") csv_dir = argv[i + 1];
  }
  if (!csv_dir.empty()) {
    const std::string path = csv_dir + "/bcl_strategy_cells.csv";
    g_csv = std::fopen(path.c_str(), "w");
    if (!g_csv) {
      std::fprintf(stderr, "cannot write %s\n", path.c_str());
      return 1;
    }
    std::fprintf(g_csv, "group,structure,n,m,p,penalty,sigma,eps,strategy,"
                        "us,it,fac,max_it,resets,fails\n");
  }

  std::printf(
      "bench_bcl_strategies: warm forward solve under each BCL strategy\n"
      "generation, on the saturation-creep failure regime (penalty 1e4,\n"
      "small drift) isolated from bench_relax_warm @ b995082. %d warm\n"
      "ticks/cell, qh drift. rst = ticks with >= 1 cold reset;\n"
      "cold-ref = shipped defaults with warm_start off.\n\n",
      kTicks);

  std::printf("=== creep regime: penalty 1e4, ruiz on ===\n");
  Header();
  for (const double eps : {1e-5, 1e-8}) {
    for (const Structure st :
         {Structure::kFeas, Structure::kInfeas, Structure::kDegen}) {
      for (const Size sz : {Size{14, 0, 100}, Size{30, 8, 200}}) {
        for (const double sigma : {1e-4, 1e-3}) {
          RunCell("creep", sz, st, 1e4, sigma, eps);
        }
      }
    }
  }

  std::printf(
      "=== mixed penalties: creep regime, per-row w in {10, 1e4} ===\n"
      "alt = 50/50 | spike = 1 stiff row in 8 | dip = 1 soft row in 8\n"
      "(dip is the suspected jump worst case: a soft row can win the\n"
      "worst-residual argmax while the stiff majority creeps).\n");
  Header();
  for (const char* pattern : {"alt", "spike", "dip"}) {
    for (const Structure st :
         {Structure::kFeas, Structure::kInfeas, Structure::kDegen}) {
      for (const double sigma : {1e-4, 1e-3}) {
        const Size sz{30, 8, 200};
        RunCell("mixed", sz, st, MixedPenalty(sz.p, pattern, 10.0, 1e4),
                pattern, sigma, 1e-5);
      }
    }
  }

  std::printf(
      "=== control: no reset pathology ===\n"
      "penalty 10 (no saturation pressure) and sigma 1e-1 (large drift:\n"
      "the classic ladder is already sufficient). The reset never fires\n"
      "here, so proxqp / cap1 / noreset must be identical; the elastic\n"
      "departures must not regress.\n");
  Header();
  for (const Structure st : {Structure::kFeas, Structure::kInfeas}) {
    RunCell("control", {30, 8, 200}, st, 10.0, 1e-4, 1e-5);
    RunCell("control", {30, 8, 200}, st, 1e4, 1e-1, 1e-5);
  }

  if (g_csv) std::fclose(g_csv);
  return 0;
}
