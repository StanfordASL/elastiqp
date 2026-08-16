// When does the FORWARD pass's warm starting pay off, and where does it
// misbehave?
//
// solve() with settings.warm_start = true (the default) reuses the
// previous solve's (x, y, z) and cached factorization, resetting the AL
// penalties to their defaults; the BCL loop re-tightens mu from a
// near-optimal iterate. This is the production path in control loops.
// This benchmark maps warm vs cold across the same grid as
// bench_relax_warm (problem size, structure, penalty tier, drift size
// and composition, on random drifting trajectories from
// common/drift_traj.hpp), and doubles as a regression watch for
// warm-start-specific pathologies: the rst column counts ticks where
// the BCL cold reset fired (Settings::cold_reset_limit) -- the
// mechanism behind the 2026-08 warm-solve stall, which fired only on
// warm starts (never on 720 random + 144 Maros-Meszaros cold solves).
//
// Per tick both solvers receive the same set_*() updates and solve the
// same data: "warm" is a persistent solver with warm_start = true (the
// chained iterate), "cold" a persistent solver with warm_start = false
// (cold_init every tick, but keeping the setup()-time allocation and
// Ruiz scaling -- pure solve cost, no setup overhead). Reported per
// cell (100 ticks):
//   cold us (it, fac) | warm us (it, fac) | spd = cold/warm wall |
//   dAS (rows whose 3-state activity -- inactive / active / saturated,
//   from z_ineq vs [0, penalty] -- changed between consecutive warm
//   solutions) | rst (ticks where the warm solve fired a cold reset) |
//   worst warm-vs-cold |dx| (same-point check) | fails cold/warm.
//
// Usage: bench_fwd_warm [--csv <dir>]  (writes fwd_warm_cells.csv)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "drift_traj.hpp"
#include "elastiqp/elastiqp.hpp"

using Eigen::VectorXd;
using drift_traj::Drift;
using drift_traj::Name;
using drift_traj::Size;
using drift_traj::Structure;
using drift_traj::Trajectory;
using std::chrono::duration;
using std::chrono::steady_clock;

namespace {

constexpr int kTicks = 100;
constexpr double kEpsMain = 1e-5;  // headline: the shipped default

Trajectory MakeTrajectory(Size sz, Structure st, double penalty_w,
                          double sigma, Drift drift, unsigned seed) {
  return drift_traj::MakeTrajectory(sz, st, penalty_w, sigma, drift, seed,
                                    kTicks);
}

// Three-state activity of a row from the converged bounded multiplier:
// 0 inactive (z ~ 0), 1 active (interior), 2 saturated (z ~ penalty).
// The margin is relative to the penalty; converged multipliers sit
// either exactly on the box (clamped) or well inside it, so the metric
// is insensitive to the margin over several orders of magnitude.
int RowState(double z, double w) {
  const double margin = 1e-6 * w;
  if (z <= margin) return 0;
  if (z >= w - margin) return 2;
  return 1;
}

int CountActivityChanges(const elastiqp::Solution& a,
                         const elastiqp::Solution& b, const VectorXd& w) {
  int changes = 0;
  for (Eigen::Index i = 0; i < w.size(); ++i) {
    if (RowState(a.z_ineq[i], w[i]) != RowState(b.z_ineq[i], w[i])) {
      changes++;
    }
  }
  return changes;
}

struct CellResult {
  double cold_us = 0, cold_it = 0, cold_fac = 0;
  double warm_us = 0, warm_it = 0, warm_fac = 0;
  double das = 0;  // mean activity changes per tick (warm chain)
  double dx = 0;   // worst warm-vs-cold |x| diff (relative)
  int cold_fails = 0, warm_fails = 0;
  int warm_resets = 0;  // ticks where the warm solve fired a cold reset
  int cold_resets = 0;  // same for the cold solve (expected 0)
};

void ApplyTick(elastiqp::Solver& s, const Trajectory& traj, int k) {
  s.set_q(traj.q[k]);
  s.set_h(traj.h[k]);
  if (traj.base.b.size() > 0) s.set_b(traj.b[k]);
  if (!traj.G.empty()) s.set_G(traj.G[k]);
  if (!traj.Q.empty()) s.set_Q(traj.Q[k]);
}

CellResult RunCell(const Trajectory& traj, double eps, bool ruiz) {
  elastiqp::Solver warm, cold;
  for (elastiqp::Solver* s : {&warm, &cold}) {
    s->settings.eps_abs = eps;
    s->settings.eps_rel = 0;
    s->settings.ruiz = ruiz;
    s->setup(traj.Q.empty() ? traj.base.Q : traj.Q[0], traj.q[0],
             traj.base.A, traj.b[0],
             traj.G.empty() ? traj.base.G : traj.G[0], traj.h[0],
             traj.penalty);
  }
  warm.settings.warm_start = true;
  cold.settings.warm_start = false;

  CellResult r;
  elastiqp::Solution prev;
  bool have_prev = false;
  for (int k = 0; k < kTicks; ++k) {
    ApplyTick(warm, traj, k);
    ApplyTick(cold, traj, k);

    auto t0 = steady_clock::now();
    const elastiqp::Solution w = warm.solve();
    r.warm_us +=
        duration<double, std::micro>(steady_clock::now() - t0).count();
    r.warm_it += w.iters;
    r.warm_fac += warm.factorizations();
    r.warm_resets += warm.cold_resets() > 0;
    if (w.converged != 1) r.warm_fails++;

    t0 = steady_clock::now();
    const elastiqp::Solution& c = cold.solve();
    r.cold_us +=
        duration<double, std::micro>(steady_clock::now() - t0).count();
    r.cold_it += c.iters;
    r.cold_fac += cold.factorizations();
    r.cold_resets += cold.cold_resets() > 0;
    if (c.converged != 1) r.cold_fails++;

    const double dxk = (w.x - c.x).lpNorm<Eigen::Infinity>();
    r.dx = std::max(r.dx, dxk / (1.0 + c.x.lpNorm<Eigen::Infinity>()));
    if (have_prev) r.das += CountActivityChanges(prev, w, traj.penalty);
    prev = w;
    have_prev = true;
  }
  const double nt = kTicks;
  r.cold_us /= nt;
  r.cold_it /= nt;
  r.cold_fac /= nt;
  r.warm_us /= nt;
  r.warm_it /= nt;
  r.warm_fac /= nt;
  r.das /= (nt - 1);
  return r;
}

std::FILE* g_csv = nullptr;

void Report(const char* group, Size sz, Structure st, double penalty_w,
            Drift drift, double sigma, double eps, const CellResult& r) {
  const bool fails = r.cold_fails + r.warm_fails > 0;
  std::printf(
      "  %-6s %3d %3d %4d %5s %6.0e %6.0e | %8.1f (%5.1f,%4.1f) | %8.1f "
      "(%5.1f,%4.1f) | %5.2fx | %6.2f | %3d | %7.1e | %d/%d%s\n",
      Name(st), sz.n, sz.m, sz.p, Name(drift), sigma, eps, r.cold_us,
      r.cold_it, r.cold_fac, r.warm_us, r.warm_it, r.warm_fac,
      r.cold_us / r.warm_us, r.das, r.warm_resets, r.dx, r.cold_fails,
      r.warm_fails, fails ? " (FAILS)" : "");
  if (g_csv) {
    std::fprintf(
        g_csv,
        "%s,%s,%d,%d,%d,%g,%s,%g,%g,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
        "%d,%d,%.3e,%d,%d\n",
        group, Name(st), sz.n, sz.m, sz.p, penalty_w, Name(drift), sigma,
        eps, r.cold_us, r.cold_it, r.cold_fac, r.warm_us, r.warm_it,
        r.warm_fac, r.das, r.warm_resets, r.cold_resets, r.dx, r.cold_fails,
        r.warm_fails);
  }
}

void Header() {
  std::printf(
      "  %-6s %3s %3s %4s %5s %6s %6s | %8s (%5s,%4s) | %8s (%5s,%4s) | "
      "%6s | %6s | %3s | %7s | %s\n",
      "struct", "n", "m", "p", "drift", "sigma", "eps", "cold", "it", "fac",
      "warm", "it", "fac", "spd", "dAS", "rst", "|dx|", "fails c/w");
}

}  // namespace

int main(int argc, char** argv) {
  std::string csv_dir;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--csv") csv_dir = argv[i + 1];
  }
  if (!csv_dir.empty()) {
    const std::string path = csv_dir + "/fwd_warm_cells.csv";
    g_csv = std::fopen(path.c_str(), "w");
    if (!g_csv) {
      std::fprintf(stderr, "cannot write %s\n", path.c_str());
      return 1;
    }
    std::fprintf(g_csv,
                 "group,structure,n,m,p,penalty,drift,sigma,eps,"
                 "cold_us,cold_it,cold_fac,warm_us,warm_it,warm_fac,"
                 "das,warm_resets,cold_resets,dx,fails_cold,fails_warm\n");
  }

  const Size kSizes[] = {{14, 0, 100}, {30, 8, 200}, {58, 15, 400}};
  const double kSigmas[] = {1e-4, 1e-3, 1e-2, 1e-1};

  std::printf(
      "bench_fwd_warm: forward solve() warm start (chained iterate +\n"
      "cached factorization) vs cold start on drifting random QPs.\n"
      "%d ticks/cell, headline eps=%g. dAS = warm-chain activity changes\n"
      "per tick (inactive/active/saturated from z_ineq); rst = ticks\n"
      "where the warm solve fired a BCL cold reset; spd = cold/warm wall\n"
      "time.\n\n",
      kTicks, kEpsMain);

  for (const double penalty_w : {10.0, 1e4}) {
    const bool ruiz = penalty_w >= 1e4;
    std::printf("=== penalty %g (ruiz %s), drift on q+h(+b) ===\n",
                penalty_w, ruiz ? "on" : "off");
    Header();
    for (const Structure st :
         {Structure::kFeas, Structure::kInfeas, Structure::kDegen}) {
      for (const Size sz : kSizes) {
        for (const double sigma : kSigmas) {
          const unsigned seed =
              91u * static_cast<unsigned>(sz.n) +
              static_cast<unsigned>(sz.p) + 7u * static_cast<unsigned>(st);
          const Trajectory traj = MakeTrajectory(sz, st, penalty_w, sigma,
                                                 Drift::kQH, seed);
          Report("main", sz, st, penalty_w, Drift::kQH, sigma, kEpsMain,
                 RunCell(traj, kEpsMain, ruiz));
        }
      }
      std::printf("\n");
    }
  }

  std::printf(
      "=== drift composition (n=30, m=8, p=200, infeas, penalty 1e4, ruiz) "
      "===\n"
      "q: linear cost only | qQ: +quadratic cost (Hessian drifts, geometry\n"
      "fixed) | qh: +bounds | qhG: +constraint matrix | all: everything\n");
  Header();
  for (const Drift drift :
       {Drift::kQ, Drift::kQQ, Drift::kQH, Drift::kQHG, Drift::kAll}) {
    for (const double sigma : kSigmas) {
      const Trajectory traj = MakeTrajectory({30, 8, 200}, Structure::kInfeas,
                                             1e4, sigma, drift, 4242);
      Report("drift", {30, 8, 200}, Structure::kInfeas, 1e4, drift, sigma,
             kEpsMain, RunCell(traj, kEpsMain, true));
    }
    std::printf("\n");
  }

  std::printf(
      "=== accuracy tier (n=30, m=8, p=200, qh drift) ===\n"
      "eps 1e-5 (default) vs 1e-8: the warm iterate is worth more when\n"
      "the target is tighter -- unless the endgame stalls.\n");
  Header();
  for (const double penalty_w : {10.0, 1e4}) {
    for (const double eps : {1e-5, 1e-8}) {
      for (const double sigma : {1e-4, 1e-2}) {
        const Trajectory traj = MakeTrajectory(
            {30, 8, 200}, Structure::kInfeas, penalty_w, sigma, Drift::kQH,
            4242);
        Report("eps", {30, 8, 200}, Structure::kInfeas, penalty_w,
               Drift::kQH, sigma, eps, RunCell(traj, eps, penalty_w >= 1e4));
      }
    }
    std::printf("\n");
  }

  if (g_csv) std::fclose(g_csv);
  return 0;
}
