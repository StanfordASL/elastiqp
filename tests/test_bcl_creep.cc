// PDAL regression test for the warm-start saturation-creep failure mode and
// the mixed-penalty jump regression.
//
// Uniform cells: penalty 1e4, sigma 1e-4 qh drift, n=14 p=100 (feas and
// degen) -- the worst cells of bench_bcl_strategies, on the exact
// trajectories (same generator, same seed formula) where the
// proxqp-parity BCL failed 18-19 of 100 warm ticks at b995082. The
// mechanism: drift pushes weakly-active rows toward elastic saturation,
// the multipliers creep there at ~r/mu per outer round, and proxqp's
// cold reset reads the creep as "no progress" and re-widens mu -- a
// limit cycle to kMaxIter.
//
// Mixed cell: 1 stiff row (1e4) in 8 soft rows (10), infeas n=30 m=8
// p=200, sigma 1e-3 -- the regime where the original worst-residual-row
// mu jump over-tightened (a stiff row far from saturation won the
// argmax while the soft majority was still settling: measured 36.9 mean
// / 105 worst-tick iterations vs 23.3 / 54 without the jump). The
// shipped shallowest-first jump measures 25.1 mean / 54 worst tick.
//
// All bounds hold ~1.3-2x headroom over the shipped measurements and
// each discriminates a rejected generation: proxqp-parity fails ticks
// outright; the capped reset fires resets and breaks the uniform
// iteration bounds; the worst-residual-row jump breaks both mixed-cell
// bounds.

#include <algorithm>
#include <cstdio>

#include "drift_traj.hpp"
#include "elastiqp/elastiqp.hpp"

using drift_traj::Drift;
using drift_traj::Size;
using drift_traj::Structure;
using drift_traj::Trajectory;
using Eigen::VectorXd;

namespace {

constexpr int kTicks = 100;

bool g_all_ok = true;

void Check(const char* name, bool ok, double val, const char* what) {
  std::printf("  %-38s %s=%9.2e %s\n", name, what, val, ok ? "OK" : "FAIL");
  g_all_ok &= ok;
}

void RunCell(Structure st, Size sz, const VectorXd& penalty, double sigma,
             const char* label, int max_total_iters, int max_tick_iters) {
  const unsigned seed = 91u * static_cast<unsigned>(sz.n) +
                        static_cast<unsigned>(sz.p) +
                        7u * static_cast<unsigned>(st);
  const Trajectory traj = drift_traj::MakeTrajectory(sz, st, penalty, sigma,
                                                     Drift::kQH, seed, kTicks);

  elastiqp::pdal::Solver s;  // shipped defaults; only eps/ruiz pinned
  s.settings.eps_abs = 1e-5;
  s.settings.eps_rel = 0;
  s.settings.ruiz = true;
  s.setup(traj.base.Q, traj.q[0], traj.base.A, traj.b[0], traj.base.G,
          traj.h[0], traj.penalty);

  int fails = 0, reset_ticks = 0, total_iters = 0, worst_tick = 0;
  for (int k = 0; k < kTicks; ++k) {
    s.set_q(traj.q[k]);
    s.set_h(traj.h[k]);
    if (sz.m > 0) s.set_b(traj.b[k]);
    const elastiqp::Solution& sol = s.solve();
    if (sol.converged != 1) fails++;
    if (s.cold_resets() > 0) reset_ticks++;
    total_iters += sol.iters;
    worst_tick = std::max(worst_tick, sol.iters);
  }

  char name[64];
  std::snprintf(name, sizeof(name), "%s: all ticks converge", label);
  Check(name, fails == 0, fails, "fails");
  std::snprintf(name, sizeof(name), "%s: no cold reset fires", label);
  Check(name, reset_ticks == 0, reset_ticks, "ticks");
  std::snprintf(name, sizeof(name), "%s: total iters bounded", label);
  Check(name, total_iters <= max_total_iters, total_iters, "iters");
  std::snprintf(name, sizeof(name), "%s: worst tick bounded", label);
  Check(name, worst_tick <= max_tick_iters, worst_tick, "iters");
}

}  // namespace

int main() {
  std::printf("BCL creep regression: warm chain, qh drift\n");
  RunCell(Structure::kFeas, {14, 0, 100}, VectorXd::Constant(100, 1e4), 1e-4,
          "feas uniform 1e4", 4000, 120);
  RunCell(Structure::kDegen, {14, 0, 100}, VectorXd::Constant(100, 1e4), 1e-4,
          "degen uniform 1e4", 5000, 120);

  // Mixed penalties: 1 stiff row in 8 (see the header comment).
  VectorXd spike(200);
  for (int i = 0; i < 200; ++i) spike[i] = (i % 8 == 0) ? 1e4 : 10.0;
  RunCell(Structure::kInfeas, {30, 8, 200}, spike, 1e-3, "infeas spike 10/1e4",
          3200, 80);
  return g_all_ok ? 0 : 1;
}
