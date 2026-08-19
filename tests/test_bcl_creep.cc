// Regression test for the warm-start saturation-creep failure mode.
//
// The cell: penalty 1e4, sigma 1e-4 qh drift, n=14 p=100 (feas and
// degen structures) -- the worst cells of bench_bcl_strategies, on the
// exact trajectories (same generator, same seed formula) where the
// proxqp-parity BCL failed 18-19 of 100 warm ticks at b995082. The
// mechanism: drift pushes weakly-active rows toward elastic
// saturation, the multipliers creep there at ~r/mu per outer round,
// and proxqp's cold reset reads the creep as "no progress" and
// re-widens mu -- a limit cycle to kMaxIter.
//
// With the shipped elastic BCL (bcl_split + bcl_mu_jump + bcl_warm_eta,
// cold_reset_limit = 0) the measured cost is ~19-25 mean iterations per
// tick, worst tick <= 59. The bounds below hold ~2x headroom and still
// discriminate every rejected generation: proxqp-parity fails ticks
// outright (and totals ~8500-10200 iterations); the capped-reset
// variant (cold_reset_limit = 1) passes all ticks but fires resets and
// totals ~6000-6800 iterations with worst ticks ~150.

#include <algorithm>
#include <cstdio>

#include "drift_traj.hpp"
#include "elastiqp/elastiqp.hpp"

using drift_traj::Drift;
using drift_traj::Size;
using drift_traj::Structure;
using drift_traj::Trajectory;

namespace {

bool g_all_ok = true;

void Check(const char* name, bool ok, double val, const char* what) {
  std::printf("  %-38s %s=%9.2e %s\n", name, what, val, ok ? "OK" : "FAIL");
  g_all_ok &= ok;
}

void RunCell(Structure st, const char* label, int max_total_iters) {
  constexpr int kTicks = 100;
  const Size sz{14, 0, 100};
  const unsigned seed = 91u * static_cast<unsigned>(sz.n) +
                        static_cast<unsigned>(sz.p) +
                        7u * static_cast<unsigned>(st);
  const Trajectory traj = drift_traj::MakeTrajectory(
      sz, st, /*penalty_w=*/1e4, /*sigma=*/1e-4, Drift::kQH, seed, kTicks);

  elastiqp::Solver s;  // shipped defaults; only eps/ruiz pinned
  s.settings.eps_abs = 1e-5;
  s.settings.eps_rel = 0;
  s.settings.ruiz = true;
  s.setup(traj.base.Q, traj.q[0], traj.base.A, traj.b[0], traj.base.G,
          traj.h[0], traj.penalty);

  int fails = 0, reset_ticks = 0, total_iters = 0, max_tick_iters = 0;
  for (int k = 0; k < kTicks; ++k) {
    s.set_q(traj.q[k]);
    s.set_h(traj.h[k]);
    const elastiqp::Solution& sol = s.solve();
    if (sol.converged != 1) fails++;
    if (s.cold_resets() > 0) reset_ticks++;
    total_iters += sol.iters;
    max_tick_iters = std::max(max_tick_iters, sol.iters);
  }

  char name[64];
  std::snprintf(name, sizeof(name), "%s: all ticks converge", label);
  Check(name, fails == 0, fails, "fails");
  std::snprintf(name, sizeof(name), "%s: no cold reset fires", label);
  Check(name, reset_ticks == 0, reset_ticks, "ticks");
  std::snprintf(name, sizeof(name), "%s: total iters bounded", label);
  Check(name, total_iters <= max_total_iters, total_iters, "iters");
  std::snprintf(name, sizeof(name), "%s: worst tick bounded", label);
  Check(name, max_tick_iters <= 120, max_tick_iters, "iters");
}

}  // namespace

int main() {
  std::printf("BCL creep regression: warm chain, penalty 1e4, sigma 1e-4\n");
  RunCell(Structure::kFeas, "feas n=14 p=100", 4000);
  RunCell(Structure::kDegen, "degen n=14 p=100", 5000);
  return g_all_ok ? 0 : 1;
}
