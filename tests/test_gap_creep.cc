// PDAL regression test for the warm-start deactivation-creep failure mode.
//
// Scenario captured from examples/constraint_conflict_demo.py
// ("Equal penalties", ticks 463/464 at dt=0.01): a 2D double-integrator
// CBF controller squeezed between a wall and a moving obstacle. During
// the pinch the two conflicting rows ride the elastic penalty cap
// (z = w = 1000); one tick later the pinch releases and the true duals
// drop to ~77 with the warm x strictly feasible for the new data.
//
// The mechanism (mirror of the saturation creep in test_bcl_creep): a
// primal-feasible warm start has pri = 0 identically, so every BCL round
// is a "good" step and mu_in stays frozen at mu_in_init. The residual
// clauses converge immediately; the only remaining error is
// complementarity (rows strictly inside the feasible set holding large
// duals), which only the duality-gap clause sees. The excess dual
// contracts at the PPM rate ~ mu/(mu + lambda) per round -- arbitrarily
// slow here because the two active rows are nearly anti-parallel
// (lambda ~ 3e-3) -- so the pre-fix solver burned all 250 outer rounds
// with pri = 0, dua ~ 1e-9, gap stuck at 0.28: kMaxIter.
//
// The fix under test is Settings::bcl_release_jump: on a good step where the
// residual clauses pass, the gap clause fails, and the gap's decay rate
// projects to more than bcl_release_jump_horizon further rounds, jump mu one
// mu_update_factor past the shallowest deactivation point
// max_i(-r_i / z_i). Measured post-fix: 24 iterations
// (bounds below hold ~2x headroom). With the flag off the warm solve
// still fails -- reported informationally to confirm the repro
// discriminates.

#include <cmath>
#include <cstdio>

#include "elastiqp/elastiqp.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

bool g_all_ok = true;

void Check(const char* name, bool ok, double val, const char* what) {
  std::printf("  %-44s %s=%9.2e %s\n", name, what, val, ok ? "OK" : "FAIL");
  g_all_ok &= ok;
}

}  // namespace

int main() {
  std::printf("Deactivation-creep regression: pinch-release warm tick\n");

  const MatrixXd Q = MatrixXd::Identity(2, 2);
  const VectorXd penalty = VectorXd::Constant(3, 1000.0);

  VectorXd q463(2), h463(3), q464(2), h464(3);
  MatrixXd G463(3, 2), G464(3, 2);
  q463 << 0.21920301334862091, 4.1139326748986216;
  G463 << 1, 0, 0, 1, 0.048596449461558944, -0.99361503907741344;
  h463 << 2.4088071318512285, -5.7350402445204285, 2.1510357789205634;
  q464 << -1.2682556160930214, 3.9605375674075187;
  G464 << 1, 0, 0, 1, 0.076622662009877995, -0.98842746321542529;
  h464 << 5.3894716545331169, -5.4329140545125938, 5.017615727279078;

  // Ground truth for the release tick (a cold solve never stalls here).
  const elastiqp::Solution ref = elastiqp::pdal::Solve(Q, q464, G464, h464, penalty);
  Check("cold reference converges", ref.converged == 1, ref.iters, "iters");

  // The warm chain at shipped defaults.
  elastiqp::pdal::Solver s;
  s.setup(Q, q463, G463, h463, penalty);
  const elastiqp::Solution pinch = s.solve();
  Check("pinch tick converges", pinch.converged == 1, pinch.iters, "iters");
  // The scenario premise: conflicting rows at/near the penalty cap.
  Check("pinch duals ride the cap", pinch.z.maxCoeff() > 0.99e3,
        pinch.z.maxCoeff(), "zmax");

  s.set_q(q464);
  s.set_G(G464);
  s.set_h(h464);
  const elastiqp::Solution rel = s.solve();
  Check("release tick converges warm", rel.converged == 1, rel.iters,
        "iters");
  Check("release tick iters bounded", rel.iters <= 50, rel.iters, "iters");
  const double xerr = (rel.x - ref.x).lpNorm<Eigen::Infinity>();
  Check("release x matches cold", xerr <= 1e-3, xerr, "err");
  const double zerr = (rel.z - ref.z).lpNorm<Eigen::Infinity>();
  Check("release duals leave the cap", zerr <= 1.0, zerr, "err");

  // Informational: the repro still discriminates (not gated -- a future
  // deeper fix may legitimately clear it with the jump off).
  elastiqp::pdal::Solver off;
  off.settings.bcl_release_jump = false;
  off.setup(Q, q463, G463, h463, penalty);
  off.solve();
  off.set_q(q464);
  off.set_G(G464);
  off.set_h(h464);
  const elastiqp::Solution noj = off.solve();
  std::printf("  [info] bcl_release_jump=false: status=%d iters=%d gap=%.2e\n",
              static_cast<int>(noj.status), noj.iters, noj.duality_gap);

  return g_all_ok ? 0 : 1;
}
