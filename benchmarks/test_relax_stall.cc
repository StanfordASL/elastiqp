// Regression test for the relax() small-kappa stall (fixed by the
// Levenberg-Marquardt stall handling in relax()).
//
// The humanoid WBC sequence has penalty tiers up to 1e5. An inactive row
// with penalty w relaxes to the pair (z, s) = (w, kappa/w), and the relax()
// Newton elimination amplifies roundoff by z/s = w^2/kappa -- 1e12..1e14 at
// kappa <= 1e-4 -- so below the resulting noise floor the Newton direction
// is garbage. Before the fix, the line search accepted the failed step
// after exhausting its halvings and the iterate drifted: replaying hum-wbc
// WITHOUT Ruiz equilibration at tol 1e-8 failed 22/250 ticks at
// kappa = 1e-4 and 14/250 at kappa = 1e-6. With the fix (revert the
// iterate, escalate the relax_reg damping, refactor), every tick converges.
//
// This deliberately runs WITHOUT Ruiz: equilibration also removes the
// amplification (and is what bench_diff_robot uses), but relax() must not
// depend on it for robustness.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "elastiqp/elastiqp.hpp"
#include "qp_io.hpp"

using robot_control::NamedSequence;
using robot_control::RobotQP;

namespace {

bool g_all_ok = true;

void Check(const char* name, bool ok, double val, const char* what) {
  std::printf("  %-44s %s=%9.2e %s\n", name, what, val, ok ? "OK" : "FAIL");
  g_all_ok &= ok;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : ELASTIQP_SEQUENCE_FILE;
  std::vector<NamedSequence> seqs;
  try {
    seqs = robot_control::LoadSequences(path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }

  constexpr double kTol = 1e-8;
  constexpr double kKappas[] = {1e-3, 1e-4, 1e-6};

  for (const NamedSequence& seq : seqs) {
    std::printf("%s: warm solve + relax(kappa), no Ruiz, tol %g\n",
                seq.name.c_str(), kTol);
    for (const double kappa : kKappas) {
      elastiqp::Solver solver;
      solver.settings.eps_abs = 1e-8;
      solver.settings.eps_rel = 0;
      solver.settings.warm_start = true;

      int fails = 0;
      long iters = 0;
      double worst_comp = 0.0;  // |s.z - kappa| over converged ticks
      bool first = true;
      for (const RobotQP& qp : seq.qps) {
        if (first) {
          solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, qp.penalty);
          first = false;
        } else {
          solver.set_Q(qp.Q);
          solver.set_q(qp.q);
          if (qp.b.size() > 0) {
            solver.set_A(qp.A);
            solver.set_b(qp.b);
          }
          solver.set_G(qp.G);
          solver.set_h(qp.h);
          solver.set_penalty(qp.penalty);
        }
        solver.solve();
        const elastiqp::Solution& rel = solver.relax(kappa, kTol, 50);
        iters += rel.iters;
        if (rel.converged != 1) {
          fails++;
          continue;
        }
        for (Eigen::Index i = 0; i < qp.h.size(); ++i) {
          worst_comp = std::max(
              worst_comp, std::abs(rel.s_t[i] * rel.z_t[i] - kappa));
          worst_comp = std::max(
              worst_comp, std::abs(rel.s_ineq[i] * rel.z_ineq[i] - kappa));
        }
      }
      char name[64];
      std::snprintf(name, sizeof(name), "kappa=%.0e (%.1f iters/tick)",
                    kappa,
                    static_cast<double>(iters) /
                        static_cast<double>(seq.qps.size()));
      Check(name, fails == 0 && worst_comp < 1e-8,
            fails > 0 ? static_cast<double>(fails) : worst_comp,
            fails > 0 ? "fails" : "comp");
    }
  }

  if (!g_all_ok) {
    std::printf("\nFAILURES\n");
    return 1;
  }
  std::printf("\nAll relax stall checks passed\n");
  return 0;
}
