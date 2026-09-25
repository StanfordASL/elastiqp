// PDAL-specific behaviour of elastiqp::pdal::Solver (the elastic QP itself
// is covered for every backend in test_solvers.cc; the BCL regression
// watches are test_bcl_creep.cc and test_gap_creep.cc):
//
// (1) the factorization cache and rank-one active-set updates: warm
//     starts must reuse the cached K, matrix updates must invalidate it,
//     and incremental updates must reach the same point as full
//     refactorizations with fewer of them;
// (2) the explicit set_warm_start() hook and its Ruiz roundtrip;
// (3) Ruiz re-equilibration: the exact remap of the solve() and relax()
//     warm iterates into the refreshed frame;
// (4) relax(kappa): agreement with the IPM's kappa-relaxed central point,
//     the tight iterate left untouched, the warm relax chain across data
//     updates, and the KKT VJP against finite differences of the relaxed
//     solution map.

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

#include "elastiqp/elastiqp.hpp"
#include "problem_gen.hpp"
#include "test_util.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;
using test_util::Backend;
using test_util::Check;
using test_util::InfNorm;
using test_util::SolveWith;
namespace ipm = elastiqp::ipm;
namespace pdal = elastiqp::pdal;

// The termination / warm-start / Ruiz defaults shared by the PDAL and IPM
// settings must agree (the IPM is the oracle at matching tolerances).
static_assert(
    pdal::Settings{}.eps_abs == ipm::Settings{}.eps_abs &&
        pdal::Settings{}.eps_rel == ipm::Settings{}.eps_rel &&
        pdal::Settings{}.check_duality_gap ==
            ipm::Settings{}.check_duality_gap &&
        pdal::Settings{}.eps_duality_gap_abs ==
            ipm::Settings{}.eps_duality_gap_abs &&
        pdal::Settings{}.eps_duality_gap_rel ==
            ipm::Settings{}.eps_duality_gap_rel &&
        pdal::Settings{}.max_factor_retries ==
            ipm::Settings{}.max_factor_retries &&
        pdal::Settings{}.check_eq_consistency ==
            ipm::Settings{}.check_eq_consistency &&
        pdal::Settings{}.ruiz == ipm::Settings{}.ruiz &&
        pdal::Settings{}.ruiz_max_iter == ipm::Settings{}.ruiz_max_iter &&
        pdal::Settings{}.ruiz_tol == ipm::Settings{}.ruiz_tol,
    "the termination/ruiz defaults shared by pdal::Settings and "
    "ipm::Settings must agree");

namespace {

pdal::Settings TightSettings() { return Backend<pdal::Solver>::Tight(); }
ipm::Settings TightIpmSettings() { return Backend<ipm::Solver>::Tight(); }

}  // namespace

int main() {
  std::mt19937 rng(42);

  std::printf("PDAL: incremental active-set updates vs full refactorization\n");
  {
    // Rank-one LLT flip updates (settings.incremental_updates) must reach
    // the same point as refactoring on every active-set change, with
    // fewer factorizations, cold and along a warm chain.
    const int n = 30, m = 8, p = 200, ticks = 10;
    const QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    pdal::Solver inc, full;
    inc.settings = TightSettings();
    full.settings = TightSettings();
    full.settings.incremental_updates = false;
    inc.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, penalty);
    full.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, penalty);
    std::normal_distribution<double> dist;
    VectorXd q = qp0.q, h = qp0.h;
    int inc_factors = 0, full_factors = 0;
    double worst_dx = 0;
    bool all_conv = true;
    for (int k = 0; k <= ticks; ++k) {
      if (k > 0) {
        for (int i = 0; i < n; ++i) q[i] += 0.01 * dist(rng);
        for (int i = 0; i < p; ++i) h[i] += 0.01 * dist(rng);
        inc.set_q(q);
        inc.set_h(h);
        full.set_q(q);
        full.set_h(h);
      }
      const auto& si = inc.solve();
      const auto& sf = full.solve();
      inc_factors += inc.factorizations();
      full_factors += full.factorizations();
      all_conv &= si.converged == 1 && sf.converged == 1;
      worst_dx = std::max(worst_dx, InfNorm(si.x - sf.x));
    }
    std::printf("  factorizations: incremental=%d full=%d\n", inc_factors,
                full_factors);
    Check("same point, fewer factorizations",
          all_conv && worst_dx < 1e-6 && inc_factors < full_factors, worst_dx,
          "|dx|");
  }

  std::printf("PDAL: warm start with equalities (drift q, h, b)\n");
  {
    const int n = 30, m = 8, p = 200, ticks = 20;
    const QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);

    pdal::Solver warm;
    warm.settings = TightSettings();
    warm.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, penalty);

    std::normal_distribution<double> dist;
    VectorXd q = qp0.q, h = qp0.h, b = qp0.b;
    int warm_iters = 0, cold_iters = 0;
    int warm_factors = 0, cold_factors = 0;
    double worst_dx = 0, worst_eq = 0, worst_kkt = 0;
    bool all_conv = true;
    for (int k = 0; k < ticks; ++k) {
      for (int i = 0; i < n; ++i) q[i] += 0.01 * dist(rng);
      for (int i = 0; i < p; ++i) h[i] += 0.01 * dist(rng);
      for (int i = 0; i < m; ++i) b[i] += 0.01 * dist(rng);
      warm.set_q(q);
      warm.set_h(h);
      warm.set_b(b);
      const auto& ws = warm.solve();
      warm_factors += warm.factorizations();
      pdal::Solver cold;
      cold.settings = TightSettings();
      cold.setup(qp0.Q, q, qp0.A, b, qp0.G, h, penalty);
      const auto& cs = cold.solve();
      cold_factors += cold.factorizations();
      all_conv &= ws.converged == 1 && cs.converged == 1;
      warm_iters += ws.iters;
      cold_iters += cs.iters;
      worst_dx = std::max(worst_dx, (ws.x - cs.x).lpNorm<Eigen::Infinity>());
      worst_eq =
          std::max(worst_eq, (qp0.A * ws.x - b).lpNorm<Eigen::Infinity>());
      worst_kkt = std::max(worst_kkt, problem_gen::ElasticKKTResidual(
                                          qp0.Q, q, qp0.A, b, qp0.G, h, penalty,
                                          ws.x, ws.t, ws.y, ws.z_t, ws.z));
    }
    std::printf(
        "  cold iters=%d warm iters=%d cold factors=%d warm factors=%d\n"
        "  worst_eq=%9.2e kkt=%9.2e\n",
        cold_iters, warm_iters, cold_factors, warm_factors, worst_eq,
        worst_kkt);
    Check("n=30 m=8 p=200 20 ticks",
          all_conv && worst_dx < 1e-4 && worst_eq < 1e-6 && worst_kkt < 1e-6 &&
              warm_iters < cold_iters && warm_factors < cold_factors,
          worst_dx, "|dx|");
  }

  std::printf("PDAL: explicit warm start via set_warm_start\n");
  {
    const int n = 30, m = 8, p = 200, ticks = 20;
    const QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);

    pdal::Solver solver;
    solver.settings = TightSettings();
    solver.settings.warm_start = false;
    solver.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, penalty);

    std::normal_distribution<double> dist;
    VectorXd q = qp0.q, h = qp0.h, b = qp0.b;
    int explicit_iters = 0, cold_iters = 0;
    double worst_dx = 0, worst_kkt = 0;
    bool all_conv = true;
    for (int k = 0; k < ticks; ++k) {
      for (int i = 0; i < n; ++i) q[i] += 0.01 * dist(rng);
      for (int i = 0; i < p; ++i) h[i] += 0.01 * dist(rng);
      for (int i = 0; i < m; ++i) b[i] += 0.01 * dist(rng);
      solver.set_q(q);
      solver.set_h(h);
      solver.set_b(b);
      if (k > 0) {
        const auto& prev = solver.solution();
        solver.set_warm_start(prev.x, prev.y, prev.z);
      }
      const auto& ws = solver.solve();
      const auto cs =
          SolveWith<pdal::Solver>(qp0.Q, q, qp0.A, b, qp0.G, h, penalty);
      all_conv &= ws.converged == 1 && cs.converged == 1;
      explicit_iters += ws.iters;
      cold_iters += cs.iters;
      worst_dx = std::max(worst_dx, (ws.x - cs.x).lpNorm<Eigen::Infinity>());
      worst_kkt = std::max(worst_kkt, problem_gen::ElasticKKTResidual(
                                          qp0.Q, q, qp0.A, b, qp0.G, h, penalty,
                                          ws.x, ws.t, ws.y, ws.z_t, ws.z));
    }
    std::printf("  cold iters=%d explicit iters=%d worst_kkt=%9.2e\n",
                cold_iters, explicit_iters, worst_kkt);
    Check("n=30 m=8 p=200 20 ticks (explicit)",
          all_conv && worst_dx < 1e-4 && worst_kkt < 1e-6 &&
              explicit_iters < cold_iters,
          worst_dx, "|dx|");
  }

  std::printf("PDAL: Ruiz re-equilibration (drift G, A row scales)\n");
  {
    // Matrix updates keep the setup()-time scaling exact but let it drift;
    // solve() must re-equilibrate past settings.ruiz_refresh_ratio and
    // carry the warm-start iterate (solve + relax) into the new frame.
    const int n = 30, m = 8, p = 200, ticks = 24;
    const double kappa = 1e-4;
    QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    VectorXd penalty = VectorXd::Constant(p, 10.0);
    pdal::Solver warm;
    warm.settings = TightSettings();
    warm.settings.ruiz = true;
    warm.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, penalty);

    std::normal_distribution<double> dist;
    MatrixXd G = qp0.G, A = qp0.A;
    VectorXd q = qp0.q, h = qp0.h, b = qp0.b;
    // Per-tick row scale growth: 1.5x on a fifth of the rows (G and A) so the
    // drift crosses the 4x refresh ratio every few ticks
    int refreshes = 0, warm_iters = 0, cold_iters = 0;
    int relax_warm_iters = 0, relax_cold_iters = 0;
    double worst_kkt = 0, worst_dx = 0, worst_rdx = 0, worst_drift = 0;
    bool all_conv = true;
    VectorXd last_relax_x;
    for (int k = 0; k < ticks; ++k) {
      for (int i = 0; i < p; i += 5) {
        G.row(i) *= 1.5;
        h[i] *= 1.5;
        penalty[i] /= 1.5;
      }
      for (int i = 0; i < m; i += 4) {
        A.row(i) *= 1.5;
        b[i] *= 1.5;
      }
      for (int i = 0; i < n; ++i) q[i] += 0.01 * dist(rng);
      warm.set_G(G);
      warm.set_h(h);
      warm.set_penalty(penalty);
      warm.set_A(A);
      warm.set_b(b);
      warm.set_q(q);
      const double drift = warm.scaling_drift();
      refreshes += drift > warm.settings.ruiz_refresh_ratio;
      const auto ws = warm.solve();
      worst_drift = std::max(worst_drift, warm.scaling_drift());
      const auto wr = warm.relax(kappa);
      pdal::Solver cold;
      cold.settings = TightSettings();
      cold.settings.ruiz = true;
      cold.setup(qp0.Q, q, A, b, G, h, penalty);
      const auto cs = cold.solve();
      const auto cr = cold.relax(kappa);
      all_conv &= ws.converged == 1 && cs.converged == 1 && wr.converged == 1 &&
                  cr.converged == 1;
      warm_iters += ws.iters;
      cold_iters += cs.iters;
      relax_warm_iters += wr.iters;
      relax_cold_iters += cr.iters;
      worst_dx = std::max(worst_dx, (ws.x - cs.x).lpNorm<Eigen::Infinity>());
      worst_rdx = std::max(worst_rdx, (wr.x - cr.x).lpNorm<Eigen::Infinity>());
      last_relax_x = wr.x;
      // The rows grow 1.5^k, so termination is by the relative clause late
      // in the run: bound the KKT residual relative to the data scale
      worst_kkt = std::max(
          worst_kkt,
          problem_gen::ElasticKKTResidual(qp0.Q, q, A, b, G, h, penalty, ws.x,
                                          ws.t, ws.y, ws.z_t, ws.z) /
              std::max({1.0, h.lpNorm<Eigen::Infinity>(),
                        b.lpNorm<Eigen::Infinity>()}));
    }
    std::printf(
        "  refreshes=%d/%d worst_drift_after=%.2f | solve iters warm=%d "
        "cold=%d | relax iters warm=%d cold=%d | worst_kkt=%9.2e\n",
        refreshes, ticks, worst_drift, warm_iters, cold_iters, relax_warm_iters,
        relax_cold_iters, worst_kkt);
    Check("auto refresh fires and settles",
          refreshes >= 3 && refreshes < ticks &&
              worst_drift <= warm.settings.ruiz_refresh_ratio,
          worst_drift, "drift");
    Check("solve matches fresh setup",
          all_conv && worst_dx < 1e-4 && worst_kkt < 1e-8, worst_dx, "|dx|");
    // (The relax warm chain is mostly rejected by the flip gate on these
    // 1.5x row jumps, so only accuracy is checked here; the exact remap of
    // the relax iterate is checked below.)
    Check("relax matches fresh setup", worst_rdx < 1e-4, worst_rdx, "|dx|");
    // Manual call on equilibrated data is a no-op
    warm.reequilibrate();
    const auto ws2 = warm.solve();
    Check("manual reequilibrate no-op on fresh data",
          ws2.converged == 1 && ws2.iters == 0 && warm.factorizations() == 0,
          warm.scaling_drift(), "drift");
    // Exact remap check: re-equilibrate with the problem UNCHANGED (the
    // setup-time scaling was deliberately left half-converged), so the
    // remapped solve() and relax() warm iterates must still be converged,
    // and the refreshed scaling must be the one a fresh setup() computes
    {
      pdal::Solver fresh;
      fresh.settings = TightSettings();
      fresh.settings.ruiz = true;
      fresh.setup(qp0.Q, q, A, b, G, h, penalty);
      pdal::Solver half;
      half.settings = TightSettings();
      half.settings.ruiz = true;
      half.settings.ruiz_max_iter = 1;
      half.settings.ruiz_refresh_ratio = 0;
      half.setup(qp0.Q, q, A, b, G, h, penalty);
      const auto hs = half.solve();
      const auto hr = half.relax(kappa);
      const double drift_before = half.scaling_drift();
      half.settings.ruiz_max_iter = 10;
      half.reequilibrate();
      const double drift_after = half.scaling_drift();
      const auto hs2 = half.solve();
      const auto hr2 = half.relax(kappa);
      const double rdx = (hr2.x - hr.x).lpNorm<Eigen::Infinity>();
      std::printf(
          "  unchanged problem: drift %.2f -> %.2f | solve iters %d "
          "-> %d | relax iters %d -> %d |dx|=%9.2e\n",
          drift_before, drift_after, hs.iters, hs2.iters, hr.iters, hr2.iters,
          rdx);
      Check("remap keeps solve() warm iterate",
            hs.converged == 1 && drift_before > 1.5 &&
                drift_after <= fresh.scaling_drift() * (1 + 1e-9) &&
                hs2.converged == 1 && hs2.iters == 0,
            hs2.iters, "iters");
      Check("remap keeps relax() warm iterate",
            hr.converged == 1 && hr2.converged == 1 && hr2.iters == 0 &&
                rdx < 1e-9,
            rdx, "|dx|");
    }
    // Drifted problem: from the same user-frame iterate, the re-equilibrated
    // solver must reach the same solution as one left on the stale scaling
    // (iteration counts differ at noise level either way on a 5x row drift)
    for (int i = 0; i < p; ++i) {
      const double sc = (i % 3 == 0) ? 3.0 : (i % 3 == 1) ? 0.2 : 1.0;
      G.row(i) *= sc;
      h[i] *= sc;
      penalty[i] /= sc;
    }
    warm.set_G(G);
    warm.set_h(h);
    warm.set_penalty(penalty);
    warm.settings.ruiz_refresh_ratio = 0;  // manual only
    pdal::Solver stale = warm;             // same state, scaling left as is
    warm.reequilibrate();
    const auto ss3 = stale.solve();
    const auto ws3 = warm.solve();
    const auto sr3 = stale.relax(kappa);
    const auto wr3 = warm.relax(kappa);
    const double dx3 = (ws3.x - ss3.x).lpNorm<Eigen::Infinity>();
    const double rdx3 = (wr3.x - sr3.x).lpNorm<Eigen::Infinity>();
    std::printf(
        "  drifted rows: solve iters stale=%d remapped=%d |dx|=%9.2e "
        "| relax iters stale=%d remapped=%d |dx|=%9.2e\n",
        ss3.iters, ws3.iters, dx3, sr3.iters, wr3.iters, rdx3);
    Check("remapped warm solve matches stale",
          ss3.converged == 1 && ws3.converged == 1 && dx3 < 1e-6 &&
              ws3.iters <= 2 * ss3.iters,
          dx3, "|dx|");
    Check("remapped warm relax matches stale",
          sr3.converged == 1 && wr3.converged == 1 && rdx3 < 1e-6, rdx3,
          "|dx|");
  }

  std::printf("PDAL: warm start under matrix drift (Q, A, G)\n");
  {
    const int n = 20, m = 5, p = 80, ticks = 10;
    const QPData qp0 = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd penalty = VectorXd::Constant(p, 10.0);
    // PSD Hessian drift keeps Q positive definite for every tick.
    const MatrixXd R = problem_gen::Randn(rng, n, n);
    const MatrixXd dQ = 0.02 * (R.transpose() * R) / n;
    const MatrixXd dA = 0.002 * problem_gen::Randn(rng, m, n);
    const MatrixXd dG = 0.002 * problem_gen::Randn(rng, p, n);

    pdal::Solver warm;
    warm.settings = TightSettings();
    warm.setup(qp0.Q, qp0.q, qp0.A, qp0.b, qp0.G, qp0.h, penalty);
    double worst_dx = 0, worst_kkt = 0;
    bool all_ok = true;
    int min_factors = 1 << 30;
    for (int k = 1; k <= ticks; ++k) {
      const MatrixXd Q = qp0.Q + k * dQ;
      const MatrixXd A = qp0.A + k * dA;
      const MatrixXd G = qp0.G + k * dG;
      warm.set_Q(Q);
      warm.set_A(A);
      warm.set_G(G);
      const auto& ws = warm.solve();
      min_factors = std::min(min_factors, warm.factorizations());
      const auto cs =
          SolveWith<pdal::Solver>(Q, qp0.q, A, qp0.b, G, qp0.h, penalty);
      all_ok &= ws.converged == 1 && cs.converged == 1;
      worst_dx = std::max(worst_dx, (ws.x - cs.x).lpNorm<Eigen::Infinity>());
      worst_kkt = std::max(worst_kkt, problem_gen::ElasticKKTResidual(
                                          Q, qp0.q, A, qp0.b, G, qp0.h, penalty,
                                          ws.x, ws.t, ws.y, ws.z_t, ws.z));
    }
    // A matrix update must invalidate the cached factorization.
    Check("n=20 m=5 p=80 10 matrix ticks",
          all_ok && worst_dx < 1e-4 && worst_kkt < 1e-6 && min_factors >= 1,
          worst_dx, "|dx|");
  }

  std::printf("PDAL: relax(kappa) reaches the kappa-relaxed central point\n");
  {
    // The relaxed point satisfies the elastic KKT with complementarity
    // s.z = kappa on both blocks; it is the same point ipm::Solver::relax
    // targets, so the two solvers must agree on it. This is the key
    // independent check of the differentiability machinery. Cover
    // equalities, conflicts (active elastic slacks), and a range of kappa.
    for (const double kappa : {1e-2, 1e-3, 1e-6}) {
      const QPData qp = problem_gen::InfeasibleEq(rng, 14, 4, 60, 15);
      const VectorXd penalty = VectorXd::Constant(60, 10.0);
      pdal::Solver solver;
      solver.settings = TightSettings();
      solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
      const elastiqp::Solution tight = solver.solve();
      const elastiqp::Solution rel = solver.relax(kappa, 1e-10, 50);

      ipm::Solver ipm;
      ipm.settings = TightIpmSettings();
      ipm.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
      ipm.solve();
      const elastiqp::Solution iref = ipm.relax(kappa, 1e-10, 100);

      const double dx = (rel.x - iref.x).lpNorm<Eigen::Infinity>();
      const double dz = (rel.z - iref.z).lpNorm<Eigen::Infinity>();
      // Complementarity is enforced by the retraction exactly; seen through
      // the reported (t, z_t, z) with the slacks reconstructed as s_t = t,
      // s_ineq = h + t - G x, it holds to the relaxed point's primal
      // residual (1e-10) times the dual (<= penalty).
      const VectorXd s_ineq = qp.h + rel.t - qp.G * rel.x;
      double comp = 0;
      for (int i = 0; i < 60; ++i) {
        comp = std::max(comp, std::abs(rel.t[i] * rel.z_t[i] - kappa));
        comp = std::max(comp, std::abs(s_ineq[i] * rel.z[i] - kappa));
      }
      char name[64];
      std::snprintf(name, sizeof(name), "kappa=%.0e matches IPM relax", kappa);
      Check(name,
            tight.converged == 1 && rel.converged == 1 && iref.converged == 1 &&
                dx < 1e-7 && dz < 1e-6 && comp < 1e-8 && rel.iters <= 12,
            std::max(dx, dz), "|dx|,|dz|");
    }
  }

  std::printf("PDAL: relax leaves the tight iterate untouched\n");
  {
    // relax() must not disturb warm starting: a re-solve after relax should
    // converge as immediately as one without.
    const QPData qp = problem_gen::InfeasibleEq(rng, 20, 5, 80, 20);
    const VectorXd penalty = VectorXd::Constant(80, 10.0);
    pdal::Solver solver;
    solver.settings = TightSettings();
    solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    const elastiqp::Solution tight = solver.solve();
    const elastiqp::Solution rel = solver.relax(1e-3, 1e-10, 50);
    const double moved = (rel.x - tight.x).lpNorm<Eigen::Infinity>();
    const elastiqp::Solution again = solver.solve();
    const double dx = (again.x - tight.x).lpNorm<Eigen::Infinity>();
    Check("re-solve after relax converges in place",
          tight.converged == 1 && rel.converged == 1 && moved > 1e-8 &&
              again.converged == 1 && again.iters <= 2 && dx < 1e-9,
          dx, "|dx|");
  }

  std::printf("PDAL: relax(warm) chains across data updates\n");
  {
    // Warm-started relax (the default): on a drifting problem the next
    // call continues from the previous relaxed point, whose offset is
    // dominated by LINEAR residual drift that Newton removes
    // quadratically; the tight-retraction start instead re-pays the
    // linear-rate barrier-curvature walk on every call. The chain must
    // land on the cold-start point and never cost more total iterations.
    // Dedicated rng: keeps the shared stream's downstream instances
    // intact (see the fd_rng note below).
    std::mt19937 wrng(7);
    QPData qp = problem_gen::InfeasibleEq(wrng, 16, 4, 50, 12);
    const VectorXd penalty = VectorXd::Constant(50, 10.0);
    pdal::Solver solver;
    solver.settings = TightSettings();
    solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    bool ok = true;
    double dmax = 0;
    int warm_iters = 0, cold_iters = 0;
    for (int tick = 0; tick < 8; ++tick) {
      if (tick > 0) {  // smooth heterogeneous control-loop-scale drift
        for (Eigen::Index i = 0; i < qp.q.size(); ++i) {
          qp.q[i] += 1e-3 * std::sin(0.7 * tick + static_cast<double>(i));
        }
        for (Eigen::Index i = 0; i < qp.h.size(); ++i) {
          qp.h[i] += 1e-3 * std::cos(0.3 * tick + static_cast<double>(i));
        }
        solver.set_q(qp.q);
        solver.set_h(qp.h);
      }
      ok = ok && solver.solve().converged == 1;
      const elastiqp::Solution w = solver.relax(1e-3, 1e-10, 50);  // warm
      const elastiqp::Solution c =
          solver.relax(1e-3, 1e-10, 50, /*warm=*/false);
      ok = ok && w.converged == 1 && c.converged == 1;
      warm_iters += w.iters;
      cold_iters += c.iters;
      dmax = std::max(dmax, (w.x - c.x).lpNorm<Eigen::Infinity>());
    }
    // At control-loop drift the chain wins (measured here ~40 vs ~60
    // iterations; ~5x wall time at robot scale, see bench_diff_robot).
    // The margin below is deliberate slack, not the expectation: when a
    // data step flips the activity of many weakly-active rows (drift
    // ~1e-2 on this instance does), the warm start re-pays the barrier
    // curvature on the flipped rows and can cost MORE than the
    // retraction start -- the chain's advantage is regime-dependent, and
    // this check only pins "same point, no blow-up".
    std::printf("  warm iters=%d cold iters=%d\n", warm_iters, cold_iters);
    Check("warm chain lands on the cold-start point",
          ok && dmax < 1e-7 && warm_iters <= cold_iters + 8, dmax, "|dx|");

    // Gradient-only pattern: a data update followed by relax() with NO
    // intervening solve() must still converge from the chain (Ruiz
    // scaling is fixed at setup, so the frame is unchanged).
    qp.q.array() += 1e-2;
    solver.set_q(qp.q);
    const elastiqp::Solution g = solver.relax(1e-3, 1e-10, 50);
    pdal::Solver ref;
    ref.settings = TightSettings();
    ref.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    ref.solve();
    const elastiqp::Solution rc = ref.relax(1e-3, 1e-10, 50);
    const double dg = (g.x - rc.x).lpNorm<Eigen::Infinity>();
    Check("gradient-only relax (no solve) matches",
          g.converged == 1 && rc.converged == 1 && dg < 1e-7, dg, "|dx|");
  }

  std::printf("PDAL: Ruiz + explicit warm start roundtrip\n");
  {
    const int n = 14, p = 40;
    QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    VectorXd pen(p);
    std::uniform_real_distribution<double> unif(-3.0, 3.0);
    for (int i = 0; i < p; ++i) {
      const double s = std::pow(10.0, unif(rng));
      qp.G.row(i) *= s;
      qp.h[i] *= s;
      pen[i] = 10.0 / s;
    }
    pdal::Solver solver;
    solver.settings = TightSettings();
    solver.settings.ruiz = true;
    solver.settings.warm_start = false;
    solver.setup(qp.Q, qp.q, qp.G, qp.h, pen);
    const auto first = solver.solve();
    // Seeding with the (unscaled) solution must roundtrip through the
    // scaling and converge immediately.
    solver.set_warm_start(first.x, VectorXd(0), first.z);
    const auto& again = solver.solve();
    Check("seeded resolve converges immediately",
          first.converged == 1 && again.converged == 1 && again.iters <= 2,
          static_cast<double>(again.iters), "iters");
  }

  std::printf("PDAL: KktVjp vs finite differences of the relaxed map\n");
  {
    // Directional central-difference check of the implicit-KKT backward
    // pass (elastiqp/kkt_vjp.hpp, the C++ mirror of _kkt_bwd in
    // python/elastiqp/jax.py) against the kappa-relaxed solution map,
    // with a random LINEAR loss on the full relaxed certificate
    // (x, t, y, z_t, z). Linear matters: vjp and FD then both
    // differentiate the same (relaxed) map exactly, with no O(kappa)
    // curvature term (see docs/pdal_differentiability.md). Each
    // direction perturbs ALL data (Q, q, A, b, G, h, penalty) at once,
    // so any wrong term/sign in any gradient block shows up as an O(1)
    // mismatch.
    const double kappa = 1e-3;
    // eps balances FD truncation (third derivatives of the relaxed map
    // scale like 1/kappa^2, giving ~eps^2/kappa^2) against relax_tol
    // solution noise (~relax_tol/eps); measured agreement here is ~5e-6,
    // and any wrong term would be O(1), so the 1e-4 gate has margin on
    // both sides.
    const double eps = 3e-6;
    const double relax_tol = 1e-11;
    // Dedicated generator: the eps balance above is sensitive to the
    // instance's KKT conditioning (most seeds put the feasible+eq case in
    // the 1e-4..1e-3 noise band), so keep these instances pinned and
    // independent of how many draws earlier tests consume.
    std::mt19937 fd_rng(14);
    for (const bool with_eq : {false, true}) {
      const int n = 8, m = with_eq ? 3 : 0, p = 20;
      const QPData qp = with_eq ? problem_gen::RandomFeasible(fd_rng, n, m, p)
                                : problem_gen::Infeasible(fd_rng, n, p, p / 4);
      const VectorXd pen = VectorXd::Constant(p, 10.0);

      elastiqp::Cotangents ct;
      ct.x = problem_gen::Randn(fd_rng, n, 1);
      ct.t = problem_gen::Randn(fd_rng, p, 1);
      ct.y = problem_gen::Randn(fd_rng, m, 1);
      ct.z_t = problem_gen::Randn(fd_rng, p, 1);
      ct.z = problem_gen::Randn(fd_rng, p, 1);

      const auto loss = [&](const MatrixXd& Q, const VectorXd& q,
                            const MatrixXd& A, const VectorXd& b,
                            const MatrixXd& G, const VectorXd& h,
                            const VectorXd& w, elastiqp::Solution* out) {
        pdal::Solver s;
        s.settings = TightSettings();
        s.setup(Q, q, A, b, G, h, w);
        const bool ok = s.solve().converged == 1;
        const elastiqp::Solution& r = s.relax(kappa, relax_tol, 100);
        if (!ok || r.converged != 1) {
          return std::numeric_limits<double>::quiet_NaN();
        }
        if (out != nullptr) *out = r;
        double L =
            ct.x.dot(r.x) + ct.t.dot(r.t) + ct.z_t.dot(r.z_t) + ct.z.dot(r.z);
        if (m > 0) L += ct.y.dot(r.y);
        return L;
      };

      elastiqp::Solution rsol;
      loss(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, pen, &rsol);
      elastiqp::KktVjp vjp;
      vjp.setup(n, m, p);
      const elastiqp::DataGrads& g =
          vjp.compute(qp.Q, qp.A, qp.G, qp.h, rsol, ct);

      double worst = 0.0;
      for (int dir = 0; dir < 3; ++dir) {
        MatrixXd dQ = problem_gen::Randn(fd_rng, n, n);
        dQ = 0.5 * (dQ + dQ.transpose());
        const MatrixXd dA = problem_gen::Randn(fd_rng, m, n);
        const MatrixXd dG = problem_gen::Randn(fd_rng, p, n);
        const VectorXd dq = problem_gen::Randn(fd_rng, n, 1);
        const VectorXd db = problem_gen::Randn(fd_rng, m, 1);
        const VectorXd dh = problem_gen::Randn(fd_rng, p, 1);
        const VectorXd dw = problem_gen::Randn(fd_rng, p, 1);

        const double lp = loss(
            qp.Q + eps * dQ, qp.q + eps * dq, qp.A + eps * dA, qp.b + eps * db,
            qp.G + eps * dG, qp.h + eps * dh, pen + eps * dw, nullptr);
        const double lm = loss(
            qp.Q - eps * dQ, qp.q - eps * dq, qp.A - eps * dA, qp.b - eps * db,
            qp.G - eps * dG, qp.h - eps * dh, pen - eps * dw, nullptr);
        const double fd = (lp - lm) / (2.0 * eps);
        double an = (g.Q.array() * dQ.array()).sum() + g.q.dot(dq) +
                    (g.G.array() * dG.array()).sum() + g.h.dot(dh) +
                    g.penalty.dot(dw);
        if (m > 0) an += (g.A.array() * dA.array()).sum() + g.b.dot(db);
        const double err = std::abs(fd - an) / std::max(1.0, std::abs(fd));
        if (!std::isfinite(err)) {
          worst = err;
          break;
        }
        worst = std::max(worst, err);
      }
      Check(with_eq ? "feasible+eq n=8 m=3 p=20" : "infeasible n=8 p=20",
            std::isfinite(worst) && worst < 1e-4, worst, "relerr");
    }
  }

  std::printf(test_util::g_all_ok ? "\nAll PDAL tests passed.\n"
                                  : "\nFAILURES\n");
  return test_util::g_all_ok ? 0 : 1;
}
