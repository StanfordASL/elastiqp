// Interior-point-specific behaviour of elastiqp::ipm::Solver (the elastic
// QP itself is covered for every backend in test_solvers.cc; the IPM is
// also the oracle there, so this file is what pins the oracle's own
// mechanics):
//
// (1) relax(kappa): the kappa-central point (s.z = kappa on every pair,
//     stationarity and equalities intact), with and without Ruiz, and
//     agreement with the PDAL's relax() on the same point;
// (2) Ruiz on/off equivalence and the certified equality gate (every
//     solve() is cold: the IPM has no warm start).

#include <cmath>
#include <cstdio>
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

namespace {

ipm::Settings Tight() { return Backend<ipm::Solver>::Tight(); }

double Kkt(const MatrixXd& Q, const VectorXd& q, const MatrixXd& A,
           const VectorXd& b, const MatrixXd& G, const VectorXd& h,
           const VectorXd& w, const elastiqp::Solution& s) {
  return problem_gen::ElasticKKTResidual(Q, q, A, b, G, h, w, s.x, s.t, s.y,
                                         s.z_t, s.z);
}

// Worst |s.z - kappa| over both blocks, plus stationarity and equality
// residuals at a relaxed point. The slacks are reconstructed from t and
// G x - h (s_t = t, s_ineq = h + t - G x), so the complementarity error
// includes the relaxed point's primal residual times the dual.
struct RelaxRes {
  double comp, stat, eq;
};
RelaxRes RelaxResiduals(const QPData& qp, double kappa,
                        const elastiqp::Solution& r) {
  RelaxRes res{0, 0, 0};
  const VectorXd s_ineq = qp.h + r.t - qp.G * r.x;
  for (Eigen::Index i = 0; i < qp.h.size(); ++i) {
    res.comp = std::max(res.comp, std::abs(r.t[i] * r.z_t[i] - kappa));
    res.comp = std::max(res.comp, std::abs(s_ineq[i] * r.z[i] - kappa));
  }
  VectorXd stat = qp.Q * r.x + qp.q + qp.G.transpose() * r.z;
  if (qp.b.size() > 0) stat += qp.A.transpose() * r.y;
  res.stat = InfNorm(stat);
  res.eq = qp.b.size() > 0 ? InfNorm(qp.A * r.x - qp.b) : 0.0;
  return res;
}

}  // namespace

int main() {
  std::mt19937 rng(42);

  std::printf("IPM: kappa relaxation (smoothed-gradient point)\n");
  for (const bool ruiz : {false, true}) {
    // relax(kappa) must land on the kappa-central path: same stationarity
    // and primal feasibility, but s.z = kappa on every pair. kappa is in
    // the user's frame: the row scaling cancels within each s.z pair.
    const int n = 20, m = 5, p = 80;
    const double kappa = 1e-3;
    QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    VectorXd penalty = VectorXd::Constant(p, 10.0);
    if (ruiz) {
      std::uniform_real_distribution<double> unif(-3.0, 3.0);
      for (int i = 0; i < p; ++i) {
        const double s = std::pow(10.0, unif(rng));
        qp.G.row(i) *= s;
        qp.h[i] *= s;
        penalty[i] = 10.0 / s;
      }
    }
    ipm::Solver solver;
    solver.settings = Tight();
    solver.settings.ruiz = ruiz;
    solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    const auto tight = solver.solve();
    const auto& rsol = solver.relax(kappa);
    const RelaxRes r = RelaxResiduals(qp, kappa, rsol);
    const double dx = InfNorm(rsol.x - tight.x);
    std::printf("  ruiz=%d comp_err=%.1e stat=%.1e eq=%.1e |x_r - x|=%.1e\n",
                ruiz, r.comp, r.stat, r.eq, dx);
    Check(ruiz ? "n=20 m=5 p=80 ruiz kappa=1e-3" : "n=20 m=5 p=80 kappa=1e-3",
          tight.converged == 1 && rsol.converged == 1 && r.comp < 1e-7 &&
              r.stat < 1e-6 && r.eq < 1e-7 && dx < 0.1 && dx > 0,
          r.comp, "comp");
  }

  std::printf("IPM: relax agrees with the PDAL's relax\n");
  for (const double kappa : {1e-2, 1e-3, 1e-6}) {
    const QPData qp = problem_gen::InfeasibleEq(rng, 14, 4, 60, 15);
    const VectorXd penalty = VectorXd::Constant(60, 10.0);
    ipm::Solver solver;
    solver.settings = Tight();
    solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    solver.solve();
    const elastiqp::Solution ir = solver.relax(kappa, 1e-10, 100);
    pdal::Solver ps;
    ps.settings = Backend<pdal::Solver>::Tight();
    ps.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty);
    ps.solve();
    const elastiqp::Solution pr = ps.relax(kappa, 1e-10, 50);
    const double dx = InfNorm(ir.x - pr.x);
    const double dz = InfNorm(ir.z - pr.z);
    char name[64];
    std::snprintf(name, sizeof name, "kappa=%.0e matches pdal relax", kappa);
    Check(name,
          ir.converged == 1 && pr.converged == 1 && dx < 1e-7 && dz < 1e-6,
          std::max(dx, dz), "|dx|,|dz|");
  }

  std::printf("IPM: Ruiz on/off reach the same point\n");
  {
    const int n = 20, m = 5, p = 80;
    QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    VectorXd penalty(p);
    std::uniform_real_distribution<double> unif(-2.0, 2.0);
    for (int i = 0; i < p; ++i) {
      const double s = std::pow(10.0, unif(rng));
      qp.G.row(i) *= s;
      qp.h[i] *= s;
      penalty[i] = 10.0 / s;
    }
    ipm::Settings on = Tight(), off = Tight();
    on.ruiz = true;
    off.ruiz = false;
    const auto son =
        SolveWith<ipm::Solver>(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty, on);
    const auto soff = SolveWith<ipm::Solver>(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h,
                                             penalty, off);
    const double dx = InfNorm(son.x - soff.x);
    const double dobj = std::abs(son.primal_obj - soff.primal_obj) /
                        std::max(1.0, std::abs(soff.primal_obj));
    std::printf(
        "  ruiz on iters=%d off iters=%d |dx|=%.1e dobj=%.1e pri=%.1e "
        "dua=%.1e\n",
        son.iters, soff.iters, dx, dobj, son.primal_res, son.dual_res);
    Check("x and objective agree",
          son.converged == 1 && soff.converged == 1 && dx < 1e-5 && dobj < 1e-7,
          dx, "|dx|");
    Check("reported residuals in user frame",
          son.primal_res < 1e-8 && son.dual_res < 1e-8 &&
              Kkt(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty, son) < 1e-7,
          Kkt(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, penalty, son), "kkt");
  }

  std::printf("IPM: equality gate, then a consistent b solves again\n");
  {
    const int n = 10, p = 20;
    const QPData qp = problem_gen::Feasible(rng, n, p);
    MatrixXd Ai(2, n);
    Ai.row(0) = problem_gen::Randn(rng, 1, n);
    Ai.row(1) = Ai.row(0);
    VectorXd bc(2), bi(2);
    bc << 0.5, 0.5;
    bi << 0.0, 1.0;
    ipm::Solver solver;  // defaults: eps_rel = 0 gates
    solver.setup(qp.Q, qp.q, Ai, bc, qp.G, qp.h, 10.0);
    const auto good = solver.solve();
    solver.set_b(bi);
    const auto fail = solver.solve();
    const double lb = solver.eq_infeasibility();
    solver.set_b(bc);
    const auto again = solver.solve();
    std::printf("  good iters=%d again iters=%d\n", good.iters, again.iters);
    Check("gated tick: kInfeasible, 0 iters, bound reported",
          good.converged == 1 && fail.status == elastiqp::Status::kInfeasible &&
              fail.iters == 0 && lb > 0.1,
          lb, "eq_infeas");
    Check("cold re-solve after the gated tick matches",
          again.converged == 1 && again.iters == good.iters &&
              InfNorm(again.x - good.x) < 1e-9 &&
              solver.eq_infeasibility() < 1e-10,
          static_cast<double>(again.iters), "iters");
  }

  std::printf(test_util::g_all_ok ? "\nAll IPM tests passed.\n"
                                  : "\nFAILURES\n");
  return test_util::g_all_ok ? 0 : 1;
}
