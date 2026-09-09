// The sparse PDAL backend (elastiqp::sparse_pdal::Solver) against the dense
// backends on identical data:
//
// (1) random dense problems fed through sparseView(): feasible, conflicting,
//     with equalities, p == 0, inconsistent equalities (the certificate),
//     per-row penalties, Ruiz on/off -- each against the IPM oracle and the
//     dense PDAL;
// (2) multiple-shooting MPC problems (tests/support/mpc_gen.hpp): cold
//     solves with a feasible and an out-of-box initial state, against the
//     dense PDAL;
// (3) the Woodbury active-set correction: incremental updates on vs off
//     reach the same point with fewer factorizations;
// (4) warm-started closed-loop chains, with the box conflict entering and
//     leaving: every tick converges, warm iterations below cold, the
//     solution matching the dense warm chain;
// (5) matrix updates (set_Q / set_A / set_G with the same and with a new
//     pattern) are picked up.

#include <cmath>
#include <cstdio>
#include <random>

#include "elastiqp/elastiqp.hpp"
#include "mpc_gen.hpp"
#include "problem_gen.hpp"
#include "test_util.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;
using test_util::Check;
using test_util::InfNorm;
using test_util::IpmRef;
using test_util::MaxT;
using test_util::SolveWith;
namespace pdal = elastiqp::pdal;
namespace spdal = elastiqp::sparse_pdal;
using spdal::SpMat;

namespace {

spdal::Settings SparseTight() {
  spdal::Settings s;
  static_cast<pdal::Settings&>(s) = test_util::Backend<pdal::Solver>::Tight();
  return s;
}
pdal::Settings DenseTight() {
  return test_util::Backend<pdal::Solver>::Tight();
}

SpMat Sparse(const MatrixXd& M) {
  SpMat S = M.sparseView();
  S.makeCompressed();
  return S;
}

elastiqp::Solution SolveSparse(const QPData& qp, const VectorXd& w,
                               const spdal::Settings& s = SparseTight()) {
  spdal::Solver solver;
  solver.settings = s;
  solver.setup(Sparse(qp.Q), qp.q, Sparse(qp.A), qp.b, Sparse(qp.G), qp.h, w);
  return solver.solve();
}

double Kkt(const QPData& qp, const VectorXd& w, const elastiqp::Solution& s) {
  return problem_gen::ElasticKKTResidual(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w,
                                         s.x, s.t, s.y, s.z_t, s.z);
}
double KktMPC(const mpc_gen::MPCQP& qp, const VectorXd& w,
              const elastiqp::Solution& s) {
  return problem_gen::ElasticKKTResidual(MatrixXd(qp.Q), qp.q, MatrixXd(qp.A),
                                         qp.b, MatrixXd(qp.G), qp.h, w, s.x,
                                         s.t, s.y, s.z_t, s.z);
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  char cell[128];

  std::printf("sparse PDAL: random dense problems through sparseView\n");
  for (auto [n, m, p] : {std::tuple{8, 0, 10}, {14, 0, 60}, {30, 8, 120},
                         {58, 20, 300}}) {
    const QPData qp = problem_gen::RandomFeasible(rng, n, m, p);
    const VectorXd w = VectorXd::Constant(p, 10.0);
    const auto sol = SolveSparse(qp, w);
    const auto ref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const double dx = InfNorm(sol.x - ref.x);
    const double res = Kkt(qp, w, sol);
    std::snprintf(cell, sizeof cell, "feasible n=%d m=%d p=%d", n, m, p);
    Check(cell, sol.converged == 1 && dx < 1e-5 && res < 1e-6, dx, "|dx|");
  }
  for (auto [n, m, p] : {std::tuple{8, 0, 10}, {14, 0, 100}, {30, 6, 200},
                         {58, 12, 300}}) {
    const QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    const VectorXd w = VectorXd::Constant(p, 10.0);
    const auto sol = SolveSparse(qp, w);
    const auto ref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const auto den = SolveWith<pdal::Solver>(qp.Q, qp.q, qp.A, qp.b, qp.G,
                                             qp.h, w);
    const double dx = InfNorm(sol.x - ref.x);
    const double dxd = InfNorm(sol.x - den.x);
    std::snprintf(cell, sizeof cell, "conflict n=%d m=%d p=%d (t>0)", n, m, p);
    Check(cell,
          sol.converged == 1 && den.converged == 1 && dx < 1e-5 &&
              dxd < 1e-5 && MaxT(sol) > 0.1,
          dx, "|dx|");
  }
  {
    const int n = 12, p = 40;
    const QPData qp = problem_gen::Infeasible(rng, n, p, p / 4);
    VectorXd w(p);
    for (int i = 0; i < p; ++i) w[i] = (i % 2) ? 100.0 : 5.0;
    const auto sol = SolveSparse(qp, w);
    const auto ref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const double dx = InfNorm(sol.x - ref.x);
    Check("per-row penalties", sol.converged == 1 && dx < 1e-5, dx, "|dx|");
  }
  {
    // Ruiz on, badly scaled data
    const int n = 20, m = 4, p = 80;
    QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, p / 4);
    VectorXd rs(p);
    for (int i = 0; i < p; ++i) rs[i] = std::pow(10.0, (i % 5) - 2);
    qp.G = rs.asDiagonal() * qp.G;
    qp.h = rs.cwiseProduct(qp.h);
    VectorXd w = 10.0 * rs;
    spdal::Settings s = SparseTight();
    s.ruiz = true;
    const auto sol = SolveSparse(qp, w, s);
    const auto ref = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const double dx = InfNorm(sol.x - ref.x);
    const double res = Kkt(qp, w, sol);
    Check("ruiz on, row-scaled conflict", sol.converged == 1 && dx < 1e-5 &&
                                              res < 1e-6, dx, "|dx|");
  }
  {
    // p == 0: equality-constrained QP, also with a singular Q
    const int n = 20, m = 6;
    QPData qp = problem_gen::RandomFeasible(rng, n, m, 4);
    qp.G.resize(0, n);
    qp.h.resize(0);
    const VectorXd w(0);
    const auto sol = SolveSparse(qp, w);
    const auto den = SolveWith<pdal::Solver>(qp.Q, qp.q, qp.A, qp.b, qp.G,
                                             qp.h, w);
    const double dx = InfNorm(sol.x - den.x);
    const double eqr = InfNorm(qp.A * sol.x - qp.b);
    Check("p == 0 equality QP", sol.converged == 1 && dx < 1e-6 && eqr < 1e-8,
          dx, "|dx|");
    qp.Q.bottomRightCorner(5, 5).setZero();
    qp.Q.bottomRows(5).setZero();
    qp.Q.rightCols(5).setZero();
    const auto sol2 = SolveSparse(qp, w);
    const auto den2 = SolveWith<pdal::Solver>(qp.Q, qp.q, qp.A, qp.b, qp.G,
                                              qp.h, w);
    const double dx2 = InfNorm(sol2.x - den2.x);
    Check("p == 0 singular Q", sol2.converged == 1 && den2.converged == 1 &&
                                    dx2 < 1e-5, dx2, "|dx|");
  }
  {
    // Inconsistent equalities: certificate fires, no iterations
    const int n = 10, p = 20;
    QPData qp = problem_gen::RandomFeasible(rng, n, 0, p);
    qp.A = MatrixXd::Zero(2, n);
    qp.A.row(0).setOnes();
    qp.A.row(1).setOnes();
    qp.b.resize(2);
    qp.b << 0.0, 1.0;
    const VectorXd w = VectorXd::Constant(p, 10.0);
    spdal::Settings gate;  // default settings: eps_rel 0 enables the gate
    const auto sol = SolveSparse(qp, w, gate);
    Check("inconsistent b fails fast",
          sol.status == elastiqp::Status::kInfeasible && sol.iters == 0,
          static_cast<double>(sol.iters), "iters");
    // Consistent but rank-deficient rows: solved
    qp.b << 1.0, 1.0;
    const auto sol2 = SolveSparse(qp, w, gate);
    const auto ref2 = IpmRef(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const double dx2 = InfNorm(sol2.x - ref2.x);
    Check("rank-deficient consistent eqs", sol2.converged == 1 && dx2 < 1e-4,
          dx2, "|dx|");
  }

  std::printf("sparse PDAL: multiple-shooting MPC vs dense PDAL\n");
  const auto sys = mpc_gen::RandomSystem(rng, 8, 3);
  const int N = 15;
  {
    VectorXd x0 = 0.5 * sys.xmax;
    VectorXd xref = VectorXd::Zero(sys.nx);
    const auto qp = mpc_gen::MakeMPC(sys, N, x0, xref);
    const VectorXd w = VectorXd::Constant(qp.p, 100.0);
    spdal::Solver s;
    s.settings = SparseTight();
    s.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const auto sol = s.solve();
    const auto den = SolveWith<pdal::Solver>(MatrixXd(qp.Q), qp.q,
                                             MatrixXd(qp.A), qp.b,
                                             MatrixXd(qp.G), qp.h, w);
    const double dx = InfNorm(sol.x - den.x);
    const double res = KktMPC(qp, w, sol);
    std::snprintf(cell, sizeof cell, "MPC feasible n=%d m=%d p=%d",
                  static_cast<int>(qp.n), static_cast<int>(qp.m),
                  static_cast<int>(qp.p));
    Check(cell, sol.converged == 1 && den.converged == 1 && dx < 1e-5 &&
                    res < 1e-6, dx, "|dx|");
    std::printf("    sparse: %d iters, %d outer, %d factorizations, "
                "%d woodbury solves, nnz(L)=%ld; dense: %d iters\n",
                sol.iters, sol.outer_iters, s.factorizations(),
                s.woodbury_solves(), static_cast<long>(s.factor_nnz()),
                den.iters);
  }
  {
    // Initial state outside its box: the x_0 box rows must saturate
    VectorXd x0 = 1.3 * sys.xmax;
    VectorXd xref = VectorXd::Zero(sys.nx);
    const auto qp = mpc_gen::MakeMPC(sys, N, x0, xref);
    const VectorXd w = VectorXd::Constant(qp.p, 100.0);
    spdal::Solver s;
    s.settings = SparseTight();
    s.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const auto sol = s.solve();
    const auto den = SolveWith<pdal::Solver>(MatrixXd(qp.Q), qp.q,
                                             MatrixXd(qp.A), qp.b,
                                             MatrixXd(qp.G), qp.h, w);
    const double dx = InfNorm(sol.x - den.x);
    const double res = KktMPC(qp, w, sol);
    Check("MPC x0 outside box (conflict)",
          sol.converged == 1 && den.converged == 1 && dx < 1e-5 &&
              res < 1e-6 && MaxT(sol) > 0.1,
          dx, "|dx|");
    std::printf("    sparse: %d iters, %d factorizations, %d saturated; "
                "dense: %d iters\n",
                sol.iters, s.factorizations(), sol.n_saturated, den.iters);

    // Woodbury correction vs refactor on every active-set change
    spdal::Solver full;
    full.settings = SparseTight();
    full.settings.incremental_updates = false;
    full.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    const auto fsol = full.solve();
    const double dxf = InfNorm(sol.x - fsol.x);
    Check("woodbury == refactor-every-change",
          fsol.converged == 1 && dxf < 1e-6 &&
              s.factorizations() < full.factorizations(),
          dxf, "|dx|");
    std::printf("    factorizations: woodbury %d vs full %d (iters %d vs %d)\n",
                s.factorizations(), full.factorizations(), sol.iters,
                fsol.iters);
  }

  std::printf("sparse PDAL: warm-started closed loop through a conflict\n");
  {
    VectorXd x0 = 0.4 * sys.xmax;
    VectorXd xref = VectorXd::Zero(sys.nx);
    auto qp = mpc_gen::MakeMPC(sys, N, x0, xref);
    const VectorXd w = VectorXd::Constant(qp.p, 100.0);
    spdal::Solver warm, cold, shifted;
    warm.settings = SparseTight();
    cold.settings = SparseTight();
    cold.settings.warm_start = false;
    shifted.settings = SparseTight();
    shifted.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    pdal::Solver dwarm;
    dwarm.settings = DenseTight();
    warm.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    cold.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    dwarm.setup(MatrixXd(qp.Q), qp.q, MatrixXd(qp.A), qp.b, MatrixXd(qp.G),
                qp.h, w);
    int fails = 0;
    long warm_iters = 0, cold_iters = 0, dense_iters = 0, shift_iters = 0;
    VectorXd xs, ys, zs;
    int warm_factors = 0;
    double worst_dx = 0.0, worst_kkt = 0.0;
    int conflict_ticks = 0;
    const int ticks = 40;
    for (int t = 0; t < ticks; ++t) {
      // A disturbance kicks the state out of its box in the middle of the
      // run; the loop then brings it back (release)
      if (t == 15) x0 = 1.4 * sys.xmax;
      mpc_gen::SetInitialState(qp, x0);
      warm.set_b(qp.b);
      cold.set_b(qp.b);
      dwarm.set_b(qp.b);
      shifted.set_b(qp.b);
      const auto ws = warm.solve();
      const auto cs = cold.solve();
      const auto ds = dwarm.solve();
      if (t > 0) {
        mpc_gen::ShiftWarmStart(qp, xs, ys, zs);
        shifted.set_warm_start(xs, ys, zs);
      }
      const auto ss = shifted.solve();
      xs = ss.x;
      ys = ss.y;
      zs = ss.z;
      shift_iters += ss.iters;
      if (ws.converged != 1 || cs.converged != 1 || ds.converged != 1 ||
          ss.converged != 1) {
        ++fails;
      }
      warm_iters += ws.iters;
      cold_iters += cs.iters;
      dense_iters += ds.iters;
      warm_factors += warm.factorizations();
      worst_dx = std::max(worst_dx, InfNorm(ws.x - ds.x));
      worst_kkt = std::max(worst_kkt, KktMPC(qp, w, ws));
      if (MaxT(ws) > 1e-6) ++conflict_ticks;
      std::printf("    tick %2d: warm %2d it (%d fac) shifted %2d it cold %2d "
                  "it, t_max %.2e\n",
                  t, ws.iters, warm.factorizations(), ss.iters, cs.iters,
                  MaxT(ws));
      x0 = mpc_gen::Step(sys, x0, qp, ws.x, rng, 0.02);
    }
    Check("closed loop: all ticks converge (sparse warm/cold, dense warm)",
          fails == 0, fails, "fails");
    Check("closed loop: sparse warm matches dense warm", worst_dx < 1e-5,
          worst_dx, "|dx|");
    Check("closed loop: worst elastic KKT residual", worst_kkt < 1e-6,
          worst_kkt, "kkt");
    Check("closed loop: shifted warm iterations < cold",
          shift_iters < cold_iters,
          static_cast<double>(shift_iters) / cold_iters, "shifted/cold");
    Check("closed loop: conflict entered and left",
          conflict_ticks > 0 && conflict_ticks < ticks, conflict_ticks,
          "ticks");
    std::printf("    iters: warm %ld, shifted %ld, cold %ld, dense warm %ld "
                "over %d ticks; warm factorizations %d (%.1f/tick)\n",
                warm_iters, shift_iters, cold_iters, dense_iters, ticks,
                warm_factors,
                static_cast<double>(warm_factors) / ticks);
  }

  std::printf("sparse PDAL: matrix updates\n");
  {
    VectorXd x0 = 0.4 * sys.xmax;
    VectorXd xref = VectorXd::Zero(sys.nx);
    auto qp = mpc_gen::MakeMPC(sys, N, x0, xref);
    const VectorXd w = VectorXd::Constant(qp.p, 100.0);
    spdal::Solver s;
    s.settings = SparseTight();
    s.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
    s.solve();
    // Same pattern, new values (a different system with the same nnz)
    mpc_gen::LinearSystem sys2 = sys;
    sys2.A *= 0.9;
    sys2.B *= 1.2;
    auto qp2 = mpc_gen::MakeMPC(sys2, N, x0, xref);
    s.set_A(qp2.A);
    s.set_b(qp2.b);
    const auto sol = s.solve();
    const auto den = SolveWith<pdal::Solver>(MatrixXd(qp2.Q), qp2.q,
                                             MatrixXd(qp2.A), qp2.b,
                                             MatrixXd(qp2.G), qp2.h, w);
    const double dx = InfNorm(sol.x - den.x);
    Check("set_A same pattern", sol.converged == 1 && dx < 1e-5, dx, "|dx|");
    // New pattern: a coupled input cost and a state-input constraint
    MatrixXd Qd(qp2.Q);
    Qd.block(qp2.u_index(0), qp2.u_index(0), sys.nu, sys.nu) +=
        0.05 * MatrixXd::Ones(sys.nu, sys.nu);
    MatrixXd Gd(qp2.G);
    Gd.row(qp2.uup_row(0)).segment(qp2.x_index(0), sys.nx).setConstant(0.1);
    s.set_Q(Sparse(Qd));
    s.set_G(Sparse(Gd));
    const auto sol2 = s.solve();
    const auto den2 = SolveWith<pdal::Solver>(Qd, qp2.q, MatrixXd(qp2.A),
                                              qp2.b, Gd, qp2.h, w);
    const double dx2 = InfNorm(sol2.x - den2.x);
    Check("set_Q / set_G new pattern", sol2.converged == 1 && dx2 < 1e-5, dx2,
          "|dx|");
  }

  std::printf("%s\n", test_util::g_all_ok ? "ALL OK" : "FAILURES");
  return test_util::g_all_ok ? 0 : 1;
}
