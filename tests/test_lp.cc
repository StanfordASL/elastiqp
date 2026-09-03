// Linear programs (Q = 0).
//
// LPs are not the focus of elastiqp, but nothing in the method requires
// strict convexity: the factored matrix is Q + rho I + (1/mu_eq) A'A +
// (1/mu_in) G_act'G_act with rho > 0, so the LLT exists with a zero
// Hessian and the proximal method of multipliers converges as usual. This
// test pins that down without an external LP solver by constructing
// problems whose optimum is known analytically:
//
//   * pick x*, n active rows G_act (full rank a.s.) with h_act = G_act x*,
//     strictly positive duals z_act, and set q = -G_act' z_act (dual
//     feasibility). Strict complementarity + n independent active rows
//     make x* the unique vertex optimum. Extra rows are slack at x* and a
//     box keeps the polytope bounded.
//   * with equalities: b = A x*, n-m active inequality rows, and
//     q = -A' y* - G_act' z_act, so [A; G_act] is square and x* unique.
//
// Also covered: a degenerate optimal face (objective correct, x on the
// face), tight tolerances, and the documented limitation that an unbounded
// LP is not certified as such -- it just fails to converge.

#include <cmath>
#include <cstdio>
#include <random>

#include "elastiqp/elastiqp.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

bool g_all_ok = true;

void Check(const char* name, bool ok, double val, const char* what) {
  std::printf("  %-44s %s=%9.2e %s\n", name, what, val, ok ? "OK" : "FAIL");
  g_all_ok &= ok;
}

struct Lp {
  VectorXd q, b, h, x_star, z_star;  // z_star: duals on all inequality rows
  MatrixXd A, G;
};

// Vertex-optimum LP: n_act = n - m active inequality rows, p_slack slack
// rows, a box of half-width 10 (slack: |x*| < 10), m equality rows.
Lp MakeVertexLp(std::mt19937& rng, int n, int m, int p_slack) {
  std::normal_distribution<double> N(0.0, 1.0);
  std::uniform_real_distribution<double> U(0.5, 2.0);
  auto randn = [&](Eigen::Index r, Eigen::Index c) {
    MatrixXd M(r, c);
    for (Eigen::Index i = 0; i < r; ++i)
      for (Eigen::Index j = 0; j < c; ++j) M(i, j) = N(rng);
    return M;
  };

  Lp lp;
  const int n_act = n - m;
  const int p = n_act + p_slack + 2 * n;
  lp.x_star = randn(n, 1);
  lp.x_star *= 2.0 / std::max(1.0, lp.x_star.lpNorm<Eigen::Infinity>());

  lp.A = randn(m, n);
  lp.b = lp.A * lp.x_star;
  VectorXd y_star = randn(m, 1);

  lp.G.resize(p, n);
  lp.h.resize(p);
  lp.z_star = VectorXd::Zero(p);
  // Active rows: tight at x*, strictly positive duals.
  lp.G.topRows(n_act) = randn(n_act, n);
  for (int i = 0; i < n_act; ++i) {
    lp.h[i] = lp.G.row(i).dot(lp.x_star);
    lp.z_star[i] = U(rng);
  }
  // Slack rows.
  lp.G.middleRows(n_act, p_slack) = randn(p_slack, n);
  for (int i = n_act; i < n_act + p_slack; ++i) {
    lp.h[i] = lp.G.row(i).dot(lp.x_star) + U(rng);
  }
  // Box |x| <= 10.
  lp.G.middleRows(n_act + p_slack, n) = MatrixXd::Identity(n, n);
  lp.G.bottomRows(n) = -MatrixXd::Identity(n, n);
  lp.h.tail(2 * n).setConstant(10.0);

  lp.q = -lp.A.transpose() * y_star - lp.G.transpose() * lp.z_star;
  return lp;
}

void CheckVertexLp(const char* name, const Lp& lp, double penalty,
                   const elastiqp::Settings& settings, double tol) {
  const int n = static_cast<int>(lp.q.size());
  const MatrixXd Q = MatrixXd::Zero(n, n);
  const elastiqp::Solution sol =
      elastiqp::Solve(Q, lp.q, lp.A, lp.b, lp.G, lp.h, penalty, settings);
  const double dx = (sol.x - lp.x_star).lpNorm<Eigen::Infinity>();
  const double dz = (sol.z_ineq - lp.z_star).lpNorm<Eigen::Infinity>();
  const double viol = (lp.G * sol.x - lp.h).cwiseMax(0.0).maxCoeff();
  const double eqres =
      lp.b.size() > 0 ? (lp.A * sol.x - lp.b).lpNorm<Eigen::Infinity>() : 0.0;
  char buf[96];
  std::snprintf(buf, sizeof buf, "%s: converged", name);
  Check(buf, sol.converged == 1, sol.iters, "iters");
  std::snprintf(buf, sizeof buf, "%s: x = x*", name);
  Check(buf, dx < tol, dx, "|dx|");
  std::snprintf(buf, sizeof buf, "%s: z = z*", name);
  Check(buf, dz < tol, dz, "|dz|");
  std::snprintf(buf, sizeof buf, "%s: feasible", name);
  Check(buf, viol < tol && eqres < tol, std::max(viol, eqres), "viol");
  std::snprintf(buf, sizeof buf, "%s: penalty not engaged", name);
  Check(buf, sol.t.maxCoeff() < tol, sol.t.maxCoeff(), "tmax");
}

}  // namespace

int main() {
  std::printf("LP (Q = 0) solves\n");
  const double penalty = 1e4;
  elastiqp::Settings def;

  // min -x1 - 2 x2  s.t. 0 <= x <= 1  ->  x* = (1, 1), z* = (1, 2, 0, 0)
  {
    const MatrixXd Q = MatrixXd::Zero(2, 2);
    VectorXd q(2), h(4);
    MatrixXd G(4, 2);
    q << -1, -2;
    G << 1, 0, 0, 1, -1, 0, 0, -1;
    h << 1, 1, 0, 0;
    const elastiqp::Solution sol = elastiqp::Solve(Q, q, G, h, penalty);
    VectorXd xs(2), zs(4);
    xs << 1, 1;
    zs << 1, 2, 0, 0;
    Check("box 2d: converged", sol.converged == 1, sol.iters, "iters");
    Check("box 2d: x = (1,1)", (sol.x - xs).lpNorm<Eigen::Infinity>() < 1e-6,
          (sol.x - xs).lpNorm<Eigen::Infinity>(), "|dx|");
    Check("box 2d: z = (1,2,0,0)",
          (sol.z_ineq - zs).lpNorm<Eigen::Infinity>() < 1e-6,
          (sol.z_ineq - zs).lpNorm<Eigen::Infinity>(), "|dz|");

    elastiqp::Settings tight;
    tight.eps_abs = 1e-8;
    tight.eps_duality_gap_abs = 1e-8;
    const elastiqp::Solution ts = elastiqp::Solve(Q, q, G, h, penalty, tight);
    Check("box 2d eps=1e-8: x = (1,1)",
          ts.converged == 1 && (ts.x - xs).lpNorm<Eigen::Infinity>() < 1e-9,
          (ts.x - xs).lpNorm<Eigen::Infinity>(), "|dx|");
  }

  // Random vertex-optimum LPs, inequality-only and with equalities.
  std::mt19937 rng(7);
  CheckVertexLp("n=5 p=20", MakeVertexLp(rng, 5, 0, 15), penalty, def, 1e-5);
  CheckVertexLp("n=20 p=80", MakeVertexLp(rng, 20, 0, 60), penalty, def, 1e-5);
  CheckVertexLp("n=50 p=200", MakeVertexLp(rng, 50, 0, 150), penalty, def,
                1e-5);
  CheckVertexLp("n=20 m=5 p=60", MakeVertexLp(rng, 20, 5, 45), penalty, def,
                1e-5);
  {
    elastiqp::Settings tight;
    tight.eps_abs = 1e-8;
    tight.eps_duality_gap_abs = 1e-8;
    CheckVertexLp("n=20 m=5 p=60 eps=1e-8", MakeVertexLp(rng, 20, 5, 45),
                  penalty, tight, 1e-7);
  }

  // Degenerate optimal face: min -x1 s.t. |x| <= 1. Any x2 in [-1, 1] is
  // optimal; check the objective and that x lies on the face.
  {
    const MatrixXd Q = MatrixXd::Zero(2, 2);
    VectorXd q(2), h(4);
    MatrixXd G(4, 2);
    q << -1, 0;
    G << 1, 0, 0, 1, -1, 0, 0, -1;
    h << 1, 1, 1, 1;
    const elastiqp::Solution sol = elastiqp::Solve(Q, q, G, h, penalty);
    Check("degenerate face: converged", sol.converged == 1, sol.iters,
          "iters");
    Check("degenerate face: obj = -1", std::abs(q.dot(sol.x) + 1.0) < 1e-6,
          std::abs(q.dot(sol.x) + 1.0), "|dobj|");
    Check("degenerate face: |x2| <= 1", std::abs(sol.x[1]) <= 1.0 + 1e-6,
          std::abs(sol.x[1]), "|x2|");
  }

  // Unbounded LP: min -x1 s.t. |x2| <= 1. Documented limitation: there is
  // no dual-infeasibility certificate, the solve just fails to converge.
  {
    const MatrixXd Q = MatrixXd::Zero(2, 2);
    VectorXd q(2), h(2);
    MatrixXd G(2, 2);
    q << -1, 0;
    G << 0, 1, 0, -1;
    h << 1, 1;
    elastiqp::Settings s;
    s.max_outer_iter = 50;
    const elastiqp::Solution sol = elastiqp::Solve(Q, q, G, h, penalty, s);
    Check("unbounded: not reported solved", sol.converged == 0, sol.x[0],
          "x1");
  }

  std::printf("%s\n", g_all_ok ? "ALL OK" : "FAILURES");
  return g_all_ok ? 0 : 1;
}
