// Random problem generators and solver-agnostic residuals shared by the
// ElastiQP tests and benchmarks.
//
// Elastic QP:
//   minimize    0.5 x^T Q x + q^T x + penalty^T t
//   subject to  A x == b (hard),  G x - t <= h,  t >= 0

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <random>

namespace problem_gen {

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct QPData {
  MatrixXd Q, A, G;
  VectorXd q, b, h;
  VectorXd x_opt;  // known optimizer of the STRICT problem
};

inline MatrixXd Randn(std::mt19937& rng, int rows, int cols) {
  std::normal_distribution<double> dist;
  MatrixXd M(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) M(i, j) = dist(rng);
  return M;
}

// Strictly convex QP with a feasible primal-dual point (qpax's construction:
// complementary s,z so half the inequalities are active at the solution).
inline QPData RandomFeasible(std::mt19937& rng, int n, int m, int p) {
  QPData qp;
  const MatrixXd P0 = Randn(rng, n, n);
  qp.Q = P0.transpose() * P0 + 1e-3 * MatrixXd::Identity(n, n);
  qp.A = Randn(rng, m, n);
  qp.G = Randn(rng, p, n);

  const VectorXd x = Randn(rng, n, 1);
  VectorXd s = Randn(rng, p, 1).cwiseAbs();
  VectorXd z = Randn(rng, p, 1).cwiseAbs();
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  for (int i = 0; i < p; ++i) {
    if (unif(rng) < 0.5)
      s[i] = 0.0;
    else
      z[i] = 0.0;
  }
  qp.h = qp.G * x + s;
  qp.b = qp.A * x;
  const VectorXd y = Randn(rng, m, 1);
  qp.q = -qp.Q * x - qp.G.transpose() * z - qp.A.transpose() * y;
  qp.x_opt = x;
  return qp;
}

// Feasible instance without equalities: high-penalty elastic solution should
// match the strict QP.
inline QPData Feasible(std::mt19937& rng, int n, int p) {
  return RandomFeasible(rng, n, 0, p);
}

// Instantaneously infeasible instance: pairs of conflicting rows
// g'x <= c and -g'x <= -(c + gap), which no x can satisfy simultaneously.
// This is the regime the elastic form exists for.
inline QPData Infeasible(std::mt19937& rng, int n, int p, int n_conflicts,
                         double gap = 1.0) {
  QPData qp = RandomFeasible(rng, n, 0, p);
  for (int k = 0; k < n_conflicts; ++k) {
    const int i = 2 * k;
    const int j = 2 * k + 1;
    if (j >= p) break;
    qp.G.row(j) = -qp.G.row(i);
    qp.h[j] = -(qp.h[i] + gap);
  }
  return qp;
}

// Instantaneously infeasible instance WITH hard equality constraints:
// feasible base problem (so Ax = b is consistent by construction) plus
// conflicting inequality pairs.
inline QPData InfeasibleEq(std::mt19937& rng, int n, int m, int p,
                           int n_conflicts, double gap = 1.0) {
  QPData qp = RandomFeasible(rng, n, m, p);
  for (int k = 0; k < n_conflicts; ++k) {
    const int i = 2 * k;
    const int j = 2 * k + 1;
    if (j >= p) break;
    qp.G.row(j) = -qp.G.row(i);
    qp.h[j] = -(qp.h[i] + gap);
  }
  return qp;
}

// Max KKT residual of the elastic QP (with hard equalities Ax = b, dual y)
// at (x, t, y, z_t, z), where z_t are the duals of t >= 0 and z those of
// Gx - t <= h. Covers stationarity in x and t, primal feasibility
// (equality and inequality), dual feasibility, and complementarity.
inline double ElasticKKTResidual(const MatrixXd& Q, const VectorXd& q,
                                 const MatrixXd& A, const VectorXd& b,
                                 const MatrixXd& G, const VectorXd& h,
                                 const VectorXd& penalty, const VectorXd& x,
                                 const VectorXd& t, const VectorXd& y,
                                 const VectorXd& z_t, const VectorXd& z) {
  VectorXd stat = Q * x + q + G.transpose() * z;
  if (b.size() > 0) stat += A.transpose() * y;
  double res = stat.lpNorm<Eigen::Infinity>();
  if (b.size() > 0)
    res = std::max(res, (A * x - b).lpNorm<Eigen::Infinity>());
  res = std::max(res, (penalty - z_t - z).lpNorm<Eigen::Infinity>());
  const VectorXd viol = G * x - t - h;  // <= 0 at feasibility
  res = std::max(res, viol.cwiseMax(0.0).maxCoeff());
  res = std::max(res, (-t).cwiseMax(0.0).maxCoeff());
  res = std::max(res, (-z_t).cwiseMax(0.0).maxCoeff());
  res = std::max(res, (-z).cwiseMax(0.0).maxCoeff());
  res = std::max(res, z.cwiseProduct(viol).lpNorm<Eigen::Infinity>());
  res = std::max(res, z_t.cwiseProduct(t).lpNorm<Eigen::Infinity>());
  return res;
}

// Equality-free overload.
inline double ElasticKKTResidual(const MatrixXd& Q, const VectorXd& q,
                                 const MatrixXd& G, const VectorXd& h,
                                 const VectorXd& penalty, const VectorXd& x,
                                 const VectorXd& t, const VectorXd& z_t,
                                 const VectorXd& z) {
  return ElasticKKTResidual(Q, q, MatrixXd(0, q.size()), VectorXd(0), G, h,
                            penalty, x, t, VectorXd(0), z_t, z);
}

// Elastic objective 0.5 x'Qx + q'x + penalty' max(0, Gx - h) evaluated with
// the implied optimal slack t = max(0, Gx - h) (exact for any x, so solvers
// can be compared on x alone).
inline double ElasticObjective(const MatrixXd& Q, const VectorXd& q,
                               const MatrixXd& G, const VectorXd& h,
                               const VectorXd& penalty, const VectorXd& x) {
  const VectorXd t = (G * x - h).cwiseMax(0.0);
  return 0.5 * x.dot(Q * x) + q.dot(x) + penalty.dot(t);
}

// Expanded formulation with explicit slack variables for reference solvers:
// variables xt = (x, t) in R^{n+p},
//   minimize   0.5 xt' [Q 0; 0 0] xt + [q; penalty]' xt
//   subject to [A 0] xt == b,   [G -I] xt <= h,   xt_l = (-inf, 0) <= xt
struct ExpandedElastic {
  MatrixXd P;   // (n+p) x (n+p)
  VectorXd c;   // n+p
  MatrixXd A;   // m x (n+p) equality block (0 rows if no equalities)
  VectorXd b;   // m
  MatrixXd Gt;  // p x (n+p)
  VectorXd h;   // p (upper bound)
  VectorXd lb;  // n+p (lower bound: -inf for x, 0 for t)
};

inline ExpandedElastic MakeExpanded(const MatrixXd& Q, const VectorXd& q,
                                    const MatrixXd& A, const VectorXd& b,
                                    const MatrixXd& G, const VectorXd& h,
                                    const VectorXd& penalty) {
  const Eigen::Index n = q.size();
  const Eigen::Index m = b.size();
  const Eigen::Index p = h.size();
  ExpandedElastic e;
  e.P = MatrixXd::Zero(n + p, n + p);
  e.P.topLeftCorner(n, n) = Q;
  e.c.resize(n + p);
  e.c << q, penalty;
  e.A = MatrixXd::Zero(m, n + p);
  e.A.leftCols(n) = A;
  e.b = b;
  e.Gt = MatrixXd::Zero(p, n + p);
  e.Gt.leftCols(n) = G;
  e.Gt.rightCols(p) = -MatrixXd::Identity(p, p);
  e.h = h;
  e.lb = VectorXd::Constant(n + p, -std::numeric_limits<double>::infinity());
  e.lb.tail(p).setZero();
  return e;
}

inline ExpandedElastic MakeExpanded(const MatrixXd& Q, const VectorXd& q,
                                    const MatrixXd& G, const VectorXd& h,
                                    const VectorXd& penalty) {
  return MakeExpanded(Q, q, MatrixXd(0, q.size()), VectorXd(0), G, h, penalty);
}

}  // namespace problem_gen
