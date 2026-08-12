// Reverse-mode (vjp) implicit differentiation of the elastic KKT system,
// shared by the tests and benchmarks. C++ port of _kkt_bwd in
// python/elastiqp/jax.py -- see that docstring for the derivation; the
// block algebra and variable names (r1..r5, v1..v5, E) match it line for
// line.
//
// Evaluate at a kappa-RELAXED point from elastiqp::Solver::relax() (all
// complementarity margins ~kappa, so z_t > 0 strictly and the divisions
// below are safe); the tight certificate sits exactly on the boundary,
// where these formulas divide by zero.

#pragma once

#include <Eigen/Core>
#include <Eigen/LU>

#include "elastiqp/elastiqp.hpp"

namespace kkt_vjp {

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Cotangents of the solution map (seed with dLoss/d{x,t,y,z_t,z_ineq}).
// Empty vectors are treated as zero.
struct Cotangents {
  VectorXd x, t, y, z_t, z_ineq;
};

// Gradients w.r.t. the problem data.
struct DataGrads {
  MatrixXd Q, A, G;
  VectorXd q, b, h, penalty;
};

// theta_bar = -(dF/dtheta)' K^{-T} wbar with F the elastic KKT residual,
// evaluated at the relaxed solution `sol`. Condensed: eliminates the
// diagonal t/z_t/z_ineq blocks and solves one (n+m) system, O(p n^2 +
// (n+m)^3).
inline DataGrads Vjp(const MatrixXd& Q, const MatrixXd& A, const MatrixXd& G,
                     const VectorXd& h, const elastiqp::Solution& sol,
                     const Cotangents& ct) {
  const Eigen::Index n = Q.rows();
  const Eigen::Index m = A.rows();
  const Eigen::Index p = G.rows();
  const VectorXd& x = sol.x;
  const VectorXd& t = sol.t;
  const VectorXd& y = sol.y;
  const VectorXd& z1 = sol.z_t;
  const VectorXd& z2 = sol.z_ineq;

  const auto or_zero = [](const VectorXd& v, Eigen::Index d) {
    return v.size() > 0 ? v : VectorXd(VectorXd::Zero(d));
  };
  const VectorXd r1 = or_zero(ct.x, n);
  const VectorXd r2 = or_zero(ct.t, p);
  const VectorXd r3 = or_zero(ct.y, m);
  const VectorXd r4 = -z1.cwiseProduct(or_zero(ct.z_t, p));
  const VectorXd r5 = z2.cwiseProduct(or_zero(ct.z_ineq, p));

  const MatrixXd Qs = 0.5 * (Q + Q.transpose());
  const VectorXd D = G * x - t - h;  // = -s_ineq <= 0
  const VectorXd E = D - z2.cwiseProduct(t).cwiseQuotient(z1);  // < 0

  const VectorXd r5t =
      r5 + z2.cwiseProduct(r4 + t.cwiseProduct(r2)).cwiseQuotient(z1);
  VectorXd rhs_x = r1;
  rhs_x.noalias() -= G.transpose() * r5t.cwiseQuotient(E);

  MatrixXd H = Qs;
  H.noalias() += G.transpose() * (-z2.cwiseQuotient(E)).asDiagonal() * G;

  VectorXd v1(n), v3(m);
  if (m > 0) {
    MatrixXd KKT = MatrixXd::Zero(n + m, n + m);
    KKT.topLeftCorner(n, n) = H;
    KKT.topRightCorner(n, m) = A.transpose();
    KKT.bottomLeftCorner(m, n) = A;
    VectorXd rhs(n + m);
    rhs << rhs_x, r3;
    const VectorXd v13 = KKT.partialPivLu().solve(rhs);
    v1 = v13.head(n);
    v3 = v13.tail(m);
  } else {
    v1 = H.partialPivLu().solve(rhs_x);
  }

  VectorXd v5 = r5t;
  v5.noalias() -= z2.cwiseProduct(G * v1);
  v5 = v5.cwiseQuotient(E);
  const VectorXd v2 =
      (r4 + t.cwiseProduct(r2) + t.cwiseProduct(v5)).cwiseQuotient(z1);

  DataGrads g;
  g.q = -v1;
  g.penalty = -v2;
  g.b = v3;
  g.h = v5;
  g.Q = -0.5 * (v1 * x.transpose() + x * v1.transpose());
  g.A = -(y * v1.transpose() + v3 * x.transpose());
  g.G = -(z2 * v1.transpose() + v5 * x.transpose());
  return g;
}

}  // namespace kkt_vjp
