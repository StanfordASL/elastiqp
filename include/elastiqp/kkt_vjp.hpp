// Reverse-mode (vjp) implicit differentiation of the elastic KKT system:
// gradients of the solution map w.r.t. the problem data, with no autodiff
// framework required. C++ mirror of _kkt_bwd in python/elastiqp/jax.py --
// see that docstring for the derivation; the block algebra and variable
// names (rt/r5t, v1..v5, E) match it line for line. Keep the two in sync:
// tests/test_pdal.cc pins this implementation against finite differences
// of the relaxed solution map, tests/test_jax_ffi.py pins the JAX one.
//
// Evaluate at a kappa-RELAXED point from elastiqp::Solver::relax() (all
// complementarity margins ~kappa, so z_t > 0 strictly and the divisions
// below are safe); the tight certificate sits exactly on the boundary,
// where these formulas divide by zero. Pass data in the USER frame (the
// same matrices given to setup()/set_*()), never the solver's internal,
// possibly Ruiz-equilibrated copies; relax() returns its Solution
// unscaled, so the two always pair up.
//
// KktVjp follows the Solver's workspace pattern: setup() allocates
// everything once, compute() is then allocation-free -- intended for
// control loops that pull gradients every tick. Vjp() is a one-shot
// convenience wrapper for everything else.

#pragma once

#include <Eigen/Core>
#include <Eigen/LU>

#include "elastiqp/elastiqp.hpp"

namespace elastiqp {

// Cotangents of the solution map (seed with dLoss/d{x,t,y,z_t,z_ineq}
// evaluated at the solution). Empty vectors are treated as zero.
struct Cotangents {
  VectorXd x, t, y, z_t, z_ineq;
};

// Gradients w.r.t. the problem data. Q is symmetrized (matching the
// solver, which only sees 0.5 (Q + Q')); A and b are 0 x n / empty when
// the problem has no equalities.
struct DataGrads {
  MatrixXd Q, A, G;
  VectorXd q, b, h, penalty;
};

// theta_bar = -(dF/dtheta)' K^{-T} wbar with F the elastic KKT residual
// and K = dF/dw, evaluated at the relaxed solution. Condensed, mirroring
// the forward solver: the diagonal t/z_t/z_ineq blocks are eliminated and
// one (n+m) saddle system is factored per call -- O(p n^2 + (n+m)^3)
// instead of O((n+3p+m)^3) for the dense Jacobian.
class KktVjp {
 public:
  // Allocates all workspace for problem dimensions (n, m, p); compute()
  // performs no heap allocation afterwards.
  void setup(Eigen::Index n, Eigen::Index m, Eigen::Index p) {
    n_ = n;
    m_ = m;
    p_ = p;
    g_.Q.resize(n, n);
    g_.A.resize(m, n);
    g_.G.resize(p, n);
    g_.q.resize(n);
    g_.b.resize(m);
    g_.h.resize(p);
    g_.penalty.resize(p);
    E_.resize(p);
    rt_.resize(p);
    r5t_.resize(p);
    w_.resize(p);
    GW_.resize(p, n);
    KKT_.resize(n + m, n + m);
    rhs_.resize(n + m);
    v13_.resize(n + m);
    v2_.resize(p);
    v5_.resize(p);
    lu_ = Eigen::PartialPivLU<MatrixXd>(n + m);
  }

  const DataGrads& grads() const { return g_; }

  const DataGrads& compute(const MatrixXd& Q, const MatrixXd& A,
                           const MatrixXd& G, const VectorXd& h,
                           const Solution& sol, const Cotangents& ct) {
    const VectorXd& x = sol.x;
    const VectorXd& t = sol.t;
    const VectorXd& y = sol.y;
    const VectorXd& z1 = sol.z_t;
    const VectorXd& z2 = sol.z_ineq;

    // E = (G x - t - h) - z2.*t./z1, strictly negative at a relaxed point
    // (first term is -s_ineq).
    E_.noalias() = G * x;
    E_ -= t;
    E_ -= h;
    E_ -= z2.cwiseProduct(t).cwiseQuotient(z1);

    // Rescaled rhs of the symmetrized transpose solve:
    // rt = r4 + t.*r2, r5t = r5 + z2.*rt./z1 (jax.py notation).
    rt_.setZero();
    if (ct.z_t.size() > 0) rt_ -= z1.cwiseProduct(ct.z_t);
    if (ct.t.size() > 0) rt_ += t.cwiseProduct(ct.t);
    r5t_ = z2.cwiseProduct(rt_).cwiseQuotient(z1);
    if (ct.z_ineq.size() > 0) r5t_ += z2.cwiseProduct(ct.z_ineq);

    w_ = r5t_.cwiseQuotient(E_);
    if (ct.x.size() > 0) {
      rhs_.head(n_) = ct.x;
    } else {
      rhs_.head(n_).setZero();
    }
    rhs_.head(n_).noalias() -= G.transpose() * w_;
    if (m_ > 0) {
      if (ct.y.size() > 0) {
        rhs_.tail(m_) = ct.y;
      } else {
        rhs_.tail(m_).setZero();
      }
    }

    // (n+m) saddle system [Qs + G' diag(-z2/E) G, A'; A, 0].
    w_ = -z2.cwiseQuotient(E_);
    GW_.noalias() = w_.asDiagonal() * G;
    KKT_.topLeftCorner(n_, n_) = 0.5 * (Q + Q.transpose());
    KKT_.topLeftCorner(n_, n_).noalias() += G.transpose() * GW_;
    if (m_ > 0) {
      KKT_.topRightCorner(n_, m_) = A.transpose();
      KKT_.bottomLeftCorner(m_, n_) = A;
      KKT_.bottomRightCorner(m_, m_).setZero();
    }
    lu_.compute(KKT_);
    v13_ = lu_.solve(rhs_);
    const auto v1 = v13_.head(n_);
    const auto v3 = v13_.tail(m_);

    v5_.noalias() = G * v1;
    v5_ = (r5t_ - z2.cwiseProduct(v5_)).cwiseQuotient(E_);
    v2_ = (rt_ + t.cwiseProduct(v5_)).cwiseQuotient(z1);

    g_.q = -v1;
    g_.penalty = -v2_;
    g_.b = v3;
    g_.h = v5_;
    g_.Q.noalias() = -0.5 * (v1 * x.transpose());
    g_.Q.noalias() -= 0.5 * (x * v1.transpose());
    g_.G.noalias() = -(z2 * v1.transpose());
    g_.G.noalias() -= v5_ * x.transpose();
    if (m_ > 0) {
      g_.A.noalias() = -(y * v1.transpose());
      g_.A.noalias() -= v3 * x.transpose();
    }
    return g_;
  }

 private:
  Eigen::Index n_ = 0, m_ = 0, p_ = 0;
  DataGrads g_;
  VectorXd E_, rt_, r5t_, w_;
  MatrixXd GW_, KKT_;
  VectorXd rhs_, v13_, v2_, v5_;
  Eigen::PartialPivLU<MatrixXd> lu_;
};

// One-shot convenience wrapper (allocates; use the KktVjp class in loops).
inline DataGrads Vjp(const MatrixXd& Q, const MatrixXd& A, const MatrixXd& G,
                     const VectorXd& h, const Solution& sol,
                     const Cotangents& ct) {
  KktVjp vjp;
  vjp.setup(Q.rows(), A.rows(), G.rows());
  return vjp.compute(Q, A, G, h, sol, ct);
}

// Equality-free overload.
inline DataGrads Vjp(const MatrixXd& Q, const MatrixXd& G, const VectorXd& h,
                     const Solution& sol, const Cotangents& ct) {
  return Vjp(Q, MatrixXd(0, Q.rows()), G, h, sol, ct);
}

}  // namespace elastiqp
