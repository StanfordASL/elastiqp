#pragma once

#include <Eigen/Core>
#include <Eigen/LU>

#include "elastiqp/common.hpp"

namespace elastiqp {

// Cotangents of the solution fields; an empty vector means zero.
struct Cotangents {
  VectorXd x, t, y, z_t, z;
};

// Gradients of the loss with respect to the QP data.
struct DataGrads {
  MatrixXd Q, A, G;
  VectorXd q, b, h, penalty;
};

// Reverse-mode derivative of the QP solution map via the implicit function
// theorem. Solves one (n+m) reduced KKT system per call; setup() preallocates.
// Evaluate at a relax()ed solution (z_t > 0, so the divisions are safe) with
// data in the user frame. Mirrors _kkt_bwd in python/elastiqp/jax.py.
class KktVjp {
 public:
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
    rh_.resize(p);
    w_.resize(p);
    GW_.resize(p, n);
    KKT_.resize(n + m, n + m);
    rhs_.resize(n + m);
    vxy_.resize(n + m);
    vpen_.resize(p);
    vh_.resize(p);
    lu_ = Eigen::PartialPivLU<MatrixXd>(n + m);
  }

  const DataGrads& grads() const { return g_; }

  const DataGrads& compute(const MatrixXd& Q, const MatrixXd& A,
                           const MatrixXd& G, const VectorXd& h,
                           const Solution& sol, const Cotangents& ct) {
    const VectorXd& x = sol.x;
    const VectorXd& t = sol.t;
    const VectorXd& y = sol.y;
    const VectorXd& zt = sol.z_t;
    const VectorXd& z = sol.z;

    // Eliminating the t, z_t, z rows reduces the adjoint system to (x, y).
    E_.noalias() = G * x;
    E_ -= t;
    E_ -= h;
    E_ -= z.cwiseProduct(t).cwiseQuotient(zt);

    rt_.setZero();
    if (ct.z_t.size() > 0) rt_ -= zt.cwiseProduct(ct.z_t);
    if (ct.t.size() > 0) rt_ += t.cwiseProduct(ct.t);
    rh_ = z.cwiseProduct(rt_).cwiseQuotient(zt);
    if (ct.z.size() > 0) rh_ += z.cwiseProduct(ct.z);

    w_ = rh_.cwiseQuotient(E_);
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

    // Reduced KKT matrix; w_ is reused for the Schur diagonal.
    w_ = -z.cwiseQuotient(E_);
    GW_.noalias() = w_.asDiagonal() * G;
    KKT_.topLeftCorner(n_, n_) = 0.5 * (Q + Q.transpose());
    KKT_.topLeftCorner(n_, n_).noalias() += G.transpose() * GW_;
    if (m_ > 0) {
      KKT_.topRightCorner(n_, m_) = A.transpose();
      KKT_.bottomLeftCorner(m_, n_) = A;
      KKT_.bottomRightCorner(m_, m_).setZero();
    }
    lu_.compute(KKT_);
    vxy_ = lu_.solve(rhs_);
    const auto vx = vxy_.head(n_);
    const auto vy = vxy_.tail(m_);

    vh_.noalias() = G * vx;
    vh_ = (rh_ - z.cwiseProduct(vh_)).cwiseQuotient(E_);
    vpen_ = (rt_ + t.cwiseProduct(vh_)).cwiseQuotient(zt);

    g_.q = -vx;
    g_.penalty = -vpen_;
    g_.b = vy;
    g_.h = vh_;
    g_.Q.noalias() = -0.5 * (vx * x.transpose());
    g_.Q.noalias() -= 0.5 * (x * vx.transpose());
    g_.G.noalias() = -(z * vx.transpose());
    g_.G.noalias() -= vh_ * x.transpose();
    if (m_ > 0) {
      g_.A.noalias() = -(y * vx.transpose());
      g_.A.noalias() -= vy * x.transpose();
    }
    return g_;
  }

 private:
  Eigen::Index n_ = 0, m_ = 0, p_ = 0;
  DataGrads g_;
  VectorXd E_, rt_, rh_, w_;
  MatrixXd GW_, KKT_;
  VectorXd rhs_, vxy_, vpen_, vh_;
  Eigen::PartialPivLU<MatrixXd> lu_;
};

inline DataGrads Vjp(const MatrixXd& Q, const MatrixXd& A, const MatrixXd& G,
                     const VectorXd& h, const Solution& sol,
                     const Cotangents& ct) {
  KktVjp vjp;
  vjp.setup(Q.rows(), A.rows(), G.rows());
  return vjp.compute(Q, A, G, h, sol, ct);
}

// No equality constraints.
inline DataGrads Vjp(const MatrixXd& Q, const MatrixXd& G, const VectorXd& h,
                     const Solution& sol, const Cotangents& ct) {
  return Vjp(Q, MatrixXd(0, Q.rows()), G, h, sol, ct);
}

}  // namespace elastiqp
