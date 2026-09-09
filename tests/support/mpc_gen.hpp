// Multiple-shooting linear MPC problems for the sparse tests and benchmarks.
//
//   minimize    sum_k 0.5 (x_k - xref)' Qx (x_k - xref) + 0.5 u_k' Ru u_k
//                 + 0.5 (x_N - xref)' QN (x_N - xref)
//   subject to  x_0 = x0,  x_{k+1} = A x_k + B u_k + c     (hard, m rows)
//               xmin <= x_k <= xmax  (k = 0..N)             (elastic)
//               umin <= u_k <= umax  (k = 0..N-1)           (elastic)
//               Cx x_k <= cx         (k = 0..N)             (elastic)
//               Cu u_k <= cu         (k = 0..N-1)           (elastic)
//               Eu_k u_k == eu_k     (k = 0..N-1)           (hard, optional)
//
// The Eu rows are per-stage hard equalities on the inputs with a FIXED
// pattern (the nonzeros of the template Eu; their values may differ per
// stage and may be zero, e.g. a contact schedule pinning the forces of
// swing feet, with all-zero rows for the stance feet). Explicit zeros are
// kept so the sparsity pattern never changes.
//
// Variables are stage-interleaved, [x_0; u_0; x_1; u_1; ...; x_N], so the
// KKT matrix is block-banded. Only finite bounds produce rows. The random
// system follows the OSQP benchmark Control class (A = I + 0.1 randn,
// spectral radius pushed below 1, B randn, Qx a random 70%-dense
// nonnegative diagonal, Ru = 0.1 I, QN the DARE cost-to-go), with the boxes
// drawn the same way. Elastic conflicts come from an initial state outside
// its box (x_0 is pinned by the equalities, so its box row must saturate)
// or from a reference the input box cannot track.
//
// The robot models of the benchmarks repo (robotics/mpc_models.hpp) fill a
// LinearSystem the same way; everything here is Eigen-only.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace mpc_gen {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;

constexpr double kInf = std::numeric_limits<double>::infinity();

struct LinearSystem {
  int nx = 0, nu = 0;
  MatrixXd A, B;
  VectorXd c;           // affine drift (zero by default)
  MatrixXd Qx, Ru, QN;  // stage and terminal costs
  VectorXd xmin, xmax, umin, umax;  // +-inf = no row
  MatrixXd Cx, Cu;      // general per-stage rows (may be empty)
  VectorXd cx, cu;
  MatrixXd Eu;          // per-stage input equalities, pattern template
  VectorXd eu;          //   (may be empty); values set per stage
};

inline MatrixXd Randn(std::mt19937& rng, int rows, int cols) {
  std::normal_distribution<double> dist;
  MatrixXd M(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) M(i, j) = dist(rng);
  }
  return M;
}

// Discrete-time Riccati fixed point (terminal cost).
inline MatrixXd RiccatiCost(const LinearSystem& s) {
  MatrixXd P = s.Qx;
  for (int it = 0; it < 1000; ++it) {
    const MatrixXd BtPB = s.B.transpose() * P * s.B + s.Ru;
    const MatrixXd Pn =
        s.Qx + s.A.transpose() * P * s.A -
        s.A.transpose() * P * s.B * BtPB.ldlt().solve(s.B.transpose() * P * s.A);
    const double diff = (Pn - P).norm();
    P = 0.5 * (Pn + Pn.transpose());
    if (diff < 1e-12 * std::max(1.0, P.norm())) break;
  }
  return P;
}

// OSQP-Control-like random system: marginally stable A, dense B.
inline LinearSystem RandomSystem(std::mt19937& rng, int nx, int nu) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  LinearSystem s;
  s.nx = nx;
  s.nu = nu;
  s.A = MatrixXd::Identity(nx, nx) + 0.1 * Randn(rng, nx, nx);
  {
    // Push the spectral radius just below 1, as the OSQP generator does
    // (it rescales individual eigenvalues; a uniform rescale of A has the
    // same effect on the dynamics' stability and keeps A real).
    const Eigen::EigenSolver<MatrixXd> es(s.A, false);
    const double radius = es.eigenvalues().cwiseAbs().maxCoeff();
    if (radius >= 0.99) s.A *= 0.99 / (radius + 0.01);
  }
  s.B = Randn(rng, nx, nu);
  s.c = VectorXd::Zero(nx);
  VectorXd dq(nx);
  for (int i = 0; i < nx; ++i) dq[i] = unif(rng) < 0.7 ? unif(rng) : 0.0;
  s.Qx = dq.asDiagonal();
  s.Ru = 0.1 * MatrixXd::Identity(nu, nu);
  s.QN = RiccatiCost(s);
  s.umax.resize(nu);
  for (int i = 0; i < nu; ++i) s.umax[i] = unif(rng);
  s.umin = -s.umax;
  s.xmax.resize(nx);
  for (int i = 0; i < nx; ++i) s.xmax[i] = 1.0 + unif(rng);
  s.xmin = -s.xmax;
  return s;
}

// The elastic QP of one MPC tick. Q, A, G are fixed by (system, N) up to
// values; q and b carry x0 / xref; h carries the bounds. Inequality rows
// are grouped by kind and stage:
//   [x_up(0..N)] [x_lo(0..N)] [u_up(0..N-1)] [u_lo(0..N-1)]
//   [Cx(0..N)] [Cu(0..N-1)]
// with per-kind row counts n_xup etc. (finite bounds only).
struct MPCQP {
  int nx = 0, nu = 0, N = 0;
  Eigen::Index n = 0, m = 0, p = 0;
  SpMat Q, A, G;
  VectorXd q, b, h;
  // Which bound entries produced rows, and the row-block layout
  std::vector<int> xup_idx, xlo_idx, uup_idx, ulo_idx;
  Eigen::Index row_xup = 0, row_xlo = 0, row_uup = 0, row_ulo = 0, row_cx = 0,
               row_cu = 0;
  Eigen::Index n_cx = 0, n_cu = 0, n_eu = 0;
  Eigen::Index row_eu = 0;  // first Eu equality row (after the dynamics)

  Eigen::Index x_index(int k) const {
    return static_cast<Eigen::Index>(k) * (nx + nu);
  }
  Eigen::Index u_index(int k) const { return x_index(k) + nx; }
  VectorXd x_stage(const VectorXd& x, int k) const {
    return x.segment(x_index(k), nx);
  }
  VectorXd u_stage(const VectorXd& x, int k) const {
    return x.segment(u_index(k), nu);
  }
  // Row blocks per stage
  Eigen::Index xup_row(int k) const {
    return row_xup + static_cast<Eigen::Index>(k) * xup_idx.size();
  }
  Eigen::Index xlo_row(int k) const {
    return row_xlo + static_cast<Eigen::Index>(k) * xlo_idx.size();
  }
  Eigen::Index uup_row(int k) const {
    return row_uup + static_cast<Eigen::Index>(k) * uup_idx.size();
  }
  Eigen::Index ulo_row(int k) const {
    return row_ulo + static_cast<Eigen::Index>(k) * ulo_idx.size();
  }
  Eigen::Index cx_row(int k) const {
    return row_cx + static_cast<Eigen::Index>(k) * n_cx;
  }
  Eigen::Index cu_row(int k) const {
    return row_cu + static_cast<Eigen::Index>(k) * n_cu;
  }
  // Equality row block of stage k (block 0 pins x_0)
  Eigen::Index eq_row(int k) const { return static_cast<Eigen::Index>(k) * nx; }
  Eigen::Index eu_row(int k) const {
    return row_eu + static_cast<Eigen::Index>(k) * n_eu;
  }
};

inline MPCQP MakeMPC(const LinearSystem& s, int N, const VectorXd& x0,
                     const VectorXd& xref) {
  MPCQP qp;
  qp.nx = s.nx;
  qp.nu = s.nu;
  qp.N = N;
  const int nx = s.nx, nu = s.nu;
  qp.n = static_cast<Eigen::Index>(N) * (nx + nu) + nx;
  qp.n_eu = s.Eu.rows();
  qp.row_eu = static_cast<Eigen::Index>(N + 1) * nx;
  qp.m = qp.row_eu + static_cast<Eigen::Index>(N) * qp.n_eu;
  for (int i = 0; i < nx; ++i) {
    if (std::isfinite(s.xmax[i])) qp.xup_idx.push_back(i);
    if (std::isfinite(s.xmin[i])) qp.xlo_idx.push_back(i);
  }
  for (int i = 0; i < nu; ++i) {
    if (std::isfinite(s.umax[i])) qp.uup_idx.push_back(i);
    if (std::isfinite(s.umin[i])) qp.ulo_idx.push_back(i);
  }
  qp.n_cx = s.Cx.rows();
  qp.n_cu = s.Cu.rows();
  qp.row_xup = 0;
  qp.row_xlo = qp.row_xup + static_cast<Eigen::Index>(N + 1) * qp.xup_idx.size();
  qp.row_uup = qp.row_xlo + static_cast<Eigen::Index>(N + 1) * qp.xlo_idx.size();
  qp.row_ulo = qp.row_uup + static_cast<Eigen::Index>(N) * qp.uup_idx.size();
  qp.row_cx = qp.row_ulo + static_cast<Eigen::Index>(N) * qp.ulo_idx.size();
  qp.row_cu = qp.row_cx + static_cast<Eigen::Index>(N + 1) * qp.n_cx;
  qp.p = qp.row_cu + static_cast<Eigen::Index>(N) * qp.n_cu;

  using T = Eigen::Triplet<double>;
  std::vector<T> tq, ta, tg;
  qp.q = VectorXd::Zero(qp.n);
  for (int k = 0; k <= N; ++k) {
    const MatrixXd& Qk = k == N ? s.QN : s.Qx;
    const Eigen::Index xi = qp.x_index(k);
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < nx; ++j) {
        if (Qk(i, j) != 0.0) tq.emplace_back(xi + i, xi + j, Qk(i, j));
      }
    }
    qp.q.segment(xi, nx) = -Qk * xref;
    if (k < N) {
      const Eigen::Index ui = qp.u_index(k);
      for (int i = 0; i < nu; ++i) {
        for (int j = 0; j < nu; ++j) {
          if (s.Ru(i, j) != 0.0) tq.emplace_back(ui + i, ui + j, s.Ru(i, j));
        }
      }
    }
  }
  qp.Q.resize(qp.n, qp.n);
  qp.Q.setFromTriplets(tq.begin(), tq.end());

  qp.b = VectorXd::Zero(qp.m);
  for (int i = 0; i < nx; ++i) ta.emplace_back(i, qp.x_index(0) + i, 1.0);
  qp.b.head(nx) = x0;
  for (int k = 0; k < N; ++k) {
    const Eigen::Index row = qp.eq_row(k + 1);
    const Eigen::Index xi = qp.x_index(k), ui = qp.u_index(k),
                       xn = qp.x_index(k + 1);
    for (int i = 0; i < nx; ++i) {
      ta.emplace_back(row + i, xn + i, 1.0);
      for (int j = 0; j < nx; ++j) {
        if (s.A(i, j) != 0.0) ta.emplace_back(row + i, xi + j, -s.A(i, j));
      }
      for (int j = 0; j < nu; ++j) {
        if (s.B(i, j) != 0.0) ta.emplace_back(row + i, ui + j, -s.B(i, j));
      }
    }
    qp.b.segment(row, nx) = s.c;
  }
  for (int k = 0; k < N; ++k) {
    const Eigen::Index ui = qp.u_index(k);
    for (Eigen::Index r = 0; r < qp.n_eu; ++r) {
      for (int j = 0; j < nu; ++j) {
        // template pattern (the values are set per stage by SetStageEqU)
        if (s.Eu(r, j) != 0.0) ta.emplace_back(qp.eu_row(k) + r, ui + j, s.Eu(r, j));
      }
      qp.b[qp.eu_row(k) + r] = s.eu.size() > r ? s.eu[r] : 0.0;
    }
  }
  qp.A.resize(qp.m, qp.n);
  qp.A.setFromTriplets(ta.begin(), ta.end());

  qp.h.resize(qp.p);
  for (int k = 0; k <= N; ++k) {
    const Eigen::Index xi = qp.x_index(k);
    for (size_t r = 0; r < qp.xup_idx.size(); ++r) {
      const int i = qp.xup_idx[r];
      tg.emplace_back(qp.xup_row(k) + static_cast<Eigen::Index>(r), xi + i, 1.0);
      qp.h[qp.xup_row(k) + static_cast<Eigen::Index>(r)] = s.xmax[i];
    }
    for (size_t r = 0; r < qp.xlo_idx.size(); ++r) {
      const int i = qp.xlo_idx[r];
      tg.emplace_back(qp.xlo_row(k) + static_cast<Eigen::Index>(r), xi + i, -1.0);
      qp.h[qp.xlo_row(k) + static_cast<Eigen::Index>(r)] = -s.xmin[i];
    }
    for (Eigen::Index r = 0; r < qp.n_cx; ++r) {
      for (int j = 0; j < nx; ++j) {
        if (s.Cx(r, j) != 0.0) tg.emplace_back(qp.cx_row(k) + r, xi + j, s.Cx(r, j));
      }
      qp.h[qp.cx_row(k) + r] = s.cx[r];
    }
  }
  for (int k = 0; k < N; ++k) {
    const Eigen::Index ui = qp.u_index(k);
    for (size_t r = 0; r < qp.uup_idx.size(); ++r) {
      const int i = qp.uup_idx[r];
      tg.emplace_back(qp.uup_row(k) + static_cast<Eigen::Index>(r), ui + i, 1.0);
      qp.h[qp.uup_row(k) + static_cast<Eigen::Index>(r)] = s.umax[i];
    }
    for (size_t r = 0; r < qp.ulo_idx.size(); ++r) {
      const int i = qp.ulo_idx[r];
      tg.emplace_back(qp.ulo_row(k) + static_cast<Eigen::Index>(r), ui + i, -1.0);
      qp.h[qp.ulo_row(k) + static_cast<Eigen::Index>(r)] = -s.umin[i];
    }
    for (Eigen::Index r = 0; r < qp.n_cu; ++r) {
      for (int j = 0; j < nu; ++j) {
        if (s.Cu(r, j) != 0.0) tg.emplace_back(qp.cu_row(k) + r, ui + j, s.Cu(r, j));
      }
      qp.h[qp.cu_row(k) + r] = s.cu[r];
    }
  }
  qp.G.resize(qp.p, qp.n);
  qp.G.setFromTriplets(tg.begin(), tg.end());
  qp.Q.makeCompressed();
  qp.A.makeCompressed();
  qp.G.makeCompressed();
  return qp;
}

// Vector updates of an existing tick (same matrices).
inline void SetInitialState(MPCQP& qp, const VectorXd& x0) {
  qp.b.head(qp.nx) = x0;
}
inline void SetReference(const LinearSystem& s, MPCQP& qp,
                         const VectorXd& xref) {
  for (int k = 0; k <= qp.N; ++k) {
    qp.q.segment(qp.x_index(k), qp.nx) = -(k == qp.N ? s.QN : s.Qx) * xref;
  }
}
// Per-stage reference trajectory (xref_k for k = 0..N)
inline void SetReferenceTrajectory(const LinearSystem& s, MPCQP& qp,
                                   const std::vector<VectorXd>& xref) {
  for (int k = 0; k <= qp.N; ++k) {
    qp.q.segment(qp.x_index(k), qp.nx) =
        -(k == qp.N ? s.QN : s.Qx) * xref[static_cast<size_t>(k)];
  }
}
// Dynamics values (A, B, c) written in place; the pattern must be
// unchanged (an entry that was zero at MakeMPC() must stay zero).
inline void SetDynamics(const LinearSystem& s, MPCQP& qp) {
  const int nx = s.nx, nu = s.nu;
  for (int k = 0; k < qp.N; ++k) {
    const Eigen::Index row = qp.eq_row(k + 1);
    const Eigen::Index xi = qp.x_index(k), ui = qp.u_index(k);
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < nx; ++j) {
        if (s.A(i, j) != 0.0) qp.A.coeffRef(row + i, xi + j) = -s.A(i, j);
      }
      for (int j = 0; j < nu; ++j) {
        if (s.B(i, j) != 0.0) qp.A.coeffRef(row + i, ui + j) = -s.B(i, j);
      }
    }
    qp.b.segment(row, nx) = s.c;
  }
}
// Per-stage dynamics values (A_k, B_k, c_k) written in place, same pattern
// as the template (sys.A, sys.B) used at MakeMPC().
inline void SetStageDynamics(const LinearSystem& s, MPCQP& qp, int k,
                             const MatrixXd& Ak, const MatrixXd& Bk,
                             const VectorXd& ck) {
  const int nx = s.nx, nu = s.nu;
  const Eigen::Index row = qp.eq_row(k + 1);
  const Eigen::Index xi = qp.x_index(k), ui = qp.u_index(k);
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < nx; ++j) {
      if (s.A(i, j) != 0.0) qp.A.coeffRef(row + i, xi + j) = -Ak(i, j);
    }
    for (int j = 0; j < nu; ++j) {
      if (s.B(i, j) != 0.0) qp.A.coeffRef(row + i, ui + j) = -Bk(i, j);
    }
  }
  qp.b.segment(row, nx) = ck;
}
// Per-stage input equality values (template pattern of sys.Eu).
inline void SetStageEqU(const LinearSystem& s, MPCQP& qp, int k,
                        const MatrixXd& Eu_k, const VectorXd& eu_k) {
  const Eigen::Index ui = qp.u_index(k);
  for (Eigen::Index r = 0; r < qp.n_eu; ++r) {
    for (int j = 0; j < s.nu; ++j) {
      if (s.Eu(r, j) != 0.0) qp.A.coeffRef(qp.eu_row(k) + r, ui + j) = Eu_k(r, j);
    }
    qp.b[qp.eu_row(k) + r] = eu_k[r];
  }
}
// Per-stage general-row bounds on u (Cu rows of stage k)
inline void SetStageCu(MPCQP& qp, int k, const VectorXd& cu_k) {
  for (Eigen::Index r = 0; r < qp.n_cu; ++r) qp.h[qp.cu_row(k) + r] = cu_k[r];
}
// Per-stage general-row bounds (e.g. a contact schedule)
inline void SetStageBoundsU(MPCQP& qp, int k, const VectorXd& umax,
                            const VectorXd& umin) {
  for (size_t r = 0; r < qp.uup_idx.size(); ++r) {
    qp.h[qp.uup_row(k) + static_cast<Eigen::Index>(r)] = umax[qp.uup_idx[r]];
  }
  for (size_t r = 0; r < qp.ulo_idx.size(); ++r) {
    qp.h[qp.ulo_row(k) + static_cast<Eigen::Index>(r)] = -umin[qp.ulo_idx[r]];
  }
}

// Row-kind penalty vector: (state boxes, input boxes, Cx rows, Cu rows).
inline VectorXd MakePenalty(const MPCQP& qp, double w_xbox, double w_ubox,
                            double w_cx, double w_cu) {
  VectorXd w(qp.p);
  w.segment(qp.row_xup, qp.row_uup - qp.row_xup).setConstant(w_xbox);
  w.segment(qp.row_uup, qp.row_cx - qp.row_uup).setConstant(w_ubox);
  w.segment(qp.row_cx, qp.row_cu - qp.row_cx).setConstant(w_cx);
  w.segment(qp.row_cu, qp.p - qp.row_cu).setConstant(w_cu);
  return w;
}

// Shifted warm start: the next tick's solution is close to the previous
// one advanced by one stage, so (x, y, z) are shifted stage-wise with the
// last stage duplicated (the receding-horizon warm start of the MPC
// literature; the unshifted iterate is off by one stage).
inline void ShiftWarmStart(const MPCQP& qp, VectorXd& x, VectorXd& y,
                           VectorXd& z) {
  const int nx = qp.nx, nu = qp.nu, N = qp.N;
  VectorXd xs = x, ys = y, zs = z;
  auto shift_block = [&](VectorXd& dst, const VectorXd& src,
                         Eigen::Index row_k, Eigen::Index row_k1,
                         Eigen::Index len) {
    if (len > 0) dst.segment(row_k, len) = src.segment(row_k1, len);
  };
  for (int k = 0; k < N; ++k) {
    xs.segment(qp.x_index(k), nx) = x.segment(qp.x_index(k + 1), nx);
    if (k + 1 < N) {
      xs.segment(qp.u_index(k), nu) = x.segment(qp.u_index(k + 1), nu);
    }
    // equality duals: dynamics block k+1 -> k, Eu block k+1 -> k
    if (k + 1 < N) {
      ys.segment(qp.eq_row(k + 1), nx) = y.segment(qp.eq_row(k + 2), nx);
      shift_block(ys, y, qp.eu_row(k), qp.eu_row(k + 1), qp.n_eu);
    }
    shift_block(zs, z, qp.xup_row(k), qp.xup_row(k + 1),
                static_cast<Eigen::Index>(qp.xup_idx.size()));
    shift_block(zs, z, qp.xlo_row(k), qp.xlo_row(k + 1),
                static_cast<Eigen::Index>(qp.xlo_idx.size()));
    shift_block(zs, z, qp.cx_row(k), qp.cx_row(k + 1), qp.n_cx);
    if (k + 1 < N) {
      shift_block(zs, z, qp.uup_row(k), qp.uup_row(k + 1),
                  static_cast<Eigen::Index>(qp.uup_idx.size()));
      shift_block(zs, z, qp.ulo_row(k), qp.ulo_row(k + 1),
                  static_cast<Eigen::Index>(qp.ulo_idx.size()));
      shift_block(zs, z, qp.cu_row(k), qp.cu_row(k + 1), qp.n_cu);
    }
  }
  ys.head(nx) = y.segment(nx, nx);  // x_0 pin dual ~ the first dynamics dual
  x = xs;
  y = ys;
  z = zs;
}

// One closed-loop step: apply the first input of the solution, plus
// process noise of the given scale.
inline VectorXd Step(const LinearSystem& s, const VectorXd& x0,
                     const MPCQP& qp, const VectorXd& xsol,
                     std::mt19937& rng, double noise) {
  VectorXd xn = s.A * x0 + s.B * qp.u_stage(xsol, 0) + s.c;
  if (noise > 0.0) xn += noise * Randn(rng, s.nx, 1);
  return xn;
}

}  // namespace mpc_gen
