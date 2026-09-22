// Shared drifting-trajectory generator for the warm-start benchmarks
// (bench_relax_warm, bench_fwd_warm): random control-loop-style QP
// sequences with a relative-sigma random walk per tick over a chosen
// subset of the data.
#pragma once

#include <Eigen/Dense>
#include <random>
#include <vector>

#include "problem_gen.hpp"

namespace drift_traj {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;

enum class Structure { kFeas, kInfeas, kDegen };

// What random-walks per tick. kQ/kQQ drift only the cost (linear /
// linear+quadratic): the constraint geometry is fixed, so the smoothed
// row configuration should be stable regardless of how far the optimum
// moves. Q drifts through its Cholesky factor (relative, entrywise) so
// it stays positive definite.
enum class Drift { kQ, kQQ, kQH, kQHG, kAll };

inline const char* Name(Structure s) {
  switch (s) {
    case Structure::kFeas:
      return "feas";
    case Structure::kInfeas:
      return "infeas";
    default:
      return "degen";
  }
}
inline const char* Name(Drift d) {
  switch (d) {
    case Drift::kQ:
      return "q";
    case Drift::kQQ:
      return "qQ";
    case Drift::kQH:
      return "qh";
    case Drift::kQHG:
      return "qhG";
    default:
      return "all";
  }
}
inline bool DriftsHB(Drift d) {
  return d == Drift::kQH || d == Drift::kQHG || d == Drift::kAll;
}
inline bool DriftsG(Drift d) { return d == Drift::kQHG || d == Drift::kAll; }
inline bool DriftsQ(Drift d) { return d == Drift::kQQ || d == Drift::kAll; }

struct Size {
  int n, m, p;
};

// Pre-generated trajectory. G/Q are per-tick only when they drift.
struct Trajectory {
  QPData base;
  VectorXd penalty;
  std::vector<VectorXd> q, h, b;
  std::vector<MatrixXd> G;  // empty unless DriftsG
  std::vector<MatrixXd> Q;  // empty unless DriftsQ
};

// penalty may be per-row (mixed penalties); the scalar overload below
// keeps the historical uniform-penalty call sites unchanged.
inline Trajectory MakeTrajectory(Size sz, Structure st, const VectorXd& penalty,
                                 double sigma, Drift drift, unsigned seed,
                                 int ticks) {
  std::mt19937 rng(seed);
  Trajectory traj;
  switch (st) {
    case Structure::kFeas:
      traj.base = problem_gen::RandomFeasible(rng, sz.n, sz.m, sz.p);
      break;
    case Structure::kInfeas:
      traj.base = problem_gen::InfeasibleEq(rng, sz.n, sz.m, sz.p, sz.p / 4);
      break;
    case Structure::kDegen: {
      traj.base = problem_gen::RandomFeasible(rng, sz.n, sz.m, sz.p);
      // Put every other row exactly on the boundary through the
      // construction point, with zero multiplier: a weakly-active tie
      // that drift pushes to either side of the smoothed hyperbola.
      const VectorXd gx = traj.base.G * traj.base.x_opt;
      for (int i = 0; i < sz.p; i += 2) traj.base.h[i] = gx[i];
      break;
    }
  }
  traj.penalty = penalty;

  const double qs = sigma * traj.base.q.lpNorm<Eigen::Infinity>();
  const double hs = sigma * traj.base.h.lpNorm<Eigen::Infinity>();
  const double bs =
      sz.m > 0 ? sigma * traj.base.b.lpNorm<Eigen::Infinity>() : 0.0;
  const double Gs = sigma;  // relative, entrywise on G
  std::normal_distribution<double> dist;
  VectorXd q = traj.base.q, h = traj.base.h, b = traj.base.b;
  MatrixXd G = traj.base.G;
  // Q drifts via its Cholesky factor (Q stays positive definite).
  MatrixXd L, L0;
  if (DriftsQ(drift)) {
    L = traj.base.Q.llt().matrixL();
    L0 = L;
  }
  for (int k = 0; k < ticks; ++k) {
    for (int i = 0; i < sz.n; ++i) q[i] += qs * dist(rng);
    if (DriftsHB(drift)) {
      for (int i = 0; i < sz.p; ++i) h[i] += hs * dist(rng);
      for (int i = 0; i < sz.m; ++i) b[i] += bs * dist(rng);
    }
    if (DriftsG(drift)) {
      for (int i = 0; i < sz.p; ++i) {
        for (int j = 0; j < sz.n; ++j) {
          G(i, j) += Gs * std::abs(traj.base.G(i, j)) * dist(rng);
        }
      }
      traj.G.push_back(G);
    }
    if (DriftsQ(drift)) {
      for (int i = 0; i < sz.n; ++i) {
        for (int j = 0; j <= i; ++j) {
          L(i, j) += sigma * std::abs(L0(i, j)) * dist(rng);
        }
      }
      traj.Q.push_back(L * L.transpose());
    }
    traj.q.push_back(q);
    traj.h.push_back(h);
    traj.b.push_back(b);
  }
  return traj;
}

inline Trajectory MakeTrajectory(Size sz, Structure st, double penalty_w,
                                 double sigma, Drift drift, unsigned seed,
                                 int ticks) {
  return MakeTrajectory(sz, st, VectorXd::Constant(sz.p, penalty_w), sigma,
                        drift, seed, ticks);
}

}  // namespace drift_traj
