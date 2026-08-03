// ElastiQP: shared types for both solver backends

#pragma once

#include <Eigen/Core>

namespace elastiqp {

using Eigen::MatrixXd;
using Eigen::VectorXd;

enum class Status {
  kUnsolved = 0,
  kSolved = 1,
  kMaxIter = 2,
  kNumerics = 3,
};

// The elastic KKT certificate, returned by both backends. One slack/dual pair
// per constraint block: _t for the bound t >= 0, _ineq for the elastic rows
// Gx - t <= h. Both pairs have length p; y (length m) is the equality dual.
// At a solution z_t = penalty - z_ineq, so z_ineq in [0, penalty] -- the
// bounded multiplier that makes the L1 penalty exact.
struct Solution {
  VectorXd x;
  VectorXd t;               // per-constraint elastic slacks (= violations)
  VectorXd y;               // equality duals
  VectorXd s_t, s_ineq;     // slacks for t >= 0 and Gx - t <= h
  VectorXd z_t, z_ineq;     // duals for t >= 0 and Gx - t <= h
  Status status = Status::kUnsolved;
  int converged = 0;  // 1 iff status == kSolved
  // Work done, in each method's own unit: interior-point iterations for
  // IpmSolver, total inner semismooth Newton steps for Solver.
  int iters = 0;
  double primal_obj = 0.0;
  double primal_res = 0.0;
  double dual_res = 0.0;
  double duality_gap = 0.0;
};

}  // namespace elastiqp
