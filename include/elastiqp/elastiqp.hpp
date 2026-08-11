// ElastiQP: an elastic QP solver for robot control.
//
//   minimize    0.5 x^T Q x + q^T x + penalty^T t
//   subject to  A x == b          (hard, dual y)
//               G x - t <= h      (soft, slack s_ineq, dual z_ineq)
//               t >= 0            (slack s_t, dual z_t)
//
// Every inequality gets its own L1-penalized slack t_i. As such, the
// inequalities cannot cause infeasibility, and the problem is feasible
// iff the equality constraints are consistent. The slacks are handled
// analytically rather than added as explicit decision variables.
//
// This umbrella header pulls in the following:
//
//   elastiqp/types.hpp  Status, Solution (shared by both backends)
//   elastiqp/pdal.hpp   Solver, Settings, Solve -- a primal-dual augmented
//                       Lagrangian method based on ProxQP. (Default backend)
//   elastiqp/ipm.hpp    IpmSolver, IpmSettings, IpmSolve -- a proximal
//                       interior-point method based on PIQP.

#pragma once

#include "elastiqp/ipm.hpp"
#include "elastiqp/pdal.hpp"
#include "elastiqp/types.hpp"
