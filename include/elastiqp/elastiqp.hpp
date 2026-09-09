// ElastiQP: an always-feasible QP solver for constrained robot control.
//
// Umbrella header. The three backends solve the same elastic QP (stated in
// elastiqp/common.hpp) and return the same Solution certificate:
//
//   elastiqp::das::Solver   DAS:  dual active set (DAQP-style)           das.hpp
//   elastiqp::pdal::Solver  PDAL: primal-dual augmented Lagrangian       pdal.hpp
//   elastiqp::ipm::Solver   IPM:  proximal interior-point method (PIQP)  ipm.hpp
//   elastiqp::sparse_pdal::Solver  PDAL on Eigen::SparseMatrix data     sparse_pdal.hpp
//
// elastiqp::Solver / Settings / Solve() name the default backend, the
// active-set method. Each header is self-contained and can be included on
// its own. relax() and the KKT VJP (kkt_vjp.hpp) are available on the PDAL
// and IPM backends only.

#pragma once

#include "elastiqp/common.hpp"
#include "elastiqp/das.hpp"
#include "elastiqp/ipm.hpp"
#include "elastiqp/pdal.hpp"
#include "elastiqp/sparse_pdal.hpp"
#include "elastiqp/kkt_vjp.hpp"

namespace elastiqp {

using das::Settings;
using das::Solve;
using das::Solver;

}  // namespace elastiqp
