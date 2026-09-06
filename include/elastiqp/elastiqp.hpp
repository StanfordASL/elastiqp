// ElastiQP: an always-feasible QP solver for constrained robot control.
//
// Umbrella header. The three backends solve the same elastic QP (stated in
// elastiqp/common.hpp) and return the same Solution certificate:
//
//   elastiqp::das::Solver    dual active set (DAQP-style)      elastiqp_das.hpp
//   elastiqp::pdal::Solver  primal-dual augmented Lagrangian  elastiqp_pdal.hpp
//   elastiqp::ipm::Solver   proximal interior point (PIQP)    elastiqp_ipm.hpp
//
// elastiqp::Solver / Settings / Solve() name the default backend, the
// active-set method. Each header is self-contained and can be included on
// its own. relax() and the KKT VJP (kkt_vjp.hpp) are available on the PDAL
// and IPM backends only.

#pragma once

#include "elastiqp/common.hpp"
#include "elastiqp/elastiqp_das.hpp"
#include "elastiqp/elastiqp_ipm.hpp"
#include "elastiqp/elastiqp_pdal.hpp"
#include "elastiqp/kkt_vjp.hpp"

namespace elastiqp {

using das::Settings;
using das::Solve;
using das::Solver;

}  // namespace elastiqp
