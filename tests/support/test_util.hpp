// Shared helpers for the ElastiQP test suite: the plain-assert Check(),
// per-backend traits so a cell can be written once and run for the
// active-set, PDAL and IPM solvers, and the tight tolerances the
// cross-validation cells pin.
//
// The cells compare against references solved at eps ~ 1e-8..1e-10 with
// agreement thresholds around 1e-5..1e-6, which needs more accuracy than the
// control-sized library defaults; every backend solve in the tests
// therefore pins its tolerances through Backend<Solver>::Tight().

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "elastiqp/elastiqp.hpp"

namespace test_util {

using Eigen::MatrixXd;
using Eigen::VectorXd;

inline bool g_all_ok = true;

inline void Check(const char* name, bool ok, double val, const char* what) {
  std::printf("  %-44s %s=%9.2e %s\n", name, what, val, ok ? "OK" : "FAIL");
  g_all_ok &= ok;
}

inline double InfNorm(const VectorXd& v) {
  return v.size() > 0 ? v.lpNorm<Eigen::Infinity>() : 0.0;
}

// |a - b|_inf relative to the scale of a
inline double RelDiff(const VectorXd& a, const VectorXd& b) {
  return InfNorm(a - b) / std::max(1.0, InfNorm(a));
}

inline double MaxT(const elastiqp::Solution& s) {
  return s.t.size() > 0 ? s.t.maxCoeff() : 0.0;
}

// Backend traits. Settings names differ per method (a BCL round cap, an
// active-set change cap, an interior-point iteration cap), as do the
// exactness properties of the certificate: the PDAL clamps, so a feasible
// row's slack is identically 0; the active set satisfies its rows to
// roundoff; the interior point converges to within eps.
template <class Solver>
struct Backend;

template <>
struct Backend<elastiqp::das::Solver> {
  using Settings = elastiqp::das::Settings;
  static constexpr const char* name = "das";
  static constexpr bool has_relax = false;
  static constexpr bool has_refresh = true;       // drift-gated Ruiz refresh
  static constexpr bool has_warm_start = true;
  static constexpr double slack_tol = 1e-10;      // t at a feasible solution
  static constexpr double invariant_tol = 1e-12;  // z_t + z == w
  static Settings Tight() {
    Settings s;
    s.eps_abs = 1e-8;  // eta_prox follows it
    s.eps_rel = 1e-9;
    return s;
  }
  static void CapIters(Settings& s, int k) { s.max_iter = k; }
  static void CapOuter(Settings& s, int k) { s.max_outer = k; }
};

template <>
struct Backend<elastiqp::pdal::Solver> {
  using Settings = elastiqp::pdal::Settings;
  static constexpr const char* name = "pdal";
  static constexpr bool has_relax = true;
  static constexpr bool has_refresh = true;
  static constexpr bool has_warm_start = true;
  static constexpr double slack_tol = 0.0;
  static constexpr double invariant_tol = 1e-12;
  static Settings Tight() {
    Settings s;
    s.eps_abs = 1e-8;
    s.eps_rel = 1e-9;
    s.eps_duality_gap_abs = 1e-8;
    s.eps_duality_gap_rel = 1e-9;
    return s;
  }
  static void CapIters(Settings& s, int k) { s.max_outer_iter = k; }
  static void CapOuter(Settings& s, int k) { s.max_outer_iter = k; }
};

template <>
struct Backend<elastiqp::ipm::Solver> {
  using Settings = elastiqp::ipm::Settings;
  static constexpr const char* name = "ipm";
  static constexpr bool has_relax = true;
  static constexpr bool has_refresh = false;     // re-equilibrates instead
  static constexpr bool has_warm_start = false;  // every solve() is cold
  static constexpr double slack_tol = 1e-6;
  static constexpr double invariant_tol = 1e-6;
  static Settings Tight() {
    Settings s;
    s.eps_abs = 1e-8;
    s.eps_rel = 1e-9;
    s.eps_duality_gap_abs = 1e-8;
    s.eps_duality_gap_rel = 1e-9;
    return s;
  }
  static void CapIters(Settings& s, int k) { s.max_iter = k; }
  static void CapOuter(Settings& s, int k) { s.max_iter = k; }
};

// Cold solve of one backend at its tight settings (or given settings).
template <class Solver>
elastiqp::Solution SolveWith(
    const MatrixXd& Q, const VectorXd& q, const MatrixXd& A, const VectorXd& b,
    const MatrixXd& G, const VectorXd& h, const VectorXd& penalty,
    const typename Backend<Solver>::Settings& settings =
        Backend<Solver>::Tight()) {
  Solver s;
  s.settings = settings;
  s.setup(Q, q, A, b, G, h, penalty);
  return s.solve();
}
template <class Solver>
elastiqp::Solution SolveWith(const MatrixXd& Q, const VectorXd& q,
                             const MatrixXd& G, const VectorXd& h,
                             const VectorXd& penalty,
                             const typename Backend<Solver>::Settings&
                                 settings = Backend<Solver>::Tight()) {
  return SolveWith<Solver>(Q, q, MatrixXd(0, q.size()), VectorXd(0), G, h,
                           penalty, settings);
}

// IPM reference on the identical elastic problem: the oracle for every
// "matches" check. The benchmarks repo runs the same checks with vanilla
// PIQP on the expanded (n+p) formulation as the oracle.
inline elastiqp::Solution IpmRef(const MatrixXd& Q, const VectorXd& q,
                                 const MatrixXd& A, const VectorXd& b,
                                 const MatrixXd& G, const VectorXd& h,
                                 const VectorXd& penalty) {
  return SolveWith<elastiqp::ipm::Solver>(Q, q, A, b, G, h, penalty);
}

// Label "<backend>: <cell>" into a fixed buffer.
template <class Solver>
const char* Label(char* buf, size_t n, const char* cell) {
  std::snprintf(buf, n, "%s: %s", Backend<Solver>::name, cell);
  return buf;
}

}  // namespace test_util
