// nanobind bindings for ElastiQP: the three backends (active set, PDAL,
// interior point) behind one module, sharing Status and Solution.

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

#include "elastiqp/elastiqp.hpp"

namespace nb = nanobind;
using elastiqp::Solution;
using elastiqp::Status;

namespace {

enum class Method { kDAS = 0, kPDAL = 1, kIPM = 2 };

Method parse_method(const std::string& method) {
  if (method == "das") return Method::kDAS;
  if (method == "pdal") return Method::kPDAL;
  if (method == "ipm") return Method::kIPM;
  throw std::invalid_argument("method must be one of 'das', 'pdal', 'ipm', got '" +
                              method + "'");
}

// Per-backend mapping of the generic one-shot options (eps_abs, max_iter,
// ruiz) onto the backend's own settings. max_iter counts the backend's
// outer budget: BCL rounds (PDAL), interior-point iterations (IPM),
// active-set iterations (DAS).
inline void apply_options(elastiqp::das::Settings& s, std::optional<double> eps,
                          std::optional<int> max_iter, std::optional<bool> ruiz) {
  if (eps) s.eps_abs = *eps;
  if (max_iter) s.max_iter = *max_iter;
  if (ruiz) s.ruiz = *ruiz;
}
inline void apply_options(elastiqp::pdal::Settings& s, std::optional<double> eps,
                          std::optional<int> max_iter, std::optional<bool> ruiz) {
  if (eps) {
    s.eps_abs = *eps;
    s.eps_duality_gap_abs = *eps;
  }
  if (max_iter) s.max_outer_iter = *max_iter;
  if (ruiz) s.ruiz = *ruiz;
}
inline void apply_options(elastiqp::ipm::Settings& s, std::optional<double> eps,
                          std::optional<int> max_iter, std::optional<bool> ruiz) {
  if (eps) {
    s.eps_abs = *eps;
    s.eps_duality_gap_abs = *eps;
  }
  if (max_iter) s.max_iter = *max_iter;
  if (ruiz) s.ruiz = *ruiz;
}

template <typename SolverT>
Solution solve_with(const Eigen::MatrixXd& Q, const Eigen::VectorXd& q,
                    const std::optional<Eigen::MatrixXd>& A,
                    const std::optional<Eigen::VectorXd>& b,
                    const Eigen::MatrixXd& G, const Eigen::VectorXd& h,
                    const Eigen::VectorXd& penalty,
                    typename SolverT::Settings_t settings,
                    std::optional<double> eps_abs, std::optional<int> max_iter,
                    std::optional<bool> ruiz) {
  apply_options(settings, eps_abs, max_iter, ruiz);
  SolverT solver;
  solver.settings = settings;
  if (A) {
    solver.setup(Q, q, *A, *b, G, h, penalty);
  } else {
    solver.setup(Q, q, G, h, penalty);
  }
  return solver.solve();
}

// Settings_t helper: the backends name their settings type `Settings` in
// their own namespace; give solve_with a uniform way to reach it.
struct DAS : elastiqp::das::Solver { using Settings_t = elastiqp::das::Settings; };
struct PDAL : elastiqp::pdal::Solver { using Settings_t = elastiqp::pdal::Settings; };
struct IPM : elastiqp::ipm::Solver { using Settings_t = elastiqp::ipm::Settings; };

template <typename SolverT>
void def_update(nb::class_<SolverT>& cls) {
  cls.def(
      "update",
      [](SolverT& s, const std::optional<Eigen::MatrixXd>& Q,
         const std::optional<Eigen::VectorXd>& q,
         const std::optional<Eigen::MatrixXd>& A,
         const std::optional<Eigen::VectorXd>& b,
         const std::optional<Eigen::MatrixXd>& G,
         const std::optional<Eigen::VectorXd>& h,
         const std::optional<std::variant<Eigen::VectorXd, double>>& penalty) {
        const Eigen::Index n = s.n(), m = s.m(), p = s.p();
        if (n == 0) {
          throw std::runtime_error("update() requires a prior setup()");
        }
        if ((A || b) && m == 0) {
          throw std::invalid_argument(
              "cannot update A/b: the solver was set up without "
              "equality constraints");
        }
        auto check_mat = [](const char* name, const Eigen::MatrixXd& M,
                            Eigen::Index rows, Eigen::Index cols) {
          if (M.rows() != rows || M.cols() != cols) {
            throw std::invalid_argument(
                std::string(name) + " must be " + std::to_string(rows) + "x" +
                std::to_string(cols) + ", got " + std::to_string(M.rows()) +
                "x" + std::to_string(M.cols()));
          }
        };
        auto check_vec = [](const char* name, const Eigen::VectorXd& v,
                            Eigen::Index size) {
          if (v.size() != size) {
            throw std::invalid_argument(
                std::string(name) + " must have size " + std::to_string(size) +
                ", got " + std::to_string(v.size()));
          }
        };
        if (Q) check_mat("Q", *Q, n, n);
        if (q) check_vec("q", *q, n);
        if (A) check_mat("A", *A, m, n);
        if (b) check_vec("b", *b, m);
        if (G) check_mat("G", *G, p, n);
        if (h) check_vec("h", *h, p);
        if (penalty) {
          if (const auto* v = std::get_if<Eigen::VectorXd>(&*penalty)) {
            check_vec("penalty", *v, p);
          }
        }
        if (Q) s.set_Q(*Q);
        if (q) s.set_q(*q);
        if (A) s.set_A(*A);
        if (b) s.set_b(*b);
        if (G) s.set_G(*G);
        if (h) s.set_h(*h);
        if (penalty) {
          if (const auto* v = std::get_if<Eigen::VectorXd>(&*penalty)) {
            s.set_penalty(*v);
          } else {
            s.set_penalty(
                Eigen::VectorXd::Constant(p, std::get<double>(*penalty)));
          }
        }
      },
      nb::kw_only(), nb::arg("Q") = nb::none(), nb::arg("q") = nb::none(),
      nb::arg("A") = nb::none(), nb::arg("b") = nb::none(),
      nb::arg("G") = nb::none(), nb::arg("h") = nb::none(),
      nb::arg("penalty") = nb::none(),
      "Update a subset of the problem data between solves");
}

// Shared setup() overloads (vector and scalar penalty), the set_* updates,
// and the accessors every backend has.
template <typename SolverT>
void def_common(nb::class_<SolverT>& cls) {
  cls.def(nb::init<>())
      .def_rw("settings", &SolverT::settings)
      .def(
          "setup",
          [](SolverT& s, const Eigen::MatrixXd& Q, const Eigen::VectorXd& q,
             const Eigen::MatrixXd& G, const Eigen::VectorXd& h,
             const Eigen::VectorXd& penalty,
             const std::optional<Eigen::MatrixXd>& A,
             const std::optional<Eigen::VectorXd>& b) {
            if (A.has_value() != b.has_value()) {
              throw std::invalid_argument("A and b must be provided together");
            }
            if (A) {
              s.setup(Q, q, *A, *b, G, h, penalty);
            } else {
              s.setup(Q, q, G, h, penalty);
            }
          },
          nb::arg("Q"), nb::arg("q"), nb::arg("G"), nb::arg("h"),
          nb::arg("penalty"), nb::kw_only(), nb::arg("A") = nb::none(),
          nb::arg("b") = nb::none())
      .def(
          "setup",
          [](SolverT& s, const Eigen::MatrixXd& Q, const Eigen::VectorXd& q,
             const Eigen::MatrixXd& G, const Eigen::VectorXd& h,
             double penalty, const std::optional<Eigen::MatrixXd>& A,
             const std::optional<Eigen::VectorXd>& b) {
            if (A.has_value() != b.has_value()) {
              throw std::invalid_argument("A and b must be provided together");
            }
            if (A) {
              s.setup(Q, q, *A, *b, G, h, penalty);
            } else {
              s.setup(Q, q, G, h, penalty);
            }
          },
          nb::arg("Q"), nb::arg("q"), nb::arg("G"), nb::arg("h"),
          nb::arg("penalty"), nb::kw_only(), nb::arg("A") = nb::none(),
          nb::arg("b") = nb::none())
      .def("set_Q", &SolverT::set_Q, nb::arg("Q"))
      .def("set_q", &SolverT::set_q, nb::arg("q"))
      .def("set_A", &SolverT::set_A, nb::arg("A"))
      .def("set_b", &SolverT::set_b, nb::arg("b"))
      .def("set_G", &SolverT::set_G, nb::arg("G"))
      .def("set_h", &SolverT::set_h, nb::arg("h"))
      .def("set_penalty", &SolverT::set_penalty, nb::arg("penalty"))
      .def("n", [](const SolverT& s) { return static_cast<int>(s.n()); })
      .def("m", [](const SolverT& s) { return static_cast<int>(s.m()); })
      .def("p", [](const SolverT& s) { return static_cast<int>(s.p()); })
      .def("solve", [](SolverT& s) -> Solution { return s.solve(); })
      .def("eq_infeasibility", &SolverT::eq_infeasibility,
           "certified lower bound on the reachable ||Ax - b|| "
           "(0 when consistent or unchecked)");
  def_update(cls);
}

// relax() for the backends that have it (PDAL, IPM).
template <typename SolverT>
void def_relax(nb::class_<SolverT>& cls, int default_max_iter) {
  cls.def(
      "relax",
      [](SolverT& s, double kappa, double tol, int max_iter) -> Solution {
        return s.relax(kappa, tol, max_iter);
      },
      nb::arg("kappa"), nb::arg("tol") = 1e-6,
      nb::arg("max_iter") = default_max_iter,
      "Walk the converged solution to the kappa-relaxed central point "
      "(s.z = kappa) for smooth differentiation. Call after solve(); the "
      "returned Solution is the relaxed point.");
}

}  // namespace

// Compiled as elastiqp._core
NB_MODULE(_core, m) {
  m.doc() =
      "ElastiQP: an elastic QP solver with per-constraint L1 slack "
      "relaxation and hard equality constraints. Three backends, each a "
      "submodule with its own Settings and Solver mirroring the C++ "
      "namespaces: das (dual active set, the default), pdal (primal-dual "
      "augmented Lagrangian) and ipm (interior point).";
  auto das = m.def_submodule("das", "Dual active-set backend (default)");
  auto pdal = m.def_submodule("pdal", "Primal-dual augmented Lagrangian backend");
  auto ipm = m.def_submodule("ipm", "Proximal interior-point backend");

  nb::enum_<Status>(m, "Status")
      .value("Unsolved", Status::kUnsolved)
      .value("Solved", Status::kSolved)
      .value("MaxIter", Status::kMaxIter)
      .value("Numerics", Status::kNumerics)
      .value("Infeasible", Status::kInfeasible);

  nb::class_<Solution>(m, "Solution")
      .def_prop_ro("x", [](const Solution& s) { return s.x; })
      .def_prop_ro("t", [](const Solution& s) { return s.t; },
                   "elastic slacks (per-row constraint violations)")
      .def_prop_ro("y", [](const Solution& s) { return s.y; },
                   "equality duals (empty without A, b)")
      .def_prop_ro("z", [](const Solution& s) { return s.z; },
                   "inequality duals of G x - t <= h, in [0, penalty]")
      .def_prop_ro("s_t", [](const Solution& s) { return s.s_t; },
                   "slacks of t >= 0 (= t at a solution)")
      .def_prop_ro("s_ineq", [](const Solution& s) { return s.s_ineq; },
                   "slacks of G x - t <= h")
      .def_prop_ro("z_t", [](const Solution& s) { return s.z_t; },
                   "duals of t >= 0 (= penalty - z at a solution)")
      .def_ro("status", &Solution::status)
      .def_ro("converged", &Solution::converged,
              "1 iff status == Status.Solved")
      .def_ro("iters", &Solution::iters,
              "backend's inner iterations: semismooth Newton steps (PDAL), "
              "interior-point iterations (IPM), working-set changes (DAS)")
      .def_ro("outer_iters", &Solution::outer_iters,
              "BCL rounds (PDAL), proximal-point rounds (AS), 0 (IPM)")
      .def_ro("n_active", &Solution::n_active,
              "inequality rows with 0 < z < penalty")
      .def_ro("n_saturated", &Solution::n_saturated,
              "inequality rows at z = penalty (violated, t > 0)")
      .def_ro("primal_obj", &Solution::primal_obj)
      .def_ro("primal_res", &Solution::primal_res)
      .def_ro("dual_res", &Solution::dual_res)
      .def_ro("duality_gap", &Solution::duality_gap);

  // --- active set --------------------------------------------------------
  {
    using S = elastiqp::das::Settings;
    nb::class_<S>(das, "Settings")
        .def(nb::init<>())
        .def_rw("eps_abs", &S::eps_abs)
        .def_rw("eps_rel", &S::eps_rel)
        .def_rw("sing_tol", &S::sing_tol)
        .def_rw("zero_tol", &S::zero_tol)
        .def_rw("eps_prox", &S::eps_prox)
        .def_rw("eta_prox", &S::eta_prox)
        .def_rw("prox_relaxation", &S::prox_relaxation)
        .def_rw("max_iter", &S::max_iter)
        .def_rw("max_outer", &S::max_outer)
        .def_rw("warm_start", &S::warm_start)
        .def_rw("reuse_factorization", &S::reuse_factorization)
        .def_rw("ruiz", &S::ruiz)
        .def_rw("ruiz_max_iter", &S::ruiz_max_iter)
        .def_rw("ruiz_tol", &S::ruiz_tol)
        .def_rw("ruiz_refresh_ratio", &S::ruiz_refresh_ratio)
        .def_rw("check_eq_consistency", &S::check_eq_consistency)
        .def_rw("progress_tol", &S::progress_tol)
        .def_rw("cycle_tol", &S::cycle_tol)
        .def_rw("refactor_tol", &S::refactor_tol);

    using Sv = elastiqp::das::Solver;
    nb::enum_<Sv::RowState>(das, "RowState")
        .value("Inactive", Sv::RowState::kInactive)
        .value("Active", Sv::RowState::kActive)
        .value("Saturated", Sv::RowState::kSaturated)
        .value("Equality", Sv::RowState::kEquality)
        .value("Dropped", Sv::RowState::kDropped);
    auto cls = nb::class_<Sv>(das, "Solver",
                              "Dual active-set backend (default)");
    def_common(cls);
    cls.def("row_state", &Sv::row_state, nb::arg("i"),
            "three-state classification of inequality row i")
        .def("proximal", &Sv::proximal,
             "True if Q was shifted (proximal-point outer loop active)")
        .def("prox_eps", &Sv::prox_eps, "proximal shift added to Q (0 if none)")
        .def("scaling_drift", &Sv::scaling_drift,
             "largest factor a scaled column/row max-norm is off from 1")
        .def("rescaled", &Sv::rescaled,
             "Ruiz factors were recomputed during the last solve()")
        .def("rows_updated", &Sv::rows_updated,
             "rows of [A; G] re-solved against R in the last solve()")
        .def("refactored", &Sv::refactored,
             "the Cholesky of Q was recomputed in the last solve()")
        .def("refactors", &Sv::refactors,
             "cycle-guard / pivot working-set refactorizations in the last solve()");
  }

  // --- PDAL ----------------------------------------------------------------
  {
    using S = elastiqp::pdal::Settings;
    nb::class_<S>(pdal, "Settings")
        .def(nb::init<>())
        .def_rw("eps_abs", &S::eps_abs)
        .def_rw("eps_rel", &S::eps_rel)
        .def_rw("check_duality_gap", &S::check_duality_gap)
        .def_rw("eps_duality_gap_abs", &S::eps_duality_gap_abs)
        .def_rw("eps_duality_gap_rel", &S::eps_duality_gap_rel)
        .def_rw("max_factor_retries", &S::max_factor_retries)
        .def_rw("incremental_updates", &S::incremental_updates)
        .def_rw("incremental_update_budget", &S::incremental_update_budget)
        .def_rw("incremental_update_max_flips", &S::incremental_update_max_flips)
        .def_rw("check_eq_consistency", &S::check_eq_consistency)
        .def_rw("warm_start", &S::warm_start)
        .def_rw("max_outer_iter", &S::max_outer_iter)
        .def_rw("max_iter_in", &S::max_iter_in)
        .def_rw("rho", &S::rho)
        .def_rw("mu_eq_init", &S::mu_eq_init)
        .def_rw("mu_in_init", &S::mu_in_init)
        .def_rw("mu_min_eq", &S::mu_min_eq)
        .def_rw("mu_min_in", &S::mu_min_in)
        .def_rw("mu_update_factor", &S::mu_update_factor)
        .def_rw("alpha_bcl", &S::alpha_bcl)
        .def_rw("beta_bcl", &S::beta_bcl)
        .def_rw("bcl_split", &S::bcl_split)
        .def_rw("bcl_saturation_jump", &S::bcl_saturation_jump)
        .def_rw("bcl_release_jump", &S::bcl_release_jump)
        .def_rw("bcl_release_jump_horizon", &S::bcl_release_jump_horizon)
        .def_rw("bcl_warm_eta", &S::bcl_warm_eta)
        .def_rw("cold_reset_mu", &S::cold_reset_mu)
        .def_rw("cold_reset_threshold", &S::cold_reset_threshold)
        .def_rw("cold_reset_residual", &S::cold_reset_residual)
        .def_rw("cold_reset_limit", &S::cold_reset_limit)
        .def_rw("safe_guard", &S::safe_guard)
        .def_rw("ruiz", &S::ruiz)
        .def_rw("ruiz_max_iter", &S::ruiz_max_iter)
        .def_rw("ruiz_tol", &S::ruiz_tol)
        .def_rw("ruiz_refresh_ratio", &S::ruiz_refresh_ratio)
        .def_rw("relax_reg", &S::relax_reg)
        .def_rw("relax_warm_budget", &S::relax_warm_budget)
        .def_rw("relax_warm_flip_tol", &S::relax_warm_flip_tol);

    using Sv = elastiqp::pdal::Solver;
    auto cls = nb::class_<Sv>(pdal, "Solver",
                              "Primal-dual augmented Lagrangian backend");
    def_common(cls);
    cls.def("solution", [](const Sv& s) -> Solution { return s.solution(); },
            "the last solve() / relax() certificate")
        .def("factorizations", &Sv::factorizations,
            "KKT factorizations performed by the last solve()")
        .def("cold_resets", &Sv::cold_resets,
             "BCL cold resets performed by the last solve()")
        .def("reequilibrate", &Sv::reequilibrate,
             "Recompute the Ruiz scaling for the current matrices and "
             "rescale the warm-start state in place (no-op with ruiz "
             "off). solve() does this automatically when the drift "
             "exceeds settings.ruiz_refresh_ratio.")
        .def("scaling_drift", &Sv::scaling_drift,
             "Largest factor by which a scaled column/row max-norm has "
             "drifted from 1 since the last equilibration (1 = none)")
        .def(
            "relax",
            [](Sv& s, double kappa, double tol, int max_iter, bool warm)
                -> Solution { return s.relax(kappa, tol, max_iter, warm); },
            nb::arg("kappa"), nb::arg("tol") = 1e-6, nb::arg("max_iter") = 50,
            nb::arg("warm") = true,
            "Walk the converged solution to the kappa-relaxed central "
            "point (s.z = kappa) for smooth differentiation, via the "
            "log-barrier retraction. Call after solve(); the returned "
            "Solution is the relaxed point, while the solver's own "
            "iterate (used for warm starts) stays at the tight solution. "
            "With warm=True (default), repeated calls on a persistent "
            "solver continue from the previous relaxed point, falling "
            "back to the retraction start automatically if the warm run "
            "does not converge.")
        .def(
            "set_warm_start",
            [](Sv& s, const Eigen::VectorXd& x, const Eigen::VectorXd& y,
               const Eigen::VectorXd& z, double rho, double mu_eq,
               double mu_in) { s.set_warm_start(x, y, z, rho, mu_eq, mu_in); },
            nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::kw_only(),
            nb::arg("rho") = 0.0, nb::arg("mu_eq") = 0.0,
            nb::arg("mu_in") = 0.0,
            "Seed the next solve() with an explicit iterate (x, y, z). Pass "
            "y=zeros(0) without equalities.");
  }

  // --- interior point ------------------------------------------------------
  {
    using S = elastiqp::ipm::Settings;
    nb::class_<S>(ipm, "Settings")
        .def(nb::init<>())
        .def_rw("eps_abs", &S::eps_abs)
        .def_rw("eps_rel", &S::eps_rel)
        .def_rw("check_duality_gap", &S::check_duality_gap)
        .def_rw("eps_duality_gap_abs", &S::eps_duality_gap_abs)
        .def_rw("eps_duality_gap_rel", &S::eps_duality_gap_rel)
        .def_rw("max_factor_retries", &S::max_factor_retries)
        .def_rw("warm_start", &S::warm_start)
        .def_rw("check_eq_consistency", &S::check_eq_consistency)
        .def_rw("max_iter", &S::max_iter)
        .def_rw("rho_init", &S::rho_init)
        .def_rw("delta_init", &S::delta_init)
        .def_rw("infeasibility_threshold", &S::infeasibility_threshold)
        .def_rw("reg_lower_limit", &S::reg_lower_limit)
        .def_rw("reg_finetune_lower_limit", &S::reg_finetune_lower_limit)
        .def_rw("reg_finetune_primal_update_threshold",
                &S::reg_finetune_primal_update_threshold)
        .def_rw("reg_finetune_dual_update_threshold",
                &S::reg_finetune_dual_update_threshold)
        .def_rw("tau", &S::tau)
        .def_rw("warm_start_fraction", &S::warm_start_fraction)
        .def_rw("warm_start_min_floor", &S::warm_start_min_floor)
        .def_rw("warm_start_max_floor", &S::warm_start_max_floor)
        .def_rw("ruiz", &S::ruiz)
        .def_rw("ruiz_max_iter", &S::ruiz_max_iter)
        .def_rw("ruiz_tol", &S::ruiz_tol);

    using Sv = elastiqp::ipm::Solver;
    auto cls = nb::class_<Sv>(ipm, "Solver",
                              "Proximal interior-point backend (PIQP-style)");
    def_common(cls);
    def_relax(cls, 30);
    cls.def("solution", [](const Sv& s) -> Solution { return s.solution(); },
            "the last solve() / relax() certificate")
        .def(
           "set_warm_start",
           [](Sv& s, const Eigen::VectorXd& x, const Eigen::VectorXd& t,
              const Eigen::VectorXd& y, const Eigen::VectorXd& s_t,
              const Eigen::VectorXd& s_ineq, const Eigen::VectorXd& z_t,
              const Eigen::VectorXd& z, double rho, double delta) {
             s.set_warm_start(x, t, y, s_t, s_ineq, z_t, z, rho, delta);
           },
           nb::arg("x"), nb::arg("t"), nb::arg("y"), nb::arg("s_t"),
           nb::arg("s_ineq"), nb::arg("z_t"), nb::arg("z"), nb::kw_only(),
           nb::arg("rho") = 0.0, nb::arg("delta") = 0.0,
           "Seed the next solve() with an explicit interior iterate, used "
           "exactly as given (slacks and duals must be strictly positive).")
        .def("warm_start_from", &Sv::warm_start_from, nb::arg("solution"),
             "Seed the next solve() from any backend's Solution, applying "
             "the interior-point boundary floor.");
  }

  // --- one-shot solve ------------------------------------------------------
  auto solve_py = [](const Eigen::MatrixXd& Q, const Eigen::VectorXd& q,
                     const Eigen::MatrixXd& G, const Eigen::VectorXd& h,
                     const Eigen::VectorXd& penalty,
                     const std::optional<Eigen::MatrixXd>& A,
                     const std::optional<Eigen::VectorXd>& b,
                     const std::string& method, std::optional<double> eps_abs,
                     std::optional<int> max_iter, std::optional<bool> ruiz,
                     nb::object settings) -> Solution {
    if (A.has_value() != b.has_value()) {
      throw std::invalid_argument("A and b must be provided together");
    }
    auto pick = [&](auto default_settings) {
      using S = decltype(default_settings);
      if (settings.is_none()) return default_settings;
      if (!nb::isinstance<S>(settings)) {
        throw std::invalid_argument(
            "settings does not match method='" + method + "'");
      }
      return nb::cast<S>(settings);
    };
    switch (parse_method(method)) {
      case Method::kDAS:
        return solve_with<DAS>(Q, q, A, b, G, h, penalty,
                              pick(elastiqp::das::Settings{}), eps_abs,
                              max_iter, ruiz);
      case Method::kPDAL:
        return solve_with<PDAL>(Q, q, A, b, G, h, penalty,
                                pick(elastiqp::pdal::Settings{}), eps_abs,
                                max_iter, ruiz);
      case Method::kIPM:
        return solve_with<IPM>(Q, q, A, b, G, h, penalty,
                               pick(elastiqp::ipm::Settings{}), eps_abs,
                               max_iter, ruiz);
    }
    throw std::logic_error("unreachable");
  };

  const char* solve_doc =
      "Solve the elastic QP  min 0.5 x'Qx + q'x + penalty't  s.t. "
      "A x == b (hard, optional), G x - t <= h, t >= 0 (elastic).\n\n"
      "method selects the backend: 'das' (dual active set, default), 'pdal' "
      "(primal-dual augmented Lagrangian) or 'ipm' (interior point). "
      "eps_abs / max_iter / ruiz override the backend's defaults when "
      "given (max_iter counts BCL rounds for pdal, interior-point "
      "iterations for ipm, active-set iterations for das); settings= "
      "passes a full das.Settings / pdal.Settings / ipm.Settings object.\n\n"
      "For differentiation, use pdal.Solver or ipm.Solver and relax(kappa) "
      "for the smoothed differentiation point.\n\n"
      "Returns a Solution (see help(elastiqp.Solution) for the fields).";

  m.def(
      "solve",
      [solve_py](const Eigen::MatrixXd& Q, const Eigen::VectorXd& q,
                 const Eigen::MatrixXd& G, const Eigen::VectorXd& h,
                 const Eigen::VectorXd& penalty,
                 const std::optional<Eigen::MatrixXd>& A,
                 const std::optional<Eigen::VectorXd>& b,
                 const std::string& method, std::optional<double> eps_abs,
                 std::optional<int> max_iter, std::optional<bool> ruiz,
                 nb::object settings) {
        return solve_py(Q, q, G, h, penalty, A, b, method, eps_abs, max_iter,
                        ruiz, settings);
      },
      nb::arg("Q"), nb::arg("q"), nb::arg("G"), nb::arg("h"),
      nb::arg("penalty"), nb::kw_only(), nb::arg("A") = nb::none(),
      nb::arg("b") = nb::none(), nb::arg("method") = "das",
      nb::arg("eps_abs") = nb::none(), nb::arg("max_iter") = nb::none(),
      nb::arg("ruiz") = nb::none(), nb::arg("settings") = nb::none(),
      solve_doc);

  m.def(
      "solve",
      [solve_py](const Eigen::MatrixXd& Q, const Eigen::VectorXd& q,
                 const Eigen::MatrixXd& G, const Eigen::VectorXd& h,
                 double penalty, const std::optional<Eigen::MatrixXd>& A,
                 const std::optional<Eigen::VectorXd>& b,
                 const std::string& method, std::optional<double> eps_abs,
                 std::optional<int> max_iter, std::optional<bool> ruiz,
                 nb::object settings) {
        return solve_py(Q, q, G, h,
                        Eigen::VectorXd::Constant(h.size(), penalty), A, b,
                        method, eps_abs, max_iter, ruiz, settings);
      },
      nb::arg("Q"), nb::arg("q"), nb::arg("G"), nb::arg("h"),
      nb::arg("penalty"), nb::kw_only(), nb::arg("A") = nb::none(),
      nb::arg("b") = nb::none(), nb::arg("method") = "das",
      nb::arg("eps_abs") = nb::none(), nb::arg("max_iter") = nb::none(),
      nb::arg("ruiz") = nb::none(), nb::arg("settings") = nb::none(),
      solve_doc);

  // --- differentiation primitives (used by elastiqp.torch) -----------------
  //
  // Mirror of the JAX FFI handler (bindings/jax_ffi.cc): one cold solve, plus
  // the kappa-relaxed central point when target_kappa > 0, in a single call
  // so an autograd framework pays one crossing per solve. Returns the tuple
  //   (x, t, y, z_t, z, xr, tr, yr, z_t_r, z_r, info)
  // with info = [converged, iters, relax_converged] as float64. method is
  // 'das', 'pdal' or 'ipm'; the relaxed block is only computed for pdal /
  // ipm (the active-set backend has no relax(); with target_kappa > 0 it
  // raises), otherwise it is a copy of the tight solution.
  m.def(
      "_solve_relaxed",
      [](const Eigen::MatrixXd& Q, const Eigen::VectorXd& q,
         const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
         const Eigen::MatrixXd& G, const Eigen::VectorXd& h,
         const Eigen::VectorXd& penalty, double eps_abs, int max_iter,
         bool ruiz, double target_kappa, const std::string& method) {
        const Method mth = parse_method(method);
        const bool want_relax = target_kappa > 0 && h.size() > 0;
        if (want_relax && mth == Method::kDAS) {
          throw std::invalid_argument(
              "method='das' has no kappa relaxation (not differentiable); "
              "use method='pdal' or 'ipm' for gradients");
        }
        Solution sol, rsol;
        // Same relax tolerance rule as the JAX FFI: decoupled from eps_abs,
        // since the VJP's accuracy is set by this residual.
        const double rtol = std::min(eps_abs, 1e-6);
        if (mth == Method::kDAS) {
          elastiqp::das::Solver solver;
          apply_options(solver.settings, eps_abs, max_iter, ruiz);
          solver.setup(Q, q, A, b, G, h, penalty);
          sol = solver.solve();
          rsol = sol;
        } else if (mth == Method::kPDAL) {
          elastiqp::pdal::Solver solver;
          apply_options(solver.settings, eps_abs, max_iter, ruiz);
          solver.setup(Q, q, A, b, G, h, penalty);
          sol = solver.solve();
          rsol = want_relax ? solver.relax(target_kappa, rtol, 50) : sol;
        } else {
          elastiqp::ipm::Solver solver;
          apply_options(solver.settings, eps_abs, max_iter, ruiz);
          solver.setup(Q, q, A, b, G, h, penalty);
          sol = solver.solve();
          rsol = want_relax ? solver.relax(target_kappa, rtol, 50) : sol;
        }
        Eigen::Vector3d info(static_cast<double>(sol.converged),
                             static_cast<double>(sol.iters),
                             static_cast<double>(rsol.converged));
        return std::make_tuple(sol.x, sol.t, sol.y, sol.z_t, sol.z, rsol.x,
                               rsol.t, rsol.y, rsol.z_t, rsol.z,
                               Eigen::VectorXd(info));
      },
      nb::arg("Q"), nb::arg("q"), nb::arg("A"), nb::arg("b"), nb::arg("G"),
      nb::arg("h"), nb::arg("penalty"), nb::arg("eps_abs"),
      nb::arg("max_iter"), nb::arg("ruiz"), nb::arg("target_kappa"),
      nb::arg("method") = "pdal");

  // Reverse-mode implicit differentiation of the elastic KKT system at a
  // relaxed point (include/elastiqp/kkt_vjp.hpp). Pass the relaxed block of
  // _solve_relaxed as (x, t, y, z_t, z) and the loss cotangents; empty
  // cotangent vectors are treated as zero. Returns
  //   (Q_bar, q_bar, A_bar, b_bar, G_bar, h_bar, penalty_bar).
  m.def(
      "_kkt_vjp",
      [](const Eigen::MatrixXd& Q, const Eigen::MatrixXd& A,
         const Eigen::MatrixXd& G, const Eigen::VectorXd& h,
         const Eigen::VectorXd& x, const Eigen::VectorXd& t,
         const Eigen::VectorXd& y, const Eigen::VectorXd& z_t,
         const Eigen::VectorXd& z, const Eigen::VectorXd& ct_x,
         const Eigen::VectorXd& ct_t, const Eigen::VectorXd& ct_y,
         const Eigen::VectorXd& ct_z_t, const Eigen::VectorXd& ct_z) {
        Solution sol;
        sol.x = x;
        sol.t = t;
        sol.y = y;
        sol.z_t = z_t;
        sol.z = z;
        elastiqp::Cotangents ct{ct_x, ct_t, ct_y, ct_z_t, ct_z};
        const elastiqp::DataGrads g = elastiqp::Vjp(Q, A, G, h, sol, ct);
        return std::make_tuple(g.Q, g.q, g.A, g.b, g.G, g.h, g.penalty);
      },
      nb::arg("Q"), nb::arg("A"), nb::arg("G"), nb::arg("h"), nb::arg("x"),
      nb::arg("t"), nb::arg("y"), nb::arg("z_t"), nb::arg("z"),
      nb::arg("ct_x"), nb::arg("ct_t"), nb::arg("ct_y"), nb::arg("ct_z_t"),
      nb::arg("ct_z"));
}
