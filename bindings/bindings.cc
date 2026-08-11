// nanobind bindings for ElastiQP

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

#include "elastiqp/ipm.hpp"
#include "elastiqp/pdal.hpp"

namespace nb = nanobind;
using elastiqp::IpmSettings;
using elastiqp::IpmSolver;
using elastiqp::Settings;
using elastiqp::Solution;
using elastiqp::Solver;
using elastiqp::Status;

// Shared by Solver and IpmSolver
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

// Shared setup() overloads (vector and scalar penalty)
template <typename SolverT>
void def_setup(nb::class_<SolverT>& cls) {
  cls.def(
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
      .def("set_penalty", &SolverT::set_penalty, nb::arg("penalty"));
}

// Compiled as elastiqp._core
NB_MODULE(_core, m) {
  m.doc() =
      "ElastiQP: an elastic QP solver with per-constraint L1 "
      "slack relaxation and hard equality constraints.";

  nb::enum_<Status>(m, "Status")
      .value("Unsolved", Status::kUnsolved)
      .value("Solved", Status::kSolved)
      .value("MaxIter", Status::kMaxIter)
      .value("Numerics", Status::kNumerics);

  nb::class_<Solution>(m, "Solution")
      .def_prop_ro("x", [](const Solution& s) { return s.x; })
      .def_prop_ro("t", [](const Solution& s) { return s.t; },
                   "elastic slacks (per-row constraint violations)")
      .def_prop_ro("y", [](const Solution& s) { return s.y; },
                   "equality duals (empty without A, b)")
      .def_prop_ro("s_t", [](const Solution& s) { return s.s_t; },
                   "slacks of t >= 0")
      .def_prop_ro("s_ineq", [](const Solution& s) { return s.s_ineq; },
                   "slacks of G x - t <= h")
      .def_prop_ro("z_t", [](const Solution& s) { return s.z_t; },
                   "duals of t >= 0 (= penalty - z_ineq at a solution)")
      .def_prop_ro("z_ineq", [](const Solution& s) { return s.z_ineq; },
                   "duals of G x - t <= h, in [0, penalty]")
      .def_ro("status", &Solution::status)
      .def_ro("converged", &Solution::converged,
              "1 iff status == Status.Solved")
      .def_ro("iters", &Solution::iters,
              "interior-point iterations (IpmSolver) or inner semismooth "
              "Newton steps (Solver, whose budget is max_outer_iter x "
              "max_iter_in)")
      .def_ro("primal_obj", &Solution::primal_obj)
      .def_ro("primal_res", &Solution::primal_res)
      .def_ro("dual_res", &Solution::dual_res)
      .def_ro("duality_gap", &Solution::duality_gap);

  // Note: many deeper settings stay C++-only
  nb::class_<Settings>(m, "Settings")
      .def(nb::init<>())
      .def_rw("rho", &Settings::rho)
      .def_rw("mu_eq_init", &Settings::mu_eq_init)
      .def_rw("mu_in_init", &Settings::mu_in_init)
      .def_rw("mu_min_eq", &Settings::mu_min_eq)
      .def_rw("mu_min_in", &Settings::mu_min_in)
      .def_rw("mu_update_factor", &Settings::mu_update_factor)
      .def_rw("alpha_bcl", &Settings::alpha_bcl)
      .def_rw("beta_bcl", &Settings::beta_bcl)
      .def_rw("eps_abs", &Settings::eps_abs)
      .def_rw("eps_rel", &Settings::eps_rel)
      .def_rw("check_duality_gap", &Settings::check_duality_gap)
      .def_rw("eps_duality_gap_abs", &Settings::eps_duality_gap_abs)
      .def_rw("eps_duality_gap_rel", &Settings::eps_duality_gap_rel)
      .def_rw("max_outer_iter", &Settings::max_outer_iter)
      .def_rw("max_iter_in", &Settings::max_iter_in)
      .def_rw("max_factor_retries", &Settings::max_factor_retries)
      .def_rw("warm_start", &Settings::warm_start)
      .def_rw("ruiz", &Settings::ruiz)
      .def_rw("ruiz_max_iter", &Settings::ruiz_max_iter)
      .def_rw("ruiz_tol", &Settings::ruiz_tol)
      .def_rw("relax_reg", &Settings::relax_reg)
      .def_rw("relax_warm_start", &Settings::relax_warm_start);

  auto solver_cls =
      nb::class_<Solver>(m, "Solver")
          .def(nb::init<>())
          .def_rw("settings", &Solver::settings)
          .def("solve", [](Solver& s) -> Solution { return s.solve(); })
          .def("solution",
               [](const Solver& s) -> Solution { return s.solution(); })
          .def("factorizations", &Solver::factorizations,
               "KKT factorizations performed by the last solve()")
          .def(
              "relax",
              [](Solver& s, double kappa, double tol, int max_iter)
                  -> Solution { return s.relax(kappa, tol, max_iter); },
              nb::arg("kappa"), nb::arg("tol") = 1e-8,
              nb::arg("max_iter") = 30,
              "Walk the converged solution to the kappa-relaxed central "
              "point (s.z = kappa) for smooth differentiation, via the "
              "log-barrier retraction. Call after solve(); the returned "
              "Solution is the relaxed point, while the solver's own "
              "iterate (used for warm starts) stays at the tight solution.")
          .def(
              "set_warm_start",
              [](Solver& s, const Eigen::VectorXd& x,
                 const Eigen::VectorXd& y, const Eigen::VectorXd& z_ineq,
                 double rho, double mu_eq, double mu_in) {
                s.set_warm_start(x, y, z_ineq, rho, mu_eq, mu_in);
              },
              nb::arg("x"), nb::arg("y"), nb::arg("z_ineq"), nb::kw_only(),
              nb::arg("rho") = 0.0, nb::arg("mu_eq") = 0.0,
              nb::arg("mu_in") = 0.0,
              "Seed the next solve() with an explicit iterate. Pass "
              "y=zeros(0) without equalities.");
  def_setup(solver_cls);
  def_update(solver_cls);

  nb::class_<IpmSettings>(m, "IpmSettings")
      .def(nb::init<>())
      .def_rw("rho_init", &IpmSettings::rho_init)
      .def_rw("delta_init", &IpmSettings::delta_init)
      .def_rw("eps_abs", &IpmSettings::eps_abs)
      .def_rw("eps_rel", &IpmSettings::eps_rel)
      .def_rw("check_duality_gap", &IpmSettings::check_duality_gap)
      .def_rw("eps_duality_gap_abs", &IpmSettings::eps_duality_gap_abs)
      .def_rw("eps_duality_gap_rel", &IpmSettings::eps_duality_gap_rel)
      .def_rw("max_iter", &IpmSettings::max_iter)
      .def_rw("max_factor_retries", &IpmSettings::max_factor_retries)
      .def_rw("tau", &IpmSettings::tau)
      .def_rw("warm_start", &IpmSettings::warm_start)
      .def_rw("warm_start_fraction", &IpmSettings::warm_start_fraction)
      .def_rw("warm_start_min_floor", &IpmSettings::warm_start_min_floor)
      .def_rw("warm_start_max_floor", &IpmSettings::warm_start_max_floor)
      .def_rw("ruiz", &IpmSettings::ruiz)
      .def_rw("ruiz_max_iter", &IpmSettings::ruiz_max_iter)
      .def_rw("ruiz_tol", &IpmSettings::ruiz_tol);

  auto ipm_cls =
      nb::class_<IpmSolver>(m, "IpmSolver")
          .def(nb::init<>())
          .def_rw("settings", &IpmSolver::settings)
          .def("solve", [](IpmSolver& s) -> Solution { return s.solve(); })
          .def(
              "relax",
              [](IpmSolver& s, double kappa, double tol,
                 int max_iter) -> Solution {
                return s.relax(kappa, tol, max_iter);
              },
              nb::arg("kappa"), nb::arg("tol") = 1e-8,
              nb::arg("max_iter") = 30)
          .def("solution",
               [](const IpmSolver& s) -> Solution { return s.solution(); })
          .def(
              "warm_start_from",
              [](IpmSolver& s, const Solution& sol) {
                if (sol.x.size() != s.n() || sol.t.size() != s.p() ||
                    sol.s_t.size() != s.p() || sol.s_ineq.size() != s.p() ||
                    sol.z_t.size() != s.p() || sol.z_ineq.size() != s.p() ||
                    (s.m() > 0 && sol.y.size() != s.m())) {
                  throw std::invalid_argument(
                      "solution dimensions do not match setup()");
                }
                s.warm_start_from(sol);
              },
              nb::arg("solution"),
              "Seed the next solve() from a Solution produced by either "
              "backend");
  def_setup(ipm_cls);
  def_update(ipm_cls);

  auto solve_py = [](const Eigen::MatrixXd& Q, const Eigen::VectorXd& q,
                     const Eigen::MatrixXd& G, const Eigen::VectorXd& h,
                     const Eigen::VectorXd& penalty,
                     const std::optional<Eigen::MatrixXd>& A,
                     const std::optional<Eigen::VectorXd>& b,
                     const std::string& backend, double eps_abs, int max_iter,
                     bool ruiz) {
    if (A.has_value() != b.has_value()) {
      throw std::invalid_argument("A and b must be provided together");
    }
    if (backend == "pdal") {
      Settings settings;
      settings.eps_abs = eps_abs;
      settings.eps_duality_gap_abs = eps_abs;
      settings.max_outer_iter = max_iter;  // BCL rounds on this backend
      settings.ruiz = ruiz;
      return A ? elastiqp::Solve(Q, q, *A, *b, G, h, penalty, settings)
               : elastiqp::Solve(Q, q, G, h, penalty, settings);
    }
    if (backend == "ipm") {
      elastiqp::IpmSettings settings;
      settings.eps_abs = eps_abs;
      settings.eps_duality_gap_abs = eps_abs;
      settings.max_iter = max_iter;
      settings.ruiz = ruiz;
      return A ? elastiqp::IpmSolve(Q, q, *A, *b, G, h, penalty, settings)
               : elastiqp::IpmSolve(Q, q, G, h, penalty, settings);
    }
    throw std::invalid_argument("unknown backend '" + backend +
                                "': expected 'pdal' or 'ipm'");
  };

  const char* solve_doc =
      "Solve the elastic QP  min 0.5 x'Qx + q'x + penalty't  s.t. "
      "A x == b (hard, optional), G x - t <= h, t >= 0 (elastic).\n\n"
      "backend='pdal' (default) uses the primal-dual augmented Lagrangian "
      "solver (elastiqp.Solver); backend='ipm' uses the proximal "
      "interior-point solver (elastiqp.IpmSolver), which is slower but "
      "holds equalities to ~1e-11. Both backends are differentiable: use "
      "the Solver/IpmSolver classes and relax(kappa) for the smoothed "
      "differentiation point. ruiz= enables Ruiz equilibration on either "
      "backend.\n\n"
      "Returns a Solution (see help(elastiqp.Solution) for the fields).";

  m.def(
      "solve",
      [solve_py](const Eigen::MatrixXd& Q, const Eigen::VectorXd& q,
                 const Eigen::MatrixXd& G, const Eigen::VectorXd& h,
                 const Eigen::VectorXd& penalty,
                 const std::optional<Eigen::MatrixXd>& A,
                 const std::optional<Eigen::VectorXd>& b,
                 const std::string& backend, double eps_abs, int max_iter,
                 bool ruiz) {
        return solve_py(Q, q, G, h, penalty, A, b, backend, eps_abs, max_iter,
                        ruiz);
      },
      nb::arg("Q"), nb::arg("q"), nb::arg("G"), nb::arg("h"),
      nb::arg("penalty"), nb::kw_only(), nb::arg("A") = nb::none(),
      nb::arg("b") = nb::none(), nb::arg("backend") = "pdal",
      nb::arg("eps_abs") = 1e-8, nb::arg("max_iter") = 250,
      nb::arg("ruiz") = false, solve_doc);

  m.def(
      "solve",
      [solve_py](const Eigen::MatrixXd& Q, const Eigen::VectorXd& q,
                 const Eigen::MatrixXd& G, const Eigen::VectorXd& h,
                 double penalty, const std::optional<Eigen::MatrixXd>& A,
                 const std::optional<Eigen::VectorXd>& b,
                 const std::string& backend, double eps_abs, int max_iter,
                 bool ruiz) {
        return solve_py(Q, q, G, h,
                        Eigen::VectorXd::Constant(h.size(), penalty), A, b,
                        backend, eps_abs, max_iter, ruiz);
      },
      nb::arg("Q"), nb::arg("q"), nb::arg("G"), nb::arg("h"),
      nb::arg("penalty"), nb::kw_only(), nb::arg("A") = nb::none(),
      nb::arg("b") = nb::none(), nb::arg("backend") = "pdal",
      nb::arg("eps_abs") = 1e-8, nb::arg("max_iter") = 250,
      nb::arg("ruiz") = false, solve_doc);
}
