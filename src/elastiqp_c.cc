// C interface to the dual active-set backend; see elastiqp/elastiqp_c.h.

#include "elastiqp/elastiqp_c.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

#include "elastiqp/das.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>;
using RowMajorMap = Eigen::Map<const RowMajor>;
using VectorMap = Eigen::Map<const VectorXd>;
namespace das = elastiqp::das;

// Preallocated staging buffers; A/G rows are read directly from caller data.
struct ElastiQPWorkspace {
  das::Solver solver;
  MatrixXd Q;
  VectorXd q, b, h, penalty, x, y, z;
  // Successful solve on current data; required by relax().
  bool solved = false;
};

namespace {

using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point since) {
  return std::chrono::duration<double>(Clock::now() - since).count();
}

int ExitFlag(elastiqp::Status s) {
  switch (s) {
    case elastiqp::Status::kSolved:
      return ELASTIQP_SOLVED;
    case elastiqp::Status::kUnsolved:
      return ELASTIQP_UNSOLVED;
    case elastiqp::Status::kMaxIter:
      return ELASTIQP_MAX_ITER;
    case elastiqp::Status::kNumerics:
      return ELASTIQP_NUMERICS;
    case elastiqp::Status::kInfeasible:
      return ELASTIQP_INFEASIBLE;
  }
  return ELASTIQP_ERR_INTERNAL;
}

// Shared metadata for settings conversion, validation, and named options.
// Add new fields here and in tests/test_c_parity.cc.
struct Option {
  const char* name;
  int type;
  double lower, upper;
  const char* description;
  double das::Settings::*d_cpp;
  double ElastiQPSettings::*d_c;
  int das::Settings::*i_cpp;
  bool das::Settings::*b_cpp;
  int ElastiQPSettings::*i_c;  // INT and BOOL options

  double Get(const das::Settings& s) const {
    if (type == ELASTIQP_OPTION_DOUBLE) return s.*d_cpp;
    if (type == ELASTIQP_OPTION_INT) return s.*i_cpp;
    return s.*b_cpp ? 1.0 : 0.0;
  }
  // value must be Valid().
  void Set(das::Settings& s, double value) const {
    if (type == ELASTIQP_OPTION_DOUBLE) {
      s.*d_cpp = value;
    } else if (type == ELASTIQP_OPTION_INT) {
      s.*i_cpp = static_cast<int>(value);
    } else {
      s.*b_cpp = value != 0.0;
    }
  }
  // C booleans are 0 / nonzero; read them as 0 / 1.
  double Get(const ElastiQPSettings& c) const {
    if (type == ELASTIQP_OPTION_DOUBLE) return c.*d_c;
    if (type == ELASTIQP_OPTION_INT) return c.*i_c;
    return c.*i_c != 0 ? 1.0 : 0.0;
  }
  void Set(ElastiQPSettings& c, double value) const {
    if (type == ELASTIQP_OPTION_DOUBLE) {
      c.*d_c = value;
    } else {
      c.*i_c = static_cast<int>(value);
    }
  }
  bool Valid(double value) const {
    if (!(value >= lower && value <= upper)) return false;  // NaN fails
    return type == ELASTIQP_OPTION_DOUBLE || value == std::floor(value);
  }
};

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kIntMax = std::numeric_limits<int>::max();
constexpr double kPositive = std::numeric_limits<double>::min();

constexpr Option D(const char* name, double das::Settings::*cpp,
                   double ElastiQPSettings::*c, double lower,
                   const char* description) {
  return {name,  ELASTIQP_OPTION_DOUBLE, lower,   kInf,   description,
          cpp,   c,                      nullptr, nullptr, nullptr};
}
constexpr Option I(const char* name, int das::Settings::*cpp,
                   int ElastiQPSettings::*c, const char* description) {
  return {name,    ELASTIQP_OPTION_INT, 0.0, kIntMax, description,
          nullptr, nullptr,             cpp, nullptr, c};
}
constexpr Option B(const char* name, bool das::Settings::*cpp,
                   int ElastiQPSettings::*c, const char* description) {
  return {name,    ELASTIQP_OPTION_BOOL, 0.0,     1.0, description,
          nullptr, nullptr,              nullptr, cpp, c};
}

#define ELASTIQP_D(f, lower, desc) \
  D(#f, &das::Settings::f, &ElastiQPSettings::f, lower, desc)
#define ELASTIQP_I(f, desc) \
  I(#f, &das::Settings::f, &ElastiQPSettings::f, desc)
#define ELASTIQP_B(f, desc) \
  B(#f, &das::Settings::f, &ElastiQPSettings::f, desc)

constexpr Option kOptions[] = {
    ELASTIQP_D(eps_abs, kPositive, "Absolute tolerance."),
    ELASTIQP_D(eps_rel, 0.0, "Relative tolerance."),
    ELASTIQP_D(sing_tol, 0.0,
               "Working-set LDL' pivot below this = dependent row."),
    ELASTIQP_D(zero_tol, 0.0,
               "Min/max Cholesky pivot ratio of Q_s accepted before adding "
               "a prox shift."),
    ELASTIQP_D(eps_prox, 0.0,
               "Initial prox shift (x max|diag Q_s|) when Q_s is not PD; 0 "
               "disables."),
    ELASTIQP_D(eta_prox, 0.0,
               "Outer-loop tolerance on eps*|x - xc|_inf; 0: eps_abs."),
    ELASTIQP_D(prox_relaxation, 0.0,
               "Over-relaxation of the prox center when the working set "
               "stopped changing; <= 1 disables."),
    ELASTIQP_I(prox_escalations,
               "Max 100x prox-shift increases after an inner numerical "
               "failure."),
    ELASTIQP_I(max_iter, "Inner active-set iterations per outer iteration."),
    ELASTIQP_I(max_outer, "Outer (proximal) iterations."),
    ELASTIQP_B(warm_start, "Warm-start from the previous solve."),
    ELASTIQP_B(reuse_factorization,
               "Keep the Cholesky of Q_s across solves when only rows or "
               "vectors changed."),
    ELASTIQP_B(ruiz, "Ruiz equilibration."),
    ELASTIQP_I(ruiz_max_iter, "Ruiz sweeps."),
    ELASTIQP_D(ruiz_tol, 0.0, "Ruiz convergence tolerance."),
    ELASTIQP_D(ruiz_refresh_ratio, 0.0,
               "Re-equilibrate when scaling drift exceeds this; 0: never."),
    ELASTIQP_B(check_eq_consistency,
               "Return infeasible early on inconsistent equalities."),
    ELASTIQP_D(progress_tol, 0.0,
               "Absolute dual-objective increase counted as progress by the "
               "cycle guard."),
    ELASTIQP_I(cycle_tol,
               "Stalled iterations after a removal before repair / "
               "numerics."),
    ELASTIQP_D(refactor_tol, 0.0,
               "At optimality, refactor once if the smallest pivot is below "
               "this."),
    ELASTIQP_D(relax_reg, 0.0, "relax(): Newton regularization."),
    ELASTIQP_I(relax_factor_retries,
               "relax(): x100 regularization retries on a failed "
               "factorization."),
};

#undef ELASTIQP_D
#undef ELASTIQP_I
#undef ELASTIQP_B

constexpr int kNumOptions = sizeof(kOptions) / sizeof(kOptions[0]);

const Option* FindOption(const char* name) {
  if (name == nullptr) return nullptr;
  for (const Option& o : kOptions) {
    if (std::strcmp(o.name, name) == 0) return &o;
  }
  return nullptr;
}

bool ValidSettings(const ElastiQPSettings& c) {
  for (const Option& o : kOptions) {
    if (!o.Valid(o.Get(c))) return false;
  }
  return true;
}

das::Settings ToCpp(const ElastiQPSettings& c) {
  das::Settings s;
  for (const Option& o : kOptions) o.Set(s, o.Get(c));
  return s;
}

ElastiQPSettings ToC(const das::Settings& s) {
  ElastiQPSettings c{};
  for (const Option& o : kOptions) o.Set(c, o.Get(s));
  return c;
}

// Positive penalties only; rejects NaN.
bool ValidPenalty(const double* w, int p) {
  for (int i = 0; i < p; ++i) {
    if (!(w[i] > 0.0)) return false;
  }
  return true;
}

bool Valid(const ElastiQPProblem* qp) {
  if (qp == nullptr || qp->n < 1 || qp->m < 0 || qp->p < 0) return false;
  if (qp->m > 0 && (qp->A == nullptr || qp->b == nullptr)) return false;
  if (qp->p > 0 &&
      (qp->G == nullptr || qp->h == nullptr || qp->penalty == nullptr ||
       !ValidPenalty(qp->penalty, qp->p))) {
    return false;
  }
  return true;
}

// Upper triangle of the row-major Q, mirrored; zero for NULL.
void StageQ(MatrixXd& dst, const double* Q, int n) {
  if (Q) {
    dst = RowMajorMap(Q, n, n).selfadjointView<Eigen::Upper>();
  } else {
    dst.setZero();
  }
}

// Penalties at or above ELASTIQP_INF become infinite (hard rows).
void StagePenalty(VectorXd& dst, const double* w) {
  for (Eigen::Index i = 0; i < dst.size(); ++i) {
    dst[i] = w[i] >= ELASTIQP_INF ? std::numeric_limits<double>::infinity()
                                  : w[i];
  }
}

// Clear statistics on API errors.
int Fail(ElastiQPResult* res, int flag) {
  res->exitflag = flag;
  res->iter = res->outer_iter = res->n_active = res->n_saturated = 0;
  res->fval = res->primal_res = res->dual_res = res->duality_gap = 0.0;
  res->setup_time = res->solve_time = 0.0;
  return flag;
}

// Fills the caller's buffers from sol; returns the exit flag.
int Extract(ElastiQPResult* res, const elastiqp::Solution& sol) {
  using Eigen::Map;
  if (res->x) Map<VectorXd>(res->x, sol.x.size()) = sol.x;
  if (res->y) Map<VectorXd>(res->y, sol.y.size()) = sol.y;
  if (res->z) Map<VectorXd>(res->z, sol.z.size()) = sol.z;
  if (res->t) Map<VectorXd>(res->t, sol.t.size()) = sol.t;
  if (res->z_t) Map<VectorXd>(res->z_t, sol.z_t.size()) = sol.z_t;
  res->exitflag = ExitFlag(sol.status);
  res->iter = sol.iters;
  res->outer_iter = sol.outer_iters;
  res->n_active = sol.n_active;
  res->n_saturated = sol.n_saturated;
  res->fval = sol.primal_obj;
  res->primal_res = sol.primal_res;
  res->dual_res = sol.dual_res;
  res->duality_gap = sol.duality_gap;
  return res->exitflag;
}

// Runs f, mapping C++ exceptions to error codes.
template <class F>
int Guard(F&& f) {
  try {
    return f();
  } catch (const std::bad_alloc&) {
    return ELASTIQP_ERR_ALLOC;
  } catch (...) {
    return ELASTIQP_ERR_INTERNAL;
  }
}

int SetupImpl(ElastiQPWorkspace& w, const ElastiQPProblem& qp,
              const ElastiQPSettings* settings) {
  const int n = qp.n, m = qp.m, p = qp.p;
  w.Q.resize(n, n);
  StageQ(w.Q, qp.Q, n);
  if (qp.q) {
    w.q = VectorMap(qp.q, n);
  } else {
    w.q.setZero(n);
  }
  MatrixXd A(m, n), G(p, n);
  w.b.resize(m);
  if (m > 0) {
    A = RowMajorMap(qp.A, m, n);
    w.b = VectorMap(qp.b, m);
  }
  w.h.resize(p);
  w.penalty.resize(p);
  if (p > 0) {
    G = RowMajorMap(qp.G, p, n);
    w.h = VectorMap(qp.h, p);
    StagePenalty(w.penalty, qp.penalty);
  }
  w.x.resize(n);
  w.y.resize(m);
  w.z.resize(p);
  if (settings) w.solver.settings = ToCpp(*settings);
  w.solver.setup(w.Q, w.q, A, w.b, G, w.h, w.penalty);
  return 0;
}

// Validates a [first, first + count) row block of a matrix with `rows` rows.
bool ValidRows(int first, int count, int rows, const double* data) {
  return first >= 0 && count >= 0 && first <= rows - count &&
         (count == 0 || data != nullptr);
}

}  // namespace

extern "C" {

void elastiqp_default_settings(ElastiQPSettings* settings) {
  if (settings) *settings = ToC(das::Settings{});
}

const char* elastiqp_status_name(int exitflag) {
  switch (exitflag) {
    case ELASTIQP_SOLVED:
      return "solved";
    case ELASTIQP_UNSOLVED:
      return "unsolved";
    case ELASTIQP_MAX_ITER:
      return "max_iter";
    case ELASTIQP_NUMERICS:
      return "numerics";
    case ELASTIQP_INFEASIBLE:
      return "infeasible";
    case ELASTIQP_ERR_INVALID_ARG:
      return "invalid_arg";
    case ELASTIQP_ERR_NOT_SOLVED:
      return "not_solved";
    case ELASTIQP_ERR_ALLOC:
      return "alloc";
    case ELASTIQP_ERR_INTERNAL:
      return "internal";
  }
  return "?";
}

int elastiqp_quadprog(ElastiQPResult* res, const ElastiQPProblem* qp,
                      const ElastiQPSettings* settings) {
  if (res == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  if (!Valid(qp) || (settings && !ValidSettings(*settings))) {
    return Fail(res, ELASTIQP_ERR_INVALID_ARG);
  }
  double setup_time = 0.0, solve_time = 0.0;
  const int flag = Guard([&] {
    ElastiQPWorkspace w;
    const Clock::time_point t0 = Clock::now();
    SetupImpl(w, *qp, settings);
    setup_time = Seconds(t0);
    const Clock::time_point t1 = Clock::now();
    const elastiqp::Solution& sol = w.solver.solve();
    solve_time = Seconds(t1);
    return Extract(res, sol);
  });
  if (flag <= ELASTIQP_ERR_INVALID_ARG) return Fail(res, flag);
  res->setup_time = setup_time;
  res->solve_time = solve_time;
  return flag;
}

int elastiqp_setup(ElastiQPWorkspace** work, const ElastiQPProblem* qp,
                   const ElastiQPSettings* settings) {
  if (work == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  *work = nullptr;
  if (!Valid(qp) || (settings && !ValidSettings(*settings))) {
    return ELASTIQP_ERR_INVALID_ARG;
  }
  ElastiQPWorkspace* w = new (std::nothrow) ElastiQPWorkspace;
  if (w == nullptr) return ELASTIQP_ERR_ALLOC;
  const int ret = Guard([&] { return SetupImpl(*w, *qp, settings); });
  if (ret != 0) {
    delete w;
    return ret;
  }
  *work = w;
  return 0;
}

int elastiqp_solve(ElastiQPResult* res, ElastiQPWorkspace* work) {
  if (res == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  if (work == nullptr) return Fail(res, ELASTIQP_ERR_INVALID_ARG);
  double solve_time = 0.0;
  const int flag = Guard([&] {
    const Clock::time_point t0 = Clock::now();
    const elastiqp::Solution& sol = work->solver.solve();
    solve_time = Seconds(t0);
    return Extract(res, sol);
  });
  work->solved = flag == ELASTIQP_SOLVED;
  if (flag <= ELASTIQP_ERR_INVALID_ARG) return Fail(res, flag);
  res->setup_time = 0.0;
  res->solve_time = solve_time;
  return flag;
}

void elastiqp_free(ElastiQPWorkspace* work) { delete work; }

void elastiqp_get_dims(const ElastiQPWorkspace* work, int* n, int* m, int* p) {
  if (n) *n = work ? work->solver.n() : 0;
  if (m) *m = work ? work->solver.m() : 0;
  if (p) *p = work ? work->solver.p() : 0;
}

int elastiqp_get_settings(const ElastiQPWorkspace* work,
                          ElastiQPSettings* settings) {
  if (work == nullptr || settings == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  *settings = ToC(work->solver.settings);
  return 0;
}

int elastiqp_set_settings(ElastiQPWorkspace* work,
                          const ElastiQPSettings* settings) {
  if (work == nullptr || settings == nullptr || !ValidSettings(*settings)) {
    return ELASTIQP_ERR_INVALID_ARG;
  }
  work->solver.settings = ToCpp(*settings);
  return 0;
}

int elastiqp_set_option(ElastiQPWorkspace* work, const char* name,
                        double value) {
  const Option* o = FindOption(name);
  if (work == nullptr || o == nullptr || !o->Valid(value)) {
    return ELASTIQP_ERR_INVALID_ARG;
  }
  o->Set(work->solver.settings, value);
  return 0;
}

int elastiqp_get_option(const ElastiQPWorkspace* work, const char* name,
                        double* value) {
  const Option* o = FindOption(name);
  if (work == nullptr || o == nullptr || value == nullptr) {
    return ELASTIQP_ERR_INVALID_ARG;
  }
  *value = o->Get(work->solver.settings);
  return 0;
}

int elastiqp_num_options(void) { return kNumOptions; }

int elastiqp_option_info(int index, const char** name, int* type,
                         double* default_value, double* lower, double* upper,
                         const char** description) {
  if (index < 0 || index >= kNumOptions) return ELASTIQP_ERR_INVALID_ARG;
  const Option& o = kOptions[index];
  if (name) *name = o.name;
  if (type) *type = o.type;
  if (default_value) *default_value = o.Get(das::Settings{});
  if (lower) *lower = o.lower;
  if (upper) *upper = o.upper;
  if (description) *description = o.description;
  return 0;
}

int elastiqp_update_Q(ElastiQPWorkspace* work, const double* Q) {
  if (work == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  StageQ(work->Q, Q, work->solver.n());
  work->solver.set_Q(work->Q);
  work->solved = false;
  return 0;
}

int elastiqp_update_q(ElastiQPWorkspace* work, const double* q) {
  if (work == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  if (q) {
    work->q = VectorMap(q, work->solver.n());
  } else {
    work->q.setZero();
  }
  work->solver.set_q(work->q);
  work->solved = false;
  return 0;
}

int elastiqp_update_A(ElastiQPWorkspace* work, const double* A) {
  if (work == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  return elastiqp_update_A_rows(work, 0, work->solver.m(), A);
}

int elastiqp_update_b(ElastiQPWorkspace* work, const double* b) {
  if (work == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  const int m = work->solver.m();
  if (m == 0) return 0;
  if (b == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  work->b = VectorMap(b, m);
  work->solver.set_b(work->b);
  work->solved = false;
  return 0;
}

int elastiqp_update_G(ElastiQPWorkspace* work, const double* G) {
  if (work == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  return elastiqp_update_G_rows(work, 0, work->solver.p(), G);
}

int elastiqp_update_h(ElastiQPWorkspace* work, const double* h) {
  if (work == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  const int p = work->solver.p();
  if (p == 0) return 0;
  if (h == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  work->h = VectorMap(h, p);
  work->solver.set_h(work->h);
  work->solved = false;
  return 0;
}

int elastiqp_update_penalty(ElastiQPWorkspace* work, const double* penalty) {
  if (work == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  const int p = work->solver.p();
  if (p == 0) return 0;
  if (penalty == nullptr || !ValidPenalty(penalty, p)) {
    return ELASTIQP_ERR_INVALID_ARG;
  }
  StagePenalty(work->penalty, penalty);
  work->solver.set_penalty(work->penalty);
  work->solved = false;
  return 0;
}

int elastiqp_update_A_rows(ElastiQPWorkspace* work, int first, int count,
                           const double* rows) {
  if (work == nullptr || !ValidRows(first, count, work->solver.m(), rows)) {
    return ELASTIQP_ERR_INVALID_ARG;
  }
  if (count == 0) return 0;
  work->solver.set_A_rows(first, RowMajorMap(rows, count, work->solver.n()));
  work->solved = false;
  return 0;
}

int elastiqp_update_G_rows(ElastiQPWorkspace* work, int first, int count,
                           const double* rows) {
  if (work == nullptr || !ValidRows(first, count, work->solver.p(), rows)) {
    return ELASTIQP_ERR_INVALID_ARG;
  }
  if (count == 0) return 0;
  work->solver.set_G_rows(first, RowMajorMap(rows, count, work->solver.n()));
  work->solved = false;
  return 0;
}

int elastiqp_set_warm_start(ElastiQPWorkspace* work, const double* x,
                            const double* y, const double* z) {
  if (work == nullptr || x == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  const int n = work->solver.n(), m = work->solver.m(), p = work->solver.p();
  work->x = VectorMap(x, n);
  if (y) {
    work->y = VectorMap(y, m);
  } else {
    work->y.setZero();
  }
  if (z) {
    work->z = VectorMap(z, p);
  } else {
    work->z.setZero();
  }
  work->solver.set_warm_start(work->x, work->y, work->z);
  return 0;
}

int elastiqp_relax(ElastiQPResult* res, ElastiQPWorkspace* work, double kappa,
                   double tol, int max_iter) {
  if (res == nullptr) return ELASTIQP_ERR_INVALID_ARG;
  if (work == nullptr || !(kappa > 0.0) || !(tol > 0.0) || max_iter < 0) {
    return Fail(res, ELASTIQP_ERR_INVALID_ARG);
  }
  if (!work->solved) return Fail(res, ELASTIQP_ERR_NOT_SOLVED);
  double solve_time = 0.0;
  const int flag = Guard([&] {
    const Clock::time_point t0 = Clock::now();
    const elastiqp::Solution& sol = work->solver.relax(kappa, tol, max_iter);
    solve_time = Seconds(t0);
    return Extract(res, sol);
  });
  if (flag <= ELASTIQP_ERR_INVALID_ARG) return Fail(res, flag);
  res->setup_time = 0.0;
  res->solve_time = solve_time;
  return flag;
}

}  // extern "C"
