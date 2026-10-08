/*
 * ElastiQP C interface (dual active-set backend, see das.hpp).
 *
 * Solves
 *
 *   minimize    0.5 x'Q x + q'x + penalty't
 *   subject to  A x = b
 *               G x - t <= h,  t >= 0
 *
 * Dense row-major matrices: M(i, j) = M[i * cols + j].
 * Q is symmetric; only its upper triangle is read.
 * Two-sided constraints and variable bounds use two rows of G (g and -g).
 * Penalties must be > 0; values >= ELASTIQP_INF make the row hard (t = 0).
 * Exit flags: > 0 success, 0 unsolved, < 0 failure.
 *
 * Use elastiqp_quadprog() for one-shot solves, or setup/update/solve/free
 * for repeated solves. Input data is copied; caller buffers can be reused.
 * Allocation-free update/solve coverage: tests/test_das_alloc.cc.
 *
 * Windows DLL users outside CMake: define ELASTIQP_C_SHARED before including.
 */

#ifndef ELASTIQP_C_H
#define ELASTIQP_C_H

#if defined(_WIN32) && defined(ELASTIQP_C_SHARED)
#ifdef ELASTIQP_C_BUILDING
#define ELASTIQP_C_API __declspec(dllexport)
#else
#define ELASTIQP_C_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define ELASTIQP_C_API __attribute__((visibility("default")))
#else
#define ELASTIQP_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Hard-row penalty threshold. */
#define ELASTIQP_INF 1e30

/* Exit flags. */
#define ELASTIQP_SOLVED 1
#define ELASTIQP_UNSOLVED 0     /* no solve yet */
#define ELASTIQP_MAX_ITER (-1)  /* iteration limit reached */
#define ELASTIQP_NUMERICS (-2)  /* numerical failure */
#define ELASTIQP_INFEASIBLE (-3) /* inconsistent equalities or hard rows */
#define ELASTIQP_ERR_INVALID_ARG (-10) /* NULL, bad dimension, value, name */
#define ELASTIQP_ERR_NOT_SOLVED (-11)  /* relax() without a current solve */
#define ELASTIQP_ERR_ALLOC (-12)       /* out of memory */
#define ELASTIQP_ERR_INTERNAL (-13)    /* unexpected C++ exception */

typedef struct {
  int n; /* number of variables, >= 1 */
  int m; /* number of equality constraints (rows of A) */
  int p; /* number of elastic inequality constraints (rows of G) */

  const double* Q;       /* n x n, upper triangle read; NULL: zero (LP) */
  const double* q;       /* n; NULL: zero */
  const double* A;       /* m x n, row-major */
  const double* b;       /* m */
  const double* G;       /* p x n, row-major */
  const double* h;       /* p */
  const double* penalty; /* p, > 0; >= ELASTIQP_INF: hard row */
} ElastiQPProblem;

/*
 * Booleans use 0 / nonzero. Fields are also accessible as named options;
 * elastiqp_option_info() gives their ranges. Out-of-range values are rejected.
 *
 * Tolerances and iteration limits apply on the next solve. Scaling and
 * factorization options (ruiz*, zero_tol, eps_prox, reuse_factorization)
 * apply on refactorization; updating Q forces it. Disabling ruiz retains
 * existing scaling. Explicit warm starts apply even with warm_start = 0.
 */
typedef struct {
  double eps_abs;
  double eps_rel;
  double sing_tol;
  double zero_tol;
  double eps_prox;
  double eta_prox;
  double prox_relaxation;
  int prox_escalations;
  int max_iter;
  int max_outer;
  int warm_start;
  int reuse_factorization;

  int ruiz;
  int ruiz_max_iter;
  double ruiz_tol;
  double ruiz_refresh_ratio;

  int check_eq_consistency;

  double progress_tol;
  int cycle_tol;
  double refactor_tol;

  double relax_reg;
  int relax_factor_retries;
} ElastiQPSettings;

/*
 * Caller-owned output arrays; NULL arrays are skipped.
 * KKT: Q x + q + A'y + G'z = 0, 0 <= z <= penalty, t = max(G x - h, 0).
 * z_t = penalty - z after solve; relax() computes it independently.
 * On API errors, only exitflag is valid; statistics are zeroed.
 */
typedef struct {
  double* x;   /* n */
  double* y;   /* m, equality duals */
  double* z;   /* p, inequality duals */
  double* t;   /* p, elastic slacks */
  double* z_t; /* p, duals of t >= 0 (INFINITY on hard rows) */

  int exitflag;    /* ELASTIQP_SOLVED, ... */
  int iter;        /* inner active-set (or relax Newton) iterations */
  int outer_iter;  /* proximal rounds */
  int n_active;    /* rows with t = 0 and z > 0 */
  int n_saturated; /* rows with t > 0 (z at its penalty) */
  double fval;     /* primal objective, penalty term included */
  double primal_res;
  double dual_res;
  double duality_gap;

  double setup_time; /* seconds; elastiqp_quadprog only */
  double solve_time; /* seconds */
} ElastiQPResult;

/* Opaque solver workspace. */
typedef struct ElastiQPWorkspace ElastiQPWorkspace;

ELASTIQP_C_API void elastiqp_default_settings(ElastiQPSettings* settings);

/* Name of an exit flag ("solved", "max_iter", ..., "invalid_arg"). */
ELASTIQP_C_API const char* elastiqp_status_name(int exitflag);

/* One-shot solve; settings may be NULL (defaults). Returns res->exitflag. */
ELASTIQP_C_API int elastiqp_quadprog(ElastiQPResult* res,
                                     const ElastiQPProblem* qp,
                                     const ElastiQPSettings* settings);

/*
 * Creates a workspace for qp (settings may be NULL). Returns 0 and sets *work,
 * or a negative error and sets *work to NULL.
 */
ELASTIQP_C_API int elastiqp_setup(ElastiQPWorkspace** work,
                                  const ElastiQPProblem* qp,
                                  const ElastiQPSettings* settings);

/* Solves the current problem, warm-started from the previous solve when
 * settings.warm_start is set. Returns res->exitflag. */
ELASTIQP_C_API int elastiqp_solve(ElastiQPResult* res, ElastiQPWorkspace* work);

ELASTIQP_C_API void elastiqp_free(ElastiQPWorkspace* work);

/* Problem dimensions of a workspace; any output may be NULL. */
ELASTIQP_C_API void elastiqp_get_dims(const ElastiQPWorkspace* work, int* n,
                                      int* m, int* p);

ELASTIQP_C_API int elastiqp_get_settings(const ElastiQPWorkspace* work,
                                         ElastiQPSettings* settings);
ELASTIQP_C_API int elastiqp_set_settings(ElastiQPWorkspace* work,
                                         const ElastiQPSettings* settings);

/*
 * Named workspace options, independent of the settings struct layout.
 * Values use double: INT requires an integer, BOOL requires 0 or 1.
 * All values must be in [lower, upper]. Returns 0 or ELASTIQP_ERR_INVALID_ARG;
 * rejected options remain unchanged.
 */
#define ELASTIQP_OPTION_DOUBLE 0
#define ELASTIQP_OPTION_INT 1
#define ELASTIQP_OPTION_BOOL 2

ELASTIQP_C_API int elastiqp_set_option(ElastiQPWorkspace* work,
                                       const char* name, double value);
ELASTIQP_C_API int elastiqp_get_option(const ElastiQPWorkspace* work,
                                       const char* name, double* value);

/* Option count and metadata for 0 <= index < count. Outputs may be NULL.
 * option_info returns 0 or ELASTIQP_ERR_INVALID_ARG; strings are static. */
ELASTIQP_C_API int elastiqp_num_options(void);
ELASTIQP_C_API int elastiqp_option_info(int index, const char** name,
                                        int* type, double* default_value,
                                        double* lower, double* upper,
                                        const char** description);

/*
 * Replace data using setup dimensions and ElastiQPProblem conventions.
 * NULL Q or q means zero. Only changed A/G rows are refreshed.
 * Zero-sized updates do nothing. Returns 0 or a negative error;
 * rejected updates leave the workspace unchanged.
 */
ELASTIQP_C_API int elastiqp_update_Q(ElastiQPWorkspace* work, const double* Q);
ELASTIQP_C_API int elastiqp_update_q(ElastiQPWorkspace* work, const double* q);
ELASTIQP_C_API int elastiqp_update_A(ElastiQPWorkspace* work, const double* A);
ELASTIQP_C_API int elastiqp_update_b(ElastiQPWorkspace* work, const double* b);
ELASTIQP_C_API int elastiqp_update_G(ElastiQPWorkspace* work, const double* G);
ELASTIQP_C_API int elastiqp_update_h(ElastiQPWorkspace* work, const double* h);
ELASTIQP_C_API int elastiqp_update_penalty(ElastiQPWorkspace* work,
                                           const double* penalty);

/*
 * Replace rows [first, first + count) with a count x n row-major block.
 * Cost: O(count * n).
 */
ELASTIQP_C_API int elastiqp_update_A_rows(ElastiQPWorkspace* work, int first,
                                          int count, const double* rows);
ELASTIQP_C_API int elastiqp_update_G_rows(ElastiQPWorkspace* work, int first,
                                          int count, const double* rows);

/*
 * Seed the next solve with (x, y, z). Requires x; NULL y or z means zero.
 */
ELASTIQP_C_API int elastiqp_set_warm_start(ElastiQPWorkspace* work,
                                           const double* x, const double* y,
                                           const double* z);

/*
 * Compute the kappa-relaxed central point for differentiation (kappa > 0).
 * Requires a successful solve on current data, else ELASTIQP_ERR_NOT_SOLVED.
 * Preserves the working set and warm start; p == 0 returns the solution.
 * Suggested: tol = 1e-6, max_iter = 50. Returns res->exitflag.
 */
ELASTIQP_C_API int elastiqp_relax(ElastiQPResult* res, ElastiQPWorkspace* work,
                                  double kappa, double tol, int max_iter);

#ifdef __cplusplus
}
#endif

#endif /* ELASTIQP_C_H */
