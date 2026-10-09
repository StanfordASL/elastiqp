/* C API: analytic solutions, updates, warm starts, and validation. */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "elastiqp/elastiqp_c.h"

static int g_all_ok = 1;

static void check(const char* name, int ok) {
  printf("  %-52s %s\n", name, ok ? "OK" : "FAIL");
  g_all_ok &= ok;
}

static int near(const double* a, const double* b, int k, double tol) {
  int i;
  for (i = 0; i < k; ++i) {
    if (!(fabs(a[i] - b[i]) <= tol)) return 0;
  }
  return 1;
}

static ElastiQPSettings tight(void) {
  ElastiQPSettings s;
  elastiqp_default_settings(&s);
  s.eps_abs = 1e-10;
  return s;
}

/* min 0.5|x|^2 - x1 - x2  s.t. x1 + x2 <= 1: x = (0.5, 0.5), z = 0.5. */
static void one_shot(void) {
  const double Q[] = {1, 0, 0, 1}, q[] = {-1, -1};
  const double G[] = {1, 1}, h[] = {1}, w[] = {10};
  const double x_ref[] = {0.5, 0.5}, z_ref[] = {0.5}, t_ref[] = {0};
  double x[2], z[1], t[1];
  ElastiQPProblem qp = {2, 0, 1, Q, q, NULL, NULL, G, h, w};
  ElastiQPSolution res = {x, NULL, z, t};
  ElastiQPSettings s = tight();
  int flag = elastiqp_solve_once(&res, &qp, &s);
  check("one-shot: solved", flag == ELASTIQP_SOLVED && res.status == flag);
  check("one-shot: x, z, t", near(x, x_ref, 2, 1e-8) &&
                                 near(z, z_ref, 1, 1e-8) &&
                                 near(t, t_ref, 1, 1e-12));
  check("one-shot: active row counted", res.n_active == 1 &&
                                            res.n_saturated == 0);
  check("one-shot: objective", fabs(res.primal_obj + 0.75) < 1e-8);
  check("one-shot: default settings (NULL)",
        elastiqp_solve_once(&res, &qp, NULL) == ELASTIQP_SOLVED &&
            near(x, x_ref, 2, 1e-5));
}

/*
 * x <= -1, x >= 1, penalty = 0.5, Q = 1: x = 0, both rows saturated.
 * Making x >= 1 hard gives x = 1.
 */
static void elastic_and_hard(void) {
  const double Q[] = {1}, G[] = {1, -1}, h[] = {-1, -1};
  double w[] = {0.5, 0.5};
  double x[1], z[2], t[2];
  const double t_ref[] = {1, 1}, z_ref[] = {0.5, 0.5};
  const double t_hard[] = {2, 0};
  ElastiQPProblem qp = {1, 0, 2, Q, NULL, NULL, NULL, G, h, w};
  ElastiQPSolution res = {x, NULL, z, t};
  ElastiQPSettings s = tight();
  int flag = elastiqp_solve_once(&res, &qp, &s);
  check("conflict: solved", flag == ELASTIQP_SOLVED);
  check("conflict: x = 0, t = 1, z at penalty",
        fabs(x[0]) < 1e-8 && near(t, t_ref, 2, 1e-8) &&
            near(z, z_ref, 2, 1e-8));
  check("conflict: both rows saturated", res.n_saturated == 2);
  w[1] = INFINITY;
  flag = elastiqp_solve_once(&res, &qp, &s);
  check("hard row: solved", flag == ELASTIQP_SOLVED);
  check("hard row: x = 1, hard slack 0",
        fabs(x[0] - 1) < 1e-8 && near(t, t_hard, 2, 1e-8));
}

/* x1 = 0.2, x2 <= h: solution x2 = min(h, 1). */
static void workspace(void) {
  const double Q[] = {1, 0, 0, 1}, q[] = {-1, -1};
  const double A[] = {1, 0}, b[] = {0.2};
  const double G[] = {0, 1}, w[] = {10};
  double h[] = {0.8};
  double x[2], y[1], z[1], t[1];
  ElastiQPProblem qp = {2, 1, 1, Q, q, A, b, G, h, w};
  ElastiQPSolution res = {x, y, z, t};
  ElastiQPSettings s = tight();
  ElastiQPWorkspace* work = NULL;
  int n = 0, m = 0, p = 0, flag;
  const double x1[] = {0.2, 0.8}, y1[] = {0.8}, z1[] = {0.2};
  const double x2[] = {0.2, 1.0}, z2[] = {0.0};
  const double x3[] = {-0.5, 1.0};

  check("setup: returns 0", elastiqp_setup(&work, &qp, &s) == 0 && work);
  elastiqp_get_dims(work, &n, &m, &p);
  check("setup: dims", n == 2 && m == 1 && p == 1);
  flag = elastiqp_solve(&res, work);
  check("solve: solved", flag == ELASTIQP_SOLVED);
  check("solve: x, y, z", near(x, x1, 2, 1e-8) && near(y, y1, 1, 1e-8) &&
                              near(z, z1, 1, 1e-8));

  h[0] = 2.0; /* workspace retains its copy */
  elastiqp_solve(&res, work);
  check("data copied at setup", near(x, x1, 2, 1e-8));
  check("set_h", elastiqp_set_h(work, h) == 0);
  elastiqp_solve(&res, work);
  check("set_h: constraint released",
        near(x, x2, 2, 1e-8) && near(z, z2, 1, 1e-8));

  {
    const double b2[] = {-0.5};
    elastiqp_set_b(work, b2);
    elastiqp_solve(&res, work);
    check("set_b", res.status == ELASTIQP_SOLVED && near(x, x3, 2, 1e-8));
  }
  {
    const double q2[] = {-1, -3}, x4[] = {-0.5, 2.0};
    elastiqp_set_q(work, q2);
    elastiqp_solve(&res, work);
    check("set_q (row active at h = 2)", near(x, x4, 2, 1e-8));
  }
  {
    /* Penalty 0.5 < marginal cost 1 at x2 = 2: the row gives way. */
    const double w2[] = {0.5}, x5[] = {-0.5, 2.5}, t5[] = {0.5};
    elastiqp_set_penalty(work, w2);
    elastiqp_solve(&res, work);
    check("set_penalty: row saturates",
          near(x, x5, 2, 1e-8) && near(t, t5, 1, 1e-8) && res.n_saturated == 1);
  }
  {
    const double Q2[] = {2, 0, 0, 2}, A2[] = {0, 1}, G2[] = {1, 0};
    const double x6[] = {0.5, -0.5};
    /* min |x|^2 - x1 - 3 x2 s.t. x2 = -0.5, x1 <= 2 (inactive). */
    elastiqp_set_Q(work, Q2);
    elastiqp_set_A(work, A2);
    elastiqp_set_G(work, G2);
    elastiqp_solve(&res, work);
    check("set_Q/A/G", near(x, x6, 2, 1e-8));
  }
  {
    ElastiQPSettings got;
    const double xw[] = {0.5, -0.5};
    elastiqp_get_settings(work, &got);
    check("get_settings", got.eps_abs == 1e-10 && got.warm_start);
    got.max_iter = 0;
    elastiqp_set_settings(work, &got);
    elastiqp_set_warm_start(work, xw, NULL, NULL);
    elastiqp_solve(&res, work);
    check("set_settings: max_iter = 0 fails",
          res.status == ELASTIQP_MAX_ITER && res.status < 0);
    elastiqp_set_settings(work, &s);
    elastiqp_solve(&res, work);
    check("set_settings: restored", res.status == ELASTIQP_SOLVED);
  }
  elastiqp_free(work);
}

/*
 * Row-major layout: G is 2 x 3. min 0.5|x|^2 + q'x with
 * G = [[1, 2, 3], [0, 0, 1]], h = [0, 10] (second row inactive). The
 * minimizer of 0.5|x - c|^2 over g'x <= 0 is c - (g'c / g'g) g.
 */
static void row_major(void) {
  const double Q[] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, q[] = {-1, -1, -1};
  const double G[] = {1, 2, 3, 0, 0, 1}, h[] = {0, 10}, w[] = {100, 100};
  /* c = (1,1,1), g'c = 6, g'g = 14 */
  const double x_ref[] = {1 - 6.0 / 14, 1 - 12.0 / 14, 1 - 18.0 / 14};
  double x[3];
  ElastiQPProblem qp = {3, 0, 2, Q, q, NULL, NULL, G, h, w};
  ElastiQPSolution res = {x, NULL, NULL, NULL};
  ElastiQPSettings s = tight();
  elastiqp_solve_once(&res, &qp, &s);
  check("row-major G",
        res.status == ELASTIQP_SOLVED && near(x, x_ref, 3, 1e-8));
}

/* Only the upper triangle of Q is read: full and upper-only Q agree. */
static void q_upper_triangle(void) {
  const double Qfull[] = {2, 1, 1, 2}, Qupper[] = {2, 1, -99, 2};
  const double q[] = {-1, 0};
  double x1[2], x2[2];
  ElastiQPProblem qp = {2, 0, 0, Qfull, q, NULL, NULL, NULL, NULL, NULL};
  ElastiQPSolution r1 = {x1}, r2 = {x2};
  /* 2 x1 + x2 = 1, x1 + 2 x2 = 0 */
  const double x_ref[] = {2.0 / 3, -1.0 / 3};
  elastiqp_solve_once(&r1, &qp, NULL);
  qp.Q = Qupper;
  elastiqp_solve_once(&r2, &qp, NULL);
  check("Q: full matrix", near(x1, x_ref, 2, 1e-10));
  check("Q: upper triangle only, lower ignored", near(x2, x_ref, 2, 1e-10));
}

/* ELASTIQP_INF enforces a hard row. */
static void inf_convention(void) {
  const double Q[] = {1}, G[] = {1, -1}, h[] = {-1, -1};
  const double w[] = {0.5, ELASTIQP_INF};
  double x[1], t[2], z_t[2];
  ElastiQPProblem qp = {1, 0, 2, Q, NULL, NULL, NULL, G, h, w};
  ElastiQPSolution res = {x, NULL, NULL, t, z_t};
  elastiqp_solve_once(&res, &qp, NULL);
  check("ELASTIQP_INF penalty is a hard row",
        res.status == ELASTIQP_SOLVED && fabs(x[0] - 1) < 1e-8 &&
            fabs(t[1]) < 1e-12 && isinf(z_t[1]));
}

/* Rows of a 3 x 2 G replaced in blocks, against fresh one-shot solves. */
static void row_updates(void) {
  const double Q[] = {1, 0, 0, 1}, q[] = {-2, -2}, h[] = {1, 1, 1};
  const double w[] = {10, 10, 10};
  double G[] = {1, 0, 0, 1, 1, 1};
  const double rows[] = {0.5, 1, 2, 0.5};
  double x[2], x_ref[2];
  ElastiQPProblem qp = {2, 0, 3, Q, q, NULL, NULL, G, h, w};
  ElastiQPSolution res = {x}, ref = {x_ref};
  ElastiQPWorkspace* work = NULL;
  elastiqp_setup(&work, &qp, NULL);
  elastiqp_solve(&res, work);
  check("set_G_rows: rows 1..2", elastiqp_set_G_rows(work, 1, 2, rows) == 0);
  G[2] = 0.5, G[3] = 1, G[4] = 2, G[5] = 0.5;
  elastiqp_solve(&res, work);
  elastiqp_solve_once(&ref, &qp, NULL);
  check("set_G_rows: matches a fresh solve",
        res.status == ELASTIQP_SOLVED && near(x, x_ref, 2, 1e-8));
  check(
      "set_G_rows: out of range rejected",
      elastiqp_set_G_rows(work, 2, 2, rows) == ELASTIQP_ERR_INVALID_ARG &&
          elastiqp_set_G_rows(work, -1, 1, rows) == ELASTIQP_ERR_INVALID_ARG &&
          elastiqp_set_A_rows(work, 0, 1, rows) == ELASTIQP_ERR_INVALID_ARG);
  check("set_G_rows: count 0 is a no-op",
        elastiqp_set_G_rows(work, 3, 0, NULL) == 0);
  elastiqp_free(work);
}

static void relax(void) {
  const double Q[] = {1, 0, 0, 1}, q[] = {-1, -1}, q2[] = {-1, -0.5};
  const double G[] = {1, 1}, h[] = {1}, w[] = {10};
  double x[2], z[1], t[1], z_t[1];
  ElastiQPProblem qp = {2, 0, 1, Q, q, NULL, NULL, G, h, w};
  ElastiQPSolution res = {x, NULL, z, t, z_t};
  ElastiQPWorkspace* work = NULL;
  elastiqp_setup(&work, &qp, NULL);
  check("relax before solve: not_solved",
        elastiqp_relax(&res, work, 1e-3, 1e-8, 50) == ELASTIQP_ERR_NOT_SOLVED);
  elastiqp_solve(&res, work);
  check("solve: z_t = penalty - z", fabs(z_t[0] - (10 - z[0])) < 1e-12);
  check("relax: kappa <= 0 rejected",
        elastiqp_relax(&res, work, 0.0, 1e-8, 50) == ELASTIQP_ERR_INVALID_ARG &&
            res.iters == 0);
  elastiqp_relax(&res, work, 1e-3, 1e-8, 50);
  /* At the relaxed point the inequality slack s = h - Gx + t is O(kappa). */
  check("relax: solved, interior point",
        res.status == ELASTIQP_SOLVED && 1 - x[0] - x[1] + t[0] > 0 &&
            z[0] > 0 && z[0] < 10 && z_t[0] > 0);
  elastiqp_set_q(work, q2);
  check("relax after an update: not_solved",
        elastiqp_relax(&res, work, 1e-3, 1e-8, 50) == ELASTIQP_ERR_NOT_SOLVED);
  elastiqp_solve(&res, work);
  check("relax after re-solve",
        elastiqp_relax(&res, work, 1e-3, 1e-8, 50) == ELASTIQP_SOLVED);
  elastiqp_free(work);
}

/* Options by name: listing, defaults, validation, effect on a solve. */
static void options(void) {
  const double Q[] = {1, 0, 0, 1}, q[] = {-1, -1};
  const double G[] = {1, 1}, h[] = {1}, w[] = {10};
  double x[2], v = 0, def = 0, lo = 0, hi = 0;
  ElastiQPProblem qp = {2, 0, 1, Q, q, NULL, NULL, G, h, w};
  ElastiQPSolution res = {x};
  ElastiQPSettings s;
  ElastiQPWorkspace* work = NULL;
  const char *name = NULL, *name2 = NULL, *desc = NULL;
  int i, j, type = -1, ok = 1, unique = 1, n_opt = elastiqp_num_options();

  elastiqp_setup(&work, &qp, NULL);
  check("options: 22 listed", n_opt == 22);
  for (i = 0; i < n_opt; ++i) {
    ok &= elastiqp_option_info(i, &name, &type, &def, &lo, &hi, &desc) == 0;
    ok &= name != NULL && desc != NULL && desc[0] != '\0';
    ok &= type >= ELASTIQP_OPTION_DOUBLE && type <= ELASTIQP_OPTION_BOOL;
    ok &= lo <= def && def <= hi;
    ok &= elastiqp_get_option(work, name, &v) == 0 && v == def;
    for (j = 0; j < i; ++j) {
      elastiqp_option_info(j, &name2, NULL, NULL, NULL, NULL, NULL);
      unique &= strcmp(name, name2) != 0;
    }
  }
  check("options: info consistent with workspace defaults", ok);
  check("options: names unique", unique);
  check("options: index out of range rejected",
        elastiqp_option_info(n_opt, &name, NULL, NULL, NULL, NULL, NULL) ==
                ELASTIQP_ERR_INVALID_ARG &&
            elastiqp_option_info(-1, NULL, NULL, NULL, NULL, NULL, NULL) ==
                ELASTIQP_ERR_INVALID_ARG);

  check("set_option: unknown name rejected",
        elastiqp_set_option(work, "eps_absolute", 1e-8) ==
                ELASTIQP_ERR_INVALID_ARG &&
            elastiqp_set_option(work, NULL, 1e-8) ==
                ELASTIQP_ERR_INVALID_ARG &&
            elastiqp_get_option(work, "nope", &v) ==
                ELASTIQP_ERR_INVALID_ARG);
  check("set_option: non-integral int rejected",
        elastiqp_set_option(work, "max_iter", 2.5) ==
            ELASTIQP_ERR_INVALID_ARG);
  check("set_option: bool must be 0 or 1",
        elastiqp_set_option(work, "ruiz", 2) == ELASTIQP_ERR_INVALID_ARG &&
            elastiqp_set_option(work, "ruiz", 0) == 0 &&
            elastiqp_get_option(work, "ruiz", &v) == 0 && v == 0);
  check("set_option: out of range / NaN rejected",
        elastiqp_set_option(work, "eps_abs", 0) == ELASTIQP_ERR_INVALID_ARG &&
            elastiqp_set_option(work, "eps_rel", -1) ==
                ELASTIQP_ERR_INVALID_ARG &&
            elastiqp_set_option(work, "max_iter", -1) ==
                ELASTIQP_ERR_INVALID_ARG &&
            elastiqp_set_option(work, "eps_abs", NAN) ==
                ELASTIQP_ERR_INVALID_ARG);
  check("set_option: rejected value leaves the option",
        elastiqp_get_option(work, "eps_abs", &v) == 0 && v == 1e-6);

  elastiqp_set_option(work, "max_iter", 0);
  check("set_option: max_iter = 0 takes effect",
        elastiqp_solve(&res, work) == ELASTIQP_MAX_ITER);
  elastiqp_set_option(work, "max_iter", 10000);
  elastiqp_set_option(work, "eps_abs", 1e-10);
  elastiqp_get_settings(work, &s);
  check("set_option: visible through get_settings",
        s.eps_abs == 1e-10 && s.ruiz == 0 && s.max_iter == 10000);
  check("set_option: solves", elastiqp_solve(&res, work) == ELASTIQP_SOLVED &&
                                  fabs(x[0] - 0.5) < 1e-8);

  s.eps_abs = -1;
  check("set_settings: out-of-range struct rejected",
        elastiqp_set_settings(work, &s) == ELASTIQP_ERR_INVALID_ARG &&
            elastiqp_get_option(work, "eps_abs", &v) == 0 && v == 1e-10);
  elastiqp_free(work);
  check("setup: out-of-range settings rejected",
        elastiqp_setup(&work, &qp, &s) == ELASTIQP_ERR_INVALID_ARG &&
            elastiqp_solve_once(&res, &qp, &s) == ELASTIQP_ERR_INVALID_ARG);
  elastiqp_default_settings(&s);
  s.warm_start = 5; /* struct booleans are 0 / nonzero */
  elastiqp_setup(&work, &qp, &s);
  check("struct bool nonzero accepted, reads as 1",
        work != NULL && elastiqp_get_option(work, "warm_start", &v) == 0 &&
            v == 1);
  elastiqp_free(work);
}

static void invalid_args(void) {
  const double Q[] = {1}, G[] = {1}, h[] = {0};
  double x[1];
  double bad[] = {0};
  ElastiQPProblem qp = {1, 0, 1, Q, NULL, NULL, NULL, G, h, NULL};
  ElastiQPSolution res = {x};
  ElastiQPWorkspace* work = (ElastiQPWorkspace*)&qp;
  res.iters = 7;
  check("missing penalty rejected, stats cleared",
        elastiqp_solve_once(&res, &qp, NULL) == ELASTIQP_ERR_INVALID_ARG &&
            res.status == ELASTIQP_ERR_INVALID_ARG && res.iters == 0);
  check("setup: rejects and clears *work",
        elastiqp_setup(&work, &qp, NULL) == ELASTIQP_ERR_INVALID_ARG &&
            work == NULL);
  qp.penalty = bad;
  check("penalty 0 rejected",
        elastiqp_solve_once(&res, &qp, NULL) == ELASTIQP_ERR_INVALID_ARG);
  bad[0] = -1;
  check("negative penalty rejected",
        elastiqp_solve_once(&res, &qp, NULL) == ELASTIQP_ERR_INVALID_ARG);
  bad[0] = NAN;
  check("NaN penalty rejected",
        elastiqp_solve_once(&res, &qp, NULL) == ELASTIQP_ERR_INVALID_ARG);
  {
    const double w[] = {1}, x_ref[] = {0};
    qp.penalty = w;
    elastiqp_setup(&work, &qp, NULL);
    check("set_penalty: bad penalty rejected",
          elastiqp_set_penalty(work, bad) == ELASTIQP_ERR_INVALID_ARG);
    check("set_penalty: rejected update changed nothing",
          elastiqp_solve(&res, work) == ELASTIQP_SOLVED &&
              near(x, x_ref, 1, 1e-10));
    elastiqp_free(work);
  }
  qp.n = 0;
  check("n = 0 rejected",
        elastiqp_solve_once(&res, &qp, NULL) == ELASTIQP_ERR_INVALID_ARG);
  check("NULL workspace rejected",
        elastiqp_solve(&res, NULL) == ELASTIQP_ERR_INVALID_ARG &&
            elastiqp_set_q(NULL, x) == ELASTIQP_ERR_INVALID_ARG);
  elastiqp_free(NULL);
  check("status names",
        strcmp(elastiqp_status_name(ELASTIQP_SOLVED), "solved") == 0 &&
            strcmp(elastiqp_status_name(ELASTIQP_INFEASIBLE),
                   "infeasible") == 0 &&
            strcmp(elastiqp_status_name(ELASTIQP_ERR_NOT_SOLVED),
                   "not_solved") == 0);
}

int main(void) {
  printf("C interface\n");
  one_shot();
  elastic_and_hard();
  workspace();
  row_major();
  q_upper_triangle();
  inf_convention();
  row_updates();
  relax();
  options();
  invalid_args();
  printf("%s\n", g_all_ok ? "ALL OK" : "FAILURES");
  return g_all_ok ? 0 : 1;
}
