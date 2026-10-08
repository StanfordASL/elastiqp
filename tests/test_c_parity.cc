// C/C++ parity for solves, updates, warm starts, and relaxation.

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "elastiqp/elastiqp.hpp"
#include "elastiqp/elastiqp_c.h"
#include "problem_gen.hpp"
#include "test_util.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;
using test_util::Check;
namespace das = elastiqp::das;

namespace {

using RowMajor =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Row-major copies of the problem data, kept alive for the C calls.
struct CData {
  RowMajor Q, A, G;
  VectorXd q, b, h, w;
  ElastiQPProblem Problem() const {
    return {static_cast<int>(q.size()),
            static_cast<int>(b.size()),
            static_cast<int>(h.size()),
            Q.data(),
            q.data(),
            A.data(),
            b.data(),
            G.data(),
            h.data(),
            w.data()};
  }
};

// Ensure both APIs read a bitwise-identical symmetric Q.
void Symmetrize(MatrixXd& Q) { Q = (0.5 * (Q + Q.transpose())).eval(); }

CData ToC(const QPData& qp, const VectorXd& w) {
  return {qp.Q, qp.A, qp.G, qp.q, qp.b, qp.h, w};
}

struct CSolution {
  VectorXd x, y, z, t, z_t;
  ElastiQPResult res{};
  CSolution(int n, int m, int p) : x(n), y(m), z(p), t(p), z_t(p) {
    res.x = x.data();
    res.y = y.data();
    res.z = z.data();
    res.t = t.data();
    res.z_t = z_t.data();
  }
};

// Largest difference between the C and C++ results (0 when identical).
double Diff(const CSolution& c, const elastiqp::Solution& s) {
  double d = 0.0;
  d = std::max(d, test_util::InfNorm(c.x - s.x));
  d = std::max(d, test_util::InfNorm(c.y - s.y));
  d = std::max(d, test_util::InfNorm(c.z - s.z));
  d = std::max(d, test_util::InfNorm(c.t - s.t));
  for (Eigen::Index i = 0; i < s.z_t.size(); ++i) {  // inf on hard rows
    if (c.z_t[i] != s.z_t[i]) d = std::max(d, std::abs(c.z_t[i] - s.z_t[i]));
  }
  const bool same_meta = std::strcmp(elastiqp_status_name(c.res.exitflag),
                                     elastiqp::status_name(s.status)) == 0 &&
                         c.res.iter == s.iters &&
                         c.res.outer_iter == s.outer_iters &&
                         c.res.n_active == s.n_active &&
                         c.res.n_saturated == s.n_saturated &&
                         c.res.fval == s.primal_obj;
  return same_meta ? d : std::max(d, 1.0);
}

void OneShot(std::mt19937& rng) {
  double worst = 0.0;
  for (int trial = 0; trial < 20; ++trial) {
    const int n = 5 + trial % 7, m = trial % 3, p = 3 * n;
    QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, trial % 4);
    Symmetrize(qp.Q);
    VectorXd w = VectorXd::Constant(p, 5.0);
    if (trial % 5 == 0) w[0] = std::numeric_limits<double>::infinity();
    das::Settings st;
    st.eps_abs = 1e-9;
    const elastiqp::Solution ref =
        das::Solve(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w, st);

    ElastiQPSettings cs;
    elastiqp_default_settings(&cs);
    cs.eps_abs = 1e-9;
    const CData cd = ToC(qp, w);
    const ElastiQPProblem cp = cd.Problem();
    CSolution sol(n, m, p);
    elastiqp_quadprog(&sol.res, &cp, &cs);
    worst = std::max(worst, Diff(sol, ref));
  }
  Check("quadprog matches das::Solve (20 problems)", worst == 0.0, worst,
        "diff");
}

void Chain(std::mt19937& rng) {
  const int n = 12, m = 3, p = 40;
  QPData qp = problem_gen::InfeasibleEq(rng, n, m, p, 4);
  Symmetrize(qp.Q);
  VectorXd w = VectorXd::Constant(p, 10.0);
  das::Solver ref;
  ref.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);

  CData cd = ToC(qp, w);
  const ElastiQPProblem cp = cd.Problem();
  ElastiQPWorkspace* work = nullptr;
  const int ret = elastiqp_setup(&work, &cp, nullptr);
  CSolution sol(n, m, p);

  double worst = 0.0;
  int solved = 0;
  for (int tick = 0; tick < 60 && ret == 0; ++tick) {
    qp.q += 0.05 * problem_gen::Randn(rng, n, 1);
    qp.h += 0.05 * problem_gen::Randn(rng, p, 1);
    qp.b += 0.01 * problem_gen::Randn(rng, m, 1);
    cd.q = qp.q;
    cd.h = qp.h;
    cd.b = qp.b;
    ref.set_q(qp.q);
    ref.set_h(qp.h);
    ref.set_b(qp.b);
    elastiqp_update_q(work, cd.q.data());
    elastiqp_update_h(work, cd.h.data());
    elastiqp_update_b(work, cd.b.data());
    if (tick % 4 == 1) {  // a few rows of G and A move, in full
      qp.G.row(tick % p) += 0.1 * problem_gen::Randn(rng, 1, n);
      qp.A.row(tick % m) += 0.01 * problem_gen::Randn(rng, 1, n);
      cd.G = qp.G;
      cd.A = qp.A;
      ref.set_G(qp.G);
      ref.set_A(qp.A);
      elastiqp_update_G(work, cd.G.data());
      elastiqp_update_A(work, cd.A.data());
    }
    if (tick % 4 == 3) {  // a block of G rows, through update_G_rows
      const int first = tick % (p - 3);
      qp.G.middleRows(first, 3) += 0.1 * problem_gen::Randn(rng, 3, n);
      cd.G = qp.G;
      ref.set_G(qp.G);
      elastiqp_update_G_rows(work, first, 3, cd.G.row(first).data());
    }
    if (tick % 10 == 3) {
      const MatrixXd dQ = 0.05 * problem_gen::Randn(rng, n, n);
      qp.Q += dQ * dQ.transpose();
      Symmetrize(qp.Q);
      cd.Q = qp.Q;
      ref.set_Q(qp.Q);
      elastiqp_update_Q(work, cd.Q.data());
    }
    if (tick % 10 == 7) {
      w = VectorXd::Constant(p, 2.0 + tick % 3);
      cd.w = w;
      ref.set_penalty(w);
      elastiqp_update_penalty(work, cd.w.data());
    }
    if (tick % 15 == 9) {  // explicit warm start from the previous solution
      const VectorXd x = sol.x, y = sol.y, z = sol.z;
      ref.set_warm_start(x, y, z);
      elastiqp_set_warm_start(work, x.data(), y.data(), z.data());
    }
    const elastiqp::Solution& r = ref.solve();
    elastiqp_solve(&sol.res, work);
    worst = std::max(worst, Diff(sol, r));
    solved += sol.res.exitflag == ELASTIQP_SOLVED;
  }
  Check("setup", ret == 0, ret, "ret");
  Check("update chain matches das::Solver (60 ticks)", worst == 0.0, worst,
        "diff");
  Check("update chain solved", solved == 60, solved, "solved");

  const elastiqp::Solution& r = ref.relax(1e-4);
  elastiqp_relax(&sol.res, work, 1e-4, 1e-6, 50);
  Check("relax matches", Diff(sol, r) == 0.0, Diff(sol, r), "diff");
  elastiqp_free(work);
}

// Check defaults and named access for every settings field.
void SettingsTable() {
  const das::Settings d;
  ElastiQPSettings c;
  elastiqp_default_settings(&c);
  bool ok = true;
  int fields = 0;
  ElastiQPWorkspace* work = nullptr;
  const double Q[] = {1}, q[] = {0};
  const ElastiQPProblem qp = {1,       0,       0,       Q,      q,
                              nullptr, nullptr, nullptr, nullptr, nullptr};
  elastiqp_setup(&work, &qp, nullptr);
  const auto field = [&](const char* name, double c_value, double cpp_value) {
    double v = -1;
    ok &= c_value == cpp_value;
    ok &= elastiqp_get_option(work, name, &v) == 0 && v == cpp_value;
    ++fields;
  };
#define FIELD(f) field(#f, c.f, d.f)
  FIELD(eps_abs);
  FIELD(eps_rel);
  FIELD(sing_tol);
  FIELD(zero_tol);
  FIELD(eps_prox);
  FIELD(eta_prox);
  FIELD(prox_relaxation);
  FIELD(prox_escalations);
  FIELD(max_iter);
  FIELD(max_outer);
  FIELD(warm_start);
  FIELD(reuse_factorization);
  FIELD(ruiz);
  FIELD(ruiz_max_iter);
  FIELD(ruiz_tol);
  FIELD(ruiz_refresh_ratio);
  FIELD(check_eq_consistency);
  FIELD(progress_tol);
  FIELD(cycle_tol);
  FIELD(refactor_tol);
  FIELD(relax_reg);
  FIELD(relax_factor_retries);
#undef FIELD
  elastiqp_free(work);
  Check("option table covers every settings field",
        ok && fields == elastiqp_num_options(), fields, "fields");
}

}  // namespace

int main() {
  std::printf("C interface vs C++ API\n");
  std::mt19937 rng(7);
  SettingsTable();
  OneShot(rng);
  Chain(rng);
  std::printf("%s\n", test_util::g_all_ok ? "ALL OK" : "FAILURES");
  return test_util::g_all_ok ? 0 : 1;
}
