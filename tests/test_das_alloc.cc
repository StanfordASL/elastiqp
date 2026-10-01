// elastiqp::das::Solver performs no heap allocation after setup() on
// robot-sized problems: every solve(), including the first, and every set_*
// update in between, across
// the solver's code paths (cold / warm, full refactorization / row update /
// vectors only, Ruiz on / off / refresh, saturated rows, hard rows and
// infeasibility, dependent and inconsistent equalities, dependent working
// sets, proximal rounds, explicit warm start, no constraints). relax() sizes
// its workspace on its first call and is allocation-free after that.
//
// Eigen heap allocations are caught through EIGEN_RUNTIME_NO_MALLOC, whose
// check goes through eigen_assert (redefined here to count instead of
// abort), and everything else (std::vector growth) through a replaced
// global operator new.
//
// Eigen's blocked triangular solve takes its scratch from the heap once
// n * (m + p) > 16384, or once n is past roughly 160 (see Solver::setup);
// that is accepted, so the sizes here stay well below both.

#include <cstdio>
#include <cstdlib>
#include <new>

namespace alloc_guard {
long g_eigen = 0;  // failed eigen_assert, i.e. a forbidden Eigen allocation
long g_new = 0;    // operator new while armed
bool g_armed = false;
}  // namespace alloc_guard

#define EIGEN_RUNTIME_NO_MALLOC
#include <map>
#include <string>
namespace alloc_guard {
inline std::map<std::string, long>& Sites() {
  static std::map<std::string, long> m;
  return m;
}
inline void Fail(const char* what, const char* file, int line) {
  ++g_eigen;
  const bool armed = g_armed;
  g_armed = false;
  Sites()[std::string(what).substr(0, 60) + " @ " + file + ":" +
          std::to_string(line)]++;
  g_armed = armed;
}
}  // namespace alloc_guard
#define eigen_assert(x)                                  \
  do {                                                   \
    if (!(x)) alloc_guard::Fail(#x, __FILE__, __LINE__); \
  } while (false)

void* operator new(std::size_t n) {
  if (alloc_guard::g_armed) ++alloc_guard::g_new;
  if (void* p = std::malloc(n > 0 ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

#include <cmath>
#include <limits>
#include <random>
#include <string>

#include "elastiqp/elastiqp.hpp"
#include "problem_gen.hpp"
#include "test_util.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;
using test_util::Check;
namespace das = elastiqp::das;

namespace {

// Heap allocations made by f().
template <class F>
long Allocations(F&& f) {
  const long e0 = alloc_guard::g_eigen, n0 = alloc_guard::g_new;
  Eigen::internal::set_is_malloc_allowed(false);
  alloc_guard::g_armed = true;
  f();
  alloc_guard::g_armed = false;
  Eigen::internal::set_is_malloc_allowed(true);
  return (alloc_guard::g_eigen - e0) + (alloc_guard::g_new - n0);
}

// Solves with allocations counted; the first solve after setup included.
struct Run {
  das::Solver s;
  long allocs = 0;
  int solves = 0, solved = 0;
  elastiqp::Status last = elastiqp::Status::kUnsolved;

  explicit Run(const das::Settings& st = {}) { s.settings = st; }
  void Solve() {
    allocs += Allocations([&] { last = s.solve().status; });
    solves++;
    solved += last == elastiqp::Status::kSolved;
  }
  // Updates are part of the allocation-free loop too.
  template <class F>
  void Update(F&& f) {
    allocs += Allocations(f);
  }
  void Report(const std::string& name, bool path_hit = true) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s: no allocations (%d solves, %d solved)",
                  name.c_str(), solves, solved);
    Check(buf, allocs == 0 && path_hit, static_cast<double>(allocs), "allocs");
  }
};

void Perturb(std::mt19937& rng, MatrixXd& M, double sigma) {
  M += sigma * problem_gen::Randn(rng, static_cast<int>(M.rows()),
                                  static_cast<int>(M.cols()));
}
void Perturb(std::mt19937& rng, VectorXd& v, double sigma) {
  v += sigma * problem_gen::Randn(rng, static_cast<int>(v.size()), 1);
}

// Drifting chain with equalities and conflicts. what: which data drifts.
enum class Drift { kAll, kRows, kVectors };

void Chain(std::mt19937& rng, Drift what, bool warm, bool ruiz) {
  QPData qp = problem_gen::InfeasibleEq(rng, 20, 4, 50, 4);
  const VectorXd w = VectorXd::Constant(50, 10.0);
  das::Settings st;
  st.warm_start = warm;
  st.ruiz = ruiz;
  Run r(st);
  r.s.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
  r.Solve();
  for (int tick = 0; tick < 40; ++tick) {
    if (what == Drift::kAll) {
      MatrixXd dQ = 0.01 * problem_gen::Randn(rng, 20, 20);
      qp.Q += dQ * dQ.transpose();
      Perturb(rng, qp.A, 0.01);
    }
    if (what != Drift::kVectors) {
      for (int i = 0; i < 50; i += 5) {  // a fifth of the rows
        MatrixXd row = qp.G.row(i);
        Perturb(rng, row, 0.02);
        qp.G.row(i) = row;
      }
    }
    Perturb(rng, qp.q, 0.05);
    Perturb(rng, qp.h, 0.05);
    if (what == Drift::kAll) qp.b = qp.A * qp.x_opt;
    r.Update([&] {
      if (what == Drift::kAll) {
        r.s.set_Q(qp.Q);
        r.s.set_A(qp.A);
        r.s.set_b(qp.b);
      }
      if (what != Drift::kVectors) r.s.set_G(qp.G);
      r.s.set_q(qp.q);
      r.s.set_h(qp.h);
    });
    r.Solve();
  }
  const char* names[] = {"Q/A/G/vectors", "G rows/vectors", "vectors"};
  r.Report(std::string("chain, ") + names[static_cast<int>(what)] + " drift, " +
           (warm ? "warm" : "cold") + (ruiz ? ", ruiz" : ", no ruiz"));
}

void HardRows(std::mt19937& rng) {
  const double inf = std::numeric_limits<double>::infinity();
  const QPData c = problem_gen::Infeasible(rng, 12, 30, 1);
  VectorXd w = VectorXd::Constant(30, inf);
  Run hard;
  hard.s.setup(c.Q, c.q, c.G, c.h, w);
  hard.Solve();
  hard.Report("conflicting hard rows (kInfeasible)",
              hard.last == elastiqp::Status::kInfeasible);
  w[1] = 10.0;
  Run mix;
  mix.s.setup(c.Q, c.q, c.G, c.h, w);
  mix.Solve();
  mix.Report("hard vs elastic conflict");
}

void Equalities(std::mt19937& rng) {
  const QPData qp = problem_gen::InfeasibleEq(rng, 20, 4, 30, 2);
  MatrixXd A(5, 20);
  VectorXd b(5);
  A << qp.A, qp.A.row(1);
  b << qp.b, qp.b[1];
  Run r;
  r.s.setup(qp.Q, qp.q, A, b, qp.G, qp.h, VectorXd::Constant(30, 10.0));
  r.Solve();
  b[4] += 0.5;
  r.Update([&] { r.s.set_b(b); });
  r.Solve();
  const bool infeasible = r.last == elastiqp::Status::kInfeasible;
  b[4] -= 0.5;
  r.Update([&] { r.s.set_b(b); });
  r.Solve();
  r.Report("dependent / inconsistent equalities", infeasible);
}

void Duplicated(std::mt19937& rng) {
  const QPData base = problem_gen::Infeasible(rng, 15, 30, 3);
  MatrixXd G(60, 15);
  VectorXd h(60);
  G << base.G, base.G;
  h << base.h, base.h;
  // The repair (refactor_working_set) is too rare to reach by problem data
  // alone. Pivots of the normalized rows are at most 1, so this threshold
  // takes it at every optimum with more than two working rows.
  das::Settings st;
  st.refactor_tol = 1.0;
  Run r(st);
  r.s.setup(base.Q, base.q, G, h, VectorXd::Constant(60, 5.0));
  r.Solve();
  int refactors = r.s.refactors();
  for (int tick = 0; tick < 10; ++tick) {
    VectorXd q = base.q;
    Perturb(rng, q, 0.1);
    r.Update([&] { r.s.set_q(q); });
    r.Solve();
    refactors += r.s.refactors();
  }
  r.Report("duplicated rows (dependent working sets, repair)", refactors > 0);
}

void Proximal(std::mt19937& rng) {
  const QPData qp = problem_gen::Feasible(rng, 10, 20);
  MatrixXd Qs = MatrixXd::Zero(10, 10);
  Qs(0, 0) = 1.0;
  MatrixXd G(40, 10);
  VectorXd h(40), w(40);
  G << qp.G, MatrixXd::Identity(10, 10), -MatrixXd::Identity(10, 10);
  h << qp.h, VectorXd::Constant(20, 5.0);
  w << VectorXd::Constant(20, 10.0), VectorXd::Constant(20, 1e4);
  Run r;
  r.s.setup(Qs, qp.q, G, h, w);
  r.Solve();
  const bool prox = r.s.proximal();
  VectorXd q = qp.q;
  for (int tick = 0; tick < 10; ++tick) {
    Perturb(rng, q, 0.1);
    r.Update([&] { r.s.set_q(q); });
    r.Solve();
  }
  r.Report("singular Q, proximal rounds", prox);

  Run lp;  // Q = 0
  lp.s.setup(MatrixXd::Zero(10, 10), qp.q, G, h, w);
  lp.Solve();
  lp.Report("LP (Q = 0)");

  das::Settings capped;
  capped.max_outer = 50;
  Run unb(capped);  // unbounded: runs out of outer iterations
  unb.s.setup(Qs, qp.q, MatrixXd(0, 10), VectorXd(0), VectorXd(0));
  unb.Solve();
  unb.Report("singular unconstrained (kMaxIter)",
             unb.last == elastiqp::Status::kMaxIter);

  das::Settings refuse;
  refuse.eps_prox = 0.0;
  Run rf(refuse);
  rf.s.setup(Qs, qp.q, qp.G, qp.h, 10.0);
  rf.Solve();
  rf.Report("eps_prox = 0, singular Q (kNumerics)",
            rf.last == elastiqp::Status::kNumerics);
}

void RuizRefresh(std::mt19937& rng) {
  const QPData qp = problem_gen::InfeasibleEq(rng, 16, 4, 60, 15);
  VectorXd pen = VectorXd::Constant(60, 10.0);
  Run r;
  r.s.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, pen);
  r.Solve();
  MatrixXd G = qp.G;
  VectorXd h = qp.h;
  G.row(3) *= 1e7;
  h[3] *= 1e7;
  pen[3] /= 1e7;
  r.Update([&] {
    r.s.set_G(G);
    r.s.set_h(h);
    r.s.set_penalty(pen);
  });
  r.Solve();
  const bool rescaled = r.s.rescaled();
  pen *= 2.0;  // penalty only
  r.Update([&] { r.s.set_penalty(pen); });
  r.Solve();
  r.Report("Ruiz refresh + penalty updates", rescaled);
}

void ExplicitWarm(std::mt19937& rng) {
  const QPData qp = problem_gen::InfeasibleEq(rng, 20, 4, 40, 4);
  const VectorXd w = VectorXd::Constant(40, 10.0);
  const elastiqp::Solution ref =
      das::Solve(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
  Run r;
  r.s.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
  r.Update([&] { r.s.set_warm_start(ref.x, ref.y, ref.z); });
  r.Solve();
  r.Update([&] { r.s.set_warm_start(ref.x, ref.y, ref.z); });
  r.Solve();
  r.Report("explicit warm start");
}

// relax() sizes its workspace on the first call after setup(); later calls
// (and the solves in between) are allocation-free.
void Relax(std::mt19937& rng) {
  const double inf = std::numeric_limits<double>::infinity();
  QPData qp = problem_gen::InfeasibleEq(rng, 20, 4, 50, 4);
  VectorXd w = VectorXd::Constant(50, 10.0);
  w[10] = inf;
  Run r;
  r.s.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, w);
  r.Solve();
  bool relaxed = r.s.relax(1e-3).converged == 1;  // sizes the workspace
  for (int tick = 0; tick < 10; ++tick) {
    Perturb(rng, qp.q, 0.05);
    Perturb(rng, qp.h, 0.05);
    r.Update([&] {
      r.s.set_q(qp.q);
      r.s.set_h(qp.h);
    });
    r.Solve();
    r.Update([&] { relaxed &= r.s.relax(1e-3).converged == 1; });
  }
  r.Report("relax after its first call", relaxed);
}

void Unconstrained(std::mt19937& rng) {
  const QPData qp = problem_gen::Feasible(rng, 8, 0);
  Run r;
  r.s.setup(qp.Q, qp.q, MatrixXd(0, 8), VectorXd(0), VectorXd(0));
  r.Solve();
  r.Solve();
  r.Report("no constraints");
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  for (Drift d : {Drift::kAll, Drift::kRows, Drift::kVectors})
    for (int warm = 0; warm < 2; ++warm)
      for (int ruiz = 0; ruiz < 2; ++ruiz) Chain(rng, d, warm == 1, ruiz == 1);
  HardRows(rng);
  Equalities(rng);
  Duplicated(rng);
  Proximal(rng);
  RuizRefresh(rng);
  ExplicitWarm(rng);
  Relax(rng);
  Unconstrained(rng);
  for (const auto& [site, k] : alloc_guard::Sites())
    std::printf("  [eigen_assert] %6ld x %s\n", k, site.c_str());
  std::printf("%s\n", test_util::g_all_ok ? "ALL PASSED" : "SOME FAILED");
  return test_util::g_all_ok ? 0 : 1;
}
