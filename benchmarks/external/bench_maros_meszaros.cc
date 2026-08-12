// Maros-Meszaros benchmark for ElastiQP (small dense subset).
//
// The problems come from the proxsuite release tarball's copy of the set
// (test/data/maros_meszaros_data, OSQP-style form l <= Ax <= u, fetched by
// CMake), packed at build time by tools/convert_maros_meszaros.py into a
// flat binary -- no data is duplicated in this repo and no .mat reader is
// needed here.
//
// ElastiQP solves the *elastic* relaxation of each problem: rows with
// l == u become hard equalities, every finite one-sided inequality becomes
// an elastic row. With the penalty above the largest dual of the hard
// problem, the exact-penalty property means the elastic solution IS the
// hard solution, which this benchmark verifies against a high-accuracy
// vanilla-PIQP reference (the penalty is set to 10x the reference's largest
// dual).
//
// Four routes, all timed in-process through the solvers' C++ APIs on
// identical data (cold solves):
//   elastiqp       elastiqp::Solve (PDAL) on the elastic form
//   piqp-hard      vanilla dense PIQP on the hard two-sided problem
//   proxqp-hard    dense ProxQP on the hard two-sided problem
//   piqp-expanded  vanilla dense PIQP on the expanded (n+p)-variable
//                  elastic formulation at matched tolerance -- the
//                  general-purpose-solver route to the same problem class
//                  -- for problems small enough to be practical
//
// Note these problems are far from ElastiQP's design point (robot control:
// n <= ~60, well-scaled data, no preconditioning needed). Maros-Meszaros is
// notoriously badly scaled, so ElastiQP runs with Ruiz equilibration ON
// (settings.ruiz = true) -- these instances are what the flag exists for.
//
// Usage: bench_maros_meszaros [packed_file]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <proxsuite/proxqp/dense/dense.hpp>

#include "elastiqp/elastiqp.hpp"
#include "piqp/piqp.hpp"
#include "problem_gen.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::chrono::duration;
using std::chrono::steady_clock;

namespace {

constexpr double kEps = 1e-6;
constexpr int kMaxIter = 250;
// Expanded-formulation comparison only below this (n + p): its dense
// factorization is O((n+p)^3) per iteration and becomes impractical.
constexpr int kExpandedLimit = 700;

struct MMProblem {
  std::string name;
  MatrixXd P;   // n x n
  VectorXd q;   // n
  MatrixXd A;   // rows x n, with l <= A x <= u
  VectorXd l, u;
};

std::int32_t ReadI32(std::istream& is) {
  std::int32_t v = 0;
  is.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}

// Reads the packed file written by tools/convert_maros_meszaros.py
// (row-major float64 matrices).
std::vector<MMProblem> LoadProblems(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) throw std::runtime_error("cannot open: " + path);
  char magic[4];
  is.read(magic, 4);
  if (!is || std::string(magic, 4) != "MMQP") {
    throw std::runtime_error("bad magic in " + path);
  }
  if (ReadI32(is) != 1) throw std::runtime_error("unsupported version");
  std::vector<MMProblem> problems(static_cast<std::size_t>(ReadI32(is)));
  using RowMajor =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  auto read = [&is](double* data, Eigen::Index count) {
    is.read(reinterpret_cast<char*>(data),
            static_cast<std::streamsize>(count) *
                static_cast<std::streamsize>(sizeof(double)));
  };
  for (MMProblem& prob : problems) {
    std::string name(static_cast<std::size_t>(ReadI32(is)), '\0');
    is.read(name.data(), static_cast<std::streamsize>(name.size()));
    prob.name = name;
    const std::int32_t n = ReadI32(is);
    const std::int32_t rows = ReadI32(is);
    RowMajor P(n, n), A(rows, n);
    prob.q.resize(n);
    prob.l.resize(rows);
    prob.u.resize(rows);
    read(P.data(), P.size());
    read(prob.q.data(), n);
    read(A.data(), A.size());
    read(prob.l.data(), rows);
    read(prob.u.data(), rows);
    prob.P = P;
    prob.A = A;
    if (!is) throw std::runtime_error("truncated file: " + path);
  }
  return problems;
}

// Split l <= Ax <= u into hard equalities (l == u) and one-sided elastic
// rows; two-sided finite inequalities become two rows.
struct ElasticForm {
  MatrixXd Aeq, G;
  VectorXd beq, h;
};

ElasticForm ToElastic(const MMProblem& prob) {
  const Eigen::Index n = prob.q.size();
  const Eigen::Index rows = prob.l.size();
  Eigen::Index meq = 0, p = 0;
  for (Eigen::Index i = 0; i < rows; ++i) {
    if (prob.l[i] == prob.u[i]) {
      meq++;
    } else {
      if (std::isfinite(prob.u[i])) p++;
      if (std::isfinite(prob.l[i])) p++;
    }
  }
  ElasticForm e;
  e.Aeq.resize(meq, n);
  e.beq.resize(meq);
  e.G.resize(p, n);
  e.h.resize(p);
  Eigen::Index ei = 0, gi = 0;
  for (Eigen::Index i = 0; i < rows; ++i) {
    if (prob.l[i] == prob.u[i]) {
      e.Aeq.row(ei) = prob.A.row(i);
      e.beq[ei++] = prob.l[i];
    } else {
      if (std::isfinite(prob.u[i])) {
        e.G.row(gi) = prob.A.row(i);
        e.h[gi++] = prob.u[i];
      }
      if (std::isfinite(prob.l[i])) {
        e.G.row(gi) = -prob.A.row(i);
        e.h[gi++] = -prob.l[i];
      }
    }
  }
  return e;
}

// The HARD problem's two-sided inequality block: rows with l != u and at
// least one finite bound (fully-free rows would just make the solvers warn
// and zero them out). Shared by the piqp-hard and proxqp-hard routes.
struct HardIneq {
  MatrixXd C;
  VectorXd cl, cu;
};

HardIneq ToHardIneq(const MMProblem& prob) {
  const Eigen::Index rows = prob.l.size();
  auto is_ineq = [&](Eigen::Index i) {
    return prob.l[i] != prob.u[i] &&
           (std::isfinite(prob.l[i]) || std::isfinite(prob.u[i]));
  };
  Eigen::Index mineq = 0;
  for (Eigen::Index i = 0; i < rows; ++i) {
    if (is_ineq(i)) mineq++;
  }
  HardIneq hi;
  hi.C.resize(mineq, prob.q.size());
  hi.cl.resize(mineq);
  hi.cu.resize(mineq);
  Eigen::Index ci = 0;
  for (Eigen::Index i = 0; i < rows; ++i) {
    if (is_ineq(i)) {
      hi.C.row(ci) = prob.A.row(i);
      hi.cl[ci] = prob.l[i];
      hi.cu[ci] = prob.u[i];
      ci++;
    }
  }
  return hi;
}

int Repeats(Eigen::Index n, Eigen::Index p) {
  const long work = static_cast<long>(n) * n * n +
                    static_cast<long>(p) * n * n;
  if (work < 1000000) return 50;
  if (work < 20000000) return 10;
  return 3;
}

struct HardRef {
  bool ok = false;
  VectorXd x;
  double max_dual = 0;
  double us = 0;
};

// High-accuracy solve of the HARD problem with vanilla dense PIQP
// (two-sided inequality interface). Also timed at kEps for the piqp-hard
// route.
HardRef SolveHardReference(const MMProblem& prob, const ElasticForm& e,
                           const HardIneq& hi) {
  HardRef ref;

  auto configure_and_solve = [&](piqp::DenseSolver<double>& solver,
                                 double eps) {
    solver.settings().eps_abs = eps;
    solver.settings().eps_rel = 0;
    solver.settings().eps_duality_gap_abs = eps;
    solver.settings().eps_duality_gap_rel = 0;
    solver.settings().max_iter = kMaxIter;
    solver.settings().verbose = false;
    if (e.beq.size() > 0) {
      solver.setup(prob.P, prob.q, e.Aeq, e.beq, hi.C, hi.cl, hi.cu,
                   piqp::nullopt, piqp::nullopt);
    } else {
      solver.setup(prob.P, prob.q, piqp::nullopt, piqp::nullopt, hi.C, hi.cl,
                   hi.cu, piqp::nullopt, piqp::nullopt);
    }
    return solver.solve();
  };

  // High-accuracy reference for x and the duals (sets the penalty scale).
  {
    piqp::DenseSolver<double> solver;
    if (configure_and_solve(solver, 1e-9) == piqp::PIQP_SOLVED) {
      ref.ok = true;
      ref.x = solver.result().x;
      const auto& res = solver.result();
      ref.max_dual = std::max(
          {res.y.size() > 0 ? res.y.lpNorm<Eigen::Infinity>() : 0.0,
           res.z_l.size() > 0 ? res.z_l.lpNorm<Eigen::Infinity>() : 0.0,
           res.z_u.size() > 0 ? res.z_u.lpNorm<Eigen::Infinity>() : 0.0});
    }
  }
  // Timing at the benchmark tolerance.
  const int reps = Repeats(prob.q.size(), hi.cl.size());
  const auto t0 = steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    piqp::DenseSolver<double> solver;
    configure_and_solve(solver, kEps);
  }
  const auto t1 = steady_clock::now();
  ref.us = duration<double, std::micro>(t1 - t0).count() / reps;
  return ref;
}

struct ProxqpHard {
  bool converged = false;
  VectorXd x;
  long iters = 0;
  double us = 0;
};

// Dense ProxQP on the same HARD two-sided problem, cold (fresh solver per
// rep, EQUALITY_CONSTRAINED_INITIAL_GUESS -- ProxQP's default), at the same
// convergence targets as the piqp-hard route. Note that ProxQP checks
// convergence on the Ruiz-preconditioned residuals, so the unscaled KKT
// residual it delivers can be looser than eps_abs even when it reports
// PROXQP_SOLVED. Its max_iter counts INNER iterations, so kMaxIter (an
// IPM-iteration cap) would be unfairly tight; the default budget is kept.
ProxqpHard SolveProxqpHard(const MMProblem& prob, const ElasticForm& e,
                           const HardIneq& hi) {
  namespace pq = proxsuite::proxqp;
  const Eigen::Index n = prob.q.size();
  const Eigen::Index meq = e.beq.size();
  const Eigen::Index mineq = hi.cl.size();

  auto configure_and_solve = [&](pq::dense::QP<double>& qp) {
    qp.settings.eps_abs = kEps;
    qp.settings.eps_rel = 0;
    qp.settings.check_duality_gap = true;
    qp.settings.eps_duality_gap_abs = kEps;
    qp.settings.eps_duality_gap_rel = 0;
    qp.settings.verbose = false;
    qp.settings.initial_guess =
        pq::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    if (meq > 0 && mineq > 0) {
      qp.init(prob.P, prob.q, e.Aeq, e.beq, hi.C, hi.cl, hi.cu);
    } else if (meq > 0) {
      qp.init(prob.P, prob.q, e.Aeq, e.beq, proxsuite::nullopt,
              proxsuite::nullopt, proxsuite::nullopt);
    } else if (mineq > 0) {
      qp.init(prob.P, prob.q, proxsuite::nullopt, proxsuite::nullopt, hi.C,
              hi.cl, hi.cu);
    } else {
      qp.init(prob.P, prob.q, proxsuite::nullopt, proxsuite::nullopt,
              proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt);
    }
    qp.solve();
  };

  ProxqpHard out;
  // Accuracy pass, off the clock.
  {
    pq::dense::QP<double> qp(n, meq, mineq);
    configure_and_solve(qp);
    out.converged =
        qp.results.info.status == pq::QPSolverOutput::PROXQP_SOLVED;
    out.x = qp.results.x;
    out.iters = static_cast<long>(qp.results.info.iter);
  }
  // Timing at the benchmark tolerance.
  const int reps = Repeats(n, mineq);
  const auto t0 = steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    pq::dense::QP<double> qp(n, meq, mineq);
    configure_and_solve(qp);
  }
  const auto t1 = steady_clock::now();
  out.us = duration<double, std::micro>(t1 - t0).count() / reps;
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  std::string csv_dir;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--csv" && i + 1 < argc) {
      csv_dir = argv[++i];
    } else {
      path = argv[i];
    }
  }
  if (path.empty()) {
#ifdef ELASTIQP_MM_DATA
    path = ELASTIQP_MM_DATA;
#else
    path = "maros_meszaros_small.bin";
#endif
  }
  std::FILE* fcsv = nullptr;
  if (!csv_dir.empty()) {
    const std::string csv_path = csv_dir + "/maros_meszaros_results.csv";
    fcsv = std::fopen(csv_path.c_str(), "w");
    if (!fcsv) {
      std::fprintf(stderr, "cannot write %s\n", csv_path.c_str());
      return 1;
    }
    // ok = converged AND matched the hard reference objective to 1e-5
    // (d_obj is NaN when the reference itself failed).
    std::fprintf(fcsv, "name,arch,route,n,m,p,time_us,iters,ok,d_obj\n");
  }
  std::vector<MMProblem> problems;
  try {
    problems = LoadProblems(path);
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "%s\nRun tools/convert_maros_meszaros.py first.\n",
                 ex.what());
    return 1;
  }

  std::printf(
      "Maros-Meszaros small dense subset (%d problems, eps = %.0e)\n"
      "  elastiqp      = elastiqp::Solve (PDAL) on the elastic form\n"
      "                  (penalty = 10x max ref dual; cold, Ruiz\n"
      "                  equilibration ON -- these badly-scaled instances\n"
      "                  are what the flag exists for)\n"
      "  piqp-hard     = vanilla dense piqp on the HARD two-sided problem\n"
      "  proxqp-hard   = dense ProxQP on the HARD two-sided problem (cold,\n"
      "                  EQUALITY_CONSTRAINED_INITIAL_GUESS; converges on\n"
      "                  preconditioned residuals, it = inner iterations)\n"
      "  piqp-expanded = vanilla dense piqp on the expanded elastic\n"
      "                  formulation (skipped when n+p > %d)\n"
      "  d_obj = relative hard-objective gap vs the reference; '!' = not\n"
      "  converged\n\n",
      static_cast<int>(problems.size()), kEps, kExpandedLimit);
  std::printf(
      "  %-10s %4s %4s %5s | %9s | %12s %4s %8s %8s | %13s | %15s %5s %8s | "
      "%17s %8s\n",
      "problem", "n", "m", "p", "penalty", "elastiqp[us]", "it", "d_obj",
      "max_t", "piqp-hard[us]", "proxqp-hard[us]", "it", "d_obj",
      "piqp-expanded[us]", "speedup");

  int solved = 0, matched = 0, ref_failed = 0;
  int proxqp_solved = 0, proxqp_matched = 0;
  for (const MMProblem& prob : problems) {
    const ElasticForm e = ToElastic(prob);
    const HardIneq hi = ToHardIneq(prob);
    const Eigen::Index n = prob.q.size();
    const Eigen::Index m = e.beq.size();
    const Eigen::Index p = e.h.size();

    const HardRef ref = SolveHardReference(prob, e, hi);
    if (!ref.ok) ref_failed++;
    // Exact-penalty rule: safely above the largest dual of the hard
    // problem, clamped to a sane range when the reference fails.
    const double penalty_val =
        ref.ok ? std::clamp(10.0 * ref.max_dual, 1e3, 1e8) : 1e6;
    const VectorXd penalty = VectorXd::Constant(p, penalty_val);

    // The elastiqp route: PDAL on the elastic form (cold one-shot).
    elastiqp::Settings settings;
    settings.eps_abs = kEps;
    settings.eps_rel = 0;
    settings.eps_duality_gap_abs = kEps;
    settings.eps_duality_gap_rel = 0;
    settings.max_outer_iter = kMaxIter;
    settings.ruiz = true;

    const int reps = Repeats(n, p);
    elastiqp::Solution sol;
    const auto t0 = steady_clock::now();
    for (int r = 0; r < reps; ++r) {
      sol = elastiqp::Solve(prob.P, prob.q, e.Aeq, e.beq, e.G, e.h, penalty,
                            settings);
    }
    const auto t1 = steady_clock::now();
    const double elastiqp_us =
        duration<double, std::micro>(t1 - t0).count() / reps;

    // Hard-objective gap vs the reference. The x themselves can differ on
    // problems with non-unique optima (the LP-like instances), so compare
    // objective values, which every optimizer must agree on.
    auto hard_obj = [&](const VectorXd& x) {
      return 0.5 * x.dot(prob.P * x) + prob.q.dot(x);
    };
    double dobj = std::numeric_limits<double>::quiet_NaN();
    if (ref.ok) {
      dobj = std::abs(hard_obj(sol.x) - hard_obj(ref.x)) /
             (1.0 + std::abs(hard_obj(ref.x)));
    }
    const double max_t = p > 0 ? sol.t.maxCoeff() : 0.0;
    if (sol.converged == 1) solved++;
    if (sol.converged == 1 && ref.ok && dobj < 1e-5) matched++;

    // ProxQP on the hard two-sided problem.
    const ProxqpHard prox = SolveProxqpHard(prob, e, hi);
    double prox_dobj = std::numeric_limits<double>::quiet_NaN();
    if (ref.ok) {
      prox_dobj = std::abs(hard_obj(prox.x) - hard_obj(ref.x)) /
                  (1.0 + std::abs(hard_obj(ref.x)));
    }
    if (prox.converged) proxqp_solved++;
    if (prox.converged && ref.ok && prox_dobj < 1e-5) proxqp_matched++;

    // Expanded-formulation comparison at matched tolerance.
    double piqp_exp_us = -1;
    bool piqp_exp_ok = false;
    long piqp_exp_iters = -1;
    if (n + p <= kExpandedLimit) {
      const problem_gen::ExpandedElastic ex = problem_gen::MakeExpanded(
          prob.P, prob.q, e.Aeq, e.beq, e.G, e.h, penalty);
      const int ereps = std::max(1, Repeats(n + p, 0) / 2);
      const auto e0 = steady_clock::now();
      for (int r = 0; r < ereps; ++r) {
        piqp::DenseSolver<double> solver;
        solver.settings().eps_abs = kEps;
        solver.settings().eps_rel = 0;
        solver.settings().eps_duality_gap_abs = kEps;
        solver.settings().eps_duality_gap_rel = 0;
        solver.settings().max_iter = kMaxIter;
        solver.settings().verbose = false;
        if (m > 0) {
          solver.setup(ex.P, ex.c, ex.A, ex.b, ex.Gt, piqp::nullopt, ex.h,
                       ex.lb, piqp::nullopt);
        } else {
          solver.setup(ex.P, ex.c, piqp::nullopt, piqp::nullopt, ex.Gt,
                       piqp::nullopt, ex.h, ex.lb, piqp::nullopt);
        }
        const piqp::Status est = solver.solve();
        piqp_exp_ok = est == piqp::PIQP_SOLVED;
        piqp_exp_iters = solver.result().info.iter;
      }
      const auto e1 = steady_clock::now();
      piqp_exp_us = duration<double, std::micro>(e1 - e0).count() / ereps;
    }

    char exp_buf[32], speedup_buf[32];
    if (piqp_exp_us >= 0) {
      std::snprintf(exp_buf, sizeof(exp_buf), "%17.1f", piqp_exp_us);
      std::snprintf(speedup_buf, sizeof(speedup_buf), "%7.1fx",
                    piqp_exp_us / elastiqp_us);
    } else {
      std::snprintf(exp_buf, sizeof(exp_buf), "%17s", "-");
      std::snprintf(speedup_buf, sizeof(speedup_buf), "%8s", "-");
    }
    std::printf(
        "  %-10s %4d %4d %5d | %9.1e | %12.1f %4d%s %8.1e %8.1e | "
        "%12.1f%s | %14.1f %5ld%s %8.1e | %s %s\n",
        prob.name.c_str(), static_cast<int>(n), static_cast<int>(m),
        static_cast<int>(p), penalty_val, elastiqp_us, sol.iters,
        sol.converged == 1 ? " " : "!", dobj, max_t, ref.us,
        ref.ok ? " " : "!", prox.us, prox.iters, prox.converged ? " " : "!",
        prox_dobj, exp_buf, speedup_buf);

    if (fcsv) {
      const auto row = [&](const char* route, double us, long iters, bool ok,
                           double d_obj) {
        std::fprintf(fcsv, "%s,%s,%s,%d,%d,%d,%.3f,%ld,%d,%.3e\n",
                     prob.name.c_str(), ELASTIQP_ARCH_LABEL, route,
                     static_cast<int>(n), static_cast<int>(m),
                     static_cast<int>(p), us, iters, ok ? 1 : 0, d_obj);
      };
      row("elastiqp", elastiqp_us, sol.iters,
          sol.converged == 1 && ref.ok && dobj < 1e-5, dobj);
      // The piqp-hard timing is the reference solver re-run at the
      // benchmark tolerance; it defines d_obj = 0 by construction.
      row("piqp-hard", ref.us, -1, ref.ok, ref.ok ? 0.0 : NAN);
      row("proxqp-hard", prox.us, prox.iters,
          prox.converged && ref.ok && prox_dobj < 1e-5, prox_dobj);
      if (piqp_exp_us >= 0) {
        row("piqp-expanded", piqp_exp_us, piqp_exp_iters, piqp_exp_ok, NAN);
      }
    }
  }

  std::printf(
      "\nelastiqp %d/%d converged, %d matched the hard reference objective "
      "to 1e-5; proxqp-hard %d/%d converged, %d matched%s\n",
      solved, static_cast<int>(problems.size()), matched, proxqp_solved,
      static_cast<int>(problems.size()), proxqp_matched,
      ref_failed ? " (some references failed)" : "");
  if (fcsv) {
    std::fclose(fcsv);
    std::printf("wrote %s/maros_meszaros_results.csv\n", csv_dir.c_str());
  }
  return 0;
}
