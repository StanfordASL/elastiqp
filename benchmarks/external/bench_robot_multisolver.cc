// The cross-solver robot control-loop benchmark: every solver route replayed
// over the robotics sequences (diff-ik / arm-osc / hum-wbc, from
// gen_robot_sequences), on both a FEASIBLE and a CONFLICT variant of each
// sequence — so a single binary produces the cross-solver timing table
// (feasible rows) and the graceful-degradation study (conflict rows).
// Fairness: all solvers are timed in-process through their C++ APIs on
// identical problem data, with persistent solvers and workspace reuse
// wherever the API supports it.
//
// Two variants of every sequence:
//
//   feasible  the original control loops (inequalities satisfiable
//             essentially everywhere)
//   conflict  the same ticks with a few opposing constraint pairs
//             collapsed into contradiction around the operating point:
//             for a selected pair of rows with G_j = -G_i (box/damper
//             pairs), both sides are moved to the feasible tick's solution
//             value v* = g_i^T x* with an empty interval of width `gap`
//             (upper: g_i^T x <= v* - gap/2, lower: g_i^T x >= v* + gap/2,
//             gap = 5% of the row's bound scale). The minimal l1 violation
//             is exactly `gap` per pair, near the trajectory the loop was
//             already flying — "slightly infeasible", with a known
//             conflict set.
//
// Solver routes, all dense, all cold (persistent solver, workspace reuse,
// update() between ticks; elastiqp additionally warm):
//
//   elastiqp      ElastiQP (primal-dual augmented Lagrangian) on the
//                 native elastic formulation
//   elastiqp-warm same, with iterate warm starting across ticks
//   piqp-hard     vanilla PIQP on the hard problem (expected: reports
//                 primal infeasibility on the conflict variant; its time
//                 there is the cost of the certificate, with no usable x)
//   piqp-slack    vanilla PIQP on the expanded (n+p)-variable l1-slack
//                 formulation ("add slack variables by hand")
//   proxqp-hard   vanilla ProxQP dense on the hard problem (default
//                 infeasibility heuristics — what a user gets out of the
//                 box)
//   proxqp-clfeas ProxQP dense with primal_infeasibility_solving = true:
//                 the closest-feasible QP, "closest" = minimal l2-norm
//                 constraint shift
//   proxqp-slack  vanilla ProxQP dense on the expanded l1-slack formulation
//
// Reported per (scenario, variant, route): per-tick time distribution
// (mean, std, p50, p95, max), iterations, tick status counts
// (solved / infeasible-reported / other), and the violation structure of
// the returned solution: rows violated (> 1e-4), rows violated OUTSIDE the
// known conflict pairs (shift leaked onto constraints that never needed to
// move), l1/linf violation norms, and the worst hard-equality residual —
// the elastic formulation's core claim is that equalities hold exactly
// even when the inequalities conflict.
//
// Usage: bench_robot_multisolver [sequence_file] [--csv <dir>]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <proxsuite/proxqp/dense/dense.hpp>

#include "elastiqp/elastiqp.hpp"
#include "piqp/piqp.hpp"
#include "problem_gen.hpp"
#include "qp_io.hpp"

#ifndef ELASTIQP_ARCH_LABEL
#define ELASTIQP_ARCH_LABEL "default"
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using robot_control::NamedSequence;
using robot_control::RobotQP;
using std::chrono::duration;
using std::chrono::steady_clock;

namespace {

constexpr double kEps = 1e-6;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kViolThresh = 1e-4;  // above tolerance, below the gaps
constexpr double kGapFrac = 0.05;     // conflict gap: 5% of the row bound
// ProxQP's default max_iter is 10000 (inner); on ticks where its
// infeasibility heuristics never fire that is ~minutes per sequence, so
// bound it. 1000 inner iterations is far above anything a successful solve
// needs here (tens); ticks that hit the cap are counted under 'other'.
constexpr int kProxMaxIter = 1000;

using Sequence = std::vector<RobotQP>;

// ------------------------------------------------------------ conflict gen

// Rows made contradictory: pair (upper_row, lower_row) with
// G.row(lower) == -G.row(upper).
struct ConflictSpec {
  std::vector<std::pair<int, int>> pairs;
  bool Contains(int row) const {
    for (const auto& pr : pairs) {
      if (pr.first == row || pr.second == row) return true;
    }
    return false;
  }
};

// Opposing-row pairs in G (box and damper structures produce them); each
// row is used at most once.
std::vector<std::pair<int, int>> OpposingPairs(const MatrixXd& G) {
  const int p = static_cast<int>(G.rows());
  std::vector<bool> used(p, false);
  std::vector<std::pair<int, int>> pairs;
  for (int i = 0; i < p; ++i) {
    if (used[i] || G.row(i).norm() == 0) continue;
    for (int j = i + 1; j < p; ++j) {
      if (used[j]) continue;
      if ((G.row(j) + G.row(i)).norm() <= 1e-12 * G.row(i).norm()) {
        pairs.emplace_back(i, j);
        used[i] = used[j] = true;
        break;
      }
    }
  }
  return pairs;
}

// Picks ~2.5% of the rows as conflict pairs (at least one), spread across
// the available pair list.
ConflictSpec MakeConflictSpec(const RobotQP& qp) {
  const auto pairs = OpposingPairs(qp.G);
  ConflictSpec spec;
  if (pairs.empty()) return spec;
  const int want = std::max(
      1, static_cast<int>(std::lround(0.025 * static_cast<double>(
                                          qp.G.rows()))));
  const int count = std::min<int>(want, static_cast<int>(pairs.size()));
  for (int k = 0; k < count; ++k) {
    spec.pairs.push_back(
        pairs[static_cast<std::size_t>(k) * pairs.size() / count]);
  }
  return spec;
}

// The conflict variant of a sequence: per tick, collapse each selected
// pair to an empty interval of width `gap` centered on the feasible
// tick's solution value v* = g^T x* (from a warm-started elastic solve of
// the unmodified tick, off the clock), so the minimal violation is `gap`
// and it occurs right where the control loop was already operating.
Sequence MakeConflictSequence(const Sequence& seq, const ConflictSpec& spec) {
  Sequence out = seq;
  elastiqp::Solver solver;
  solver.settings.eps_abs = kEps;
  solver.settings.eps_rel = 0;
  solver.settings.warm_start = true;
  const RobotQP& first = seq.front();
  solver.setup(first.Q, first.q, first.A, first.b, first.G, first.h,
               first.penalty);
  for (std::size_t k = 0; k < seq.size(); ++k) {
    const RobotQP& qp = seq[k];
    solver.set_Q(qp.Q);
    solver.set_q(qp.q);
    if (qp.b.size() > 0) {
      solver.set_A(qp.A);
      solver.set_b(qp.b);
    }
    solver.set_G(qp.G);
    solver.set_h(qp.h);
    const auto& sol = solver.solve();
    for (const auto& pr : spec.pairs) {
      const double v_star = qp.G.row(pr.first).dot(sol.x);
      const double gap =
          kGapFrac * std::max(1.0, std::abs(qp.h[pr.first]));
      out[k].h[pr.first] = v_star - 0.5 * gap;
      out[k].h[pr.second] = -(v_star + 0.5 * gap);
    }
  }
  return out;
}

// ------------------------------------------------------------------ stats

struct RouteStats {
  std::vector<double> tick_us;
  double mean_us = 0, std_us = 0, p50_us = 0, p95_us = 0, max_us = 0;
  double iters = 0;
  long iter_total = 0;
  int solved = 0;    // usable primal returned (incl. closest-feasible)
  int infeas = 0;    // infeasibility reported, no usable primal
  int other = 0;     // max-iter / anything else
  // Violation structure of the returned x, averaged over ticks WITH a
  // usable primal; worst-cases tracked separately.
  double viol_rows = 0, spurious_rows = 0, viol_l1 = 0, viol_linf = 0;
  double worst_eq = 0;
  int quality_ticks = 0;

  void AddTime(double us) { tick_us.push_back(us); }
  void AddQuality(const RobotQP& qp, const VectorXd& x,
                  const ConflictSpec& spec) {
    const VectorXd v = (qp.G * x - qp.h).cwiseMax(0.0);
    int nnz = 0, spurious = 0;
    for (int i = 0; i < v.size(); ++i) {
      if (v[i] > kViolThresh) {
        nnz++;
        if (!spec.Contains(i)) spurious++;
      }
    }
    viol_rows += nnz;
    spurious_rows += spurious;
    viol_l1 += v.lpNorm<1>();
    viol_linf += v.size() ? v.lpNorm<Eigen::Infinity>() : 0.0;
    if (qp.b.size() > 0) {
      worst_eq = std::max(worst_eq,
                          (qp.A * x - qp.b).lpNorm<Eigen::Infinity>());
    }
    quality_ticks++;
  }
  void Finalize() {
    const auto ticks = static_cast<double>(tick_us.size());
    if (ticks == 0) return;
    double sum = 0, sum2 = 0;
    for (double us : tick_us) {
      sum += us;
      sum2 += us * us;
    }
    mean_us = sum / ticks;
    std_us = std::sqrt(std::max(0.0, sum2 / ticks - mean_us * mean_us));
    std::vector<double> sorted = tick_us;
    std::sort(sorted.begin(), sorted.end());
    p50_us = sorted[sorted.size() / 2];
    p95_us = sorted[static_cast<std::size_t>(
        0.95 * static_cast<double>(sorted.size() - 1) + 0.5)];
    max_us = sorted.back();
    iters = static_cast<double>(iter_total) / ticks;
    if (quality_ticks > 0) {
      viol_rows /= quality_ticks;
      spurious_rows /= quality_ticks;
      viol_l1 /= quality_ticks;
      viol_linf /= quality_ticks;
    }
  }
};

// ---------------------------------------------------------------- elastiqp

RouteStats RunElastiqp(const Sequence& seq, const ConflictSpec& spec,
                       bool warm) {
  elastiqp::Solver solver;
  solver.settings.eps_abs = kEps;
  solver.settings.eps_rel = 0;
  solver.settings.eps_duality_gap_abs = kEps;
  solver.settings.eps_duality_gap_rel = 0;
  solver.settings.warm_start = warm;
  const RobotQP& first = seq.front();
  solver.setup(first.Q, first.q, first.A, first.b, first.G, first.h,
               first.penalty);
  RouteStats st;
  for (const RobotQP& qp : seq) {
    solver.set_Q(qp.Q);
    solver.set_q(qp.q);
    if (qp.b.size() > 0) {
      solver.set_A(qp.A);
      solver.set_b(qp.b);
    }
    solver.set_G(qp.G);
    solver.set_h(qp.h);
    const auto t0 = steady_clock::now();
    const elastiqp::Solution& sol = solver.solve();
    const auto t1 = steady_clock::now();
    st.AddTime(duration<double, std::micro>(t1 - t0).count());
    st.iter_total += sol.iters;
    if (sol.converged == 1) {
      st.solved++;
      st.AddQuality(qp, sol.x, spec);
    } else {
      st.other++;
    }
  }
  st.Finalize();
  return st;
}

// -------------------------------------------------------------------- piqp

RouteStats RunPiqpHard(const Sequence& seq, const ConflictSpec& spec) {
  const Eigen::Index m = seq.front().b.size();
  piqp::DenseSolver<double> solver;
  solver.settings().eps_abs = kEps;
  solver.settings().eps_rel = 0;
  solver.settings().eps_duality_gap_abs = kEps;
  solver.settings().eps_duality_gap_rel = 0;
  RouteStats st;
  bool first = true;
  for (const RobotQP& qp : seq) {
    const auto t0 = steady_clock::now();
    piqp::Status status;
    if (first) {
      if (m > 0) {
        solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, piqp::nullopt, qp.h,
                     piqp::nullopt, piqp::nullopt);
      } else {
        solver.setup(qp.Q, qp.q, piqp::nullopt, piqp::nullopt, qp.G,
                     piqp::nullopt, qp.h, piqp::nullopt, piqp::nullopt);
      }
      first = false;
    } else {
      if (m > 0) {
        solver.update(qp.Q, qp.q, qp.A, qp.b, qp.G, piqp::nullopt, qp.h);
      } else {
        solver.update(qp.Q, qp.q, piqp::nullopt, piqp::nullopt, qp.G,
                      piqp::nullopt, qp.h);
      }
    }
    status = solver.solve();
    const auto t1 = steady_clock::now();
    st.AddTime(duration<double, std::micro>(t1 - t0).count());
    st.iter_total += solver.result().info.iter;
    if (status == piqp::PIQP_SOLVED) {
      st.solved++;
      st.AddQuality(qp, solver.result().x, spec);
    } else if (status == piqp::PIQP_PRIMAL_INFEASIBLE ||
               status == piqp::PIQP_DUAL_INFEASIBLE) {
      st.infeas++;
    } else {
      st.other++;
    }
  }
  st.Finalize();
  return st;
}

// Expanded (n+p)-variable l1-slack data, shared by the piqp/proxqp slack
// routes.
struct Expanded {
  MatrixXd H, Aexp, C;
  VectorXd g, u, lb;

  void Resize(Eigen::Index n, Eigen::Index m, Eigen::Index p) {
    H = MatrixXd::Zero(n + p, n + p);
    g.resize(n + p);
    Aexp = MatrixXd::Zero(m, n + p);
    C = MatrixXd::Zero(p, n + p);
    C.rightCols(p) = -MatrixXd::Identity(p, p);
    u.resize(p);
    lb = VectorXd::Constant(n + p, -kInf);
    lb.tail(p).setZero();
  }

  void Fill(const RobotQP& qp) {
    const Eigen::Index n = qp.q.size();
    const Eigen::Index p = qp.h.size();
    H.topLeftCorner(n, n) = qp.Q;
    g.head(n) = qp.q;
    g.tail(p) = qp.penalty;
    if (qp.b.size() > 0) Aexp.leftCols(n) = qp.A;
    C.leftCols(n) = qp.G;
    u = qp.h;
  }
};

RouteStats RunPiqpSlack(const Sequence& seq, const ConflictSpec& spec) {
  const Eigen::Index n = seq.front().q.size();
  const Eigen::Index m = seq.front().b.size();
  const Eigen::Index p = seq.front().h.size();
  Expanded e;
  e.Resize(n, m, p);
  piqp::DenseSolver<double> solver;
  solver.settings().eps_abs = kEps;
  solver.settings().eps_rel = 0;
  solver.settings().eps_duality_gap_abs = kEps;
  solver.settings().eps_duality_gap_rel = 0;
  RouteStats st;
  bool first = true;
  for (const RobotQP& qp : seq) {
    e.Fill(qp);
    const auto t0 = steady_clock::now();
    piqp::Status status;
    if (first) {
      if (m > 0) {
        solver.setup(e.H, e.g, e.Aexp, qp.b, e.C, piqp::nullopt, e.u, e.lb,
                     piqp::nullopt);
      } else {
        solver.setup(e.H, e.g, piqp::nullopt, piqp::nullopt, e.C,
                     piqp::nullopt, e.u, e.lb, piqp::nullopt);
      }
      first = false;
    } else {
      if (m > 0) {
        solver.update(e.H, e.g, e.Aexp, qp.b, e.C, piqp::nullopt, e.u, e.lb);
      } else {
        solver.update(e.H, e.g, piqp::nullopt, piqp::nullopt, e.C,
                      piqp::nullopt, e.u, e.lb);
      }
    }
    status = solver.solve();
    const auto t1 = steady_clock::now();
    st.AddTime(duration<double, std::micro>(t1 - t0).count());
    st.iter_total += solver.result().info.iter;
    if (status == piqp::PIQP_SOLVED) {
      st.solved++;
      st.AddQuality(qp, solver.result().x.head(n), spec);
    } else if (status == piqp::PIQP_PRIMAL_INFEASIBLE ||
               status == piqp::PIQP_DUAL_INFEASIBLE) {
      st.infeas++;
    } else {
      st.other++;
    }
  }
  st.Finalize();
  return st;
}

// ------------------------------------------------------------------ proxqp

// mode: 0 = hard vanilla, 1 = hard closest-feasible, 2 = expanded slack.
//
// Modes 0 and 2 keep a persistent solver (init once, update per tick,
// workspace/preconditioner reuse) like the other routes. Mode 1 builds a
// fresh solver per tick: primal_infeasibility_solving through the
// update() path misbehaves (the closest-feasible phase spins for ~1e5
// inner iterations, ignoring max_iter, and still reports plain
// PROXQP_PRIMAL_INFEASIBLE), while a fresh init per solve — matching the
// proxsuite examples — behaves; its per-tick times therefore include the
// init.
RouteStats RunProxqp(const Sequence& seq, const ConflictSpec& spec,
                     int mode) {
  namespace pq = proxsuite::proxqp;
  const Eigen::Index n = seq.front().q.size();
  const Eigen::Index m = seq.front().b.size();
  const Eigen::Index p = seq.front().h.size();
  const bool slack = mode == 2;
  const bool fresh = mode == 1;
  const VectorXd l = VectorXd::Constant(p, -kInf);

  Expanded e;
  if (slack) e.Resize(n, m, p);

  auto make = [&] {
    auto qp = std::make_unique<pq::dense::QP<double>>(
        slack ? n + p : n, m, p, /*box_constraints=*/slack);
    qp->settings.eps_abs = kEps;
    qp->settings.eps_rel = 0;
    qp->settings.max_iter = kProxMaxIter;
    // The duality-gap check is unattainable in the closest-feasible phase,
    // so run that mode without it.
    qp->settings.check_duality_gap = mode != 1;
    qp->settings.eps_duality_gap_abs = kEps;
    qp->settings.eps_duality_gap_rel = 0;
    qp->settings.initial_guess =
        pq::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    if (mode == 1) qp->settings.primal_infeasibility_solving = true;
    if (slack) {
      // The expanded elastic problem is feasible by construction, but
      // ProxQP's default infeasibility heuristics (eps 1e-4) can
      // false-positive on it (observed on these sequences), so tighten
      // them to machine precision.
      qp->settings.eps_primal_inf = 1e-14;
      qp->settings.eps_dual_inf = 1e-14;
    }
    return qp;
  };

  RouteStats st;
  bool first = true;
  const VectorXd ub_box = VectorXd::Constant(n + p, kInf);
  auto qp = make();
  for (const RobotQP& tick : seq) {
    if (slack) e.Fill(tick);
    const auto t0 = steady_clock::now();
    if (fresh && !first) qp = make();
    if (slack) {
      if (first) {
        if (m > 0) {
          qp->init(e.H, e.g, e.Aexp, tick.b, e.C, l, e.u, e.lb, ub_box);
        } else {
          qp->init(e.H, e.g, proxsuite::nullopt, proxsuite::nullopt, e.C, l,
                   e.u, e.lb, ub_box);
        }
      } else {
        if (m > 0) {
          qp->update(e.H, e.g, e.Aexp, tick.b, e.C, l, e.u, e.lb, ub_box);
        } else {
          qp->update(e.H, e.g, proxsuite::nullopt, proxsuite::nullopt, e.C,
                     l, e.u, e.lb, ub_box);
        }
      }
    } else {
      if (first || fresh) {
        if (m > 0) {
          qp->init(tick.Q, tick.q, tick.A, tick.b, tick.G, l, tick.h);
        } else {
          qp->init(tick.Q, tick.q, proxsuite::nullopt, proxsuite::nullopt,
                   tick.G, l, tick.h);
        }
      } else {
        if (m > 0) {
          qp->update(tick.Q, tick.q, tick.A, tick.b, tick.G, l, tick.h);
        } else {
          qp->update(tick.Q, tick.q, proxsuite::nullopt, proxsuite::nullopt,
                     tick.G, l, tick.h);
        }
      }
    }
    first = false;
    qp->solve();
    const auto t1 = steady_clock::now();
    st.AddTime(duration<double, std::micro>(t1 - t0).count());
    st.iter_total += qp->results.info.iter;
    const auto status = qp->results.info.status;
    const bool usable =
        status == pq::QPSolverOutput::PROXQP_SOLVED ||
        status == pq::QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE;
    if (usable) {
      st.solved++;
      st.AddQuality(tick,
                    slack ? VectorXd(qp->results.x.head(n)) : qp->results.x,
                    spec);
    } else if (status == pq::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE ||
               status == pq::QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
      st.infeas++;
    } else {
      st.other++;
    }
  }
  st.Finalize();
  return st;
}

// ------------------------------------------------------------------ report

struct Row {
  std::string scenario, variant, route;
  int n, m, p, ticks, conflict_rows;
  RouteStats st;
};

void PrintRow(const Row& r) {
  std::printf(
      "%-8s %-8s %-13s | %9.1f +- %8.1f us (p95 %9.1f) %5.1f it | "
      "ok %3d inf %3d oth %3d | viol %5.1f (spur %4.1f) l1 %8.3f "
      "linf %7.3f | eq %.0e\n",
      r.scenario.c_str(), r.variant.c_str(), r.route.c_str(), r.st.mean_us,
      r.st.std_us, r.st.p95_us, r.st.iters, r.st.solved, r.st.infeas,
      r.st.other, r.st.viol_rows, r.st.spurious_rows, r.st.viol_l1,
      r.st.viol_linf, r.st.worst_eq);
}

void WriteCsv(const std::string& dir, const std::vector<Row>& rows) {
  const std::string path = dir + "/robot_multisolver.csv";
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    return;
  }
  std::fprintf(
      f,
      "scenario,variant,route,arch,n,m,p,ticks,conflict_rows,mean_us,std_us,"
      "p50_us,p95_us,max_us,mean_iters,solved,infeasible_reported,other,"
      "viol_rows_mean,spurious_viol_rows_mean,viol_l1_mean,viol_linf_mean,"
      "worst_eq_residual\n");
  for (const Row& r : rows) {
    std::fprintf(f,
                 "%s,%s,%s,%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,"
                 "%d,%d,%d,%.3f,%.3f,%.6f,%.6f,%.3e\n",
                 r.scenario.c_str(), r.variant.c_str(), r.route.c_str(),
                 ELASTIQP_ARCH_LABEL, r.n, r.m, r.p, r.ticks, r.conflict_rows,
                 r.st.mean_us, r.st.std_us, r.st.p50_us, r.st.p95_us,
                 r.st.max_us, r.st.iters, r.st.solved, r.st.infeas,
                 r.st.other, r.st.viol_rows, r.st.spurious_rows, r.st.viol_l1,
                 r.st.viol_linf, r.st.worst_eq);
  }
  std::fclose(f);
  std::printf("wrote %s\n", path.c_str());
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
#ifdef ELASTIQP_SEQUENCE_FILE
    path = ELASTIQP_SEQUENCE_FILE;
#else
    path = "robot_sequences.bin";
#endif
  }
  std::vector<NamedSequence> seqs;
  try {
    seqs = robot_control::LoadSequences(path);
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "%s\nGenerate the sequence file first: gen_robot_sequences "
                 "%s\n",
                 e.what(), path.c_str());
    return 1;
  }
  std::printf(
      "Cross-solver robot control replay, feasible + conflict variants "
      "[arch: %s]\n"
      "(eps=%g, conflict gap = %.0f%% of the row bound, ~2.5%% of rows "
      "conflicting; %s)\n\n",
      ELASTIQP_ARCH_LABEL, kEps, 100 * kGapFrac, path.c_str());

  std::vector<Row> rows;
  for (const NamedSequence& seq : seqs) {
    const RobotQP& first = seq.qps.front();
    const int n = static_cast<int>(first.q.size());
    const int m = static_cast<int>(first.b.size());
    const int p = static_cast<int>(first.h.size());
    const ConflictSpec spec = MakeConflictSpec(first);
    const ConflictSpec no_conflict;  // feasible variant: empty conflict set
    const Sequence conflict = MakeConflictSequence(seq.qps, spec);

    auto add = [&](const std::string& variant, const std::string& route,
                   const ConflictSpec& sp, RouteStats st) {
      rows.push_back(Row{seq.name, variant, route, n, m, p,
                         static_cast<int>(seq.qps.size()),
                         2 * static_cast<int>(spec.pairs.size()),
                         std::move(st)});
      PrintRow(rows.back());
    };

    for (int variant = 0; variant < 2; ++variant) {
      const bool conf = variant == 1;
      const Sequence& s = conf ? conflict : seq.qps;
      const ConflictSpec& sp = conf ? spec : no_conflict;
      const char* vname = conf ? "conflict" : "feasible";
      add(vname, "elastiqp", sp, RunElastiqp(s, sp, /*warm=*/false));
      add(vname, "elastiqp-warm", sp, RunElastiqp(s, sp, /*warm=*/true));
      add(vname, "piqp-hard", sp, RunPiqpHard(s, sp));
      add(vname, "piqp-slack", sp, RunPiqpSlack(s, sp));
      add(vname, "proxqp-hard", sp, RunProxqp(s, sp, 0));
      add(vname, "proxqp-clfeas", sp, RunProxqp(s, sp, 1));
      add(vname, "proxqp-slack", sp, RunProxqp(s, sp, 2));
      std::fflush(stdout);
    }
    std::printf("\n");
  }
  if (!csv_dir.empty()) WriteCsv(csv_dir, rows);
  return 0;
}
