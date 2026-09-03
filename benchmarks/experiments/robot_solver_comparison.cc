// Cross-solver comparison on the recorded robot control loops (diff-ik,
// arm-osc, hum-wbc, biman-ik): every route to a usable answer, on the feasible
// sequences and on a conflict variant of each, in one binary. Produces the
// data for the combined feasible/infeasible table in the paper.
//
// Conflict variant: one row (fixed per scenario) is tightened at every tick
// to v_min - gap, where v_min is the smallest value of that row the REMAINING
// constraints admit (an auxiliary LP). Violating the tightened row alone by
// `gap` restores feasibility, so every other violated row is shift that did
// not need to happen. The row is the first one of the softest penalty class
// that is held by several other constraints, so the conflict is genuinely
// multi-row (a row held by a single parallel partner is a one-dimensional
// conflict on which every norm agrees); diff-ik has only single-variable
// bounds, so it falls back to the first soft row.
//
// Usage: robot_solver_comparison [sequence_file] [--csv <dir>]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/SparseCore>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/sparse/sparse.hpp>

#include "elastiqp/elastiqp.hpp"
#include "piqp/piqp.hpp"
#include "qp_io.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using robot_control::NamedSequence;
using robot_control::RobotQP;
using Sequence = std::vector<RobotQP>;
namespace pq = proxsuite::proxqp;

namespace {

constexpr double kEps = 1e-6;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kViolThresh = 1e-4;
constexpr double kGapFrac = 0.05;
constexpr int kProxMaxIter = 1000;

double Now() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch())
      .count();
}

// ----------------------------------------------------------- conflict gen

// min g_a^T x over the other constraints (tiny Tikhonov term keeps the LP
// strictly convex for PIQP). Returns v_min and the number of rows holding it.
struct LpResult {
  double v_min;
  int holders;
};

LpResult TightestValue(const RobotQP& qp, int a) {
  const int n = static_cast<int>(qp.q.size()), p = static_cast<int>(qp.h.size());
  MatrixXd G(p - 1, n);
  VectorXd h(p - 1);
  G << qp.G.topRows(a), qp.G.bottomRows(p - a - 1);
  h << qp.h.head(a), qp.h.tail(p - a - 1);
  piqp::DenseSolver<double> lp;
  lp.settings().verbose = false;
  lp.settings().eps_abs = 1e-8;
  lp.settings().eps_rel = 0;
  lp.settings().max_iter = 2000;
  lp.setup(1e-6 * MatrixXd::Identity(n, n), qp.G.row(a).transpose(), qp.A,
           qp.b, G, piqp::nullopt, h, piqp::nullopt, piqp::nullopt);
  if (lp.solve() != piqp::PIQP_SOLVED) return {-kInf, 0};
  const VectorXd& z = lp.result().z_u;
  const double z_max = z.maxCoeff();
  return {qp.G.row(a).dot(lp.result().x),
          static_cast<int>((z.array() > 0.01 * z_max).count())};
}

int ChooseConflictRow(const RobotQP& qp) {
  const double soft = qp.penalty.minCoeff();
  int fallback = -1;
  for (int a = 0; a < qp.h.size(); ++a) {
    if (qp.penalty[a] > soft) continue;
    if (fallback < 0) fallback = a;
    if (TightestValue(qp, a).holders >= 2) return a;
  }
  return fallback;
}

Sequence MakeConflictSequence(const Sequence& seq, int a) {
  Sequence out = seq;
  for (std::size_t k = 0; k < seq.size(); ++k) {
    const double gap = kGapFrac * std::max(1.0, std::abs(seq[k].h[a]));
    out[k].h[a] = TightestValue(seq[k], a).v_min - gap;
  }
  return out;
}

// ------------------------------------------------------------------ stats

struct RouteStats {
  std::vector<double> us;
  long iters = 0;
  int solved = 0, infeasible = 0, other = 0;
  // Violation structure of the returned x, averaged over ticks with one.
  int primal_ticks = 0;
  double viol_rows = 0, spurious_rows = 0, viol_l1 = 0, worst_eq = 0;

  void Add(double t, int it) {
    us.push_back(t);
    iters += it;
  }
  void AddPrimal(const RobotQP& qp, const VectorXd& x, int conflict_row) {
    primal_ticks++;
    const VectorXd v = (qp.G * x - qp.h).cwiseMax(0.0);
    for (int i = 0; i < v.size(); ++i) {
      if (v[i] <= kViolThresh) continue;
      viol_rows++;
      if (i != conflict_row) spurious_rows++;
    }
    viol_l1 += v.lpNorm<1>();
    if (qp.b.size() > 0)
      worst_eq = std::max(worst_eq, (qp.A * x - qp.b).lpNorm<Eigen::Infinity>());
  }
  double Mean() const {
    double s = 0;
    for (double t : us) s += t;
    return s / us.size();
  }
  double Percentile(double f) const {
    std::vector<double> v = us;
    std::sort(v.begin(), v.end());
    return v[static_cast<std::size_t>(f * (v.size() - 1) + 0.5)];
  }
  void Finalize() {
    if (primal_ticks == 0) return;
    viol_rows /= primal_ticks;
    spurious_rows /= primal_ticks;
    viol_l1 /= primal_ticks;
  }
};

// ----------------------------------------------------------------- routes

RouteStats RunElastiqp(const Sequence& seq, int conflict_row, bool warm) {
  elastiqp::Solver solver;
  solver.settings.eps_abs = kEps;
  solver.settings.eps_rel = 0;
  solver.settings.eps_duality_gap_abs = kEps;
  solver.settings.eps_duality_gap_rel = 0;
  solver.settings.warm_start = warm;
  const RobotQP& f = seq.front();
  solver.setup(f.Q, f.q, f.A, f.b, f.G, f.h, f.penalty);
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
    const double t0 = Now();
    const elastiqp::Solution& sol = solver.solve();
    st.Add(Now() - t0, sol.iters);
    if (sol.converged == 1) {
      st.solved++;
      st.AddPrimal(qp, sol.x, conflict_row);
    } else {
      st.other++;
    }
  }
  st.Finalize();
  return st;
}

// Expanded (n+p)-variable l1-slack formulation: x' = [x; t],
// minimize 0.5 x'Q x + q'x + penalty't, s.t. A x = b, G x - t <= h, t >= 0.
struct Expanded {
  MatrixXd H, A, C;
  VectorXd g, u, lb, ub;
  int n, p;

  Expanded(const RobotQP& qp)
      : n(static_cast<int>(qp.q.size())), p(static_cast<int>(qp.h.size())) {
    const int m = static_cast<int>(qp.b.size());
    H = MatrixXd::Zero(n + p, n + p);
    g.resize(n + p);
    A = MatrixXd::Zero(m, n + p);
    C = MatrixXd::Zero(p, n + p);
    C.rightCols(p) = -MatrixXd::Identity(p, p);
    lb = VectorXd::Constant(n + p, -kInf);
    lb.tail(p).setZero();
    ub = VectorXd::Constant(n + p, kInf);
  }
  void Fill(const RobotQP& qp) {
    H.topLeftCorner(n, n) = qp.Q;
    g.head(n) = qp.q;
    g.tail(p) = qp.penalty;
    A.leftCols(n) = qp.A;
    C.leftCols(n) = qp.G;
    u = qp.h;
  }
};

RouteStats RunPiqp(const Sequence& seq, int conflict_row, bool slack) {
  Expanded e(seq.front());
  piqp::DenseSolver<double> solver;
  solver.settings().verbose = false;
  solver.settings().eps_abs = kEps;
  solver.settings().eps_rel = 0;
  solver.settings().eps_duality_gap_abs = kEps;
  solver.settings().eps_duality_gap_rel = 0;
  RouteStats st;
  bool first = true;
  for (const RobotQP& qp : seq) {
    if (slack) e.Fill(qp);
    const double t0 = Now();
    if (slack && first) {
      solver.setup(e.H, e.g, e.A, qp.b, e.C, piqp::nullopt, e.u, e.lb, piqp::nullopt);
    } else if (slack) {
      solver.update(e.H, e.g, e.A, qp.b, e.C, piqp::nullopt, e.u, e.lb);
    } else if (first) {
      solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, piqp::nullopt, qp.h, piqp::nullopt,
                   piqp::nullopt);
    } else {
      solver.update(qp.Q, qp.q, qp.A, qp.b, qp.G, piqp::nullopt, qp.h);
    }
    first = false;
    const piqp::Status status = solver.solve();
    st.Add(Now() - t0, solver.result().info.iter);
    if (status == piqp::PIQP_SOLVED) {
      st.solved++;
      st.AddPrimal(qp, solver.result().x.head(e.n), conflict_row);
    } else if (status == piqp::PIQP_PRIMAL_INFEASIBLE) {
      st.infeasible++;
    } else {
      st.other++;
    }
  }
  st.Finalize();
  return st;
}

// Sparse view of the expanded formulation. The dense blocks (Q, A, G) are
// stored with every entry explicit, zeros included, so the sparsity pattern is
// identical on every tick: ProxQP-sparse's update() compares the pattern with
// the one given to init() and silently keeps the old matrices when it differs.
// The slack columns are the only structural sparsity ([-I] in C, empty in H).
// The sparse API has no box block, so t >= 0 is p further rows [0 I] of C.
struct SparseExpanded {
  using Mat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
  Mat H, A, C;
  VectorXd g, l, u;
  int n, p;

  SparseExpanded(const RobotQP& qp)
      : n(static_cast<int>(qp.q.size())), p(static_cast<int>(qp.h.size())) {
    const int m = static_cast<int>(qp.b.size());
    H.resize(n + p, n + p);
    A.resize(m, n + p);
    C.resize(2 * p, n + p);
    g.resize(n + p);
    l = VectorXd::Constant(2 * p, -kInf);
    l.tail(p).setZero();
    u = VectorXd::Constant(2 * p, kInf);
    Fill(qp);
  }
  static void DenseBlock(std::vector<Eigen::Triplet<double>>& t, const MatrixXd& M) {
    for (int j = 0; j < M.cols(); ++j)
      for (int i = 0; i < M.rows(); ++i) t.emplace_back(i, j, M(i, j));
  }
  void Fill(const RobotQP& qp) {
    std::vector<Eigen::Triplet<double>> t;
    DenseBlock(t, qp.Q);
    H.setFromTriplets(t.begin(), t.end());
    t.clear();
    DenseBlock(t, qp.A);
    A.setFromTriplets(t.begin(), t.end());
    t.clear();
    DenseBlock(t, qp.G);
    for (int i = 0; i < p; ++i) {
      t.emplace_back(i, n + i, -1.0);
      t.emplace_back(p + i, n + i, 1.0);
    }
    C.setFromTriplets(t.begin(), t.end());
    g.head(n) = qp.q;
    g.tail(p) = qp.penalty;
    u.head(p) = qp.h;
  }
};

enum class ProxMode { kHard, kClosestFeasible, kSlack };

// Closest-feasible mode misbehaves through update() (spins past max_iter in
// the closest-feasible phase and still reports plain infeasibility), so that
// route builds a fresh solver per tick, as in the proxsuite examples; its
// per-tick time includes init. In that mode the iterate converges to the
// closest-feasible (min l2 shift) solution whether or not ProxQP certifies
// it, so its violation structure is recorded on every tick and the
// certification rate is reported separately via solved/infeasible.
RouteStats RunProxqp(const Sequence& seq, int conflict_row, ProxMode mode,
                     bool warm) {
  Expanded e(seq.front());
  const bool slack = mode == ProxMode::kSlack;
  const bool fresh = mode == ProxMode::kClosestFeasible;
  const int m = static_cast<int>(seq.front().b.size());
  const VectorXd l = VectorXd::Constant(e.p, -kInf);
  auto make = [&] {
    auto qp = std::make_unique<pq::dense::QP<double>>(slack ? e.n + e.p : e.n, m,
                                                      e.p, slack);
    qp->settings.eps_abs = kEps;
    qp->settings.eps_rel = 0;
    qp->settings.max_iter = kProxMaxIter;
    qp->settings.check_duality_gap = !fresh;
    qp->settings.eps_duality_gap_abs = kEps;
    qp->settings.eps_duality_gap_rel = 0;
    qp->settings.primal_infeasibility_solving = fresh;
    qp->settings.initial_guess =
        warm ? pq::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT
             : pq::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    if (slack) {  // feasible by construction; disable the infeasibility heuristics
      qp->settings.eps_primal_inf = 1e-14;
      qp->settings.eps_dual_inf = 1e-14;
    }
    return qp;
  };
  RouteStats st;
  auto qp = make();
  bool first = true;
  for (const RobotQP& tick : seq) {
    if (slack) e.Fill(tick);
    const double t0 = Now();
    if (fresh && !first) qp = make();
    if (slack && (first || fresh)) {
      qp->init(e.H, e.g, e.A, tick.b, e.C, l, e.u, e.lb, e.ub);
    } else if (slack) {
      qp->update(e.H, e.g, e.A, tick.b, e.C, l, e.u, e.lb, e.ub);
    } else if (first || fresh) {
      qp->init(tick.Q, tick.q, tick.A, tick.b, tick.G, l, tick.h);
    } else {
      qp->update(tick.Q, tick.q, tick.A, tick.b, tick.G, l, tick.h);
    }
    first = false;
    qp->solve();
    st.Add(Now() - t0, static_cast<int>(qp->results.info.iter));
    const auto status = qp->results.info.status;
    const bool certified =
        status == pq::QPSolverOutput::PROXQP_SOLVED ||
        status == pq::QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE;
    if (certified) {
      st.solved++;
    } else if (status == pq::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
      st.infeasible++;
    } else {
      st.other++;
    }
    if (certified || fresh) st.AddPrimal(tick, qp->results.x.head(e.n), conflict_row);
  }
  st.Finalize();
  return st;
}

// ProxQP sparse backend on the expanded l1-slack formulation.
RouteStats RunProxqpSparse(const Sequence& seq, int conflict_row, bool warm) {
  SparseExpanded e(seq.front());
  const int m = static_cast<int>(seq.front().b.size());
  pq::sparse::QP<double, int> qp(e.n + e.p, m, 2 * e.p);
  qp.settings.eps_abs = kEps;
  qp.settings.eps_rel = 0;
  qp.settings.max_iter = kProxMaxIter;
  qp.settings.check_duality_gap = true;
  qp.settings.eps_duality_gap_abs = kEps;
  qp.settings.eps_duality_gap_rel = 0;
  qp.settings.eps_primal_inf = 1e-14;  // feasible by construction
  qp.settings.eps_dual_inf = 1e-14;
  qp.settings.initial_guess =
      warm ? pq::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT
           : pq::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  RouteStats st;
  bool first = true;
  for (const RobotQP& tick : seq) {
    e.Fill(tick);
    const double t0 = Now();
    if (first) {
      qp.init(e.H, e.g, e.A, tick.b, e.C, e.l, e.u);
    } else {
      qp.update(e.H, e.g, e.A, tick.b, e.C, e.l, e.u);
    }
    first = false;
    qp.solve();
    st.Add(Now() - t0, static_cast<int>(qp.results.info.iter));
    const auto status = qp.results.info.status;
    if (status == pq::QPSolverOutput::PROXQP_SOLVED) {
      st.solved++;
      st.AddPrimal(tick, qp.results.x.head(e.n), conflict_row);
    } else if (status == pq::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
      st.infeasible++;
    } else {
      st.other++;
    }
  }
  st.Finalize();
  return st;
}

// ----------------------------------------------------------------- report

struct Row {
  std::string scenario, variant, route;
  RouteStats st;
};

void Print(const Row& r) {
  std::printf(
      "%-8s %-8s %-18s | %9.1f us (p95 %9.1f) %5.1f it | ok %3d inf %3d "
      "oth %3d | viol %5.1f spur %5.1f l1 %8.3f | eq %.0e\n",
      r.scenario.c_str(), r.variant.c_str(), r.route.c_str(), r.st.Mean(),
      r.st.Percentile(0.95), static_cast<double>(r.st.iters) / r.st.us.size(),
      r.st.solved, r.st.infeasible, r.st.other, r.st.viol_rows,
      r.st.spurious_rows, r.st.viol_l1, r.st.worst_eq);
}

void WriteCsv(const std::string& path, const std::vector<Row>& rows) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return;
  std::fprintf(f,
               "scenario,variant,route,ticks,mean_us,p50_us,p95_us,max_us,"
               "mean_iters,solved,infeasible,other,viol_rows,spurious_rows,"
               "viol_l1,worst_eq\n");
  for (const Row& r : rows) {
    std::fprintf(f, "%s,%s,%s,%zu,%.3f,%.3f,%.3f,%.3f,%.2f,%d,%d,%d,%.3f,%.3f,%.6f,%.3e\n",
                 r.scenario.c_str(), r.variant.c_str(), r.route.c_str(),
                 r.st.us.size(), r.st.Mean(), r.st.Percentile(0.5),
                 r.st.Percentile(0.95), r.st.Percentile(1.0),
                 static_cast<double>(r.st.iters) / r.st.us.size(), r.st.solved,
                 r.st.infeasible, r.st.other, r.st.viol_rows, r.st.spurious_rows,
                 r.st.viol_l1, r.st.worst_eq);
  }
  std::fclose(f);
  std::printf("wrote %s\n", path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  std::string path = ELASTIQP_SEQUENCE_FILE, csv_dir;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--csv" && i + 1 < argc) {
      csv_dir = argv[++i];
    } else {
      path = argv[i];
    }
  }
  const std::vector<NamedSequence> seqs = robot_control::LoadSequences(path);
  std::printf("eps=%g, conflict gap = %.0f%% of the row bound\n\n", kEps,
              100 * kGapFrac);

  std::vector<Row> rows;
  for (const NamedSequence& seq : seqs) {
    const int conflict_row = ChooseConflictRow(seq.qps.front());
    std::printf("%s: tightened row %d\n", seq.name.c_str(), conflict_row);
    const Sequence conflict_seq = MakeConflictSequence(seq.qps, conflict_row);

    for (int variant = 0; variant < 2; ++variant) {
      const bool conf = variant == 1;
      const Sequence& s = conf ? conflict_seq : seq.qps;
      const int row_c = conf ? conflict_row : -1;
      auto add = [&](const std::string& route, RouteStats st) {
        rows.push_back(Row{seq.name, conf ? "conflict" : "feasible", route, std::move(st)});
        Print(rows.back());
      };
      add("elastiqp", RunElastiqp(s, row_c, false));
      add("elastiqp-warm", RunElastiqp(s, row_c, true));
      add("piqp-hard", RunPiqp(s, row_c, false));
      add("piqp-slack", RunPiqp(s, row_c, true));
      add("proxqp-hard", RunProxqp(s, row_c, ProxMode::kHard, false));
      add("proxqp-hard-warm", RunProxqp(s, row_c, ProxMode::kHard, true));
      add("proxqp-clfeas", RunProxqp(s, row_c, ProxMode::kClosestFeasible, false));
      add("proxqp-slack", RunProxqp(s, row_c, ProxMode::kSlack, false));
      add("proxqp-slack-warm", RunProxqp(s, row_c, ProxMode::kSlack, true));
      add("proxqp-sparse-slack", RunProxqpSparse(s, row_c, false));
      add("proxqp-sparse-slack-warm", RunProxqpSparse(s, row_c, true));
      std::fflush(stdout);
    }
    std::printf("\n");
  }
  if (!csv_dir.empty()) WriteCsv(csv_dir + "/robot_solver_comparison.csv", rows);
  return 0;
}
