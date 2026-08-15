// When does relax() warm starting pay off?
//
// relax(kappa) supports two starts: cold (warm=false, the retraction of
// the tick's tight certificate) and warm (default, the previous relaxed
// point). The cold start has a tiny O(kappa) residual but pays
// ~log2(res0/kappa) linear-rate halvings traversing the barrier
// curvature from the boundary; the warm start has a LARGE residual after
// a data step, but if the drift lives in the linear residual blocks
// Newton removes it in ~2 steps. The warm start's failure mode is a data
// step that flips the smoothed configuration of many rows (which side of
// the s.z = kappa hyperbola a pair sits on): each flipped row re-pays
// the curvature walk. This benchmark maps that trade across problem
// size, structure, penalty tier, drift size, and drift composition, on
// random control-loop-style trajectories (fixed matrices unless G
// drifts; relative-sigma random walk per tick, as in the forward-pass
// warm-start studies).
//
// Per tick: warm forward solve, then relax warm-first (the chain must
// run BEFORE the cold probe: any converged relax refreshes the stored
// chain iterate, so cold-first would hand the warm call this tick's
// already-converged point and measure 0 iterations), then the cold
// probe. Reported per cell (100 ticks):
//   fwd us | cold relax us (it) | warm relax us (it) | speedup |
//   flips/tick (v-sign changes between consecutive relaxed points,
//   t-block + ineq-block) | worst warm-vs-cold |dx| (same-point check) |
//   fails fwd/cold/warm.
//
// Structures: feas (no conflicts), infeas (p/4 conflicting pairs --
// active elastic slacks), degen (half the rows exactly at the boundary
// with zero multiplier -- maximally flip-prone under drift).
// Penalty tiers: 10 (ruiz off) and 1e4 (ruiz on, per the documented
// "penalty >= ~1e4 + differentiation => ruiz" rule).
//
// Usage: bench_relax_warm [--csv <dir>]  (writes relax_warm_cells.csv)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "elastiqp/elastiqp.hpp"
#include "problem_gen.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using problem_gen::QPData;
using std::chrono::duration;
using std::chrono::steady_clock;

namespace {

constexpr int kTicks = 100;
constexpr double kFwdEps = 1e-8;
constexpr double kRelaxTol = 1e-8;
constexpr int kRelaxMaxIter = 50;
constexpr double kKappa = 1e-3;  // headline; see the kappa sweep at the end

enum class Structure { kFeas, kInfeas, kDegen };
enum class Drift { kQ, kQH, kQHG };  // what random-walks per tick

const char* Name(Structure s) {
  switch (s) {
    case Structure::kFeas: return "feas";
    case Structure::kInfeas: return "infeas";
    default: return "degen";
  }
}
const char* Name(Drift d) {
  switch (d) {
    case Drift::kQ: return "q";
    case Drift::kQH: return "qh";
    default: return "qhG";
  }
}

struct Size {
  int n, m, p;
};

// Pre-generated trajectory. G is per-tick only under kQHG (else one copy).
struct Trajectory {
  QPData base;
  VectorXd penalty;
  std::vector<VectorXd> q, h, b;
  std::vector<MatrixXd> G;  // empty unless Drift::kQHG
};

Trajectory MakeTrajectory(Size sz, Structure st, double penalty_w,
                          double sigma, Drift drift, unsigned seed) {
  std::mt19937 rng(seed);
  Trajectory traj;
  switch (st) {
    case Structure::kFeas:
      traj.base = problem_gen::RandomFeasible(rng, sz.n, sz.m, sz.p);
      break;
    case Structure::kInfeas:
      traj.base =
          problem_gen::InfeasibleEq(rng, sz.n, sz.m, sz.p, sz.p / 4);
      break;
    case Structure::kDegen: {
      traj.base = problem_gen::RandomFeasible(rng, sz.n, sz.m, sz.p);
      // Put every other row exactly on the boundary through the
      // construction point, with zero multiplier: a weakly-active tie
      // that drift pushes to either side of the smoothed hyperbola.
      const VectorXd gx = traj.base.G * traj.base.x_opt;
      for (int i = 0; i < sz.p; i += 2) traj.base.h[i] = gx[i];
      break;
    }
  }
  traj.penalty = VectorXd::Constant(sz.p, penalty_w);

  const double qs = sigma * traj.base.q.lpNorm<Eigen::Infinity>();
  const double hs = sigma * traj.base.h.lpNorm<Eigen::Infinity>();
  const double bs =
      sz.m > 0 ? sigma * traj.base.b.lpNorm<Eigen::Infinity>() : 0.0;
  const double Gs = sigma;  // relative, entrywise on G
  std::normal_distribution<double> dist;
  VectorXd q = traj.base.q, h = traj.base.h, b = traj.base.b;
  MatrixXd G = traj.base.G;
  for (int k = 0; k < kTicks; ++k) {
    for (int i = 0; i < sz.n; ++i) q[i] += qs * dist(rng);
    if (drift != Drift::kQ) {
      for (int i = 0; i < sz.p; ++i) h[i] += hs * dist(rng);
      for (int i = 0; i < sz.m; ++i) b[i] += bs * dist(rng);
    }
    if (drift == Drift::kQHG) {
      for (int i = 0; i < sz.p; ++i) {
        for (int j = 0; j < sz.n; ++j) {
          G(i, j) += Gs * std::abs(traj.base.G(i, j)) * dist(rng);
        }
      }
      traj.G.push_back(G);
    }
    traj.q.push_back(q);
    traj.h.push_back(h);
    traj.b.push_back(b);
  }
  return traj;
}

// Smoothed-configuration flips between consecutive relaxed points: sign
// changes of v = z - s on both blocks (which side of the hyperbola the
// pair sits on).
int CountFlips(const elastiqp::Solution& a, const elastiqp::Solution& b) {
  int flips = 0;
  for (Eigen::Index i = 0; i < a.z_t.size(); ++i) {
    if ((a.z_t[i] - a.s_t[i] > 0) != (b.z_t[i] - b.s_t[i] > 0)) flips++;
    if ((a.z_ineq[i] - a.s_ineq[i] > 0) != (b.z_ineq[i] - b.s_ineq[i] > 0)) {
      flips++;
    }
  }
  return flips;
}

struct CellResult {
  double fwd_us = 0;
  double cold_us = 0, cold_it = 0;
  double warm_us = 0, warm_it = 0;
  double flips = 0;  // mean per tick
  double dx = 0;     // worst warm-vs-cold |x| diff (relative)
  int fwd_fails = 0, cold_fails = 0, warm_fails = 0;
  int fallbacks = 0;  // ticks where the warm attempt gave up and the
                      // retraction fallback produced the result
};

CellResult RunCell(const Trajectory& traj, double kappa, bool ruiz) {
  elastiqp::Solver solver;
  solver.settings.eps_abs = kFwdEps;
  solver.settings.eps_rel = 0;
  solver.settings.warm_start = true;
  solver.settings.ruiz = ruiz;
  solver.setup(traj.base.Q, traj.q[0], traj.base.A, traj.b[0],
               traj.G.empty() ? traj.base.G : traj.G[0], traj.h[0],
               traj.penalty);

  CellResult r;
  elastiqp::Solution prev;
  bool have_prev = false;
  for (int k = 0; k < kTicks; ++k) {
    solver.set_q(traj.q[k]);
    solver.set_h(traj.h[k]);
    if (traj.base.b.size() > 0) solver.set_b(traj.b[k]);
    if (!traj.G.empty()) solver.set_G(traj.G[k]);

    auto t0 = steady_clock::now();
    const int fwd_ok = solver.solve().converged;
    r.fwd_us += duration<double, std::micro>(steady_clock::now() - t0).count();
    if (fwd_ok != 1) r.fwd_fails++;

    // Warm chain first (see the header comment on ordering).
    t0 = steady_clock::now();
    const elastiqp::Solution w =
        solver.relax(kappa, kRelaxTol, kRelaxMaxIter);
    r.warm_us +=
        duration<double, std::micro>(steady_clock::now() - t0).count();
    r.warm_it += w.iters;
    if (w.converged != 1) r.warm_fails++;

    t0 = steady_clock::now();
    const elastiqp::Solution& c =
        solver.relax(kappa, kRelaxTol, kRelaxMaxIter, /*warm=*/false);
    r.cold_us +=
        duration<double, std::micro>(steady_clock::now() - t0).count();
    r.cold_it += c.iters;
    if (c.converged != 1) r.cold_fails++;

    const double dxk = (w.x - c.x).lpNorm<Eigen::Infinity>();
    r.dx = std::max(r.dx, dxk / (1.0 + c.x.lpNorm<Eigen::Infinity>()));
    // A fallback re-runs the retraction start -- the exact arithmetic the
    // cold probe performs -- so its result is bitwise identical to the
    // cold one while its iters also count the failed warm attempt.
    if (dxk == 0.0 && w.iters > c.iters) r.fallbacks++;
    if (have_prev) r.flips += CountFlips(prev, w);
    prev = w;
    have_prev = true;
  }
  const double nt = kTicks;
  r.fwd_us /= nt;
  r.cold_us /= nt;
  r.cold_it /= nt;
  r.warm_us /= nt;
  r.warm_it /= nt;
  r.flips /= (nt - 1);
  return r;
}

std::FILE* g_csv = nullptr;

void Report(const char* group, Size sz, Structure st, double penalty_w,
            Drift drift, double sigma, double kappa, const CellResult& r) {
  const bool fails = r.fwd_fails + r.cold_fails + r.warm_fails > 0;
  std::printf(
      "  %-6s %3d %3d %4d %5s %6.0e %6.0e | %7.1f | %8.1f (%4.1f) | %8.1f "
      "(%4.1f) | %5.2fx | %6.2f | %3d | %7.1e | %d/%d/%d%s\n",
      Name(st), sz.n, sz.m, sz.p, Name(drift), sigma, kappa, r.fwd_us,
      r.cold_us, r.cold_it, r.warm_us, r.warm_it, r.cold_us / r.warm_us,
      r.flips, r.fallbacks, r.dx, r.fwd_fails, r.cold_fails, r.warm_fails,
      fails ? " (FAILS)" : "");
  if (g_csv) {
    std::fprintf(
        g_csv, "%s,%s,%d,%d,%d,%g,%s,%g,%g,%.3f,%.3f,%.2f,%.3f,%.2f,%.3f,"
        "%d,%.3e,%d,%d,%d\n",
        group, Name(st), sz.n, sz.m, sz.p, penalty_w, Name(drift), sigma,
        kappa, r.fwd_us, r.cold_us, r.cold_it, r.warm_us, r.warm_it, r.flips,
        r.fallbacks, r.dx, r.fwd_fails, r.cold_fails, r.warm_fails);
  }
}

void Header() {
  std::printf(
      "  %-6s %3s %3s %4s %5s %6s %6s | %7s | %8s (%4s) | %8s (%4s) | %6s "
      "| %6s | %3s | %7s | %s\n",
      "struct", "n", "m", "p", "drift", "sigma", "kappa", "fwd", "rx cold",
      "it", "rx warm", "it", "spd", "flips", "fb", "|dx|", "fails f/c/w");
}

}  // namespace

int main(int argc, char** argv) {
  std::string csv_dir;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--csv") csv_dir = argv[i + 1];
  }
  if (!csv_dir.empty()) {
    const std::string path = csv_dir + "/relax_warm_cells.csv";
    g_csv = std::fopen(path.c_str(), "w");
    if (!g_csv) {
      std::fprintf(stderr, "cannot write %s\n", path.c_str());
      return 1;
    }
    std::fprintf(g_csv,
                 "group,structure,n,m,p,penalty,drift,sigma,kappa,"
                 "fwd_us,cold_us,cold_it,warm_us,warm_it,flips,fallbacks,"
                 "dx,fails_fwd,fails_cold,fails_warm\n");
  }

  const Size kSizes[] = {{14, 0, 100}, {30, 8, 200}, {58, 15, 400}};
  const double kSigmas[] = {1e-4, 1e-3, 1e-2, 1e-1};

  std::printf(
      "bench_relax_warm: relax() warm chain vs cold retraction start on\n"
      "drifting random QPs. %d ticks/cell, kappa=%g, fwd eps=%g, relax\n"
      "tol=%g. flips = smoothed-configuration changes per tick; spd =\n"
      "cold/warm wall time.\n\n",
      kTicks, kKappa, kFwdEps, kRelaxTol);

  for (const double penalty_w : {10.0, 1e4}) {
    const bool ruiz = penalty_w >= 1e4;
    std::printf("=== penalty %g (ruiz %s), drift on q+h(+b) ===\n",
                penalty_w, ruiz ? "on" : "off");
    Header();
    for (const Structure st :
         {Structure::kFeas, Structure::kInfeas, Structure::kDegen}) {
      for (const Size sz : kSizes) {
        for (const double sigma : kSigmas) {
          const unsigned seed =
              91u * static_cast<unsigned>(sz.n) +
              static_cast<unsigned>(sz.p) + 7u * static_cast<unsigned>(st);
          const Trajectory traj = MakeTrajectory(sz, st, penalty_w, sigma,
                                                 Drift::kQH, seed);
          Report("main", sz, st, penalty_w, Drift::kQH, sigma, kKappa,
                 RunCell(traj, kKappa, ruiz));
        }
      }
      std::printf("\n");
    }
  }

  std::printf(
      "=== drift composition (n=30, m=8, p=200, infeas, penalty 1e4, ruiz) "
      "===\n"
      "q: cost only (stable geometry) | qh: +bounds | qhG: +matrix\n");
  Header();
  for (const Drift drift : {Drift::kQ, Drift::kQH, Drift::kQHG}) {
    for (const double sigma : kSigmas) {
      const Trajectory traj = MakeTrajectory({30, 8, 200}, Structure::kInfeas,
                                             1e4, sigma, drift, 4242);
      Report("drift", {30, 8, 200}, Structure::kInfeas, 1e4, drift, sigma,
             kKappa, RunCell(traj, kKappa, true));
    }
    std::printf("\n");
  }

  std::printf(
      "=== ruiz on/off at penalty 1e4 (n=30, m=8, p=200, infeas, qh) ===\n"
      "Does equilibration change the warm chain's flip cost? Measured: no\n"
      "-- the scaled penalty is O(1) but the scaled complementarity\n"
      "target shrinks to c_s*kappa, so the flip traversal length (and its\n"
      "z/s amplification) is preserved.\n");
  Header();
  for (const bool ruiz : {false, true}) {
    for (const double sigma : kSigmas) {
      const Trajectory traj = MakeTrajectory({30, 8, 200}, Structure::kInfeas,
                                             1e4, sigma, Drift::kQH, 4242);
      Report(ruiz ? "ruiz_on" : "ruiz_off", {30, 8, 200}, Structure::kInfeas,
             1e4, Drift::kQH, sigma, kKappa, RunCell(traj, kKappa, ruiz));
    }
    std::printf("\n");
  }

  std::printf(
      "=== kappa sweep (n=30, m=8, p=200, sigma=1e-3, qh drift) ===\n"
      "Smaller kappa moves BOTH costs against the chain: the cold start\n"
      "gets cheaper (its residual is O(kappa) against a fixed absolute\n"
      "tol) while each flip's traversal gets longer (~log2(scale^2/\n"
      "kappa)).\n");
  Header();
  for (const double penalty_w : {10.0, 1e4}) {
    for (const double kappa : {1e-2, 1e-3, 1e-4, 1e-6}) {
      const Trajectory traj = MakeTrajectory(
          {30, 8, 200}, Structure::kInfeas, penalty_w, 1e-3, Drift::kQH,
          4242);
      Report("kappa", {30, 8, 200}, Structure::kInfeas, penalty_w,
             Drift::kQH, 1e-3, kappa, RunCell(traj, kappa, penalty_w >= 1e4));
    }
    std::printf("\n");
  }

  if (g_csv) std::fclose(g_csv);
  return 0;
}
