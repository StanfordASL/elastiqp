// Robotics correctness tests for ElastiQP: differential IK, torque-level
// OSC, and humanoid whole-body control built with Pinocchio (see
// robot_control.hpp).
//
// Checks, per problem family:
//   * every solve in a closed-loop trajectory converges, with small elastic
//     KKT residuals
//   * hard equality constraints (base dynamics, contacts) hold to ~1e-9 even
//     when the elastic inequalities conflict
//   * in feasible scenarios the elastic slacks are ~0 and the joint,
//     velocity, and torque limits are respected (torques reconstructed from
//     the eliminated dynamics)
//   * in deliberately conflicting scenarios the solver still converges and
//     degrades gracefully (slacks > 0) instead of failing
//   * (with the vendored piqp/) solutions match vanilla PIQP on the expanded
//     (n+p)-variable formulation
//
// The closed-loop tests integrate the QP solution (velocity or acceleration)
// and verify tracking, which exercises the warm-started Solver exactly as a
// control loop would.

#include <cstdio>

#include "elastiqp/elastiqp.hpp"
#include "problem_gen.hpp"
#include "robot_control.hpp"

#ifdef ELASTIQP_HAVE_PIQP
#include "piqp/piqp.hpp"
#endif

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using robot_control::Manipulator;
using robot_control::RobotQP;

namespace {

bool g_all_ok = true;

void Check(const char* name, bool ok, double val, const char* what) {
  std::printf("  %-44s %s=%9.2e %s\n", name, what, val, ok ? "OK" : "FAIL");
  g_all_ok &= ok;
}

double ElasticKKT(const RobotQP& qp, const elastiqp::Solution& sol) {
  return problem_gen::ElasticKKTResidual(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h,
                                         qp.penalty, sol.x, sol.t, sol.y,
                                         sol.z_t, sol.z_ineq);
}

double EqualityResidual(const RobotQP& qp, const VectorXd& x) {
  if (qp.b.size() == 0) return 0.0;
  return (qp.A * x - qp.b).lpNorm<Eigen::Infinity>();
}

// Absolute-only termination, pinned tight: the 1e-6 KKT / 1e-5 PIQP
// agreement thresholds below need more accuracy than the control-sized
// library default (eps_abs = 1e-5), and the conflict scenarios have
// 1e5-scale penalty data where relative criteria would stop at absolute
// KKT residuals around 1e-4 -- accurate, but the tests want a scale-free
// bound.
elastiqp::Settings TestSettings() {
  elastiqp::Settings s;
  s.eps_abs = 1e-8;
  s.eps_duality_gap_abs = 1e-8;
  s.eps_rel = 0;
  s.eps_duality_gap_rel = 0;
  return s;
}

elastiqp::Solver MakeSolver(const RobotQP& qp) {
  elastiqp::Solver solver;
  solver.settings = TestSettings();
  solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, qp.penalty);
  return solver;
}

void Update(elastiqp::Solver& solver, const RobotQP& qp) {
  solver.set_Q(qp.Q);
  solver.set_q(qp.q);
  if (qp.b.size() > 0) {
    solver.set_A(qp.A);
    solver.set_b(qp.b);
  }
  solver.set_G(qp.G);
  solver.set_h(qp.h);
}

// Circular EE target around a center, in the y-z plane.
robot_control::TaskTarget CircleTarget(const Vector3d& center,
                                       const Eigen::Matrix3d& rot, double t,
                                       double radius, double period) {
  const double w = 2 * M_PI / period;
  robot_control::TaskTarget target;
  target.rot = rot;
  target.pos = center + radius * Vector3d(0.0, std::cos(w * t) - 1.0,
                                          std::sin(w * t));
  target.vel = radius * w * Vector3d(0.0, -std::sin(w * t), std::cos(w * t));
  return target;
}

#ifdef ELASTIQP_HAVE_PIQP
// Cross-validate one instance against vanilla PIQP on the expanded
// (n+p)-variable formulation.
void CrossValidate(const char* name, const RobotQP& qp) {
  const auto esol = elastiqp::Solve(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h,
                                    qp.penalty, TestSettings());

  const problem_gen::ExpandedElastic e = problem_gen::MakeExpanded(
      qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, qp.penalty);
  piqp::DenseSolver<double> ref;
  ref.settings().eps_abs = 1e-10;
  ref.settings().eps_rel = 0;
  ref.settings().verbose = false;
  if (qp.b.size() > 0) {
    ref.setup(e.P, e.c, e.A, e.b, e.Gt, piqp::nullopt, e.h, e.lb,
              piqp::nullopt);
  } else {
    ref.setup(e.P, e.c, piqp::nullopt, piqp::nullopt, e.Gt, piqp::nullopt, e.h,
              e.lb, piqp::nullopt);
  }
  const piqp::Status st = ref.solve();
  const VectorXd x_ref = ref.result().x.head(qp.q.size());
  const double dx = (esol.x - x_ref).lpNorm<Eigen::Infinity>();
  Check(name, esol.converged == 1 && st == piqp::PIQP_SOLVED && dx < 1e-5, dx,
        "|dx|");
}
#endif

// ---------------------------------------------------------------------------
// Differential IK
// ---------------------------------------------------------------------------

void TestDiffIKTracking() {
  std::printf("Differential IK: closed-loop tracking\n");
  Manipulator arm;
  robot_control::ArmParams prm;

  const double dt = 0.01;
  const int ticks = 400;
  auto q = Manipulator::Home();
  arm.Compute(q, Manipulator::Vector6d::Zero());
  const Vector3d center = arm.ee_pos();
  const Eigen::Matrix3d rot0 = arm.ee_rot();

  RobotQP qp = robot_control::BuildDiffIK(arm, prm, q, {});
  elastiqp::Solver solver = MakeSolver(qp);

  int fails = 0;
  double worst_kkt = 0, worst_slack = 0, worst_track = 0;
  bool limits_ok = true;
  for (int k = 0; k < ticks; ++k) {
    const auto target = CircleTarget(center, rot0, k * dt, 0.10, 2.0);
    arm.Compute(q, Manipulator::Vector6d::Zero());
    qp = robot_control::BuildDiffIK(arm, prm, q, target);
    Update(solver, qp);
    const auto& sol = solver.solve();
    if (sol.converged != 1) fails++;
    worst_kkt = std::max(worst_kkt, ElasticKKT(qp, sol));
    worst_slack = std::max(worst_slack, sol.t.maxCoeff());
    const auto qd = sol.x;
    q += dt * qd;
    for (int i = 0; i < Manipulator::kNv; ++i) {
      limits_ok &= q[i] >= arm.q_min()[i] - 1e-6 &&
                   q[i] <= arm.q_max()[i] + 1e-6 &&
                   std::abs(qd[i]) <= arm.qd_max()[i] + 1e-6;
    }
    if (k > 50) {  // after the initial transient
      arm.Compute(q, Manipulator::Vector6d::Zero());
      worst_track =
          std::max(worst_track, (arm.ee_pos() - target.pos).norm());
    }
  }
  Check("all ticks converge", fails == 0, fails, "fails");
  Check("worst elastic KKT residual", worst_kkt < 1e-6, worst_kkt, "kkt");
  Check("feasible => slacks ~ 0", worst_slack < 1e-6, worst_slack, "max t");
  Check("joint/velocity limits respected", limits_ok, limits_ok ? 0.0 : 1.0,
        "viol");
  Check("EE tracking error < 1 cm", worst_track < 0.01, worst_track, "m");
}

void TestDiffIKConflict() {
  std::printf("Differential IK: conflicting constraints\n");
  Manipulator arm;
  robot_control::ArmParams prm;
  // Tighten the velocity box to 1 rad/s and start with a joint pushed well
  // past its position limit: the damper row then demands a retreat velocity
  // of damper_gain * 0.3 = 3 rad/s, which the box cannot deliver -- the QP
  // has no feasible point and the solver must return a least-violating
  // command (retreat at the box limit).
  prm.qd_cap = 1.0;

  auto q = Manipulator::Home();
  q[1] = arm.q_max()[1] + 0.3;
  arm.Compute(q, Manipulator::Vector6d::Zero());

  robot_control::TaskTarget target;
  target.pos = arm.ee_pos();
  target.rot = arm.ee_rot();

  const double dt = 0.01;
  int fails = 0;
  double worst_kkt = 0, max_slack = 0;
  bool finite = true;
  for (int k = 0; k < 100; ++k) {
    arm.Compute(q, Manipulator::Vector6d::Zero());
    const RobotQP qp = robot_control::BuildDiffIK(arm, prm, q, target);
    const auto sol =
        elastiqp::Solve(qp.Q, qp.q, qp.G, qp.h, qp.penalty, TestSettings());
    if (sol.converged != 1) fails++;
    worst_kkt = std::max(worst_kkt, ElasticKKT(qp, sol));
    max_slack = std::max(max_slack, sol.t.maxCoeff());
    finite &= sol.x.allFinite();
    q += dt * sol.x;
  }
  Check("all ticks converge", fails == 0 && finite, fails, "fails");
  Check("worst elastic KKT residual", worst_kkt < 1e-6, worst_kkt, "kkt");
  Check("conflict => some slack active", max_slack > 1e-1, max_slack, "max t");
  // The damper drives the joint back exponentially, so it settles AT the
  // limit (from above), not strictly inside it.
  Check("joint recovered to its limit", q[1] <= arm.q_max()[1] + 1e-2,
        q[1] - arm.q_max()[1], "q1 - q_max");
}

// ---------------------------------------------------------------------------
// Bimanual differential IK (rigid grasp as hard equality rows)
// ---------------------------------------------------------------------------

void TestBimanualDiffIK(bool conflict) {
  std::printf("Bimanual diff IK (x = [qd1; qd2], rigid grasp): %s\n",
              conflict ? "conflicting constraints" : "closed-loop tracking");
  Manipulator arm1, arm2;
  robot_control::ArmParams prm;
  const Vector3d offset(0.0, 0.6, 0.0);

  const double dt = 0.01;
  const int ticks = 400;
  auto q1 = Manipulator::Home();
  auto q2 = Manipulator::Home();
  const auto qd0 = Manipulator::Vector6d::Zero();
  if (conflict) {
    // Tightened velocity box + a joint pushed past its limit, as in the
    // single-arm conflict test: the damper demands a retreat the box cannot
    // deliver. The grasp equalities must keep holding regardless.
    prm.qd_cap = 1.0;
    q1[1] = arm1.q_max()[1] + 0.3;
  }
  arm1.Compute(q1, qd0);
  arm2.Compute(q2, qd0);
  const robot_control::BimanualGrasp grasp =
      robot_control::MakeGrasp(arm1, arm2, offset);
  const Vector3d center = arm1.ee_pos();
  const Eigen::Matrix3d rot0 = arm1.ee_rot();

  RobotQP qp = robot_control::BuildBimanualDiffIK(arm1, arm2, offset, grasp,
                                                  prm, q1, q2, {}, {});
  elastiqp::Solver solver = MakeSolver(qp);

  int fails = 0;
  double worst_kkt = 0, worst_eq = 0, max_slack = 0, worst_track = 0;
  double worst_grasp = 0;
  bool limits_ok = true;
  for (int k = 0; k < ticks; ++k) {
    robot_control::TaskTarget t1 = CircleTarget(center, rot0, k * dt,
                                                conflict ? 0.0 : 0.10, 2.0);
    if (!conflict) {
      const double wy = 2 * M_PI / 3.0;
      const double yaw = 0.15 * std::sin(wy * k * dt);
      t1.rot = Eigen::AngleAxisd(yaw, Vector3d::UnitZ()) * rot0;
      t1.omega = Vector3d(0.0, 0.0, 0.15 * wy * std::cos(wy * k * dt));
    }
    const robot_control::TaskTarget t2 =
        robot_control::GraspConsistentTarget(grasp, t1);
    arm1.Compute(q1, qd0);
    arm2.Compute(q2, qd0);
    qp = robot_control::BuildBimanualDiffIK(arm1, arm2, offset, grasp, prm, q1,
                                            q2, t1, t2);
    Update(solver, qp);
    const auto& sol = solver.solve();
    if (sol.converged != 1) fails++;
    worst_kkt = std::max(worst_kkt, ElasticKKT(qp, sol));
    worst_eq = std::max(worst_eq, EqualityResidual(qp, sol.x));
    max_slack = std::max(max_slack, sol.t.maxCoeff());
    const VectorXd qd = sol.x;
    q1 += dt * qd.head<6>();
    q2 += dt * qd.tail<6>();
    for (int i = 0; i < Manipulator::kNv; ++i) {
      limits_ok &= std::abs(qd[i]) <= arm1.qd_max()[i] + 1e-6 &&
                   std::abs(qd[6 + i]) <= arm2.qd_max()[i] + 1e-6;
    }
    arm1.Compute(q1, qd0);
    arm2.Compute(q2, qd0);
    // Relative-pose (grasp) error: the servo in b must hold it near zero.
    const Vector3d e_pos = (robot_control::Arm2WorldPos(arm2, offset) -
                            arm1.ee_pos()) -
                           arm1.ee_rot() * grasp.r12;
    const Vector3d e_rot = robot_control::OrientationError(
        arm2.ee_rot(), arm1.ee_rot() * grasp.R12);
    worst_grasp = std::max(worst_grasp,
                           std::max(e_pos.norm(), e_rot.norm()));
    if (!conflict && k > 50) {
      worst_track = std::max(worst_track, (arm1.ee_pos() - t1.pos).norm());
    }
  }
  Check("all ticks converge", fails == 0, fails, "fails");
  Check("worst elastic KKT residual", worst_kkt < 1e-6, worst_kkt, "kkt");
  Check("grasp equality residual ~ 0", worst_eq < 1e-8, worst_eq, "|Ax-b|");
  Check("grasp pose error stays small", worst_grasp < 1e-3, worst_grasp,
        "|e|");
  if (conflict) {
    Check("conflict => some slack active", max_slack > 1e-1, max_slack,
          "max t");
    Check("joint recovered to its limit", q1[1] <= arm1.q_max()[1] + 1e-2,
          q1[1] - arm1.q_max()[1], "q1 - q_max");
  } else {
    Check("feasible => slacks ~ 0", max_slack < 1e-6, max_slack, "max t");
    Check("velocity limits respected", limits_ok, limits_ok ? 0.0 : 1.0,
          "viol");
    Check("EE tracking error < 1 cm", worst_track < 0.01, worst_track, "m");
  }
}

// ---------------------------------------------------------------------------
// Torque-level operational-space control (accelerations eliminated, x = tau)
// ---------------------------------------------------------------------------

void TestArmOSC(bool conflict) {
  std::printf("Arm OSC (x = tau, accelerations eliminated): %s\n",
              conflict ? "conflicting torque/velocity limits"
                       : "closed-loop tracking");
  Manipulator arm;
  robot_control::ArmParams prm;
  if (conflict) {
    // A 1 Nm torque cap cannot hold the arm against gravity (worst gravity
    // torque ~5.6 Nm), so it accelerates; once a joint nears the tightened
    // 1 rad/s velocity cap, the velocity-limit rows demand a deceleration
    // the capped torque rows cannot produce, and slacks must activate.
    prm.tau_cap = 1.0;
    prm.qd_cap = 1.0;
  }

  const double dt = 1e-3;
  const int ticks = 500;
  auto q = Manipulator::Home();
  Manipulator::Vector6d qd = Manipulator::Vector6d::Zero();
  arm.Compute(q, qd);
  const Vector3d center = arm.ee_pos();
  const Eigen::Matrix3d rot0 = arm.ee_rot();

  RobotQP qp = robot_control::BuildArmOSC(
      arm, prm, q, qd, robot_control::TaskAcceleration(arm, qd, {}));
  elastiqp::Solver solver = MakeSolver(qp);
  int fails = 0;
  double worst_kkt = 0, max_slack = 0, worst_track = 0, worst_tau = 0;
  bool finite = true;
  for (int k = 0; k < ticks; ++k) {
    const auto target = CircleTarget(center, rot0, k * dt, 0.05, 2.0);
    arm.Compute(q, qd);
    qp = robot_control::BuildArmOSC(
        arm, prm, q, qd, robot_control::TaskAcceleration(arm, qd, target));
    Update(solver, qp);
    const auto& sol = solver.solve();
    if (sol.converged != 1) fails++;
    worst_kkt = std::max(worst_kkt, ElasticKKT(qp, sol));
    max_slack = std::max(max_slack, sol.t.maxCoeff());
    finite &= sol.x.allFinite();
    const Manipulator::Vector6d tau = sol.x;
    worst_tau = std::max(worst_tau, tau.lpNorm<Eigen::Infinity>());
    const Manipulator::Vector6d qdd = arm.Accelerations(tau);
    qd += dt * qdd;
    q += dt * qd;
    if (!conflict && k > 200) {
      worst_track = std::max(worst_track, (arm.ee_pos() - target.pos).norm());
    }
  }
  Check("all ticks converge", fails == 0 && finite, fails, "fails");
  Check("worst elastic KKT residual", worst_kkt < 1e-6, worst_kkt, "kkt");
  if (conflict) {
    Check("conflict => slack active", max_slack > 1e-1, max_slack, "max t");
  } else {
    Check("feasible => slacks ~ 0", max_slack < 1e-6, max_slack, "max t");
    Check("commanded torques within limits",
          worst_tau <= arm.tau_max().minCoeff() + 1e-6, worst_tau, "|tau|");
    Check("EE tracking error < 2 cm", worst_track < 0.02, worst_track, "m");
  }
}

// ---------------------------------------------------------------------------
// Humanoid whole-body control (torques eliminated, x = [qdd; f])
// ---------------------------------------------------------------------------

void TestHumanoidWBC(bool conflict) {
  std::printf("Humanoid WBC (x = [qdd; f], torques eliminated): %s\n",
              conflict ? "conflicting limits" : "perturbed configurations");
  robot_control::Humanoid robot;
  robot_control::HumanoidParams prm;
  if (conflict) {
    // Torque and acceleration caps far below what the commanded CoM
    // acceleration and gravity require.
    prm.tau_max = 1.0;
    prm.qdd_max = 0.5;
  }

  const int nv = robot.nv();
  int fails = 0;
  double worst_kkt = 0, worst_eq = 0, max_slack = 0, min_fz = 1e30;
  double worst_tau = 0;
  elastiqp::Solver solver;
  bool first = true;
  for (int k = 0; k < 20; ++k) {
    // Smoothly varying configuration/velocity around neutral.
    VectorXd dq = VectorXd::Zero(nv), qd = VectorXd::Zero(nv);
    for (int i = 6; i < nv; ++i) {
      dq[i] = 0.3 * std::sin(0.1 * k + 0.4 * i);
      qd[i] = 0.2 * std::cos(0.1 * k + 0.7 * i);
    }
    const VectorXd q = robot.Configuration(dq);
    robot.Compute(q, qd);
    const Vector3d acom_des =
        conflict ? Vector3d(50.0, 50.0, 100.0)
                 : Vector3d(0.5 * std::sin(0.1 * k), 0.0, -0.5) -
                       2.0 * robot.vcom();
    const RobotQP qp = robot_control::BuildHumanoidWBC(robot, prm, qd,
                                                       acom_des);
    if (first) {
      solver.settings = TestSettings();
      solver.setup(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h, qp.penalty);
      first = false;
    } else {
      Update(solver, qp);
    }
    const auto& sol = solver.solve();
    if (sol.converged != 1) fails++;
    worst_kkt = std::max(worst_kkt, ElasticKKT(qp, sol));
    worst_eq = std::max(worst_eq, EqualityResidual(qp, sol.x));
    max_slack = std::max(max_slack, sol.t.maxCoeff());
    const VectorXd f = sol.x.tail(12);
    worst_tau = std::max(
        worst_tau,
        robot.Torques(sol.x.head(nv), f).lpNorm<Eigen::Infinity>());
    for (int c = 0; c < 2; ++c) {
      min_fz = std::min(min_fz, f[6 * c + 2]);
    }
  }
  Check("all solves converge", fails == 0, fails, "fails");
  Check("worst elastic KKT residual", worst_kkt < 1e-6, worst_kkt, "kkt");
  Check("base dynamics + contact residual ~ 0", worst_eq < 1e-8, worst_eq,
        "|Ax-b|");
  if (conflict) {
    Check("conflict => slacks active", max_slack > 1e-2, max_slack, "max t");
  } else {
    Check("feasible => slacks ~ 0", max_slack < 1e-6, max_slack, "max t");
    Check("reconstructed torques within limits", worst_tau <= prm.tau_max,
          worst_tau, "|tau|");
    Check("normal forces support the robot", min_fz > 0, min_fz, "min fz");
  }
}

}  // namespace

int main() {
  TestDiffIKTracking();
  TestDiffIKConflict();
  TestBimanualDiffIK(/*conflict=*/false);
  TestBimanualDiffIK(/*conflict=*/true);
  TestArmOSC(/*conflict=*/false);
  TestArmOSC(/*conflict=*/true);
  TestHumanoidWBC(/*conflict=*/false);
  TestHumanoidWBC(/*conflict=*/true);

#ifdef ELASTIQP_HAVE_PIQP
  std::printf("Cross-validation vs vanilla PIQP (expanded formulation)\n");
  {
    Manipulator arm;
    robot_control::ArmParams prm;
    auto q = Manipulator::Home();
    auto qd = Manipulator::Vector6d::Constant(0.3);
    arm.Compute(q, qd);
    robot_control::TaskTarget target;
    target.pos = arm.ee_pos() + Vector3d(0.1, -0.1, 0.1);
    target.rot = arm.ee_rot();
    CrossValidate("diff IK", robot_control::BuildDiffIK(arm, prm, q, target));
    {
      Manipulator b1, b2;
      const Vector3d offset(0.0, 0.6, 0.0);
      auto qb1 = Manipulator::Home();
      auto qb2 = Manipulator::Home();
      qb2[2] += 0.2;  // break the symmetry between the arms
      b1.Compute(qb1, Manipulator::Vector6d::Zero());
      b2.Compute(qb2, Manipulator::Vector6d::Zero());
      const auto grasp = robot_control::MakeGrasp(b1, b2, offset);
      robot_control::TaskTarget t1;
      t1.pos = b1.ee_pos() + Vector3d(0.05, -0.05, 0.05);
      t1.rot = b1.ee_rot();
      CrossValidate("bimanual diff IK",
                    robot_control::BuildBimanualDiffIK(
                        b1, b2, offset, grasp, prm, qb1, qb2, t1,
                        robot_control::GraspConsistentTarget(grasp, t1)));
    }
    CrossValidate("arm OSC",
                  robot_control::BuildArmOSC(
                      arm, prm, q, qd,
                      robot_control::TaskAcceleration(arm, qd, target)));
    robot_control::ArmParams conflict_prm = prm;
    conflict_prm.tau_cap = 1.0;
    conflict_prm.qd_cap = 1.0;
    CrossValidate("arm OSC (conflict)",
                  robot_control::BuildArmOSC(
                      arm, conflict_prm, q, qd,
                      robot_control::TaskAcceleration(arm, qd, target)));

    robot_control::Humanoid robot;
    robot_control::HumanoidParams hprm;
    // The default 1e-6 force regularization leaves the force split nearly
    // non-unique (two solvers can return equally optimal but different f),
    // so cross-validate a well-conditioned instance.
    hprm.w_force = 1e-3;
    VectorXd dq = VectorXd::Zero(robot.nv()), hqd = VectorXd::Zero(robot.nv());
    for (int i = 6; i < robot.nv(); ++i) dq[i] = 0.3 * std::sin(0.4 * i);
    robot.Compute(robot.Configuration(dq), hqd);
    CrossValidate("humanoid WBC",
                  robot_control::BuildHumanoidWBC(robot, hprm, hqd,
                                                  Vector3d(0.2, 0.0, -0.5)));
  }
#endif

  std::printf(g_all_ok ? "ALL OK\n" : "FAILURES\n");
  return g_all_ok ? 0 : 1;
}
