// Robotics problem generators for the ElastiQP tests and benchmarks:
// realistic differential-IK and whole-body-control QPs built with Pinocchio.
// Task-space objectives, joint/velocity/torque limits as elastic
// inequalities, and (for the floating-base humanoid) the unactuated dynamics
// and contact constraints as hard equalities.
//
// Four problem families, all with the elastic-QP shape
//   minimize    0.5 x^T Q x + q^T x + penalty^T t
//   subject to  A x == b (hard),  G x - t <= h,  t >= 0
//
//  1. DiffIK  — velocity-control differential IK on Pinocchio's sample
//               6-DOF manipulator (n = 6, no equalities, p = 24 elastic
//               rows: joint-limit dampers, velocity box)
//  2. ArmOSC  — torque-level operational-space control on the same arm,
//               with the joint torques as the decision variables. The
//               accelerations are eliminated analytically (the arm is
//               fully actuated: qdd = M^-1 (tau - nle)), so x = tau
//               (n = 6, no equalities, p = 24: velocity-limit rows in
//               tau and the torque box)
//  3. BimanualDiffIK — two of the sample manipulators mounted side by side,
//               both holding one rigid object; x = [qd1; qd2] (n = 12) with
//               the rigid grasp as m = 6 hard equality rows coupling the two
//               EE twists and p = 48 elastic rows (per-arm dampers and
//               velocity boxes)
//  4. HumanoidWBC — floating-base whole-body control on Pinocchio's sample
//               humanoid, standing on two 6D "foot" contacts. The actuated
//               torques are eliminated through the actuated dynamics rows
//               (tau = [M qdd + nle - Jc^T f]_actuated), leaving
//               x = [qdd (34); f (12)] (n = 46) with the 6 unactuated
//               base dynamics rows and 12 contact rows hard (m = 18) and
//               p = 132 elastic rows: acceleration box, torque box (dense
//               rows in qdd and f), friction/CoP cones.
//
// Only Pinocchio + Eigen are required, and both robots are Pinocchio sample
// models (pinocchio::buildModels), so there are no URDFs, data files, or
// runtime paths.

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/sample-models.hpp>

#include "qp_io.hpp"  // RobotQP

namespace robot_control {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

// frax's orientation_error_3D: ~rotation vector from R_cur to R_des.
inline Vector3d OrientationError(const Matrix3d& R_cur, const Matrix3d& R_des) {
  return -0.5 * (R_cur.col(0).cross(R_des.col(0)) +
                 R_cur.col(1).cross(R_des.col(1)) +
                 R_cur.col(2).cross(R_des.col(2)));
}

// ---------------------------------------------------------------------------
// 6-DOF manipulator (differential IK and torque-level OSC)
// ---------------------------------------------------------------------------

// Pinocchio's deterministic sample manipulator (buildModels::manipulator):
// shoulder-elbow-wrist, two 1 m links, limits +-3.14 rad, 10 rad/s, 10 Nm.
class Manipulator {
 public:
  static constexpr int kNv = 6;
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  Manipulator() : data_(MakeModel()) {
    // MakeModel() stored the model in model_ before Data was constructed.
    ee_frame_ = model_.getFrameId("effector_body");
    q_min_ = model_.lowerPositionLimit;
    q_max_ = model_.upperPositionLimit;
    qd_max_ = model_.velocityLimit;
    tau_max_ = model_.effortLimit;
  }

  const pinocchio::Model& model() const { return model_; }
  const Vector6d& q_min() const { return q_min_; }
  const Vector6d& q_max() const { return q_max_; }
  const Vector6d& qd_max() const { return qd_max_; }
  const Vector6d& tau_max() const { return tau_max_; }

  // Non-singular, low-gravity-torque configuration (worst gravity torque
  // ~5.6 Nm against the 10 Nm effort limit).
  static Vector6d Home() {
    return (Vector6d() << 0.0, 0.4, 0.0, -0.8, 0.0, 0.5).finished();
  }

  // Recomputes every state-dependent quantity used by the problem builders.
  void Compute(const Vector6d& q, const Vector6d& qd) {
    pinocchio::crba(model_, data_, q);
    M_ = data_.M.template selfadjointView<Eigen::Upper>();
    Minv_ = M_.llt().solve(Eigen::Matrix<double, 6, 6>::Identity());
    pinocchio::nonLinearEffects(model_, data_, q, qd);
    nle_ = data_.nle;
    pinocchio::computeJointJacobians(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);
    // Classical accelerations with qdd = 0 give the Jdot*qd bias terms.
    pinocchio::forwardKinematics(model_, data_, q, qd, VectorXd::Zero(kNv));

    J_ee_.setZero(6, kNv);
    pinocchio::getFrameJacobian(model_, data_, ee_frame_,
                                pinocchio::LOCAL_WORLD_ALIGNED, J_ee_);
    ee_pos_ = data_.oMf[ee_frame_].translation();
    ee_rot_ = data_.oMf[ee_frame_].rotation();
    ee_bias_ = pinocchio::getFrameClassicalAcceleration(
                   model_, data_, ee_frame_, pinocchio::LOCAL_WORLD_ALIGNED)
                   .toVector();
  }

  // --- valid after Compute ---
  const Eigen::Matrix<double, 6, 6>& M() const { return M_; }
  const Eigen::Matrix<double, 6, 6>& Minv() const { return Minv_; }
  const Vector6d& nle() const { return nle_; }
  const MatrixXd& J_ee() const { return J_ee_; }
  const Vector3d& ee_pos() const { return ee_pos_; }
  const Matrix3d& ee_rot() const { return ee_rot_; }
  const Eigen::Matrix<double, 6, 1>& ee_bias() const { return ee_bias_; }

  // Forward/inverse dynamics at the Compute state.
  Vector6d Torques(const Vector6d& qdd) const { return M_ * qdd + nle_; }
  Vector6d Accelerations(const Vector6d& tau) const {
    return Minv_ * (tau - nle_);
  }

 private:
  pinocchio::Model& MakeModel() {
    pinocchio::buildModels::manipulator(model_);
    return model_;
  }

  pinocchio::Model model_;
  pinocchio::Data data_;
  pinocchio::FrameIndex ee_frame_;

  Vector6d q_min_, q_max_, qd_max_, tau_max_;
  Eigen::Matrix<double, 6, 6> M_, Minv_;
  Vector6d nle_;
  MatrixXd J_ee_;
  Vector3d ee_pos_;
  Matrix3d ee_rot_;
  Eigen::Matrix<double, 6, 1> ee_bias_;
};

struct ArmParams {
  double kp_pos = 5.0;       // task-space P gains (diff IK)
  double kp_rot = 2.0;
  double damper_gain = 10.0; // joint-limit damper gain [1/s]
  double vel_horizon = 0.1;  // accel-level velocity-limit horizon [s]
  double qd_cap = -1.0;      // velocity cap; < 0 uses the model limit
  double tau_cap = -1.0;     // torque cap; < 0 uses the model limit
  double soft_penalty = 1e3;   // dampers / velocity-limit rows
  double limit_penalty = 1e5;  // velocity box / torque box
  double grasp_gain = 10.0;    // bimanual rigid-grasp servo gain [1/s]
};

struct TaskTarget {
  Vector3d pos = Vector3d::Zero();
  Matrix3d rot = Matrix3d::Identity();
  Vector3d vel = Vector3d::Zero();
  Vector3d omega = Vector3d::Zero();
};

// --- Differential IK (velocity control), n = 6, m = 0, p = 24 -------------
//
// minimize 0.5 ||J qd - v_des||^2 + 0.5 w_n ||qd||^2 over qd, with elastic
// rows [joint-limit dampers (12); velocity box (12)]. Damper rows use
// soft_penalty, the velocity box limit_penalty.
inline RobotQP BuildDiffIK(const Manipulator& arm, const ArmParams& prm,
                           const Manipulator::Vector6d& q,
                           const TaskTarget& target) {
  const int n = Manipulator::kNv;
  const int p = 4 * n;

  RobotQP qp;
  qp.A.resize(0, n);
  qp.b.resize(0);
  qp.G = MatrixXd::Zero(p, n);
  qp.h.resize(p);
  qp.penalty.resize(p);

  // Objective: task velocity tracking + a small uniform damping term.
  Eigen::Matrix<double, 6, 1> v_des;
  v_des.head<3>() = target.vel + prm.kp_pos * (target.pos - arm.ee_pos());
  v_des.tail<3>() =
      target.omega - prm.kp_rot * OrientationError(arm.ee_rot(), target.rot);
  const double w_n = 1e-2;
  const MatrixXd& J = arm.J_ee();
  qp.Q = J.transpose() * J + w_n * MatrixXd::Identity(n, n);
  qp.q = -J.transpose() * v_des;

  int r = 0;
  // Joint-limit velocity dampers: qd_i <= gain (q_max - q), and lower.
  for (int i = 0; i < n; ++i, ++r) {
    qp.G(r, i) = 1.0;
    qp.h[r] = prm.damper_gain * (arm.q_max()[i] - q[i]);
  }
  for (int i = 0; i < n; ++i, ++r) {
    qp.G(r, i) = -1.0;
    qp.h[r] = prm.damper_gain * (q[i] - arm.q_min()[i]);
  }
  qp.penalty.head(r).setConstant(prm.soft_penalty);
  // Velocity box.
  const int box_start = r;
  for (int i = 0; i < n; ++i, ++r) {
    qp.G(r, i) = 1.0;
    qp.h[r] = prm.qd_cap > 0 ? prm.qd_cap : arm.qd_max()[i];
  }
  for (int i = 0; i < n; ++i, ++r) {
    qp.G(r, i) = -1.0;
    qp.h[r] = prm.qd_cap > 0 ? prm.qd_cap : arm.qd_max()[i];
  }
  qp.penalty.tail(p - box_start).setConstant(prm.limit_penalty);
  return qp;
}

// --- Torque-level OSC / inverse dynamics, x = tau -------------------------
//
// The joint torques are the decision variables; the accelerations are
// eliminated analytically (the arm is fully actuated, so the dynamics give
// qdd = M^-1 (tau - nle) uniquely) rather than carried alongside tau with
// the dynamics as hard equalities. n = 6, m = 0, elastic rows
// [velocity-limit rows, dense in tau (12); torque box, sparse (12)]
// => p = 24.
//
// a_des is the desired 6D task acceleration; the objective tracks it with
// J qdd(tau) + Jdot qd, plus a small posture term pulling qdd(tau) toward a
// PD acceleration on the home configuration.
inline RobotQP BuildArmOSC(const Manipulator& arm, const ArmParams& prm,
                           const Manipulator::Vector6d& q,
                           const Manipulator::Vector6d& qd,
                           const Eigen::Matrix<double, 6, 1>& a_des) {
  const int n = Manipulator::kNv;
  const int p = 4 * n;

  RobotQP qp;
  // Substituting qdd = Minv (tau - nle) into
  //   0.5 ||J qdd - (a_des - Jdot qd)||^2 + 0.5 w ||qdd - qdd_post||^2
  // gives a quadratic in tau with task map C = J Minv and posture map Minv.
  const Eigen::Matrix<double, 6, 6>& Minv = arm.Minv();
  const Manipulator::Vector6d qdd_bias = Minv * arm.nle();
  const MatrixXd C = arm.J_ee() * Minv;
  const double w_posture = 1e-2;
  const Manipulator::Vector6d qdd_post =
      -10.0 * (q - Manipulator::Home()) - 5.0 * qd;
  qp.Q = C.transpose() * C +
         w_posture * (Minv.transpose() * Minv);
  qp.q = -C.transpose() * (a_des - arm.ee_bias() + C * arm.nle()) -
         w_posture * (Minv.transpose() * (qdd_post + qdd_bias));

  qp.A.resize(0, n);
  qp.b.resize(0);
  qp.G = MatrixXd::Zero(p, n);
  qp.h.resize(p);
  qp.penalty.resize(p);

  int r = 0;
  // Velocity-limit rows: keep qd within the box over the next vel_horizon
  // seconds at constant qdd: qdd_i(tau) <= (qd_max - qd_i) / horizon, and
  // lower — dense rows Minv.row(i) in tau.
  for (int i = 0; i < n; ++i, ++r) {
    const double cap = prm.qd_cap > 0 ? prm.qd_cap : arm.qd_max()[i];
    qp.G.row(r) = Minv.row(i);
    qp.h[r] = (cap - qd[i]) / prm.vel_horizon + qdd_bias[i];
  }
  for (int i = 0; i < n; ++i, ++r) {
    const double cap = prm.qd_cap > 0 ? prm.qd_cap : arm.qd_max()[i];
    qp.G.row(r) = -Minv.row(i);
    qp.h[r] = (qd[i] + cap) / prm.vel_horizon - qdd_bias[i];
  }
  qp.penalty.head(r).setConstant(prm.soft_penalty);
  // Torque box: +-tau_i <= tau_cap (plain box on the variables).
  const int box_start = r;
  for (int i = 0; i < n; ++i, ++r) {
    qp.G(r, i) = 1.0;
    qp.h[r] = prm.tau_cap > 0 ? prm.tau_cap : arm.tau_max()[i];
  }
  for (int i = 0; i < n; ++i, ++r) {
    qp.G(r, i) = -1.0;
    qp.h[r] = prm.tau_cap > 0 ? prm.tau_cap : arm.tau_max()[i];
  }
  qp.penalty.tail(p - box_start).setConstant(prm.limit_penalty);
  return qp;
}

// Desired task acceleration for the OSC controller (PD on the EE pose).
inline Eigen::Matrix<double, 6, 1> TaskAcceleration(
    const Manipulator& arm, const Manipulator::Vector6d& qd,
    const TaskTarget& target, double kp_pos = 50.0, double kp_rot = 20.0,
    double kd_pos = 20.0, double kd_rot = 10.0) {
  const Eigen::Matrix<double, 6, 1> twist = arm.J_ee() * qd;
  Eigen::Matrix<double, 6, 1> a;
  a.head<3>() = kp_pos * (target.pos - arm.ee_pos()) +
                kd_pos * (target.vel - twist.head<3>());
  a.tail<3>() = -kp_rot * OrientationError(arm.ee_rot(), target.rot) +
                kd_rot * (target.omega - twist.tail<3>());
  return a;
}

// ---------------------------------------------------------------------------
// Bimanual differential IK (two manipulators, rigid grasp)
// ---------------------------------------------------------------------------
//
// Two sample manipulators mounted side by side (arm 2's base translated by
// base2_offset, no base rotation), both holding one rigid object. The grasp
// is the relative EE pose captured at grasp time and the coupling is enforced
// at the velocity level as 6 HARD equality rows on x = [qd1; qd2] (world-
// aligned twists, lever arm a = R1 * r12):
//   position:  v2 - v1 - w1 x a = -k e_pos      (rigid-body point velocity)
//   rotation:  w2 - w1          = -k e_rot
// with a proportional servo (grasp_gain) on the relative-pose error so the
// velocity-level constraint does not drift under integration. b is therefore
// state-dependent, like every other matrix in the problem.

// Relative EE pose at grasp time, fixed in arm 1's EE frame.
struct BimanualGrasp {
  Vector3d r12;  // p2 - p1, expressed in arm-1 EE frame
  Matrix3d R12;  // R1^T R2
};

// World position of arm 2's EE (its base is translated, not rotated).
inline Vector3d Arm2WorldPos(const Manipulator& arm2,
                             const Vector3d& base2_offset) {
  return base2_offset + arm2.ee_pos();
}

// Captures the grasp from the current (Compute()d) arm states.
inline BimanualGrasp MakeGrasp(const Manipulator& arm1,
                               const Manipulator& arm2,
                               const Vector3d& base2_offset) {
  BimanualGrasp g;
  g.r12 = arm1.ee_rot().transpose() *
          (Arm2WorldPos(arm2, base2_offset) - arm1.ee_pos());
  g.R12 = arm1.ee_rot().transpose() * arm2.ee_rot();
  return g;
}

// Arm-2 task target rigidly consistent with an arm-1 (object) target: the
// object moves with target1's twist, so the second grasp frame follows with
// the transported velocity.
inline TaskTarget GraspConsistentTarget(const BimanualGrasp& grasp,
                                        const TaskTarget& target1) {
  const Vector3d a = target1.rot * grasp.r12;  // desired world lever arm
  TaskTarget t2;
  t2.pos = target1.pos + a;
  t2.rot = target1.rot * grasp.R12;
  t2.vel = target1.vel + target1.omega.cross(a);
  t2.omega = target1.omega;
  return t2;
}

// --- Bimanual differential IK, n = 12, m = 6, p = 48 ----------------------
//
// minimize 0.5 ||J1 qd1 - v1_des||^2 + 0.5 ||J2 qd2 - v2_des||^2
//          + 0.5 w_n ||qd||^2   subject to the rigid-grasp rows above,
// with per-arm elastic rows [dampers arm1 (12); dampers arm2 (12);
// velocity box arm1 (12); velocity box arm2 (12)]. Dampers use soft_penalty,
// the boxes limit_penalty. Both arms must be Compute()d at (q1, q2).
inline RobotQP BuildBimanualDiffIK(const Manipulator& arm1,
                                   const Manipulator& arm2,
                                   const Vector3d& base2_offset,
                                   const BimanualGrasp& grasp,
                                   const ArmParams& prm,
                                   const Manipulator::Vector6d& q1,
                                   const Manipulator::Vector6d& q2,
                                   const TaskTarget& target1,
                                   const TaskTarget& target2) {
  const int na = Manipulator::kNv;
  const int n = 2 * na;
  const int p = 8 * na;

  const Vector3d p1 = arm1.ee_pos();
  const Vector3d p2 = Arm2WorldPos(arm2, base2_offset);

  RobotQP qp;
  // Objective: both EEs track their (rigid-consistent) task twists, plus a
  // small uniform damping term.
  auto task_twist = [&prm](const Manipulator& arm, const Vector3d& p_world,
                           const TaskTarget& tg) {
    Eigen::Matrix<double, 6, 1> v;
    v.head<3>() = tg.vel + prm.kp_pos * (tg.pos - p_world);
    v.tail<3>() = tg.omega - prm.kp_rot * OrientationError(arm.ee_rot(), tg.rot);
    return v;
  };
  const Eigen::Matrix<double, 6, 1> v1_des = task_twist(arm1, p1, target1);
  const Eigen::Matrix<double, 6, 1> v2_des = task_twist(arm2, p2, target2);
  const double w_n = 1e-2;
  const MatrixXd& J1 = arm1.J_ee();
  const MatrixXd& J2 = arm2.J_ee();
  qp.Q = w_n * MatrixXd::Identity(n, n);
  qp.Q.topLeftCorner(na, na) += J1.transpose() * J1;
  qp.Q.bottomRightCorner(na, na) += J2.transpose() * J2;
  qp.q.resize(n);
  qp.q.head(na) = -J1.transpose() * v1_des;
  qp.q.tail(na) = -J2.transpose() * v2_des;

  // Hard rigid-grasp rows. On arm 1's twist the position rows carry the
  // lever-arm term (-w1 x a = skew(a) w1); arm 2's twist enters plainly.
  const Vector3d a = arm1.ee_rot() * grasp.r12;
  const Vector3d e_pos = (p2 - p1) - a;
  const Vector3d e_rot =
      OrientationError(arm2.ee_rot(), arm1.ee_rot() * grasp.R12);
  Eigen::Matrix<double, 6, 6> C1 = -Eigen::Matrix<double, 6, 6>::Identity();
  C1.block<3, 3>(0, 3) << 0, -a.z(), a.y(), a.z(), 0, -a.x(), -a.y(), a.x(), 0;
  C1.block<3, 3>(3, 0).setZero();
  qp.A.resize(6, n);
  qp.A.leftCols(na) = C1 * J1;
  qp.A.rightCols(na) = J2;
  qp.b.resize(6);
  qp.b.head<3>() = -prm.grasp_gain * e_pos;
  qp.b.tail<3>() = -prm.grasp_gain * e_rot;

  // Elastic rows: dampers for both arms (soft), then both velocity boxes.
  qp.G = MatrixXd::Zero(p, n);
  qp.h.resize(p);
  qp.penalty.resize(p);
  int r = 0;
  auto dampers = [&](const Manipulator& arm, const Manipulator::Vector6d& q,
                     int col) {
    for (int i = 0; i < na; ++i, ++r) {
      qp.G(r, col + i) = 1.0;
      qp.h[r] = prm.damper_gain * (arm.q_max()[i] - q[i]);
    }
    for (int i = 0; i < na; ++i, ++r) {
      qp.G(r, col + i) = -1.0;
      qp.h[r] = prm.damper_gain * (q[i] - arm.q_min()[i]);
    }
  };
  dampers(arm1, q1, 0);
  dampers(arm2, q2, na);
  qp.penalty.head(r).setConstant(prm.soft_penalty);
  const int box_start = r;
  auto vel_box = [&](const Manipulator& arm, int col) {
    for (int s = 0; s < 2; ++s) {
      for (int i = 0; i < na; ++i, ++r) {
        qp.G(r, col + i) = s == 0 ? 1.0 : -1.0;
        qp.h[r] = prm.qd_cap > 0 ? prm.qd_cap : arm.qd_max()[i];
      }
    }
  };
  vel_box(arm1, 0);
  vel_box(arm2, na);
  qp.penalty.tail(p - box_start).setConstant(prm.limit_penalty);
  return qp;
}

// ---------------------------------------------------------------------------
// Floating-base humanoid whole-body control
// ---------------------------------------------------------------------------
//
// Pinocchio's sample humanoid (free-flyer base, nv = 34, 28 actuated joints)
// standing on two 6D "foot" contacts (the leg wrist2 joints). The actuated
// torques are eliminated through the actuated rows of the dynamics,
//   tau = [M qdd + nle - Jc^T f]_actuated,
// so x = [qdd (34); f (12)], n = 46.
// Hard equalities (m = 18):
//   base dynamics  [M qdd + nle - Jc^T f]_base = 0   (6, unactuated)
//   contacts       Jc qdd = -Jcdot qd                (12)
// Elastic rows (p = 132): acceleration box (56), torque box (56, dense rows
// in qdd and f), and per foot a 10-row wrench cone [fz >= 0;
// +-fx, +-fy <= mu fz; +-taux, +-tauy <= cop_box fz; tauz <= tauz_max fz].
class Humanoid {
 public:
  Humanoid() : data_(MakeModel()) {
    contact_joint_[0] = model_.getJointId("rleg_wrist2_joint");
    contact_joint_[1] = model_.getJointId("lleg_wrist2_joint");
    q0_ = pinocchio::neutral(model_);
  }

  const pinocchio::Model& model() const { return model_; }
  int nv() const { return model_.nv; }
  int na() const { return model_.nv - 6; }  // actuated dofs
  static constexpr int kNumContacts = 2;
  static constexpr int kContactDim = 6;

  const VectorXd& q_neutral() const { return q0_; }

  // Configuration from a tangent-space perturbation of neutral (handles the
  // free-flyer quaternion correctly).
  VectorXd Configuration(const VectorXd& dq) const {
    return pinocchio::integrate(model_, q0_, dq);
  }

  void Compute(const VectorXd& q, const VectorXd& qd) {
    pinocchio::crba(model_, data_, q);
    M_ = data_.M.selfadjointView<Eigen::Upper>();
    pinocchio::nonLinearEffects(model_, data_, q, qd);
    nle_ = data_.nle;
    pinocchio::computeJointJacobians(model_, data_, q);
    pinocchio::forwardKinematics(model_, data_, q, qd,
                                 VectorXd::Zero(model_.nv));
    for (int c = 0; c < kNumContacts; ++c) {
      const auto jid = contact_joint_[c];
      Jc_[c].setZero(6, model_.nv);
      pinocchio::getJointJacobian(model_, data_, jid,
                                  pinocchio::LOCAL_WORLD_ALIGNED, Jc_[c]);
      Jcdot_qd_[c] = pinocchio::getClassicalAcceleration(
                         model_, data_, jid, pinocchio::LOCAL_WORLD_ALIGNED)
                         .toVector();
    }
    Jcom_ = pinocchio::jacobianCenterOfMass(model_, data_, q);
    pinocchio::centerOfMass(model_, data_, q, qd, VectorXd::Zero(model_.nv));
    com_ = data_.com[0];
    vcom_ = data_.vcom[0];
    acom_bias_ = data_.acom[0];  // Jcomdot * qd (computed with qdd = 0)
  }

  const MatrixXd& M() const { return M_; }
  const VectorXd& nle() const { return nle_; }
  const MatrixXd& Jc(int c) const { return Jc_[c]; }
  const Eigen::Matrix<double, 6, 1>& Jcdot_qd(int c) const {
    return Jcdot_qd_[c];
  }
  const MatrixXd& Jcom() const { return Jcom_; }
  const Vector3d& com() const { return com_; }
  const Vector3d& vcom() const { return vcom_; }
  const Vector3d& acom_bias() const { return acom_bias_; }

  // Actuated torques implied by (qdd, f) through the eliminated dynamics
  // rows (valid after Compute): tau = [M qdd + nle - Jc^T f]_actuated.
  VectorXd Torques(const VectorXd& qdd, const VectorXd& f) const {
    VectorXd gen = M_ * qdd + nle_;
    for (int c = 0; c < kNumContacts; ++c) {
      gen -= Jc_[c].transpose() * f.segment<kContactDim>(kContactDim * c);
    }
    return gen.tail(model_.nv - 6);
  }

 private:
  pinocchio::Model& MakeModel() {
    pinocchio::buildModels::humanoid(model_, /*usingFF=*/true);
    return model_;
  }

  pinocchio::Model model_;
  pinocchio::Data data_;
  pinocchio::JointIndex contact_joint_[kNumContacts];
  VectorXd q0_;

  MatrixXd M_;
  VectorXd nle_;
  MatrixXd Jc_[kNumContacts];
  Eigen::Matrix<double, 6, 1> Jcdot_qd_[kNumContacts];
  MatrixXd Jcom_;
  Vector3d com_, vcom_, acom_bias_;
};

struct HumanoidParams {
  double tau_max = 100.0;   // sample model has no meaningful effort limits
  double qdd_max = 50.0;
  double mu = 0.6;          // friction coefficient
  double cop_box = 0.05;    // CoP box: |taux|, |tauy| <= cop_box * fz
  double tauz_max = 0.4;    // |tauz| <= tauz_max * fz
  double w_com = 1.0;       // CoM task weight
  double w_posture = 1e-2;
  double w_force = 1e-6;
  double accel_penalty = 1e3;
  double torque_penalty = 1e5;
  double friction_penalty = 1e4;
};

// Builds the humanoid WBC QP at the current model state for a desired CoM
// acceleration. Variable layout x = [qdd (nv); f (12)].
inline RobotQP BuildHumanoidWBC(const Humanoid& robot,
                                const HumanoidParams& prm,
                                const VectorXd& qd, const Vector3d& acom_des) {
  const int nv = robot.nv();
  const int na = robot.na();
  const int nc = Humanoid::kNumContacts;
  const int nf = nc * Humanoid::kContactDim;
  const int n = nv + nf;
  const int m = 6 + nc * Humanoid::kContactDim;
  const int p = 2 * na + 2 * na + nc * 10;  // accel box uses actuated dofs

  RobotQP qp;
  // Objective: CoM acceleration task + posture + force regularizer.
  qp.Q = MatrixXd::Zero(n, n);
  qp.q = VectorXd::Zero(n);
  const MatrixXd& Jcom = robot.Jcom();
  qp.Q.topLeftCorner(nv, nv) =
      prm.w_com * (Jcom.transpose() * Jcom) +
      prm.w_posture * MatrixXd::Identity(nv, nv);
  qp.q.head(nv) =
      -prm.w_com * (Jcom.transpose() * (acom_des - robot.acom_bias()));
  qp.Q.bottomRightCorner(nf, nf) = prm.w_force * MatrixXd::Identity(nf, nf);

  // Hard equalities.
  qp.A = MatrixXd::Zero(m, n);
  qp.b = VectorXd::Zero(m);
  // Unactuated base dynamics: [M qdd - Jc^T f]_base = -nle_base.
  qp.A.topLeftCorner(6, nv) = robot.M().topRows(6);
  for (int c = 0; c < nc; ++c) {
    qp.A.block(0, nv + 6 * c, 6, 6) =
        -robot.Jc(c).leftCols(6).transpose();
  }
  qp.b.head(6) = -robot.nle().head(6);
  // Contacts: Jc qdd = -Jcdot qd.
  for (int c = 0; c < nc; ++c) {
    qp.A.block(6 + 6 * c, 0, 6, nv) = robot.Jc(c);
    qp.b.segment(6 + 6 * c, 6) = -robot.Jcdot_qd(c);
  }

  qp.G = MatrixXd::Zero(p, n);
  qp.h.resize(p);
  qp.penalty.resize(p);
  int r = 0;
  // Acceleration box on the actuated dofs.
  for (int i = 0; i < na; ++i, ++r) {
    qp.G(r, 6 + i) = 1.0;
    qp.h[r] = prm.qdd_max;
    qp.penalty[r] = prm.accel_penalty;
  }
  for (int i = 0; i < na; ++i, ++r) {
    qp.G(r, 6 + i) = -1.0;
    qp.h[r] = prm.qdd_max;
    qp.penalty[r] = prm.accel_penalty;
  }
  // Torque box through the eliminated dynamics: tau_i = M_(6+i) qdd
  // + nle_(6+i) - sum_c (Jc^T f)_(6+i); +-tau_i <= tau_max.
  for (int sign = 0; sign < 2; ++sign) {
    const double s = sign == 0 ? 1.0 : -1.0;
    for (int i = 0; i < na; ++i, ++r) {
      qp.G.row(r).head(nv) = s * robot.M().row(6 + i);
      for (int c = 0; c < nc; ++c) {
        qp.G.row(r).segment(nv + 6 * c, 6) =
            -s * robot.Jc(c).col(6 + i).transpose();
      }
      qp.h[r] = prm.tau_max - s * robot.nle()[6 + i];
      qp.penalty[r] = prm.torque_penalty;
    }
  }
  // Contact wrench cone per foot, f = [fx fy fz taux tauy tauz], 10 rows:
  //   -fz <= 0;  +-fx <= mu fz;  +-fy <= mu fz;
  //   +-taux <= cop_box fz;  +-tauy <= cop_box fz;  tauz <= tauz_max fz.
  for (int c = 0; c < nc; ++c) {
    const int f0 = nv + 6 * c;
    auto cone_row = [&](int axis, double sign, double coef) {
      qp.G(r, f0 + axis) = sign;
      qp.G(r, f0 + 2) = -coef;  // relative to fz
      qp.h[r] = 0.0;
      qp.penalty[r] = prm.friction_penalty;
      ++r;
    };
    qp.G(r, f0 + 2) = -1.0;  // fz >= 0
    qp.h[r] = 0.0;
    qp.penalty[r] = prm.friction_penalty;
    ++r;
    cone_row(0, 1.0, prm.mu);
    cone_row(0, -1.0, prm.mu);
    cone_row(1, 1.0, prm.mu);
    cone_row(1, -1.0, prm.mu);
    cone_row(3, 1.0, prm.cop_box);
    cone_row(3, -1.0, prm.cop_box);
    cone_row(4, 1.0, prm.cop_box);
    cone_row(4, -1.0, prm.cop_box);
    cone_row(5, 1.0, prm.tauz_max);
  }
  return qp;
}

}  // namespace robot_control
