// Generates the robotics control-loop QP sequences (see robot_control.hpp)
// and writes them to a binary file for the robot benchmarks to replay.
//
// The split exists so the solver benchmarks need no Pinocchio at build OR
// run time: binary Pinocchio distributions are generic (non-AVX) builds, so
// this generator must be compiled with a matching generic -march, while the
// benchmarks can be rebuilt freely (e.g. with -march=native) for codegen
// comparisons. The model evaluation is done here, up front, so the
// benchmark timings measure the solver only.
//
//   diff-ik   differential IK, x = qd        n =  6, m =  0, p =  24
//   arm-osc   torque-level OSC, x = tau      n =  6, m =  0, p =  24
//   hum-wbc   humanoid WBC, x = [qdd; f]     n = 46, m = 18, p = 132
// (arm scenarios are closed loop, the humanoid follows a prescribed sway)
//
// The closed loops are driven by a warm-started elastiqp::Solver at default
// settings; every problem matrix drifts every tick (Q, q, A, b, G, h all
// depend on the state), which is the honest robot-control workload.
//
// Regeneration is not bit-reproducible across machines (FMA availability and
// Eigen version perturb the closed loop), so the file in benchmarks/data/ is
// the canonical artifact. The per-sequence fingerprints printed below let a
// regenerated file be compared statistically: matching hashes mean identical
// bytes; differing hashes with matching stats (to a few significant digits)
// mean an equivalent trajectory on different hardware.
//
// Usage: gen_robot_sequences [output_file]   (default robot_sequences.bin)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "elastiqp/elastiqp.hpp"
#include "qp_io.hpp"
#include "robot_control.hpp"

using Eigen::Vector3d;
using Eigen::VectorXd;
using robot_control::NamedSequence;
using robot_control::RobotQP;

namespace {

std::vector<RobotQP> DiffIKSequence(int ticks) {
  robot_control::Manipulator arm;
  robot_control::ArmParams prm;
  const double dt = 0.01;
  auto q = robot_control::Manipulator::Home();
  arm.Compute(q, robot_control::Manipulator::Vector6d::Zero());
  const Vector3d center = arm.ee_pos();
  const Eigen::Matrix3d rot0 = arm.ee_rot();

  elastiqp::Solver solver;
  std::vector<RobotQP> seq;
  for (int k = 0; k < ticks; ++k) {
    const double t = k * dt;
    const double w = 2 * M_PI / 2.0;
    robot_control::TaskTarget target;
    target.rot = rot0;
    target.pos =
        center + 0.10 * Vector3d(0.0, std::cos(w * t) - 1.0, std::sin(w * t));
    target.vel =
        0.10 * w * Vector3d(0.0, -std::sin(w * t), std::cos(w * t));
    arm.Compute(q, robot_control::Manipulator::Vector6d::Zero());
    seq.push_back(robot_control::BuildDiffIK(arm, prm, q, target));
    const RobotQP& qp = seq.back();
    if (k == 0) {
      solver.setup(qp.Q, qp.q, qp.G, qp.h, qp.penalty);
    } else {
      solver.set_Q(qp.Q);
      solver.set_q(qp.q);
      solver.set_G(qp.G);
      solver.set_h(qp.h);
    }
    q += dt * solver.solve().x;
  }
  return seq;
}

std::vector<RobotQP> ArmOSCSequence(int ticks) {
  robot_control::Manipulator arm;
  robot_control::ArmParams prm;
  const double dt = 1e-3;
  auto q = robot_control::Manipulator::Home();
  robot_control::Manipulator::Vector6d qd =
      robot_control::Manipulator::Vector6d::Zero();
  arm.Compute(q, qd);
  const Vector3d center = arm.ee_pos();
  const Eigen::Matrix3d rot0 = arm.ee_rot();

  elastiqp::Solver solver;
  std::vector<RobotQP> seq;
  for (int k = 0; k < ticks; ++k) {
    const double t = k * dt;
    const double w = 2 * M_PI / 2.0;
    robot_control::TaskTarget target;
    target.rot = rot0;
    target.pos =
        center + 0.05 * Vector3d(0.0, std::cos(w * t) - 1.0, std::sin(w * t));
    target.vel =
        0.05 * w * Vector3d(0.0, -std::sin(w * t), std::cos(w * t));
    arm.Compute(q, qd);
    seq.push_back(robot_control::BuildArmOSC(
        arm, prm, q, qd, robot_control::TaskAcceleration(arm, qd, target)));
    const RobotQP& qp = seq.back();
    if (k == 0) {
      solver.setup(qp.Q, qp.q, qp.G, qp.h, qp.penalty);
    } else {
      solver.set_Q(qp.Q);
      solver.set_q(qp.q);
      solver.set_G(qp.G);
      solver.set_h(qp.h);
    }
    // The OSC solution is the torque command; integrate the resulting
    // acceleration.
    const robot_control::Manipulator::Vector6d qdd =
        arm.Accelerations(solver.solve().x);
    qd += dt * qdd;
    q += dt * qd;
  }
  return seq;
}

std::vector<RobotQP> HumanoidSequence(int ticks) {
  robot_control::Humanoid robot;
  robot_control::HumanoidParams prm;
  const int nv = robot.nv();
  std::vector<RobotQP> seq;
  for (int k = 0; k < ticks; ++k) {
    // Prescribed smooth joint trajectory (a control-loop-rate sway).
    const double t = 0.002 * k;
    VectorXd dq = VectorXd::Zero(nv), qd = VectorXd::Zero(nv);
    for (int i = 6; i < nv; ++i) {
      dq[i] = 0.25 * std::sin(2.0 * t + 0.4 * i);
      qd[i] = 0.5 * std::cos(2.0 * t + 0.4 * i);
    }
    robot.Compute(robot.Configuration(dq), qd);
    const Vector3d acom_des =
        Vector3d(0.5 * std::sin(2.0 * t), 0.5 * std::cos(2.0 * t), -0.5) -
        2.0 * robot.vcom();
    seq.push_back(robot_control::BuildHumanoidWBC(robot, prm, qd, acom_des));
  }
  return seq;
}

// Fingerprints hash the float32 representation -- exactly the bytes that
// SaveSequences(..., float32=true) writes -- so a byte-identical file yields
// an identical hash.
void HashF32(std::uint64_t& h, const double* data, std::ptrdiff_t count) {
  for (std::ptrdiff_t i = 0; i < count; ++i) {
    const float f = static_cast<float>(data[i]);
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    for (int byte = 0; byte < 4; ++byte) {
      h ^= (bits >> (8 * byte)) & 0xff;
      h *= 1099511628211ull;  // FNV-1a
    }
  }
}

void PrintFingerprint(const NamedSequence& s) {
  std::uint64_t h = 14695981039346656037ull;
  double sum_q = 0, sum_h = 0;
  for (const RobotQP& qp : s.qps) {
    HashF32(h, qp.Q.data(), qp.Q.size());
    HashF32(h, qp.q.data(), qp.q.size());
    HashF32(h, qp.A.data(), qp.A.size());
    HashF32(h, qp.b.data(), qp.b.size());
    HashF32(h, qp.G.data(), qp.G.size());
    HashF32(h, qp.h.data(), qp.h.size());
    HashF32(h, qp.penalty.data(), qp.penalty.size());
    sum_q += qp.q.norm();
    sum_h += qp.h.norm();
  }
  const RobotQP& qp = s.qps.front();
  std::printf(
      "%-8s %4d ticks  n=%3d m=%3d p=%3d  hash=%016llx  sum|q|=%.6e  "
      "sum|h|=%.6e\n",
      s.name.c_str(), static_cast<int>(s.qps.size()),
      static_cast<int>(qp.q.size()), static_cast<int>(qp.b.size()),
      static_cast<int>(qp.h.size()), static_cast<unsigned long long>(h), sum_q,
      sum_h);
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "robot_sequences.bin";
  constexpr int kTicks = 500;
  std::vector<NamedSequence> seqs;
  seqs.push_back({"diff-ik", DiffIKSequence(kTicks)});
  seqs.push_back({"arm-osc", ArmOSCSequence(kTicks)});
  seqs.push_back({"hum-wbc", HumanoidSequence(kTicks / 2)});
  robot_control::SaveSequences(path, seqs, /*float32=*/true);
  for (const NamedSequence& s : seqs) PrintFingerprint(s);
  std::printf("wrote %s\n", path);
  return 0;
}
