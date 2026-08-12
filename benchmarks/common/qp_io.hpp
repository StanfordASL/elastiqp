// Elastic-QP sequence serialization, shared between gen_robot_sequences
// (built against Pinocchio, always compiled with a generic -march to match
// binary Pinocchio distributions) and the robot benchmarks (no Pinocchio
// dependency, so they can be freely rebuilt with -march=native for
// codegen-comparison runs).
//
// File format (little-endian):
//   "EQPS" | int32 version | [int32 dtype (v2 only)] | int32 num_sequences
//   per sequence: int32 name_len | name bytes | int32 num_qps
//     per QP: int32 n, m, p | Q (n*n) | q (n) | A (m*n) | b (m)
//             | G (p*n) | h (p) | penalty (p)
// Matrices are column-major. v1 stores float64; v2 adds a dtype field
// (0 = float64, 1 = float32). The loader accepts both versions and always
// returns double matrices.
//
// float32 storage halves the committed data file. The ~1e-7 relative
// rounding it introduces is harmless for benchmarking: every consumer
// solves the problem exactly as stored, so all solvers see identical data.

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace robot_control {

// One elastic QP instance:
//   minimize    0.5 x^T Q x + q^T x + penalty^T t
//   subject to  A x == b (hard),  G x - t <= h,  t >= 0
struct RobotQP {
  Eigen::MatrixXd Q, A, G;
  Eigen::VectorXd q, b, h, penalty;
};

struct NamedSequence {
  std::string name;
  std::vector<RobotQP> qps;
};

namespace qp_io_detail {

inline void WriteI32(std::ostream& os, std::int32_t v) {
  os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

inline std::int32_t ReadI32(std::istream& is) {
  std::int32_t v = 0;
  is.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}

inline void WriteArray(std::ostream& os, const double* data, std::size_t count,
                       bool float32) {
  if (float32) {
    std::vector<float> staged(data, data + count);
    os.write(reinterpret_cast<const char*>(staged.data()),
             static_cast<std::streamsize>(count * sizeof(float)));
  } else {
    os.write(reinterpret_cast<const char*>(data),
             static_cast<std::streamsize>(count * sizeof(double)));
  }
}

inline void ReadArray(std::istream& is, double* data, std::size_t count,
                      bool float32) {
  if (float32) {
    std::vector<float> staged(count);
    is.read(reinterpret_cast<char*>(staged.data()),
            static_cast<std::streamsize>(count * sizeof(float)));
    for (std::size_t i = 0; i < count; ++i) data[i] = staged[i];
  } else {
    is.read(reinterpret_cast<char*>(data),
            static_cast<std::streamsize>(count * sizeof(double)));
  }
}

}  // namespace qp_io_detail

inline void SaveSequences(const std::string& path,
                          const std::vector<NamedSequence>& seqs,
                          bool float32 = false) {
  namespace d = qp_io_detail;
  std::ofstream os(path, std::ios::binary);
  if (!os) throw std::runtime_error("cannot open for writing: " + path);
  os.write("EQPS", 4);
  d::WriteI32(os, 2);
  d::WriteI32(os, float32 ? 1 : 0);
  d::WriteI32(os, static_cast<std::int32_t>(seqs.size()));
  for (const NamedSequence& seq : seqs) {
    d::WriteI32(os, static_cast<std::int32_t>(seq.name.size()));
    os.write(seq.name.data(),
             static_cast<std::streamsize>(seq.name.size()));
    d::WriteI32(os, static_cast<std::int32_t>(seq.qps.size()));
    for (const RobotQP& qp : seq.qps) {
      const auto n = static_cast<std::int32_t>(qp.q.size());
      const auto m = static_cast<std::int32_t>(qp.b.size());
      const auto p = static_cast<std::int32_t>(qp.h.size());
      d::WriteI32(os, n);
      d::WriteI32(os, m);
      d::WriteI32(os, p);
      d::WriteArray(os, qp.Q.data(), qp.Q.size(), float32);
      d::WriteArray(os, qp.q.data(), qp.q.size(), float32);
      d::WriteArray(os, qp.A.data(), qp.A.size(), float32);
      d::WriteArray(os, qp.b.data(), qp.b.size(), float32);
      d::WriteArray(os, qp.G.data(), qp.G.size(), float32);
      d::WriteArray(os, qp.h.data(), qp.h.size(), float32);
      d::WriteArray(os, qp.penalty.data(), qp.penalty.size(), float32);
    }
  }
  if (!os) throw std::runtime_error("write failed: " + path);
}

inline std::vector<NamedSequence> LoadSequences(const std::string& path) {
  namespace d = qp_io_detail;
  std::ifstream is(path, std::ios::binary);
  if (!is) throw std::runtime_error("cannot open: " + path);
  char magic[4];
  is.read(magic, 4);
  if (!is || std::string(magic, 4) != "EQPS") {
    throw std::runtime_error("bad magic in " + path);
  }
  const std::int32_t version = d::ReadI32(is);
  if (version != 1 && version != 2) {
    throw std::runtime_error("unsupported version");
  }
  bool float32 = false;
  if (version == 2) {
    const std::int32_t dtype = d::ReadI32(is);
    if (dtype != 0 && dtype != 1) throw std::runtime_error("unsupported dtype");
    float32 = dtype == 1;
  }
  std::vector<NamedSequence> seqs(
      static_cast<std::size_t>(d::ReadI32(is)));
  for (NamedSequence& seq : seqs) {
    std::string name(static_cast<std::size_t>(d::ReadI32(is)), '\0');
    is.read(name.data(), static_cast<std::streamsize>(name.size()));
    seq.name = name;
    seq.qps.resize(static_cast<std::size_t>(d::ReadI32(is)));
    for (RobotQP& qp : seq.qps) {
      const std::int32_t n = d::ReadI32(is);
      const std::int32_t m = d::ReadI32(is);
      const std::int32_t p = d::ReadI32(is);
      qp.Q.resize(n, n);
      qp.q.resize(n);
      qp.A.resize(m, n);
      qp.b.resize(m);
      qp.G.resize(p, n);
      qp.h.resize(p);
      qp.penalty.resize(p);
      d::ReadArray(is, qp.Q.data(), qp.Q.size(), float32);
      d::ReadArray(is, qp.q.data(), qp.q.size(), float32);
      d::ReadArray(is, qp.A.data(), qp.A.size(), float32);
      d::ReadArray(is, qp.b.data(), qp.b.size(), float32);
      d::ReadArray(is, qp.G.data(), qp.G.size(), float32);
      d::ReadArray(is, qp.h.data(), qp.h.size(), float32);
      d::ReadArray(is, qp.penalty.data(), qp.penalty.size(), float32);
    }
    if (!is) throw std::runtime_error("truncated file: " + path);
  }
  return seqs;
}

}  // namespace robot_control
