// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DENSE_KKT_HPP
#define PIQP_DENSE_KKT_HPP

#include "piqp/kkt_solver_base.hpp"
#include "piqp/kkt_fwd.hpp"
#include "piqp/dense/data.hpp"
#include "piqp/dense/ldlt_no_pivot.hpp"

namespace piqp
{

namespace dense
{

template<typename T>
class KKT : public KKTSolverBase<T, int, PIQP_DENSE> {
protected:
    T m_delta;

    Vec<T> m_z_reg_inv;

    Mat<T> kkt_mat;
    Eigen::LLT<Mat<T>, Eigen::Lower> llt;

    Mat<T> AT_A;
    Mat<T> W_delta_inv_G;
    Vec<T> work_z; // working variable

public:
    explicit KKT(const Data<T>& data);

    std::unique_ptr<KKTSolverBase<T, int, PIQP_DENSE>> clone() const override;

    void update_data(const Data<T>& data, int options) override;

    bool update_scalings_and_factor(const Data<T>& data, const T& delta, const Vec<T>& x_reg, const Vec<T>& z_reg) override;

    void solve(const Data<T>& data, const Vec<T>& rhs_x, const Vec<T>& rhs_y, const Vec<T>& rhs_z, Vec<T>& lhs_x, Vec<T>& lhs_y, Vec<T>& lhs_z) override;

    // z = alpha * P * x
    void eval_P_x(const Data<T>& data, const T& alpha, const Vec<T>& x, Vec<T>& z) override;

    // zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
    void eval_A_xn_and_AT_xt(const Data<T>& data, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt) override;

    // zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
    void eval_G_xn_and_GT_xt(const Data<T>& data, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt) override;

    Mat<T>& internal_kkt_mat()
    {
        return kkt_mat;
    }

protected:
    void update_kkt(const Data<T>& data, const Vec<T>& x_reg);

    void solve_ldlt_in_place(Vec<T>& x);
};

} // namespace dense

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/common.hpp"

namespace piqp
{

namespace dense
{

extern template class KKT<common::Scalar>;

} // namespace dense

} // namespace piqp

#else
#include "piqp/dense/kkt.tpp"
#endif

#endif //PIQP_DENSE_KKT_HPP
