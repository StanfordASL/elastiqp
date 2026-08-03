// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_KKT_HPP
#define PIQP_SPARSE_KKT_HPP

#include "piqp/sparse/data.hpp"
#include "piqp/sparse/ldlt.hpp"
#include "piqp/sparse/ordering.hpp"
#include "piqp/sparse/utils.hpp"
#include "piqp/kkt_solver_base.hpp"
#include "piqp/kkt_fwd.hpp"
#include "piqp/sparse/kkt_full.hpp"
#include "piqp/sparse/kkt_eq_eliminated.hpp"
#include "piqp/sparse/kkt_ineq_eliminated.hpp"
#include "piqp/sparse/kkt_all_eliminated.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I, int Mode = KKTMode::KKT_FULL, typename Ordering = AMDOrdering<I>>
class KKT : public KKTSolverBase<T, I, PIQP_SPARSE>, public KKTImpl<KKT<T, I, Mode, Ordering>, T, I, Mode>
{
protected:
    friend class KKTImpl<KKT<T, I, Mode, Ordering>, T, I, Mode>;

    T m_delta;

    Vec<T> m_z_reg_inv;

    Ordering ordering;
    SparseMat<T, I> PKPt; // permuted KKT matrix, upper triangular only
    Vec<I> PKi;           // mapping of row indices of KKT matrix to permuted KKT matrix

    LDLt<T, I> ldlt;

    Vec<T> work_z;        // working variable
    Vec<T> rhs;           // stores the rhs and the solution
    Vec<T> rhs_perm;      // permuted rhs

public:
    explicit KKT(const Data<T, I>& data);

    std::unique_ptr<KKTSolverBase<T, I, PIQP_SPARSE>> clone() const override;

    void update_data(const Data<T, I>& data, int options) override;

    bool update_scalings_and_factor(const Data<T, I>& data, const T& delta, const Vec<T>& x_reg, const Vec<T>& z_reg) override;

    void solve(const Data<T, I>& data, const Vec<T>& rhs_x, const Vec<T>& rhs_y, const Vec<T>& rhs_z, Vec<T>& lhs_x, Vec<T>& lhs_y, Vec<T>& lhs_z) override;

    // z = alpha * P * x
    void eval_P_x(const Data<T, I>& data, const T& alpha, const Vec<T>& x, Vec<T>& z) override;

    // zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
    void eval_A_xn_and_AT_xt(const Data<T, I>& data, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt) override;

    // zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
    void eval_G_xn_and_GT_xt(const Data<T, I>& data, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt) override;

    SparseMat<T, I>& internal_kkt_mat()
    {
        return PKPt;
    }

protected:
    isize kkt_size(const Data<T, I>& data);

    void solve_ldlt_in_place(Vec<T>& x);
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/common.hpp"

namespace piqp
{

namespace sparse
{

extern template class KKT<common::Scalar, common::StorageIndex, KKTMode::KKT_FULL, common::sparse::Ordering>;
extern template class KKT<common::Scalar, common::StorageIndex, KKTMode::KKT_EQ_ELIMINATED, common::sparse::Ordering>;
extern template class KKT<common::Scalar, common::StorageIndex, KKTMode::KKT_INEQ_ELIMINATED, common::sparse::Ordering>;
extern template class KKT<common::Scalar, common::StorageIndex, KKTMode::KKT_ALL_ELIMINATED, common::sparse::Ordering>;

} // namespace sparse

} // namespace piqp

#else
#include "piqp/sparse/kkt.tpp"
#endif

#endif //PIQP_SPARSE_KKT_HPP
