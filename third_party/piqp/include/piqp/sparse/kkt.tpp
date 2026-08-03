// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_KKT_TPP
#define PIQP_SPARSE_KKT_TPP

#include "piqp/sparse/kkt.hpp"
#include "piqp/utils/tracy.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I, int Mode, typename Ordering>
KKT<T, I, Mode, Ordering>::KKT(const Data<T, I>& data)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::constructor");

    isize n_kkt = kkt_size(data);

    // init workspace
    m_z_reg_inv.resize(data.m);
    work_z.resize(data.m);
    rhs.resize(n_kkt);
    rhs_perm.resize(n_kkt);

    this->init_workspace(data);
    SparseMat<T, I> KKT = this->create_kkt_matrix(data);

    ordering.init(KKT);
    PKi = permute_sparse_symmetric_matrix(KKT, PKPt, ordering);

    ldlt.factorize_symbolic_upper_triangular(PKPt);
}

template<typename T, typename I, int Mode, typename Ordering>
std::unique_ptr<KKTSolverBase<T, I, PIQP_SPARSE>> KKT<T, I, Mode, Ordering>::clone() const
{
    return std::make_unique<KKT>(*this);
}

template<typename T, typename I, int Mode, typename Ordering>
void KKT<T, I, Mode, Ordering>::update_data(const Data<T, I>& data, int options)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::update_data");
    this->update_data_impl(data, options);
}

template<typename T, typename I, int Mode, typename Ordering>
bool KKT<T, I, Mode, Ordering>::update_scalings_and_factor(const Data<T, I>& data, const T& delta, const Vec<T>& x_reg, const Vec<T>& z_reg)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::update_scalings_and_factor");

    m_delta = delta;
    m_z_reg_inv.array() = z_reg.array().inverse();

    {
        PIQP_TRACY_ZoneScopedN("piqp::KKT::update_scalings_and_factor::kkt_cost");
        this->update_kkt_cost_scalings(data, x_reg);
    }
    {
        PIQP_TRACY_ZoneScopedN("piqp::KKT::update_scalings_and_factor::kkt_equality");
        this->update_kkt_equality_scalings(data);
    }
    {
        PIQP_TRACY_ZoneScopedN("piqp::KKT::update_scalings_and_factor::kkt_inequality");
        this->update_kkt_inequality_scaling(data, z_reg);
    }

    isize n = ldlt.factorize_numeric_upper_triangular(PKPt);
    return n == PKPt.cols();
}

template<typename T, typename I, int Mode, typename Ordering>
void KKT<T, I, Mode, Ordering>::solve(const Data<T, I>& data, const Vec<T>& rhs_x, const Vec<T>& rhs_y, const Vec<T>& rhs_z, Vec<T>& lhs_x, Vec<T>& lhs_y, Vec<T>& lhs_z)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::solve");

    T delta_inv = T(1) / m_delta;

    rhs.head(data.n).noalias() = rhs_x;
    if (Mode == KKTMode::KKT_FULL)
    {
        rhs.segment(data.n, data.p).noalias() = rhs_y;
        rhs.tail(data.m).noalias() = rhs_z;
    }
    else if (Mode == KKTMode::KKT_EQ_ELIMINATED)
    {
        rhs.head(data.n).noalias() += delta_inv * data.AT * rhs_y;
        rhs.tail(data.m).noalias() = rhs_z;
    }
    else if (Mode == KKTMode::KKT_INEQ_ELIMINATED)
    {
        work_z.array() = m_z_reg_inv.array() * rhs_z.array();
        rhs.head(data.n).noalias() += data.GT * work_z;
        rhs.tail(data.p).noalias() = rhs_y;
    }
    else
    {
        work_z.array() = m_z_reg_inv.array() * rhs_z.array();
        rhs.noalias() += data.GT * work_z;
        rhs.noalias() += delta_inv * data.AT * rhs_y;
    }

    ordering.template perm<T>(rhs_perm, rhs);

    solve_ldlt_in_place(rhs_perm);

    // we reuse the memory of rhs for the solution
    ordering.template permt<T>(rhs, rhs_perm);

    lhs_x.noalias() = rhs.head(data.n);

    if (Mode == KKTMode::KKT_FULL)
    {
        lhs_y.noalias() = rhs.segment(data.n, data.p);

        lhs_z.noalias() = rhs.tail(data.m);
    }
    else if (Mode == KKTMode::KKT_EQ_ELIMINATED)
    {
        lhs_y.noalias() = delta_inv * data.AT.transpose() * lhs_x;
        lhs_y.noalias() -= delta_inv * rhs_y;

        lhs_z.noalias() = rhs.tail(data.m);
    }
    else if (Mode == KKTMode::KKT_INEQ_ELIMINATED)
    {
        lhs_y.noalias() = rhs.tail(data.p);

        lhs_z.noalias() = data.GT.transpose() * lhs_x;
        lhs_z.noalias() -= rhs_z;
        lhs_z.array() *= m_z_reg_inv.array();
    }
    else
    {
        lhs_y.noalias() = delta_inv * data.AT.transpose() * lhs_x;
        lhs_y.noalias() -= delta_inv * rhs_y;

        lhs_z.noalias() = data.GT.transpose() * lhs_x;
        lhs_z.noalias() -= rhs_z;
        lhs_z.array() *= m_z_reg_inv.array();
    }
}

template<typename T, typename I, int Mode, typename Ordering>
void KKT<T, I, Mode, Ordering>::eval_P_x(const Data<T, I>& data, const T& alpha, const Vec<T>& x, Vec<T>& z)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::eval_P_x");

    z.noalias() = alpha * data.P_utri * x;
    z.noalias() += alpha * data.P_utri.transpose().template triangularView<Eigen::StrictlyLower>() * x;
}

template<typename T, typename I, int Mode, typename Ordering>
void KKT<T, I, Mode, Ordering>::eval_A_xn_and_AT_xt(const Data<T, I>& data, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::eval_A_xn_and_AT_xt");

    zn.noalias() = alpha_n * data.AT.transpose() * xn;
    zt.noalias() = alpha_t * data.AT * xt;
}

template<typename T, typename I, int Mode, typename Ordering>
void KKT<T, I, Mode, Ordering>::eval_G_xn_and_GT_xt(const Data<T, I>& data, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::eval_G_xn_and_GT_xt");

    zn.noalias() = alpha_n * data.GT.transpose() * xn;
    zt.noalias() = alpha_t * data.GT * xt;
}

template<typename T, typename I, int Mode, typename Ordering>
isize KKT<T, I, Mode, Ordering>::kkt_size(const Data<T, I>& data)
{
    isize n_kkt;
    if (Mode == KKTMode::KKT_FULL)
    {
        n_kkt = data.n + data.p + data.m;
    }
    else if (Mode == KKTMode::KKT_EQ_ELIMINATED)
    {
        n_kkt = data.n + data.m;
    }
    else if (Mode == KKTMode::KKT_INEQ_ELIMINATED)
    {
        n_kkt = data.n + data.p;
    }
    else
    {
        n_kkt = data.n;
    }
    return n_kkt;
}

template<typename T, typename I, int Mode, typename Ordering>
void KKT<T, I, Mode, Ordering>::solve_ldlt_in_place(Vec<T>& x)
{
#ifdef PIQP_DEBUG_PRINT
    Vec<T> x_copy = x;
#endif

    ldlt.solve_inplace(x);

#ifdef PIQP_DEBUG_PRINT
    Vec<T> rhs_x = PKPt.template triangularView<Eigen::Upper>() * x;
    rhs_x += PKPt.transpose().template triangularView<Eigen::StrictlyLower>() * x;
    std::cout << "ldlt_error: " << (x_copy - rhs_x).template lpNorm<Eigen::Infinity>() << std::endl;
#endif
}

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_KKT_TPP
