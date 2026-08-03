// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2005-2022 by Timothy A. Davis.
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_LDLT_HPP
#define PIQP_SPARSE_LDLT_HPP

#include "piqp/fwd.hpp"
#include "piqp/typedefs.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I>
struct LDLt
{
    Vec<I> etree;  // elimination tree
    // L in CSC
    Vec<I> L_cols; // column starts[n+1]
    Vec<I> L_nnz;  // number of non-zeros per column[n]
    Vec<I> L_ind;  // row indices
    Vec<T> L_vals; // values

    Vec<T> D;      // diagonal matrix D
    Vec<T> D_inv;  // inverse of D

    // working variables used in numerical factorization
    struct {
        Vec<I> flag;
        Vec<I> pattern;
        Vec<T> y;
    } work;

    void factorize_symbolic_upper_triangular(const SparseMat<T, I>& A);

    isize factorize_numeric_upper_triangular(const SparseMat<T, I>& A);

    void lsolve(Vec<T>& x);

    void dsolve(Vec<T>& x);

    void ltsolve(Vec<T>& x);

    void solve_inplace(Vec<T>& x);
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/common.hpp"

namespace piqp
{

namespace sparse
{

extern template struct LDLt<common::Scalar, common::StorageIndex>;

} // namespace sparse

} // namespace piqp

#else
#include "piqp/sparse/ldlt.tpp"
#endif

#endif //PIQP_SPARSE_LDLT_HPP
