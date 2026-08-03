// This file is part of PIQP.
//
// Copyright (c) 2025 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_KKT_SYSTEM_HPP
#define PIQP_KKT_SYSTEM_HPP

#include "piqp/kkt_fwd.hpp"
#include "piqp/kkt_solver_base.hpp"
#include "piqp/settings.hpp"
#include "piqp/variables.hpp"
#include "piqp/dense/data.hpp"
#include "piqp/sparse/data.hpp"

namespace piqp
{

template<typename T, typename I, int MatrixType>
class KKTSystem
{
protected:
	using DataType = std::conditional_t<MatrixType == PIQP_DENSE, dense::Data<T>, sparse::Data<T, I>>;

	T m_rho;
	T m_delta;

	Vec<T> P_diag;

	Vec<T> m_s_l;
	Vec<T> m_s_u;
	Vec<T> m_s_bl;
	Vec<T> m_s_bu;
	Vec<T> m_z_l_inv;
	Vec<T> m_z_u_inv;
	Vec<T> m_z_bl_inv;
	Vec<T> m_z_bu_inv;

	Vec<T> m_x_reg;
	Vec<T> m_z_reg;

	Vec<T> rhs_x_bar;
	Vec<T> rhs_z_bar;

	// working variables
	Vec<T> work_x;
	Vec<T> work_z;

	// iterative refinement variables
	Vec<T> ref_err_x;
	Vec<T> ref_err_y;
	Vec<T> ref_err_z;
	Vec<T> ref_lhs_x;
	Vec<T> ref_lhs_y;
	Vec<T> ref_lhs_z;

	bool use_iterative_refinement = false;
	std::unique_ptr<KKTSolverBase<T, I, MatrixType>> kkt_solver;

public:
	KKTSystem() = default;

	KKTSystem(const KKTSystem& other);

	bool init(const DataType& data, const Settings<T>& settings);

	void update_data(const DataType& data, int options);

	bool update_scalings_and_factor(const DataType& data, const Settings<T>& settings, bool iterative_refinement,
		                            const T& rho, const T& delta, const Variables<T>& vars);

	bool solve(const DataType& data, const Settings<T>& settings, const Variables<T>& rhs, Variables<T>& lhs);

	// z = alpha * P * x
	void eval_P_x(const DataType& data, const T& alpha, const Vec<T>& x, Vec<T>& z);

	// zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
	void eval_A_xn_and_AT_xt(const DataType& data, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt);

	// zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
	void eval_G_xn_and_GT_xt(const DataType& data, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt);

	void mul(const DataType& data, const Variables<T>& lhs, Variables<T>& rhs);

	void print_info();

protected:
	void extract_P_diag(const DataType& data);

	bool init_kkt_solver(const DataType& data, const Settings<T>& settings);

	T inf_norm(const Vec<T>& x, const Vec<T>& y, const Vec<T>& z);

	void mul_condensed_kkt(const DataType& data,
                           const Vec<T>& lhs_x, const Vec<T>& lhs_y, const Vec<T>& lhs_z,
						   Vec<T>& rhs_x, Vec<T>& rhs_y, Vec<T>& rhs_z);

	T get_refine_error(const DataType& data,
		              const Vec<T>& lhs_x, const Vec<T>& lhs_y, const Vec<T>& lhs_z,
		              const Vec<T>& rhs_x, const Vec<T>& rhs_y, const Vec<T>& rhs_z,
		              Vec<T>& err_x, Vec<T>& err_y, Vec<T>& err_z);
};

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION

#include "piqp/typedefs.hpp"

namespace piqp
{

extern template class KKTSystem<common::Scalar, common::StorageIndex, PIQP_DENSE>;
extern template class KKTSystem<common::Scalar, common::StorageIndex, PIQP_SPARSE>;

} // namespace piqp

#else
#include "piqp/kkt_system.tpp"
#endif

#endif //PIQP_KKT_SYSTEM_HPP
