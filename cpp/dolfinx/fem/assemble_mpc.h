
#pragma once

#include "Function.h"
#include "FunctionSpace.h"
#include "MPC.h"
#include "assembler.h"
#include "pack.h"
#include "traits.h"
#include "utils.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace dolfinx::fem
{

/// @brief Assemble bilinear form with a multipoint constraint into a matrix.
/// Matrix must already be initialised, with suitable sparsity.
/// Does not zero or finalise the matrix.
/// @param[in] mpc Multipoint constraint.
/// @param[in] mat_add The function for adding values into the matrix.
/// @param[in] a The bilinear form to assemble.
/// @param[in] bcs Dirichlet boundary conditions.
template <dolfinx::scalar T, std::floating_point U>
void assemble_matrix_mpc(
    const fem::MPC<T, U>& mpc, auto mat_add, const fem::Form<T, U>& a,
    const std::vector<std::reference_wrapper<const DirichletBC<T, U>>>& bcs)
{
  // Check functionspace is the same for rows and cols
  if (a.function_spaces().size() != 2)
    throw std::runtime_error("Bilinear form required");

  // Ensure column and row space are the same
  if (a.function_spaces()[0].get() != a.function_spaces()[1].get())
    throw std::runtime_error("Different FunctionSpaces not supported.");

  // Check that DirichletBCs and MPC constraints do not conflict
  for (auto bc : bcs)
  {
    for (std::int32_t dof : bc.get().dof_indices().first)
    {
      spdlog::debug("BC dof {}", dof);
      if (!mpc.constraint(dof).first.empty())
        throw std::runtime_error("Clashing MPC constraint and DirichletBC");
    }
  }

  using cmdspan2_t = md::mdspan<const T, md::dextents<std::size_t, 2>>;
  using cmdspan2T_t
      = md::mdspan<const T, md::dextents<std::size_t, 2>, md::layout_left>;
  using mdspan2_t = md::mdspan<T, md::dextents<std::size_t, 2>>;

  std::vector<T> cache;
  auto mat_add_mpc = [mat_add, &mpc, &cache](std::span<const std::int32_t> rows,
                                             std::span<const std::int32_t> cols,
                                             std::span<const T> vals) mutable
  {
    std::vector<std::int32_t> mod_rows = mpc.modified_dofs(rows);
    int kmat_size = rows.size() * mod_rows.size();
    cache.resize(kmat_size * 2 + mod_rows.size() * mod_rows.size());
    std::fill(cache.begin(), cache.end(), 0.0);
    std::span<T> Kmat(cache.data(), kmat_size);
    mpc.Kmat(rows, Kmat);
    cmdspan2_t K(Kmat.data(), rows.size(), mod_rows.size());
    cmdspan2T_t KT(Kmat.data(), rows.size(), mod_rows.size());
    cmdspan2_t Ae(vals.data(), rows.size(), cols.size());
    mdspan2_t A0(cache.data() + kmat_size, rows.size(), mod_rows.size());
    mdspan2_t A1(cache.data() + 2 * kmat_size, mod_rows.size(),
                 mod_rows.size());
    // A0 = Ae.K^T
    math::dot(Ae, KT, A0);
    // A1 = K.Ae.K^T
    math::dot(K, A0, A1);

    // Revise rows, cols and vals for MPC
    mat_add(mod_rows, mod_rows, std::span<T>(A1.data_handle(), A1.size()));
  };

  // Prepare constants and coefficients
  const std::vector<T> constants = pack_constants(a);
  auto coefficients = allocate_coefficient_storage(a);
  pack_coefficients(a, coefficients);

  // Main assembly
  assemble_matrix(mat_add_mpc, a, bcs);

  // Insert constraint u_i = sum(a_j u_j)
  // N.B. assumes b_i = 0 (make sure this is done in RHS)
  for (int dof = 0; dof < mpc.V()->dofmap()->index_map->size_local(); ++dof)
  {
    std::pair<std::vector<std::int32_t>, std::vector<T>> c
        = mpc.constraint(dof);
    if (!c.first.empty())
    {
      c.first.push_back(dof);
      c.second.push_back(-1.0);
      std::vector<T> v;
      v.reserve(c.second.size() * c.second.size());
      for (std::size_t i = 0; i < c.second.size(); ++i)
        for (std::size_t j = 0; j < c.second.size(); ++j)
        {
          v.push_back(c.second[i] * c.second[j]);
        }
      mat_add(c.first, c.first, v);
    }
  }
}

} // namespace dolfinx::fem
