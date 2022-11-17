// Copyright (C) 2018-2022 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "assemble_matrix_impl.h"
#include "assemble_scalar_impl.h"
#include "assemble_vector_impl.h"
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace dolfinx::fem
{
template <typename T>
class DirichletBC;
template <typename T>
class Form;
class FunctionSpace;

// -- Helper functions -----------------------------------------------------

/// @brief Create a map of `std::span`s from a map of `std::vector`s
template <typename T>
std::map<std::pair<fem::IntegralType, int>, std::pair<std::span<const T>, int>>
make_coefficients_span(const std::map<std::pair<IntegralType, int>,
                                      std::pair<std::vector<T>, int>>& coeffs)
{
  using Key = typename std::remove_reference_t<decltype(coeffs)>::key_type;
  std::map<Key, std::pair<std::span<const T>, int>> c;
  std::transform(coeffs.cbegin(), coeffs.cend(), std::inserter(c, c.end()),
                 [](auto& e) -> typename decltype(c)::value_type {
                   return {e.first, {e.second.first, e.second.second}};
                 });
  return c;
}

// -- Scalar ----------------------------------------------------------------

/// @brief Assemble functional into scalar.
///
/// The caller supplies the form constants and coefficients for this
/// version, which has efficiency benefits if the data can be re-used
/// for multiple calls.
/// @note Caller is responsible for accumulation across processes.
/// @param[in] M The form (functional) to assemble
/// @param[in] constants The constants that appear in `M`
/// @param[in] coefficients The coefficients that appear in `M`
/// @return The contribution to the form (functional) from the local
/// process
template <typename T>
T assemble_scalar(
    const Form<T>& M, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  return impl::assemble_scalar(M, constants, coefficients);
}

/// Assemble functional into scalar
/// @note Caller is responsible for accumulation across processes.
/// @param[in] M The form (functional) to assemble
/// @return The contribution to the form (functional) from the local
///   process
template <typename T>
T assemble_scalar(const Form<T>& M)
{
  const std::vector<T> constants = pack_constants(M);
  auto coefficients = allocate_coefficient_storage(M);
  pack_coefficients(M, coefficients);
  return assemble_scalar(M, std::span(constants),
                         make_coefficients_span(coefficients));
}

// -- Vectors ----------------------------------------------------------------

/// @brief Assemble linear form into a vector.
///
/// The caller supplies the form constants and coefficients for this
/// version, which has efficiency benefits if the data can be re-used
/// for multiple calls.
/// @param[in,out] b The vector to be assembled. It will not be zeroed
/// before assembly.
/// @param[in] L The linear forms to assemble into b
/// @param[in] constants The constants that appear in `L`
/// @param[in] coefficients The coefficients that appear in `L`
template <typename T>
void assemble_vector(
    std::span<T> b, const Form<T>& L, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  impl::assemble_vector(b, L, constants, coefficients);
}

/// @brief Assemble linear form into a vector
/// @param[in,out] b The vector to be assembled. It will not be zeroed
/// before assembly.
/// @param[in] L The linear forms to assemble into b
template <typename T>
void assemble_vector(std::span<T> b, const Form<T>& L)
{
  auto coefficients = allocate_coefficient_storage(L);
  pack_coefficients(L, coefficients);
  const std::vector<T> constants = pack_constants(L);
  assemble_vector(b, L, std::span(constants),
                  make_coefficients_span(coefficients));
}

// FIXME: clarify how x0 is used
// FIXME: if bcs entries are set

// FIXME: need to pass an array of Vec for x0?
// FIXME: clarify zeroing of vector

/// Modify b such that:
///
///   b <- b - scale * A_j (g_j - x0_j)
///
/// where j is a block (nest) index. For a non-blocked problem j = 0. The
/// boundary conditions bcs1 are on the trial spaces V_j. The forms in
/// [a] must have the same test space as L (from which b was built), but the
/// trial space may differ. If x0 is not supplied, then it is treated as
/// zero.
///
/// Ghost contributions are not accumulated (not sent to owner). Caller
/// is responsible for calling VecGhostUpdateBegin/End.
template <typename T>
void apply_lifting(
    std::span<T> b, const std::vector<std::shared_ptr<const Form<T>>>& a,
    const std::vector<std::span<const T>>& constants,
    const std::vector<std::map<std::pair<IntegralType, int>,
                               std::pair<std::span<const T>, int>>>& coeffs,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC<T>>>>& bcs1,
    const std::vector<std::span<const T>>& x0, double scale)
{
  impl::apply_lifting(b, a, constants, coeffs, bcs1, x0, scale);
}

/// Modify b such that:
///
///   b <- b - scale * A_j (g_j - x0_j)
///
/// where j is a block (nest) index. For a non-blocked problem j = 0. The
/// boundary conditions bcs1 are on the trial spaces V_j. The forms in
/// [a] must have the same test space as L (from which b was built), but the
/// trial space may differ. If x0 is not supplied, then it is treated as
/// zero.
///
/// Ghost contributions are not accumulated (not sent to owner). Caller
/// is responsible for calling VecGhostUpdateBegin/End.
template <typename T>
void apply_lifting(
    std::span<T> b, const std::vector<std::shared_ptr<const Form<T>>>& a,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC<T>>>>& bcs1,
    const std::vector<std::span<const T>>& x0, double scale)
{
  std::vector<
      std::map<std::pair<IntegralType, int>, std::pair<std::vector<T>, int>>>
      coeffs;
  std::vector<std::vector<T>> constants;
  for (auto _a : a)
  {
    if (_a)
    {
      auto coefficients = allocate_coefficient_storage(*_a);
      pack_coefficients(*_a, coefficients);
      coeffs.push_back(coefficients);
      constants.push_back(pack_constants(*_a));
    }
    else
    {
      coeffs.emplace_back();
      constants.emplace_back();
    }
  }

  std::vector<std::span<const T>> _constants(constants.begin(),
                                             constants.end());
  std::vector<std::map<std::pair<IntegralType, int>,
                       std::pair<std::span<const T>, int>>>
      _coeffs;
  std::transform(coeffs.cbegin(), coeffs.cend(), std::back_inserter(_coeffs),
                 [](auto& c) { return make_coefficients_span(c); });
  apply_lifting(b, a, _constants, _coeffs, bcs1, x0, scale);
}

// -- Matrices ---------------------------------------------------------------

/// Assemble bilinear form into a matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear from to assemble
/// @param[in] constants Constants that appear in `a`
/// @param[in] coefficients Coefficients that appear in `a`
/// @param[in] bcs Boundary conditions to apply. For boundary condition
///  dofs the row and column are zeroed. The diagonal  entry is not set.
template <typename T>
void assemble_matrix(
    auto mat_add, const Form<T>& a, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients,
    const std::vector<std::shared_ptr<const DirichletBC<T>>>& bcs)
{
  // Index maps for dof ranges
  auto map0 = a.function_spaces().at(0)->dofmap()->index_map;
  auto map1 = a.function_spaces().at(1)->dofmap()->index_map;
  auto bs0 = a.function_spaces().at(0)->dofmap()->index_map_bs();
  auto bs1 = a.function_spaces().at(1)->dofmap()->index_map_bs();

  // Build dof markers
  std::vector<std::int8_t> dof_marker0, dof_marker1;
  assert(map0);
  std::int32_t dim0 = bs0 * (map0->size_local() + map0->num_ghosts());
  assert(map1);
  std::int32_t dim1 = bs1 * (map1->size_local() + map1->num_ghosts());
  for (std::size_t k = 0; k < bcs.size(); ++k)
  {
    assert(bcs[k]);
    assert(bcs[k]->function_space());
    if (a.function_spaces().at(0)->contains(*bcs[k]->function_space()))
    {
      dof_marker0.resize(dim0, false);
      bcs[k]->mark_dofs(dof_marker0);
    }

    if (a.function_spaces().at(1)->contains(*bcs[k]->function_space()))
    {
      dof_marker1.resize(dim1, false);
      bcs[k]->mark_dofs(dof_marker1);
    }
  }

  // Assemble
  impl::assemble_matrix(mat_add, a, constants, coefficients, dof_marker0,
                        dof_marker1);
}

/// Assemble bilinear form into a matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear from to assemble
/// @param[in] bcs Boundary conditions to apply. For boundary condition
///  dofs the row and column are zeroed. The diagonal  entry is not set.
template <typename T>
void assemble_matrix(
    auto mat_add, const Form<T>& a,
    const std::vector<std::shared_ptr<const DirichletBC<T>>>& bcs)
{
  // Prepare constants and coefficients
  const std::vector<T> constants = pack_constants(a);
  auto coefficients = allocate_coefficient_storage(a);
  pack_coefficients(a, coefficients);

  // Assemble
  assemble_matrix(mat_add, a, std::span(constants),
                  make_coefficients_span(coefficients), bcs);
}

/// @brief Assemble bilinear form into a matrix. Matrix must already be
/// initialised. Does not zero or finalise the matrix.
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear form to assemble
/// @param[in] constants Constants that appear in `a`
/// @param[in] coefficients Coefficients that appear in `a`
/// @param[in] dof_marker0 Boundary condition markers for the rows. If
/// bc[i] is true then rows i in A will be zeroed. The index i is a
/// local index.
/// @param[in] dof_marker1 Boundary condition markers for the columns.
/// If bc[i] is true then rows i in A will be zeroed. The index i is a
/// local index.
template <typename T>
void assemble_matrix(
    auto mat_add, const Form<T>& a, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients,
    std::span<const std::int8_t> dof_marker0,
    std::span<const std::int8_t> dof_marker1)

{
  impl::assemble_matrix(mat_add, a, constants, coefficients, dof_marker0,
                        dof_marker1);
}

/// @brief Assemble bilinear form into a matrix. Matrix must already be
/// initialised. Does not zero or finalise the matrix.
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear form to assemble
/// @param[in] dof_marker0 Boundary condition markers for the rows. If
/// bc[i] is true then rows i in A will be zeroed. The index i is a
/// local index.
/// @param[in] dof_marker1 Boundary condition markers for the columns.
/// If bc[i] is true then rows i in A will be zeroed. The index i is a
/// local index.
template <typename T>
void assemble_matrix(auto mat_add, const Form<T>& a,
                     std::span<const std::int8_t> dof_marker0,
                     std::span<const std::int8_t> dof_marker1)

{
  // Prepare constants and coefficients
  const std::vector<T> constants = pack_constants(a);
  auto coefficients = allocate_coefficient_storage(a);
  pack_coefficients(a, coefficients);

  // Assemble
  assemble_matrix(mat_add, a, std::span(constants),
                  make_coefficients_span(coefficients), dof_marker0,
                  dof_marker1);
}

/// @brief Sets a value to the diagonal of a matrix for specified rows.
///
/// This function is typically called after assembly. The assembly
/// function zeroes Dirichlet rows and columns. For block matrices, this
/// function should normally be called only on the diagonal blocks, i.e.
/// blocks for which the test and trial spaces are the same.
/// @param[in] set_fn The function for setting values to a matrix
/// @param[in] rows The row blocks, in local indices, for which to add a
/// value to the diagonal
/// @param[in] diagonal The value to add to the diagonal for the
/// specified rows
template <typename T>
void set_diagonal(auto set_fn, std::span<const std::int32_t> rows,
                  T diagonal = 1.0)
{
  for (std::size_t i = 0; i < rows.size(); ++i)
  {
    std::span diag_span(&diagonal, 1);
    set_fn(rows.subspan(i, 1), rows.subspan(i, 1), diag_span);
  }
}

/// @brief Sets a value to the diagonal of the matrix for rows with a
/// Dirichlet boundary conditions applied.
///
/// This function is typically called after assembly. The assembly
/// function zeroes Dirichlet rows and columns. This function adds the
/// value only to rows that are locally owned, and therefore does not
/// create a need for parallel communication. For block matrices, this
/// function should normally be called only on the diagonal blocks, i.e.
/// blocks for which the test and trial spaces are the same.
/// @param[in] set_fn The function for setting values to a matrix
/// @param[in] V The function space for the rows and columns of the
/// matrix. It is used to extract only the Dirichlet boundary conditions
/// that are define on V or subspaces of V.
/// @param[in] bcs The Dirichlet boundary conditions
/// @param[in] diagonal The value to add to the diagonal for rows with a
/// boundary condition applied
template <typename T>
void set_diagonal(auto set_fn, const fem::FunctionSpace& V,
                  const std::vector<std::shared_ptr<const DirichletBC<T>>>& bcs,
                  T diagonal = 1.0)
{
  for (const auto& bc : bcs)
  {
    assert(bc);
    if (V.contains(*bc->function_space()))
    {
      const auto [dofs, range] = bc->dof_indices();
      set_diagonal(set_fn, dofs.first(range), diagonal);
    }
  }
}

// -- Setting bcs ------------------------------------------------------------

// FIXME: Move these function elsewhere?

// FIXME: clarify x0
// FIXME: clarify what happens with ghosts

/// Set bc values in owned (local) part of the vector, multiplied by
/// 'scale'. The vectors b and x0 must have the same local size. The bcs
/// should be on (sub-)spaces of the form L that b represents.
template <typename T>
void set_bc(std::span<T> b,
            const std::vector<std::shared_ptr<const DirichletBC<T>>>& bcs,
            std::span<const T> x0, double scale = 1.0)
{
  if (b.size() > x0.size())
    throw std::runtime_error("Size mismatch between b and x0 vectors.");
  for (const auto& bc : bcs)
  {
    assert(bc);
    bc->set(b, x0, scale);
  }
}

/// Set bc values in owned (local) part of the vector, multiplied by
/// 'scale'. The bcs should be on (sub-)spaces of the form L that b
/// represents.
template <typename T>
void set_bc(std::span<T> b,
            const std::vector<std::shared_ptr<const DirichletBC<T>>>& bcs,
            double scale = 1.0)
{
  for (const auto& bc : bcs)
  {
    assert(bc);
    bc->set(b, scale);
  }
}

} // namespace dolfinx::fem
