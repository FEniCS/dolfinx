// Copyright (C) 2018-2026 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FunctionSpace.h"
#include "assemble_matrix_impl.h"
#include "assemble_scalar_impl.h"
#include "assemble_vector_impl.h"
#include "pack.h"
#include "traits.h"
#include <algorithm>
#include <array>
#include <basix/mdspan.hpp>
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/common/types.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/EntityMap.h>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

/// @file assembler.h
/// @brief Functions supporting assembly of a finite element fem::Form.

namespace dolfinx::fem
{
template <dolfinx::scalar T, std::floating_point U>
class DirichletBC;
template <dolfinx::scalar T, std::floating_point U>
class Form;
template <std::floating_point T>
class FunctionSpace;

// -- Helper functions -----------------------------------------------------

/// @brief Create a map of `std::span`s from a map of `std::vector`s
template <dolfinx::scalar T>
std::map<std::pair<IntegralType, int>, std::pair<std::span<const T>, int>>
make_coefficients_span(const std::map<std::pair<IntegralType, int>,
                                      std::pair<std::vector<T>, int>>& coeffs)
{
  using Key = typename std::remove_reference_t<decltype(coeffs)>::key_type;
  std::map<Key, std::pair<std::span<const T>, int>> c;
  std::ranges::transform(
      coeffs, std::inserter(c, c.end()),
      [](auto& e) -> typename decltype(c)::value_type
      { return {e.first, {e.second.first, e.second.second}}; });
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
template <dolfinx::scalar T, std::floating_point U>
T assemble_scalar(
    const Form<T, U>& M, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  using mdspanx3_t
      = md::mdspan<const U, md::extents<std::size_t, md::dynamic_extent, 3>>;

  std::shared_ptr<const mesh::Mesh<U>> mesh = M.mesh();
  assert(mesh);
  std::span x = mesh->geometry().x();

  // Accumulate contributions from each cell type
  const int num_cell_types = mesh->topology()->cell_types().size();
  T val = 0;
  for (int cell_type_idx = 0; cell_type_idx < num_cell_types; ++cell_type_idx)
  {
    // Geometry dofmap and data
    md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>> x_dofmap
        = mesh->geometry().dofmaps().at(cell_type_idx);
    val += impl::assemble_scalar(M, x_dofmap,
                                 mdspanx3_t(x.data(), x.size() / 3, 3),
                                 constants, coefficients, cell_type_idx);
  }
  return val;
}

/// @brief Assemble functional into scalar.
///
/// @note Caller is responsible for accumulation across processes.
///
/// @param[in] M The form (functional) to assemble.
/// @return The contribution to the form (functional) from the local
/// process.
template <dolfinx::scalar T, std::floating_point U>
T assemble_scalar(const Form<T, U>& M)
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
/// @param[in] L The linear forms to assemble into b.
/// @param[in] constants The constants that appear in `L`.
/// @param[in] coefficients The coefficients that appear in `L`.
// template <dolfinx::scalar T, std::floating_point U>
template <typename V, std::floating_point U,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void assemble_vector(
    V&& b, const Form<T, U>& L, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  impl::assemble_vector(b, L, constants, coefficients);
}

/// @brief Assemble linear form into a vector.
/// @param[in,out] b Vector to be assembled. It will not be zeroed
/// before assembly.
/// @param[in] L Linear forms to assemble into b.
// template <dolfinx::scalar T, std::floating_point U>
// void assemble_vector(std::span<T> b, const Form<T, U>& L)
template <typename V, std::floating_point U,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void assemble_vector(V&& b, const Form<T, U>& L)
{
  auto coefficients = allocate_coefficient_storage(L);
  pack_coefficients(L, coefficients);
  const std::vector<T> constants = pack_constants(L);
  assemble_vector(b, L, std::span(constants),
                  make_coefficients_span(coefficients));
}

namespace impl
{
/// @brief Mark the dofs of `V` (owned and ghost, unrolled) constrained
/// by the boundary conditions in `bcs` that are defined on `V` or a
/// subspace of it.
/// @return Dof markers, or an empty array if no boundary condition
/// applies.
template <dolfinx::scalar T, std::floating_point U>
std::vector<std::int8_t> bc_dof_markers(
    const FunctionSpace<U>& V,
    const std::vector<std::reference_wrapper<const DirichletBC<T, U>>>& bcs)
{
  std::vector<std::int8_t> markers;
  for (auto& bc : bcs)
  {
    assert(bc.get().function_space());
    if (V.contains(*bc.get().function_space()))
    {
      if (markers.empty())
      {
        std::shared_ptr<const DofMap> dofmap = V.dofmaps().front();
        std::shared_ptr<const common::IndexMap> map = dofmap->index_map;
        assert(map);
        markers.resize(dofmap->index_map_bs()
                           * (map->size_local() + map->num_ghosts()),
                       0);
      }
      bc.get().mark_dofs(markers);
    }
  }
  return markers;
}

/// @brief Mark the dofs of the test and trial spaces of `a` that the
/// boundary conditions in `bcs` constrain.
///
/// Markers depend only on the space, so a form whose test and trial
/// spaces are the same marks its dofs once and shares the array
/// between its rows and its columns. The two returned spans are then
/// the same array rather than equal copies.
///
/// @param[in] a Bilinear form whose spaces are marked.
/// @param[in] bcs Boundary conditions. Only those defined on a space
/// or a subspace of it mark that space.
/// @param[out] storage Backing storage for the returned spans, which
/// are views into it and are valid for as long as it is.
/// @return Markers on the test space, indexing the rows, and on the
/// trial space, indexing the columns.
template <dolfinx::scalar T, std::floating_point U>
std::array<std::span<const std::int8_t>, 2> bc_dof_markers_pair(
    const Form<T, U>& a,
    const std::vector<std::reference_wrapper<const DirichletBC<T, U>>>& bcs,
    std::array<std::vector<std::int8_t>, 2>& storage)
{
  storage[0] = bc_dof_markers(*a.function_spaces().at(0), bcs);
  std::span<const std::int8_t> marker0(storage[0]);
  if (a.function_spaces().at(0) == a.function_spaces().at(1))
    return {marker0, marker0};

  storage[1] = bc_dof_markers(*a.function_spaces().at(1), bcs);
  return {marker0, std::span<const std::int8_t>(storage[1])};
}

/// @brief Constrained dof markers and boundary condition values on the
/// trial space of each form in `a`, as used by apply_lifting().
/// @return Markers and values for each block `j`, both empty if `a[j]`
/// is null or `bcs1[j]` is empty.
template <dolfinx::scalar T, std::floating_point U>
std::pair<std::vector<std::vector<std::int8_t>>, std::vector<std::vector<T>>>
bc_lifting_data(
    const std::vector<std::optional<std::reference_wrapper<const Form<T, U>>>>&
        a,
    const std::vector<
        std::vector<std::reference_wrapper<const DirichletBC<T, U>>>>& bcs1)
{
  if (a.size() != bcs1.size())
  {
    throw std::invalid_argument(
        "Mismatch in size between a and bcs in assembler.");
  }

  std::vector<std::vector<std::int8_t>> markers(a.size());
  std::vector<std::vector<T>> values(a.size());
  for (std::size_t j = 0; j < a.size(); ++j)
  {
    if (a[j] and !bcs1[j].empty())
    {
      std::shared_ptr<const DofMap> dofmap
          = a[j]->get().function_spaces().at(1)->dofmaps().front();
      std::shared_ptr<const common::IndexMap> map1 = dofmap->index_map;
      assert(map1);
      const std::int32_t crange
          = dofmap->index_map_bs() * (map1->size_local() + map1->num_ghosts());
      markers[j].assign(crange, 0);
      values[j].assign(crange, 0);
      for (auto& bc : bcs1[j])
      {
        bc.get().mark_dofs(markers[j]);
        bc.get().set(values[j], std::nullopt, 1);
      }
    }
  }

  return {std::move(markers), std::move(values)};
}
} // namespace impl

/// @brief Modify the right-hand side vector to account for constraints
/// (Dirichlet boundary condition constraints), with the constrained
/// dofs and their values given as arrays.
///
/// Computes
/// \f[
///  b \leftarrow b - \alpha A_{j}^{(1)} (g_{j} - x_{j})
/// \f]
/// for each block `j`, as described in the apply_lifting() overload
/// that takes Dirichlet boundary conditions. That overload builds
/// `bc_markers1` and `bc_values1` from the boundary conditions and calls
/// this function.
///
/// @note Ghost contributions are not accumulated (not sent to owner).
/// Caller is responsible for reverse-scatter to update the ghosts.
///
/// @param[in,out] b The vector to modify inplace.
/// @param[in] a List of bilinear forms, where `a[j]` is the form that
/// generates the matrix \f$A_{j}\f$. All forms in `a` must share the
/// same test function space. The trial function spaces can differ.
/// @param[in] constants Constant data appearing in the forms `a`.
/// @param[in] coeffs Coefficient data appearing in the forms `a`.
/// @param[in] bc_markers1 Constrained dof markers on the trial space of
/// `a[j]`, owned and ghost (unrolled): `bc_markers1[j][i]` is non-zero
/// if dof `i` is constrained. An empty `bc_markers1[j]` means block `j`
/// has no constraints and is skipped.
/// @param[in] bc_values1 Boundary condition values \f$g_{j}\f$, with
/// `bc_values1[j][i]` the value for dof `i`. Read only where
/// `bc_markers1[j][i]` is non-zero. Must have the same length as
/// `bc_markers1[j]`.
/// @param[in] x0 The vectors \f$x_{j}\f$. If empty, \f$x_{j}\f$ is
/// treated as zero. Otherwise must have the same length as `a`.
/// @param[in] alpha Scalar used in the modification of `b`.
template <typename V,
          std::floating_point U
          = scalar_value_t<typename std::remove_cvref_t<V>::value_type>,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void apply_lifting(
    V&& b,
    const std::vector<std::optional<std::reference_wrapper<const Form<T, U>>>>&
        a,
    const std::vector<std::span<const T>>& constants,
    const std::vector<std::map<std::pair<IntegralType, int>,
                               std::pair<std::span<const T>, int>>>& coeffs,
    const std::vector<std::span<const std::int8_t>>& bc_markers1,
    const std::vector<std::span<const T>>& bc_values1,
    const std::vector<std::span<const T>>& x0, T alpha)
{
  // If all forms are null, there is nothing to do
  if (std::ranges::all_of(a, [](auto ai) { return !ai; }))
    return;

  common::Timer t("[Apply lifting]");

  if (!x0.empty() and x0.size() != a.size())
  {
    throw std::invalid_argument(
        "Mismatch in size between x0 and bilinear form in assembler.");
  }

  if (bc_markers1.size() != a.size() or bc_values1.size() != a.size())
  {
    throw std::invalid_argument(
        "Mismatch in size between a and bc markers/values in assembler.");
  }

  for (std::size_t j = 0; j < a.size(); ++j)
  {
    if (!a[j] or bc_markers1[j].empty())
      continue;

    assert(a[j]->get().function_spaces().at(0));
    auto V1 = a[j]->get().function_spaces()[1];
    assert(V1);

    std::shared_ptr<const DofMap> dofmap = V1->dofmaps().front();
    auto map1 = dofmap->index_map;
    assert(map1);
    const std::size_t crange
        = dofmap->index_map_bs() * (map1->size_local() + map1->num_ghosts());
    if (bc_markers1[j].size() != crange or bc_values1[j].size() != crange)
    {
      throw std::invalid_argument(std::format(
          "bc markers/values for block {} have length {}/{}, expected {}.", j,
          bc_markers1[j].size(), bc_values1[j].size(), crange));
    }

    const int bs0 = a[j]->get().function_spaces()[0]->dofmaps().front()->bs();
    const int bs1 = dofmap->bs();

    std::span<const T> _x0;
    if (!x0.empty())
      _x0 = x0[j];

    impl::dispatch_bs(bs0, bs1,
                      [&b, &a, j, &constants, &coeffs, &bc_values1,
                       &bc_markers1, &_x0, alpha](auto bs0, auto bs1)
                      {
                        impl::lift_bc(b, a[j]->get(), bs0, bs1, constants[j],
                                      coeffs[j], bc_values1[j], bc_markers1[j],
                                      _x0, alpha);
                      });
  }
}

/// @brief Modify the right-hand side vector to account for constraints
/// (Dirichlet boundary condition constraints). This modification is
/// known as 'lifting'.
///
/// Consider the discrete algebraic system
/// \f[
/// \begin{bmatrix}
/// A_{0} & A_{1}
/// \end{bmatrix}
/// \begin{bmatrix}
/// u_{0} \\ u_{1}
/// \end{bmatrix}
/// = b,
/// \f]
/// where \f$A_{i}\f$ is a matrix. Partitioning each vector \f$u_{i}\f$
/// into 'unknown' (\f$u_{i}^{(0)}\f$) and prescribed
/// (\f$u_{i}^{(1)}\f$) groups,
/// \f[
/// \begin{bmatrix}
/// A_{0}^{(0)} & A_{0}^{(1)} & A_{1}^{(0)} & A_{1}^{(1)}
/// \end{bmatrix}
/// \begin{bmatrix}
/// u_{0}^{(0)} \\ u_{0}^{(1)} \\ u_{1}^{(0)} \\ u_{1}^{(1)}
/// \end{bmatrix}
/// = b.
/// \f]
/// If \f$u_{i}^{(1)} = \alpha(g_{i} - x_{i})\f$, where \f$g_{i}\f$ is
/// the Dirichlet boundary condition value, \f$x_{i}\f$ is provided and
/// \f$\alpha\f$ is a constant, then
/// \f[
/// \begin{bmatrix}
/// A_{0}^{(0)} & A_{0}^{(1)} & A_{1}^{(0)} & A_{1}^{(1)}
/// \end{bmatrix}
/// \begin{bmatrix}
/// u_{0}^{(0)} \\ \alpha(g_{0} - x_{0}) \\ u_{1}^{(0)} \\ \alpha(g_{1} - x_{1})
/// \end{bmatrix}
/// = b.
/// \f]
/// Rearranging,
/// \f[
/// \begin{bmatrix}
/// A_{0}^{(0)} & A_{1}^{(0)}
/// \end{bmatrix}
/// \begin{bmatrix}
/// u_{0}^{(0)} \\ u_{1}^{(0)}
/// \end{bmatrix}
/// = b - \alpha A_{0}^{(1)} (g_{0} - x_{0}) - \alpha A_{1}^{(1)} (g_{1} -
/// x_{1}).
/// \f]
///
/// The modified \f$b\f$ vector is
/// \f[
///  b \leftarrow b - \alpha A_{0}^{(1)} (g_{0} - x_{0}) - \alpha A_{1}^{(1)}
///  (g_{1} - x_{1})
/// \f]
/// More generally,
/// \f[
///  b \leftarrow b - \alpha A_{i}^{(1)} (g_{i} - x_{i}).
/// \f]
///
/// @note Ghost contributions are not accumulated (not sent to owner).
/// Caller is responsible for reverse-scatter to update the ghosts.
///
/// @note Boundary condition values are *not* set in `b` by this
/// function. Use DirichletBC::set to set values in `b`.
///
/// @note Convenience overload for callers that have boundary
/// conditions. It rebuilds the constrained dof markers and values on
/// every call, and should not be called internally by the library;
/// call the overload taking `bc_markers1` and `bc_values1` instead.
///
/// @param[in,out] b The vector to modify inplace.
/// @param[in] a List of bilinear forms, where `a[i]` is the form that
/// generates the matrix \f$A_{i}\f$. All forms in `a` must share the
/// same test function space. The trial function spaces can differ.
/// @param[in] constants Constant data appearing in the forms `a`.
/// @param[in] coeffs Coefficient data appearing in the forms `a`.
/// @param[in] x0 The vector \f$x_{i}\f$ above. If empty it is set to
/// zero.
/// @param[in] bcs1 Boundary conditions that provide the \f$g_{i}\f$
/// values. `bcs1[i]` is the list of boundary conditions on \f$u_{i}\f$.
/// @param[in] alpha Scalar used in the modification of `b`.
template <typename V,
          std::floating_point U
          = scalar_value_t<typename std::remove_cvref_t<V>::value_type>,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void apply_lifting(
    V&& b,
    const std::vector<std::optional<std::reference_wrapper<const Form<T, U>>>>&
        a,
    const std::vector<std::span<const T>>& constants,
    const std::vector<std::map<std::pair<IntegralType, int>,
                               std::pair<std::span<const T>, int>>>& coeffs,
    const std::vector<
        std::vector<std::reference_wrapper<const DirichletBC<T, U>>>>& bcs1,
    const std::vector<std::span<const T>>& x0, T alpha)
{
  auto [bc_markers1, bc_values1] = impl::bc_lifting_data(a, bcs1);
  apply_lifting(
      b, a, constants, coeffs,
      std::vector<std::span<const std::int8_t>>(bc_markers1.begin(),
                                                bc_markers1.end()),
      std::vector<std::span<const T>>(bc_values1.begin(), bc_values1.end()), x0,
      alpha);
}

/// @brief Modify the right-hand side vector to account for constraints
/// (Dirichlet boundary conditions constraints). This modification is
/// known as 'lifting'.
///
/// See apply_lifting() for a detailed explanation of the lifting. The
/// difference between this function and apply_lifting() is that
/// apply_lifting() requires packed form constant and coefficient data
/// to be passed to the function, whereas this function packs the
/// constant and coefficient form data and then calls apply_lifting().
///
/// @note Ghost contributions are not accumulated (not sent to owner).
/// Caller is responsible for reverse-scatter to update the ghosts.
///
/// @note Boundary condition values are *not* set in `b` by this
/// function. Use DirichletBC::set to set values in `b`.
///
/// @note Convenience overload for callers that have boundary
/// conditions. It rebuilds the constrained dof markers and values on
/// every call, and should not be called internally by the library;
/// call the overload taking `bc_markers1` and `bc_values1` instead.
///
/// @param[in,out] b The vector to modify inplace.
/// @param[in] a List of bilinear forms, where `a[i]` is the form that
/// generates the matrix \f$A_{i}\f$ (see apply_lifting()). All forms in
/// `a` must share the same test function space. The trial function
/// spaces can differ.
/// @param[in] x0 The vector \f$x_{i}\f$ described in apply_lifting().
/// If empty it is set to zero.
/// @param[in] bcs1 Boundary conditions that provide the \f$g_{i}\f$
/// values described in apply_lifting(). `bcs1[i]` is the list of
/// boundary conditions on \f$u_{i}\f$.
/// @param[in] alpha Scalar used in the modification of `b` (see
/// described in apply_lifting()).
template <typename V,
          std::floating_point U
          = scalar_value_t<typename std::remove_cvref_t<V>::value_type>,
          dolfinx::scalar T = typename std::remove_cvref_t<V>::value_type>
  requires std::is_same_v<typename std::remove_cvref_t<V>::value_type, T>
void apply_lifting(
    V&& b,
    const std::vector<std::optional<std::reference_wrapper<const Form<T, U>>>>&
        a,
    const std::vector<
        std::vector<std::reference_wrapper<const DirichletBC<T, U>>>>& bcs1,
    const std::vector<std::span<const T>>& x0, T alpha)
{
  std::vector<
      std::map<std::pair<IntegralType, int>, std::pair<std::vector<T>, int>>>
      coeffs;
  std::vector<std::vector<T>> constants;
  for (const auto& _a : a)
  {
    if (_a)
    {
      auto coefficients = allocate_coefficient_storage(_a->get());
      pack_coefficients(_a->get(), coefficients);
      coeffs.push_back(coefficients);
      constants.push_back(pack_constants(_a->get()));
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
  std::ranges::transform(coeffs, std::back_inserter(_coeffs),
                         [](auto& c) { return make_coefficients_span(c); });

  auto [bc_markers1, bc_values1] = impl::bc_lifting_data(a, bcs1);
  apply_lifting(
      b, a, _constants, _coeffs,
      std::vector<std::span<const std::int8_t>>(bc_markers1.begin(),
                                                bc_markers1.end()),
      std::vector<std::span<const T>>(bc_values1.begin(), bc_values1.end()), x0,
      alpha);
}

// -- Matrices ---------------------------------------------------------------

/// @brief Assemble bilinear form into a matrix. Matrix must already be
/// initialised. Does not zero or finalise the matrix.
/// @note This function can be used to insert kernels into different objects by
/// replacing the mat_add function appropriately.
/// @tparam T scalar type
/// @tparam U geometry scalar type
/// @param[in] mat_add The function for adding values into the matrix.
/// @param[in] a The bilinear form to assemble.
/// @param[in] constants Constants that appear in `a`.
/// @param[in] coefficients Coefficients that appear in `a`.
/// @param[in] dof_marker0 Boundary condition markers for the rows. If
/// bc[i] is true then rows i in A will be zeroed. The index i is a
/// local index.
/// @param[in] dof_marker1 Boundary condition markers for the columns.
/// If bc[i] is true then rows i in A will be zeroed. The index i is a
/// local index.
template <dolfinx::scalar T, std::floating_point U>
void assemble_matrix(
    la::MatSet<T> auto mat_add, const Form<T, U>& a,
    std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients,
    std::span<const std::int8_t> dof_marker0,
    std::span<const std::int8_t> dof_marker1)

{
  common::Timer t_assm("[Assemble Matrix]");
  using mdspanx3_t
      = md::mdspan<const U, md::extents<std::size_t, md::dynamic_extent, 3>>;

  std::shared_ptr<const mesh::Mesh<U>> mesh = a.mesh();
  assert(mesh);
  std::span x = mesh->geometry().x();
  impl::assemble_matrix<false>(mat_add, a,
                               mdspanx3_t(x.data(), x.size() / 3, 3), constants,
                               coefficients, dof_marker0, dof_marker1);
}

/// @brief Assemble bilinear form into a matrix.
/// @note Convenience overload for callers that have boundary
/// conditions. It rebuilds the dof markers on every call, and should
/// not be called internally by the library; call the overload taking
/// `dof_marker0` and `dof_marker1` instead.
///
/// @param[in] mat_add The function for adding values into the matrix.
/// @param[in] a The bilinear from to assemble.
/// @param[in] bcs Boundary conditions to apply. For boundary condition
/// dofs the row and column are zeroed. The diagonal  entry is not set.
template <dolfinx::scalar T, std::floating_point U>
void assemble_matrix(
    auto mat_add, const Form<T, U>& a,
    const std::vector<std::reference_wrapper<const DirichletBC<T, U>>>& bcs)
{
  // Prepare constants and coefficients
  const std::vector<T> constants = pack_constants(a);
  auto coefficients = allocate_coefficient_storage(a);
  pack_coefficients(a, coefficients);

  std::array<std::vector<std::int8_t>, 2> markers;
  auto [dof_marker0, dof_marker1] = impl::bc_dof_markers_pair(a, bcs, markers);

  // Assemble
  assemble_matrix(mat_add, a, std::span<const T>(constants),
                  make_coefficients_span(coefficients), dof_marker0,
                  dof_marker1);
}

/// @brief Assemble bilinear form into a matrix. Matrix must already be
/// initialised. Does not zero or finalise the matrix.
///
/// @param[in] mat_add The function for adding values into the matrix.
/// @param[in] a The bilinear form to assemble.
/// @param[in] dof_marker0 Boundary condition markers for the rows. If
/// `bc[i]` is `true` then rows `i` in A` `will be zeroed. The index `i`
/// is a local index.
/// @param[in] dof_marker1 Boundary condition markers for the columns.
/// If `bc[i]` is `true` then rows `i` in `A` will be zeroed. The index
/// `i` is a local index.
template <dolfinx::scalar T, std::floating_point U>
void assemble_matrix(auto mat_add, const Form<T, U>& a,
                     std::span<const std::int8_t> dof_marker0,
                     std::span<const std::int8_t> dof_marker1)

{
  // Prepare constants and coefficients
  const std::vector<T> constants = pack_constants(a);
  auto coefficients = allocate_coefficient_storage(a);
  pack_coefficients(a, coefficients);

  // Assemble
  assemble_matrix(mat_add, a, std::span<const T>(constants),
                  make_coefficients_span(coefficients), dof_marker0,
                  dof_marker1);
}

/// @brief Set a value on the diagonal of the locally owned rows that a
/// Dirichlet boundary condition constrains.
///
/// Set only locally owned rows to prevent accumulation when finalising
/// `A`. A constrained degree-of-freedom that is a ghost on the calling
/// rank is left untouched here and is set by the rank that owns it, so
/// no communication is needed from this function.
///
/// This function is typically called after assembly, which zeroes
/// Dirichlet rows and columns. For block matrices, it should normally
/// be called only on the diagonal blocks, i.e. blocks for which the
/// test and trial spaces are the same.
///
/// @note Convenience overload for callers holding `V` and `bcs` rather
/// than the row list, which it rebuilds on every call. Library code
/// should cache the rows across repeated calls and set them with
/// la::set_diagonal.
///
/// @note Each row is set exactly once, even where several boundary
/// conditions constrain the same degree-of-freedom, so `set_fn` may
/// add rather than insert. Every condition sets the same `diagonal`
/// value, so their order in `bcs` does not matter here.
///
/// @param[in] set_fn The function for setting values to a matrix.
/// @param[in] V The function space for the rows and columns of the
/// matrix. It is used to extract only the Dirichlet boundary conditions
/// that are define on V or subspaces of V.
/// @param[in] bcs The Dirichlet boundary conditions. Only conditions
/// defined on `V` or a subspace of it contribute, and of those only
/// their locally owned dofs.
/// @param[in] diagonal Value to set on the diagonal of each owned
/// constrained row.
template <dolfinx::scalar T, std::floating_point U>
void set_diagonal(
    auto set_fn, const FunctionSpace<U>& V,
    const std::vector<std::reference_wrapper<const DirichletBC<T, U>>>& bcs,
    T diagonal = T(1))
{
  spdlog::debug("Set diagonal");

  std::vector<std::int32_t> rows;
  for (auto& bc : bcs)
  {
    if (V.contains(*bc.get().function_space()))
    {
      const auto [dofs, range] = bc.get().dof_indices();
      std::span<const std::int32_t> owned = dofs.first(range);
      rows.insert(rows.end(), owned.begin(), owned.end());
    }
  }

  // A condition's dofs are strictly increasing (a DirichletBC
  // precondition), so one condition needs no sort. Several give sorted
  // runs, which a comparison sort handles poorly and which are often
  // already in order, hence the check before radix sorting.
  if (!std::ranges::is_sorted(rows))
    dolfinx::radix_sort(rows);

  // Overlapping conditions can repeat a row
  rows.erase(std::ranges::unique(rows).begin(), rows.end());
  la::set_diagonal(set_fn, rows, diagonal);
}

} // namespace dolfinx::fem
