// Copyright (C) 2018-2026 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Expression.h"
#include "FiniteElement.h"
#include "Function.h"
#include "FunctionSpace.h"
#include "assemble_expression_impl.h"
#include "interpolate.h"
#include "pack.h"
#include "traits.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

/// @file expression_evaluate.h
/// @brief Evaluation of a fem::Expression, and interpolation of an
/// Expression into a fem::Function.
///
/// These operations sit above fem::Function because evaluating an
/// Expression requires its coefficient data to be packed, which in turn
/// requires a complete fem::Function.

namespace dolfinx::fem
{
/// @brief Evaluate an Expression on cells or facets.
///
/// This function accepts packed coefficient data, which allows it be
/// called without re-packing all coefficient data at each evaluation.
///
/// @tparam T Scalar type.
/// @tparam U Geometry type
/// @param[in,out] values Array to fill with computed values. Shape is
/// `(num_entities, num_points, value_size, num_argument_dofs)` and
/// storage is row-major.
/// @param[in] e Expression to evaluate.
/// @param[in] coeffs Packed coefficients for the Expressions. Typically
/// computed using fem::pack_coefficients.
/// @param[in] constants Packed constant data. Typically computed using
/// fem::pack_constants.
/// @param[in] entities Mesh entities to evaluate the expression over.
/// For cells it is a list of cell indices. For facets is is a list of
/// (cell index, local facet index) index pairs, i.e. `entities=[cell0,
/// facet_local0, cell1, facet_local1, ...]`.
/// @param[in] mesh Mesh that the Expression is evaluated on.
/// @param[in] element Argument element and argument space dimension.
template <dolfinx::scalar T, std::floating_point U>
void tabulate_expression(
    std::span<T> values, const fem::Expression<T, U>& e,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    std::span<const T> constants, const mesh::Mesh<U>& mesh,
    fem::MDSpan2 auto entities,
    std::optional<
        std::pair<std::reference_wrapper<const FiniteElement<U>>, std::size_t>>
        element)
{
  // Check that domain is the same as mesh of the expression
  if (e.coordinate_element_hash() != mesh.geometry().cmaps().front().hash())
  {
    throw std::invalid_argument(
        "Expression was created on a different mesh. Cannot tabulate.");
  }
  auto [X, Xshape] = e.X();
  impl::tabulate_expression(values, e.kernel(), Xshape, e.value_size(), coeffs,
                            constants, mesh, entities, element);
}

/// @brief Evaluate an Expression on cells or facets.
///
/// @tparam T Scalar type.
/// @tparam U Geometry type
/// @param[in,out] values Array to fill with computed values. Row major
/// storage. Sizing should be `(num_cells, num_points * value_size *
/// num_all_argument_dofs columns)`. facet index) tuples. Array is
/// flattened per entity.
/// @param[in] e Expression to evaluate.
/// @param[in] mesh Mesh to compute `e` on.
/// @param[in] entities Mesh entities to evaluate the expression over.
/// For expressions executed on cells, rank is 1 and size is the number
/// of cells. For expressions executed on facets rank is 2, and shape is
/// `(num_facets, 2)`, where `entities[i, 0]` is the cell index and
/// `entities[i, 1]` is the local index of the facet relative to the
/// cell.
template <dolfinx::scalar T, std::floating_point U>
void tabulate_expression(std::span<T> values, const fem::Expression<T, U>& e,
                         const mesh::Mesh<U>& mesh, fem::MDSpan2 auto entities)
{
  // Check that domain is the same as mesh of the expression
  if (e.coordinate_element_hash() != mesh.geometry().cmaps().front().hash())
  {
    throw std::invalid_argument(
        "Expression was created on a different mesh. Cannot tabulate.");
  }

  std::optional<
      std::pair<std::reference_wrapper<const FiniteElement<U>>, std::size_t>>
      element = std::nullopt;
  if (auto V = e.argument_space(); V)
  {
    std::size_t num_argument_dofs
        = V->dofmap()->element_dof_layout().num_dofs() * V->dofmap()->bs();
    assert(V->element());
    element = {std::cref(*V->element()), num_argument_dofs};
  }

  std::vector<int> coffsets = e.coefficient_offsets();
  const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients
      = e.coefficients();
  std::vector<T> coeffs(entities.extent(0) * coffsets.back());
  int cstride = coffsets.back();
  {
    std::vector<std::reference_wrapper<const Function<T, U>>> c;
    std::ranges::transform(coefficients, std::back_inserter(c),
                           [](auto c) -> const Function<T, U>& { return *c; });
    fem::pack_coefficients(c, mesh, entities, e.entity_maps(), coffsets,
                           std::span(coeffs));
  }
  std::vector<T> constants = fem::pack_constants(e);

  tabulate_expression<T, U>(
      values, e, md::mdspan(coeffs.data(), entities.extent(0), cstride),
      std::span<const T>(constants), mesh, entities, element);
}

/// @brief Interpolate an Expression into a Function over a subset of
/// cells.
///
/// @param[out] u1 Function to interpolate into. The Expression must
/// have been created using the reference coordinates returned by
/// FiniteElement::interpolation_points for the element of `u1`.
/// @param[in] cells1 Cell indices associated with the mesh of `u1` that
/// will be interpolated onto.
/// @param[in] e0 Expression to be interpolated from.
/// @param[in] cells0 Cell indices associated with the mesh of `e0` that
/// will be interpolated from, used if `e0` has Function coefficients.
/// If no mesh can be associated with `e0` then the mesh of `u1` is
/// used. If `cells1[i]` is the index of a cell in the mesh associated
/// with `u1`, then `cells0[i]` is the index of the *same* cell but in
/// the mesh associated with `e0`.
///
/// @pre `cells0` and `cells1` have the same size.
template <dolfinx::scalar T, std::floating_point U>
void interpolate(Function<T, U>& u1, mesh::CellRange auto&& cells1,
                 const Expression<T, U>& e0, mesh::CellRange auto&& cells0)
{
  // Extract mesh
  const mesh::Mesh<U>* mesh0 = nullptr;
  for (auto& c : e0.coefficients())
  {
    assert(c);
    assert(c->function_space());
    assert(c->function_space()->mesh());
    if (auto mesh = c->function_space()->mesh().get(); !mesh0)
      mesh0 = mesh;
    else if (mesh != mesh0)
    {
      throw std::invalid_argument(
          "Expression coefficient Functions have different meshes.");
    }
  }

  // If Expression has no Function coefficients take mesh from `u1`.
  auto V1 = u1.function_space();
  assert(V1);
  assert(V1->mesh());
  if (!mesh0)
    mesh0 = V1->mesh().get();

  if (cells0.size() != cells1.size())
    throw std::invalid_argument("Cell lists have different lengths.");

  // Check that Function and Expression spaces are compatible
  assert(V1->element());
  std::size_t value_size = e0.value_size();
  if (e0.argument_space())
    throw std::invalid_argument("Cannot interpolate Expression with Argument.");

  if (value_size != (std::size_t)V1->element()->value_size())
  {
    throw std::invalid_argument(
        "Function value size not equal to Expression value size.");
  }

  // Compatibility check
  {
    auto [X0, shape0] = e0.X();
    auto [X1, shape1] = V1->element()->interpolation_points();
    if (shape0 != shape1)
    {
      throw std::invalid_argument(
          "Function element interpolation points has different shape to "
          "Expression interpolation points");
    }

    for (std::size_t i = 0; i < X0.size(); ++i)
    {
      if (std::abs(X0[i] - X1[i]) > 1.0e-10)
      {
        throw std::invalid_argument("Function element interpolation points not "
                                    "equal to Expression interpolation points");
      }
    }
  }

  // Array to hold evaluated Expression
  std::size_t num_cells = cells0.size();
  std::size_t num_points = e0.X().second[0];
  std::vector<T> fdata(num_cells * num_points * value_size);
  md::mdspan<const T, md::dextents<std::size_t, 3>> f(fdata.data(), num_cells,
                                                      num_points, value_size);

  // Evaluate Expression at points
  std::vector<std::int32_t> _cells0(cells0.begin(), cells0.end());
  tabulate_expression(std::span(fdata), e0, *mesh0,
                      md::mdspan(_cells0.data(), _cells0.size()));

  // Reshape evaluated data to fit interpolate.
  // Expression returns matrix of shape (num_cells, num_points *
  // value_size), i.e. xyzxyz ordering of dof values per cell per
  // point. The interpolation uses xxyyzz input, ordered for all
  // points of each cell, i.e. (value_size, num_cells*num_points).
  std::vector<T> fdata1(num_cells * num_points * value_size);
  md::mdspan<T, md::dextents<std::size_t, 3>> f1(fdata1.data(), value_size,
                                                 num_cells, num_points);
  for (std::size_t i = 0; i < f.extent(0); ++i)
    for (std::size_t j = 0; j < f.extent(1); ++j)
      for (std::size_t k = 0; k < f.extent(2); ++k)
        f1(k, i, j) = f(i, j, k);

  // Interpolate values into appropriate space
  fem::interpolate<T>(u1, std::span<const T>(fdata1.data(), fdata1.size()),
                      {value_size, num_cells * num_points}, cells1);
}

/// @brief Interpolate an Expression into a Function over a subset of
/// cells.
///
/// @param[out] u1 Function to interpolate into. The Expression must
/// have been created using the reference coordinates returned by
/// FiniteElement::interpolation_points for the element of `u1`.
/// @param[in] e0 Expression to be interpolated from.
/// @param[in] cells Cell indices to interpolate on. If `e0` has
/// Function coefficients, the cells are interpreted relative to the
/// mesh of `e0` as well as the mesh of `u1`.
template <dolfinx::scalar T, std::floating_point U>
void interpolate(Function<T, U>& u1, const Expression<T, U>& e0,
                 mesh::CellRange auto&& cells)
{
  interpolate<T, U>(u1, cells, e0, cells);
}

/// @brief Interpolate an Expression into a Function on all cells.
///
/// @param[out] u1 Function to interpolate into. The Expression must
/// have been created using the reference coordinates returned by
/// FiniteElement::interpolation_points for the element of `u1`.
/// @param[in] e0 Expression to be interpolated from.
///
/// @pre If a mesh is associated with the Function coefficients of `e0`,
/// it must be the same as the mesh::Mesh associated with `u1`.
template <dolfinx::scalar T, std::floating_point U>
void interpolate(Function<T, U>& u1, const Expression<T, U>& e0)
{
  assert(u1.function_space());
  assert(u1.function_space()->mesh());
  int tdim = u1.function_space()->mesh()->topology()->dim();
  auto map = u1.function_space()->mesh()->topology()->index_map(tdim);
  assert(map);
  interpolate<T, U>(
      u1, e0, std::ranges::iota_view(0, map->size_local() + map->num_ghosts()));
}
} // namespace dolfinx::fem
