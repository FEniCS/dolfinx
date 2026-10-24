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
#include <dolfinx/mesh/EntityMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <format>
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
namespace impl
{
/// @brief Check that an Expression can be evaluated over entities of a
/// mesh.
///
/// @param[in] e Expression to evaluate.
/// @param[in] mesh Mesh that the Expression is evaluated on.
/// @param[in] rank Rank of the entities array: 1 for cell indices, 2
/// for (cell, local entity index) pairs.
/// @throws std::invalid_argument if `mesh` has more than one cell type,
/// `e` was created on another mesh, its entity dimension does not fit
/// `mesh` or `rank`, or its argument is on another mesh whose cells
/// are not the entities evaluated over.
template <dolfinx::scalar T, std::floating_point U>
void check_expression_entities(const fem::Expression<T, U>& e,
                               const mesh::Mesh<U>& mesh, std::size_t rank)
{
  // An Expression is compiled for one cell type, and cell indices are
  // local to a cell type, so neither cells nor (cell, local index)
  // pairs identify entities of a mesh with several cell types
  if (mesh.topology()->cell_types().size() > 1)
  {
    throw std::invalid_argument("Evaluating an Expression on a mesh with "
                                "multiple cell types is not supported.");
  }

  if (e.coordinate_element_hash() != mesh.geometry().cmaps().front().hash())
  {
    throw std::invalid_argument(
        "Expression was created on a different mesh. Cannot tabulate.");
  }

  const int tdim = mesh.topology()->dim();
  if (e.entity_dim() > tdim)
  {
    throw std::invalid_argument(
        std::format("Cannot evaluate an Expression on entities of dimension "
                    "{} on a mesh of dimension {}.",
                    e.entity_dim(), tdim));
  }

  if (rank == 1 and e.entity_dim() != tdim)
  {
    throw std::invalid_argument(
        "An Expression on facets, ridges or peaks must be evaluated on "
        "(cell, local entity index) pairs.");
  }

  // An argument on another mesh must live on the evaluated entities
  if (std::shared_ptr<const FunctionSpace<U>> V = e.argument_space();
      V and V->mesh()->topology() != mesh.topology())
  {
    mesh::check_entity_cell_mapping(*V->mesh()->topology(), *mesh.topology(),
                                    e.entity_dim());
  }
}
} // namespace impl

/// @brief Pack the coefficients of an Expression over entities of a
/// mesh.
///
/// @tparam T Scalar type.
/// @tparam U Geometry type.
/// @param[in] e Expression to pack coefficients for.
/// @param[in] mesh Mesh that the Expression is evaluated on.
/// @param[in] entities Entities to pack over: either a rank-1 list of
/// cell indices, or a rank-2 list of (cell, local entity index) pairs.
/// @return Packed coefficients, shape `(entities.extent(0), cstride)`
/// and row-major, and `cstride`.
/// @throws std::invalid_argument if `e` cannot be evaluated over
/// `entities` of `mesh`.
template <dolfinx::scalar T, std::floating_point U>
std::pair<std::vector<T>, int> pack_coefficients(const fem::Expression<T, U>& e,
                                                 const mesh::Mesh<U>& mesh,
                                                 fem::MDSpan2 auto entities)
{
  // Validate before packing, so that an invalid request is rejected
  // before any coefficient is packed
  impl::check_expression_entities(e, mesh, entities.rank());

  std::vector<int> coffsets = e.coefficient_offsets();
  const int cstride = coffsets.back();
  std::vector<T> coeffs(entities.extent(0) * cstride);
  std::vector<std::reference_wrapper<const Function<T, U>>> c;
  std::ranges::transform(e.coefficients(), std::back_inserter(c),
                         [](const std::shared_ptr<const Function<T, U>>& f)
                             -> const Function<T, U>& { return *f; });
  fem::pack_coefficients(c, *mesh.topology(), entities, e.entity_dim(),
                         e.entity_maps(), coffsets, std::span(coeffs));
  return {std::move(coeffs), cstride};
}

/// @brief Evaluate an Expression on cells, facets, ridges or peaks.
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
/// For cells it is a list of cell indices. For facets, ridges and peaks
/// it is a list of (cell index, local entity index) index pairs, i.e.
/// `entities=[cell0, entity_local0, cell1, entity_local1, ...]`.
/// @param[in] mesh Mesh that the Expression is evaluated on.
/// @param[in] element Argument element and argument space dimension.
/// @note An argument on a mesh other than `mesh` is mapped to its own
/// cells through the Expression's entity maps, and its dof
/// transformations use that mesh's cell orientation data.
/// @pre If the argument is on a mesh other than `mesh` and needs dof
/// transformations, each of `entities` has a cell in that mesh.
/// @throws std::invalid_argument if `e` cannot be evaluated over
/// `entities` of `mesh`.
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
  impl::check_expression_entities(e, mesh, entities.rank());

  std::shared_ptr<const FunctionSpace<U>> V = e.argument_space();

  // Topology providing the argument's cell orientation data and, when
  // the argument is on another mesh, its cell for each of `entities`.
  // `topology_v` and `argument_cells` back the spans passed to
  // impl::tabulate_expression.
  std::shared_ptr<mesh::Topology> topology_v = mesh.topology_mutable();
  std::vector<std::int32_t> argument_cells;
  std::span<const std::uint32_t> cell_info;
  if (element and element->first.get().needs_dof_transformations())
  {
    if (V and V->mesh()->topology() != mesh.topology())
      topology_v = V->mesh()->topology_mutable();

    topology_v->create_cell_permutations();
    cell_info = topology_v->get_cell_permutation_info();

    if (topology_v != mesh.topology_mutable())
    {
      const mesh::Topology& topology = *mesh.topology();
      const mesh::EntityMap& emap
          = mesh::find_entity_map(e.entity_maps(), topology, *topology_v);
      argument_cells = mesh::extract_cells_from_entities(
          *topology_v, topology, entities, std::cref(emap));
#ifndef NDEBUG
      if (std::ranges::find(argument_cells, -1) != argument_cells.end())
      {
        throw std::invalid_argument(
            "An entity has no cell in the mesh of the argument.");
      }
#endif
    }
  }

  impl::tabulate_expression(values, e.kernel(), e.X().second, e.value_size(),
                            coeffs, constants, mesh, entities, element,
                            cell_info, argument_cells);
}

/// @brief Evaluate an Expression on cells, facets, ridges or peaks.
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
/// of cells. For expressions executed on facets, ridges or peaks rank
/// is 2, and shape is `(num_entities, 2)`, where `entities[i, 0]` is the
/// cell index and `entities[i, 1]` is the local index of the entity
/// relative to the cell.
template <dolfinx::scalar T, std::floating_point U>
void tabulate_expression(std::span<T> values, const fem::Expression<T, U>& e,
                         const mesh::Mesh<U>& mesh, fem::MDSpan2 auto entities)
{
  // Packing validates `e` against `mesh` and `entities` first
  auto [coeffs, cstride] = fem::pack_coefficients(e, mesh, entities);
  std::vector<T> constants = fem::pack_constants(e);

  std::optional<
      std::pair<std::reference_wrapper<const FiniteElement<U>>, std::size_t>>
      element = std::nullopt;
  if (std::shared_ptr<const FunctionSpace<U>> V = e.argument_space(); V)
  {
    std::size_t num_argument_dofs
        = V->dofmap()->element_dof_layout().num_dofs() * V->dofmap()->bs();
    assert(V->element());
    element = {std::cref(*V->element()), num_argument_dofs};
  }

  tabulate_expression(
      values, e,
      md::mdspan(std::as_const(coeffs).data(), entities.extent(0), cstride),
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
/// @note If `u1` is in a symmetric rank-2 tensor space, `e0` has the
/// value size of the full `(d, d)` tensor. The input should be symmetric.
/// Symmetry is not checked. The diagonal is taken from the diagonal of
/// the input; the choice of off-diagonal components retained from
/// non-symmetric data is unspecified.
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
    const auto& [X0, shape0] = e0.X();
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
  std::size_t num_points = e0.num_points();
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
