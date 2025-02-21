// Copyright (C) 2025 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Expression.h"
#include "FunctionSpace.h"
#include "traits.h"
#include "utils.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <memory>
#include <vector>

namespace dolfinx::fem::impl
{
/// @brief Tabulate an Expression at points.
///
/// Executes an Expression kernel over a list of mesh entities.
///
/// @tparam T Scalar type of expression.
/// @tparam U Geometry type.
/// @param[in,out] values Array in which the tabulated expressions are
/// set. It should have shape `(entities.extent(0), Xshape[0],
/// value_size, num_argument_dofs)`. Storage is row-major. Data is set
/// in `values` (not accumulated).
/// @param[in] fn Expression kernel to execute.
/// @param[in] Xshape Shape `(num_points, geometric dimensions)` of
/// points array at which the expression is evaluated by the kernel.
/// @param[in] value_size Value size of the evaluated expression at a
/// point, e.g. 1 for a scalar field, 3 for a vector field in 3D, 4 for
/// a second-order tensor in 2D.
/// @param[in] num_argument_dofs Dimension of an argument function.
/// Greater than 1 when computing a 1-form expression, e.g. can be used
/// to create a matrix that when applied to a degree-of-freedom vector
/// gives the expression values at the evaluation points.
/// @param[in] x_dofmap Geometry degree-of-freedom map.
/// @param[in] x Geometry coordinate of the mesh.
/// @param[in] coeffs Coefficient data that appears in the Expression.
/// Shape is `(num_cells, coeff data per cell)`. Usually packed using
/// fem::pack_coefficients.
/// @param[in] constants Constant (coefficient) data that appears in
/// expression. Usually packed using fem::pack_constants.
/// @param[in] entities Mesh entities to evaluate the expression over.
/// For expressions executed on cells, rank is 1 and size is the number
/// of cells. For expressions executed on facets rank is 2, and shape is
/// `(num_facets, 2)`, where `entities[i, 0]` is the cell index and
/// `entities[i, 1]` is the local index of the facet relative to the
/// cell.
/// @param[in] cell_info Cell orientation data for use in `P0`.
/// @param[in] P0 Degree-of-freedom transformation function. Applied when
/// expressions includes an argument function that requires a
/// transformation.
template <dolfinx::scalar T, std::floating_point U>
void tabulate_expression(
    std::span<T> values, fem::FEkernel<T> auto fn,
    std::array<std::size_t, 2> Xshape, std::size_t value_size,
    std::size_t num_argument_dofs,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x_dofmap,
    std::span<const scalar_value_type_t<T>> x,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        coeffs,
    std::span<const T> constants, fem::MDSpan2 auto entities,
    std::span<const std::uint32_t> cell_info,
    fem::DofTransformKernel<T> auto P0)
{
  namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;
  static_assert(entities.rank() == 1 or entities.rank() == 2);

  // Create data structures used in evaluation
  std::vector<U> coord_dofs(3 * x_dofmap.extent(1));

  // Iterate over cells and 'assemble' into values
  int size0 = Xshape[0] * value_size;
  std::vector<T> values_local(size0 * num_argument_dofs, 0);
  std::size_t offset = values_local.size();
  for (std::size_t e = 0; e < entities.extent(0); ++e)
  {
    std::ranges::fill(values_local, 0);
    if constexpr (entities.rank() == 1)
    {
      std::int32_t entity = entities(e);
      auto x_dofs = md::submdspan(x_dofmap, entity, md::full_extent);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                    std::next(coord_dofs.begin(), 3 * i));
      }
      fn(values_local.data(), &coeffs(e, 0), constants.data(),
         coord_dofs.data(), nullptr, nullptr, nullptr);
    }
    else
    {
      std::int32_t entity = entities(e, 0);
      auto x_dofs = md::submdspan(x_dofmap, entity, md::full_extent);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                    std::next(coord_dofs.begin(), 3 * i));
      }
      fn(values_local.data(), &coeffs(e, 0), constants.data(),
         coord_dofs.data(), &entities(e, 1), nullptr, nullptr);
    }

    P0(values_local, cell_info, e, size0);
    for (std::size_t j = 0; j < values_local.size(); ++j)
      values[e * offset + j] = values_local[j];
  }
}

/// @brief Tabulate an Expression at points.
///
/// Executes an Expression kernel over a list of mesh entities.
///
/// @tparam T Scalar type
/// @tparam U Geometry type
/// @param[in,out] values Array in which the tabulated expressions are
/// set. If `V` is available, it should have shape (`entities.extent(0),
/// Xshape[0], value_size, dim(V))`. Otherwise is has shape
/// (`entities.extent(0), Xshape[0], value_size)`. Storage is row-major.
/// Data is set (not accumulated).
/// @param[in] fn Expression kernel to execute.
/// @param[in] Xshape Shape `(num_points, geometric dimensions)` of
/// points array at which the expression is evaluated by the kernel.
/// @param[in] value_size Value size of the evaluated expression at a
/// point, e.g. 1 for a scalar field and 3 for a vector field in 3D.
/// @param[in] coeffs Coefficient data that appears in the Expression.
/// Shape is `(num_cells, coeff data per cell)`. Usually packed using
/// fem::pack_coefficients.
/// @param[in] constants Constant (coefficient) data that appears in
/// expression. Usually packed using fem::pack_constants.
/// @param[in] mesh Mesh to execute the expression kernel on.
/// @param[in] entities Mesh entities to evaluate the expression over.
/// For expressions executed on cells, rank is 1 and size is the number
/// of cells. For expressions executed on facets rank is 2, and shape is
/// `(num_facets, 2)`, where `entities[i, 0]` is the cell index and
/// `entities[i, 1]` is the local index of the facet relative to the
/// cell.
/// @param[in] element Argument element and argument space dimension.
/// Used to computed a 1-form expression, e.g. can be used to create a
/// matrix that when applied to a degree-of-freedom vector gives the
/// expression values at the evaluation points.
template <dolfinx::scalar T, std::floating_point U>
void tabulate_expression(
    std::span<T> values, fem::FEkernel<T> auto fn,
    std::array<std::size_t, 2> Xshape, std::size_t value_size,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        coeffs,
    std::span<const T> constants, const mesh::Mesh<U>& mesh,
    fem::MDSpan2 auto entities,
    std::optional<
        std::pair<std::reference_wrapper<const FiniteElement<U>>, std::size_t>>
        element)
{
  namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

  std::function<void(std::span<T>, std::span<const std::uint32_t>, std::int32_t,
                     int)>
      post_dof_transform
      = [](std::span<T>, std::span<const std::uint32_t>, std::int32_t, int)
  {
    // Do nothing
  };

  std::shared_ptr<const mesh::Topology> topology = mesh.topology();
  assert(topology);
  std::size_t num_argument_dofs = 1;
  std::span<const std::uint32_t> cell_info;
  if (element)
  {
    num_argument_dofs = element->second;
    if (element->first.get().needs_dof_transformations())
    {
      mesh.topology_mutable()->create_entity_permutations();
      cell_info = std::span(topology->get_cell_permutation_info());
      post_dof_transform
          = element->first.get().template dof_transformation_right_fn<T>(
              doftransform::transpose);
    }
  }

  tabulate_expression<T, U>(values, fn, Xshape, value_size, num_argument_dofs,
                            mesh.geometry().dofmap(), mesh.geometry().x(),
                            coeffs, constants, entities, cell_info,
                            post_dof_transform);
}
} // namespace dolfinx::fem::impl
