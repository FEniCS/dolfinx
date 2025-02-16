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
/// Function executes an Expression kernel over a list of mesh entities.
///
/// @tparam T Scalar type
/// @tparam U Geometry type
/// @param[in,out] values
/// @param[in] vshape
/// @param[in] fn Expression kernel to execute.
/// @param[in] Xshape
/// @param[in] value_size
/// @param[in] coeffs Coefficient data that appears in expression.
/// @param[in] cstride
/// @param[in] constant_data Constant (coefficient) data that appears in
/// expression.
/// @param[in] mesh Mesh to evaluate expression on.
/// @param[in] entities Mesh entities to evaluate the expression over.
/// @param[in] V
template <dolfinx::scalar T, std::floating_point U>
void tabulate_expression(
    std::span<T> values, fem::FEkernel<T> auto fn,
    std::array<std::size_t, 2> Xshape, std::size_t value_size,
    std::span<const T> coeffs, std::size_t cstride,
    std::span<const T> constant_data, const mesh::Mesh<U>& mesh,
    std::span<const std::int32_t> entities,
    std::optional<std::reference_wrapper<const FunctionSpace<U>>> V)
{
  namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

  std::size_t estride;
  if (mesh.topology()->dim() == Xshape[1])
    estride = 1;
  else if (mesh.topology()->dim() == Xshape[1] + 1)
    estride = 2;
  else
    throw std::runtime_error("Invalid dimension of evaluation points.");

  // Prepare cell geometry
  auto x_dofmap = mesh.geometry().dofmap();
  auto& cmap = mesh.geometry().cmap();
  std::size_t num_dofs_g = cmap.dim();
  auto x_g = mesh.geometry().x();

  // Create data structures used in evaluation
  std::vector<U> coord_dofs(3 * num_dofs_g);

  int num_argument_dofs = 1;
  std::span<const std::uint32_t> cell_info;
  std::function<void(std::span<T>, std::span<const std::uint32_t>, std::int32_t,
                     int)>
      post_dof_transform
      = [](std::span<T>, std::span<const std::uint32_t>, std::int32_t, int)
  {
    // Do nothing
  };

  if (V)
  {
    num_argument_dofs = V->get().dofmap()->element_dof_layout().num_dofs();
    auto element = V->get().element();
    num_argument_dofs *= V->get().dofmap()->bs();
    assert(element);
    if (element->needs_dof_transformations())
    {
      mesh.topology_mutable()->create_entity_permutations();
      cell_info = std::span(mesh.topology()->get_cell_permutation_info());
      post_dof_transform = element->template dof_transformation_right_fn<T>(
          doftransform::transpose);
    }
  }

  // Create get entity index function
  std::function<const std::int32_t*(std::span<const std::int32_t>, std::size_t)>
      get_entity_index
      = [](std::span<const std::int32_t> /*entities*/, std::size_t /*idx*/)
  { return nullptr; };
  if (estride == 2)
  {
    get_entity_index
        = [](std::span<const std::int32_t> entities, std::size_t idx)
    { return entities.data() + 2 * idx + 1; };
  }

  // Iterate over cells and 'assemble' into values
  int size0 = Xshape[0] * value_size;
  std::vector<T> values_local(size0 * num_argument_dofs, 0);
  for (std::size_t e = 0; e < entities.size() / estride; ++e)
  {
    std::int32_t entity = entities[e * estride];
    auto x_dofs = md::submdspan(x_dofmap, entity, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                  std::next(coord_dofs.begin(), 3 * i));
    }

    const T* coeff_cell = coeffs.data() + e * cstride;
    const int* entity_index = get_entity_index(entities, e);

    std::ranges::fill(values_local, 0);
    fn(values_local.data(), coeff_cell, constant_data.data(), coord_dofs.data(),
       entity_index, nullptr);
    post_dof_transform(values_local, cell_info, e, size0);
    for (std::size_t j = 0; j < values_local.size(); ++j)
      values[e * Xshape[0] + j] = values_local[j];
  }
}
} // namespace dolfinx::fem::impl
