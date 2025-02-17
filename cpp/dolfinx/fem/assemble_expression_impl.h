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
/// Executes an Expression kernel over a list of mesh entities.
///
/// @tparam T Scalar type
/// @tparam U Geometry type
/// @param[in,out] values Array in which the tabulated expressions are
/// set. If `V` is available, it should have shape (`entities.size(),
/// Xshape[0], value_size, dim(V))`. Otherwise is has shape
/// (`entities.size(), Xshape[0], value_size)`. Storage is row-major.
/// Data is set (not accumulated).
/// @param[in] fn Expression kernel to execute.
/// @param[in] Xshape Shape `(num_points, geometric dimensions)` of
/// points array at which the expression is evaluated by the kernel.
/// @param[in] value_size Value size of the evaluated expression at a
/// point, e.g. 1 for a scalar field and 3 for a vector field in 3D.
/// @param[in] coeffs Coefficient data that appears in the expression.
/// Usually packed using fem::pack_coefficients.
/// @param[in] cstride
/// @param[in] constant_data Constant (coefficient) data that appears in
/// expression. Usually packed using em::pack_constants.
/// @param[in] mesh Mesh to execute the expression kernel on.
/// @param[in] entities Mesh entities to evaluate the expression over.
/// @param[in] V Argument space. Used to computed a 1-form expression,
/// e.g. can be used to create a matrix that when applied to a
/// degree-of-freedom vector gives the expression values at the
/// evaluation points.
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

  const mesh::Geometry<U>& geometry = mesh.geometry();
  std::shared_ptr<const mesh::Topology> topology = mesh.topology();

  assert(topology);
  std::size_t estride;
  if (std::size_t(topology->dim()) == Xshape[1])
    estride = 1;
  else if (std::size_t(topology->dim()) == Xshape[1] + 1)
    estride = 2;
  else
    throw std::runtime_error("Invalid dimension of evaluation points.");

  // Prepare cell geometry
  auto x_dofmap = geometry.dofmap();
  auto& cmap = geometry.cmap();
  std::size_t num_dofs_g = cmap.dim();
  auto x_g = geometry.x();

  // Create data structures used in evaluation
  std::vector<U> coord_dofs(3 * num_dofs_g);

  std::span<const std::uint32_t> cell_info;
  std::function<void(std::span<T>, std::span<const std::uint32_t>, std::int32_t,
                     int)>
      post_dof_transform
      = [](std::span<T>, std::span<const std::uint32_t>, std::int32_t, int)
  {
    // Do nothing
  };

  int num_argument_dofs = 1;
  if (V)
  {
    num_argument_dofs = V->get().dofmap()->element_dof_layout().num_dofs();
    num_argument_dofs *= V->get().dofmap()->bs();
    auto element = V->get().element();
    assert(element);
    if (element->needs_dof_transformations())
    {
      mesh.topology_mutable()->create_entity_permutations();
      cell_info = std::span(topology->get_cell_permutation_info());
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
  std::size_t offset = values_local.size();
  for (std::size_t e = 0; e < entities.size() / estride; ++e)
  {
    std::int32_t entity = entities[e * estride];
    auto x_dofs = md::submdspan(x_dofmap, entity, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                  std::next(coord_dofs.begin(), 3 * i));
    }

    std::ranges::fill(values_local, 0);
    fn(values_local.data(), coeffs.data() + e * cstride, constant_data.data(),
       coord_dofs.data(), get_entity_index(entities, e), nullptr);

    post_dof_transform(values_local, cell_info, e, size0);
    for (std::size_t j = 0; j < values_local.size(); ++j)
      values[e * offset + j] = values_local[j];
  }
}
} // namespace dolfinx::fem::impl
