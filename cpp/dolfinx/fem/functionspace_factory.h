// Copyright (C) 2013-2026 Johan Hake, Jan Blechta, Garth N. Wells and Paul T.
// Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "ElementDofLayout.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <memory>
#include <mpi.h>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

/// @file functionspace_factory.h
/// @brief Factories for finite element dofmaps and function spaces.

namespace dolfinx::fem
{
/// Create an ElementDofLayout from a FiniteElement
template <std::floating_point T>
ElementDofLayout create_element_dof_layout(const fem::FiniteElement<T>& element,
                                           const std::vector<int>& parent_map
                                           = {})
{
  // Create subdofmaps and compute offset
  std::vector<int> offsets(1, 0);
  std::vector<dolfinx::fem::ElementDofLayout> sub_doflayout;
  int bs = element.block_size();
  for (int i = 0; i < element.num_sub_elements(); ++i)
  {
    // The ith sub-element. For mixed elements this is subelements()[i]. For
    // blocked elements, the sub-element will always be the same, so we'll use
    // sub_elements()[0]
    std::shared_ptr<const fem::FiniteElement<T>> sub_e
        = element.sub_elements()[bs > 1 ? 0 : i];

    // In a mixed element DOFs are ordered element by element, so the offset to
    // the next sub-element is sub_e->space_dimension(). Blocked elements use
    // xxyyzz ordering, so the offset to the next sub-element is 1

    std::vector<int> parent_map_sub(sub_e->space_dimension(), offsets.back());
    for (std::size_t j = 0; j < parent_map_sub.size(); ++j)
      parent_map_sub[j] += bs * j;
    offsets.push_back(offsets.back() + (bs > 1 ? 1 : sub_e->space_dimension()));
    sub_doflayout.push_back(
        dolfinx::fem::create_element_dof_layout(*sub_e, parent_map_sub));
  }

  return ElementDofLayout(bs, element.entity_dofs(),
                          element.entity_closure_dofs(), parent_map,
                          sub_doflayout);
}

/// @brief Create a dof map on mesh
/// @param[in] comm MPI communicator
/// @param[in] layout Dof layout on an element
/// @param[in] topology Mesh topology
/// @param[in] permute_inv Function to un-permute dofs. `nullptr`
/// when transformation is not required.
/// @param[in] reorder_fn Graph reordering function called on the dofmap
/// @return A new dof map
DofMap
create_dofmap(MPI_Comm comm, const ElementDofLayout& layout,
              mesh::Topology& topology,
              const std::function<void(std::span<std::int32_t>, std::uint32_t)>&
                  permute_inv,
              const std::function<std::vector<int>(
                  const graph::AdjacencyList<std::int32_t>&)>& reorder_fn);

/// @brief Create a set of dofmaps on a given topology
/// @param[in] comm MPI communicator
/// @param[in] layouts Dof layout on each element type
/// @param[in] topology Mesh topology
/// @param[in] permute_inv Function to un-permute dofs. `nullptr`
/// when transformation is not required.
/// @param[in] reorder_fn Graph reordering function called on the dofmaps
/// @return The list of new dof maps
/// @note The number of layouts must match the number of cell types in the
/// topology
std::vector<DofMap> create_dofmaps(
    MPI_Comm comm, const std::vector<ElementDofLayout>& layouts,
    mesh::Topology& topology,
    const std::function<void(std::span<std::int32_t>, std::uint32_t)>&
        permute_inv,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn);

/// @brief NEW Create a function space from a fem::FiniteElement.
template <std::floating_point T>
FunctionSpace<T> create_functionspace(
    std::shared_ptr<mesh::Mesh<T>> mesh,
    std::shared_ptr<const fem::FiniteElement<T>> e,
    std::function<std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>
        reorder_fn = nullptr)
{
  // TODO: check cell type of e (need to add method to fem::FiniteElement)
  assert(e);
  assert(mesh);
  assert(mesh->topology());
  if (e->cell_type() != mesh->topology()->cell_type())
    throw std::invalid_argument("Cell type of element and mesh must match.");

  // Create element dof layout
  fem::ElementDofLayout layout = fem::create_element_dof_layout(*e);

  // Create a dofmap
  std::function<void(std::span<std::int32_t>, std::uint32_t)> permute_inv
      = e->needs_dof_permutations() ? e->dof_permutation_fn(true, true)
                                    : nullptr;
  auto dofmap = std::make_shared<const DofMap>(create_dofmap(
      mesh->comm(), layout, *mesh->topology(), permute_inv, reorder_fn));

  return FunctionSpace(mesh, e, dofmap);
}
} // namespace dolfinx::fem
