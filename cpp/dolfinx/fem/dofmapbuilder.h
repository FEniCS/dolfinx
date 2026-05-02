// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <mpi.h>
#include <tuple>
#include <vector>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::mesh
{
class Topology;
}

namespace dolfinx::fem
{
class ElementDofLayout;
class DofMap;

/// Build dofmap data for elements on a mesh topology
/// @param[in] comm MPI communicator
/// @param[in] topology The mesh topology
/// @param[in] element_dof_layouts The element dof layouts for each cell
/// type in `topology`.
/// @param[in] reorder_fn Graph reordering function that is applied to
/// the dofmaps
/// @return The index map, block size, and dofmaps for each element type
std::tuple<common::IndexMap, int, std::vector<std::vector<std::int32_t>>>
build_dofmap_data(MPI_Comm comm, const mesh::Topology& topology,
                  const std::vector<ElementDofLayout>& element_dof_layouts,
                  const std::function<std::vector<int>(
                      const graph::AdjacencyList<std::int32_t>&)>& reorder_fn);

/// @brief Build a dofmap on a real element, i.e. a single constant dof shared
/// by all cells.
///
/// @param[in] topology The mesh topology.
/// @param[in] entity_dofs The dofs for each mesh entity.
/// @param[in] entity_closure_dofs The closure dofs for each mesh entity.
/// @param[in] value_size The number of components for the real element.
/// @return The dofmap for the real element.
fem::DofMap build_real_element_dofmap(
    const mesh::Topology& topology,
    const std::vector<std::vector<std::vector<int>>>& entity_dofs,
    const std::vector<std::vector<std::vector<int>>>& entity_closure_dofs,
    int value_size);

} // namespace dolfinx::fem
