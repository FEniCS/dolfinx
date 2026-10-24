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
class ElementDofLayout;
class Topology;
} // namespace dolfinx::mesh

namespace dolfinx::fem
{
class DofMap;

/// @deprecated Use mesh::build_dofmap_data instead.
[[deprecated("Use mesh::build_dofmap_data instead.")]]
std::tuple<common::IndexMap, int, std::vector<std::vector<std::int32_t>>>
build_dofmap_data(
    MPI_Comm comm, const mesh::Topology& topology,
    const std::vector<mesh::ElementDofLayout>& element_dof_layouts,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn);

/// @brief Build a dofmap on a real element, i.e. a single constant dof shared
/// by all cells.
///
/// @param[in] topology The mesh topology.
/// @param[in] dof_layout The element dof layout for the real element.
/// @return The dofmap for the real element.
fem::DofMap build_real_element_dofmap(const mesh::Topology& topology,
                                      const mesh::ElementDofLayout& dof_layout);

} // namespace dolfinx::fem
