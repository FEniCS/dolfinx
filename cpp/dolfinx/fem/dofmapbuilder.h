// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <functional>
#include <mpi.h>
#include <tuple>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::mesh
{
class Topology;
}

namespace dolfinx::graph
{
template <typename T>
class AdjacencyList;
}

namespace dolfinx::fem
{
class ElementDofLayout;

/// Build dofmap data for an element on a mesh topology
/// @param[in] comm MPI communicator
/// @param[in] topology The mesh topology
/// @param[in] element_dof_layouts The element dof layouts for the
/// function space
/// @param[in] reorder_fn Graph reordering function that is applied to
/// the dofmap
/// @return The index map and local to global DOF data for the DOF map
std::tuple<common::IndexMap, int, std::vector<std::int32_t>>
build_dofmap_data(MPI_Comm comm, const mesh::Topology& topology,
                  const std::vector<ElementDofLayout>& element_dof_layouts,
                  const std::function<std::vector<int>(
                      const graph::AdjacencyList<std::int32_t>&)>& reorder_fn);

} // namespace dolfinx::fem
