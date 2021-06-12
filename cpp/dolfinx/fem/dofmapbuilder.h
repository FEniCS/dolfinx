// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <memory>
#include <mpi.h>
#include <tuple>

#include <dolfinx/graph/scotch.h>

namespace dolfinx
{

namespace common
{
class IndexMap;
}

namespace mesh
{
class Topology;
} // namespace mesh

namespace fem
{
class ElementDofLayout;
class CoordinateElement;

/// Reorder graph using the SCOTCH GPS implementation
inline std::vector<int>
scotch_reorder(const graph::AdjacencyList<std::int32_t>& graph)
{
  return graph::scotch::compute_gps(graph, 2).first;
}

// /// Random graph reordering
// /// @note: Randomised dof ordering should only be used for
// /// testing/benchmarking
// std::vector<int> random_reorder(const graph::AdjacencyList<std::int32_t>&
// graph)
// {
//   std::vector<int> node_remap(graph.num_nodes());
//   std::iota(node_remap.begin(), node_remap.end(), 0);
//   std::random_device rd;
//   std::default_random_engine g(rd());
//   std::shuffle(node_remap.begin(), node_remap.end(), g);
//   return node_remap;
// }

/// Build dofmap data for an element on a mesh topology
/// @param[in] comm MPI communicator
/// @param[in] topology The mesh topology
/// @param[in] element_dof_layout The element dof layout for the
/// function space
/// @param[in] reorder_fn Graph reordering function that is applied to
/// the dofmap
/// @return The index map and local to global DOF data for the DOF map
std::tuple<std::shared_ptr<common::IndexMap>, int,
           graph::AdjacencyList<std::int32_t>>
build_dofmap_data(MPI_Comm comm, const mesh::Topology& topology,
                  const ElementDofLayout& element_dof_layout,
                  const std::function<std::vector<int>(
                      const graph::AdjacencyList<std::int32_t>)>& reorder_fn
                  = scotch_reorder);

} // namespace fem
} // namespace dolfinx
