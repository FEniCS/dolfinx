// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Geometry.h"
#include "Mesh.h"
#include "Topology.h"
#include "cell_types.h"
#include "graphbuild.h"
#include <algorithm>
#include <cstdlib>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/math.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <span>
#include <stdexcept>
#include <vector>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::vector<std::int64_t>
mesh::extract_topology(CellType cell_type, const fem::ElementDofLayout& layout,
                       std::span<const std::int64_t> cells)
{
  // Use ElementDofLayout to get vertex dof indices (local to a cell)
  const int num_vertices_per_cell = num_cell_vertices(cell_type);
  const int num_node_per_cell = layout.num_dofs();
  std::vector<int> local_vertices(num_vertices_per_cell);
  for (int i = 0; i < num_vertices_per_cell; ++i)
  {
    const std::vector<int>& local_index = layout.entity_dofs(0, i);
    assert(local_index.size() == 1);
    local_vertices[i] = local_index[0];
  }

  // Extract vertices
  std::vector<std::int64_t> topology((cells.size() / num_node_per_cell)
                                     * num_vertices_per_cell);
  for (std::size_t c = 0; c < cells.size() / num_node_per_cell; ++c)
  {
    auto p = cells.subspan(c * num_node_per_cell, num_node_per_cell);
    std::span t(topology.data() + c * num_vertices_per_cell,
                num_vertices_per_cell);
    for (int j = 0; j < num_vertices_per_cell; ++j)
      t[j] = p[local_vertices[j]];
  }

  return topology;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> mesh::exterior_facet_indices(const Topology& topology,
                                                       int facet_type_idx)
{
  const int tdim = topology.dim();
  auto f_to_c = topology.connectivity(tdim - 1, tdim);
  if (!f_to_c)
  {
    throw std::runtime_error(
        "Facet to cell connectivity has not been computed.");
  }

  // Find all owned facets (not ghost) with only one attached cell
  auto facet_map = topology.index_maps(tdim - 1).at(facet_type_idx);

  std::vector<std::int32_t> facets;
  for (std::int32_t f = 0; f < facet_map->size_local(); ++f)
  {
    if (f_to_c->num_links(f) == 1)
      facets.push_back(f);
  }

  // Remove facets on internal inter-process boundary
  std::vector<std::int32_t> ext_facets;
  std::ranges::set_difference(facets,
                              topology.interprocess_facets(facet_type_idx),
                              std::back_inserter(ext_facets));

  return ext_facets;
}
//------------------------------------------------------------------------------
std::vector<std::int32_t> mesh::exterior_facet_indices(const Topology& topology)
{
  if (topology.entity_types(topology.dim() - 1).size() > 1)
  {
    throw std::runtime_error("Multiple facet types in mesh. Call "
                             "exterior_facet_indices with facet type index.");
  }

  return mesh::exterior_facet_indices(topology, 0);
}
//------------------------------------------------------------------------------
mesh::CellPartitionFunction
mesh::create_cell_partitioner(mesh::GhostMode ghost_mode,
                              const graph::partition_fn& partfn)
{
  return [partfn, ghost_mode](
             MPI_Comm comm, int nparts, const std::vector<CellType>& cell_types,
             const std::vector<std::span<const std::int64_t>>& cells)
             -> graph::AdjacencyList<std::int32_t>
  {
    spdlog::info("Compute partition of cells across ranks");

    // Compute distributed dual graph (for the cells on this process)
    const graph::AdjacencyList dual_graph
        = build_dual_graph(comm, cell_types, cells);

    // Just flag any kind of ghosting for now
    bool ghosting = (ghost_mode != GhostMode::none);

    // Compute partition
    return partfn(comm, nparts, dual_graph, ghosting);
  };
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
mesh::compute_incident_entities(const Topology& topology,
                                std::span<const std::int32_t> entities, int d0,
                                int d1)
{
  auto map0 = topology.index_map(d0);
  if (!map0)
  {
    throw std::runtime_error("Mesh entities of dimension " + std::to_string(d0)
                             + " have not been created.");
  }

  auto map1 = topology.index_map(d1);
  if (!map1)
  {
    throw std::runtime_error("Mesh entities of dimension " + std::to_string(d1)
                             + " have not been created.");
  }

  auto e0_to_e1 = topology.connectivity(d0, d1);
  if (!e0_to_e1)
  {
    throw std::runtime_error("Connectivity missing: (" + std::to_string(d0)
                             + ", " + std::to_string(d1) + ")");
  }

  std::vector<std::int32_t> entities1;
  for (std::int32_t entity : entities)
  {
    auto e = e0_to_e1->links(entity);
    entities1.insert(entities1.end(), e.begin(), e.end());
  }

  std::ranges::sort(entities1);
  auto [unique_end, range_end] = std::ranges::unique(entities1);
  entities1.erase(unique_end, range_end);

  return entities1;
}
//-----------------------------------------------------------------------------
