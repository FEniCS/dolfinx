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
#include <stdexcept>
#include <vector>

using namespace dolfinx;

//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int64_t>
mesh::extract_topology(const CellType& cell_type,
                       const fem::ElementDofLayout& layout,
                       const graph::AdjacencyList<std::int64_t>& cells)
{
  // Use ElementDofLayout to get vertex dof indices (local to a cell)
  const int num_vertices_per_cell = num_cell_vertices(cell_type);
  std::vector<int> local_vertices(num_vertices_per_cell);
  for (int i = 0; i < num_vertices_per_cell; ++i)
  {
    const std::vector<int> local_index = layout.entity_dofs(0, i);
    assert(local_index.size() == 1);
    local_vertices[i] = local_index[0];
  }

  // Extract vertices
  std::vector<std::int64_t> topology(cells.num_nodes() * num_vertices_per_cell);
  for (int c = 0; c < cells.num_nodes(); ++c)
  {
    auto p = cells.links(c);
    for (int j = 0; j < num_vertices_per_cell; ++j)
      topology[num_vertices_per_cell * c + j] = p[local_vertices[j]];
  }

  return graph::regular_adjacency_list(std::move(topology),
                                       num_vertices_per_cell);
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> mesh::exterior_facet_indices(const Topology& topology)
{
  const int tdim = topology.dim();
  auto facet_map = topology.index_map(tdim - 1);
  if (!facet_map)
    throw std::runtime_error("Facets have not been computed.");

  // Find all owned facets (not ghost) with only one attached cell
  const int num_facets = facet_map->size_local();
  auto f_to_c = topology.connectivity(tdim - 1, tdim);
  assert(f_to_c);
  std::vector<std::int32_t> facets;
  for (std::int32_t f = 0; f < num_facets; ++f)
  {
    if (f_to_c->num_links(f) == 1)
      facets.push_back(f);
  }

  // Remove facets on internal inter-process boundary
  const std::vector<std::int32_t>& interprocess_facets
      = topology.interprocess_facets();
  std::vector<std::int32_t> ext_facets;
  std::set_difference(facets.begin(), facets.end(), interprocess_facets.begin(),
                      interprocess_facets.end(),
                      std::back_inserter(ext_facets));
  return ext_facets;
}
//------------------------------------------------------------------------------
mesh::CellPartitionFunction
mesh::create_cell_partitioner(mesh::GhostMode ghost_mode,
                              const graph::partition_fn& partfn)
{
  return [partfn, ghost_mode](MPI_Comm comm, int nparts, int tdim,
                              const graph::AdjacencyList<std::int64_t>& cells)
             -> graph::AdjacencyList<std::int32_t>
  {
    LOG(INFO) << "Compute partition of cells across ranks";

    // Compute distributed dual graph (for the cells on this process)
    const graph::AdjacencyList dual_graph = build_dual_graph(comm, cells, tdim);

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

  std::sort(entities1.begin(), entities1.end());
  entities1.erase(std::unique(entities1.begin(), entities1.end()),
                  entities1.end());
  return entities1;
}
//-----------------------------------------------------------------------------
std::pair<mesh::Topology, graph::AdjacencyList<std::int64_t>>
mesh::build_topology(MPI_Comm comm, MPI_Comm commt, mesh::CellType celltype,
                     const fem::ElementDofLayout& dof_layout,
                     const graph::AdjacencyList<std::int64_t>& cells,
                     const CellPartitionFunction& partitioner)
{
  // -- Partition topology across ranks of comm

  // Note: the function extract_topology (returns an
  // AdjacencyList<std::int64_t>) extract topology data, e.g. just the
  // vertices. For P1 geometry this should just be the identity
  // operator. For other elements the filtered lists may have 'gaps',
  // i.e. the indices might not be contiguous. We don't create an
  // object before calling partitioner to ensure that memory is
  // freed immediately.
  //
  // Note: extract_topology could be skipped for 'P1' elements since
  // it is just the identity

  // Compute the destination rank for cells on this process via graph
  // partitioning.
  const int tdim = cell_dim(celltype);

  graph::AdjacencyList<std::int32_t> dest(0);
  graph::AdjacencyList<std::int64_t> cell_nodes(0);
  std::vector<std::int64_t> original_cell_index0;
  std::vector<int> ghost_owners;
  if (partitioner)
  {
    if (commt != MPI_COMM_NULL)
    {
      const int size = dolfinx::MPI::size(comm);
      dest = partitioner(commt, size, tdim,
                         extract_topology(celltype, dof_layout, cells));
    }

    // -- Distribute cells (topology, includes higher-order 'nodes')

    // Distribute cells to destination rank
    std::vector<int> src;
    std::tie(cell_nodes, src, original_cell_index0, ghost_owners)
        = graph::build::distribute(comm, cells, dest);
  }
  else
  {
    int rank = dolfinx::MPI::rank(comm);
    dest = graph::regular_adjacency_list(
        std::vector<std::int32_t>(cells.num_nodes(), rank), 1);
    cell_nodes = cells;
    std::int64_t offset(0), num_owned(cells.num_nodes());
    MPI_Exscan(&num_owned, &offset, 1, MPI_INT64_T, MPI_SUM, comm);
    original_cell_index0.resize(cell_nodes.num_nodes());
    std::iota(original_cell_index0.begin(), original_cell_index0.end(), offset);
  }

  // -- Extract cell topology

  // Extract cell 'topology', i.e. extract the vertices for each cell
  // and discard any 'higher-order' nodes
  graph::AdjacencyList<std::int64_t> cells_extracted
      = extract_topology(celltype, dof_layout, cell_nodes);

  // -- Re-order cells

  // Build local dual graph for owned cells to apply re-ordering
  const std::int32_t num_owned_cells
      = cells_extracted.num_nodes() - ghost_owners.size();
  auto [graph, unmatched_facets, max_v, facet_attached_cells]
      = build_local_dual_graph(
          std::span<const std::int64_t>(
              cells_extracted.array().data(),
              cells_extracted.offsets()[num_owned_cells]),
          std::span<const std::int32_t>(cells_extracted.offsets().data(),
                                        num_owned_cells + 1),
          tdim);
  const std::vector<int> remap = graph::reorder_gps(graph);

  // Create re-ordered cell lists (leaves ghosts unchanged)
  std::vector<std::int64_t> original_cell_index(original_cell_index0.size());
  for (std::size_t i = 0; i < remap.size(); ++i)
    original_cell_index[remap[i]] = original_cell_index0[i];
  std::copy_n(std::next(original_cell_index0.cbegin(), num_owned_cells),
              ghost_owners.size(),
              std::next(original_cell_index.begin(), num_owned_cells));
  cells_extracted = impl::reorder_list(cells_extracted, remap);
  cell_nodes = impl::reorder_list(cell_nodes, remap);

  // -- Create Topology

  // Boundary vertices are marked as unknown
  std::vector<std::int64_t> boundary_vertices(unmatched_facets);
  std::sort(boundary_vertices.begin(), boundary_vertices.end());
  boundary_vertices.erase(
      std::unique(boundary_vertices.begin(), boundary_vertices.end()),
      boundary_vertices.end());

  // Remove -1 if it occurs in boundary vertices (may occur in mixed topology)
  if (boundary_vertices.size() > 0 and boundary_vertices[0] == -1)
    boundary_vertices.erase(boundary_vertices.begin());

  // Create cells and vertices with the ghosting requested. Input
  // topology includes cells shared via facet, but ghosts will be
  // removed later if not required by ghost_mode.

  std::vector<std::int32_t> cell_group_offsets
      = {0, std::int32_t(cells_extracted.num_nodes() - ghost_owners.size()),
         cells_extracted.num_nodes()};
  return std::pair{create_topology(comm, cells_extracted, original_cell_index,
                                   ghost_owners, {celltype}, cell_group_offsets,
                                   boundary_vertices),
                   std::move(cell_nodes)};
}
//-----------------------------------------------------------------------------
