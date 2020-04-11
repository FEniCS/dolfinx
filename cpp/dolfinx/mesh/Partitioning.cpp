// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Partitioning.h"
#include "Mesh.h"
#include "Topology.h"
#include "cell_types.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/SCOTCH.h>
#include <dolfinx/mesh/GraphBuilder.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
std::vector<bool> Partitioning::compute_vertex_exterior_markers(
    const mesh::Topology& topology_local)
{
  // Get list of boundary vertices
  const int dim = topology_local.dim();
  auto facet_cell = topology_local.connectivity(dim - 1, dim);
  if (!facet_cell)
  {
    throw std::runtime_error(
        "Need facet-cell connectivity to build distributed adjacency list.");
  }

  auto facet_vertex = topology_local.connectivity(dim - 1, 0);
  if (!facet_vertex)
  {
    throw std::runtime_error(
        "Need facet-vertex connectivity to build distributed adjacency list.");
  }

  auto map_vertex = topology_local.index_map(0);
  if (!map_vertex)
    throw std::runtime_error("Need vertex IndexMap from topology.");
  assert(map_vertex->num_ghosts() == 0);

  std::vector<bool> exterior_vertex(map_vertex->size_local(), false);
  for (int f = 0; f < facet_cell->num_nodes(); ++f)
  {
    if (facet_cell->num_links(f) == 1)
    {
      auto vertices = facet_vertex->links(f);
      for (int j = 0; j < vertices.rows(); ++j)
        exterior_vertex[vertices[j]] = true;
    }
  }

  return exterior_vertex;
}
//-------------------------------------------------------------
graph::AdjacencyList<std::int32_t> Partitioning::partition_cells(
    MPI_Comm comm, int n, const mesh::CellType cell_type,
    const graph::AdjacencyList<std::int64_t>& cells, mesh::GhostMode ghost_mode)
{
  common::Timer timer("Partition cells across processes");

  LOG(INFO) << "Compute partition of cells across processes";

  if (cells.num_nodes() > 0)
  {
    if (cells.num_links(0) != mesh::num_cell_vertices(cell_type))
    {
      throw std::runtime_error(
          "Inconsistent number of cell vertices. Got "
          + std::to_string(cells.num_links(0)) + ", expected "
          + std::to_string(mesh::num_cell_vertices(cell_type)) + ".");
    }
  }

  // FIXME: Update GraphBuilder to use AdjacencyList
  // Wrap AdjacencyList
  const Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>
      _cells(cells.array().data(), cells.num_nodes(),
             mesh::num_cell_vertices(cell_type));

  // Compute distributed dual graph (for the cells on this process)
  const auto [dual_graph, graph_info]
      = mesh::GraphBuilder::compute_dual_graph(comm, _cells, cell_type);

  // Extract data from graph_info
  const auto [num_ghost_nodes, num_local_edges, num_nonlocal_edges]
      = graph_info;

  graph::AdjacencyList<SCOTCH_Num> adj_graph(dual_graph);
  std::vector<std::size_t> weights;

  // Just flag any kind of ghosting for now
  bool ghosting = (ghost_mode != mesh::GhostMode::none);

  // Call partitioner
  graph::AdjacencyList<std::int32_t> partition = graph::SCOTCH::partition(
      comm, n, adj_graph, weights, num_ghost_nodes, ghosting);

  return partition;
}
//-----------------------------------------------------------------------------
