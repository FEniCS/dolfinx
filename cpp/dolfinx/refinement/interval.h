// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <mpi.h>

#include <concepts>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "dolfinx/mesh/Mesh.h"
#include "dolfinx/mesh/cell_types.h"
#include "dolfinx/mesh/utils.h"
#include "dolfinx/refinement/plaza.h"

namespace dolfinx::refinement
{

namespace impl
{

/// Refine with markers returning new mesh data.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] edges Indices of the edges that are marked for refinement
///
/// @return New mesh data: cell topology, vertex coordinates and parent
/// edge indices.
template <std::floating_point T>
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<T>,
           std::array<std::size_t, 2>, std::vector<std::int32_t>>
compute_interval_refinement(const mesh::Mesh<T>& mesh,
                            std::optional<std::span<const std::int32_t>> edges)
{
  auto topology = mesh.topology();
  assert(topology);
  assert(topology->dim() == 1);

  auto map_e = topology->index_map(1);
  assert(map_e);

  // TODO: creation of sharing ranks in external function? Also same code in use
  // for plaza
  // Get sharing ranks for each edge
  graph::AdjacencyList<int> edge_ranks = map_e->index_to_dest_ranks();

  // Create unique list of ranks that share edges (owners of ghosts plus
  // ranks that ghost owned indices)
  std::vector<int> ranks = edge_ranks.array();
  std::ranges::sort(ranks);
  auto to_remove = std::ranges::unique(ranks);
  ranks.erase(to_remove.begin(), to_remove.end());

  // Convert edge_ranks from global rank to to neighbourhood ranks
  std::ranges::transform(edge_ranks.array(), edge_ranks.array().begin(),
                         [&ranks](auto r)
                         {
                           auto it = std::lower_bound(ranks.begin(),
                                                      ranks.end(), r);
                           assert(it != ranks.end() and *it == r);
                           return std::distance(ranks.begin(), it);
                         });

  // create refinement flag for edges
  // TODO: vector of bools? -> make of use std specialization for type bool
  std::vector<std::int8_t> refinement_marker(
      map_e->size_local() + map_e->num_ghosts(), !edges.has_value());

  // mark edges for refinement
  std::vector<std::vector<std::int32_t>> marked_for_update(ranks.size());
  if (edges.has_value())
  {
    std::ranges::for_each(edges.value(),
                          [&](auto edge)
                          {
                            if (!refinement_marker[edge])
                            {
                              refinement_marker[edge] = true;
                              for (int rank : edge_ranks.links(edge))
                                marked_for_update[rank].push_back(edge);
                            }
                          });
  }

  // create neighborhood communicator for vertex creation
  MPI_Comm neighbor_comm;
  MPI_Dist_graph_create_adjacent(
      mesh.comm(), ranks.size(), ranks.data(), MPI_UNWEIGHTED, ranks.size(),
      ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neighbor_comm);

  // Communicate ghost edges that might have been marked. This is not necessary
  // for a uniform refinement.
  if (edges.has_value())
    update_logical_edgefunction(neighbor_comm, marked_for_update,
                                refinement_marker, *map_e);

  // Construct the new vertices
  const auto [new_vertex_map, new_vertex_coords, xshape]
      = create_new_vertices(neighbor_comm, edge_ranks, mesh, refinement_marker);
  MPI_Comm_free(&neighbor_comm);

  auto e_to_v = mesh.topology()->connectivity(1, 0);
  assert(e_to_v);

  // get the count of edges to refine, note: we only consider non-ghost edges
  std::int32_t number_of_refined_edges
      = std::count(refinement_marker.begin(),
                   std::next(refinement_marker.begin(),
                             mesh.topology()->index_map(1)->size_local()),
                   true);

  // Produce local global indices, by padding out the previous index map
  std::vector<std::int64_t> global_indices
      = adjust_indices(*mesh.topology()->index_map(0), number_of_refined_edges);

  // Build the topology on the new vertices
  const auto refined_cell_count = mesh.topology()->index_map(1)->size_local()
                                  + mesh.topology()->index_map(1)->num_ghosts()
                                  + number_of_refined_edges;

  std::vector<std::int64_t> edge_topology;
  edge_topology.reserve(refined_cell_count * 2);

  std::vector<std::int32_t> parent_edge;
  parent_edge.reserve(refined_cell_count);

  for (std::int32_t edge = 0; edge < map_e->size_local(); ++edge)
  {
    const auto& vertices = e_to_v->links(edge);
    assert(vertices.size() == 2);

    // we consider a (previous) edge of (global) vertices
    // a ----------- b
    const std::int64_t a = global_indices[vertices[0]];
    const std::int64_t b = global_indices[vertices[1]];

    if (refinement_marker[edge])
    {
      // find (global) index of new midpoint vertex:
      // a --- c --- b
      auto it = new_vertex_map.find(edge);
      assert(it != new_vertex_map.end());
      const std::int64_t c = it->second;

      // add new edges to refined topology
      edge_topology.insert(edge_topology.end(), {a, c, c, b});
      parent_edge.insert(parent_edge.end(), {edge, edge});
    }
    else
    {
      // copy the previous edge
      edge_topology.insert(edge_topology.end(), {a, b});
      parent_edge.push_back(edge);
    }
  }

  assert(edge_topology.size() == refined_cell_count * 2);
  assert(parent_edge.size() == refined_cell_count);

  std::vector<std::int32_t> offsets(refined_cell_count + 1);
  std::ranges::generate(offsets, [i = 0]() mutable { return 2 * i++; });

  graph::AdjacencyList cell_adj(std::move(edge_topology), std::move(offsets));

  return {std::move(cell_adj), std::move(new_vertex_coords), xshape,
          std::move(parent_edge)};
}

} // namespace impl

/// Refines a (topologically) one dimensional mesh by splitting edges.
///
/// @param[in] mesh Mesh to be refined
/// @param[in] edges Optional indices of the edges that should be split by this
/// refinement. If not provided, all edges are considered marked for refinement,
/// i.e. a uniform refinement is performed.
/// @param[in] redistribute Option to enable redistribution of the refined mesh
/// across processes.
///
/// @return Refined mesh, and list of parent edges - for every new edge index
/// this contains the associated edge index of the pre-refinement mesh.
template <std::floating_point T>
std::tuple<mesh::Mesh<T>, std::vector<std::int32_t>>
refine_interval(const mesh::Mesh<T>& mesh,
                std::optional<std::span<const std::int32_t>> edges,
                bool redistribute)
{

  if (mesh.topology()->cell_type() != mesh::CellType::interval)
    throw std::runtime_error("Cell type not supported");

  if (!mesh.topology()->index_map(1))
    throw std::runtime_error("Edges must be initialised");

  assert(mesh.topology()->dim() == 1);

  auto [cell_adj, new_coords, xshape, parent_cell]
      = impl::compute_interval_refinement(mesh, edges);

  if (dolfinx::MPI::size(mesh.comm()) == 1)
  {
    return {mesh::create_mesh(mesh.comm(), cell_adj.array(),
                              mesh.geometry().cmap(), new_coords, xshape,
                              mesh::GhostMode::none),
            std::move(parent_cell)};
  }
  else
  {
    // Check if mesh has ghost cells on any rank
    // FIXME: this is not a robust test. Should be user option.
    const int num_ghost_cells = mesh.topology()->index_map(1)->num_ghosts();
    int max_ghost_cells = 0;
    MPI_Allreduce(&num_ghost_cells, &max_ghost_cells, 1, MPI_INT, MPI_MAX,
                  mesh.comm());

    // Build mesh
    const auto ghost_mode = max_ghost_cells == 0
                                ? mesh::GhostMode::none
                                : mesh::GhostMode::shared_facet;

    return {partition<T>(mesh, cell_adj, std::span(new_coords), xshape,
                         redistribute, ghost_mode),
            std::move(parent_cell)};
  }
}

} // namespace dolfinx::refinement
