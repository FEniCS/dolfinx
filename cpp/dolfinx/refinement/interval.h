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
/// @param[in] cells Indices of the cells that are marked for refinement
///
/// @return New mesh data: cell topology, vertex coordinates and parent
/// cell indices.
template <std::floating_point T>
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<T>,
           std::array<std::size_t, 2>, std::vector<std::int32_t>>
compute_interval_refinement(const mesh::Mesh<T>& mesh,
                            std::optional<std::span<const std::int32_t>> cells)
{
  auto topology = mesh.topology();
  assert(topology);
  assert(topology->dim() == 1);

  auto map_c = topology->index_map(1);
  assert(map_c);

  // TODO: creation of sharing ranks in external function? Also same code in use
  // for plaza
  // Get sharing ranks for each cell
  graph::AdjacencyList<int> cell_ranks = map_c->index_to_dest_ranks();

  // Create unique list of ranks that share cells (owners of ghosts plus
  // ranks that ghost owned indices)
  std::vector<int> ranks = cell_ranks.array();
  std::ranges::sort(ranks);
  auto to_remove = std::ranges::unique(ranks);
  ranks.erase(to_remove.begin(), to_remove.end());

  // Convert cell_ranks from global rank to to neighbourhood ranks
  std::ranges::transform(cell_ranks.array(), cell_ranks.array().begin(),
                         [&ranks](auto r)
                         {
                           auto it = std::lower_bound(ranks.begin(),
                                                      ranks.end(), r);
                           assert(it != ranks.end() and *it == r);
                           return std::distance(ranks.begin(), it);
                         });

  // Create refinement flag for cells
  std::vector<std::int8_t> refinement_marker(
      map_c->size_local() + map_c->num_ghosts(), !cells.has_value());

  // Mark cells for refinement
  std::vector<std::vector<std::int32_t>> marked_for_update(ranks.size());
  if (cells.has_value())
  {
    std::ranges::for_each(cells.value(),
                          [&](auto cell)
                          {
                            if (!refinement_marker[cell])
                            {
                              refinement_marker[cell] = true;
                              for (int rank : cell_ranks.links(cell))
                                marked_for_update[rank].push_back(cell);
                            }
                          });
  }

  // Create neighborhood communicator for vertex creation
  MPI_Comm neighbor_comm;
  MPI_Dist_graph_create_adjacent(
      mesh.comm(), ranks.size(), ranks.data(), MPI_UNWEIGHTED, ranks.size(),
      ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neighbor_comm);

  // Communicate ghost cells that might have been marked. This is not necessary
  // for a uniform refinement.
  if (cells.has_value())
    update_logical_edgefunction(neighbor_comm, marked_for_update,
                                refinement_marker, *map_c);

  // Construct the new vertices
  const auto [new_vertex_map, new_vertex_coords, xshape]
      = create_new_vertices(neighbor_comm, cell_ranks, mesh, refinement_marker);
  MPI_Comm_free(&neighbor_comm);

  auto c_to_v = mesh.topology()->connectivity(1, 0);
  assert(c_to_v);

  // Get the count of cells to refine, note: we only consider non-ghost cells
  std::int32_t number_of_refined_cells
      = std::count(refinement_marker.begin(),
                   std::next(refinement_marker.begin(),
                             mesh.topology()->index_map(1)->size_local()),
                   true);

  // Produce local global indices, by padding out the previous index map
  std::vector<std::int64_t> global_indices
      = adjust_indices(*mesh.topology()->index_map(0), number_of_refined_cells);

  // Build the topology on the new vertices
  const auto refined_cell_count = mesh.topology()->index_map(1)->size_local()
                                  + mesh.topology()->index_map(1)->num_ghosts()
                                  + number_of_refined_cells;

  std::vector<std::int64_t> cell_topology;
  cell_topology.reserve(refined_cell_count * 2);

  std::vector<std::int32_t> parent_cell;
  parent_cell.reserve(refined_cell_count);

  for (std::int32_t cell = 0; cell < map_c->size_local(); ++cell)
  {
    const auto& vertices = c_to_v->links(cell);
    assert(vertices.size() == 2);

    // We consider a cell (defined by global vertices)
    // a ----------- b
    const std::int64_t a = global_indices[vertices[0]];
    const std::int64_t b = global_indices[vertices[1]];

    if (refinement_marker[cell])
    {
      // Find (global) index of new midpoint vertex:
      // a --- c --- b
      auto it = new_vertex_map.find(cell);
      assert(it != new_vertex_map.end());
      const std::int64_t c = it->second;

      // Add new cells/edges to refined topology
      cell_topology.insert(cell_topology.end(), {a, c, c, b});
      parent_cell.insert(parent_cell.end(), {cell, cell});
    }
    else
    {
      // Copy the previous cell
      cell_topology.insert(cell_topology.end(), {a, b});
      parent_cell.push_back(cell);
    }
  }

  assert(cell_topology.size() == refined_cell_count * 2);
  assert(parent_cell.size() == refined_cell_count);

  std::vector<std::int32_t> offsets(refined_cell_count + 1);
  std::ranges::generate(offsets, [i = 0]() mutable { return 2 * i++; });

  graph::AdjacencyList cell_adj(std::move(cell_topology), std::move(offsets));

  return {std::move(cell_adj), std::move(new_vertex_coords), xshape,
          std::move(parent_cell)};
}

} // namespace impl

/// Refines a (topologically) one dimensional mesh by splitting cells, i.e.
/// edges.
///
/// @param[in] mesh Mesh to be refined
/// @param[in] cells Optional indices of the cells that should be refined by
/// this refinement. If not provided, all cells are considered marked for
/// refinement, i.e. a uniform refinement is performed.
/// @param[in] redistribute Option to enable redistribution of the refined mesh
/// across processes.
/// @param[in] ghost_mode Ghost mode of the refined mesh, default is ghost mode
/// none
///
/// @return Refined mesh, and list of parent cells and an array mapping the
/// child cell index of the refined mesh to its parent cell index in the
/// unrefined mesh.
template <std::floating_point T>
std::tuple<mesh::Mesh<T>, std::vector<std::int32_t>>
refine_interval(const mesh::Mesh<T>& mesh,
                std::optional<std::span<const std::int32_t>> cells,
                bool redistribute,
                mesh::GhostMode ghost_mode = mesh::GhostMode::none)
{

  if (mesh.topology()->cell_type() != mesh::CellType::interval)
    throw std::runtime_error("Cell type not supported");

  assert(mesh.topology()->dim() == 1);
  assert(mesh.topology()->index_map(1));

  auto [cell_adj, new_coords, xshape, parent_cell]
      = impl::compute_interval_refinement(mesh, cells);

  if (dolfinx::MPI::size(mesh.comm()) == 1)
  {
    return {mesh::create_mesh(mesh.comm(), cell_adj.array(),
                              mesh.geometry().cmap(), new_coords, xshape,
                              mesh::GhostMode::none),
            std::move(parent_cell)};
  }
  else
  {
    return {partition<T>(mesh, cell_adj, std::span(new_coords), xshape,
                         redistribute, ghost_mode),
            std::move(parent_cell)};
  }
}

} // namespace dolfinx::refinement
