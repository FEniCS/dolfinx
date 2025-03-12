// Copyright (C) 2010-2024 Garth N. Wells and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "dolfinx/common/MPI.h"
#include "dolfinx/graph/AdjacencyList.h"
#include "dolfinx/graph/partitioners.h"
#include "dolfinx/mesh/Mesh.h"
#include "dolfinx/mesh/Topology.h"
#include "dolfinx/mesh/cell_types.h"
#include "dolfinx/mesh/graphbuild.h"
#include "dolfinx/mesh/utils.h"
#include "interval.h"
#include "plaza.h"
#include <algorithm>
#include <concepts>
#include <mpi.h>
#include <optional>
#include <spdlog/spdlog.h>
#include <utility>

namespace dolfinx::refinement
{

/**
 * @brief Create a cell partitioner which maintains the partition of a coarse
 * mesh.
 *
 * @tparam T floating point type of mesh geometry.
 * @param parent_mesh mesh indicating the partition scheme to use, i.e. the
 * coarse mesh.
 * @param parent_cell list of cell indices mapping cells of the new refined mesh
 * into the coarse mesh, usually output of `refinement::refine`.
 * @return The created cell partition function.
 */
template <std::floating_point T>
mesh::CellPartitionFunction
create_identity_partitioner(const mesh::Mesh<T>& parent_mesh,
                            std::span<std::int32_t> parent_cell)
{
  // TODO: optimize for non ghosted mesh?

  return [&parent_mesh,
          parent_cell](MPI_Comm comm, int /*nparts*/,
                       std::vector<mesh::CellType> cell_types,
                       std::vector<std::span<const std::int64_t>> cells)
             -> graph::AdjacencyList<std::int32_t>
  {
    auto cell_im
        = parent_mesh.topology()->index_map(parent_mesh.topology()->dim());

    std::int32_t num_cells = cells.front().size() / mesh::num_cell_vertices(cell_types.front());
    std::vector<std::int32_t> destinations(num_cells);

    int rank = dolfinx::MPI::rank(comm);
    for (std::int32_t i = 0; i < destinations.size(); i++)
    {
      bool ghost_parent_cell = parent_cell[i] > cell_im->size_local();
      if (ghost_parent_cell)
      {
        destinations[i]
            = cell_im->owners()[parent_cell[i] - cell_im->size_local()];
      }
      else
      {
        destinations[i] = rank;
      }
    }

    if (comm == MPI_COMM_NULL)
    {
      return graph::regular_adjacency_list(std::move(destinations), 1);
    }

    auto dual_graph = mesh::build_dual_graph(comm, cell_types, cells);
    std::vector<std::int32_t> node_disp(MPI::size(comm) + 1, 0);
    std::int32_t local_size = dual_graph.num_nodes();
    MPI_Allgather(&local_size, 1, dolfinx::MPI::mpi_t<std::int32_t>,
                  node_disp.data() + 1, 1, dolfinx::MPI::mpi_t<std::int32_t>,
                  comm);
    std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());
    return compute_destination_ranks(comm, dual_graph, node_disp, destinations);
  };
}

/// @brief Refine a mesh with markers.
///
/// The refined mesh can be optionally re-partitioned across processes.
/// Passing `nullptr` for `partitioner`, refined cells will be on the
/// same process as the parent cell.
///
/// Parent-child relationships can be optionally computed. Parent-child
/// relationships can be used to create MeshTags on the refined mesh
/// from MeshTags on the parent mesh.
///
/// @warning Using the default partitioner for a refined mesh, the
/// refined mesh will **not** include ghosts cells (cells connected by
/// facet to an owned cell) even if the parent mesh is ghosted.
///
/// @warning Passing `nullptr` for `partitioner`, the refined mesh will
/// **not** have ghosts cells even if the parent mesh is ghosted. The
/// possibility to not re-partition the refined mesh and include ghost
/// cells in the refined mesh will be added in a future release.
///
/// @param[in] mesh Input mesh to be refined.
/// @param[in] edges Indices of the edges that should be split in the
/// refinement. If not provided (`std::nullopt`), uniform refinement is
/// performed.
/// @param[in] partitioner Partitioner to be used to distribute the
/// refined mesh. If not callable, refined cells will be on the same
/// process as the parent cell.
/// @param[in] option Control the computation of parent facets, parent
/// cells. If an option is not selected, an empty list is returned.
/// @return New mesh, and optional parent cell indices and parent facet
/// indices.
template <std::floating_point T>
std::tuple<mesh::Mesh<T>, std::optional<std::vector<std::int32_t>>,
           std::optional<std::vector<std::int8_t>>>
refine(const mesh::Mesh<T>& mesh,
       std::optional<std::span<const std::int32_t>> edges,
       mesh::CellPartitionFunction partitioner = nullptr,
       Option option = Option::none)
{
  auto topology = mesh.topology();
  assert(topology);
  if (!mesh::is_simplex(topology->cell_type()))
    throw std::runtime_error("Refinement only defined for simplices");

  auto [cell_adj, new_vertex_coords, xshape, parent_cell, parent_facet]
      = (topology->cell_type() == mesh::CellType::interval)
            ? interval::compute_refinement_data(mesh, edges, option)
            : plaza::compute_refinement_data(mesh, edges, option);

  if (!partitioner)
  {
    assert(parent_cell);
    partitioner = create_identity_partitioner(mesh, parent_cell.value());
  }

  mesh::Mesh<T> mesh1 = mesh::create_mesh(
      mesh.comm(), mesh.comm(), cell_adj.array(), mesh.geometry().cmap(),
      mesh.comm(), new_vertex_coords, xshape, partitioner);

  // Report the number of refined cells
  const int D = topology->dim();
  const std::int64_t n0 = topology->index_map(D)->size_global();
  const std::int64_t n1 = mesh1.topology()->index_map(D)->size_global();
  spdlog::info(
      "Number of cells increased from {} to {} ({}% increase).", n0, n1,
      100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0));

  return {std::move(mesh1), std::move(parent_cell), std::move(parent_facet)};
}

} // namespace dolfinx::refinement
