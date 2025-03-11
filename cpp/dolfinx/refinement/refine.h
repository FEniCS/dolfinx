// Copyright (C) 2010-2024 Garth N. Wells and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "dolfinx/common/MPI.h"
#include "dolfinx/graph/AdjacencyList.h"
#include "dolfinx/mesh/Mesh.h"
#include "dolfinx/mesh/Topology.h"
#include "dolfinx/mesh/cell_types.h"
#include "dolfinx/mesh/graphbuild.h"
#include "dolfinx/mesh/utils.h"
#include "interval.h"
#include "plaza.h"
#include <algorithm>
#include <concepts>
#include <optional>
#include <spdlog/spdlog.h>
#include <utility>

// TODO: Remove once works
namespace dolfinx {
namespace
{
/// @todo Is it un-documented that the owning rank must come first in
/// reach list of edges?
///
/// @param[in] comm The communicator
/// @param[in] graph Graph, using global indices for graph edges
/// @param[in] node_disp The distribution of graph nodes across MPI
/// ranks. The global index `gidx` of local index `lidx` is `lidx +
/// node_disp[my_rank]`.
/// @param[in] part The destination rank for owned nodes, i.e. `dest[i]`
/// is the destination of the node with local index `i`.
/// @return Destination ranks for each local node.
template <typename T>
graph::AdjacencyList<int> compute_destination_ranks(
    MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& graph,
    const std::vector<T>& node_disp, const std::vector<T>& part)
{
  common::Timer timer("Extend graph destination ranks for halo");

  const int rank = dolfinx::MPI::rank(comm);
  const std::int64_t range0 = node_disp[rank];
  const std::int64_t range1 = node_disp[rank + 1];
  assert(static_cast<std::int32_t>(range1 - range0) == graph.num_nodes());

  // Wherever an owned 'node' goes, so must the nodes connected to it by
  // an edge ('node1'). Task is to let the owner of node1 know the extra
  // ranks that it needs to send node1 to.
  std::vector<std::array<std::int64_t, 3>> node_to_dest;
  for (int node0 = 0; node0 < graph.num_nodes(); ++node0)
  {
    // Wherever 'node' goes to, so must the attached 'node1'
    for (auto node1 : graph.links(node0))
    {
      if (node1 < range0 or node1 >= range1)
      {
        auto it = std::ranges::upper_bound(node_disp, node1);
        int remote_rank = std::distance(node_disp.begin(), it) - 1;
        node_to_dest.push_back(
            {remote_rank, node1, static_cast<std::int64_t>(part[node0])});
      }
      else
        node_to_dest.push_back(
            {rank, node1, static_cast<std::int64_t>(part[node0])});
    }
  }

  std::ranges::sort(node_to_dest);
  auto [unique_end, range_end] = std::ranges::unique(node_to_dest);
  node_to_dest.erase(unique_end, range_end);

  // Build send data and buffer
  std::vector<int> dest, send_sizes;
  std::vector<std::int64_t> send_buffer;
  {
    auto it = node_to_dest.begin();
    while (it != node_to_dest.end())
    {
      // Current destination rank
      dest.push_back(it->front());

      // Find iterator to next destination rank and pack send data
      auto it1
          = std::find_if(it, node_to_dest.end(), [r0 = dest.back()](auto& idx)
                         { return idx[0] != r0; });
      send_sizes.push_back(2 * std::distance(it, it1));
      for (auto itx = it; itx != it1; ++itx)
      {
        send_buffer.push_back(itx->at(1));
        send_buffer.push_back(itx->at(2));
      }

      it = it1;
    }
  }

  // Prepare send displacements
  std::vector<int> send_disp(send_sizes.size() + 1, 0);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_disp.begin()));

  // Discover src ranks. ParMETIS/KaHIP are not scalable (holding an
  // array of size equal to the comm size), so no extra harm in using
  // non-scalable neighbourhood detection (which might be faster for
  // small rank counts).
  const std::vector<int> src
      = dolfinx::MPI::compute_graph_edges_pcx(comm, dest);

  // Create neighbourhood communicator
  MPI_Comm neigh_comm;
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm);

  // Determine receives sizes
  std::vector<int> recv_sizes(dest.size());
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                        MPI_INT, neigh_comm);

  // Prepare receive displacements
  std::vector<int> recv_disp(recv_sizes.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_disp.begin()));

  // Send/receive data
  std::vector<std::int64_t> recv_buffer(recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                         send_disp.data(), MPI_INT64_T, recv_buffer.data(),
                         recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                         neigh_comm);
  MPI_Comm_free(&neigh_comm);

  // Prepare (local node index, destination rank) array. Add local data,
  // then add the received data, and the make unique.
  std::vector<std::array<int, 2>> local_node_to_dest;
  for (auto d : part)
  {
    local_node_to_dest.push_back(
        {static_cast<int>(local_node_to_dest.size()), static_cast<int>(d)});
  }
  for (std::size_t i = 0; i < recv_buffer.size(); i += 2)
  {
    std::int64_t idx = recv_buffer[i];
    int d = recv_buffer[i + 1];
    assert(idx >= range0 and idx < range1);
    std::int32_t idx_local = idx - range0;
    local_node_to_dest.push_back({idx_local, d});
  }

  {
    std::ranges::sort(local_node_to_dest);
    auto [unique_end, range_end] = std::ranges::unique(local_node_to_dest);
    local_node_to_dest.erase(unique_end, range_end);
  }
  // Compute offsets
  std::vector<std::int32_t> offsets(graph.num_nodes() + 1, 0);
  {
    std::vector<std::int32_t> num_dests(graph.num_nodes(), 0);
    for (auto x : local_node_to_dest)
      ++num_dests[x[0]];
    std::partial_sum(num_dests.begin(), num_dests.end(),
                     std::next(offsets.begin()));
  }

  // Fill data array
  std::vector<int> data(offsets.back());
  {
    std::vector<std::int32_t> pos = offsets;
    for (auto [x0, x1] : local_node_to_dest)
      data[pos[x0]++] = x1;
  }

  graph::AdjacencyList<int> g(std::move(data), std::move(offsets));

  // Make sure the owning rank comes first for each node
  for (std::int32_t i = 0; i < g.num_nodes(); ++i)
  {
    auto d = g.links(i);
    auto it = std::find(d.begin(), d.end(), part[i]);
    assert(it != d.end());
    std::iter_swap(d.begin(), it);
  }

  return g;
}
} // namespace
}

namespace dolfinx::refinement
{

template <std::floating_point T>
mesh::CellPartitionFunction
create_identity_partitioner(const mesh::Mesh<T>& parent_mesh,
                            std::span<std::int32_t> parent_cell)
{
  // TODO: optimize for non ghosted mesh?

  return
      [&](MPI_Comm comm, int /*nparts*/, std::vector<mesh::CellType> cell_types,
          std::vector<std::span<const std::int64_t>> cells)
          -> graph::AdjacencyList<std::int32_t>
  {
    auto parent_top = parent_mesh.topology();
    auto parent_cell_im = parent_top->index_map(parent_top->dim());

    int num_cell_vertices = mesh::num_cell_vertices(cell_types.front());
    std::int32_t num_cells = cells.front().size() / num_cell_vertices;
    std::vector<std::int32_t> destinations(num_cells);

    std::vector<std::int32_t> dest_offsets(num_cells + 1);
    int rank = dolfinx::MPI::rank(comm);
    for (std::int32_t i = 0; i < destinations.size(); i++)
    {
      bool ghost_parent_cell = parent_cell[i] > parent_cell_im->size_local();
      if (ghost_parent_cell)
      {
        destinations[i]
            = parent_cell_im->owners()[parent_cell[i] - parent_cell_im->size_local()];
      }
      else
      {
        destinations[i] = rank;
      }
    }

    auto dual_graph = mesh::build_dual_graph(comm, cell_types, cells);
    std::vector<std::int32_t> node_disp;
    node_disp = std::vector<std::int32_t>(MPI::size(comm) + 1, 0);
    std::int32_t local_size = cells.front().size();
    MPI_Allgather(&local_size, 1, dolfinx::MPI::mpi_t<std::int32_t>,
                  node_disp.data() + 1, 1, dolfinx::MPI::mpi_t<std::int32_t>,
                  comm);
    std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());

    return compute_destination_ranks(comm, dual_graph, node_disp, destinations);

    // std::iota(dest_offsets.begin(), dest_offsets.end(), 0);
    // return graph::AdjacencyList(std::move(destinations),
    //                             std::move(dest_offsets));
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
