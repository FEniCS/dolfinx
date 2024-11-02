// Copyright (C) 2012-2020 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <tuple>
#include <vector>

namespace dolfinx::mesh
{
template <typename T>
class MeshTags;
class Topology;
enum class GhostMode;
} // namespace dolfinx::mesh

namespace dolfinx::common
{
class IndexMap;
} // namespace dolfinx::common

namespace dolfinx::refinement
{
namespace impl
{
/// @brief  Compute global index
std::int64_t local_to_global(std::int32_t local_index,
                             const common::IndexMap& map);

/// @brief Create geometric points of new Mesh, from current Mesh and a
/// edge_to_vertex map listing the new local points (midpoints of those
/// edges).
///
/// @param mesh Current mesh.
/// @param local_edge_to_new_vertex A map from a local edge to the new
/// global vertex index that will be inserted at its midpoint.
/// @return (1) Array of new (flattened) mesh geometry and (2) its
/// multi-dimensional shape.
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>> create_new_geometry(
    const mesh::Mesh<T>& mesh,
    const std::map<std::int32_t, std::int64_t>& local_edge_to_new_vertex)
{
  // Build map from vertex -> geometry dof
  auto x_dofmap = mesh.geometry().dofmap();
  const int tdim = mesh.topology()->dim();
  auto c_to_v = mesh.topology()->connectivity(tdim, 0);
  assert(c_to_v);
  auto map_v = mesh.topology()->index_map(0);
  assert(map_v);
  std::vector<std::int32_t> vertex_to_x(map_v->size_local()
                                        + map_v->num_ghosts());
  auto map_c = mesh.topology()->index_map(tdim);

  assert(map_c);
  auto dof_layout = mesh.geometry().cmap().create_dof_layout();
  auto entity_dofs_all = dof_layout.entity_dofs_all();
  for (int c = 0; c < map_c->size_local() + map_c->num_ghosts(); ++c)
  {
    auto vertices = c_to_v->links(c);
    auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
      auto vertex_pos = entity_dofs_all[0][i][0];
      vertex_to_x[vertices[i]] = dofs[vertex_pos];
    }
  }

  // Copy over existing mesh vertices
  std::span<const T> x_g = mesh.geometry().x();
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t num_vertices = map_v->size_local();
  const std::size_t num_new_vertices = local_edge_to_new_vertex.size();

  std::array<std::size_t, 2> shape = {num_vertices + num_new_vertices, gdim};
  std::vector<T> new_vertex_coords(shape[0] * shape[1]);
  for (std::size_t v = 0; v < num_vertices; ++v)
  {
    std::size_t pos = 3 * vertex_to_x[v];
    for (std::size_t j = 0; j < gdim; ++j)
      new_vertex_coords[gdim * v + j] = x_g[pos + j];
  }

  // Compute new vertices
  if (num_new_vertices > 0)
  {
    std::vector<int> edges(num_new_vertices);
    int i = 0;
    for (auto& e : local_edge_to_new_vertex)
      edges[i++] = e.first;

    // Compute midpoint of each edge (padded to 3D)
    const std::vector<T> midpoints = mesh::compute_midpoints(mesh, 1, edges);
    for (std::size_t i = 0; i < num_new_vertices; ++i)
      for (std::size_t j = 0; j < gdim; ++j)
        new_vertex_coords[gdim * (num_vertices + i) + j] = midpoints[3 * i + j];
  }

  return {std::move(new_vertex_coords), shape};
}

} // namespace impl

/// @brief Communicate edge markers between processes that share edges.
///
/// @param[in] comm MPI Communicator for neighborhood
/// @param[in] marked_for_update Lists of edges to be updated on each
/// neighbor. `marked_for_update[r]` is the list of edge indices that
/// are marked by the caller and are shared with local MPI rank `r`.
/// @param[in, out] marked_edges Marker for each edge on the calling
/// process
/// @param[in] map Index map for the mesh edges
void update_logical_edgefunction(
    MPI_Comm comm,
    const std::vector<std::vector<std::int32_t>>& marked_for_update,
    std::span<std::int8_t> marked_edges, const common::IndexMap& map);

/// @brief Add new vertex for each marked edge, and create
/// new_vertex_coordinates and global_edge->new_vertex map.
///
/// Communicate new vertices with MPI to all affected processes.
///
/// @param[in] comm MPI Communicator for neighborhood
/// @param[in] shared_edges For each local edge on a process map to ghost
/// processes
/// @param[in] mesh Existing mesh
/// @param[in] marked_edges A marker for all edges on the process (local +
/// ghosts) indicating if an edge should be refined
/// @return (0) map from local edge index to new vertex global index, (1) the
/// coordinates of the new vertices (row-major storage) and (2) the shape of the
/// new coordinates.
template <std::floating_point T>
std::tuple<std::map<std::int32_t, std::int64_t>, std::vector<T>,
           std::array<std::size_t, 2>>
create_new_vertices(MPI_Comm comm,
                    const graph::AdjacencyList<int>& shared_edges,
                    const mesh::Mesh<T>& mesh,
                    std::span<const std::int8_t> marked_edges)
{
  // Take marked_edges and use to create new vertices
  std::shared_ptr<const common::IndexMap> edge_index_map
      = mesh.topology()->index_map(1);

  // Add new edge midpoints to list of vertices. The new vertex will be owned by
  // the process owning the edge.
  int n = 0;
  std::map<std::int32_t, std::int64_t> local_edge_to_new_vertex;
  for (int local_i = 0; local_i < edge_index_map->size_local(); ++local_i)
  {
    if (marked_edges[local_i])
    {
      [[maybe_unused]] auto it = local_edge_to_new_vertex.insert({local_i, n});
      assert(it.second);
      ++n;
    }
  }

  const std::int64_t num_local = n;
  std::int64_t global_offset = 0;
  MPI_Exscan(&num_local, &global_offset, 1, dolfinx::MPI::mpi_t<std::int64_t>,
             MPI_SUM, mesh.comm());
  global_offset += mesh.topology()->index_map(0)->local_range()[1];
  std::for_each(local_edge_to_new_vertex.begin(),
                local_edge_to_new_vertex.end(),
                [global_offset](auto& e) { e.second += global_offset; });

  // Create actual points
  auto [new_vertex_coords, xshape]
      = impl::create_new_geometry(mesh, local_edge_to_new_vertex);

  // If they are shared, then the new global vertex index needs to be
  // sent off-process.

  // Get number of neighbors
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);
  assert(indegree == outdegree);
  const int num_neighbors = indegree;

  std::vector<std::vector<std::int64_t>> values_to_send(num_neighbors);
  for (auto& local_edge : local_edge_to_new_vertex)
  {
    const std::size_t local_i = local_edge.first;
    // shared, but locally owned : remote owned are not in list.

    for (int remote_process : shared_edges.links(local_i))
    {
      // Send (global edge index) -> (new global vertex index) map
      values_to_send[remote_process].push_back(
          impl::local_to_global(local_i, *edge_index_map));
      values_to_send[remote_process].push_back(local_edge.second);
    }
  }

  // Send all shared edges marked for update and receive from other
  // processes
  std::vector<std::int64_t> received_values;
  {
    int indegree(-1), outdegree(-2), weighted(-1);
    MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);
    assert(indegree == outdegree);

    std::vector<std::int64_t> send_buffer;
    std::vector<int> send_sizes;
    for (auto& x : values_to_send)
    {
      send_sizes.push_back(x.size());
      send_buffer.insert(send_buffer.end(), x.begin(), x.end());
    }
    assert((int)send_sizes.size() == outdegree);

    std::vector<int> recv_sizes(outdegree);
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                          MPI_INT, comm);

    // Build displacements
    std::vector<int> send_disp = {0};
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::back_inserter(send_disp));
    std::vector<int> recv_disp = {0};
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::back_inserter(recv_disp));

    received_values.resize(recv_disp.back());
    MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                           send_disp.data(), dolfinx::MPI::mpi_t<std::int64_t>,
                           received_values.data(), recv_sizes.data(),
                           recv_disp.data(), dolfinx::MPI::mpi_t<std::int64_t>,
                           comm);
  }

  // Add received remote global vertex indices to map
  std::vector<std::int64_t> recv_global_edge;
  assert(received_values.size() % 2 == 0);
  for (std::size_t i = 0; i < received_values.size() / 2; ++i)
    recv_global_edge.push_back(received_values[i * 2]);
  std::vector<std::int32_t> recv_local_edge(recv_global_edge.size());
  mesh.topology()->index_map(1)->global_to_local(recv_global_edge,
                                                 recv_local_edge);
  for (std::size_t i = 0; i < received_values.size() / 2; ++i)
  {
    assert(recv_local_edge[i] != -1);
    [[maybe_unused]] auto it = local_edge_to_new_vertex.insert(
        {recv_local_edge[i], received_values[i * 2 + 1]});
    assert(it.second);
  }

  return {std::move(local_edge_to_new_vertex), std::move(new_vertex_coords),
          xshape};
}

/// @todo Improve docstring.
///
/// @brief Given an index map, add "n" extra indices at the end of local
/// range.
///
/// @note The returned global indices (local and ghosts) are adjust for
/// the addition of new local indices.
///
/// @param[in] map Index map.
/// @param[in] n Number of new local indices.
/// @return New global indices for both owned and ghosted indices in
/// input index map.
std::vector<std::int64_t> adjust_indices(const common::IndexMap& map,
                                         std::int32_t n);

/// @brief Transfer facet MeshTags from coarse mesh to refined mesh.
///
/// @warning The refined mesh must not have been redistributed during
/// refinement.
///
/// @warning Mesh/topology `GhostMode` must be mesh::GhostMode::none.
///
/// @param[in] tags0 Tags on the parent mesh.
/// @param[in] topology1 Refined mesh topology.
/// @param[in] cell Parent cell of each cell in refined mesh
/// @param[in] facet Local facets of parent in each cell in refined mesh.
/// @return (0) entities and (1) values on the refined topology.
std::array<std::vector<std::int32_t>, 2> transfer_facet_meshtag(
    const mesh::MeshTags<std::int32_t>& tags0, const mesh::Topology& topology1,
    std::span<const std::int32_t> cell, std::span<const std::int8_t> facet);

/// @brief Transfer cell MeshTags from coarse mesh to refined mesh.
///
/// @warning The refined mesh must not have been redistributed during
/// refinement.
///
/// @warning Mesh/topology `GhostMode` must be mesh::GhostMode::none.
///
/// @param[in] tags0 Tags on the parent mesh.
/// @param[in] topology1 Refined mesh topology.
/// @param[in] parent_cell Parent cell of each cell in refined mesh.
/// @return (0) entities and (1) values on the refined topology.
std::array<std::vector<std::int32_t>, 2>
transfer_cell_meshtag(const mesh::MeshTags<std::int32_t>& tags0,
                      const mesh::Topology& topology1,
                      std::span<const std::int32_t> parent_cell);

} // namespace dolfinx::refinement
