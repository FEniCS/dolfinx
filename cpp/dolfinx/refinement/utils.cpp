// Copyright (C) 2013-2020 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/graphbuild.h>
#include <dolfinx/mesh/topologycomputation.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

using namespace dolfinx;

namespace
{
std::int64_t local_to_global(std::int32_t local_index,
                             const common::IndexMap& map)
{
  assert(local_index >= 0);
  const std::array local_range = map.local_range();
  const std::int32_t local_size = local_range[1] - local_range[0];
  if (local_index < local_size)
  {
    const std::int64_t global_offset = local_range[0];
    return global_offset + local_index;
  }
  else
  {
    const std::vector<std::int64_t>& ghosts = map.ghosts();
    assert((local_index - local_size) < (int)ghosts.size());
    return ghosts[local_index - local_size];
  }
}

//-----------------------------------------------------------------------------
// Create geometric points of new Mesh, from current Mesh and a edge_to_vertex
// map listing the new local points (midpoints of those edges)
// @param Mesh
// @param local_edge_to_new_vertex
// @return array of points
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_new_geometry(
    const mesh::Mesh& mesh,
    const std::map<std::int32_t, std::int64_t>& local_edge_to_new_vertex)
{
  // Build map from vertex -> geometry dof
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const int tdim = mesh.topology().dim();
  auto c_to_v = mesh.topology().connectivity(tdim, 0);
  assert(c_to_v);
  auto map_v = mesh.topology().index_map(0);
  assert(map_v);
  std::vector<std::int32_t> vertex_to_x(map_v->size_local()
                                        + map_v->num_ghosts());
  auto map_c = mesh.topology().index_map(tdim);
  assert(map_c);
  for (int c = 0; c < map_c->size_local() + map_c->num_ghosts(); ++c)
  {
    auto vertices = c_to_v->links(c);
    auto dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
      // FIXME: We are making an assumption here on the
      // ElementDofLayout. We should use an ElementDofLayout to map
      // between local vertex index and x dof index.
      vertex_to_x[vertices[i]] = dofs[i];
    }
  }

  // Copy over existing mesh vertices
  // FIXME: Use eigen map for now.
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      x_g(mesh.geometry().x().data(), mesh.geometry().x().shape[0],
          mesh.geometry().x().shape[1]);

  const std::int32_t num_vertices = map_v->size_local();
  const std::int32_t num_new_vertices = local_edge_to_new_vertex.size();
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
      new_vertex_coordinates(num_vertices + num_new_vertices, 3);

  for (int v = 0; v < num_vertices; ++v)
    new_vertex_coordinates.row(v) = x_g.row(vertex_to_x[v]);

  std::vector<int> edges(num_new_vertices);
  int i = 0;
  for (auto& e : local_edge_to_new_vertex)
    edges[i++] = e.first;

  const auto midpoints = mesh::midpoints(mesh, 1, edges);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
      midpoints_eigen(midpoints.data(), midpoints.shape[0], midpoints.shape[1]);
  new_vertex_coordinates.bottomRows(num_new_vertices) = midpoints_eigen;

  const int gdim = mesh.geometry().dim();
  return new_vertex_coordinates.leftCols(gdim);
}
} // namespace

//---------------------------------------------------------------------------------
std::pair<MPI_Comm, std::map<std::int32_t, std::vector<int>>>
refinement::compute_edge_sharing(const mesh::Mesh& mesh)
{
  if (!mesh.topology().connectivity(1, 0))
    throw std::runtime_error("Edges must be initialised");

  auto map_e = mesh.topology().index_map(1);
  assert(map_e);

  // Create shared edges, for both owned and ghost indices
  // returning edge -> set(global process numbers)
  std::map<std::int32_t, std::set<int>> shared_edges_by_proc
      = map_e->compute_shared_indices();

  // Compute a slightly wider neighborhood for direct communication of shared
  // edges
  std::set<int> all_neighbor_set;
  for (const auto& q : shared_edges_by_proc)
    all_neighbor_set.insert(q.second.begin(), q.second.end());
  std::vector<int> neighbors(all_neighbor_set.begin(), all_neighbor_set.end());

  MPI_Comm neighbor_comm;
  MPI_Dist_graph_create_adjacent(
      mesh.mpi_comm(), neighbors.size(), neighbors.data(), MPI_UNWEIGHTED,
      neighbors.size(), neighbors.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false,
      &neighbor_comm);

  // Create a "shared_edge to neighbor map"
  std::map<int, int> proc_to_neighbor;
  for (std::size_t i = 0; i < neighbors.size(); ++i)
    proc_to_neighbor.insert({neighbors[i], i});

  std::map<std::int32_t, std::vector<int>> shared_edges;
  for (auto& q : shared_edges_by_proc)
  {
    std::vector<int> neighbor_set;
    for (int r : q.second)
      neighbor_set.push_back(proc_to_neighbor[r]);
    std::sort(neighbor_set.begin(), neighbor_set.end());
    neighbor_set.erase(std::unique(neighbor_set.begin(), neighbor_set.end()),
                       neighbor_set.end());
    shared_edges.insert({q.first, neighbor_set});
  }

  return {neighbor_comm, std::move(shared_edges)};
}
//-----------------------------------------------------------------------------
void refinement::update_logical_edgefunction(
    const MPI_Comm& neighbor_comm,
    const std::vector<std::vector<std::int32_t>>& marked_for_update,
    std::vector<bool>& marked_edges, const common::IndexMap& map_e)
{
  std::vector<std::int32_t> send_offsets = {0};
  std::vector<std::int64_t> data_to_send;
  int num_neighbors = marked_for_update.size();
  for (int i = 0; i < num_neighbors; ++i)
  {
    for (std::int32_t q : marked_for_update[i])
      data_to_send.push_back(local_to_global(q, map_e));

    send_offsets.push_back(data_to_send.size());
  }

  // Send all shared edges marked for update and receive from other
  // processes
  const std::vector<std::int64_t> data_to_recv
      = MPI::neighbor_all_to_all(
            neighbor_comm,
            graph::AdjacencyList<std::int64_t>(data_to_send, send_offsets))
            .array();

  // Flatten received values and set marked_edges at each index received
  std::vector<std::int32_t> local_indices = map_e.global_to_local(data_to_recv);
  for (std::int32_t local_index : local_indices)
  {
    assert(local_index != -1);
    marked_edges[local_index] = true;
  }
}
//-----------------------------------------------------------------------------
std::pair<std::map<std::int32_t, std::int64_t>,
          Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
refinement::create_new_vertices(
    const MPI_Comm& neighbor_comm,
    const std::map<std::int32_t, std::vector<std::int32_t>>& shared_edges,
    const mesh::Mesh& mesh, const std::vector<bool>& marked_edges)
{
  // Take marked_edges and use to create new vertices
  const std::shared_ptr<const common::IndexMap> edge_index_map
      = mesh.topology().index_map(1);

  // Add new edge midpoints to list of vertices
  int n = 0;
  std::map<std::int32_t, std::int64_t> local_edge_to_new_vertex;
  for (int local_i = 0; local_i < edge_index_map->size_local(); ++local_i)
  {
    if (marked_edges[local_i] == true)
    {
      auto it = local_edge_to_new_vertex.insert({local_i, n});
      assert(it.second);
      ++n;
    }
  }
  const int num_new_vertices = n;
  const std::size_t global_offset
      = MPI::global_offset(mesh.mpi_comm(), num_new_vertices, true)
        + mesh.topology().index_map(0)->local_range()[1];

  for (auto& e : local_edge_to_new_vertex)
    e.second += global_offset;

  // Create actual points
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      new_vertex_coordinates
      = create_new_geometry(mesh, local_edge_to_new_vertex);

  // If they are shared, then the new global vertex index needs to be
  // sent off-process.

  // Get number of neighbors
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighbor_comm, &indegree, &outdegree,
                                 &weighted);
  assert(indegree == outdegree);
  const int num_neighbors = indegree;

  std::vector<std::vector<std::int64_t>> values_to_send(num_neighbors);
  for (auto& local_edge : local_edge_to_new_vertex)
  {
    const std::size_t local_i = local_edge.first;
    // shared, but locally owned : remote owned are not in list.

    if (auto shared_edge_i = shared_edges.find(local_i);
        shared_edge_i != shared_edges.end())
    {
      for (int remote_process : shared_edge_i->second)
      {
        // send map from global edge index to new global vertex index
        values_to_send[remote_process].push_back(
            local_to_global(local_edge.first, *edge_index_map));
        values_to_send[remote_process].push_back(local_edge.second);
      }
    }
  }

  const std::vector<std::int64_t> received_values
      = MPI::neighbor_all_to_all(
            neighbor_comm, graph::AdjacencyList<std::int64_t>(values_to_send))
            .array();

  // Add received remote global vertex indices to map
  std::vector<std::int64_t> recv_global_edge;
  assert(received_values.size() % 2 == 0);
  for (std::size_t i = 0; i < received_values.size() / 2; ++i)
    recv_global_edge.push_back(received_values[i * 2]);
  const std::vector<std::int32_t> recv_local_edge
      = mesh.topology().index_map(1)->global_to_local(recv_global_edge);
  for (std::size_t i = 0; i < received_values.size() / 2; ++i)
  {
    assert(recv_local_edge[i] != -1);
    auto it = local_edge_to_new_vertex.insert(
        {recv_local_edge[i], received_values[i * 2 + 1]});
    assert(it.second);
  }

  return {std::move(local_edge_to_new_vertex),
          std::move(new_vertex_coordinates)};
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> refinement::adjust_indices(
    const std::shared_ptr<const common::IndexMap>& index_map, std::int32_t n)
{
  // Add in an extra "n" indices at the end of the current local_range
  // of "index_map", and adjust existing indices to match.

  // Get number of new indices on all processes
  int mpi_size = dolfinx::MPI::size(index_map->comm());
  int mpi_rank = dolfinx::MPI::rank(index_map->comm());
  std::vector<std::int32_t> recvn(mpi_size);
  MPI_Allgather(&n, 1, MPI_INT32_T, recvn.data(), 1, MPI_INT32_T,
                index_map->comm());
  std::vector<std::int64_t> global_offsets = {0};
  for (std::int32_t r : recvn)
    global_offsets.push_back(global_offsets.back() + r);

  std::vector global_indices = index_map->global_indices();

  const std::vector<int>& ghost_owners = index_map->ghost_owner_rank();
  int local_size = index_map->size_local();
  for (int i = 0; i < local_size; ++i)
    global_indices[i] += global_offsets[mpi_rank];
  for (std::size_t i = 0; i < ghost_owners.size(); ++i)
    global_indices[local_size + i] += global_offsets[ghost_owners[i]];

  return global_indices;
}
//-----------------------------------------------------------------------------
mesh::Mesh refinement::partition(
    const mesh::Mesh& old_mesh,
    const graph::AdjacencyList<std::int64_t>& cell_topology,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        new_vertex_coordinates,
    bool redistribute, mesh::GhostMode gm)
{

  if (redistribute)
  {
    common::array2d<double> new_coords(new_vertex_coordinates);
    return mesh::create_mesh(old_mesh.mpi_comm(), cell_topology,
                             old_mesh.geometry().cmap(), new_coords, gm);
  }

  auto partitioner = [](MPI_Comm mpi_comm, int, const mesh::CellType cell_type,
                        const graph::AdjacencyList<std::int64_t>& cell_topology,
                        mesh::GhostMode) {
    // Find out the ghosting information
    auto [graph, info]
        = mesh::build_dual_graph(mpi_comm, cell_topology, cell_type);

    // FIXME: much of this is reverse engineering of data that is already
    // known in the GraphBuilder

    const int mpi_size = MPI::size(mpi_comm);
    const int mpi_rank = MPI::rank(mpi_comm);
    const std::int32_t local_size = graph.num_nodes();
    std::vector<std::int32_t> local_sizes(mpi_size);
    std::vector<std::int64_t> local_offsets(mpi_size + 1);

    // Get the "local range" for all processes
    MPI_Allgather(&local_size, 1, MPI_INT32_T, local_sizes.data(), 1,
                  MPI_INT32_T, mpi_comm);
    for (int i = 0; i < mpi_size; ++i)
      local_offsets[i + 1] = local_offsets[i] + local_sizes[i];

    // All cells should go to their currently assigned ranks (no change)
    // but must also be sent to their ghost destinations, which are determined
    // here.
    std::vector<std::int32_t> destinations;
    destinations.reserve(graph.num_nodes());
    std::vector<std::int32_t> dest_offsets = {0};
    dest_offsets.reserve(graph.num_nodes());
    for (int i = 0; i < graph.num_nodes(); ++i)
    {
      destinations.push_back(mpi_rank);
      for (int j = 0; j < graph.num_links(i); ++j)
      {
        std::int64_t index = graph.links(i)[j];
        if (index < local_offsets[mpi_rank]
            or index >= local_offsets[mpi_rank + 1])
        {
          // Ghosted cell - identify which process it should be sent to.
          for (std::size_t k = 0; k < local_offsets.size(); ++k)
          {
            if (index >= local_offsets[k] and index < local_offsets[k + 1])
            {
              destinations.push_back(k);
              break;
            }
          }
        }
      }
      dest_offsets.push_back(destinations.size());
    }

    return graph::AdjacencyList<std::int32_t>(std::move(destinations),
                                              std::move(dest_offsets));
  };

  common::array2d<double> new_coords(new_vertex_coordinates);

  return mesh::create_mesh(old_mesh.mpi_comm(), cell_topology,
                           old_mesh.geometry().cmap(), new_coords, gm,
                           partitioner);
}
//-----------------------------------------------------------------------------
