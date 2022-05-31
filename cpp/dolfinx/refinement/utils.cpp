// Copyright (C) 2013-2022 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
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
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using namespace xt::placeholders;

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

/// Create geometric points of new Mesh, from current Mesh and a
/// edge_to_vertex map listing the new local points (midpoints of those
/// edges)
/// @param Mesh
/// @param local_edge_to_new_vertex
/// @return array of points
xt::xtensor<double, 2> create_new_geometry(
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
  auto dof_layout = mesh.geometry().cmap().create_dof_layout();
  auto entity_dofs_all = dof_layout.entity_dofs_all();
  for (int c = 0; c < map_c->size_local() + map_c->num_ghosts(); ++c)
  {
    auto vertices = c_to_v->links(c);
    auto dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
      auto vertex_pos = entity_dofs_all[0][i][0];
      vertex_to_x[vertices[i]] = dofs[vertex_pos];
    }
  }

  // Copy over existing mesh vertices
  xtl::span<const double> x_g = mesh.geometry().x();

  const std::size_t num_vertices = map_v->size_local();
  const std::size_t num_new_vertices = local_edge_to_new_vertex.size();
  xt::xtensor<double, 2> new_vertex_coordinates(
      {num_vertices + num_new_vertices, 3});

  for (std::size_t v = 0; v < num_vertices; ++v)
  {
    const int pos = 3 * vertex_to_x[v];
    for (std::size_t j = 0; j < 3; ++j)
      new_vertex_coordinates(v, j) = x_g[pos + j];
  }

  // Compute new vertices
  if (num_new_vertices > 0)
  {
    std::vector<int> edges(num_new_vertices);
    int i = 0;
    for (auto& e : local_edge_to_new_vertex)
      edges[i++] = e.first;

    const std::vector<double> midpoints
        = mesh::compute_midpoints(mesh, 1, edges);

    std::vector<std::size_t> shape = {edges.size(), 3};
    auto _midpoints = xt::adapt(midpoints, shape);

    // The below should work, but misbehaves with the Intel icpx compiler
    // xt::view(new_vertex_coordinates, xt::range(-num_new_vertices, _),
    // xt::all())
    //     = midpoints;
    auto _vertex = xt::view(new_vertex_coordinates,
                            xt::range(-num_new_vertices, _), xt::all());
    _vertex.assign(_midpoints);
  }

  return xt::view(new_vertex_coordinates, xt::all(),
                  xt::range(0, mesh.geometry().dim()));
}
} // namespace

//---------------------------------------------------------------------------------
void refinement::update_logical_edgefunction(
    MPI_Comm neighbor_comm,
    const std::vector<std::vector<std::int32_t>>& marked_for_update,
    std::vector<std::int8_t>& marked_edges, const common::IndexMap& map)
{
  std::vector<int> send_sizes;
  std::vector<std::int64_t> data_to_send;
  for (std::size_t i = 0; i < marked_for_update.size(); ++i)
  {
    for (std::int32_t q : marked_for_update[i])
      data_to_send.push_back(local_to_global(q, map));

    send_sizes.push_back(marked_for_update[i].size());
  }

  // Send all shared edges marked for update and receive from other
  // processes
  std::vector<std::int64_t> data_to_recv;
  {
    int indegree(-1), outdegree(-2), weighted(-1);
    MPI_Dist_graph_neighbors_count(neighbor_comm, &indegree, &outdegree,
                                   &weighted);
    assert(indegree == (int)marked_for_update.size());
    assert(indegree == outdegree);

    std::vector<int> recv_sizes(outdegree);
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                          MPI_INT, neighbor_comm);

    // Build displacements
    std::vector<int> send_disp = {0};
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::back_inserter(send_disp));
    std::vector<int> recv_disp = {0};
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::back_inserter(recv_disp));

    data_to_recv.resize(recv_disp.back());
    MPI_Neighbor_alltoallv(data_to_send.data(), send_sizes.data(),
                           send_disp.data(), MPI_INT64_T, data_to_recv.data(),
                           recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                           neighbor_comm);
  }

  // Flatten received values and set marked_edges at each index received
  std::vector<std::int32_t> local_indices(data_to_recv.size());
  map.global_to_local(data_to_recv, local_indices);
  for (std::int32_t local_index : local_indices)
  {
    assert(local_index != -1);
    marked_edges[local_index] = true;
  }
}
//-----------------------------------------------------------------------------
std::pair<std::map<std::int32_t, std::int64_t>, xt::xtensor<double, 2>>
refinement::create_new_vertices(MPI_Comm neighbor_comm,
                                const graph::AdjacencyList<int>& shared_edges,
                                const mesh::Mesh& mesh,
                                const std::vector<std::int8_t>& marked_edges)
{
  // Take marked_edges and use to create new vertices
  std::shared_ptr<const common::IndexMap> edge_index_map
      = mesh.topology().index_map(1);

  // Add new edge midpoints to list of vertices
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
  MPI_Exscan(&num_local, &global_offset, 1, MPI_INT64_T, MPI_SUM, mesh.comm());
  global_offset += mesh.topology().index_map(0)->local_range()[1];
  std::for_each(local_edge_to_new_vertex.begin(),
                local_edge_to_new_vertex.end(),
                [global_offset](auto& e) { e.second += global_offset; });

  // Create actual points
  xt::xtensor<double, 2> new_vertex_coordinates
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

    for (int remote_process : shared_edges.links(local_i))
    {
      // Send (global edge index) -> (new global vertex index) map
      values_to_send[remote_process].push_back(
          local_to_global(local_i, *edge_index_map));
      values_to_send[remote_process].push_back(local_edge.second);
    }
  }

  // Send all shared edges marked for update and receive from other
  // processes
  std::vector<std::int64_t> received_values;
  {
    int indegree(-1), outdegree(-2), weighted(-1);
    MPI_Dist_graph_neighbors_count(neighbor_comm, &indegree, &outdegree,
                                   &weighted);
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
                          MPI_INT, neighbor_comm);

    // Build displacements
    std::vector<int> send_disp = {0};
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::back_inserter(send_disp));
    std::vector<int> recv_disp = {0};
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::back_inserter(recv_disp));

    received_values.resize(recv_disp.back());
    MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                           send_disp.data(), MPI_INT64_T,
                           received_values.data(), recv_sizes.data(),
                           recv_disp.data(), MPI_INT64_T, neighbor_comm);
  }

  // Add received remote global vertex indices to map
  std::vector<std::int64_t> recv_global_edge;
  assert(received_values.size() % 2 == 0);
  for (std::size_t i = 0; i < received_values.size() / 2; ++i)
    recv_global_edge.push_back(received_values[i * 2]);
  std::vector<std::int32_t> recv_local_edge(recv_global_edge.size());
  mesh.topology().index_map(1)->global_to_local(recv_global_edge,
                                                recv_local_edge);
  for (std::size_t i = 0; i < received_values.size() / 2; ++i)
  {
    assert(recv_local_edge[i] != -1);
    [[maybe_unused]] auto it = local_edge_to_new_vertex.insert(
        {recv_local_edge[i], received_values[i * 2 + 1]});
    assert(it.second);
  }

  return {std::move(local_edge_to_new_vertex),
          std::move(new_vertex_coordinates)};
}
//-----------------------------------------------------------------------------
mesh::Mesh
refinement::partition(const mesh::Mesh& old_mesh,
                      const graph::AdjacencyList<std::int64_t>& cell_topology,
                      const xt::xtensor<double, 2>& new_vertex_coordinates,
                      bool redistribute, mesh::GhostMode gm)
{
  if (redistribute)
  {
    xt::xtensor<double, 2> new_coords(new_vertex_coordinates);
    return mesh::create_mesh(old_mesh.comm(), cell_topology,
                             old_mesh.geometry().cmap(), new_coords, gm);
  }

  auto partitioner = [](MPI_Comm comm, int, int tdim,
                        const graph::AdjacencyList<std::int64_t>& cell_topology,
                        mesh::GhostMode)
  {
    // Find out the ghosting information
    graph::AdjacencyList<std::int64_t> graph
        = mesh::build_dual_graph(comm, cell_topology, tdim);

    // FIXME: much of this is reverse engineering of data that is already
    // known in the GraphBuilder

    const int mpi_size = dolfinx::MPI::size(comm);
    const int mpi_rank = dolfinx::MPI::rank(comm);
    const std::int32_t local_size = graph.num_nodes();
    std::vector<std::int32_t> local_sizes(mpi_size);

    // Get the "local range" for all processes
    MPI_Allgather(&local_size, 1, MPI_INT32_T, local_sizes.data(), 1,
                  MPI_INT32_T, comm);

    std::vector<std::int64_t> local_offsets(mpi_size + 1);
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

  return mesh::create_mesh(old_mesh.comm(), cell_topology,
                           old_mesh.geometry().cmap(), new_vertex_coordinates,
                           gm, partitioner);
}
//-----------------------------------------------------------------------------

std::vector<std::int64_t>
refinement::adjust_indices(const common::IndexMap& map, std::int32_t n)
{
  // NOTE: Is this effectively concatenating index maps?

  // Add in an extra "n" indices at the end of the current local_range
  // of "index_map", and adjust existing indices to match.

  // Get offset for 'n' for this process
  const std::int64_t num_local = n;
  std::int64_t global_offset = 0;
  MPI_Exscan(&num_local, &global_offset, 1, MPI_INT64_T, MPI_SUM, map.comm());

  const std::vector<int>& owners = map.owners();
  const std::vector<int>& src = map.src();
  const std::vector<int>& dest = map.dest();

  MPI_Comm comm;
  MPI_Dist_graph_create_adjacent(map.comm(), src.size(), src.data(),
                                 MPI_UNWEIGHTED, dest.size(), dest.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

  // Communicate offset to neighbors
  std::vector<std::int64_t> offsets(src.size(), 0);
  offsets.reserve(1);
  MPI_Neighbor_allgather(&global_offset, 1, MPI_INT64_T, offsets.data(), 1,
                         MPI_INT64_T, comm);

  MPI_Comm_free(&comm);

  int local_size = map.size_local();
  std::vector<std::int64_t> global_indices = map.global_indices();

  // Add new offset to owned indices
  std::transform(global_indices.begin(),
                 std::next(global_indices.begin(), local_size),
                 global_indices.begin(),
                 [global_offset](auto x) { return x + global_offset; });

  // Add offsets to ghost indices
  std::transform(std::next(global_indices.begin(), local_size),
                 global_indices.end(), owners.begin(),
                 std::next(global_indices.begin(), local_size),
                 [&src, &offsets](auto idx, auto r)
                 {
                   auto it = std::lower_bound(src.begin(), src.end(), r);
                   assert(it != src.end() and *it == r);
                   int rank = std::distance(src.begin(), it);
                   return idx + offsets[rank];
                 });

  return global_indices;
}
//-----------------------------------------------------------------------------
mesh::MeshTags<std::int32_t> refinement::transfer_facet_meshtag(
    const mesh::MeshTags<std::int32_t>& parent_meshtag,
    const mesh::Mesh& refined_mesh,
    const std::vector<std::int32_t>& parent_cell,
    const std::vector<std::int8_t>& parent_facet)
{
  const int tdim = parent_meshtag.mesh()->topology().dim();
  if (parent_meshtag.dim() != tdim - 1)
    throw std::runtime_error("Input meshtag is not facet-based");

  if (parent_meshtag.mesh()->topology().index_map(tdim)->num_ghosts() > 0)
    throw std::runtime_error("Ghosted meshes are not supported");

  auto parent_c_to_f
      = parent_meshtag.mesh()->topology().connectivity(tdim, tdim - 1);
  auto c_to_f = refined_mesh.topology().connectivity(tdim, tdim - 1);

  // Create map parent->child facets
  const std::int32_t num_input_facets
      = parent_meshtag.mesh()->topology().index_map(tdim - 1)->size_local()
        + parent_meshtag.mesh()->topology().index_map(tdim - 1)->num_ghosts();
  std::vector<int> count_child(num_input_facets, 0);

  // Get global index for each refined cell, before reordering in Mesh
  // construction
  const std::vector<std::int64_t>& original_cell_index
      = refined_mesh.topology().original_cell_index;
  assert(original_cell_index.size() == parent_cell.size());
  std::int64_t global_offset
      = refined_mesh.topology().index_map(tdim)->local_range()[0];
  // Map cells back to original index
  std::vector<std::int32_t> local_cell_index(original_cell_index.size());
  for (std::size_t i = 0; i < local_cell_index.size(); ++i)
  {
    assert(original_cell_index[i] >= global_offset);
    assert(original_cell_index[i] - global_offset
           < (int)local_cell_index.size());
    local_cell_index[original_cell_index[i] - global_offset] = i;
  }

  // Count number of child facets for each parent facet
  for (std::size_t c = 0; c < parent_cell.size(); ++c)
  {
    auto parent_cf = parent_c_to_f->links(parent_cell[c]);

    for (int j = 0; j < (tdim + 1); ++j)
    {
      std::int8_t fidx = parent_facet[c * (tdim + 1) + j];
      if (fidx != -1)
        ++count_child[parent_cf[fidx]];
    }
  }

  std::vector<int> offset_child(num_input_facets + 1, 0);
  std::partial_sum(count_child.begin(), count_child.end(),
                   std::next(offset_child.begin()));
  std::vector<std::int32_t> child_facet(offset_child.back());

  // Fill in data for each child facet
  for (std::size_t c = 0; c < parent_cell.size(); ++c)
  {
    std::int32_t pc = parent_cell[c];
    auto parent_cf = parent_c_to_f->links(pc);

    // Use original indexing for child cell
    const std::int32_t lc = local_cell_index[c];
    auto refined_cf = c_to_f->links(lc);

    // Get child facets for each cell
    for (int j = 0; j < (tdim + 1); ++j)
    {
      std::int8_t fidx = parent_facet[c * (tdim + 1) + j];
      if (fidx != -1)
      {
        int offset = offset_child[parent_cf[fidx]];
        child_facet[offset] = refined_cf[j];
        ++offset_child[parent_cf[fidx]];
      }
    }
  }

  // Rebuild offset
  offset_child[0] = 0;
  std::partial_sum(count_child.begin(), count_child.end(),
                   std::next(offset_child.begin()));
  graph::AdjacencyList<std::int32_t> p_to_c_facet(std::move(child_facet),
                                                  std::move(offset_child));

  // Copy facet meshtag from parent to child
  std::vector<std::int32_t> facet_indices;
  std::vector<std::int32_t> tag_values;
  const std::vector<std::int32_t>& in_index = parent_meshtag.indices();
  const std::vector<std::int32_t>& in_value = parent_meshtag.values();
  for (std::size_t i = 0; i < in_index.size(); ++i)
  {
    std::int32_t parent_index = in_index[i];
    auto pclinks = p_to_c_facet.links(parent_index);
    // eliminate duplicates
    std::sort(pclinks.begin(), pclinks.end());
    auto it_end = std::unique(pclinks.begin(), pclinks.end());
    for (auto child_it = pclinks.begin(); child_it != it_end; ++child_it)
    {
      facet_indices.push_back(*child_it);
      tag_values.push_back(in_value[i]);
    }
  }

  // Sort values into order, based on facet indices
  std::vector<std::int32_t> sort_order(tag_values.size());
  std::iota(sort_order.begin(), sort_order.end(), 0);
  std::sort(sort_order.begin(), sort_order.end(),
            [&facet_indices](auto a, auto b)
            { return facet_indices[a] < facet_indices[b]; });
  std::vector<std::int32_t> sorted_facet_indices(facet_indices.size());
  std::vector<std::int32_t> sorted_tag_values(tag_values.size());
  for (std::size_t i = 0; i < sort_order.size(); ++i)
  {
    sorted_tag_values[i] = tag_values[sort_order[i]];
    sorted_facet_indices[i] = facet_indices[sort_order[i]];
  }

  return mesh::MeshTags<std::int32_t>(
      std::make_shared<mesh::Mesh>(refined_mesh), tdim - 1,
      std::move(sorted_facet_indices), std::move(sorted_tag_values));
}
//----------------------------------------------------------------------------
mesh::MeshTags<std::int32_t> refinement::transfer_cell_meshtag(
    const mesh::MeshTags<std::int32_t>& parent_meshtag,
    const mesh::Mesh& refined_mesh,
    const std::vector<std::int32_t>& parent_cell)
{
  const int tdim = parent_meshtag.mesh()->topology().dim();
  if (parent_meshtag.dim() != tdim)
    throw std::runtime_error("Input meshtag is not cell-based");

  if (parent_meshtag.mesh()->topology().index_map(tdim)->num_ghosts() > 0)
    throw std::runtime_error("Ghosted meshes are not supported");

  // Create map parent->child facets
  const std::int32_t num_input_cells
      = parent_meshtag.mesh()->topology().index_map(tdim)->size_local()
        + parent_meshtag.mesh()->topology().index_map(tdim)->num_ghosts();
  std::vector<int> count_child(num_input_cells, 0);

  // Get global index for each refined cell, before reordering in Mesh
  // construction
  const std::vector<std::int64_t>& original_cell_index
      = refined_mesh.topology().original_cell_index;
  assert(original_cell_index.size() == parent_cell.size());
  std::int64_t global_offset
      = refined_mesh.topology().index_map(tdim)->local_range()[0];
  // Map back to original index
  std::vector<std::int32_t> local_cell_index(original_cell_index.size());
  for (std::size_t i = 0; i < local_cell_index.size(); ++i)
  {
    assert(original_cell_index[i] >= global_offset);
    assert(original_cell_index[i] - global_offset
           < (int)local_cell_index.size());
    local_cell_index[original_cell_index[i] - global_offset] = i;
  }

  // Count number of child cells for each parent cell
  for (std::int32_t pcell : parent_cell)
    ++count_child[pcell];

  std::vector<int> offset_child(num_input_cells + 1, 0);
  std::partial_sum(count_child.begin(), count_child.end(),
                   std::next(offset_child.begin()));
  std::vector<std::int32_t> child_cell(offset_child.back());

  // Fill in data for each child cell
  for (std::size_t c = 0; c < parent_cell.size(); ++c)
  {
    std::int32_t pc = parent_cell[c];
    int offset = offset_child[pc];
    // Use original indexing for child cell
    const std::int32_t lc = local_cell_index[c];
    child_cell[offset] = lc;
    ++offset_child[pc];
  }

  // Rebuild offset
  offset_child[0] = 0;
  std::partial_sum(count_child.begin(), count_child.end(),
                   std::next(offset_child.begin()));
  graph::AdjacencyList<std::int32_t> p_to_c_cell(std::move(child_cell),
                                                 std::move(offset_child));

  // Copy cell meshtag from parent to child
  std::vector<std::int32_t> cell_indices;
  std::vector<std::int32_t> tag_values;
  const std::vector<std::int32_t>& in_index = parent_meshtag.indices();
  const std::vector<std::int32_t>& in_value = parent_meshtag.values();
  for (std::size_t i = 0; i < in_index.size(); ++i)
  {
    std::int32_t parent_index = in_index[i];
    auto pclinks = p_to_c_cell.links(parent_index);
    for (std::int32_t child : pclinks)
    {
      cell_indices.push_back(child);
      tag_values.push_back(in_value[i]);
    }
  }

  // Sort values into order, based on cell indices
  std::vector<std::int32_t> sort_order(tag_values.size());
  std::iota(sort_order.begin(), sort_order.end(), 0);
  std::sort(sort_order.begin(), sort_order.end(),
            [&cell_indices](auto a, auto b)
            { return cell_indices[a] < cell_indices[b]; });
  std::vector<std::int32_t> sorted_tag_values(tag_values.size());
  std::vector<std::int32_t> sorted_cell_indices(cell_indices.size());
  for (std::size_t i = 0; i < sort_order.size(); ++i)
  {
    sorted_tag_values[i] = tag_values[sort_order[i]];
    sorted_cell_indices[i] = cell_indices[sort_order[i]];
  }

  return mesh::MeshTags<std::int32_t>(
      std::make_shared<mesh::Mesh>(refined_mesh), tdim,
      std::move(sorted_cell_indices), std::move(sorted_tag_values));
}
