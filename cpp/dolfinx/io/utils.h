// Copyright (C) 2012-2024 Chris N. Richardson, Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <basix/mdspan.hpp>
#include <dolfinx/common/types.h>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace fem
{
class ElementDofLayout;
} // namespace fem

namespace mesh
{
class Topology;
} // namespace mesh

namespace io
{

/// @brief Get owned entities and associated data from input entities
/// defined by global 'node' indices.
///
/// The input entities and data can be supplied on any rank and this
/// function will manage the communication.
///
/// @param[in] topology A mesh topology.
/// @param[in] nodes_g Global 'input' indices for the mesh, as returned
/// by Geometry::input_global_indices.
/// @param[in] num_nodes_g Global number of geometry nodes, as returned
/// by `Geometry::index_map()->size_global()`.
/// @param[in] cmap_dof_layout Coordinate element dof layout, computed
/// using `Geometry::cmap().create_dof_layout()`.
/// @param[in] xdofmap Dofmap for the mesh geometry (Geometry::dofmap).
/// @param[in] entity_dim Topological dimension of entities to extract.
/// @param[in] entities Mesh entities defined using global input indices
/// ('nodes'), typically from an input mesh file, e.g. [gi0, gi1, gi2]
/// for a triangle. Let [v0, v1, v2] be the vertex indices of some
/// triangle (using local indexing). Each vertex has a 'node' (geometry
/// dof) index, and each node has a persistent input global index, so
/// the triangle [gi0, gi1, gi2] could be identified with [v0, v1, v2].
/// The data is flattened and the shape is `(num_entities,
/// nodes_per_entity)`.
/// @param[in] data Data associated with each entity in `entities`.
/// @return (entity-vertex connectivity of owned entities, associated
/// data (values) with each entity).
///
/// @note This function involves parallel distribution and must be
/// called collectively. Global input indices for entities which are not
/// owned by current rank could be passed to this function. E.g., rank0
/// provides an entity with global input indices [gi0, gi1, gi2], but
/// this identifies a triangle that is owned by rank1. It will be
/// distributed and rank1 will receive the (local) cell-vertex
/// connectivity for this triangle.
template <typename T>
std::pair<std::vector<std::int32_t>, std::vector<T>> distribute_entity_data(
    const mesh::Topology& topology, std::span<const std::int64_t> nodes_g,
    std::int64_t num_nodes_g, const fem::ElementDofLayout& cmap_dof_layout,
    md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>> xdofmap,
    int entity_dim,
    md::mdspan<const std::int64_t, md::dextents<std::size_t, 2>> entities,
    std::span<const T> data)
{
  assert(entities.extent(0) == data.size());

  spdlog::info("XDMF distribute entity data");
  mesh::CellType cell_type = topology.cell_type();

  // Get layout of dofs on 0th cell entity of dimension entity_dim
  std::vector<int> cell_vertex_dofs;
  for (int i = 0; i < mesh::cell_num_entities(cell_type, 0); ++i)
  {
    const std::vector<int>& local_index = cmap_dof_layout.entity_dofs(0, i);
    assert(local_index.size() == 1);
    cell_vertex_dofs.push_back(local_index[0]);
  }

  // -- A. Convert from list of entities by 'nodes' to list of entities
  // by 'vertex nodes'
  auto to_vertex_entities
      = [](const fem::ElementDofLayout& cmap_dof_layout, int entity_dim,
           std::span<const int> cell_vertex_dofs, mesh::CellType cell_type,
           auto entities)
  {
    // Use ElementDofLayout of the cell to get vertex dof indices (local
    // to a cell), i.e. build a map from local vertex index to associated
    // local dof index
    const std::vector<int> entity_layout
        = cmap_dof_layout.entity_closure_dofs(entity_dim, 0);
    std::vector<int> entity_vertex_dofs;
    for (std::size_t i = 0; i < cell_vertex_dofs.size(); ++i)
    {
      auto it = std::find(entity_layout.begin(), entity_layout.end(),
                          cell_vertex_dofs[i]);
      if (it != entity_layout.end())
        entity_vertex_dofs.push_back(std::distance(entity_layout.begin(), it));
    }

    const std::size_t num_vert_per_e = mesh::cell_num_entities(
        mesh::cell_entity_type(cell_type, entity_dim, 0), 0);

    assert(entities.extent(1) == entity_layout.size());
    std::vector<std::int64_t> entities_v(entities.extent(0) * num_vert_per_e);
    for (std::size_t e = 0; e < entities.extent(0); ++e)
    {
      std::span entity(entities_v.data() + e * num_vert_per_e, num_vert_per_e);
      for (std::size_t i = 0; i < num_vert_per_e; ++i)
        entity[i] = entities(e, entity_vertex_dofs[i]);
      std::ranges::sort(entity);
    }

    std::array shape{entities.extent(0), num_vert_per_e};
    return std::pair(std::move(entities_v), shape);
  };
  const auto [entities_v_b, shapev] = to_vertex_entities(
      cmap_dof_layout, entity_dim, cell_vertex_dofs, cell_type, entities);

  md::mdspan<const std::int64_t, md::dextents<std::size_t, 2>> entities_v(
      entities_v_b.data(), shapev);

  MPI_Comm comm = topology.comm();
  MPI_Datatype compound_type;
  MPI_Type_contiguous(entities_v.extent(1), MPI_INT64_T, &compound_type);
  MPI_Type_commit(&compound_type);

  // -- B. Send entities and entity data to postmaster
  auto send_entities_to_postmaster
      = [](MPI_Comm comm, MPI_Datatype compound_type, std::int64_t num_nodes_g,
           auto entities, std::span<const T> data)
  {
    const int size = dolfinx::MPI::size(comm);

    // Determine destination by index of first vertex
    std::vector<int> dest0;
    for (std::size_t e = 0; e < entities.extent(0); ++e)
    {
      dest0.push_back(
          dolfinx::MPI::index_owner(size, entities(e, 0), num_nodes_g));
    }
    std::vector<int> perm(dest0.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::ranges::sort(perm, [&dest0](auto x0, auto x1)
                      { return dest0[x0] < dest0[x1]; });

    // Note: dest[perm[i]] is ordered with increasing i
    // Build list of neighbour dest ranks and count number of entities to
    // send to each post office
    std::vector<int> dest;
    std::vector<std::int32_t> num_items_send;
    {
      auto it = perm.begin();
      while (it != perm.end())
      {
        dest.push_back(dest0[*it]);
        auto it1
            = std::find_if(it, perm.end(), [&dest0, r = dest.back()](auto idx)
                           { return dest0[idx] != r; });
        num_items_send.push_back(std::distance(it, it1));
        it = it1;
      }
    }

    // Compute send displacements
    std::vector<int> send_disp(num_items_send.size() + 1, 0);
    std::partial_sum(num_items_send.begin(), num_items_send.end(),
                     std::next(send_disp.begin()));

    // Determine src ranks. Sort ranks so that ownership determination is
    // deterministic for a given number of ranks.
    std::vector<int> src = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
    std::ranges::sort(src);

    // Create neighbourhood communicator for sending data to post
    // offices
    MPI_Comm comm0;
    int err = MPI_Dist_graph_create_adjacent(
        comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(), dest.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
    dolfinx::MPI::check_error(comm, err);

    // Send number of items to post offices (destinations)
    std::vector<int> num_items_recv(src.size());
    num_items_send.reserve(1);
    num_items_recv.reserve(1);
    MPI_Neighbor_alltoall(num_items_send.data(), 1, MPI_INT,
                          num_items_recv.data(), 1, MPI_INT, comm0);
    dolfinx::MPI::check_error(comm, err);

    // Compute receive displacements
    std::vector<int> recv_disp(num_items_recv.size() + 1, 0);
    std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                     std::next(recv_disp.begin()));

    // Prepare send buffer
    std::vector<std::int64_t> send_buffer;
    std::vector<T> send_values_buffer;
    send_buffer.reserve(entities.size());
    send_values_buffer.reserve(data.size());
    for (std::size_t e = 0; e < entities.extent(0); ++e)
    {
      auto idx = perm[e];
      auto it = std::next(entities.data_handle(), idx * entities.extent(1));
      send_buffer.insert(send_buffer.end(), it, it + entities.extent(1));
      send_values_buffer.push_back(data[idx]);
    }

    std::vector<std::int64_t> recv_buffer(recv_disp.back()
                                          * entities.extent(1));
    err = MPI_Neighbor_alltoallv(send_buffer.data(), num_items_send.data(),
                                 send_disp.data(), compound_type,
                                 recv_buffer.data(), num_items_recv.data(),
                                 recv_disp.data(), compound_type, comm0);
    dolfinx::MPI::check_error(comm, err);
    std::vector<T> recv_values_buffer(recv_disp.back());
    err = MPI_Neighbor_alltoallv(
        send_values_buffer.data(), num_items_send.data(), send_disp.data(),
        dolfinx::MPI::mpi_t<T>, recv_values_buffer.data(),
        num_items_recv.data(), recv_disp.data(), dolfinx::MPI::mpi_t<T>, comm0);
    dolfinx::MPI::check_error(comm, err);
    err = MPI_Comm_free(&comm0);
    dolfinx::MPI::check_error(comm, err);

    std::array shape{recv_buffer.size() / (entities.extent(1)),
                     (entities.extent(1))};
    return std::tuple<std::vector<std::int64_t>, std::vector<T>,
                      std::array<std::size_t, 2>>(
        std::move(recv_buffer), std::move(recv_values_buffer), shape);
  };
  const auto [entitiesp_b, entitiesp_v, shapep] = send_entities_to_postmaster(
      comm, compound_type, num_nodes_g, entities_v, data);
  md::mdspan<const std::int64_t, md::dextents<std::size_t, 2>> entitiesp(
      entitiesp_b.data(), shapep);

  // -- C. Send mesh global indices to postmaster
  auto indices_to_postoffice = [](MPI_Comm comm, std::int64_t num_nodes,
                                  std::span<const std::int64_t> indices)
  {
    int size = dolfinx::MPI::size(comm);
    std::vector<std::pair<int, std::int64_t>> dest_to_index;
    std::ranges::transform(
        indices, std::back_inserter(dest_to_index),
        [size, num_nodes](auto n)
        {
          return std::pair(dolfinx::MPI::index_owner(size, n, num_nodes), n);
        });
    std::ranges::sort(dest_to_index);

    // Build list of neighbour dest ranks and count number of indices to
    // send to each post office
    std::vector<int> dest;
    std::vector<std::int32_t> num_items_send;
    {
      auto it = dest_to_index.begin();
      while (it != dest_to_index.end())
      {
        dest.push_back(it->first);
        auto it1
            = std::find_if(it, dest_to_index.end(), [r = dest.back()](auto idx)
                           { return idx.first != r; });
        num_items_send.push_back(std::distance(it, it1));
        it = it1;
      }
    }

    // Compute send displacements
    std::vector<int> send_disp(num_items_send.size() + 1, 0);
    std::partial_sum(num_items_send.begin(), num_items_send.end(),
                     std::next(send_disp.begin()));

    // Determine src ranks. Sort ranks so that ownership determination is
    // deterministic for a given number of ranks.
    std::vector<int> src = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
    std::ranges::sort(src);

    // Create neighbourhood communicator for sending data to post offices
    MPI_Comm comm0;
    int err = MPI_Dist_graph_create_adjacent(
        comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(), dest.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
    dolfinx::MPI::check_error(comm, err);

    // Send number of items to post offices (destination) that I will be
    // sending
    std::vector<int> num_items_recv(src.size());
    num_items_send.reserve(1);
    num_items_recv.reserve(1);
    MPI_Neighbor_alltoall(num_items_send.data(), 1, MPI_INT,
                          num_items_recv.data(), 1, MPI_INT, comm0);
    dolfinx::MPI::check_error(comm, err);

    // Compute receive displacements
    std::vector<int> recv_disp(num_items_recv.size() + 1, 0);
    std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                     std::next(recv_disp.begin()));

    // Prepare send buffer
    std::vector<std::int64_t> send_buffer;
    send_buffer.reserve(indices.size());
    std::ranges::transform(dest_to_index, std::back_inserter(send_buffer),
                           [](auto x) { return x.second; });

    std::vector<std::int64_t> recv_buffer(recv_disp.back());
    err = MPI_Neighbor_alltoallv(send_buffer.data(), num_items_send.data(),
                                 send_disp.data(), MPI_INT64_T,
                                 recv_buffer.data(), num_items_recv.data(),
                                 recv_disp.data(), MPI_INT64_T, comm0);
    dolfinx::MPI::check_error(comm, err);
    err = MPI_Comm_free(&comm0);
    dolfinx::MPI::check_error(comm, err);
    return std::tuple(std::move(recv_buffer), std::move(recv_disp),
                      std::move(src), std::move(dest));
  };
  const auto [nodes_g_p, recv_disp, src, dest]
      = indices_to_postoffice(comm, num_nodes_g, nodes_g);

  // D. Send entities to possible owners, based on first entity index
  auto candidate_ranks
      = [](MPI_Comm comm, MPI_Datatype compound_type,
           std::span<const std::int64_t> indices_recv,
           std::span<const int> indices_recv_disp, std::span<const int> src,
           std::span<const int> dest, auto entities, std::span<const T> data)
  {
    // Build map from received global node indices to neighbourhood
    // ranks that have the node
    std::multimap<std::int64_t, int> node_to_rank;
    for (std::size_t i = 0; i < indices_recv_disp.size() - 1; ++i)
      for (int j = indices_recv_disp[i]; j < indices_recv_disp[i + 1]; ++j)
        node_to_rank.insert({indices_recv[j], i});

    std::vector<std::vector<std::int64_t>> send_data(dest.size());
    std::vector<std::vector<T>> send_values(dest.size());
    for (std::size_t e = 0; e < entities.extent(0); ++e)
    {
      std::span e_recv(entities.data_handle() + e * entities.extent(1),
                       entities.extent(1));
      auto [it0, it1] = node_to_rank.equal_range(entities(e, 0));
      for (auto it = it0; it != it1; ++it)
      {
        int p = it->second;
        send_data[p].insert(send_data[p].end(), e_recv.begin(), e_recv.end());
        send_values[p].push_back(data[e]);
      }
    }

    MPI_Comm comm0;
    int err = MPI_Dist_graph_create_adjacent(
        comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(), dest.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
    dolfinx::MPI::check_error(comm, err);

    std::vector<int> num_items_send;
    for (auto& x : send_data)
      num_items_send.push_back(x.size() / entities.extent(1));

    std::vector<int> num_items_recv(src.size());
    num_items_send.reserve(1);
    num_items_recv.reserve(1);
    err = MPI_Neighbor_alltoall(num_items_send.data(), 1, MPI_INT,
                                num_items_recv.data(), 1, MPI_INT, comm0);
    dolfinx::MPI::check_error(comm, err);

    // Compute send displacements
    std::vector<std::int32_t> send_disp(num_items_send.size() + 1, 0);
    std::partial_sum(num_items_send.begin(), num_items_send.end(),
                     std::next(send_disp.begin()));

    // Compute receive displacements
    std::vector<std::int32_t> recv_disp(num_items_recv.size() + 1, 0);
    std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                     std::next(recv_disp.begin()));

    // Prepare send buffers
    std::vector<std::int64_t> send_buffer;
    std::vector<T> send_values_buffer;
    for (auto& x : send_data)
      send_buffer.insert(send_buffer.end(), x.begin(), x.end());
    for (auto& v : send_values)
      send_values_buffer.insert(send_values_buffer.end(), v.begin(), v.end());
    std::vector<std::int64_t> recv_buffer(entities.extent(1)
                                          * recv_disp.back());
    err = MPI_Neighbor_alltoallv(send_buffer.data(), num_items_send.data(),
                                 send_disp.data(), compound_type,
                                 recv_buffer.data(), num_items_recv.data(),
                                 recv_disp.data(), compound_type, comm0);

    dolfinx::MPI::check_error(comm, err);

    std::vector<T> recv_values_buffer(recv_disp.back());
    err = MPI_Neighbor_alltoallv(
        send_values_buffer.data(), num_items_send.data(), send_disp.data(),
        dolfinx::MPI::mpi_t<T>, recv_values_buffer.data(),
        num_items_recv.data(), recv_disp.data(), dolfinx::MPI::mpi_t<T>, comm0);

    dolfinx::MPI::check_error(comm, err);

    err = MPI_Comm_free(&comm0);
    dolfinx::MPI::check_error(comm, err);

    std::array shape{recv_buffer.size() / entities.extent(1),
                     entities.extent(1)};
    return std::tuple<std::vector<std::int64_t>, std::vector<T>,
                      std::array<std::size_t, 2>>(
        std::move(recv_buffer), std::move(recv_values_buffer), shape);
  };
  // NOTE: src and dest are transposed here because we're reversing the
  // direction of communication
  const auto [entities_data_b, entities_values, shape_eb]
      = candidate_ranks(comm, compound_type, nodes_g_p, recv_disp, dest, src,
                        entitiesp, std::span(entitiesp_v));
  md::mdspan<const std::int64_t, md::dextents<std::size_t, 2>> entities_data(
      entities_data_b.data(), shape_eb);

  // -- E. From the received (key, value) data, determine which keys
  //    (entities) are on this process.
  //
  // TODO: We have already received possibly tagged entities from other
  //       ranks, so we could use the received data to avoid creating
  //       the std::map for *all* entities and just for candidate
  //       entities.
  auto select_entities
      = [](const mesh::Topology& topology, auto xdofmap,
           std::span<const std::int64_t> nodes_g,
           std::span<const int> cell_vertex_dofs, auto entities_data,
           std::span<const T> entities_values)
  {
    spdlog::info("XDMF build map");
    auto c_to_v = topology.connectivity(topology.dim(), 0);
    if (!c_to_v)
      throw std::runtime_error("Missing cell-vertex connectivity.");

    std::map<std::int64_t, std::int32_t> input_idx_to_vertex;
    for (int c = 0; c < c_to_v->num_nodes(); ++c)
    {
      auto vertices = c_to_v->links(c);
      std::span xdofs(xdofmap.data_handle() + c * xdofmap.extent(1),
                      xdofmap.extent(1));
      for (std::size_t v = 0; v < vertices.size(); ++v)
        input_idx_to_vertex[nodes_g[xdofs[cell_vertex_dofs[v]]]] = vertices[v];
    }

    std::vector<std::int32_t> entities;
    std::vector<T> data;
    std::vector<std::int32_t> entity(entities_data.extent(1));
    for (std::size_t e = 0; e < entities_data.extent(0); ++e)
    {
      bool entity_found = true;
      for (std::size_t i = 0; i < entities_data.extent(1); ++i)
      {
        if (auto it = input_idx_to_vertex.find(entities_data(e, i));
            it == input_idx_to_vertex.end())
        {
          // As soon as this received index is not in locally owned
          // input global indices skip the entire entity
          entity_found = false;
          break;
        }
        else
          entity[i] = it->second;
      }

      if (entity_found)
      {
        entities.insert(entities.end(), entity.begin(), entity.end());
        data.push_back(entities_values[e]);
      }
    }

    return std::pair(std::move(entities), std::move(data));
  };

  MPI_Type_free(&compound_type);

  return select_entities(topology, xdofmap, nodes_g, cell_vertex_dofs,
                         entities_data, std::span(entities_values));
}
//-----------------------------------------------------------------------------}

} // namespace io
} // namespace dolfinx
