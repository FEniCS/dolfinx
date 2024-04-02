// Copyright (C) 2012 Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5Interface.h"
#include <array>
#include <basix/mdspan.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <complex>
#include <concepts>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <filesystem>
#include <numeric>
#include <pugixml.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace
{
template <typename T, std::size_t ndim>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, ndim>>;
} // namespace

namespace dolfinx
{

namespace fem
{
template <dolfinx::scalar T, std::floating_point U>
class Function;
} // namespace fem

namespace fem
{
template <std::floating_point T>
class CoordinateElement;
} // namespace fem

namespace mesh
{
template <std::floating_point T>
class Mesh;
} // namespace mesh

namespace io::xdmf_utils
{

/// Get DOLFINx cell type string from XML topology node
/// @return DOLFINx cell type and polynomial degree
std::pair<std::string, int> get_cell_type(const pugi::xml_node& topology_node);

/// Return (0) HDF5 filename and (1) path in HDF5 file from a DataItem
/// node.
std::array<std::string, 2> get_hdf5_paths(const pugi::xml_node& dataitem_node);

std::filesystem::path
get_hdf5_filename(const std::filesystem::path& xdmf_filename);

/// Get dimensions from an XML DataSet node
std::vector<std::int64_t> get_dataset_shape(const pugi::xml_node& dataset_node);

/// Get number of cells from an XML Topology node
std::int64_t get_num_cells(const pugi::xml_node& topology_node);

/// Get the VTK string identifier
std::string vtk_cell_type_str(mesh::CellType cell_type, int num_nodes);

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
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        xdofmap,
    int entity_dim,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        entities,
    std::span<const T> data)
{
  assert(entities.extent(0) == data.size());
  LOG(INFO) << "XDMF distribute entity data";
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
      std::sort(entity.begin(), entity.end());
    }

    std::array shape{entities.extent(0), num_vert_per_e};
    return std::pair(std::move(entities_v), shape);
  };
  const auto [entities_v_b, shapev] = to_vertex_entities(
      cmap_dof_layout, entity_dim, cell_vertex_dofs, cell_type, entities);
  mdspan_t<const std::int64_t, 2> entities_v(entities_v_b.data(), shapev);

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
    std::sort(perm.begin(), perm.end(),
              [&dest0](auto x0, auto x1) { return dest0[x0] < dest0[x1]; });

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
        auto it1 = std::find_if(it, perm.end(),
                                [&dest0, r = dest.back()](auto idx)
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
    std::sort(src.begin(), src.end());

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
        dolfinx::MPI::mpi_type<T>(), recv_values_buffer.data(),
        num_items_recv.data(), recv_disp.data(), dolfinx::MPI::mpi_type<T>(),
        comm0);
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
  mdspan_t<const std::int64_t, 2> entitiesp(entitiesp_b.data(), shapep);

  // -- C. Send mesh global indices to postmaster
  auto indices_to_postoffice = [](MPI_Comm comm, std::int64_t num_nodes,
                                  std::span<const std::int64_t> indices)
  {
    int size = dolfinx::MPI::size(comm);
    std::vector<std::pair<int, std::int64_t>> dest_to_index;
    std::transform(
        indices.begin(), indices.end(), std::back_inserter(dest_to_index),
        [size, num_nodes](auto n) {
          return std::pair(dolfinx::MPI::index_owner(size, n, num_nodes), n);
        });
    std::sort(dest_to_index.begin(), dest_to_index.end());

    // Build list of neighbour dest ranks and count number of indices to
    // send to each post office
    std::vector<int> dest;
    std::vector<std::int32_t> num_items_send;
    {
      auto it = dest_to_index.begin();
      while (it != dest_to_index.end())
      {
        dest.push_back(it->first);
        auto it1 = std::find_if(it, dest_to_index.end(),
                                [r = dest.back()](auto idx)
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
    std::sort(src.begin(), src.end());

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
    std::transform(dest_to_index.begin(), dest_to_index.end(),
                   std::back_inserter(send_buffer),
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
        dolfinx::MPI::mpi_type<T>(), recv_values_buffer.data(),
        num_items_recv.data(), recv_disp.data(), dolfinx::MPI::mpi_type<T>(),
        comm0);

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
  mdspan_t<const std::int64_t, 2> entities_data(entities_data_b.data(),
                                                shape_eb);

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
    LOG(INFO) << "XDMF build map";
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

/// TODO: Document
template <typename T>
void add_data_item(pugi::xml_node& xml_node, hid_t h5_id,
                   const std::string& h5_path, std::span<const T> x,
                   std::int64_t offset, const std::vector<std::int64_t>& shape,
                   const std::string& number_type, bool use_mpi_io)
{
  // Add DataItem node
  assert(xml_node);
  pugi::xml_node data_item_node = xml_node.append_child("DataItem");
  assert(data_item_node);

  // Add dimensions attribute
  std::string dims;
  for (auto d : shape)
    dims += std::to_string(d) + std::string(" ");
  dims.pop_back();
  data_item_node.append_attribute("Dimensions") = dims.c_str();

  // Set type for topology data (needed by XDMF to prevent default to
  // float)
  if (!number_type.empty())
    data_item_node.append_attribute("NumberType") = number_type.c_str();

  // Add format attribute
  if (h5_id < 0)
  {
    data_item_node.append_attribute("Format") = "XML";
    assert(shape.size() == 2);
    std::ostringstream s;
    s.precision(16);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      if ((i + 1) % shape[1] == 0 and shape[1] != 0)
        s << x.data()[i] << std::endl;
      else
        s << x.data()[i] << " ";
    }

    data_item_node.append_child(pugi::node_pcdata).set_value(s.str().c_str());
  }
  else
  {
    data_item_node.append_attribute("Format") = "HDF";

    // Get name of HDF5 file, including path
    const std::filesystem::path p = io::hdf5::get_filename(h5_id);
    const std::filesystem::path filename = p.filename().c_str();

    // Add HDF5 filename and HDF5 internal path to XML file
    const std::string xdmf_path
        = filename.string() + std::string(":") + h5_path;
    data_item_node.append_child(pugi::node_pcdata).set_value(xdmf_path.c_str());

    // Compute data offset and range of values
    std::int64_t local_shape0 = std::reduce(
        std::next(shape.begin()), shape.end(), x.size(), std::divides{});

    const std::array local_range{offset, offset + local_shape0};
    io::hdf5::write_dataset(h5_id, h5_path, x.data(), local_range, shape,
                            use_mpi_io, false);

    // Add partitioning attribute to dataset
    // std::vector<std::size_t> partitions;
    // std::vector<std::size_t> offset_tmp(1, offset);
    // dolfinx::MPI::gather(comm, offset_tmp, partitions);
    // dolfinx::MPI::broadcast(comm, partitions);
    // io::hdf5::add_attribute(h5_id, h5_path, "partition", partitions);
  }
}

/// @brief Get data associated with a data set node.
/// @tparam T Data type to read into.
/// @warning Data will be silently cast to type `T` if requested type
/// and storage type differ.
template <typename T>
std::vector<T> get_dataset(MPI_Comm comm, const pugi::xml_node& dataset_node,
                           hid_t h5_id,
                           std::array<std::int64_t, 2> range = {0, 0})
{
  // FIXME: Need to sort out dataset dimensions - can't depend on HDF5
  // shape, and a Topology data item is not required to have a
  // 'Dimensions' attribute since the dimensions can be determined from
  // the number of cells and the cell type (for topology, one must
  // supply cell type + (number of cells or dimensions)).
  //
  // A geometry data item must have 'Dimensions' attribute.

  assert(dataset_node);
  pugi::xml_attribute format_attr = dataset_node.attribute("Format");
  assert(format_attr);

  // Get data set shape from 'Dimensions' attribute (empty if not
  // available)
  const std::vector shape_xml = xdmf_utils::get_dataset_shape(dataset_node);

  const std::string format = format_attr.as_string();
  std::vector<T> data_vector;
  // Only read ASCII on process 0
  const int mpi_rank = dolfinx::MPI::rank(comm);
  if (format == "XML")
  {
    if (mpi_rank == 0)
    {
      // Read data and trim any leading/trailing whitespace
      pugi::xml_node data_node = dataset_node.first_child();
      assert(data_node);
      std::string data_str = data_node.value();

      // Split data based on spaces and line breaks
      std::vector<boost::iterator_range<std::string::iterator>> data_vector_str;
      boost::split(data_vector_str, data_str, boost::is_any_of(" \n"));

      // Add data to numerical vector
      data_vector.reserve(data_vector_str.size());
      for (auto& v : data_vector_str)
      {
        if (v.begin() != v.end())
          data_vector.push_back(
              boost::lexical_cast<T>(boost::copy_range<std::string>(v)));
      }
    }
  }
  else if (format == "HDF")
  {
    // Get file and data path
    auto paths = xdmf_utils::get_hdf5_paths(dataset_node);

    // Get data shape from HDF5 file
    const std::vector shape_hdf5 = io::hdf5::get_dataset_shape(h5_id, paths[1]);

    // FIXME: should we support empty data sets?
    // Check that data set is not empty
    assert(!shape_hdf5.empty());
    assert(shape_hdf5[0] != 0);

    // Determine range of data to read from HDF5 file. This is
    // complicated by the XML Dimension attribute and the HDF5 storage
    // possibly having different shapes, e.g. the HDF5 storage may be a
    // flat array.

    // If range = {0, 0} then no range is supplied and we must determine
    // the range
    if (range[0] == 0 and range[1] == 0)
    {
      if (shape_xml == shape_hdf5)
      {
        range = dolfinx::MPI::local_range(mpi_rank, shape_hdf5[0],
                                          dolfinx::MPI::size(comm));
      }
      else if (!shape_xml.empty() and shape_hdf5.size() == 1)
      {
        // Size of dims > 0
        std::int64_t d = std::reduce(shape_xml.begin(), shape_xml.end(),
                                     std::int64_t(1), std::multiplies{});

        // Check for data size consistency
        if (d * shape_xml[0] != shape_hdf5[0])
        {
          throw std::runtime_error("Data size in XDMF/XML and size of HDF5 "
                                   "dataset are inconsistent");
        }

        // Compute data range to read
        range = dolfinx::MPI::local_range(mpi_rank, shape_xml[0],
                                          dolfinx::MPI::rank(comm));
        range[0] *= d;
        range[1] *= d;
      }
      else
      {
        throw std::runtime_error("This combination of array shapes in XDMF and "
                                 "HDF5 is not supported");
      }
    }

    // Retrieve data
    if (hid_t dset_id = io::hdf5::open_dataset(h5_id, paths[1]);
        dset_id == H5I_INVALID_HID)
      throw std::runtime_error("Failed to open HDF5 global dataset.");
    else
    {
      data_vector = io::hdf5::read_dataset<T>(dset_id, range, true);
      if (herr_t err = H5Dclose(dset_id); err < 0)
        throw std::runtime_error("Failed to close HDF5 global dataset.");
    }
  }
  else
    throw std::runtime_error("Storage format \"" + format + "\" is unknown");

  // Get dimensions for consistency (if available in DataItem node)
  if (shape_xml.empty())
  {
    std::int64_t size = 1;
    for (auto dim : shape_xml)
      size *= dim;

    std::int64_t size_global = 0;
    const std::int64_t size_local = data_vector.size();
    MPI_Allreduce(&size_local, &size_global, 1, MPI_INT64_T, MPI_SUM, comm);
    if (size != size_global)
    {
      throw std::runtime_error(
          "Data sizes in attribute and size of data read are inconsistent");
    }
  }

  return data_vector;
}

} // namespace io::xdmf_utils
} // namespace dolfinx
