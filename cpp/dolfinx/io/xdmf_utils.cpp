// Copyright (C) 2012-2023 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_utils.h"
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <filesystem>
#include <map>
#include <pugixml.hpp>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
/// Get data width - normally the same as u.value_size(), but expand for
/// 2D vector/tensor because XDMF presents everything as 3D
template <std::floating_point U>
std::int64_t get_padded_width(const fem::FiniteElement<U>& e)
{
  const int width = e.value_size();
  const int rank = e.value_shape().size();
  if (rank == 1 and width == 2)
    return 3;
  else if (rank == 2 and width == 4)
    return 9;
  else
    return width;
}
} // namespace

//----------------------------------------------------------------------------
std::pair<std::string, int>
xdmf_utils::get_cell_type(const pugi::xml_node& topology_node)
{
  assert(topology_node);
  pugi::xml_attribute type_attr = topology_node.attribute("TopologyType");
  assert(type_attr);

  const static std::map<std::string, std::pair<std::string, int>> xdmf_to_dolfin
      = {{"polyvertex", {"point", 1}},
         {"polyline", {"interval", 1}},
         {"edge_3", {"interval", 2}},
         {"triangle", {"triangle", 1}},
         {"triangle_6", {"triangle", 2}},
         {"tetrahedron", {"tetrahedron", 1}},
         {"tetrahedron_10", {"tetrahedron", 2}},
         {"quadrilateral", {"quadrilateral", 1}},
         {"quadrilateral_9", {"quadrilateral", 2}},
         {"quadrilateral_16", {"quadrilateral", 3}},
         {"hexahedron", {"hexahedron", 1}},
         {"hexahedron_27", {"hexahedron", 2}}};

  // Convert XDMF cell type string to DOLFINx cell type string
  std::string cell_type = type_attr.as_string();
  std::transform(cell_type.begin(), cell_type.end(), cell_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  auto it = xdmf_to_dolfin.find(cell_type);
  if (it == xdmf_to_dolfin.end())
  {
    throw std::runtime_error("Cannot recognise cell type. Unknown value: "
                             + cell_type);
  }

  return it->second;
}
//----------------------------------------------------------------------------
std::array<std::string, 2>
xdmf_utils::get_hdf5_paths(const pugi::xml_node& dataitem_node)
{
  // Check that node is a DataItem node
  assert(dataitem_node);
  const std::string dataitem_str = "DataItem";
  if (dataitem_node.name() != dataitem_str)
  {
    throw std::runtime_error("Node name is \""
                             + std::string(dataitem_node.name())
                             + R"(", expecting "DataItem")");
  }

  // Check that format is HDF
  pugi::xml_attribute format_attr = dataitem_node.attribute("Format");
  assert(format_attr);
  const std::string format = format_attr.as_string();
  if (format.compare("HDF") != 0)
  {
    throw std::runtime_error("DataItem format \"" + format
                             + R"(" is not "HDF")");
  }

  // Get path data
  pugi::xml_node path_node = dataitem_node.first_child();
  assert(path_node);

  // Create string from path and trim leading and trailing whitespace
  std::string path = path_node.text().get();
  boost::algorithm::trim(path);

  // Split string into file path and HD5 internal path
  std::vector<std::string> paths;
  boost::split(paths, path, boost::is_any_of(":"));
  assert(paths.size() == 2);

  return {{paths[0], paths[1]}};
}
//-----------------------------------------------------------------------------
std::filesystem::path
xdmf_utils::get_hdf5_filename(const std::filesystem::path& xdmf_filename)
{
  std::filesystem::path p = xdmf_filename;
  p.replace_extension("h5");
  if (p.string() == xdmf_filename)
  {
    throw std::runtime_error("Cannot deduce name of HDF5 file from XDMF "
                             "filename. Filename clash. Check XDMF filename");
  }

  return p;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
xdmf_utils::get_dataset_shape(const pugi::xml_node& dataset_node)
{
  // Get Dimensions attribute string
  assert(dataset_node);
  pugi::xml_attribute dimensions_attr = dataset_node.attribute("Dimensions");

  // Gets dimensions, if attribute is present
  std::vector<std::int64_t> dims;
  if (dimensions_attr)
  {
    // Split dimensions string
    const std::string dims_str = dimensions_attr.as_string();
    std::vector<std::string> dims_list;
    boost::split(dims_list, dims_str, boost::is_any_of(" "));

    // Cast dims to integers
    for (auto d : dims_list)
      dims.push_back(boost::lexical_cast<std::int64_t>(d));
  }

  return dims;
}
//----------------------------------------------------------------------------
std::int64_t xdmf_utils::get_num_cells(const pugi::xml_node& topology_node)
{
  assert(topology_node);

  // Get number of cells from topology
  std::int64_t num_cells_topology = -1;
  pugi::xml_attribute num_cells_attr
      = topology_node.attribute("NumberOfElements");
  if (num_cells_attr)
    num_cells_topology = num_cells_attr.as_llong();

  // Get number of cells from topology dataset
  pugi::xml_node topology_dataset_node = topology_node.child("DataItem");
  assert(topology_dataset_node);
  const std::vector tdims = get_dataset_shape(topology_dataset_node);

  // Check that number of cells can be determined
  if (tdims.size() != 2 and num_cells_topology == -1)
    throw std::runtime_error("Cannot determine number of cells in XDMF mesh");

  // Check for consistency if number of cells appears in both the
  // topology and DataItem nodes
  if (num_cells_topology != -1 and tdims.size() == 2)
  {
    if (num_cells_topology != tdims[0])
      throw std::runtime_error("Cannot determine number of cells in XDMF mesh");
  }

  return std::max(num_cells_topology, tdims[0]);
}
//----------------------------------------------------------------------------
std::string xdmf_utils::vtk_cell_type_str(mesh::CellType cell_type,
                                          int num_nodes)
{
  static const std::map<mesh::CellType, std::map<int, std::string>> vtk_map = {
      {mesh::CellType::point, {{1, "PolyVertex"}}},
      {mesh::CellType::interval, {{2, "PolyLine"}, {3, "Edge_3"}}},
      {mesh::CellType::triangle,
       {{3, "Triangle"}, {6, "Triangle_6"}, {10, "Triangle_10"}}},
      {mesh::CellType::quadrilateral,
       {{4, "Quadrilateral"},
        {9, "Quadrilateral_9"},
        {16, "Quadrilateral_16"}}},
      {mesh::CellType::prism, {{6, "Wedge"}}},
      {mesh::CellType::tetrahedron,
       {{4, "Tetrahedron"}, {10, "Tetrahedron_10"}, {20, "Tetrahedron_20"}}},
      {mesh::CellType::hexahedron, {{8, "Hexahedron"}, {27, "Hexahedron_27"}}}};

  // Get cell family
  auto cell = vtk_map.find(cell_type);
  if (cell == vtk_map.end())
    throw std::runtime_error("Could not find cell type.");

  // Get cell string
  auto cell_str = cell->second.find(num_nodes);
  if (cell_str == cell->second.end())
    throw std::runtime_error("Could not find VTK string for cell order.");

  return cell_str->second;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
xdmf_utils::distribute_entity_data(
    const mesh::Topology& topology, std::span<const std::int64_t> nodes_g,
    std::int64_t num_nodes_g, const fem::ElementDofLayout& cmap_dof_layout,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        xdofmap,
    int entity_dim, std::span<const std::int64_t> entities,
    std::span<const std::int32_t> data)
{
  LOG(INFO) << "XDMF distribute entity data";
  auto cell_types = topology.cell_types();
  if (cell_types.size() > 1)
    throw std::runtime_error("cell type IO");

  const std::size_t num_vert_per_entity = mesh::cell_num_entities(
      mesh::cell_entity_type(cell_types.back(), entity_dim, 0), 0);

  std::vector<int> cell_vertex_dofs;
  {
    // Get layout of dofs on 0th cell entity of dimension entity_dim
    for (int i = 0; i < mesh::cell_num_entities(cell_types.back(), 0); ++i)
    {
      const std::vector<int>& local_index = cmap_dof_layout.entity_dofs(0, i);
      assert(local_index.size() == 1);
      cell_vertex_dofs.push_back(local_index[0]);
    }
  }

  // -- A. Convert from list of entities by 'nodex' to list of entities
  // by 'vertex nodes'
  std::vector<std::int64_t> entities_v;
  {
    // Use ElementDofLayout of the cell to get vertex dof indices (local
    // to a cell), i.e. build a map from local vertex index to associated
    // local dof index
    auto c_to_v = topology.connectivity(topology.dim(), 0);
    if (!c_to_v)
      throw std::runtime_error("Missing cell-vertex connectivity.");
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

    const std::size_t shape_e_1 = entity_layout.size();
    const std::size_t shape_e_0 = entities.size() / shape_e_1;
    entities_v.resize(shape_e_0 * num_vert_per_entity);
    for (std::size_t e = 0; e < shape_e_0; ++e)
    {
      std::span entity(entities_v.data() + e * num_vert_per_entity,
                       num_vert_per_entity);
      for (std::size_t i = 0; i < num_vert_per_entity; ++i)
        entity[i] = entities[e * shape_e_1 + entity_vertex_dofs[i]];
      std::sort(entity.begin(), entity.end());
    }
  }

  MPI_Comm comm = topology.comm();

  // -- B. Send entities and entity data to postmaster
  auto entities_to_postoffice
      = [comm, num_nodes = num_nodes_g](std::span<const std::int64_t> entities,
                                        int bs,
                                        std::span<const std::int32_t> data)
  {
    assert(entities.size() / bs == data.size());
    int size = dolfinx::MPI::size(comm);

    // Determine destination by index of first vertex
    std::vector<int> dest0;
    for (std::size_t i = 0; i < entities.size(); i += bs)
      dest0.push_back(dolfinx::MPI::index_owner(size, entities[i], num_nodes));
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
    std::vector<std::int32_t> send_disp(num_items_send.size() + 1, 0);
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
    std::vector<std::int32_t> recv_disp(num_items_recv.size() + 1, 0);
    std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                     std::next(recv_disp.begin()));

    // Prepare send buffer
    std::vector<std::int64_t> send_buffer;
    send_buffer.reserve(entities.size() + data.size());
    for (std::size_t i = 0; i < entities.size() / bs; ++i)
    {
      auto idx = perm[i];
      auto it = std::next(entities.begin(), idx * bs);
      send_buffer.insert(send_buffer.end(), it, it + bs);
      send_buffer.push_back(data[idx]);
    }

    std::vector<std::int64_t> recv_buffer(recv_disp.back() * (bs + 1));
    MPI_Datatype compound_type;
    MPI_Type_contiguous(bs + 1, MPI_INT64_T, &compound_type);
    MPI_Type_commit(&compound_type);
    err = MPI_Neighbor_alltoallv(send_buffer.data(), num_items_send.data(),
                                 send_disp.data(), compound_type,
                                 recv_buffer.data(), num_items_recv.data(),
                                 recv_disp.data(), compound_type, comm0);
    dolfinx::MPI::check_error(comm, err);

    err = MPI_Comm_free(&comm0);
    dolfinx::MPI::check_error(comm, err);

    return recv_buffer;
  };
  std::vector<std::int64_t> entity_data
      = entities_to_postoffice(entities_v, num_vert_per_entity, data);

  // -- C. Send mesh global indices to postmaster
  auto indices_to_postoffice
      = [comm, num_nodes = num_nodes_g](std::span<const std::int64_t> indices)
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
    std::vector<std::int32_t> send_disp(num_items_send.size() + 1, 0);
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
    std::vector<std::int32_t> recv_disp(num_items_recv.size() + 1, 0);
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
    return std::tuple(recv_buffer, recv_disp, src, dest);
  };
  auto [global_indices1, recv_disp, src, dest] = indices_to_postoffice(nodes_g);

  // D. Send entities to possible owners, based on first entity index
  auto candidate_ranks
      = [comm](auto indices_recv, auto indices_recv_disp, auto src, auto dest,
               auto entities, auto eshape1)
  {
    // Build map from received global node indices to neighbourhood
    // ranks that have the node
    std::multimap<std::int64_t, int> node_to_rank;
    for (std::size_t i = 0; i < indices_recv_disp.size() - 1; ++i)
      for (int j = indices_recv_disp[i]; j < indices_recv_disp[i + 1]; ++j)
        node_to_rank.insert({indices_recv[j], i});

    std::vector<std::vector<std::int64_t>> send_data(dest.size());
    for (std::size_t e = 0; e < entities.size() / eshape1; ++e)
    {
      std::span e_recv(entities.data() + e * eshape1, eshape1);
      auto [it0, it1] = node_to_rank.equal_range(e_recv.front());
      for (auto it = it0; it != it1; ++it)
      {
        int p = it->second;
        send_data[p].insert(send_data[p].end(), e_recv.begin(),
                            e_recv.end() - 1);
        send_data[p].push_back(e_recv.back());
      }
    }

    MPI_Comm comm0;
    int err = MPI_Dist_graph_create_adjacent(
        comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(), dest.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
    dolfinx::MPI::check_error(comm, err);

    std::vector<int> num_items_send;
    for (auto& x : send_data)
      num_items_send.push_back(x.size() / eshape1);

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

    // Prepare send buffer
    std::vector<std::int64_t> send_buffer;
    for (auto& x : send_data)
      send_buffer.insert(send_buffer.end(), x.begin(), x.end());

    std::vector<std::int64_t> recv_buffer(eshape1 * recv_disp.back());
    MPI_Datatype compound_type;
    MPI_Type_contiguous(eshape1, MPI_INT64_T, &compound_type);
    MPI_Type_commit(&compound_type);
    err = MPI_Neighbor_alltoallv(send_buffer.data(), num_items_send.data(),
                                 send_disp.data(), compound_type,
                                 recv_buffer.data(), num_items_recv.data(),
                                 recv_disp.data(), compound_type, comm0);
    err = MPI_Comm_free(&comm0);
    dolfinx::MPI::check_error(comm, err);
    return recv_buffer;
  };
  // NOTE: src and dest are transposed here because we're reversing the
  // direction of communication
  auto entity_data_revc = candidate_ranks(global_indices1, recv_disp, dest, src,
                                          entity_data, num_vert_per_entity + 1);

  // -- E. From the received (key, value) data, determine which keys
  //    (entities) are on this process.
  //
  // TODO: Rather than using std::map<std::vector<std::int64_t>,
  //       std::int32_t>, use a rectangular array to avoid the
  //       cost of std::vector<std::int64_t> allocations, and sort the
  //       Array by row.
  //
  // TODO: We have already received possibly tagged entities from other
  //       ranks, so we could use the received data to avoid creating
  //       the std::map for *all* entities and just for candidate
  //       entities.
  auto determine_my_entities =
      [&cell_vertex_dofs](const mesh::Topology& topology,
                          std::span<const std::int64_t> nodes_g, auto x_dofmap,
                          const std::span<const std::int64_t> recv_ents,
                          std::size_t num_vert_per_entity)
  {
    // Build map from input global indices to local vertex numbers
    LOG(INFO) << "XDMF build map";

    auto c_to_v = topology.connectivity(topology.dim(), 0);
    std::map<std::int64_t, std::int32_t> input_idx_to_vertex;
    for (int c = 0; c < c_to_v->num_nodes(); ++c)
    {
      auto vertices = c_to_v->links(c);
      auto xdofs = MDSPAN_IMPL_STANDARD_NAMESPACE::
          MDSPAN_IMPL_PROPOSED_NAMESPACE::submdspan(
              x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t v = 0; v < vertices.size(); ++v)
        input_idx_to_vertex[nodes_g[xdofs[cell_vertex_dofs[v]]]] = vertices[v];
    }

    std::size_t shape1 = num_vert_per_entity + 1;
    std::size_t shape0 = recv_ents.size() / shape1;
    std::vector<std::int32_t> entities_new, data_new;
    entities_new.reserve(shape0 * num_vert_per_entity);
    data_new.reserve(shape1);
    std::vector<std::int32_t> entity(num_vert_per_entity);
    for (std::size_t e = 0; e < shape0; ++e)
    {
      auto ent = recv_ents.subspan(e * shape1, num_vert_per_entity);
      bool entity_found = true;
      for (std::size_t i = 0; i < num_vert_per_entity; ++i)
      {
        if (auto it = input_idx_to_vertex.find(ent[i]);
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
        entities_new.insert(entities_new.end(), entity.begin(), entity.end());
        data_new.push_back(recv_ents[e * shape1 + num_vert_per_entity]);
      }
    }

    return std::pair(std::move(entities_new), std::move(data_new));
  };
  auto [entities_new, data_new] = determine_my_entities(
      topology, nodes_g, xdofmap, entity_data_revc, num_vert_per_entity);

  return {std::move(entities_new), std::move(data_new)};
}
//-----------------------------------------------------------------------------
