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
/// @warning Do not use. This function will be removed.
///
/// Send in_values[p0] to process p0 and receive values from process p1
/// in out_values[p1]
template <typename T>
graph::AdjacencyList<T> all_to_all(MPI_Comm comm,
                                   const graph::AdjacencyList<T>& send_data)
{
  const std::vector<std::int32_t>& send_offsets = send_data.offsets();
  const std::vector<T>& values_in = send_data.array();

  const int comm_size = dolfinx::MPI::size(comm);
  assert(send_data.num_nodes() == comm_size);

  // Data size per destination rank
  std::vector<int> send_size(comm_size);
  std::adjacent_difference(std::next(send_offsets.begin()), send_offsets.end(),
                           send_size.begin());

  // Get received data sizes from each rank
  std::vector<int> recv_size(comm_size);
  MPI_Alltoall(send_size.data(), 1, MPI_INT, recv_size.data(), 1, MPI_INT,
               comm);

  // Compute receive offset
  std::vector<std::int32_t> recv_offset(comm_size + 1, 0);
  std::partial_sum(recv_size.begin(), recv_size.end(),
                   std::next(recv_offset.begin()));

  // Send/receive data
  std::vector<T> recv_values(recv_offset.back());
  MPI_Alltoallv(values_in.data(), send_size.data(), send_offsets.data(),
                dolfinx::MPI::mpi_type<T>(), recv_values.data(),
                recv_size.data(), recv_offset.data(),
                dolfinx::MPI::mpi_type<T>(), comm);

  return graph::AdjacencyList<T>(std::move(recv_values),
                                 std::move(recv_offset));
}
//-----------------------------------------------------------------------------
// Get data width - normally the same as u.value_size(), but expand for
// 2D vector/tensor because XDMF presents everything as 3D
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
//-----------------------------------------------------------------------------

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

  std::vector<std::int64_t> entities_v;
  {
    // Use ElementDofLayout of the cell to get vertex dof indices (local
    // to a cell), i.e. build a map from local vertex index to associated
    // local dof index

    // const std::size_t num_vert_per_entity = mesh::cell_num_entities(
    //     mesh::cell_entity_type(cell_types.back(), entity_dim, 0), 0);
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

  // -- A. Send entities and entity data to postmaster
  MPI_Comm comm = topology.comm();
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

  // -- B. Send mesh global indices to postmaster
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

  // C. Send entities to possible owners, based on first entity index
  auto candidate_ranks
      = [comm](auto indices_recv, auto indices_recv_disp, auto src, auto dest,
               auto entities_v, auto eshape1)
  {
    // Build map from received global node indices to neighbourhood
    // ranks that have the node
    std::multimap<std::int64_t, int> node_to_rank;
    for (std::size_t i = 0; i < indices_recv_disp.size() - 1; ++i)
      for (int j = indices_recv_disp[i]; j < indices_recv_disp[i + 1]; ++j)
        node_to_rank.insert({indices_recv[j], i});

    std::vector<std::vector<std::int64_t>> send_data(dest.size());
    for (std::size_t e = 0; e < entities_v.size() / eshape1; ++e)
    {
      std::span e_recv(entities_v.data() + e * eshape1, eshape1);
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

  //-------------------------------------------------------

  // -------------------
  // 1. Send this rank's global "input" nodes indices to the
  //    'postmaster' rank, and receive global "input" nodes for which
  //    this rank is the postmaster

  auto postmaster_global_nodes_sendrecv
      = [](const mesh::Topology& topology,
           std::span<const std::int64_t> nodes_g, std::int64_t num_nodes_g)
  {
    const MPI_Comm comm = topology.comm();
    const int comm_size = dolfinx::MPI::size(comm);

    // Send input global indices to 'post master' rank, based on input
    // global index value
    // NOTE: could make this int32_t be sending: index <- index -
    // dest_rank_offset
    std::vector<std::vector<std::int64_t>> nodes_g_send(comm_size);
    for (std::int64_t node : nodes_g)
    {
      // Figure out which process is the postmaster for the input global
      // index
      const std::int32_t p
          = dolfinx::MPI::index_owner(comm_size, node, num_nodes_g);
      nodes_g_send[p].push_back(node);
    }

    // Send/receive
    LOG(INFO) << "XDMF send entity nodes size: (" << num_nodes_g << ")";
    graph::AdjacencyList<std::int64_t> nodes_g_recv
        = all_to_all(comm, graph::AdjacencyList<std::int64_t>(nodes_g_send));

    return nodes_g_recv;
  };
  const graph::AdjacencyList<std::int64_t> nodes_g_recv
      = postmaster_global_nodes_sendrecv(topology, nodes_g, num_nodes_g);

  // -------------------
  // 2. Send the entity key (nodes list) and tag to the postmaster based
  //    on the lowest index node in the entity 'key'
  //
  //    NOTE: Stage 2 doesn't depend on the data received in Step 1, so
  //    data (i) the communication could be combined, or (ii) the
  //    communication in Step 1 could be make non-blocking.

  auto postmaster_global_ent_sendrecv
      = [&cell_vertex_dofs](
            const mesh::Topology& topology, std::int64_t num_nodes_g,
            const fem::ElementDofLayout& cmap_dof_layout, int entity_dim,
            std::span<const std::int64_t> entities,
            std::span<const std::int32_t> data)
  {
    const MPI_Comm comm = topology.comm();
    const int comm_size = dolfinx::MPI::size(comm);

    auto cell_types = topology.cell_types();
    if (cell_types.size() > 1)
      throw std::runtime_error("cell type IO");

    const std::size_t num_vert_per_entity = mesh::cell_num_entities(
        mesh::cell_entity_type(cell_types.back(), entity_dim, 0), 0);
    auto c_to_v = topology.connectivity(topology.dim(), 0);
    if (!c_to_v)
      throw std::runtime_error("Missing cell-vertex connectivity.");

    const std::vector<int> entity_layout
        = cmap_dof_layout.entity_closure_dofs(entity_dim, 0);

    // Find map from entity vertex to local (w.r.t. dof numbering on the
    // entity) dof number. E.g., if there are dofs on entity [0 3 6 7 9]
    // and dofs 3 and 7 belong to vertices, then this produces map [1,
    // 3].
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
    std::vector<std::int64_t> entities_vertices(shape_e_0
                                                * num_vert_per_entity);
    for (std::size_t e = 0; e < shape_e_0; ++e)
    {
      for (std::size_t i = 0; i < num_vert_per_entity; ++i)
      {
        entities_vertices[e * num_vert_per_entity + i]
            = entities[e * shape_e_1 + entity_vertex_dofs[i]];
      }
    }

    std::vector<std::vector<std::int64_t>> entities_send(comm_size);
    std::vector<std::vector<std::int32_t>> data_send(comm_size);
    for (std::size_t e = 0; e < shape_e_0; ++e)
    {
      std::span<std::int64_t> entity(entities_vertices.data()
                                         + e * num_vert_per_entity,
                                     num_vert_per_entity);
      std::sort(entity.begin(), entity.end());

      // Determine postmaster based on lowest entity node
      const std::int32_t p
          = dolfinx::MPI::index_owner(comm_size, entity.front(), num_nodes_g);
      entities_send[p].insert(entities_send[p].end(), entity.begin(),
                              entity.end());
      data_send[p].push_back(data[e]);
    }

    LOG(INFO) << "XDMF send entity keys size: (" << shape_e_0 << ")";
    // TODO: Pack into one MPI call
    graph::AdjacencyList<std::int64_t> entities_recv
        = all_to_all(comm, graph::AdjacencyList<std::int64_t>(entities_send));
    graph::AdjacencyList<std::int32_t> data_recv
        = all_to_all(comm, graph::AdjacencyList<std::int32_t>(data_send));

    return std::pair(entities_recv, data_recv);
  };

  const auto [entities_recv, data_recv] = postmaster_global_ent_sendrecv(
      topology, num_nodes_g, cmap_dof_layout, entity_dim, entities, data);

  // -------------------
  // 3. As 'postmaster', send back the entity key (vertex list) and tag
  //    value to ranks that possibly need the data. Do this based on the
  //    first node index in the entity key.

  // NOTE: Could: (i) use a std::unordered_multimap, or (ii) only send
  // owned nodes to the postmaster and use map, unordered_map or
  // std::vector<pair>>, followed by a neighborhood all_to_all at the
  // end.

  auto postmaster_send_to_candidates
      = [](const mesh::Topology& topology, int entity_dim,
           const graph::AdjacencyList<std::int64_t>& nodes_g_recv,
           const graph::AdjacencyList<std::int64_t>& entities_recv,
           const graph::AdjacencyList<std::int32_t>& data_recv)
  {
    const MPI_Comm comm = topology.comm();
    const int comm_size = dolfinx::MPI::size(comm);

    auto cell_types = topology.cell_types();
    if (cell_types.size() > 1)
      throw std::runtime_error("cell type IO");
    const std::size_t num_vert_per_entity = mesh::cell_num_entities(
        mesh::cell_entity_type(cell_types.back(), entity_dim, 0), 0);

    // Build map from global node index to ranks that have the node
    std::multimap<std::int64_t, int> node_to_rank;
    for (int p = 0; p < nodes_g_recv.num_nodes(); ++p)
    {
      auto nodes = nodes_g_recv.links(p);
      for (std::int32_t node : nodes)
        node_to_rank.insert({node, p});
    }

    // Figure out which processes are owners of received nodes
    std::vector<std::vector<std::int64_t>> send_nodes_owned(comm_size);
    std::vector<std::vector<std::int32_t>> send_vals_owned(comm_size);
    const std::size_t shape0
        = entities_recv.array().size() / num_vert_per_entity;
    const std::size_t shape1 = num_vert_per_entity;
    const std::vector<std::int32_t>& _data_recv = data_recv.array();
    assert(_data_recv.size() == shape0);
    for (std::size_t e = 0; e < shape0; ++e)
    {
      std::span e_recv(entities_recv.array().data() + e * shape1, shape1);

      // Find ranks that have node0
      auto [it0, it1] = node_to_rank.equal_range(e_recv.front());
      for (auto it = it0; it != it1; ++it)
      {
        const int p1 = it->second;
        send_nodes_owned[p1].insert(send_nodes_owned[p1].end(), e_recv.begin(),
                                    e_recv.end());
        send_vals_owned[p1].push_back(_data_recv[e]);
      }
    }

    // TODO: Pack into one MPI call
    const int send_val_size = std::transform_reduce(
        send_vals_owned.begin(), send_vals_owned.end(), 0, std::plus{},
        [](const std::vector<std::int32_t>& v) { return v.size(); });
    LOG(INFO) << "XDMF return entity and value data size:(" << send_val_size
              << ")";
    graph::AdjacencyList<std::int64_t> recv_ents = all_to_all(
        comm, graph::AdjacencyList<std::int64_t>(send_nodes_owned));
    graph::AdjacencyList<std::int32_t> recv_vals
        = all_to_all(comm, graph::AdjacencyList<std::int32_t>(send_vals_owned));

    return std::pair(std::move(recv_ents), std::move(recv_vals));
  };

  const auto [recv_ents, recv_vals] = postmaster_send_to_candidates(
      topology, entity_dim, nodes_g_recv, entities_recv, data_recv);

  // -------------------
  // 4. From the received (key, value) data, determine which keys
  //    (entities) are on this process.

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
      [&cell_vertex_dofs](
          const mesh::Topology& topology, std::span<const std::int64_t> nodes_g,
          MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
              const std::int32_t,
              MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
              x_dofmap,
          int entity_dim, std::span<const std::int64_t> recv_ents,
          std::span<const std::int32_t> recv_vals)
  {
    // Build map from input global indices to local vertex numbers
    LOG(INFO) << "XDMF build map";

    auto cell_types = topology.cell_types();
    const std::size_t num_vert_per_entity = mesh::cell_num_entities(
        mesh::cell_entity_type(cell_types.back(), entity_dim, 0), 0);
    auto c_to_v = topology.connectivity(topology.dim(), 0);
    if (!c_to_v)
      throw std::runtime_error("Missing cell-vertex connectivity.");

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

    std::vector<std::int32_t> entities_new, data_new;
    entities_new.reserve(recv_ents.size());
    data_new.reserve(recv_vals.size());
    std::vector<std::int32_t> entity(num_vert_per_entity);
    for (std::size_t e = 0; e < recv_ents.size() / num_vert_per_entity; ++e)
    {
      bool entity_found = true;
      for (std::size_t i = 0; i < num_vert_per_entity; ++i)
      {
        if (auto it
            = input_idx_to_vertex.find(recv_ents[e * num_vert_per_entity + i]);
            it == input_idx_to_vertex.end())
        {
          // As soon as this received index is not in locally owned input
          // global indices skip the entire entity
          entity_found = false;
          break;
        }
        else
          entity[i] = it->second;
      }

      if (entity_found == true)
      {
        entities_new.insert(entities_new.end(), entity.begin(), entity.end());
        data_new.push_back(recv_vals[e]);
      }
    }

    return std::pair(std::move(entities_new), std::move(data_new));
  };
  auto [entities_new, data_new]
      = determine_my_entities(topology, nodes_g, xdofmap, entity_dim,
                              recv_ents.array(), recv_vals.array());

  auto Xdetermine_my_entities =
      [&cell_vertex_dofs](
          const mesh::Topology& topology, std::span<const std::int64_t> nodes_g,
          MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
              const std::int32_t,
              MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
              x_dofmap,
          int entity_dim, const std::span<const std::int64_t> recv_ents,
          int num_vert_per_entity)
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
  auto [Xentities_new, Xdata_new]
      = Xdetermine_my_entities(topology, nodes_g, xdofmap, entity_dim,
                               entity_data_revc, num_vert_per_entity);

  // std::cout << "Sizes: " << entities_new.size() << ", " <<
  // Xentities_new.size()
  //           << std::endl;
  // std::cout << "TestA: " << int(Xentities_new == Xentities_new) << std::endl;
  // std::cout << "TestB: " << int(data_new == Xdata_new) << std::endl;
  // for (std::size_t i = 0; i < data_new.size(); ++i)
  //   std::cout << "Data: " << data_new[i] << ", " << Xdata_new[i] <<
  //   std::endl;

  {
    std::vector<std::pair<std::vector<std::int64_t>, std::int32_t>> oldd, newd;
    for (std::size_t i = 0; i < Xentities_new.size() / num_vert_per_entity; ++i)
    {
      std::vector<std::int64_t> e(
          Xentities_new.begin() + i * num_vert_per_entity,
          Xentities_new.begin() + i * num_vert_per_entity
              + num_vert_per_entity);
      newd.push_back(std::pair(e, Xdata_new[i]));
    }
    for (std::size_t i = 0; i < entities_new.size() / num_vert_per_entity; ++i)
    {
      std::vector<std::int64_t> e(
          entities_new.begin() + i * num_vert_per_entity,
          entities_new.begin() + i * num_vert_per_entity + num_vert_per_entity);
      oldd.push_back(std::pair(e, data_new[i]));
    }
    std::sort(newd.begin(), newd.end());
    std::sort(oldd.begin(), oldd.end());
    assert(newd == oldd);
  }

  return {std::move(Xentities_new), std::move(Xdata_new)};
  // return {std::move(entities_new), std::move(data_new)};
}
//-----------------------------------------------------------------------------
