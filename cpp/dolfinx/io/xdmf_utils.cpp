// Copyright (C) 2012-2023 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_utils.h"
#include <array>
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
#include <span>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
template <typename T, std::size_t ndim>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, ndim>>;
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
  std::ranges::transform(cell_type, cell_type.begin(),
                         [](auto c) { return std::tolower(c); });
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
template <typename T>
std::pair<std::vector<std::int32_t>, std::vector<T>>
xdmf_utils::distribute_entity_data(
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
    std::ranges::transform(
        indices, std::back_inserter(dest_to_index),
        [size, num_nodes](auto n) {
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
//-----------------------------------------------------------------------------
/// @cond
template std::pair<std::vector<std::int32_t>, std::vector<double>>
xdmf_utils::distribute_entity_data(
    const mesh::Topology&, std::span<const std::int64_t>, std::int64_t,
    const fem::ElementDofLayout&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    std::span<const double>);
template std::pair<std::vector<std::int32_t>, std::vector<float>>
xdmf_utils::distribute_entity_data(
    const mesh::Topology&, std::span<const std::int64_t>, std::int64_t,
    const fem::ElementDofLayout&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    std::span<const float>);
template std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
xdmf_utils::distribute_entity_data(
    const mesh::Topology&, std::span<const std::int64_t>, std::int64_t,
    const fem::ElementDofLayout&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    std::span<const std::int32_t>);
template std::pair<std::vector<std::int32_t>, std::vector<std::complex<double>>>
xdmf_utils::distribute_entity_data(
    const mesh::Topology&, std::span<const std::int64_t>, std::int64_t,
    const fem::ElementDofLayout&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    std::span<const std::complex<double>>);
template std::pair<std::vector<std::int32_t>, std::vector<std::complex<float>>>
xdmf_utils::distribute_entity_data(
    const mesh::Topology&, std::span<const std::int64_t>, std::int64_t,
    const fem::ElementDofLayout&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    std::span<const std::complex<float>>);

/// @endcond
