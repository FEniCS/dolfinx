// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "xdmf_mesh.h"
#include "xdmf_read.h"
#include "xdmf_utils.h"

namespace pugi
{
class xml_document;
} // namespace pugi

namespace dolfinx
{
namespace function
{
class Function;
}

namespace io
{
/// Low-level methods for reading XDMF files
namespace xdmf_mf
{

/// TODO: Document and revise. Can it be removed?
/// Remap meshfunction data, scattering data to appropriate processes
template <typename T>
Eigen::Array<T, Eigen::Dynamic, 1>
remap_meshfunction_data(const mesh::Mesh& mesh, const int dim,
                        const std::vector<std::int64_t>& topology_data,
                        const std::vector<T>& value_data)
{
  // This function could use more local communication by sending data to
  // the vertex owner, who could then send the data via a neighbourhood
  // communicator to the owner.

  mesh::CellType cell_type
      = mesh::cell_entity_type(mesh.topology().cell_type(), dim);

  // FIXME: This should be nodes per entity
  const int vertices_per_entity = mesh::num_cell_vertices(cell_type);

  const MPI_Comm comm = mesh.mpi_comm();
  const int size = MPI::size(comm);

  assert(topology_data.size() % vertices_per_entity == 0);
  const std::int32_t num_entities = topology_data.size() / vertices_per_entity;

  // Send (sorted) entity topology and data to a post-office process
  // determined by the lowest global vertex index of the entity
  std::vector<std::vector<std::int64_t>> send_topology(size);
  std::vector<std::vector<T>> send_values(size);

  // std::cout << "Topology data (in): " << topology_data.size() << std::endl;
  // for (auto d : topology_data)
  //   std::cout << "  " << d << std::endl;

  // Get max node index
  std::int64_t max_node = 0;
  if (!topology_data.empty())
  {
    const auto max_it
        = std::max_element(topology_data.begin(), topology_data.end());
    max_node = MPI::max(comm, *max_it);
  }
  else
    max_node = MPI::max(comm, 0);

  for (int i = 0; i < num_entities; ++i)
  {
    // FIXME: This dynamic allocation inside a loop is bad
    std::vector<std::int64_t> cell(
        topology_data.begin() + i * vertices_per_entity,
        topology_data.begin() + (i + 1) * vertices_per_entity);
    std::sort(cell.begin(), cell.end());

    // Use first vertex to decide where to send this data
    const int dest = MPI::index_owner(size, cell.front(), max_node);
    send_topology[dest].insert(send_topology[dest].end(), cell.begin(),
                               cell.end());
    send_values[dest].push_back(value_data[i]);
  }

  // std::cout << "T Data to send (0): " << num_entities << ", "
  //           << send_topology.size() << std::endl;
  // for (auto d : send_topology[0])
  //   std::cout << "  " << d << std::endl;

  std::vector<std::vector<std::int64_t>> receive_topology(size);
  std::vector<std::vector<T>> receive_values(size);
  MPI::all_to_all(comm, send_topology, receive_topology);
  MPI::all_to_all(comm, send_values, receive_values);

  auto map = mesh.topology().index_map(dim);
  assert(map);
  assert(map->block_size() == 1);
  // auto c_to_v = mesh.topology().connectivity(dim, 0);
  // assert(c_to_v);

  // Generate requests for data from remote processes, based on the
  // first vertex of the mesh::MeshEntities which belong on this process
  // Send our process number, and our local index, so it can come back
  // directly to the right place
  std::vector<std::vector<std::int64_t>> send_requests(size);
  const int rank = MPI::rank(comm);
  const std::vector<std::int64_t>& global_indices
      = mesh.geometry().global_indices();

  auto map_g = mesh.geometry().index_map();
  assert(map_g);

  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const int num_cells = map->size_local() + map->num_ghosts();
  const int tdim = mesh.topology().dim();
  if (dim == tdim)
  {
    for (int c = 0; c < num_cells; ++c)
    {
      std::vector<std::int64_t> cell;
      auto nodes = x_dofmap.links(c);
      for (int v = 0; v < nodes.rows(); ++v)
      {
        std::int32_t local_index = nodes(v);
        cell.push_back(global_indices[local_index]);
      }
      std::sort(cell.begin(), cell.end());

      // Use first vertex to decide where to send this request
      int dest = MPI::index_owner(size, cell.front(), max_node);

      // Map to this process and local index by appending to send data
      cell.push_back(c);
      cell.push_back(rank);
      send_requests[dest].insert(send_requests[dest].end(), cell.begin(),
                                 cell.end());
    }
  }
  else
  {
    mesh.create_connectivity(dim, tdim);
    auto e_to_c = mesh.topology().connectivity(dim, tdim);
    assert(e_to_c);
    mesh.create_connectivity(dim, 0);
    auto e_to_v = mesh.topology().connectivity(dim, 0);
    assert(e_to_v);
    mesh.create_connectivity(tdim, 0);
    auto c_to_v = mesh.topology().connectivity(tdim, 0);
    assert(c_to_v);

    for (int e = 0; e < num_cells; ++e)
    {
      std::vector<std::int64_t> cell;

      // Get first attached cell
      std::int32_t c = e_to_c->links(e)[0];
      auto cell_vertices = c_to_v->links(c);
      auto nodes = x_dofmap.links(c);

      auto vertices = e_to_v->links(e);
      for (int v = 0; v < vertices.rows(); ++v)
      {
        const int v_index = v;
        const int vertex = vertices[v_index];
        auto it
            = std::find(cell_vertices.data(),
                        cell_vertices.data() + cell_vertices.rows(), vertex);
        assert(it != (cell_vertices.data() + cell_vertices.rows()));
        const int local_cell_vertex = std::distance(cell_vertices.data(), it);

        const std::int32_t local_index = nodes[local_cell_vertex];
        // std::cout << "   ind: " << local_index << ", "
        //           << global_indices[local_index] << std::endl;

        cell.push_back(global_indices[local_index]);
      }

      std::sort(cell.begin(), cell.end());

      // Use first vertex to decide where to send this request
      int dest = MPI::index_owner(size, cell.front(), max_node);

      // Map to this process and local index by appending to send data
      cell.push_back(e);
      cell.push_back(rank);
      send_requests[dest].insert(send_requests[dest].end(), cell.begin(),
                                 cell.end());
    }
  }

  std::vector<std::vector<std::int64_t>> receive_requests(size);
  MPI::all_to_all(comm, send_requests, receive_requests);

  // At this point, the data with its associated vertices is in
  // receive_values and receive_topology and the final destinations
  // are stored in receive_requests as
  // [vertices][index][process][vertices][index][process]...  Some
  // data will have more than one destination

  // Create a map from the topology vector to the desired data
  std::map<std::vector<std::int64_t>, T> cell_to_data;
  for (std::size_t i = 0; i < receive_values.size(); ++i)
  {
    assert(receive_values[i].size() * vertices_per_entity
           == receive_topology[i].size());
    auto p = receive_topology[i].begin();
    for (std::size_t j = 0; j < receive_values[i].size(); ++j)
    {
      const std::vector<std::int64_t> cell(p, p + vertices_per_entity);
      cell_to_data.insert({cell, receive_values[i][j]});
      p += vertices_per_entity;
    }
  }

  // Clear vectors for reuse - now to send values and indices to final
  // destination
  send_topology = std::vector<std::vector<std::int64_t>>(size);
  send_values = std::vector<std::vector<T>>(size);

  // Go through requests, which are stacked as [vertex, vertex, ...]
  // [index] [proc] etc.  Use the vertices as the key for the map
  // (above) to retrieve the data to send to proc
  for (std::size_t i = 0; i < receive_requests.size(); ++i)
  {
    for (auto p = receive_requests[i].begin(); p != receive_requests[i].end();
         p += (vertices_per_entity + 2))
    {
      const std::vector<std::int64_t> cell(p, p + vertices_per_entity);
      const int remote_index = *(p + vertices_per_entity);
      const int dest = *(p + vertices_per_entity + 1);

      const auto find_cell = cell_to_data.find(cell);
      assert(find_cell != cell_to_data.end());
      send_values[dest].push_back(find_cell->second);
      send_topology[dest].push_back(remote_index);
    }
  }

  MPI::all_to_all(comm, send_topology, receive_topology);
  MPI::all_to_all(comm, send_values, receive_values);

  const int num_values = map->size_local() + map->num_ghosts();
  Eigen::Array<T, Eigen::Dynamic, 1> mf_values(num_values);

  // At this point, receive_topology should only list the local indices
  // and received values should have the appropriate values for each
  for (std::size_t i = 0; i < receive_values.size(); ++i)
  {
    assert(receive_values[i].size() == receive_topology[i].size());
    for (std::size_t j = 0; j < receive_values[i].size(); ++j)
      mf_values[receive_topology[i][j]] = receive_values[i][j];
  }

  return mf_values;
}
//----------------------------------------------------------------------------

/// Return data which is local
template <typename T>
std::vector<T> compute_value_data(const mesh::MeshFunction<T>& meshfunction)
{
  // Create vector to store data
  std::vector<T> value_data;
  value_data.reserve(meshfunction.values().size());

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = meshfunction.mesh();
  assert(mesh);
  const int cell_dim = meshfunction.dim();

  // Get reference to mesh function data array
  const Eigen::Array<T, Eigen::Dynamic, 1>& mf_values = meshfunction.values();
  auto map = mesh->topology().index_map(cell_dim);
  assert(map);
  for (int e = 0; e < map->size_local(); ++e)
    value_data.push_back(mf_values[e]);

  return value_data;
}

/// TODO
template <typename T>
void write_mesh_function(MPI_Comm comm,
                         const mesh::MeshFunction<T>& meshfunction,
                         pugi::xml_node& domain_node, hid_t h5_id, int counter)
{
  if (meshfunction.values().size() == 0)
    throw std::runtime_error("No values in MeshFunction");

  // Get mesh
  assert(meshfunction.mesh());
  std::shared_ptr<const mesh::Mesh> mesh = meshfunction.mesh();
  const int dim = meshfunction.dim();
  const int tdim = mesh->topology().dim();

  assert(domain_node);

  const std::string mf_name = "/MeshFunction/" + std::to_string(counter);

  // If adding a mesh::MeshFunction of topology dimension dim() to an
  // existing mesh::Mesh, do not rewrite mesh::Mesh
  //
  // FIXME: do some checks on the existing mesh::Mesh to make sure it is
  // the same as the meshfunction's mesh.

  pugi::xml_node grid_node = domain_node.child("Grid");
  const bool grid_empty = grid_node.empty();

  // Check existing mesh::Mesh for compatibility.
  if (!grid_empty)
  {
    pugi::xml_node topology_node = grid_node.child("Topology");
    assert(topology_node);
    std::pair<std::string, int> cell_type_str
        = xdmf_utils::get_cell_type(topology_node);
    if (mesh::to_string(mesh->topology().cell_type()) != cell_type_str.first)
    {
      throw std::runtime_error(
          "Incompatible Mesh type. Try writing the Mesh to XDMF first");
    }
  }

  if (grid_empty or dim != tdim)
  {
    // Make new grid node
    grid_node = domain_node.append_child("Grid");
    assert(grid_node);
    grid_node.append_attribute("Name") = "mesh";
    grid_node.append_attribute("GridType") = "Uniform";

    xdmf_mesh::add_topology_data(comm, grid_node, h5_id, mf_name,
                                 mesh->topology(), mesh->geometry(), dim);

    // Add geometry node if none already, else link back to first
    // existing Mesh
    if (grid_empty)
    {
      xdmf_mesh::add_geometry_data(comm, grid_node, h5_id, mf_name,
                                   mesh->geometry());
    }
    else
    {
      // Add geometry node (reference)
      pugi::xml_node geometry_node = grid_node.append_child("Geometry");
      assert(geometry_node);
      geometry_node.append_attribute("Reference") = "XML";
      geometry_node.append_child(pugi::node_pcdata)
          .set_value("/Xdmf/Domain/Grid/Geometry");
    }
  }

  // Add attribute node with values
  pugi::xml_node attribute_node = grid_node.append_child("Attribute");
  assert(attribute_node);
  attribute_node.append_attribute("Name") = meshfunction.name.c_str();
  attribute_node.append_attribute("AttributeType") = "Scalar";
  attribute_node.append_attribute("Center") = "Cell";

  auto map = mesh->topology().index_map(dim);
  assert(map);
  const std::int64_t num_values = map->size_global();
  // Add attribute DataItem node and write data

  // Copy values to vector, removing duplicates
  std::vector<T> values = compute_value_data(meshfunction);

  const std::int64_t offset
      = dolfinx::MPI::global_offset(comm, map->size_local(), true);
  const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);
  xdmf_utils::add_data_item(attribute_node, h5_id, mf_name + "/values", values,
                            offset, {num_values, 1}, "", use_mpi_io);
}

/// TODO
template <typename T>
mesh::MeshFunction<T> read_mesh_function(std::shared_ptr<const mesh::Mesh> mesh,
                                         std::string name, std::string filename,
                                         const pugi::xml_node& domain_node)
{
  // Check all top level Grid nodes for suitable dataset
  pugi::xml_node grid_node;
  pugi::xml_node value_node;

  // Using lambda to exit nested loops
  [&] {
    for (pugi::xml_node node : domain_node.children("Grid"))
    {
      for (pugi::xml_node attr_node : node.children("Attribute"))
      {
        if (attr_node
            and (name == "" or name == attr_node.attribute("Name").as_string()))
        {
          grid_node = node;
          value_node = attr_node;
          return;
        }
      }
    }
  }();

  // Check if a TimeSeries (old format), in which case the Grid will
  // be down one level

  //   if (!grid_node)
  //   {
  //     pugi::xml_node grid_node1 = domain_node.child("Grid");
  //     if (grid_node1)
  //     {
  //       for (pugi::xml_node node : grid_node1.children("Grid"))
  //       {
  //         pugi::xml_node attr_node = node.child("Attribute");
  //         if (attr_node
  //             and (name == "" or name ==
  //             attr_node.attribute("Name").as_string()))
  //         {
  //           grid_node = node;
  //           value_node = attr_node;
  //           break;
  //         }
  //       }
  //     }
  //   }

  //   // Still can't find it
  //   if (!grid_node)
  //   {
  //     throw std::runtime_error("Mesh Grid with data Attribute not found in
  //     XDMF");
  //   }

  // Get topology node
  pugi::xml_node topology_node = grid_node.child("Topology");
  assert(topology_node);

  // Get cell type and topology of mesh::MeshFunction (may be different
  // from Mesh)
  const auto cell_type_str = xdmf_utils::get_cell_type(topology_node);
  assert(cell_type_str.second == 1);
  mesh::CellType cell_type = mesh::to_type(cell_type_str.first);
  const int num_vertices_per_cell = mesh::cell_num_entities(cell_type, 0);
  const int dim = mesh::cell_dim(cell_type);

  const std::int64_t num_entities_global
      = xdmf_utils::get_num_cells(topology_node);

  mesh->create_entities(dim);
  assert(mesh->topology().index_map(dim));
  assert(mesh->topology().index_map(dim)->size_global() == num_entities_global);

  boost::filesystem::path xdmf_filename(filename);
  const boost::filesystem::path parent_path = xdmf_filename.parent_path();

  // Get topology dataset
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  assert(topology_data_node);
  const std::vector<std::int64_t> topology_data
      = xdmf_read::get_dataset<std::int64_t>(mesh->mpi_comm(),
                                             topology_data_node, parent_path);
  assert(topology_data.size() % num_vertices_per_cell == 0);

  // Get value dataset
  pugi::xml_node value_data_node = value_node.child("DataItem");
  assert(value_data_node);
  std::vector<T> value_data = xdmf_read::get_dataset<T>(
      mesh->mpi_comm(), value_data_node, parent_path);

  // Distribute data to owner rank

  // Create mesh function and scatter/gather data across processes
  mesh::MeshFunction<T> mf(mesh, dim, 0);
  mf.values()
      = xdmf_mf::remap_meshfunction_data(*mesh, dim, topology_data, value_data);

  return mf;
}
//-----------------------------------------------------------------------------

} // namespace xdmf_mf
} // namespace io
} // namespace dolfinx
