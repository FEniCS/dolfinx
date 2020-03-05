// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

// #include "HDF5File.h"
#include "xdmf_mesh.h"
#include "xdmf_read.h"
#include "xdmf_utils.h"
#include <dolfinx/fem/ElementDofLayout.h>

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
namespace xdmf_mvc
{

/// TODO
template <typename T>
void write(MPI_Comm comm, const mesh::MeshValueCollection<T>& mvc,
           pugi::xml_node& domain_node, hid_t h5_id, int counter)
{
  std::shared_ptr<const mesh::Mesh> mesh = mvc.mesh();
  assert(mesh);

  const int tdim = mesh->topology().dim();
  const int gdim = mesh->geometry().dim();

  // Check domain node for existing mesh::Mesh Grid and check it is
  // compatible with this mesh::MeshValueCollection, or if none, add
  // Mesh
  pugi::xml_node grid_node = domain_node.child("Grid");
  if (grid_node.empty())
    xdmf_mesh::add_mesh(comm, domain_node, h5_id, *mesh, "/Mesh");
  else
  {
    // Check topology
    pugi::xml_node topology_node = grid_node.child("Topology");
    assert(topology_node);
    const std::int64_t ncells = mesh->num_entities_global(tdim);
    pugi::xml_attribute num_cells_attr
        = topology_node.attribute("NumberOfElements");
    assert(num_cells_attr);
    if (num_cells_attr.as_llong() != ncells)
    {
      throw std::runtime_error("Cannot add MeshValueCollection to file. "
                               "Incompatible mesh.");
    }

    // Check geometry
    pugi::xml_node geometry_node = grid_node.child("Geometry");
    assert(geometry_node);
    pugi::xml_node geometry_data_node = geometry_node.child("DataItem");
    assert(geometry_data_node);
    const std::string dims_str
        = geometry_data_node.attribute("Dimensions").as_string();
    std::vector<std::string> dims_list;
    boost::split(dims_list, dims_str, boost::is_any_of(" "));
    const std::int64_t npoints = mesh->num_entities_global(0);
    if (boost::lexical_cast<std::int64_t>(dims_list[0]) != npoints
        or boost::lexical_cast<std::int64_t>(dims_list[1]) != (int)gdim)
    {
      throw std::runtime_error("Cannot add MeshValueCollection to file. "
                               "Incompatible mesh.");
    }
  }

  // Add new grid node, for MVC mesh
  pugi::xml_node mvc_grid_node = domain_node.append_child("Grid");
  assert(mvc_grid_node);
  mvc_grid_node.append_attribute("Name") = mvc.name.c_str();
  mvc_grid_node.append_attribute("GridType") = "Uniform";

  // Add topology node and attributes
  const int dim = mvc.dim();

  // Get entity 'cell' type
  const mesh::CellType entity_cell_type
      = mesh::cell_entity_type(mesh->topology().cell_type(), dim);
  const int num_nodes_per_cell
      = mesh->geometry().dof_layout().num_entity_closure_dofs(dim);
  const std::string vtk_cell_str
      = xdmf_utils::vtk_cell_type_str_new(entity_cell_type, num_nodes_per_cell);

  const std::map<std::pair<std::size_t, std::size_t>, T>& values = mvc.values();
  const std::int64_t num_cells = values.size();
  const std::int64_t num_cells_global = MPI::sum(mesh->mpi_comm(), num_cells);

  pugi::xml_node topology_node = mvc_grid_node.append_child("Topology");
  assert(topology_node);
  topology_node.append_attribute("NumberOfElements")
      = std::to_string(num_cells_global).c_str();
  topology_node.append_attribute("TopologyType") = vtk_cell_str.c_str();
  topology_node.append_attribute("NodesPerElement")
      = std::to_string(num_nodes_per_cell).c_str();

  std::vector<std::int32_t> topology_data;
  std::vector<T> value_data;
  topology_data.reserve(num_cells * num_nodes_per_cell);
  value_data.reserve(num_cells);

  auto map = mesh->topology().index_map(0);
  assert(map);
  const std::vector<std::int64_t> global_indices = map->global_indices(false);
  mesh->create_connectivity(tdim, dim);

  auto e_to_c = mesh->topology().connectivity(dim, tdim);
  assert(e_to_c);
  auto e_to_v = mesh->topology().connectivity(dim, 0);
  assert(e_to_v);

  if (dim == tdim)
  {
    for (auto& p : values)
    {
      const std::int32_t entity_index = p.first.first;
      auto vertices = e_to_v->links(entity_index);
      for (int i = 0; i < vertices.rows(); ++i)
        topology_data.push_back(global_indices[vertices[i]]);
      value_data.push_back(p.second);
    }
  }
  else
  {
    auto c_to_e = mesh->topology().connectivity(tdim, dim);
    assert(c_to_e);
    for (auto& p : values)
    {
      const std::int32_t entity_index = p.first.first;
      const std::int32_t local_index = p.first.second;
      const std::int32_t c = e_to_c->links(entity_index)[0];
      const std::int32_t e = c_to_e->links(c)[local_index];

      // if cell is actually a vertex
      if (dim == 0)
        topology_data.push_back(global_indices[c]);
      else
      {
        auto vertices = e_to_v->links(e);
        for (int i = 0; i < vertices.rows(); ++i)
          topology_data.push_back(global_indices[vertices[i]]);
      }

      value_data.push_back(p.second);
    }
  }

  const std::string mvc_dataset_name
      = "/MeshValueCollection/" + std::to_string(counter);

  const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);

  const std::int64_t offset = dolfinx::MPI::global_offset(
      comm, topology_data.size() / num_nodes_per_cell, true);
  const std::int64_t num_values = MPI::sum(mesh->mpi_comm(), value_data.size());
  xdmf_utils::add_data_item(
      topology_node, h5_id, mvc_dataset_name + "/topology", topology_data,
      offset, {num_values, num_nodes_per_cell}, "Int", use_mpi_io);

  // Add geometry node (share with main mesh::Mesh)
  pugi::xml_node geometry_node = mvc_grid_node.append_child("Geometry");
  assert(geometry_node);
  geometry_node.append_attribute("Reference") = "XML";
  geometry_node.append_child(pugi::node_pcdata)
      .set_value("/Xdmf/Domain/Grid/Geometry");

  // Add attribute node with values
  pugi::xml_node attribute_node = mvc_grid_node.append_child("Attribute");
  assert(attribute_node);
  attribute_node.append_attribute("Name") = mvc.name.c_str();
  attribute_node.append_attribute("AttributeType") = "Scalar";
  attribute_node.append_attribute("Center") = "Cell";

  const std::int64_t offset1
      = dolfinx::MPI::global_offset(comm, value_data.size(), true);
  xdmf_utils::add_data_item(attribute_node, h5_id, mvc_dataset_name + "/values",
                            value_data, offset1, {num_values, 1}, "",
                            use_mpi_io);
}

/// TODO
template <typename T>
mesh::MeshValueCollection<T> read(std::shared_ptr<const mesh::Mesh> mesh,
                                  std::string name, std::string filename,
                                  const pugi::xml_node& domain_node)
{
  // Check all Grid nodes for suitable dataset
  pugi::xml_node grid_node;
  for (pugi::xml_node node : domain_node.children("Grid"))
  {
    pugi::xml_node value_node = node.child("Attribute");
    if (value_node
        and (name == "" or name == value_node.attribute("Name").as_string()))
    {
      grid_node = node;
      break;
    }
  }

  // Get MVC topology node
  pugi::xml_node topology_node = grid_node.child("Topology");
  assert(topology_node);

  // Get description of MVC cell type and dimension from topology node
  auto cell_type_str = xdmf_utils::get_cell_type(topology_node);

  const mesh::CellType cell_type = mesh::to_type(cell_type_str.first);
  const int dim = mesh::cell_dim(cell_type);

  const int num_verts_per_entity = mesh::num_cell_vertices(cell_type);
  const int num_nodes_per_entity
      = mesh->geometry().dof_layout().num_entity_closure_dofs(dim);

  // Read MVC topology
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  assert(topology_data_node);
  boost::filesystem::path xdmf_filename(filename);
  const boost::filesystem::path parent_path = xdmf_filename.parent_path();
  std::vector<std::int64_t> topology_data
      = xdmf_read::get_dataset<std::int64_t>(mesh->mpi_comm(),
                                             topology_data_node, parent_path);

  // Read values associated with each mesh::MeshEntity described by topology
  pugi::xml_node attribute_node = grid_node.child("Attribute");
  assert(attribute_node);
  pugi::xml_node attribute_data_node = attribute_node.child("DataItem");
  assert(attribute_data_node);
  std::vector<T> values_data = xdmf_read::get_dataset<T>(
      mesh->mpi_comm(), attribute_data_node, parent_path);

  // Ensure the mesh dimension is initialised
  mesh->create_entities(dim);
  const std::int64_t global_vertex_range = mesh->num_entities_global(0);
  const int num_processes = MPI::size(mesh->mpi_comm());

  // Send entities to processes based on the lowest vertex index
  std::vector<std::vector<std::int32_t>> send_entities(num_processes);
  std::vector<std::vector<std::int32_t>> recv_entities(num_processes);

  auto map = mesh->topology().index_map(0);
  assert(map);
  const std::vector<std::int64_t> global_indices = map->global_indices(false);

  auto map_e = mesh->topology().index_map(dim);
  assert(map_e);
  std::vector<std::int32_t> v(num_verts_per_entity);
  // for (auto& m : mesh::MeshRange(*mesh, dim, mesh::MeshRangeType::ALL))
  for (int e = 0; e < map_e->size_local() + map_e->num_ghosts(); ++e)
  {
    if (dim == 0)
      v[0] = global_indices[e];
    else
    {
      v.clear();
      auto e_to_v = mesh->topology().connectivity(dim, 0);
      auto vertices = e_to_v->links(e);
      for (int i = 0; i < vertices.rows(); ++i)
        v.push_back(global_indices[vertices[i]]);
      std::sort(v.begin(), v.end());
    }

    int dest = MPI::index_owner(num_processes, v[0], global_vertex_range);
    send_entities[dest].push_back(e);
    send_entities[dest].insert(send_entities[dest].end(), v.begin(), v.end());
  }
  MPI::all_to_all(mesh->mpi_comm(), send_entities, recv_entities);

  // Map from {entity vertex indices} to {process, local_index}
  std::map<std::vector<std::int32_t>, std::vector<std::int32_t>> entity_map;
  for (std::int32_t i = 0; i < num_processes; ++i)
  {
    for (auto it = recv_entities[i].begin(); it != recv_entities[i].end();
         it += (num_verts_per_entity + 1))
    {
      std::copy(it + 1, it + num_verts_per_entity + 1, v.begin());
      auto map_it = entity_map.insert({v, {i, *it}});
      if (!map_it.second)
      {
        // Entry already exists, add to it
        map_it.first->second.push_back(i);
        map_it.first->second.push_back(*it);
      }
    }
  }

  // Send data from mesh::MeshValueCollection to sorting process
  std::vector<std::vector<T>> send_data(num_processes);
  std::vector<std::vector<T>> recv_data(num_processes);
  // Reset send/recv arrays
  send_entities = std::vector<std::vector<std::int32_t>>(num_processes);
  recv_entities = std::vector<std::vector<std::int32_t>>(num_processes);

  std::vector<int> nodes_to_verts
      = mesh::cell_vertex_indices(cell_type, num_nodes_per_entity);
  std::vector<std::int32_t> entity_nodes(num_nodes_per_entity);

  std::int32_t i = 0;
  for (auto it = topology_data.begin(); it != topology_data.end();
       it += num_nodes_per_entity)
  {
    // Apply node to vertices mapping, this throws away
    // nodes read from the file
    for (int j = 0; j < num_verts_per_entity; ++j)
      v[j] = *(it + nodes_to_verts[j]);
    std::sort(v.begin(), v.end());

    std::size_t dest
        = MPI::index_owner(num_processes, v[0], global_vertex_range);
    send_entities[dest].insert(send_entities[dest].end(), v.begin(), v.end());
    send_data[dest].push_back(values_data[i]);
    ++i;
  }

  MPI::all_to_all(mesh->mpi_comm(), send_entities, recv_entities);
  MPI::all_to_all(mesh->mpi_comm(), send_data, recv_data);

  // Reset send arrays
  send_data = std::vector<std::vector<T>>(num_processes);
  send_entities = std::vector<std::vector<std::int32_t>>(num_processes);

  // Locate entity in map, and send back to data to owning processes
  for (std::int32_t i = 0; i != num_processes; ++i)
  {
    assert(recv_data[i].size() * num_verts_per_entity
           == recv_entities[i].size());

    for (std::size_t j = 0; j != recv_data[i].size(); ++j)
    {
      auto it = recv_entities[i].begin() + j * num_verts_per_entity;
      std::copy(it, it + num_verts_per_entity, v.begin());
      auto map_it = entity_map.find(v);

      if (map_it == entity_map.end())
      {
        throw std::runtime_error("Cannotfind entity in map. "
                                 "Error reading mesh::MeshValueCollection");
      }
      for (auto p = map_it->second.begin(); p != map_it->second.end(); p += 2)
      {
        const std::int32_t dest = *p;
        assert(dest < num_processes);
        send_entities[dest].push_back(*(p + 1));
        send_data[dest].push_back(recv_data[i][j]);
      }
    }
  }

  // Send to owning processes and set in mesh::MeshValueCollection
  MPI::all_to_all(mesh->mpi_comm(), send_entities, recv_entities);
  MPI::all_to_all(mesh->mpi_comm(), send_data, recv_data);

  mesh::MeshValueCollection<T> mvc(mesh, dim);
  for (std::int32_t i = 0; i != num_processes; ++i)
  {
    assert(recv_entities[i].size() == recv_data[i].size());
    for (std::size_t j = 0; j != recv_data[i].size(); ++j)
    {
      mvc.set_value(recv_entities[i][j], recv_data[i][j]);
    }
  }

  return mvc;
}

} // namespace xdmf_mvc
} // namespace io
} // namespace dolfinx