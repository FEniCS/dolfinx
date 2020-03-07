// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_read.h"
#include "xdmf_utils.h"
#include "xdmf_write.h"

#include "HDF5File.h"
#include "HDF5Utility.h"
#include "XDMFFile.h"
#include "cells.h"
#include "pugixml.hpp"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/container/vector.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/defines.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <dolfinx/mesh/MeshValueCollection.h>
#include <iomanip>
#include <memory>
#include <petscvec.h>
#include <set>
#include <string>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
//-----------------------------------------------------------------------------
// Return a vector of numerical values from a vector of stringstream
template <typename T>
std::vector<T> string_to_vector(const std::vector<std::string>& x_str)
{
  std::vector<T> data;
  for (auto& v : x_str)
  {
    if (!v.empty())
      data.push_back(boost::lexical_cast<T>(v));
  }

  return data;
}
//-----------------------------------------------------------------------------
// Return a string of the form "x y"
template <typename X, typename Y>
std::string to_string(X x, Y y)
{
  return std::to_string(x) + " " + std::to_string(y);
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
XDMFFile::XDMFFile(MPI_Comm comm, const std::string filename, Encoding encoding)
    : _mpi_comm(comm), _filename(filename), _counter(0),
      _xml_doc(new pugi::xml_document), _encoding(encoding)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XDMFFile::~XDMFFile() { close(); }
//-----------------------------------------------------------------------------
void XDMFFile::close()
{
  // Close the HDF5 file
  _hdf5_file.reset();
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const mesh::Mesh& mesh)
{
  // Check that encoding
  if (_encoding == Encoding::ASCII and MPI::size(_mpi_comm.comm()) != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }

  // Open a HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
  std::unique_ptr<HDF5File> h5_file;
  if (_encoding == Encoding::HDF5)
  {
    // Open file
    h5_file = std::make_unique<HDF5File>(
        mesh.mpi_comm(), xdmf_utils::get_hdf5_filename(_filename), "w");
    assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }

  // Reset pugi doc
  _xml_doc->reset();

  // Add XDMF node and version attribute
  _xml_doc->append_child(pugi::node_doctype)
      .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
  pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
  assert(xdmf_node);
  xdmf_node.append_attribute("Version") = "3.0";
  xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";

  // Add domain node and add name attribute
  pugi::xml_node domain_node = xdmf_node.append_child("Domain");
  assert(domain_node);

  // Add the mesh Grid to the domain
  xdmf_write::add_mesh(_mpi_comm.comm(), domain_node, h5_id, mesh, "/Mesh");

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::write_checkpoint(const function::Function& u,
                                std::string function_name, double time_step)
{
  if (_encoding == Encoding::ASCII and MPI::size(_mpi_comm.comm()) != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }

  LOG(INFO) << "Writing function \"" << function_name << "\" to XDMF file \""
            << _filename << "\" with time step " << time_step;

  // If XML file exists load it to member _xml_doc
  if (boost::filesystem::exists(_filename))
  {
    LOG(WARNING) << "Appending to an existing XDMF XML file \"" << _filename
                 << "\"";

    pugi::xml_parse_result result = _xml_doc->load_file(_filename.c_str());
    assert(result);

    if (_xml_doc->select_node("/Xdmf/Domain").node().empty())
    {
      LOG(WARNING) << "File \"" << _filename
                   << "\" contains invalid XDMF. Writing new XDMF.";
    }
  }

  bool truncate_hdf = false;

  // If the XML file doesn't have expected structure (domain) reset the file
  // and create empty structure
  if (_xml_doc->select_node("/Xdmf/Domain").node().empty())
  {
    _xml_doc->reset();

    // Prepare new XML structure
    pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
    assert(xdmf_node);
    xdmf_node.append_attribute("Version") = "3.0";

    pugi::xml_node domain_node = xdmf_node.append_child("Domain");
    assert(domain_node);

    truncate_hdf = true;
  }

  if (truncate_hdf
      and boost::filesystem::exists(xdmf_utils::get_hdf5_filename(_filename)))
  {
    LOG(WARNING) << "HDF file \"" << xdmf_utils::get_hdf5_filename(_filename)
                 << "\" will be overwritten.";
  }

  // Open the HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
  if (_encoding == Encoding::HDF5)
  {
    if (truncate_hdf)
    {
      // We are writing for the first time, any HDF file must be overwritten
      _hdf5_file = std::make_unique<HDF5File>(
          _mpi_comm.comm(), xdmf_utils::get_hdf5_filename(_filename), "w");
    }
    else if (_hdf5_file)
    {
      // Pointer to HDF file is active, we are writing time series
      // or adding function with flush_output=false
    }
    else
    {
      // Pointer is empty, we are writing time series
      // or adding function to already flushed file
      _hdf5_file = std::unique_ptr<HDF5File>(new HDF5File(
          _mpi_comm.comm(), xdmf_utils::get_hdf5_filename(_filename), "a"));
    }
    assert(_hdf5_file);
    h5_id = _hdf5_file->h5_id();
  }

  // From this point _xml_doc points to a valid XDMF XML document
  // with expected structure

  // Find temporal grid with name equal to the name of function we're about
  // to save
  pugi::xml_node func_temporal_grid_node
      = _xml_doc
            ->select_node(("/Xdmf/Domain/Grid[@CollectionType='Temporal' and "
                           "@Name='"
                           + function_name + "']")
                              .c_str())
            .node();

  // If there is no such temporal grid then create one
  if (func_temporal_grid_node.empty())
  {
    func_temporal_grid_node
        = _xml_doc->select_node("/Xdmf/Domain").node().append_child("Grid");
    func_temporal_grid_node.append_attribute("GridType") = "Collection";
    func_temporal_grid_node.append_attribute("CollectionType") = "Temporal";
    func_temporal_grid_node.append_attribute("Name") = function_name.c_str();
  }
  else
  {
    LOG(INFO) << "XDMF time series for function \"" << function_name
              << "\" not empty. Appending.";
  }

  //
  // Write mesh
  //

  std::size_t counter = func_temporal_grid_node.select_nodes("Grid").size();
  std::string function_time_name
      = function_name + "_" + std::to_string(counter);

  const mesh::Mesh& mesh = *u.function_space()->mesh();
  xdmf_write::add_mesh(_mpi_comm.comm(), func_temporal_grid_node, h5_id, mesh,
                       function_name + "/" + function_time_name);

  // Get newly (by add_mesh) created Grid
  pugi::xml_node mesh_grid_node
      = func_temporal_grid_node.select_node("Grid[@Name='mesh']").node();
  assert(mesh_grid_node);

  // Change it's name to {function_name}_{counter}
  // where counter = number of children in temporal grid node
  mesh_grid_node.attribute("Name") = function_time_name.c_str();

  pugi::xml_node time_node = mesh_grid_node.append_child("Time");
  time_node.append_attribute("Value") = std::to_string(time_step).c_str();

#ifdef PETSC_USE_COMPLEX
  std::vector<std::string> components = {"real", "imag"};
#else
  std::vector<std::string> components = {""};
#endif

  // Write function u (for each of its components)
  for (const std::string component : components)
  {
    xdmf_write::add_function(_mpi_comm.comm(), mesh_grid_node, h5_id,
                             function_name + "/" + function_time_name, u,
                             function_name, mesh, component);
  }

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
  {
    LOG(INFO) << "Saving XML file \"" << _filename << "\" (only on rank = 0)";
    _xml_doc->save_file(_filename.c_str(), "  ");
  }

  // Close the HDF5 file if in "flush" mode
  if (_encoding == Encoding::HDF5 and flush_output)
  {
    LOG(INFO) << "Writing function in \"flush_output\" mode. HDF5 "
                 "file will be flushed (closed).";
    assert(_hdf5_file);
    _hdf5_file.reset();
  }
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const mesh::MeshValueCollection<int>& mvc)
{
  write_mesh_value_collection(mvc);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const mesh::MeshValueCollection<std::size_t>& mvc)
{
  write_mesh_value_collection(mvc);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const mesh::MeshValueCollection<double>& mvc)
{
  write_mesh_value_collection(mvc);
}
//-----------------------------------------------------------------------------
template <typename T>
void XDMFFile::write_mesh_value_collection(
    const mesh::MeshValueCollection<T>& mvc)
{
  // Check that encoding
  if (_encoding == Encoding::ASCII and MPI::size(_mpi_comm.comm()) != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }

  // Provide some very basic functionality for saving
  // mesh::MeshValueCollections mainly for saving values on a boundary
  // mesh

  assert(mvc.mesh());
  std::shared_ptr<const mesh::Mesh> mesh = mvc.mesh();
  const std::size_t tdim = mesh->topology().dim();
  const std::size_t gdim = mesh->geometry().dim();

  if (MPI::sum(mesh->mpi_comm(), mvc.size()) == 0)
  {
    throw std::runtime_error("Cannot save empty mesh::MeshValueCollection"
                             "No values in mesh::MeshValueCollection");
  }

  pugi::xml_node domain_node;
  std::string hdf_filemode = "a";
  if (_xml_doc->child("Xdmf").empty())
  {
    // Reset pugi
    _xml_doc->reset();

    // Add XDMF node and version attribute
    _xml_doc->append_child(pugi::node_doctype)
        .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
    assert(xdmf_node);
    xdmf_node.append_attribute("Version") = "3.0";
    xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";

    // Add domain node and add name attribute
    domain_node = xdmf_node.append_child("Domain");
    hdf_filemode = "w";
  }
  else
    domain_node = _xml_doc->child("Xdmf").child("Domain");

  assert(domain_node);

  // Open a HDF5 file if using HDF5 encoding
  hid_t h5_id = -1;
  std::unique_ptr<HDF5File> h5_file;
  if (_encoding == Encoding::HDF5)
  {
    // Open file
    h5_file = std::make_unique<HDF5File>(
        mesh->mpi_comm(), xdmf_utils::get_hdf5_filename(_filename),
        hdf_filemode);
    assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }

  // Check domain node for existing mesh::Mesh Grid and check it is
  // compatible with this mesh::MeshValueCollection, or if none, add
  // mesh::Mesh
  pugi::xml_node grid_node = domain_node.child("Grid");
  if (grid_node.empty())
    xdmf_write::add_mesh(_mpi_comm.comm(), domain_node, h5_id, *mesh, "/Mesh");
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
  const std::size_t cell_dim = mvc.dim();
  const std::size_t degree = 1;
  const std::string vtk_cell_str = xdmf_utils::vtk_cell_type_str(
      mesh::cell_entity_type(mesh->topology().cell_type(), cell_dim), degree);
  const std::int32_t num_vertices_per_cell = mesh::num_cell_vertices(
      cell_entity_type(mesh->topology().cell_type(), cell_dim));

  const std::map<std::pair<std::size_t, std::size_t>, T>& values = mvc.values();
  const std::int64_t num_cells = values.size();
  const std::int64_t num_cells_global = MPI::sum(mesh->mpi_comm(), num_cells);

  pugi::xml_node topology_node = mvc_grid_node.append_child("Topology");
  assert(topology_node);
  topology_node.append_attribute("NumberOfElements")
      = std::to_string(num_cells_global).c_str();
  topology_node.append_attribute("TopologyType") = vtk_cell_str.c_str();
  topology_node.append_attribute("NodesPerElement")
      = std::to_string(num_vertices_per_cell).c_str();

  std::vector<std::int32_t> topology_data;
  std::vector<T> value_data;
  topology_data.reserve(num_cells * num_vertices_per_cell);
  value_data.reserve(num_cells);

  auto map = mesh->topology().index_map(0);
  assert(map);
  const std::vector<std::int64_t> global_indices = map->global_indices(false);
  mesh->create_connectivity(tdim, cell_dim);
  for (auto& p : values)
  {
    mesh::MeshEntity cell(*mesh, tdim, p.first.first);
    if (cell_dim != tdim)
    {
      const std::int32_t entity_local_idx
          = cell.entities(cell_dim)[p.first.second];
      cell = mesh::MeshEntity(*mesh, cell_dim, entity_local_idx);
    }

    // if cell is actually a vertex
    if (cell.dim() == 0)
      topology_data.push_back(global_indices[cell.index()]);
    else
    {
      for (auto& v : mesh::EntityRange(cell, 0))
        topology_data.push_back(global_indices[v.index()]);
    }

    value_data.push_back(p.second);
  }

  const std::string mvc_dataset_name
      = "/MeshValueCollection/" + std::to_string(_counter);
  const std::int64_t num_values = MPI::sum(mesh->mpi_comm(), value_data.size());
  xdmf_write::add_data_item(_mpi_comm.comm(), topology_node, h5_id,
                            mvc_dataset_name + "/topology", topology_data,
                            {num_values, num_vertices_per_cell}, "Int");

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

  xdmf_write::add_data_item(_mpi_comm.comm(), attribute_node, h5_id,
                            mvc_dataset_name + "/values", value_data,
                            {num_values, 1}, "");

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");

  ++_counter;
}
//-----------------------------------------------------------------------------
mesh::MeshValueCollection<int>
XDMFFile::read_mvc_int(std::shared_ptr<const mesh::Mesh> mesh,
                       std::string name) const
{
  return read_mesh_value_collection<int>(mesh, name);
}
//-----------------------------------------------------------------------------
mesh::MeshValueCollection<std::size_t>
XDMFFile::read_mvc_size_t(std::shared_ptr<const mesh::Mesh> mesh,
                          std::string name) const
{
  return read_mesh_value_collection<std::size_t>(mesh, name);
}
//-----------------------------------------------------------------------------
mesh::MeshValueCollection<double>
XDMFFile::read_mvc_double(std::shared_ptr<const mesh::Mesh> mesh,
                          std::string name) const
{
  return read_mesh_value_collection<double>(mesh, name);
}
//-----------------------------------------------------------------------------
template <typename T>
mesh::MeshValueCollection<T>
XDMFFile::read_mesh_value_collection(std::shared_ptr<const mesh::Mesh> mesh,
                                     std::string name) const
{
  // Load XML doc from file
  pugi::xml_document xml_doc;
  pugi::xml_parse_result result = xml_doc.load_file(_filename.c_str());
  assert(result);

  // Get XDMF node
  pugi::xml_node xdmf_node = xml_doc.child("Xdmf");
  assert(xdmf_node);

  // Get domain node
  pugi::xml_node domain_node = xdmf_node.child("Domain");
  assert(domain_node);

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

  const int degree = cell_type_str.second;
  const mesh::CellType cell_type = mesh::to_type(cell_type_str.first);
  const int dim = mesh::cell_dim(cell_type);
  const int num_verts_per_entity = mesh::num_cell_vertices(cell_type);
  const int num_nodes_per_entity = mesh::num_cell_nodes(cell_type, degree);

  // Read MVC topology
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  assert(topology_data_node);
  boost::filesystem::path xdmf_filename(_filename);
  const boost::filesystem::path parent_path = xdmf_filename.parent_path();
  std::vector<std::int32_t> topology_data
      = xdmf_read::get_dataset<std::int32_t>(_mpi_comm.comm(),
                                             topology_data_node, parent_path);

  // Read values associated with each mesh::MeshEntity described by topology
  pugi::xml_node attribute_node = grid_node.child("Attribute");
  assert(attribute_node);
  pugi::xml_node attribute_data_node = attribute_node.child("DataItem");
  assert(attribute_data_node);
  std::vector<T> values_data = xdmf_read::get_dataset<T>(
      _mpi_comm.comm(), attribute_data_node, parent_path);

  // Ensure the mesh dimension is initialised
  mesh->create_entities(dim);
  const std::size_t global_vertex_range = mesh->num_entities_global(0);
  const std::int32_t num_processes = MPI::size(_mpi_comm.comm());

  // Send entities to processes based on the lowest vertex index
  std::vector<std::vector<std::int32_t>> send_entities(num_processes);
  std::vector<std::vector<std::int32_t>> recv_entities(num_processes);

  auto map = mesh->topology().index_map(0);
  assert(map);
  const std::vector<std::int64_t> global_indices = map->global_indices(false);

  std::vector<std::int32_t> v(num_verts_per_entity);
  for (auto& m : mesh::MeshRange(*mesh, dim, mesh::MeshRangeType::ALL))
  {
    if (dim == 0)
      v[0] = global_indices[m.index()];
    else
    {
      v.clear();
      for (auto& vtx : mesh::EntityRange(m, 0))
        v.push_back(global_indices[vtx.index()]);
      std::sort(v.begin(), v.end());
    }

    std::size_t dest
        = MPI::index_owner(num_processes, v[0], global_vertex_range);
    send_entities[dest].push_back(m.index());
    send_entities[dest].insert(send_entities[dest].end(), v.begin(), v.end());
  }
  MPI::all_to_all(_mpi_comm.comm(), send_entities, recv_entities);

  // Map from {entity vertex indices} to {process, local_index}
  std::map<std::vector<std::int32_t>, std::vector<std::int32_t>> entity_map;
  for (std::int32_t i = 0; i != num_processes; ++i)
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

  MPI::all_to_all(_mpi_comm.comm(), send_entities, recv_entities);
  MPI::all_to_all(_mpi_comm.comm(), send_data, recv_data);

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
  MPI::all_to_all(_mpi_comm.comm(), send_entities, recv_entities);
  MPI::all_to_all(_mpi_comm.comm(), send_data, recv_data);

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
//-----------------------------------------------------------------------------
void XDMFFile::write(
    const Eigen::Ref<
        const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& points)
{
  // Check that encoding
  if (_encoding == Encoding::ASCII and MPI::size(_mpi_comm.comm()) != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }

  // Open a HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
  std::unique_ptr<HDF5File> h5_file;
  if (_encoding == Encoding::HDF5)
  {
    // Open file
    h5_file = std::make_unique<HDF5File>(
        _mpi_comm.comm(), xdmf_utils::get_hdf5_filename(_filename), "w");
    assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }

  // Create pugi doc
  _xml_doc->reset();
  // Add XDMF node and version attribute
  _xml_doc->append_child(pugi::node_doctype)
      .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
  pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
  assert(xdmf_node);

  xdmf_write::add_points(_mpi_comm.comm(), xdmf_node, h5_id, points);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::write(
    const Eigen::Ref<
        const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& points,
    const std::vector<double>& values)
{
  // Write clouds of points to XDMF/HDF5 with values
  assert((std::size_t)points.rows() == values.size());

  // Check that encoding is supported
  if (_encoding == Encoding::ASCII and MPI::size(_mpi_comm.comm()) != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }

  // Create pugi doc
  _xml_doc->reset();

  // Open a HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
  std::unique_ptr<HDF5File> h5_file;
  if (_encoding == Encoding::HDF5)
  {
    // Open file
    h5_file = std::make_unique<HDF5File>(
        _mpi_comm.comm(), xdmf_utils::get_hdf5_filename(_filename), "w");
    assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }

  // Add XDMF node and version attribute
  _xml_doc->append_child(pugi::node_doctype)
      .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
  pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
  assert(xdmf_node);

  xdmf_write::add_points(_mpi_comm.comm(), xdmf_node, h5_id, points);

  // Add attribute node
  pugi::xml_node domain_node = xdmf_node.child("Domain");
  assert(domain_node);
  pugi::xml_node grid_node = domain_node.child("Grid");
  assert(grid_node);
  pugi::xml_node attribute_node = grid_node.append_child("Attribute");
  assert(attribute_node);
  attribute_node.append_attribute("Name") = "Point values";
  attribute_node.append_attribute("AttributeType") = "Scalar";
  attribute_node.append_attribute("Center") = "Node";

  // Add attribute DataItem node and write data
  std::int64_t num_values = MPI::sum(_mpi_comm.comm(), values.size());
  xdmf_write::add_data_item(_mpi_comm.comm(), attribute_node, h5_id,
                            "/Points/values", values, {num_values, 1}, "");

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
std::tuple<
    mesh::CellType,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    std::vector<std::int64_t>>
XDMFFile::read_mesh_data(MPI_Comm comm) const
{
  // Extract parent filepath (required by HDF5 when XDMF stores relative
  // path of the HDF5 files(s) and the XDMF is not opened from its own
  // directory)
  boost::filesystem::path xdmf_filename(_filename);
  const boost::filesystem::path parent_path = xdmf_filename.parent_path();

  if (!boost::filesystem::exists(xdmf_filename))
    throw std::runtime_error("Cannot open XDMF file. File does not exists.");

  // Load XML doc from file
  pugi::xml_document xml_doc;
  pugi::xml_parse_result result = xml_doc.load_file(_filename.c_str());
  assert(result);

  // Get XDMF node
  pugi::xml_node xdmf_node = xml_doc.child("Xdmf");
  assert(xdmf_node);

  // Get domain node
  pugi::xml_node domain_node = xdmf_node.child("Domain");
  assert(domain_node);

  // Get grid node
  pugi::xml_node grid_node = domain_node.child("Grid");
  assert(grid_node);

  // Get topology node
  pugi::xml_node topology_node = grid_node.child("Topology");
  assert(topology_node);

  // Get cell type
  const auto cell_type_str = xdmf_utils::get_cell_type(topology_node);

  const int degree = cell_type_str.second;
  if (degree == 2)
    LOG(WARNING) << "Caution: reading quadratic mesh";

  // Get toplogical dimensions
  mesh::CellType cell_type = mesh::to_type(cell_type_str.first);

  // Get geometry node
  pugi::xml_node geometry_node = grid_node.child("Geometry");
  assert(geometry_node);

  // Determine geometric dimension
  pugi::xml_attribute geometry_type_attr
      = geometry_node.attribute("GeometryType");
  assert(geometry_type_attr);
  int gdim = -1;
  const std::string geometry_type = geometry_type_attr.value();
  if (geometry_type == "XY")
    gdim = 2;
  else if (geometry_type == "XYZ")
    gdim = 3;
  else
  {
    throw std::runtime_error("Cannot determine geometric dimension. "
                             "GeometryType \""
                             + geometry_type
                             + "\" in XDMF file is unknown or unsupported");
  }

  // Get number of points from Geometry dataitem node
  pugi::xml_node geometry_data_node = geometry_node.child("DataItem");
  assert(geometry_data_node);
  const std::vector<std::int64_t> gdims
      = xdmf_utils::get_dataset_shape(geometry_data_node);
  assert(gdims.size() == 2);
  assert(gdims[1] == gdim);

  // Get topology dataset node
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  assert(topology_data_node);
  const std::vector<std::int64_t> tdims
      = xdmf_utils::get_dataset_shape(topology_data_node);
  const std::size_t npoint_per_cell = tdims[1];

  // If this process is not in the (new) communicator, it will be MPI_COMM_NULL.
  if (comm != MPI_COMM_NULL)
  {
    // Geometry data
    const auto geometry_data
        = xdmf_read::get_dataset<double>(comm, geometry_data_node, parent_path);
    const std::size_t num_local_points = geometry_data.size() / gdim;
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        points(geometry_data.data(), num_local_points, gdim);

    // Topology data
    const std::vector<std::int64_t> tdims
        = xdmf_utils::get_dataset_shape(topology_data_node);
    const std::vector<std::int64_t> topology_data
        = xdmf_read::get_dataset<std::int64_t>(comm, topology_data_node,
                                               parent_path);
    const std::size_t num_local_cells = topology_data.size() / npoint_per_cell;
    Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        cells(topology_data.data(), num_local_cells, npoint_per_cell);

    // Set cell global indices by adding offset
    const std::int64_t cell_index_offset
        = MPI::global_offset(_mpi_comm.comm(), num_local_cells, true);
    std::vector<std::int64_t> global_cell_indices(num_local_cells);
    std::iota(global_cell_indices.begin(), global_cell_indices.end(),
              cell_index_offset);

    return std::tuple(cell_type, std::move(points), std::move(cells),
                      std::move(global_cell_indices));
  }
  else
  {
    // Create empty mesh data for processes that don't belong to the
    // communicator.
    const std::size_t num_local_cells = 0;
    MPI::global_offset(_mpi_comm.comm(), num_local_cells, true);
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        points(num_local_cells, gdim);
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        cells(num_local_cells, npoint_per_cell);
    std::vector<std::int64_t> global_cell_indices(num_local_cells);

    return std::tuple(cell_type, std::move(points), std::move(cells),
                      std::move(global_cell_indices));
  }
}
//----------------------------------------------------------------------------
function::Function
XDMFFile::read_checkpoint(std::shared_ptr<const function::FunctionSpace> V,
                          std::string func_name, std::int64_t counter) const
{
  LOG(INFO) << "Reading function \"" << func_name << "\" from XDMF file \""
            << _filename << "\" with counter " << counter;

  // Extract parent filepath (required by HDF5 when XDMF stores relative path
  // of the HDF5 files(s) and the XDMF is not opened from its own directory)
  boost::filesystem::path xdmf_filename(_filename);
  const boost::filesystem::path parent_path = xdmf_filename.parent_path();

  if (!boost::filesystem::exists(xdmf_filename))
  {
    throw std::runtime_error("Cannot open XDMF file. "
                             "XDMF file \""
                             + _filename + "\" does not exist");
  }

  // Read XML nodes = parse XML document

  // Load XML doc from file
  pugi::xml_document xml_doc;
  pugi::xml_parse_result result = xml_doc.load_file(_filename.c_str());
  assert(result);

  // Find grid with name equal to the name of function we're about
  // to save and given counter

  // If counter is negative then read with respect to last element, i.e.
  // counter = -1 == last element, counter = -2 == one before last etc.
  std::string selector;
  if (counter < -1)
    selector = "position()=last()" + std::to_string(counter + 1);
  else if (counter == -1)
    selector = "position()=last()";
  else
    selector = "@Name='" + func_name + "_" + std::to_string(counter) + "'";

  pugi::xml_node grid_node
      = xml_doc
            .select_node(("/Xdmf/Domain/Grid[@CollectionType='Temporal' and "
                          "@Name='"
                          + func_name + "']/Grid[" + selector + "]")
                             .c_str())
            .node();
  assert(grid_node);

#ifdef PETSC_USE_COMPLEX
  // Find FE attribute node of the real component (default)
  pugi::xml_node fe_attribute_node
      = grid_node
            .select_node(("Attribute[@ItemType=\"FiniteElementFunction\" and"
                          "@Name='real_"
                          + func_name + "']")
                             .c_str())
            .node();
#else
  pugi::xml_node fe_attribute_node
      = grid_node.select_node("Attribute[@ItemType=\"FiniteElementFunction\"]")
            .node();
#endif

  assert(fe_attribute_node);
  // Get cells dofs indices = dofmap
  pugi::xml_node cell_dofs_dataitem
      = fe_attribute_node.select_node("DataItem[position()=1]").node();
  assert(cell_dofs_dataitem);

  // Get vector
  pugi::xml_node vector_dataitem
      = fe_attribute_node.select_node("DataItem[position()=2]").node();
  assert(vector_dataitem);

  // Get number of dofs per cell
  pugi::xml_node x_cell_dofs_dataitem
      = fe_attribute_node.select_node("DataItem[position()=3]").node();
  assert(x_cell_dofs_dataitem);

  // Get cell ordering
  pugi::xml_node cells_dataitem
      = fe_attribute_node.select_node("DataItem[position()=4]").node();
  assert(cells_dataitem);

  // Read dataitems

  // Get existing mesh and dofmap - these should be pre-existing
  // and set up by user when defining the function::Function
  assert(V);
  assert(V->mesh());
  const mesh::Mesh& mesh = *V->mesh();
  assert(V->dofmap());
  const fem::DofMap& dofmap = *V->dofmap();

  // Read cell ordering
  std::vector<std::size_t> cells = xdmf_read::get_dataset<std::size_t>(
      _mpi_comm.comm(), cells_dataitem, parent_path);

  const std::vector<std::int64_t> x_cell_dofs_shape
      = xdmf_utils::get_dataset_shape(cells_dataitem);

  // Divide cells equally between processes
  std::array<std::int64_t, 2> cell_range
      = dolfinx::MPI::local_range(_mpi_comm.comm(), x_cell_dofs_shape[0]);

  // Read number of dofs per cell
  std::vector<std::int64_t> x_cell_dofs = xdmf_read::get_dataset<std::int64_t>(
      _mpi_comm.comm(), x_cell_dofs_dataitem, parent_path,
      {{cell_range[0], cell_range[1] + 1}});

  // Read cell dofmaps
  std::vector<std::int64_t> cell_dofs = xdmf_read::get_dataset<std::int64_t>(
      _mpi_comm.comm(), cell_dofs_dataitem, parent_path,
      {{x_cell_dofs.front(), x_cell_dofs.back()}});

  const std::vector<std::int64_t> vector_shape
      = xdmf_utils::get_dataset_shape(vector_dataitem);
  const std::size_t num_global_dofs = vector_shape[0];

  // Divide vector between processes
  const std::array<std::int64_t, 2> input_vector_range
      = MPI::local_range(_mpi_comm.comm(), num_global_dofs);

#ifdef PETSC_USE_COMPLEX
  // Read real component of function vector
  std::vector<double> real_vector = xdmf_read::get_dataset<double>(
      _mpi_comm.comm(), vector_dataitem, parent_path, input_vector_range);

  // Find FE attribute node of the imaginary component
  pugi::xml_node imag_fe_attribute_node
      = grid_node
            .select_node(("Attribute[@ItemType=\"FiniteElementFunction\" and"
                          "@Name='imag_"
                          + func_name + "']")
                             .c_str())
            .node();
  assert(imag_fe_attribute_node);

  // Get extra FE attribute of the imaginary component
  pugi::xml_node imag_vector_dataitem
      = imag_fe_attribute_node.select_node("DataItem[position()=2]").node();
  assert(imag_vector_dataitem);

  // Read imaginary component of function vector
  std::vector<double> imag_vector = xdmf_read::get_dataset<double>(
      _mpi_comm.comm(), imag_vector_dataitem, parent_path, input_vector_range);

  assert(real_vector.size() == imag_vector.size());

  // Compose complex function vector
  std::vector<PetscScalar> vector;
  vector.reserve(real_vector.size());
  std::transform(std::begin(real_vector), std::end(real_vector),
                 std::begin(imag_vector), std::back_inserter(vector),
                 [](double r, double i) { return r + i * PETSC_i; });
#else
  // Read function vector
  std::vector<double> vector = xdmf_read::get_dataset<double>(
      _mpi_comm.comm(), vector_dataitem, parent_path, input_vector_range);
#endif

  function::Function u(V);
  HDF5Utility::set_local_vector_values(_mpi_comm.comm(), u.vector(), mesh,
                                       cells, cell_dofs, x_cell_dofs, vector,
                                       input_vector_range, dofmap);

  return u;
}
//----------------------------------------------------------------------------
