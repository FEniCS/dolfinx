// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_write.h"
#include "HDF5File.h"
#include "pugixml.hpp"
#include "xdmf_utils.h"
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/log.h>
#include <dolfin/common/utils.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Topology.h>
#include <dolfin/mesh/Vertex.h>

using namespace dolfin;
using namespace dolfin::io;

namespace
{
//-----------------------------------------------------------------------------
// Convert a value_rank to the XDMF string description (Scalar, Vector,
// Tensor)
std::string rank_to_string(std::size_t value_rank)
{
  switch (value_rank)
  {
  case 0:
    return "Scalar";
  case 1:
    return "Vector";
  case 2:
    return "Tensor";
  default:
    throw std::runtime_error("Range Error");
  }

  return "";
}
//-----------------------------------------------------------------------------
// Remap meshfunction data, scattering data to appropriate processes
template <typename T>
void remap_meshfunction_data(mesh::MeshFunction<T>& meshfunction,
                             const std::vector<std::int64_t>& topology_data,
                             const std::vector<T>& value_data)
{
  // Send the read data to each process on the basis of the first
  // vertex of the entity, since we do not know the global_index
  const int cell_dim = meshfunction.dim();
  const auto mesh = meshfunction.mesh();
  // FIXME : get vertices_per_entity properly
  const int vertices_per_entity = meshfunction.dim() + 1;
  const MPI_Comm comm = mesh->mpi_comm();
  const std::size_t num_processes = dolfin::MPI::size(comm);

  // Wrap topology data in boost array
  assert(topology_data.size() % vertices_per_entity == 0);
  const std::size_t num_entities = topology_data.size() / vertices_per_entity;

  // Send (sorted) entity topology and data to a post-office process
  // determined by the lowest global vertex index of the entity
  std::vector<std::vector<std::int64_t>> send_topology(num_processes);
  std::vector<std::vector<T>> send_values(num_processes);
  const std::size_t max_vertex = mesh->num_entities_global(0);
  for (std::size_t i = 0; i < num_entities; ++i)
  {
    std::vector<std::int64_t> cell_topology(
        topology_data.begin() + i * vertices_per_entity,
        topology_data.begin() + (i + 1) * vertices_per_entity);
    std::sort(cell_topology.begin(), cell_topology.end());

    // Use first vertex to decide where to send this data
    const std::size_t destination_process
        = dolfin::MPI::index_owner(comm, cell_topology.front(), max_vertex);

    send_topology[destination_process].insert(
        send_topology[destination_process].end(), cell_topology.begin(),
        cell_topology.end());
    send_values[destination_process].push_back(value_data[i]);
  }

  std::vector<std::vector<std::int64_t>> receive_topology(num_processes);
  std::vector<std::vector<T>> receive_values(num_processes);
  dolfin::MPI::all_to_all(comm, send_topology, receive_topology);
  dolfin::MPI::all_to_all(comm, send_values, receive_values);

  // Generate requests for data from remote processes, based on the
  // first vertex of the mesh::MeshEntities which belong on this process
  // Send our process number, and our local index, so it can come back
  // directly to the right place
  std::vector<std::vector<std::int64_t>> send_requests(num_processes);
  const std::size_t rank = dolfin::MPI::rank(comm);
  for (auto& cell : mesh::MeshRange<mesh::MeshEntity>(*mesh, cell_dim,
                                                      mesh::MeshRangeType::ALL))
  {
    std::vector<std::int64_t> cell_topology;
    if (cell_dim == 0)
      cell_topology.push_back(cell.global_index());
    else
    {
      for (auto& v : mesh::EntityRange<mesh::Vertex>(cell))
        cell_topology.push_back(v.global_index());
    }

    std::sort(cell_topology.begin(), cell_topology.end());

    // Use first vertex to decide where to send this request
    std::size_t send_to_process
        = dolfin::MPI::index_owner(comm, cell_topology.front(), max_vertex);
    // Map to this process and local index by appending to send data
    cell_topology.push_back(cell.index());
    cell_topology.push_back(rank);
    send_requests[send_to_process].insert(send_requests[send_to_process].end(),
                                          cell_topology.begin(),
                                          cell_topology.end());
  }

  std::vector<std::vector<std::int64_t>> receive_requests(num_processes);
  dolfin::MPI::all_to_all(comm, send_requests, receive_requests);

  // At this point, the data with its associated vertices is in
  // receive_values and receive_topology and the final destinations
  // are stored in receive_requests as
  // [vertices][index][process][vertices][index][process]...  Some
  // data will have more than one destination

  // Create a mapping from the topology vector to the desired data
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
  send_topology = std::vector<std::vector<std::int64_t>>(num_processes);
  send_values = std::vector<std::vector<T>>(num_processes);

  // Go through requests, which are stacked as [vertex, vertex, ...]
  // [index] [proc] etc.  Use the vertices as the key for the map
  // (above) to retrieve the data to send to proc
  for (std::size_t i = 0; i < receive_requests.size(); ++i)
  {
    for (auto p = receive_requests[i].begin(); p != receive_requests[i].end();
         p += (vertices_per_entity + 2))
    {
      const std::vector<std::int64_t> cell(p, p + vertices_per_entity);
      const std::size_t remote_index = *(p + vertices_per_entity);
      const std::size_t send_to_proc = *(p + vertices_per_entity + 1);

      const auto find_cell = cell_to_data.find(cell);
      assert(find_cell != cell_to_data.end());
      send_values[send_to_proc].push_back(find_cell->second);
      send_topology[send_to_proc].push_back(remote_index);
    }
  }

  dolfin::MPI::all_to_all(comm, send_topology, receive_topology);
  dolfin::MPI::all_to_all(comm, send_values, receive_values);

  // At this point, receive_topology should only list the local indices
  // and received values should have the appropriate values for each
  for (std::size_t i = 0; i < receive_values.size(); ++i)
  {
    assert(receive_values[i].size() == receive_topology[i].size());
    for (std::size_t j = 0; j < receive_values[i].size(); ++j)
      meshfunction[receive_topology[i][j]] = receive_values[i][j];
  }
}
//----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Return topology data on this process as a flat vector
std::vector<std::int64_t> compute_topology_data(const mesh::Mesh& mesh,
                                                int cell_dim)
{
  // Create vector to store topology data
  const int num_vertices_per_cell = mesh.type().num_vertices(cell_dim);
  std::vector<std::int64_t> topology_data;
  topology_data.reserve(mesh.num_entities(cell_dim) * (num_vertices_per_cell));

  // Get mesh communicator
  MPI_Comm comm = mesh.mpi_comm();

  const std::vector<std::int8_t> perm = mesh.type().vtk_mapping();
  const int tdim = mesh.topology().dim();
  if (dolfin::MPI::size(comm) == 1 or cell_dim == tdim)
  {
    // Simple case when nothing is shared between processes
    if (cell_dim == 0)
    {
      for (auto& v : mesh::MeshRange<mesh::Vertex>(mesh))
        topology_data.push_back(v.global_index());
    }
    else
    {
      const auto& global_vertices = mesh.topology().global_indices(0);
      for (auto& c : mesh::MeshRange<mesh::MeshEntity>(mesh, cell_dim))
      {
        const std::int32_t* entities = c.entities(0);
        for (std::uint32_t i = 0; i != c.num_entities(0); ++i)
          topology_data.push_back(global_vertices[entities[perm[i]]]);
      }
    }
  }
  else
  {
    std::set<std::uint32_t> non_local_entities
        = xdmf_write::compute_nonlocal_entities(mesh, cell_dim);

    if (cell_dim == 0)
    {
      // Special case for mesh of points
      for (auto& v : mesh::MeshRange<mesh::Vertex>(mesh))
      {
        if (non_local_entities.find(v.index()) == non_local_entities.end())
          topology_data.push_back(v.global_index());
      }
    }
    else
    {
      // Local-to-global map for point indices
      const auto& global_vertices = mesh.topology().global_indices(0);
      for (auto& e : mesh::MeshRange<mesh::MeshEntity>(mesh, cell_dim))
      {
        // If not excluded, add to topology
        if (non_local_entities.find(e.index()) == non_local_entities.end())
        {
          for (std::uint32_t i = 0; i != e.num_entities(0); ++i)
          {
            const std::int32_t local_idx = e.entities(0)[perm[i]];
            topology_data.push_back(global_vertices[local_idx]);
          }
        }
      }
    }
  }

  return topology_data;
}
//-----------------------------------------------------------------------------
// Return data associated with a data set node
template <typename T>
std::vector<T> get_dataset(MPI_Comm comm, const pugi::xml_node& dataset_node,
                           const boost::filesystem::path& parent_path,
                           std::array<std::int64_t, 2> range = {{0, 0}})
{
  // FIXME: Need to sort out datasset dimensions - can't depend on
  // HDF5 shape, and a Topology data item is not required to have a
  // 'Dimensions' attribute since the dimensions can be determined
  // from the number of cells and the cell type (for topology, one
  // must supply cell type + (number of cells or dimensions).
  //
  // A geometry data item must have 'Dimensions' attribute.

  assert(dataset_node);
  pugi::xml_attribute format_attr = dataset_node.attribute("Format");
  assert(format_attr);

  // Get data set shape from 'Dimensions' attribute (empty if not available)
  const std::vector<std::int64_t> shape_xml
      = xdmf_utils::get_dataset_shape(dataset_node);

  const std::string format = format_attr.as_string();
  std::vector<T> data_vector;
  // Only read ASCII on process 0
  if (format == "XML")
  {
    if (dolfin::MPI::rank(comm) == 0)
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

    // Handle cases where file path is (a) absolute or (b) relative
    boost::filesystem::path h5_filepath(paths[0]);
    if (!h5_filepath.is_absolute())
      h5_filepath = parent_path / h5_filepath;

    // Open HDF5 for reading
    HDF5File h5_file(comm, h5_filepath.string(), "r");

    // Get data shape from HDF5 file
    const std::vector<std::int64_t> shape_hdf5
        = HDF5Interface::get_dataset_shape(h5_file.h5_id(), paths[1]);

    // FIXME: should we support empty data sets?
    // Check that data set is not empty
    assert(!shape_hdf5.empty());
    assert(shape_hdf5[0] != 0);

    // Determine range of data to read from HDF5 file. This is
    // complicated by the XML Dimension attribute and the HDF5 storage
    // possibly having different shapes, e.g. the HDF5 storgae may be a
    // flat array.

    // If range = {0, 0} then no range is supplied
    // and we must determine the range
    if (range[0] == 0 and range[1] == 0)
    {
      if (shape_xml == shape_hdf5)
        range = dolfin::MPI::local_range(comm, shape_hdf5[0]);
      else if (!shape_xml.empty() and shape_hdf5.size() == 1)
      {
        // Size of dims > 0
        std::int64_t d = 1;
        for (std::size_t i = 1; i < shape_xml.size(); ++i)
          d *= shape_xml[i];

        // Check for data size consistency
        if (d * shape_xml[0] != shape_hdf5[0])
        {
          throw std::runtime_error("Data size in XDMF/XML and size of HDF5 "
                                   "dataset are inconsistent");
        }

        // Compute data range to read
        range = dolfin::MPI::local_range(comm, shape_xml[0]);
        range[0] *= d;
        range[1] *= d;
      }
      else
      {
        throw std::runtime_error(
            "This combination of array shapes in XDMF and HDF5 "
            "is not supported");
      }
    }

    // Retrieve data
    data_vector
        = HDF5Interface::read_dataset<T>(h5_file.h5_id(), paths[1], range);
  }
  else
    throw std::runtime_error("Storage format \"" + format + "\" is unknown");

  // Get dimensions for consistency (if available in DataItem node)
  if (shape_xml.empty())
  {
    std::int64_t size = 1;
    for (auto dim : shape_xml)
      size *= dim;

    if (size != (std::int64_t)dolfin::MPI::sum(comm, data_vector.size()))
    {
      throw std::runtime_error(
          "Data sizes in attribute and size of data read are inconsistent");
    }
  }

  return data_vector;
}
//----------------------------------------------------------------------------
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

std::set<std::uint32_t>
xdmf_write::compute_nonlocal_entities(const mesh::Mesh& mesh, int cell_dim)
{
  // If not already numbered, number entities of
  // order cell_dim so we can get shared_entities
  mesh::DistributedMeshTools::number_entities(mesh, cell_dim);

  const int mpi_rank = dolfin::MPI::rank(mesh.mpi_comm());
  const std::map<std::int32_t, std::set<std::int32_t>>& shared_entities
      = mesh.topology().shared_entities(cell_dim);

  std::set<std::uint32_t> non_local_entities;

  const int tdim = mesh.topology().dim();
  bool ghosted
      = (mesh.topology().size(tdim) > mesh.topology().ghost_offset(tdim));

  if (!ghosted)
  {
    // No ghost cells - exclude shared entities which are on lower rank
    // processes
    for (const auto& e : shared_entities)
    {
      const int lowest_rank_owner = *(e.second.begin());
      if (lowest_rank_owner < mpi_rank)
        non_local_entities.insert(e.first);
    }
  }
  else
  {
    // Iterate through ghost cells, adding non-ghost entities which are
    // in lower rank process cells
    for (auto& c : mesh::MeshRange<mesh::MeshEntity>(
             mesh, tdim, mesh::MeshRangeType::GHOST))
    {
      const int cell_owner = c.owner();
      for (auto& e : mesh::EntityRange<mesh::MeshEntity>(c, cell_dim))
        if (!e.is_ghost() && cell_owner < mpi_rank)
          non_local_entities.insert(e.index());
    }
  }
  return non_local_entities;
}
//-----------------------------------------------------------------------------
void xdmf_write::add_points(MPI_Comm comm, pugi::xml_node& xdmf_node,
                            hid_t h5_id,
                            const std::vector<Eigen::Vector3d>& points)
{
  xdmf_node.append_attribute("Version") = "3.0";
  xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";
  pugi::xml_node domain_node = xdmf_node.append_child("Domain");
  assert(domain_node);

  // Add a Grid to the domain
  pugi::xml_node grid_node = domain_node.append_child("Grid");
  assert(grid_node);
  grid_node.append_attribute("GridType") = "Uniform";
  grid_node.append_attribute("Name") = "Point cloud";

  pugi::xml_node topology_node = grid_node.append_child("Topology");
  assert(topology_node);
  const std::size_t n = points.size();
  const std::int64_t nglobal = dolfin::MPI::sum(comm, n);
  topology_node.append_attribute("NumberOfElements")
      = std::to_string(nglobal).c_str();
  topology_node.append_attribute("TopologyType") = "PolyVertex";
  topology_node.append_attribute("NodesPerElement") = 1;

  pugi::xml_node geometry_node = grid_node.append_child("Geometry");
  assert(geometry_node);
  geometry_node.append_attribute("GeometryType") = "XYZ";

  // Pack data
  std::vector<double> x(3 * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      x[3 * i + j] = points[i][j];

  const std::vector<std::int64_t> shape = {nglobal, 3};
  add_data_item(comm, geometry_node, h5_id, "/Points/coordinates", x, shape,
                "");
}
//----------------------------------------------------------------------------
void xdmf_write::add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                   hid_t h5_id, const std::string path_prefix,
                                   const mesh::Mesh& mesh, int cell_dim)
{
  // Get number of cells (global) and vertices per cell from mesh
  const std::int64_t num_cells = mesh.topology().size_global(cell_dim);
  int num_nodes_per_cell = mesh.type().num_vertices(cell_dim);

  // Get VTK string for cell type and degree (linear or quadratic)
  const std::size_t degree = mesh.degree();
  const std::string vtk_cell_str = xdmf_utils::vtk_cell_type_str(
      mesh.type().entity_type(cell_dim), degree);

  pugi::xml_node topology_node = xml_node.append_child("Topology");
  assert(topology_node);
  topology_node.append_attribute("NumberOfElements")
      = std::to_string(num_cells).c_str();
  topology_node.append_attribute("TopologyType") = vtk_cell_str.c_str();

  // Compute packed topology data
  std::vector<std::int64_t> topology_data;

  if (degree > 1)
  {
    const int tdim = mesh.topology().dim();
    if (cell_dim != tdim)
    {
      throw std::runtime_error("Cannot create topology data for mesh. "
                               "Can only create mesh of cells");
    }

    const auto& global_points = mesh.geometry().global_indices();
    const mesh::Connectivity& cell_points
        = mesh.coordinate_dofs().entity_points();

    // Adjust num_nodes_per_cell to appropriate size
    num_nodes_per_cell = cell_points.size(0);
    topology_data.reserve(num_nodes_per_cell * mesh.num_entities(tdim));
    const std::vector<std::uint8_t>& perm
        = mesh.coordinate_dofs().cell_permutation();

    for (std::int32_t c = 0; c < mesh.num_entities(tdim); ++c)
    {
      const std::int32_t* points = cell_points.connections(c);
      for (std::int32_t i = 0; i < num_nodes_per_cell; ++i)
        topology_data.push_back(global_points[points[perm[i]]]);
    }
  }
  else
    topology_data = compute_topology_data(mesh, cell_dim);

  topology_node.append_attribute("NodesPerElement") = num_nodes_per_cell;

  // Add topology DataItem node
  const std::string group_name = path_prefix + "/" + "mesh";
  const std::string h5_path = group_name + "/topology";
  const std::vector<std::int64_t> shape = {num_cells, num_nodes_per_cell};
  const std::string number_type = "UInt";

  xdmf_write::add_data_item(comm, topology_node, h5_id, h5_path, topology_data,
                            shape, number_type);
}
//-----------------------------------------------------------------------------
void xdmf_write::add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                   hid_t h5_id, const std::string path_prefix,
                                   const mesh::Mesh& mesh)
{
  const mesh::Geometry& mesh_geometry = mesh.geometry();
  int gdim = mesh_geometry.dim();

  // Compute number of points (global) in mesh (equal to number of vertices
  // for affine meshes)
  const std::int64_t num_points = mesh.geometry().num_points_global();

  // Add geometry node and attributes
  pugi::xml_node geometry_node = xml_node.append_child("Geometry");
  assert(geometry_node);
  assert(gdim > 0 and gdim <= 3);
  const std::string geometry_type = (gdim == 3) ? "XYZ" : "XY";
  geometry_node.append_attribute("GeometryType") = geometry_type.c_str();

  // Pack geometry data
  EigenRowArrayXXd _x = mesh::DistributedMeshTools::reorder_by_global_indices(
      mesh.mpi_comm(), mesh.geometry().points(),
      mesh.geometry().global_indices());

  // Increase 1D to 2D because XDMF has no "X" geometry, use "XY"
  int width = (gdim == 1) ? 2 : gdim;

  std::size_t num_values = _x.rows() * width;
  std::vector<double> x(num_values, 0.0);

  if (width == 3)
    std::copy(_x.data(), _x.data() + _x.size(), x.begin());
  else
  {
    for (int i = 0; i < _x.rows(); ++i)
    {
      for (int j = 0; j < gdim; ++j)
        x[width * i + j] = _x(i, j);
    }
  }

  // Add geometry DataItem node
  const std::string group_name = path_prefix + "/" + "mesh";
  const std::string h5_path = group_name + "/geometry";
  const std::vector<std::int64_t> shape = {num_points, width};

  xdmf_write::add_data_item(comm, geometry_node, h5_id, h5_path, x, shape, "");
}
//-----------------------------------------------------------------------------
void xdmf_write::add_mesh(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                          const mesh::Mesh& mesh, const std::string path_prefix)
{
  LOG(INFO) << "Adding mesh to node \"" << xml_node.path('/') << "\"";

  // Add grid node and attributes
  pugi::xml_node grid_node = xml_node.append_child("Grid");
  assert(grid_node);
  grid_node.append_attribute("Name") = "mesh";
  grid_node.append_attribute("GridType") = "Uniform";

  // Add topology node and attributes (including writing data)
  const int tdim = mesh.topology().dim();
  add_topology_data(comm, grid_node, h5_id, path_prefix, mesh, tdim);

  // Add geometry node and attributes (including writing data)
  add_geometry_data(comm, grid_node, h5_id, path_prefix, mesh);
}
//----------------------------------------------------------------------------
void xdmf_write::add_function(MPI_Comm mpi_comm, pugi::xml_node& xml_node,
                              hid_t h5_id, std::string h5_path,
                              const function::Function& u,
                              std::string function_name, const mesh::Mesh& mesh,
                              const std::string component)
{
  LOG(INFO) << "Adding function to node \"" << xml_node.path('/') << "\"";

  std::string element_family = u.function_space()->element->family();
  const std::size_t element_degree = u.function_space()->element->degree();
  const CellType element_cell_type = u.function_space()->element->cell_shape();

  // Map of standard UFL family abbreviations for visualisation
  const std::map<std::string, std::string> family_abbr
      = {{"Lagrange", "CG"},
         {"Discontinuous Lagrange", "DG"},
         {"Raviart-Thomas", "RT"},
         {"Brezzi-Douglas-Marini", "BDM"},
         {"Crouzeix-Raviart", "CR"},
         {"Nedelec 1st kind H(curl)", "N1curl"},
         {"Nedelec 2nd kind H(curl)", "N2curl"},
         {"Q", "Q"},
         {"DQ", "DQ"}};

  const std::map<CellType, std::string> cell_shape_repr
      = {{CellType::interval, "interval"},
         {CellType::triangle, "triangle"},
         {CellType::tetrahedron, "tetrahedron"},
         {CellType::quadrilateral, "quadrilateral"},
         {CellType::hexahedron, "hexahedron"}};

  // Check that element is supported
  auto const it = family_abbr.find(element_family);
  if (it == family_abbr.end())
    throw std::runtime_error("Element type not supported for XDMF output.");
  element_family = it->second;

  // Check that cell shape is supported
  auto it_shape = cell_shape_repr.find(element_cell_type);
  if (it_shape == cell_shape_repr.end())
    throw std::runtime_error("Cell type not supported for XDMF output.");
  const std::string element_cell = it_shape->second;

  // Prepare main Attribute for the FiniteElementFunction type
  std::string attr_name;
  if (component.empty())
    attr_name = function_name;
  else
  {
    attr_name = component + "_" + function_name;
    h5_path = h5_path + "/" + component;
  }

  pugi::xml_node fe_attribute_node = xml_node.append_child("Attribute");
  fe_attribute_node.append_attribute("ItemType") = "FiniteElementFunction";
  fe_attribute_node.append_attribute("ElementFamily") = element_family.c_str();
  fe_attribute_node.append_attribute("ElementDegree")
      = std::to_string(element_degree).c_str();
  fe_attribute_node.append_attribute("ElementCell") = element_cell.c_str();
  fe_attribute_node.append_attribute("Name") = attr_name.c_str();
  fe_attribute_node.append_attribute("Center") = "Other";
  fe_attribute_node.append_attribute("AttributeType")
      = rank_to_string(u.value_rank()).c_str();

  // Prepare and save number of dofs per cell (x_cell_dofs) and cell
  // dofmaps (cell_dofs)

  assert(u.function_space()->dofmap);
  const fem::DofMap& dofmap = *u.function_space()->dofmap;

  const std::size_t tdim = mesh.topology().dim();
  std::vector<PetscInt> cell_dofs;
  std::vector<std::size_t> x_cell_dofs;
  const std::size_t n_cells = mesh.topology().ghost_offset(tdim);
  x_cell_dofs.reserve(n_cells);

  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> local_to_global_map
      = dofmap.index_map->indices(true);

  // Add number of dofs for each cell
  // Add cell dofmap
  for (std::size_t i = 0; i != n_cells; ++i)
  {
    x_cell_dofs.push_back(cell_dofs.size());
    auto cell_dofs_i = dofmap.cell_dofs(i);
    for (Eigen::Index j = 0; j < cell_dofs_i.size(); ++j)
    {
      auto p = cell_dofs_i[j];
      assert(p < (PetscInt)local_to_global_map.size());
      cell_dofs.push_back(local_to_global_map[p]);
    }
  }

  // Add offset to CSR index to be seamless in parallel
  std::size_t offset = MPI::global_offset(mpi_comm, cell_dofs.size(), true);
  std::transform(x_cell_dofs.begin(), x_cell_dofs.end(), x_cell_dofs.begin(),
                 std::bind2nd(std::plus<std::size_t>(), offset));

  const std::int64_t num_cell_dofs_global
      = MPI::sum(mpi_comm, cell_dofs.size());

  // Write dofmap = indices to the values DataItem
  xdmf_write::add_data_item(mpi_comm, fe_attribute_node, h5_id,
                            h5_path + "/cell_dofs", cell_dofs,
                            {num_cell_dofs_global, 1}, "UInt");

  // FIXME: Avoid unnecessary copying of data
  // Get all local data
  const la::PETScVector& u_vector = u.vector();
  PetscErrorCode ierr;
  const PetscScalar* u_ptr = nullptr;
  ierr = VecGetArrayRead(u_vector.vec(), &u_ptr);
  if (ierr != 0)
    la::petsc_error(ierr, __FILE__, "VecGetArrayRead");
  std::vector<PetscScalar> local_data(u_ptr, u_ptr + u_vector.local_size());
  ierr = VecRestoreArrayRead(u_vector.vec(), &u_ptr);
  if (ierr != 0)
    la::petsc_error(ierr, __FILE__, "VecRestoreArrayRead");

#ifdef PETSC_USE_COMPLEX
  // FIXME: Avoid copies by writing directly a compound data
  std::vector<double> component_data_values(local_data.size());
  for (unsigned int i = 0; i < local_data.size(); i++)
  {
    if (component == "real")
      component_data_values[i] = local_data[i].real();
    else if (component == "imag")
      component_data_values[i] = local_data[i].imag();
  }

  xdmf_write::add_data_item(mpi_comm, fe_attribute_node, h5_id,
                            h5_path + "/vector", component_data_values,
                            {(std::int64_t)u_vector.size(), 1}, "Float");
#else
  xdmf_write::add_data_item(mpi_comm, fe_attribute_node, h5_id,
                            h5_path + "/vector", local_data,
                            {(std::int64_t)u_vector.size(), 1}, "Float");
#endif

  if (MPI::rank(mpi_comm) == MPI::size(mpi_comm) - 1)
    x_cell_dofs.push_back(num_cell_dofs_global);

  const std::int64_t num_x_cell_dofs_global
      = mesh.num_entities_global(tdim) + 1;

  // Write number of dofs per cell
  xdmf_write::add_data_item(mpi_comm, fe_attribute_node, h5_id,
                            h5_path + "/x_cell_dofs", x_cell_dofs,
                            {num_x_cell_dofs_global, 1}, "UInt");

  // Save cell ordering - copy to local vector and cut off ghosts
  std::vector<std::size_t> cells(mesh.topology().global_indices(tdim).begin(),
                                 mesh.topology().global_indices(tdim).begin()
                                     + n_cells);

  const std::int64_t num_cells_global = mesh.num_entities_global(tdim);

  xdmf_write::add_data_item(mpi_comm, fe_attribute_node, h5_id,
                            h5_path + "/cells", cells, {num_cells_global, 1},
                            "UInt");
}
//-----------------------------------------------------------------------------
