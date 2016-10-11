// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2012

#include <iomanip>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/container/vector.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include "pugixml.hpp"

#include <dolfin/common/MPI.h>
#include <dolfin/common/defines.h>
#include <dolfin/common/utils.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "HDF5File.h"
#include "HDF5Utility.h"
#include "XDMFFile.h"
#include "xmlutils.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XDMFFile::XDMFFile(MPI_Comm comm, const std::string filename)
  : _mpi_comm(comm), _filename(filename),
    _counter(0), _xml_doc(new pugi::xml_document)
{
  // Rewrite the mesh at every time step in a time series. Should be
  // turned off if the mesh remains constant.
  parameters.add("rewrite_function_mesh", true);

  // FIXME: This is only relevant to HDF5
  // Flush datasets to disk at each timestep. Allows inspection of the
  // HDF5 file whilst running, at some performance cost.
  parameters.add("flush_output", false);

  // FIXME: This is only relevant to HDF5
  // HDF5 file restart interval. Use 0 to collect all output in one
  // file.
  parameters.add("multi_file", 0);
}
//-----------------------------------------------------------------------------
XDMFFile::~XDMFFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const Mesh& mesh, Encoding encoding)
{
  // Check that encoding is supported
  check_encoding(encoding);

  // Reset pugi doc
  _xml_doc->reset();

  // Open a HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
#ifdef HAS_HDF5
  std::unique_ptr<HDF5File> h5_file;
  if (encoding == Encoding::HDF5)
  {
    // Open file
    h5_file.reset(new HDF5File(mesh.mpi_comm(), get_hdf5_filename(_filename), "w"));
    dolfin_assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }
#endif

  // Add XDMF node and version attribute
  _xml_doc->append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
  pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
  dolfin_assert(xdmf_node);
  xdmf_node.append_attribute("Version") = "3.0";
  xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";

  // Add domain node and add name attribute
  pugi::xml_node domain_node = xdmf_node.append_child("Domain");
  dolfin_assert(domain_node);

  // Add the mesh Grid to the domain
  add_mesh(_mpi_comm, domain_node, h5_id, mesh, "/Mesh");

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const Function& u, Encoding encoding)
{
  check_encoding(encoding);

  // If counter is non-zero, a time series has been saved before
  if (_counter != 0)
  {
    dolfin_error("XDMFFile.cpp",
                 "write Function to XDMF",
                 "Not writing a time series");
  }

  const Mesh& mesh = *u.function_space()->mesh();

  // Clear pugi doc
  _xml_doc->reset();

  // Open the HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
#ifdef HAS_HDF5
  std::unique_ptr<HDF5File> h5_file;
  if (encoding == Encoding::HDF5)
  {
    // Open file
    h5_file.reset(new HDF5File(mesh.mpi_comm(),
                               get_hdf5_filename(_filename), "w"));
    dolfin_assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }
#endif

  // Add XDMF node and version attribute
  pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
  dolfin_assert(xdmf_node);
  xdmf_node.append_attribute("Version") = "3.0";

  // Add domain node and add name attribute
  pugi::xml_node domain_node = xdmf_node.append_child("Domain");
  dolfin_assert(domain_node);

  // Add the mesh Grid to the domain
  add_mesh(_mpi_comm, domain_node, h5_id, mesh, "/Mesh");
  pugi::xml_node grid_node = domain_node.child("Grid");
  dolfin_assert(grid_node);

  // Get Function data values and shape
  std::vector<double> data_values;
  bool cell_centred = has_cell_centred_data(u);

  if (cell_centred)
    data_values = get_cell_data_values(u);
  else
    data_values = get_point_data_values(u);

  // Add attribute node
  pugi::xml_node attribute_node = grid_node.append_child("Attribute");
  dolfin_assert(attribute_node);
  attribute_node.append_attribute("Name") = u.name().c_str();
  attribute_node.append_attribute("AttributeType")
    = rank_to_string(u.value_rank()).c_str();
  attribute_node.append_attribute("Center") = cell_centred ? "Cell" : "Node";

  // Add attribute DataItem node and write data
  std::int64_t width = get_padded_width(u);
  dolfin_assert(data_values.size()%width == 0);
  std::int64_t num_values =  cell_centred ?
    mesh.size_global(mesh.topology().dim()) : mesh.size_global(0);

  add_data_item(_mpi_comm, attribute_node, h5_id,
                "/VisualisationVector/0", data_values, {num_values, width});

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const Function& u, double time_step, Encoding encoding)
{
  check_encoding(encoding);

  const Mesh& mesh = *u.function_space()->mesh();

  // Clear the pugi doc the first time
  if (_counter == 0)
  {
    _xml_doc->reset();

    // Create XDMF header
    _xml_doc->append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
    dolfin_assert(xdmf_node);
    xdmf_node.append_attribute("Version") = "3.0";
    xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";
    pugi::xml_node domain_node = xdmf_node.append_child("Domain");
    dolfin_assert(domain_node);

    //  /Xdmf/Domain/Grid - actually a TimeSeries, not a spatial grid
    pugi::xml_node timegrid_node = domain_node.append_child("Grid");
    dolfin_assert(timegrid_node);
    timegrid_node.append_attribute("Name") = "TimeSeries";
    timegrid_node.append_attribute("GridType") = "Collection";
    timegrid_node.append_attribute("CollectionType") = "Temporal";

    //  /Xdmf/Domain/Grid/Time
    pugi::xml_node time_node = timegrid_node.append_child("Time");
    dolfin_assert(time_node);
    time_node.append_attribute("TimeType") = "List";
    pugi::xml_node timedata_node = time_node.append_child("DataItem");
    dolfin_assert(timedata_node);
    timedata_node.append_attribute("Format") = "XML";
    timedata_node.append_attribute("Dimensions") = "0";
    timedata_node.append_child(pugi::node_pcdata);

#ifdef HAS_HDF5
    // Open the HDF5 file for first time, if using HDF5 encoding
    if (encoding == Encoding::HDF5)
    {
      // Truncate the file the first time
      std::string filemode = "w";
      // Open file
      _hdf5_file.reset(new HDF5File(mesh.mpi_comm(),
                                    get_hdf5_filename(_filename), filemode));
      dolfin_assert(_hdf5_file);
    }
#endif
  }

  // Get handle from existing file, (should be already open)
  hid_t h5_id = -1;
#ifdef HAS_HDF5
  if (encoding == Encoding::HDF5)
    h5_id = _hdf5_file->h5_id();
#endif

  pugi::xml_node xdmf_node = _xml_doc->child("Xdmf");
  dolfin_assert(xdmf_node);
  pugi::xml_node domain_node = xdmf_node.child("Domain");
  dolfin_assert(domain_node);

  // Look for existing TimeSeries and check name
  pugi::xml_node timegrid_node = domain_node.first_child();
  dolfin_assert(timegrid_node);
  dolfin_assert(std::string(timegrid_node.attribute("Name").value())
                == "TimeSeries");
  pugi::xml_node time_node = timegrid_node.child("Time");
  dolfin_assert(time_node);

  // Get existing number of timesteps, and values
  pugi::xml_node timedata_node = time_node.child("DataItem");
  dolfin_assert(timedata_node);
  pugi::xml_attribute numsteps_attr = timedata_node.attribute("Dimensions");
  dolfin_assert(numsteps_attr);
  pugi::xml_node times_str_node = timedata_node.first_child();
  dolfin_assert(times_str_node);
  unsigned int numsteps = std::stoul(numsteps_attr.value());
  dolfin_assert(numsteps == _counter);
  std::string times_str(times_str_node.value());

  // Add new values
  times_str += " " + boost::str((boost::format("%g") % time_step));
  ++numsteps;
  numsteps_attr.set_value(numsteps);
  times_str_node.set_value(times_str.c_str());

  // Add the mesh Grid to the time Grid
  add_mesh(_mpi_comm, timegrid_node,
           h5_id, mesh,
           "/Mesh/" + std::to_string(_counter));
  pugi::xml_node grid_node = timegrid_node.last_child();
  dolfin_assert(grid_node);

  // Get Function data values and shape
  std::vector<double> data_values;
  bool cell_centred = has_cell_centred_data(u);

  if (cell_centred)
    data_values = get_cell_data_values(u);
  else
    data_values = get_point_data_values(u);

  // Add attribute node
  pugi::xml_node attribute_node = grid_node.append_child("Attribute");
  dolfin_assert(attribute_node);
  attribute_node.append_attribute("Name") = u.name().c_str();
  attribute_node.append_attribute("AttributeType")
    = rank_to_string(u.value_rank()).c_str();
  attribute_node.append_attribute("Center") = cell_centred ? "Cell" : "Node";

  // Add attribute DataItem node and write data
  std::int64_t width = get_padded_width(u);
  dolfin_assert(data_values.size()%width == 0);
  std::int64_t num_values =  cell_centred ?
    mesh.size_global(mesh.topology().dim()) : mesh.size_global(0);

  const std::string dataset_name = "/VisualisationVector/"
                                   + std::to_string(_counter);

  add_data_item(_mpi_comm, attribute_node, h5_id,
                dataset_name, data_values, {num_values, width});

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");

  ++_counter;
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const MeshFunction<bool>& meshfunction, Encoding encoding)
{
  write_mesh_function(meshfunction, encoding);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const MeshFunction<int>& meshfunction, Encoding encoding)
{
  write_mesh_function(meshfunction, encoding);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const MeshFunction<std::size_t>& meshfunction,
                     Encoding encoding)
{
  write_mesh_function(meshfunction, encoding);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const MeshFunction<double>& meshfunction,
                     Encoding encoding)
{
  write_mesh_function(meshfunction, encoding);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const MeshValueCollection<std::size_t>& mvc,
                     Encoding encoding)
{
  check_encoding(encoding);

  // Provide some very basic functionality for saving
  // MeshValueCollections mainly for saving values on a boundary mesh

  dolfin_assert(mvc.mesh());
  std::shared_ptr<const Mesh> mesh = mvc.mesh();

  // Reset pugi
  _xml_doc->reset();

  // Open a HDF5 file if using HDF5 encoding (append)
  hid_t h5_id = -1;
#ifdef HAS_HDF5
  std::unique_ptr<HDF5File> h5_file;
  if (encoding == Encoding::HDF5)
  {
    // Open file
    h5_file.reset(new HDF5File(mesh->mpi_comm(), get_hdf5_filename(_filename), "w"));
    dolfin_assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }
#endif

  if (MPI::sum(mesh->mpi_comm(), mvc.size()) == 0)
  {
    dolfin_error("XDMFFile.cpp",
                 "save empty MeshValueCollection",
                 "No values in MeshValueCollection");
  }

  //  const std::size_t cell_dim = mvc.dim();
  //  CellType::Type cell_type = mesh->type().entity_type(cell_dim);

  // Add XDMF node and version attribute
  _xml_doc->append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
  pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
  dolfin_assert(xdmf_node);
  xdmf_node.append_attribute("Version") = "3.0";
  xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";

  // Add domain node and add name attribute
  pugi::xml_node domain_node = xdmf_node.append_child("Domain");
  dolfin_assert(domain_node);

  // Add the mesh Grid to the domain
  add_mesh(_mpi_comm, domain_node, h5_id, *mesh, "/Mesh");

  // FIXME: Add MVC topology
  // FIXME: Reference back to Mesh geometry
  // FIXME: Add MVC data

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const std::vector<Point>& points, Encoding encoding)
{
  // Check that encoding is supported
  check_encoding(encoding);

  // Create pugi doc
  _xml_doc->reset();

  // Open a HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
#ifdef HAS_HDF5
  std::unique_ptr<HDF5File> h5_file;
  if (encoding == Encoding::HDF5)
  {
    // Open file
    h5_file.reset(new HDF5File(_mpi_comm, get_hdf5_filename(_filename), "w"));
    dolfin_assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }
#endif

  // Add XDMF node and version attribute
  _xml_doc->append_child(pugi::node_doctype)
    .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
  pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
  dolfin_assert(xdmf_node);

  add_points(_mpi_comm, xdmf_node, h5_id, points);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::add_points(MPI_Comm comm, pugi::xml_node& xdmf_node,
                          hid_t h5_id, const std::vector<Point>& points)
{
  xdmf_node.append_attribute("Version") = "3.0";
  xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";
  pugi::xml_node domain_node = xdmf_node.append_child("Domain");
  dolfin_assert(domain_node);

  // Add a Grid to the domain
  pugi::xml_node grid_node = domain_node.append_child("Grid");
  dolfin_assert(grid_node);
  grid_node.append_attribute("GridType") = "Uniform";
  grid_node.append_attribute("Name") = "Point cloud";

  pugi::xml_node topology_node = grid_node.append_child("Topology");
  dolfin_assert(topology_node);
  const std::size_t n = points.size();
  const std::int64_t nglobal = MPI::sum(comm, n);
  topology_node.append_attribute("NumberOfElements")
    = std::to_string(nglobal).c_str();
  topology_node.append_attribute("TopologyType") = "PolyVertex";
  topology_node.append_attribute("NodesPerElement") = 1;

  pugi::xml_node geometry_node = grid_node.append_child("Geometry");
  dolfin_assert(geometry_node);
  geometry_node.append_attribute("GeometryType") = "XYZ";

  // Pack data
  std::vector<double> x(3*n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      x[3*i + j] = points[i][j];

  const std::vector<std::int64_t> shape = {nglobal, 3};
  add_data_item(comm, geometry_node, h5_id, "/Points/coordinates", x, shape);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const std::vector<Point>& points,
                     const std::vector<double>& values,
                     Encoding encoding)
{
  // Write clouds of points to XDMF/HDF5 with values
  dolfin_assert(points.size() == values.size());

  // Check that encoding is supported
  check_encoding(encoding);

  // Create pugi doc
  _xml_doc->reset();

  // Open a HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
#ifdef HAS_HDF5
  std::unique_ptr<HDF5File> h5_file;
  if (encoding == Encoding::HDF5)
  {
    // Open file
    h5_file.reset(new HDF5File(_mpi_comm, get_hdf5_filename(_filename), "w"));
    dolfin_assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }
#endif

  // Add XDMF node and version attribute
  _xml_doc->append_child(pugi::node_doctype)
    .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
  pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
  dolfin_assert(xdmf_node);

  add_points(_mpi_comm, xdmf_node, h5_id, points);

  // Add attribute node
  pugi::xml_node domain_node = xdmf_node.child("Domain");
  dolfin_assert(domain_node);
  pugi::xml_node grid_node = domain_node.child("Grid");
  dolfin_assert(grid_node);
  pugi::xml_node attribute_node = grid_node.append_child("Attribute");
  dolfin_assert(attribute_node);
  attribute_node.append_attribute("Name") = "Point values";
  attribute_node.append_attribute("AttributeType") = "Scalar";
  attribute_node.append_attribute("Center") = "Node";

  // Add attribute DataItem node and write data
  std::int64_t num_values =  MPI::sum(_mpi_comm, values.size());
  add_data_item(_mpi_comm, attribute_node, h5_id,
                "/Points/values", values, {num_values, 1});

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
void XDMFFile::read(MeshFunction<bool>& meshfunction, std::string name)
{
  const std::shared_ptr<const Mesh> mesh = meshfunction.mesh();
  dolfin_assert(mesh);

  const std::size_t cell_dim = meshfunction.dim();
  MeshFunction<std::size_t> mf(mesh, cell_dim);
  read_mesh_function(mf, name);

  for (MeshEntityIterator cell(*mesh, cell_dim); !cell.end(); ++cell)
    meshfunction[cell->index()] = (mf[cell->index()] == 1);
}
//----------------------------------------------------------------------------
void XDMFFile::read(MeshFunction<int>& meshfunction, std::string name)
{
  read_mesh_function(meshfunction, name);
}
//----------------------------------------------------------------------------
void XDMFFile::read(MeshFunction<std::size_t>& meshfunction, std::string name)
{
  read_mesh_function(meshfunction, name);
}
//----------------------------------------------------------------------------
void XDMFFile::read(MeshFunction<double>& meshfunction, std::string name)
{
  read_mesh_function(meshfunction, name);
}
//----------------------------------------------------------------------------
void XDMFFile::add_mesh(MPI_Comm comm, pugi::xml_node& xml_node,
                        hid_t h5_id, const Mesh& mesh,
                        const std::string path_prefix)
{
  // Add grid node and attributes
  pugi::xml_node grid_node = xml_node.append_child("Grid");
  dolfin_assert(grid_node);
  grid_node.append_attribute("Name") = mesh.name().c_str();
  grid_node.append_attribute("GridType") = "Uniform";

  // Add topology node and attributes (including writing data)
  const int tdim = mesh.topology().dim();
  const std::int64_t num_global_cells = mesh.size_global(tdim);
  if (num_global_cells < 1e9)
    add_topology_data<std::int32_t>(comm, grid_node, h5_id, path_prefix,
                                    mesh, tdim);
  else
    add_topology_data<std::int64_t>(comm, grid_node, h5_id, path_prefix,
                                    mesh, tdim);

  // Add geometry node and attributes (including writing data)
  add_geometry_data(comm, grid_node, h5_id, path_prefix, mesh);
}
//-----------------------------------------------------------------------------
void XDMFFile::read(Mesh& mesh) const
{
  // Extract parent filepath (required by HDF5 when XDMF stores relative path
  // of the HDF5 files(s) and the XDMF is not opened from its own directory)
  boost::filesystem::path xdmf_filename(_filename);
  const boost::filesystem::path parent_path = xdmf_filename.parent_path();

  if (!boost::filesystem::exists(xdmf_filename))
  {
    dolfin_error("XDMFFile.cpp",
                 "open XDMF file",
                 "XDMF file \"%s\" does not exist", _filename.c_str());
  }

  // Load XML doc from file
  pugi::xml_document xml_doc;
  pugi::xml_parse_result result = xml_doc.load_file(_filename.c_str());
  dolfin_assert(result);

  // Get XDMF node
  pugi::xml_node xdmf_node = xml_doc.child("Xdmf");
  dolfin_assert(xdmf_node);

  // Get domain node
  pugi::xml_node domain_node = xdmf_node.child("Domain");
  dolfin_assert(domain_node);

  // Get grid node
  pugi::xml_node grid_node = domain_node.child("Grid");
  dolfin_assert(grid_node);

  // Get topology node
  pugi::xml_node topology_node = grid_node.child("Topology");
  dolfin_assert(topology_node);

  // Get cell type
  const std::string cell_type_str = get_cell_type(topology_node);

  // Get toplogical dimensions
  std::unique_ptr<CellType> cell_type(CellType::create(cell_type_str));
  dolfin_assert(cell_type);
  const int tdim = cell_type->dim();
  const std::int64_t num_cells_global = get_num_cells(topology_node);

  // Get geometry node
  pugi::xml_node geometry_node = grid_node.child("Geometry");
  dolfin_assert(geometry_node);

  // Determine geometric dimension
  pugi::xml_attribute geometry_type_attr = geometry_node.attribute("GeometryType");
  dolfin_assert(geometry_type_attr);
  int gdim = -1;
  const std::string geometry_type =  geometry_type_attr.value();
  if (geometry_type == "XY")
    gdim = 2;
  else if (geometry_type == "XYZ")
    gdim = 3;
  else
  {
    dolfin_error("XDMFFile.cpp",
                 "determine geometric dimension",
                 "GeometryType \"%s\" in XDMF file is unknown or unsupported",
                 geometry_type.c_str());
  }

  // Get number of points from Geometry dataitem node
  pugi::xml_node geometry_data_node = geometry_node.child("DataItem");
  dolfin_assert(geometry_data_node);
  const std::vector<std::int64_t> gdims = get_dataset_shape(geometry_data_node);
  dolfin_assert(gdims.size() == 2);
  const std::int64_t num_points_global = gdims[0];
  dolfin_assert(gdims[1] == gdim);

  // Get topology dataset node
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  dolfin_assert(topology_data_node);

  if (MPI::size(_mpi_comm) == 1)
  {
    build_mesh(mesh, cell_type_str, num_points_global, num_cells_global,
               cell_type->num_vertices(),
               tdim, gdim, topology_data_node, geometry_data_node,
               parent_path);
  }
  else
  {
    // Build local mesh data structure
    LocalMeshData local_mesh_data(_mpi_comm);
    build_local_mesh_data(local_mesh_data, *cell_type, num_points_global,
                          num_cells_global,
                          cell_type->num_vertices(), tdim, gdim,
                          topology_data_node, geometry_data_node,
                          parent_path);
    local_mesh_data.check();

    // Build mesh
    const std::string ghost_mode = dolfin::parameters["ghost_mode"];
    MeshPartitioning::build_distributed_mesh(mesh, local_mesh_data, ghost_mode);
  }
}
//----------------------------------------------------------------------------
void XDMFFile::build_mesh(Mesh& mesh, std::string cell_type_str,
                          std::int64_t num_points, std::int64_t num_cells,
                          int num_points_per_cell,
                          int tdim, int gdim,
                          const pugi::xml_node& topology_dataset_node,
                          const pugi::xml_node& geometry_dataset_node,
                          const boost::filesystem::path& relative_path)
{
  MeshEditor mesh_editor;
  mesh_editor.open(mesh, cell_type_str, tdim, gdim);

  // Get topology data vector and add to mesh
  {
    // Get the data
    dolfin_assert(topology_dataset_node);
    std::vector<std::int32_t> topology_data
      = get_dataset<std::int32_t>(mesh.mpi_comm(), topology_dataset_node,
                                  relative_path);

    // Check dims

    // Prepate mesh editor for addition of cells
    mesh_editor.init_cells_global(num_cells, num_cells);

    // Prepare mesh editor for addition of cells, and add cell topology
    std::vector<std::size_t> cell_topology(num_points_per_cell);
    for (std::int64_t i = 0; i < num_cells; ++i)
    {
      cell_topology.assign(topology_data.begin() +  i*num_points_per_cell,
                           topology_data.begin() +  (i + 1)*num_points_per_cell);
      mesh_editor.add_cell(i, cell_topology);
    }
  }

  // Get geometry data vector and add to mesh
  {
    // Get geometry data
    dolfin_assert(geometry_dataset_node);
    std::vector<double> geometry_data
      = get_dataset<double>(mesh.mpi_comm(), geometry_dataset_node, relative_path);

    // Check dims

    // Prepare mesh editor for addition of points, and add points
    mesh_editor.init_vertices_global(num_points, num_points);
    Point p;
    for (std::int64_t i = 0; i < num_points; ++i)
    {
      for (int j = 0; j < gdim; ++j)
        p[j] = geometry_data[i*gdim + j];
      mesh_editor.add_vertex(i, p);
    }
  }

  mesh_editor.close();
}
//----------------------------------------------------------------------------
void
XDMFFile::build_local_mesh_data(LocalMeshData& local_mesh_data,
                                const CellType& cell_type,
                                const std::int64_t num_points_global,
                                const std::int64_t num_cells_global,
                                const int num_points_per_cell,
                                const int tdim, const int gdim,
                                const pugi::xml_node& topology_dataset_node,
                                const pugi::xml_node& geometry_dataset_node,
                                const boost::filesystem::path& relative_path)
{
  // -- Topology --

  // Get number of vertices per cell from CellType
  const int num_vertices_per_cell = cell_type.num_entities(0);

  // Set topology attributes
  local_mesh_data.topology.dim = cell_type.dim();
  local_mesh_data.topology.cell_type = cell_type.cell_type();
  local_mesh_data.topology.num_vertices_per_cell = num_vertices_per_cell;
  local_mesh_data.topology.num_global_cells = num_cells_global;

  // Get share of topology data
  dolfin_assert(topology_dataset_node);
  const auto topology_data = get_dataset<std::int64_t>(local_mesh_data.mpi_comm(),
                                                       topology_dataset_node,
                                                      relative_path);
  dolfin_assert(topology_data.size() % num_vertices_per_cell == 0);

  // Wrap topology data as multi-dimensional array
  const int num_local_cells = topology_data.size()/num_vertices_per_cell;
  const boost::const_multi_array_ref<std::int64_t, 2>
    topology_data_array(topology_data.data(),
                        boost::extents[num_local_cells][num_vertices_per_cell]);

  // Remap vertices to DOLFIN ordering from VTK/XDMF ordering
  local_mesh_data.topology.cell_vertices.resize(boost::extents[num_local_cells][num_vertices_per_cell]);
  const std::vector<std::int8_t> perm = cell_type.vtk_mapping();
  for (int i = 0; i < num_local_cells; ++i)
  {
    for (int j = 0; j < num_vertices_per_cell; ++j)
      local_mesh_data.topology.cell_vertices[i][j] = topology_data_array[i][perm[j]];
  }

  // Set cell global indices by adding offset
  const std::int64_t cell_index_offset
    = MPI::global_offset(local_mesh_data.mpi_comm(), num_local_cells, true);
  local_mesh_data.topology.global_cell_indices.resize(num_local_cells);
  std::iota(local_mesh_data.topology.global_cell_indices.begin(),
            local_mesh_data.topology.global_cell_indices.end(),
            cell_index_offset);

  // -- Geometry --

  // Set geometry attributes
  local_mesh_data.geometry.num_global_vertices = num_points_global;
  local_mesh_data.geometry.dim = gdim;

  // Read geometry dataset
  dolfin_assert(geometry_dataset_node);
  const auto geometry_data = get_dataset<double>(local_mesh_data.mpi_comm(),
                                                 geometry_dataset_node,
                                                 relative_path);
  dolfin_assert(geometry_data.size() % gdim == 0);

  // Deduce number of vertices that have been read on this process
  const int num_local_vertices = geometry_data.size()/gdim;

  // Copy geometry data into LocalMeshData
  local_mesh_data.geometry.vertex_coordinates.resize(boost::extents[num_local_vertices][gdim]);
  std::copy(geometry_data.begin(), geometry_data.end(),
            local_mesh_data.geometry.vertex_coordinates.data());

  // vertex offset
  const std::int64_t vertex_index_offset
    = MPI::global_offset(local_mesh_data.mpi_comm(), num_local_vertices, true);
  local_mesh_data.geometry.vertex_indices.resize(num_local_vertices);
  std::iota(local_mesh_data.geometry.vertex_indices.begin(),
            local_mesh_data.geometry.vertex_indices.end(),
            vertex_index_offset);
}
//----------------------------------------------------------------------------
template<typename T>
void XDMFFile::add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                 hid_t h5_id, const std::string path_prefix,
                                 const Mesh& mesh, int tdim)
{
  // Get number of cells (global) and vertices per cell from mesh
  const std::int64_t num_cells = mesh.topology().size_global(tdim);
  const int num_vertices_per_cell = mesh.type().num_vertices(tdim);

  // Get VTK string for cell type
  const std::string vtk_cell_str
    = vtk_cell_type_str(mesh.type().entity_type(tdim), mesh.geometry().degree());

  pugi::xml_node topology_node = xml_node.append_child("Topology");
  dolfin_assert(topology_node);
  topology_node.append_attribute("NumberOfElements") = std::to_string(num_cells).c_str();
  topology_node.append_attribute("TopologyType") = vtk_cell_str.c_str();
  topology_node.append_attribute("NodesPerElement") = num_vertices_per_cell;

  // Compute packed topology data
  std::vector<T> topology_data = compute_topology_data<T>(mesh, tdim);

  // Add topology DataItem node
  const std::string group_name = path_prefix + "/" + mesh.name();
  const std::string h5_path = group_name + "/topology";
  const std::vector<std::int64_t> shape = {num_cells, num_vertices_per_cell};
  const std::string number_type = "UInt";

  add_data_item(comm, topology_node, h5_id, h5_path, topology_data, shape, number_type);
}
//-----------------------------------------------------------------------------
void XDMFFile::add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                 hid_t h5_id, const std::string path_prefix,
                                 const Mesh& mesh)
{
  const MeshGeometry& mesh_geometry = mesh.geometry();
  int gdim = mesh_geometry.dim();

  // Compute number of points (global) in mesh (equal to number of vertices
  // for affine meshes)
  std::int64_t num_points = 0;
  for (std::size_t i = 0; i <= mesh.topology().dim(); ++i)
  {
    num_points +=
      mesh_geometry.num_entity_coordinates(i)*mesh.size_global(i);
  }

  // Add geometry node and attributes
  pugi::xml_node geometry_node = xml_node.append_child("Geometry");
  dolfin_assert(geometry_node);
  dolfin_assert(gdim > 0 and gdim <= 3);
  const std::string geometry_type = ( gdim == 1 or gdim == 2) ? "XY" : "XYZ";
  geometry_node.append_attribute("GeometryType") = geometry_type.c_str();

  // Pack geometry data
  std::vector<double> x
    = DistributedMeshTools::reorder_vertices_by_global_indices(mesh);

  // XDMF does not support 1D, so handle as special case
  if (gdim == 1)
  {
    // Pad the coordinates with zeros for a dummy Y
    gdim = 2;
    std::vector<double> _x(2*x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i)
      _x[2*i] = x[i];
    std::swap(x, _x);
  }

  // Add geometry DataItem node
  const std::string group_name = path_prefix + "/" + mesh.name();
  const std::string h5_path = group_name + "/geometry";
  const std::vector<std::int64_t> shape = {num_points, gdim};
  add_data_item(comm, geometry_node, h5_id, h5_path, x, shape);
}
//-----------------------------------------------------------------------------
template<typename T>
void XDMFFile::add_data_item(MPI_Comm comm, pugi::xml_node& xml_node,
                             hid_t h5_id, const std::string h5_path, const T& x,
                             const std::vector<std::int64_t> shape,
                             const std::string number_type)
{
  // Add DataItem node
  dolfin_assert(xml_node);
  pugi::xml_node data_item_node = xml_node.append_child("DataItem");
  dolfin_assert(data_item_node);

  // Add dimensions attribute
  data_item_node.append_attribute("Dimensions")
    = container_to_string(shape, " ", 16).c_str();

  // Set type for topology data (needed by XDMF to prevent default to float)
  if (!number_type.empty())
    data_item_node.append_attribute("NumberType") = number_type.c_str();

  // Add format attribute
  if (h5_id < 0)
  {
    data_item_node.append_attribute("Format") = "XML";
    data_item_node.append_child(pugi::node_pcdata).set_value(container_to_string(x, " ", 16).c_str());
  }
  else
  {
#ifdef HAS_HDF5
    data_item_node.append_attribute("Format") = "HDF";

    // Get name of HDF5 file
    const std::string hdf5_filename = HDF5Interface::get_filename(h5_id);
    const boost::filesystem::path p(hdf5_filename);

    // Add HDF5 filename and HDF5 internal path to XML file
    const std::string xdmf_path = p.filename().string() + ":" + h5_path;
    data_item_node.append_child(pugi::node_pcdata).set_value(xdmf_path.c_str());

    // Compute total number of items and check for consistency with shape
    dolfin_assert(!shape.empty());
    std::int64_t num_items_total = 1;
    for (auto n : shape)
      num_items_total *= n;

    dolfin_assert(num_items_total == (std::int64_t) MPI::sum(comm, x.size()));

    // Compute data offset and range of values
    std::int64_t local_shape0 = x.size();
    for (std::size_t i = 1; i < shape.size(); ++i)
    {
      dolfin_assert(local_shape0 % shape[i] == 0);
      local_shape0 /= shape[i];
    }
    const std::int64_t offset = MPI::global_offset(comm, local_shape0, true);
    const std::pair<std::int64_t, std::int64_t> local_range
      = {offset, offset + local_shape0};

    const bool use_mpi_io = MPI::size(comm) == 1 ? false : true;
    HDF5Interface::write_dataset(h5_id, h5_path, x, local_range, shape, use_mpi_io,
                                 false);

    // Add partitioning attribute to dataset
    std::vector<std::size_t> partitions;
    std::vector<std::size_t> offset_tmp(1, offset);
    MPI::gather(comm, offset_tmp, partitions);
    MPI::broadcast(comm, partitions);
    HDF5Interface::add_attribute(h5_id, h5_path, "partition", partitions);

#else
    // Should never reach this point
    dolfin_error("XDMFFile.cpp",
                 "add dataitem",
                 "DOLFIN has not been configured with HDF5");
#endif
  }
}
//----------------------------------------------------------------------------
std::set<unsigned int>
XDMFFile::compute_nonlocal_entities(const Mesh& mesh, int cell_dim)
{
  // If not already numbered, number entities of
  // order cell_dim so we can get shared_entities
  DistributedMeshTools::number_entities(mesh, cell_dim);

  const std::size_t mpi_rank = MPI::rank(mesh.mpi_comm());
  const std::map<std::int32_t, std::set<unsigned int>>& shared_entities
    = mesh.topology().shared_entities(cell_dim);

  std::set<unsigned int> non_local_entities;

  const std::size_t tdim = mesh.topology().dim();
  bool ghosted
    = (mesh.topology().size(tdim) > mesh.topology().ghost_offset(tdim));

  if (!ghosted)
  {
    // No ghost cells - exclude shared entities
    // which are on lower rank processes
    for (const auto &e : shared_entities)
    {
      const unsigned int lowest_rank_owner = *(e.second.begin());
      if (lowest_rank_owner < mpi_rank)
        non_local_entities.insert(e.first);
    }
  }
  else
  {
    // Iterate through ghost cells, adding non-ghost entities
    // which are in lower rank process cells
    for (MeshEntityIterator c(mesh, tdim, "ghost"); !c.end(); ++c)
    {
      const unsigned int cell_owner = c->owner();
      for (MeshEntityIterator e(*c, cell_dim); !e.end(); ++e)
        if (!e->is_ghost() && cell_owner < mpi_rank)
          non_local_entities.insert(e->index());
    }
  }
  return non_local_entities;
}
//-----------------------------------------------------------------------------
template<typename T>
std::vector<T> XDMFFile::compute_topology_data(const Mesh& mesh, int cell_dim)
{
  // Create vector to store topology data
  const int num_vertices_per_cell = mesh.type().num_vertices(cell_dim);
  std::vector<T> topology_data;
  topology_data.reserve(mesh.num_entities(cell_dim)*(num_vertices_per_cell));

  // Get mesh communicator
  MPI_Comm comm = mesh.mpi_comm();

  const std::vector<std::int8_t> perm = mesh.type().vtk_mapping();
  const int tdim = mesh.topology().dim();
  if (MPI::size(comm) == 1 or cell_dim == tdim)
  {
    // Simple case when nothing is shared between processes
    if (cell_dim == 0)
    {
      for (VertexIterator v(mesh); !v.end(); ++v)
        topology_data.push_back(v->global_index());
    }
    else
    {
      const std::vector<size_t>& global_vertices
        = mesh.topology().global_indices(0);
      for (MeshEntityIterator c(mesh, cell_dim); !c.end(); ++c)
      {
        const unsigned int* entities = c->entities(0);
        for (unsigned int i = 0; i != c->num_entities(0); ++i)
          topology_data.push_back(global_vertices[entities[perm[i]]]);
      }
    }
  }
  else
  {
    std::set<unsigned int> non_local_entities
      = compute_nonlocal_entities(mesh, cell_dim);

    if (cell_dim == 0)
    {
      // Special case for mesh of points
      for (VertexIterator v(mesh); !v.end(); ++v)
      {
        if (non_local_entities.find(v->index())
            == non_local_entities.end())
          topology_data.push_back(v->global_index());
      }
    }
    else
    {
      const std::vector<size_t>& global_vertices
        = mesh.topology().global_indices(0);
      for (MeshEntityIterator e(mesh, cell_dim); !e.end(); ++e)
      {
        // If not excluded, add to topology
        if (non_local_entities.find(e->index())
            == non_local_entities.end())
        {
          for (unsigned int i = 0; i != e->num_entities(0); ++i)
          {
            const unsigned int local_idx = e->entities(0)[perm[i]];
            topology_data.push_back(global_vertices[local_idx]);
          }
        }
      }
    }
  }

  return topology_data;
}
//----------------------------------------------------------------------------
template<typename T>
std::vector<T> XDMFFile::compute_value_data(const MeshFunction<T>& meshfunction)
{
  // Create vector to store data
  std::vector<T> value_data;
  value_data.reserve(meshfunction.size());

  // Get mesh communicator
  const auto mesh = meshfunction.mesh();
  MPI_Comm comm = mesh->mpi_comm();

  const int tdim = mesh->topology().dim();
  const int cell_dim = meshfunction.dim();

  if (MPI::size(comm) == 1 or cell_dim == tdim)
  {
    // FIXME: fail with ghosts?
    value_data.resize(meshfunction.size());
    std::copy(meshfunction.values(),
              meshfunction.values() + meshfunction.size(),
              value_data.begin());
  }
  else
  {
    std::set<unsigned int> non_local_entities
      = compute_nonlocal_entities(*mesh, cell_dim);

    for (MeshEntityIterator e(*mesh, cell_dim); !e.end(); ++e)
    {
      if (non_local_entities.find(e->index())
          == non_local_entities.end())
        value_data.push_back(meshfunction[*e]);
    }
  }

  return value_data;
}
//----------------------------------------------------------------------------
std::string XDMFFile::get_cell_type(const pugi::xml_node& topology_node)
{
  dolfin_assert(topology_node);
  pugi::xml_attribute type_attr = topology_node.attribute("TopologyType");
  dolfin_assert(type_attr);

  // Convert XDMF cell type string to DOLFIN cell type string
  std::string cell_type = type_attr.as_string();
  boost::algorithm::to_lower(cell_type);
  if (cell_type == "polyline")
    cell_type = "interval";

  return cell_type;
}
//----------------------------------------------------------------------------
std::vector<std::int64_t>
XDMFFile::get_dataset_shape(const pugi::xml_node& dataset_node)
{
  // Get Dimensions attribute string
  dolfin_assert(dataset_node);
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
std::int64_t XDMFFile::get_num_cells(const pugi::xml_node& topology_node)
{
  dolfin_assert(topology_node);

  // Get number of cells from topology
  std::int64_t num_cells_topolgy = -1;
  pugi::xml_attribute num_cells_attr = topology_node.attribute("NumberOfElements");
  if (num_cells_attr)
    num_cells_topolgy = num_cells_attr.as_llong();

  // Get number of cells from topology dataset
  pugi::xml_node topology_dataset_node = topology_node.child("DataItem");
  dolfin_assert(topology_dataset_node);
  const std::vector<std::int64_t> tdims = get_dataset_shape(topology_dataset_node);

  // Check that number of cells can be determined
  if (tdims.size() != 2 and num_cells_topolgy == -1)
  {
    dolfin_error("XDMFFile.cpp",
                 "determine number of cells",
                 "Cannot determine number of cells if XMDF mesh");
  }

  // Check for consistency if number of cells appears in both the topology
  // and DataItem nodes
  if (num_cells_topolgy != -1 and tdims.size() == 2)
  {
    if (num_cells_topolgy != tdims[0])
    {
      dolfin_error("XDMFFile.cpp",
                   "determine number of cells",
                   "Cannot determine number of cells if XMDF mesh");
     }
  }

  return std::max(num_cells_topolgy, tdims[0]);
}
//----------------------------------------------------------------------------
template <typename T>
std::vector<T> XDMFFile::get_dataset(MPI_Comm comm,
                                    const pugi::xml_node& dataset_node,
                                    const boost::filesystem::path& parent_path)
{
  // FIXME: Need to sort out datasset dimensions - can't depend on
  // HDF5 shape, and a Topology data item is not required to have a
  // 'Dimensions' attribute since the dimensions can be determined
  // from the number of cells and the cell type (for topology, one
  // must supply cell type + (number of cells or dimensions).
  //
  // A geometry data item must have 'Dimensions' attribute.

  dolfin_assert(dataset_node);
  pugi::xml_attribute format_attr = dataset_node.attribute("Format");
  dolfin_assert(format_attr);

  // Get data set shape from 'Dimensions' attribute (empty if not available)
  const std::vector<std::int64_t> shape_xml = get_dataset_shape(dataset_node);

  const std::string format = format_attr.as_string();
  std::vector<T> data_vector;
  if (format == "XML")
  {
    // Read data and trim any leading/trailing whitespace
    pugi::xml_node data_node = dataset_node.first_child();
    dolfin_assert(data_node);
    std::string data_str = data_node.value();

    // Split data based on spaces and line breaks
    std::vector<boost::iterator_range<std::string::iterator>> data_vector_str;
    boost::split(data_vector_str, data_str, boost::is_any_of(" \n"));

    // Add data to numerical vector
    data_vector.reserve(data_vector_str.size());
    for (auto& v : data_vector_str)
    {
      if (v.begin() != v.end())
        data_vector.push_back(boost::lexical_cast<T>(boost::copy_range<std::string>(v)));
    }
  }
  else if (format == "HDF")
  {
    #ifdef HAS_HDF5
    // Get file and data path
    auto paths = get_hdf5_paths(dataset_node);

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
    dolfin_assert(!shape_hdf5.empty());
    dolfin_assert(shape_hdf5[0] != 0);

    // Determine range of data to read from HDF5 file. This is
    // complicated by the XML Dimension attribute and the HDF5 storage
    // possibly having different shapes, e.g. the HDF5 storgae may be a
    // flat array.
    std::pair<std::int64_t, std::int64_t> range;
    if (shape_xml == shape_hdf5)
      range = MPI::local_range(comm, shape_hdf5[0]);
    else if (!shape_xml.empty() and shape_hdf5.size() == 1)
    {
      // Size of dims > 0
      std::int64_t d = 1;
      for (std::size_t i = 1; i < shape_xml.size(); ++i)
        d *= shape_xml[i];

      // Check for data size consistency
      if (d*shape_xml[0] != shape_hdf5[0])
      {
        dolfin_error("XDMFFile.cpp",
                     "reading data from XDMF file",
                     "Data size in XDMF/XML and size of HDF5 dataset are inconsistent");
      }

      // Compute data range to read
      range = MPI::local_range(comm, shape_xml[0]);
      range.first *= d;
      range.second *= d;
    }
    else
    {
      dolfin_error("XDMFFile.cpp",
                   "reading data from XDMF file",
                   "This combination of array shapes in XDMF and HDF5 not supported");
    }

    // Retrieve data
    HDF5Interface::read_dataset(h5_file.h5_id(), paths[1], range, data_vector);
    #else
    // Should never reach this point
    dolfin_error("XDMFFile.cpp",
                 "get dataset",
                 "DOLFIN has not been configured with HDF5");
    #endif
  }
  else
  {
    dolfin_error("XDMFFile.cpp",
                 "reading data from XDMF file",
                 "Storage format \"%s\" is unknown", format.c_str());
  }

  // Get dimensions for consistency (if available in DataItem node)
  if (shape_xml.empty())
  {
    std::int64_t size = 1;
    for (auto dim : shape_xml)
      size *= dim;

    if (size != (std::int64_t) MPI::sum(comm, data_vector.size()))
    {
      dolfin_error("XDMFFile.cpp",
                   "reading data from XDMF file",
                   "Data sizes in attribute and size of data read are inconsistent");
     }
   }

   return data_vector;
}
//----------------------------------------------------------------------------
std::array<std::string, 2> XDMFFile::get_hdf5_paths(const pugi::xml_node& dataitem_node)
{
  // Check that node is a DataItem node
  dolfin_assert(dataitem_node);
  xmlutils::check_node_name(dataitem_node, "DataItem");

  // Check that format is HDF
  pugi::xml_attribute format_attr = dataitem_node.attribute("Format");
  dolfin_assert(format_attr);
  const std::string format = format_attr.as_string();
  if (format.compare("HDF") != 0)
  {
    dolfin_error("XDMFFile.cpp",
                 "extracting HDF5 filename and data path",
                 "DataItem format \"%s\" is not \"HDF\"", format.c_str());
  }

  // Get path data
  pugi::xml_node path_node = dataitem_node.first_child();
  dolfin_assert(path_node);

  // Create string from path and trim leading and trailing whitespace
  std::string path = path_node.text().get();
  boost::algorithm::trim(path);

  // Split string into file path and HD5 internal path
  std::vector<std::string> paths;
  boost::split(paths, path, boost::is_any_of(":"));
  dolfin_assert(paths.size() == 2);

  return {{paths[0], paths[1]}};
}
//-----------------------------------------------------------------------------
template<typename T>
void XDMFFile::read_mesh_function(MeshFunction<T>& meshfunction,
                                  std::string name)
{
  // Load XML doc from file
  pugi::xml_document xml_doc;
  pugi::xml_parse_result result = xml_doc.load_file(_filename.c_str());
  dolfin_assert(result);

  // Get XDMF node
  pugi::xml_node xdmf_node = xml_doc.child("Xdmf");
  dolfin_assert(xdmf_node);

  // Get domain node
  pugi::xml_node domain_node = xdmf_node.child("Domain");
  dolfin_assert(domain_node);

  // Check all Grid nodes for suitable dataset
  pugi::xml_node grid_node;
  for (pugi::xml_node node: domain_node.children("Grid"))
  {
    pugi::xml_node value_node = node.child("Attribute");
    if (value_node and (name == "" or name == value_node.attribute("Name").as_string()))
      {
        grid_node = node;
        break;
      }
  }

  if (!grid_node)
  {
    dolfin_error("XDMFFile.cpp",
                 "open MeshFunction for reading",
                 "Mesh Grid with data Attribute not found in XDMF");
  }

  // Get topology node
  pugi::xml_node topology_node = grid_node.child("Topology");
  dolfin_assert(topology_node);

  // Get value node
  pugi::xml_node value_node = grid_node.child("Attribute");
  dolfin_assert(value_node);

  // Get existing Mesh of MeshFunction
  const auto mesh = meshfunction.mesh();

  // Get cell type and topology of MeshFunction (may be different from Mesh)
  const std::string cell_type_str = get_cell_type(topology_node);
  std::unique_ptr<CellType> cell_type(CellType::create(cell_type_str));
  dolfin_assert(cell_type);
  const unsigned int num_vertices_per_cell = cell_type->num_entities(0);
  const unsigned int cell_dim = cell_type->dim();
  dolfin_assert(cell_dim == meshfunction.dim());
  const std::size_t num_entities_global = get_num_cells(topology_node);
  // Ensure size_global(cell_dim) is set and check dataset matches
  DistributedMeshTools::number_entities(*mesh, cell_dim);
  dolfin_assert(mesh->size_global(cell_dim) == num_entities_global);

  boost::filesystem::path xdmf_filename(_filename);
  const boost::filesystem::path parent_path = xdmf_filename.parent_path();

  // Get topology dataset
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  dolfin_assert(topology_data_node);
  const auto topology_data = get_dataset<std::int64_t>(mesh->mpi_comm(),
                                                       topology_data_node,
                                                       parent_path);
  dolfin_assert(topology_data.size() % num_vertices_per_cell == 0);

  // Get value dataset
  pugi::xml_node value_data_node = value_node.child("DataItem");
  dolfin_assert(value_data_node);
  std::vector<T> value_data
    = get_dataset<T>(_mpi_comm, value_data_node, parent_path);

  // Scatter/gather data across processes
  remap_meshfunction_data(meshfunction, topology_data, value_data);
}
//-----------------------------------------------------------------------------
template <typename T>
void XDMFFile::remap_meshfunction_data(MeshFunction<T>& meshfunction,
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
  const std::size_t num_processes = MPI::size(comm);

  // Wrap topology data in boost array
  dolfin_assert(topology_data.size()%vertices_per_entity == 0);
  const std::size_t num_entities = topology_data.size()/vertices_per_entity;

  // Send (sorted) entity topology and data to a post-office process
  // determined by the lowest global vertex index of the entity
  std::vector<std::vector<std::int64_t>> send_topology(num_processes);
  std::vector<std::vector<T>> send_values(num_processes);
  const std::size_t max_vertex = mesh->size_global(0);
  for (std::size_t i = 0; i < num_entities ; ++i)
  {
    std::vector<std::int64_t>
      cell_topology(topology_data.begin() + i*vertices_per_entity,
                    topology_data.begin() + (i + 1)*vertices_per_entity);
    std::sort(cell_topology.begin(), cell_topology.end());

    // Use first vertex to decide where to send this data
    const std::size_t destination_process
      = MPI::index_owner(comm, cell_topology.front(), max_vertex);

    send_topology[destination_process]
      .insert(send_topology[destination_process].end(),
              cell_topology.begin(),
              cell_topology.end());
    send_values[destination_process].push_back(value_data[i]);
  }

  std::vector<std::vector<std::int64_t>> receive_topology(num_processes);
  std::vector<std::vector<T>> receive_values(num_processes);
  MPI::all_to_all(comm, send_topology, receive_topology);
  MPI::all_to_all(comm, send_values, receive_values);

  // Generate requests for data from remote processes, based on the
  // first vertex of the MeshEntities which belong on this process
  // Send our process number, and our local index, so it can come back
  // directly to the right place
  std::vector<std::vector<std::int64_t>> send_requests(num_processes);
  const std::size_t rank = MPI::rank(comm);
  for (MeshEntityIterator cell(*mesh, cell_dim, "all"); !cell.end(); ++cell)
  {
    std::vector<std::int64_t> cell_topology;
    for (VertexIterator v(*cell); !v.end(); ++v)
      cell_topology.push_back(v->global_index());
    std::sort(cell_topology.begin(), cell_topology.end());

    // Use first vertex to decide where to send this request
    std::size_t send_to_process = MPI::index_owner(comm,
                                                   cell_topology.front(),
                                                   max_vertex);
    // Map to this process and local index by appending to send data
    cell_topology.push_back(cell->index());
    cell_topology.push_back(rank);
    send_requests[send_to_process].insert(send_requests[send_to_process].end(),
                                          cell_topology.begin(),
                                          cell_topology.end());
  }

  std::vector<std::vector<std::int64_t>> receive_requests(num_processes);
  MPI::all_to_all(comm, send_requests, receive_requests);

  // At this point, the data with its associated vertices is in
  // receive_values and receive_topology and the final destinations
  // are stored in receive_requests as
  // [vertices][index][process][vertices][index][process]...  Some
  // data will have more than one destination

  // Create a mapping from the topology vector to the desired data
  std::map<std::vector<std::int64_t>, T> cell_to_data;

  for (std::size_t i = 0; i < receive_values.size(); ++i)
  {
    dolfin_assert(receive_values[i].size()*vertices_per_entity
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
    for (auto p = receive_requests[i].begin();
         p != receive_requests[i].end(); p += (vertices_per_entity + 2))
    {
      const std::vector<std::int64_t> cell(p, p + vertices_per_entity);
      const std::size_t remote_index = *(p + vertices_per_entity);
      const std::size_t send_to_proc = *(p + vertices_per_entity + 1);

      const auto find_cell = cell_to_data.find(cell);
      dolfin_assert(find_cell != cell_to_data.end());
      send_values[send_to_proc].push_back(find_cell->second);
      send_topology[send_to_proc].push_back(remote_index);
    }
  }

  MPI::all_to_all(comm, send_topology, receive_topology);
  MPI::all_to_all(comm, send_values, receive_values);

  // At this point, receive_topology should only list the local indices
  // and received values should have the appropriate values for each
  for (std::size_t i = 0; i < receive_values.size(); ++i)
  {
    dolfin_assert(receive_values[i].size() == receive_topology[i].size());
    for (std::size_t j = 0; j < receive_values[i].size(); ++j)
      meshfunction[receive_topology[i][j]] = receive_values[i][j];
  }
}
//----------------------------------------------------------------------------
std::string XDMFFile::get_hdf5_filename(std::string xdmf_filename)
{
  boost::filesystem::path p(xdmf_filename);
  p.replace_extension(".h5");
  if (p.string() == xdmf_filename)
  {
    dolfin_error("XDMFile.cpp",
                  "deduce name of HDF5 file from XDMF filename",
                  "Filename clash. Check XDMF filename");
  }

  return p.string();
}
//----------------------------------------------------------------------------
template<typename T>
void XDMFFile::write_mesh_function(const MeshFunction<T>& meshfunction,
                                   Encoding encoding)
{
  check_encoding(encoding);

  if (meshfunction.size() == 0)
  {
    dolfin_error("XDMFFile.cpp",
                 "save empty MeshFunction",
                 "No values in MeshFunction");
  }

  // Get mesh
  dolfin_assert(meshfunction.mesh());
  std::shared_ptr<const Mesh> mesh = meshfunction.mesh();

  // Check if _xml_doc already has data. If not, create an outer structure
  // If it already has data, then we may append to it.

  pugi::xml_node domain_node;
  std::string hdf_filemode = "a";
  if (_xml_doc->child("Xdmf").empty())
  {
    // Add XDMF node and version attribute
    _xml_doc->append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
    dolfin_assert(xdmf_node);
    xdmf_node.append_attribute("Version") = "3.0";
    xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";

    // Add domain node and add name attribute
    domain_node = xdmf_node.append_child("Domain");
    hdf_filemode = "w";
  }
  else
    domain_node = _xml_doc->child("Xdmf").child("Domain");

  dolfin_assert(domain_node);

  // Open a HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
#ifdef HAS_HDF5
  std::unique_ptr<HDF5File> h5_file;
  if (encoding == Encoding::HDF5)
  {
    // Open file
    h5_file.reset(new HDF5File(mesh->mpi_comm(),
                               get_hdf5_filename(_filename), hdf_filemode));
    dolfin_assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }
#endif

  // Add grid node and attributes
  pugi::xml_node grid_node = domain_node.append_child("Grid");
  dolfin_assert(grid_node);
  grid_node.append_attribute("Name") = mesh->name().c_str();
  grid_node.append_attribute("GridType") = "Uniform";

  // Add topology node and attributes (including writing data)
  const std::size_t cell_dim = meshfunction.dim();

  // Make sure entities are numbered - only needed for Edge in 3D in parallel
  // FIXME: remove this once Edge in 3D in parallel works properly
  DistributedMeshTools::number_entities(*mesh, cell_dim);

  const std::int64_t num_global_cells = mesh->size_global(cell_dim);
  if (num_global_cells < 1e9)
    add_topology_data<std::int32_t>(_mpi_comm, grid_node, h5_id, "/MeshFunction",
                                    *mesh, cell_dim);
  else
    add_topology_data<std::int64_t>(_mpi_comm, grid_node, h5_id, "/MeshFunction",
                                    *mesh, cell_dim);

  // Add geometry node and attributes (including writing data)
  // FIXME: should/could be shared with Mesh
  add_geometry_data(_mpi_comm, grid_node, h5_id, "/MeshFunction", *mesh);

  // Add attribute node with values
  pugi::xml_node attribute_node = grid_node.append_child("Attribute");
  dolfin_assert(attribute_node);
  attribute_node.append_attribute("Name") = meshfunction.name().c_str();
  attribute_node.append_attribute("AttributeType") = "Scalar";
  attribute_node.append_attribute("Center") = "Cell";

  const std::int64_t num_values = mesh->size_global(cell_dim);
  // Add attribute DataItem node and write data

  // Copy values to vector, removing duplicates
  std::vector<T> values = compute_value_data(meshfunction);

  add_data_item(_mpi_comm, attribute_node, h5_id,
                "/MeshFunction/values", values, {num_values, 1});

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
std::vector<double> XDMFFile::get_cell_data_values(const Function& u)
{
  dolfin_assert(u.function_space()->dofmap());
  dolfin_assert(u.vector());

  const auto mesh = u.function_space()->mesh();
  const std::size_t value_size = u.value_size();
  const std::size_t value_rank = u.value_rank();

  // Allocate memory for function values at cell centres
  const std::size_t tdim = mesh->topology().dim();
  const std::size_t num_local_cells = mesh->topology().ghost_offset(tdim);
  const std::size_t local_size = num_local_cells*value_size;

  // Build lists of dofs and create map
  std::vector<dolfin::la_index> dof_set;
  dof_set.reserve(local_size);
  const auto dofmap = u.function_space()->dofmap();
  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    // Tabulate dofs
    const ArrayView<const dolfin::la_index> dofs
      = dofmap->cell_dofs(cell->index());
    const std::size_t ndofs = dofmap->num_element_dofs(cell->index());
    dolfin_assert(ndofs == value_size);
    for (std::size_t i = 0; i < ndofs; ++i)
      dof_set.push_back(dofs[i]);
  }

  // Get  values
  std::vector<double> data_values(dof_set.size());
  dolfin_assert(u.vector());
  u.vector()->get_local(data_values.data(), dof_set.size(), dof_set.data());

  if (value_rank == 1 && value_size == 2)
  {
    // Pad out data for 2D vector to 3D
    data_values.resize(3*num_local_cells);
    for (int j = (num_local_cells - 1); j >= 0; --j)
    {
      double nd[3] = {data_values[j*2], data_values[j*2 + 1], 0};
      std::copy(nd, nd + 3, &data_values[j*3]);
    }
  }
  else if (value_rank == 2 && value_size == 4)
  {
    data_values.resize(9*num_local_cells);
    for (int j = (num_local_cells - 1); j >= 0; --j)
    {
      double nd[9] = { data_values[j*4], data_values[j*4 + 1], 0,
                       data_values[j*4 + 2], data_values[j*4 + 3], 0,
                       0, 0, 0 };
      std::copy(nd, nd + 9, &data_values[j*9]);
    }
  }
  return data_values;
}
//-----------------------------------------------------------------------------
std::int64_t XDMFFile::get_padded_width(const Function& u)
{
  std::int64_t width = u.value_size();
  std::int64_t rank = u.value_rank();
  if (rank == 1 and width == 2)
    return 3;
  else if (rank == 2 and width == 4)
    return 9;
  return width;
}
//-----------------------------------------------------------------------------
bool XDMFFile::has_cell_centred_data(const Function& u)
{
  // Test for cell-centred data
  std::size_t cell_based_dim = 1;
  for (std::size_t i = 0; i < u.value_rank(); i++)
    cell_based_dim *= u.function_space()->mesh()->topology().dim();
  return (u.function_space()->dofmap()->max_element_dofs() == cell_based_dim);
}
//-----------------------------------------------------------------------------
std::vector<double> XDMFFile::get_point_data_values(const Function& u)
{
  const auto mesh = u.function_space()->mesh();

  std::vector<double> data_values;
  if (mesh->geometry().degree() == 1)
  {
    u.compute_vertex_values(data_values, *mesh);

    std::int64_t width = get_padded_width(u);

    if (u.value_rank() > 0)
    {
      // Transpose vector/tensor data arrays
      const std::size_t num_local_vertices = mesh->size(0);
      const std::size_t value_size = u.value_size();
      std::vector<double> _data_values(width*num_local_vertices, 0.0);
      for (std::size_t i = 0; i < num_local_vertices; i++)
      {
        for (std::size_t j = 0; j < value_size; j++)
        {
          std::size_t tensor_2d_offset = (j > 1 && value_size == 4) ? 1 : 0;
          _data_values[i*width + j + tensor_2d_offset]
            = data_values[i + j*num_local_vertices];
        }
      }
      data_values = _data_values;
    }

    // Remove duplicates for vertex-based data in parallel
    if (MPI::size(mesh->mpi_comm()) > 1)
    {
      DistributedMeshTools::reorder_values_by_global_indices(*mesh,
                                                             data_values, width);
    }

    return data_values;
  }

  // FIXME: need to break up
  assert(mesh->geometry().degree() == 2);

  const std::size_t value_size = u.value_size();
  const std::size_t value_rank = u.value_rank();
  const std::size_t num_local_points = mesh->size(0) + mesh->size(1);
  const std::size_t width = get_padded_width(u);
  data_values.resize(width*num_local_points);
  std::vector<dolfin::la_index> data_dofs(data_values.size(), 0);

  dolfin_assert(u.function_space()->dofmap());
  const auto dofmap = u.function_space()->dofmap();

  // Function can be P1 or P2
  if (dofmap->num_entity_dofs(1) == 0)
  {
    // P1
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
      const ArrayView<const dolfin::la_index> dofs
        = dofmap->cell_dofs(cell->index());
      std::size_t c = 0;
      for (std::size_t i = 0; i != value_size; ++i)
      {
        for (VertexIterator v(*cell); !v.end(); ++v)
        {
          const std::size_t v0 = v->index()*width;
          data_dofs[v0 + i] = dofs[c];
          ++c;
        }
      }
    }

    // Get the values at the vertex points
    const GenericVector& uvec = *u.vector();
    uvec.get_local(data_values.data(), data_dofs.size(), data_dofs.data());

    // Get midpoint values for Edge points
    for (EdgeIterator e(*mesh); !e.end(); ++e)
    {
      const std::size_t v0 = e->entities(0)[0];
      const std::size_t v1 = e->entities(0)[1];
      const std::size_t e0 = (e->index() + mesh->size(0))*width;
      for (std::size_t i = 0; i != value_size; ++i)
        data_values[e0 + i] = (data_values[v0 + i] + data_values[v1 + i])/2.0;
    }
  }
  else if (dofmap->num_entity_dofs(0) == dofmap->num_entity_dofs(1))
  {
    // P2
    // Go over all cells inserting values
    // FIXME: a lot of duplication here
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
      const ArrayView<const dolfin::la_index> dofs
        = dofmap->cell_dofs(cell->index());
      std::size_t c = 0;
      for (std::size_t i = 0; i != value_size; ++i)
      {
        for (VertexIterator v(*cell); !v.end(); ++v)
        {
          const std::size_t v0 = v->index()*width;
          data_dofs[v0 + i] = dofs[c];
          ++c;
        }
        for (EdgeIterator e(*cell); !e.end(); ++e)
        {
          const std::size_t e0 = (e->index() + mesh->size(0))*width;
          data_dofs[e0 + i] = dofs[c];
          ++c;
        }
      }
    }

    const GenericVector& uvec = *u.vector();
    uvec.get_local(data_values.data(), data_dofs.size(), data_dofs.data());
  }
  else
  {
    dolfin_error("XDMFFile.cpp",
                 "get point values for Function",
                 "Function appears not to be defined on a P1 or P2 type FunctionSpace");
  }

  // Blank out empty values of 2D vector and tensor
  if (value_rank == 1 and value_size == 2)
  {
    for (std::size_t i = 0; i < data_values.size(); i += 3)
      data_values[i + 2] = 0.0;
  }
  else if (value_rank == 2 and value_size == 4)
  {
    for (std::size_t i = 0; i < data_values.size(); i += 9)
    {
      data_values[i + 2] = 0.0;
      data_values[i + 5] = 0.0;
      data_values[i + 6] = 0.0;
      data_values[i + 7] = 0.0;
      data_values[i + 8] = 0.0;
    }
  }

  return data_values;
}
//----------------------------------------------------------------------------
void XDMFFile::check_encoding(Encoding encoding) const
{
  if (encoding == Encoding::HDF5 and !has_hdf5())
  {
    dolfin_error("XDMFFile.cpp",
                 "write XDMF file",
                 "DOLFIN has not been compiled with HDF5 support");
  }

  if (encoding == Encoding::ASCII and MPI::size(_mpi_comm) != 1)
  {
    dolfin_error("XDMFFile.cpp",
                 "write XDMF file",
                 "ASCII format is not supported in parallel, use HDF5");
  }
}
//-----------------------------------------------------------------------------
std::string XDMFFile::vtk_cell_type_str(CellType::Type cell_type, int order)
{
  // FIXME: Move to CellType?
  switch (cell_type)
  {
  case CellType::Type::point:
    switch (order)
    {
    case 1:
      return "PolyVertex";
    }
  case CellType::Type::interval:
    switch (order)
    {
    case 1:
      return "PolyLine";
    case 2:
      return "Edge_3";
    }
  case CellType::Type::triangle:
    switch (order)
    {
    case 1:
      return "Triangle";
    case 2:
      return "Tri_6";
    }
  case CellType::Type::quadrilateral:
    switch (order)
    {
    case 1:
      return "Quadrilateral";
    case 2:
      return "Quad_8";
    }
  case CellType::Type::tetrahedron:
    switch (order)
    {
    case 1:
      return "Tetrahedron";
    case 2:
      return "Tet_10";
    }
  case CellType::Type::hexahedron:
    switch (order)
    {
    case 1:
      return "Hexahedron";
    case 2:
      return "Hex_20";
    }
  default:
    dolfin_error("XDMFFile.cpp",
                 "output mesh topology",
                 "Invalid combination of cell type and order");
    return "error";
  }
}
//-----------------------------------------------------------------------------
template <typename X, typename Y>
std::string XDMFFile::to_string(X x, Y y)
{
  return std::to_string(x) + " " + std::to_string(y);
}
//-----------------------------------------------------------------------------
template <typename T>
std::vector<T> XDMFFile::string_to_vector(const std::vector<std::string>& x_str)
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
std::string XDMFFile::rank_to_string(std::size_t value_rank)
{
  if (value_rank > 2)
    dolfin_error("XDMFFile.cpp", "get rank string", "Out of range");
  if (value_rank == 0)
    return "Scalar";
  else if (value_rank == 1)
    return "Vector";
  return "Tensor";
}
//-----------------------------------------------------------------------------
