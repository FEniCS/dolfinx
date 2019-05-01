// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "XDMFFile.h"
#include "HDF5File.h"
#include "HDF5Utility.h"
#include "pugixml.hpp"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/container/vector.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <dolfin/common/MPI.h>
#include <dolfin/common/defines.h>
#include <dolfin/common/log.h>
#include <dolfin/common/utils.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/ReferenceCellTopology.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/utils.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Connectivity.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/mesh/Partitioning.h>
#include <dolfin/mesh/Vertex.h>
#include <iomanip>
#include <memory>
#include <petscvec.h>
#include <set>
#include <string>
#include <vector>

using namespace dolfin;
using namespace dolfin::io;

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
  if (_encoding == Encoding::ASCII and _mpi_comm.size() != 1)
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
    h5_file = std::make_unique<HDF5File>(mesh.mpi_comm(),
                                         get_hdf5_filename(_filename), "w");
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
  add_mesh(_mpi_comm.comm(), domain_node, h5_id, mesh, "/Mesh");

  // Save XML file (on process 0 only)
  if (_mpi_comm.rank() == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::write_checkpoint(const function::Function& u,
                                std::string function_name, double time_step)
{
  if (_encoding == Encoding::ASCII and _mpi_comm.size() != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }

  if (!name_same_on_all_procs(function_name))
  {
    throw std::runtime_error("Function name must be the same on all processes "
                             "when writing to XDMF file.");
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

  if (truncate_hdf and boost::filesystem::exists(get_hdf5_filename(_filename)))
  {
    LOG(WARNING) << "HDF file \"" << get_hdf5_filename(_filename)
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
          _mpi_comm.comm(), get_hdf5_filename(_filename), "w");
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
      _hdf5_file = std::unique_ptr<HDF5File>(
          new HDF5File(_mpi_comm.comm(), get_hdf5_filename(_filename), "a"));
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
  add_mesh(_mpi_comm.comm(), func_temporal_grid_node, h5_id, mesh,
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
    add_function(_mpi_comm.comm(), mesh_grid_node, h5_id,
                 function_name + "/" + function_time_name, u, function_name,
                 mesh, component);
  }

  // Save XML file (on process 0 only)
  if (_mpi_comm.rank() == 0)
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
void XDMFFile::write(const function::Function& u)
{
  // Check that encoding
  if (_encoding == Encoding::ASCII and _mpi_comm.size() != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }

  // If counter is non-zero, a time series has been saved before
  if (_counter != 0)
  {
    throw std::runtime_error("Cannot write function::Function to XDMF. "
                             "Not writing a time series");
  }

  const mesh::Mesh& mesh = *u.function_space()->mesh();

  // Clear pugi doc
  _xml_doc->reset();

  // Open the HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
  std::unique_ptr<HDF5File> h5_file;
  if (_encoding == Encoding::HDF5)
  {
    // Open file
    h5_file = std::make_unique<HDF5File>(mesh.mpi_comm(),
                                         get_hdf5_filename(_filename), "w");
    assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }

  // Add XDMF node and version attribute
  pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
  assert(xdmf_node);
  xdmf_node.append_attribute("Version") = "3.0";

  // Add domain node and add name attribute
  pugi::xml_node domain_node = xdmf_node.append_child("Domain");
  assert(domain_node);

  // Add the mesh Grid to the domain
  add_mesh(_mpi_comm.comm(), domain_node, h5_id, mesh, "/Mesh");

  pugi::xml_node grid_node = domain_node.child("Grid");
  assert(grid_node);

  // Get function::Function data values and shape
  std::vector<PetscScalar> data_values;
  bool cell_centred = has_cell_centred_data(u);
  if (cell_centred)
    data_values = get_cell_data_values(u);
  else
    data_values = get_point_data_values(u);

  // Add attribute DataItem node and write data
  std::int64_t width = get_padded_width(u);
  assert(data_values.size() % width == 0);

  const std::int64_t num_points = mesh.geometry().num_points_global();
  const std::int64_t num_values
      = cell_centred ? mesh.num_entities_global(mesh.topology().dim())
                     : num_points;

#ifdef PETSC_USE_COMPLEX
  std::vector<std::string> components = {"real", "imag"};
#else
  std::vector<std::string> components = {""};
#endif
  for (const std::string component : components)
  {
    std::string attr_name;
    if (component.empty())
      attr_name = u.name();
    else
      attr_name = component + "_" + u.name();

    // Add attribute node of the current component
    pugi::xml_node attribute_node = grid_node.append_child("Attribute");
    assert(attribute_node);
    attribute_node.append_attribute("Name") = attr_name.c_str();
    attribute_node.append_attribute("AttributeType")
        = rank_to_string(u.value_rank()).c_str();
    attribute_node.append_attribute("Center") = cell_centred ? "Cell" : "Node";

#ifdef PETSC_USE_COMPLEX
    // FIXME: Avoid copies by writing directly a compound data
    std::vector<double> component_data_values(data_values.size());
    for (unsigned int i = 0; i < data_values.size(); i++)
    {
      if (component == components[0])
        component_data_values[i] = data_values[i].real();
      else if (component == components[1])
        component_data_values[i] = data_values[i].imag();
    }
    // Add data item of component
    add_data_item(_mpi_comm.comm(), attribute_node, h5_id,
                  "/VisualisationVector/" + component + "/0",
                  component_data_values, {num_values, width});
#else
    // Add data item
    add_data_item(_mpi_comm.comm(), attribute_node, h5_id,
                  "/VisualisationVector/0", data_values, {num_values, width});
#endif
  }

  // Save XML file (on process 0 only)
  if (_mpi_comm.rank() == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const function::Function& u, double time_step)
{
  // Check that encoding
  if (_encoding == Encoding::ASCII and _mpi_comm.size() != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }

  const mesh::Mesh& mesh = *u.function_space()->mesh();

  // Clear the pugi doc the first time
  if (_counter == 0)
  {
    _xml_doc->reset();

    // Create XDMF header
    _xml_doc->append_child(pugi::node_doctype)
        .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
    assert(xdmf_node);
    xdmf_node.append_attribute("Version") = "3.0";
    xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";
    pugi::xml_node domain_node = xdmf_node.append_child("Domain");
    assert(domain_node);
  }

  hid_t h5_id = -1;
  // Open the HDF5 file for first time, if using HDF5 encoding
  if (_encoding == Encoding::HDF5)
  {
    // Truncate the file the first time
    if (_counter == 0)
      _hdf5_file = std::make_unique<HDF5File>(
          mesh.mpi_comm(), get_hdf5_filename(_filename), "w");
    else if (flush_output)
    {
      // Append to existing HDF5 file
      assert(!_hdf5_file);
      _hdf5_file = std::make_unique<HDF5File>(
          mesh.mpi_comm(), get_hdf5_filename(_filename), "a");
    }
    else if ((_counter != 0) and (!_hdf5_file))
    {
      // The XDMFFile was previously closed, and now must be reopened
      _hdf5_file = std::make_unique<HDF5File>(
          mesh.mpi_comm(), get_hdf5_filename(_filename), "a");
    }
    assert(_hdf5_file);
    h5_id = _hdf5_file->h5_id();
  }

  pugi::xml_node xdmf_node = _xml_doc->child("Xdmf");
  assert(xdmf_node);
  pugi::xml_node domain_node = xdmf_node.child("Domain");
  assert(domain_node);

  // Should functions share mesh or not? By default they do not
  std::string tg_name = "TimeSeries_" + u.name();
  if (functions_share_mesh)
    tg_name = "TimeSeries";

  // Look for existing time series grid node with Name == tg_name
  bool new_timegrid = false;
  std::string time_step_str = boost::lexical_cast<std::string>(time_step);
  pugi::xml_node timegrid_node, mesh_node;
  timegrid_node
      = domain_node.find_child_by_attribute("Grid", "Name", tg_name.c_str());

  // Ensure that we have a time series grid node
  if (timegrid_node)
  {
    // Get existing mesh grid node with the correct time step if it exist
    // (otherwise null)
    std::string xpath = std::string("Grid[Time/@Value=\"") + time_step_str
                        + std::string("\"]");
    mesh_node = timegrid_node.select_node(xpath.c_str()).node();
    assert(std::string(timegrid_node.attribute("CollectionType").value())
           == "Temporal");
  }
  else
  {
    //  Create a new time series grid node with Name = tg_name
    timegrid_node = domain_node.append_child("Grid");
    assert(timegrid_node);
    timegrid_node.append_attribute("Name") = tg_name.c_str();
    timegrid_node.append_attribute("GridType") = "Collection";
    timegrid_node.append_attribute("CollectionType") = "Temporal";
    new_timegrid = true;
  }

  // Only add mesh grid node at this time step if no other function has
  // previously added it (and functions_share_mesh == true)
  if (!mesh_node)
  {
    // Add the mesh grid node to to the time series grid node
    if (new_timegrid or rewrite_function_mesh)
    {
      add_mesh(_mpi_comm.comm(), timegrid_node, h5_id, mesh,
               "/Mesh/" + std::to_string(_counter));
    }
    else
    {
      // Make a grid node that references back to first mesh grid node of the
      // time series
      pugi::xml_node grid_node = timegrid_node.append_child("Grid");
      assert(grid_node);

      // Reference to previous topology and geometry document nodes via
      // XInclude
      std::string xpointer
          = std::string("xpointer(//Grid[@Name=\"") + tg_name
            + std::string("\"]/Grid[1]/*[self::Topology or self::Geometry])");
      pugi::xml_node reference = grid_node.append_child("xi:include");
      assert(reference);
      reference.append_attribute("xpointer") = xpointer.c_str();
    }

    // Get the newly created mesh grid node
    mesh_node = timegrid_node.last_child();
    assert(mesh_node);

    // Add time value to mesh grid node
    pugi::xml_node time_node = mesh_node.append_child("Time");
    time_node.append_attribute("Value") = time_step_str.c_str();
  }

  // Get function::Function data values and shape
  std::vector<PetscScalar> data_values;
  bool cell_centred = has_cell_centred_data(u);

  if (cell_centred)
    data_values = get_cell_data_values(u);
  else
    data_values = get_point_data_values(u);

  // Add attribute DataItem node and write data
  std::int64_t width = get_padded_width(u);
  assert(data_values.size() % width == 0);
  std::int64_t num_values
      = cell_centred ? mesh.num_entities_global(mesh.topology().dim())
                     : mesh.num_entities_global(0);

#ifdef PETSC_USE_COMPLEX
  std::vector<std::string> components = {"real", "imag"};
#else
  std::vector<std::string> components = {""};
#endif

  for (const std::string component : components)
  {
    std::string attr_name;
    std::string dataset_name;
    if (component.empty())
    {
      attr_name = u.name();
      dataset_name = "/VisualisationVector/" + std::to_string(_counter);
    }
    else
    {
      attr_name = component + "_" + u.name();
      dataset_name = "/VisualisationVector/" + component + "/"
                     + std::to_string(_counter);
    }
    // Add attribute node
    pugi::xml_node attribute_node = mesh_node.append_child("Attribute");
    assert(attribute_node);
    attribute_node.append_attribute("Name") = attr_name.c_str();
    attribute_node.append_attribute("AttributeType")
        = rank_to_string(u.value_rank()).c_str();
    attribute_node.append_attribute("Center") = cell_centred ? "Cell" : "Node";

#ifdef PETSC_USE_COMPLEX
    // FIXME: Avoid copies by writing directly a compound data
    std::vector<double> component_data_values(data_values.size());
    for (unsigned int i = 0; i < data_values.size(); i++)
    {
      if (component == components[0])
        component_data_values[i] = data_values[i].real();
      else if (component == components[1])
        component_data_values[i] = data_values[i].imag();
    }
    // Add data item of component
    add_data_item(_mpi_comm.comm(), attribute_node, h5_id, dataset_name,
                  component_data_values, {num_values, width});
#else
    // Add data item
    add_data_item(_mpi_comm.comm(), attribute_node, h5_id, dataset_name,
                  data_values, {num_values, width});
#endif
  }

  // Save XML file (on process 0 only)
  if (_mpi_comm.rank() == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");

  // Close the HDF5 file if in "flush" mode
  if (_encoding == Encoding::HDF5 and flush_output)
  {
    assert(_hdf5_file);
    _hdf5_file.reset();
  }

  ++_counter;
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const mesh::MeshFunction<bool>& meshfunction)
{
  write_mesh_function(meshfunction);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const mesh::MeshFunction<int>& meshfunction)
{
  write_mesh_function(meshfunction);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const mesh::MeshFunction<std::size_t>& meshfunction)
{
  write_mesh_function(meshfunction);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const mesh::MeshFunction<double>& meshfunction)
{
  write_mesh_function(meshfunction);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const mesh::MeshValueCollection<bool>& mvc)
{
  write_mesh_value_collection(mvc);
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
  if (_encoding == Encoding::ASCII and _mpi_comm.size() != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }

  // Provide some very basic functionality for saving
  // mesh::MeshValueCollections mainly for saving values on a boundary mesh

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
        mesh->mpi_comm(), get_hdf5_filename(_filename), hdf_filemode);
    assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }

  // Check domain node for existing mesh::Mesh Grid and check it is compatible
  // with
  // this mesh::MeshValueCollection, or if none, add mesh::Mesh

  pugi::xml_node grid_node = domain_node.child("Grid");
  if (grid_node.empty())
    add_mesh(_mpi_comm.comm(), domain_node, h5_id, *mesh, "/Mesh");
  else
  {
    // Check topology
    pugi::xml_node topology_node = grid_node.child("Topology");
    assert(topology_node);
    const std::int64_t ncells = mesh->topology().size_global(tdim);
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
  mvc_grid_node.append_attribute("Name") = mvc.name().c_str();
  mvc_grid_node.append_attribute("GridType") = "Uniform";

  // Add topology node and attributes
  const std::size_t cell_dim = mvc.dim();
  const std::size_t degree = 1;
  const std::string vtk_cell_str
      = vtk_cell_type_str(mesh->type().entity_type(cell_dim), degree);
  const std::int64_t num_vertices_per_cell
      = mesh->type().num_vertices(cell_dim);

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

  mesh->create_connectivity(tdim, cell_dim);
  for (auto& p : values)
  {
    mesh::MeshEntity cell = mesh::Cell(*mesh, p.first.first);
    if (cell_dim != tdim)
    {
      const std::int32_t entity_local_idx
          = cell.entities(cell_dim)[p.first.second];
      cell = mesh::MeshEntity(*mesh, cell_dim, entity_local_idx);
    }

    // if cell is actually a vertex
    if (cell.dim() == 0)
      topology_data.push_back(cell.global_index());
    else
    {
      for (auto& v : mesh::EntityRange<mesh::Vertex>(cell))
        topology_data.push_back(v.global_index());
    }

    value_data.push_back(p.second);
  }

  const std::string mvc_dataset_name
      = "/MeshValueCollection/" + std::to_string(_counter);
  const std::int64_t num_values = MPI::sum(mesh->mpi_comm(), value_data.size());
  add_data_item(_mpi_comm.comm(), topology_node, h5_id,
                mvc_dataset_name + "/topology", topology_data,
                {num_values, num_vertices_per_cell}, "UInt");

  // Add geometry node (share with main mesh::Mesh)
  pugi::xml_node geometry_node = mvc_grid_node.append_child("Geometry");
  assert(geometry_node);
  geometry_node.append_attribute("Reference") = "XML";
  geometry_node.append_child(pugi::node_pcdata)
      .set_value("/Xdmf/Domain/Grid/Geometry");

  // Add attribute node with values
  pugi::xml_node attribute_node = mvc_grid_node.append_child("Attribute");
  assert(attribute_node);
  attribute_node.append_attribute("Name") = mvc.name().c_str();
  attribute_node.append_attribute("AttributeType") = "Scalar";
  attribute_node.append_attribute("Center") = "Cell";

  add_data_item(_mpi_comm.comm(), attribute_node, h5_id,
                mvc_dataset_name + "/values", value_data, {num_values, 1});

  // Save XML file (on process 0 only)
  if (_mpi_comm.rank() == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");

  ++_counter;
}
//-----------------------------------------------------------------------------
mesh::MeshValueCollection<bool>
XDMFFile::read_mvc_bool(std::shared_ptr<const mesh::Mesh> mesh,
                        std::string name) const
{
  // Bool is not really supported, so copy from int
  mesh::MeshValueCollection<int> mvc_int
      = read_mesh_value_collection<int>(mesh, name);

  mesh::MeshValueCollection<bool> mvc(mesh, mvc_int.dim());
  for (const auto& p : mvc_int.values())
    mvc.set_value(p.first.first, p.first.second, (bool)p.second);

  return mvc;
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
  auto cell_type_str = get_cell_type(topology_node);
  assert(cell_type_str.second == 1);
  std::unique_ptr<mesh::CellType> cell_type(
      mesh::CellType::create(cell_type_str.first));
  assert(cell_type);
  const int dim = cell_type->dim();
  const int num_verts_per_entity = cell_type->num_vertices();

  // Read MVC topology
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  assert(topology_data_node);
  boost::filesystem::path xdmf_filename(_filename);
  const boost::filesystem::path parent_path = xdmf_filename.parent_path();
  std::vector<std::int32_t> topology_data = get_dataset<std::int32_t>(
      _mpi_comm.comm(), topology_data_node, parent_path);

  // Read values associated with each mesh::MeshEntity described by topology
  pugi::xml_node attribute_node = grid_node.child("Attribute");
  assert(attribute_node);
  pugi::xml_node attribute_data_node = attribute_node.child("DataItem");
  assert(attribute_data_node);
  std::vector<T> values_data
      = get_dataset<T>(_mpi_comm.comm(), attribute_data_node, parent_path);

  // Ensure the mesh dimension is initialised
  mesh->create_entities(dim);
  const std::size_t global_vertex_range = mesh->num_entities_global(0);
  const std::int32_t num_processes = _mpi_comm.size();

  // Send entities to processes based on the lowest vertex index
  std::vector<std::vector<std::int32_t>> send_entities(num_processes);
  std::vector<std::vector<std::int32_t>> recv_entities(num_processes);

  std::vector<std::int32_t> v(num_verts_per_entity);
  for (auto& m : mesh::MeshRange<mesh::MeshEntity>(*mesh, dim))
  {
    if (dim == 0)
      v[0] = m.global_index();
    else
    {
      v.clear();
      for (auto& vtx : mesh::EntityRange<mesh::Vertex>(m))
        v.push_back(vtx.global_index());
      std::sort(v.begin(), v.end());
    }

    std::size_t dest
        = MPI::index_owner(_mpi_comm.comm(), v[0], global_vertex_range);
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

  std::int32_t i = 0;
  for (auto it = topology_data.begin(); it != topology_data.end();
       it += num_verts_per_entity)
  {
    std::partial_sort_copy(it, it + num_verts_per_entity, v.begin(), v.end());
    std::size_t dest
        = MPI::index_owner(_mpi_comm.comm(), v[0], global_vertex_range);
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
void XDMFFile::write(const std::vector<geometry::Point>& points)
{
  // Check that encoding
  if (_encoding == Encoding::ASCII and _mpi_comm.size() != 1)
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
    h5_file = std::make_unique<HDF5File>(_mpi_comm.comm(),
                                         get_hdf5_filename(_filename), "w");
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

  add_points(_mpi_comm.comm(), xdmf_node, h5_id, points);

  // Save XML file (on process 0 only)
  if (_mpi_comm.rank() == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::add_points(MPI_Comm comm, pugi::xml_node& xdmf_node, hid_t h5_id,
                          const std::vector<geometry::Point>& points)
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
  const std::int64_t nglobal = MPI::sum(comm, n);
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
  add_data_item(comm, geometry_node, h5_id, "/Points/coordinates", x, shape);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const std::vector<geometry::Point>& points,
                     const std::vector<double>& values)
{
  // Write clouds of points to XDMF/HDF5 with values
  assert(points.size() == values.size());

  // Check that encoding is supported
  if (_encoding == Encoding::ASCII and _mpi_comm.size() != 1)
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
    h5_file = std::make_unique<HDF5File>(_mpi_comm.comm(),
                                         get_hdf5_filename(_filename), "w");
    assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }

  // Add XDMF node and version attribute
  _xml_doc->append_child(pugi::node_doctype)
      .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
  pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
  assert(xdmf_node);

  add_points(_mpi_comm.comm(), xdmf_node, h5_id, points);

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
  add_data_item(_mpi_comm.comm(), attribute_node, h5_id, "/Points/values",
                values, {num_values, 1});

  // Save XML file (on process 0 only)
  if (_mpi_comm.rank() == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
mesh::MeshFunction<bool>
XDMFFile::read_mf_bool(std::shared_ptr<const mesh::Mesh> mesh,
                       std::string name) const
{
  // Read mesh function
  mesh::MeshFunction<std::size_t> mf_int
      = read_mesh_function<std::size_t>(mesh, name);

  // Convert to bool
  mesh::MeshFunction<bool> mf(mesh, mf_int.dim(), 0);
  for (auto& cell : mesh::MeshRange<mesh::MeshEntity>(*mesh, mf.dim()))
    mf[cell.index()] = (mf_int[cell.index()] == 1);

  return mf;
}
//----------------------------------------------------------------------------
mesh::MeshFunction<int>
XDMFFile::read_mf_int(std::shared_ptr<const mesh::Mesh> mesh,
                      std::string name) const
{
  return read_mesh_function<int>(mesh, name);
}
//----------------------------------------------------------------------------
mesh::MeshFunction<std::size_t>
XDMFFile::read_mf_size_t(std::shared_ptr<const mesh::Mesh> mesh,
                         std::string name) const
{
  return read_mesh_function<std::size_t>(mesh, name);
}
//----------------------------------------------------------------------------
mesh::MeshFunction<double>
XDMFFile::read_mf_double(std::shared_ptr<const mesh::Mesh> mesh,
                         std::string name) const
{
  return read_mesh_function<double>(mesh, name);
}
//----------------------------------------------------------------------------
void XDMFFile::add_mesh(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
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
  const std::int64_t num_global_cells = mesh.num_entities_global(tdim);
  if (num_global_cells < 1e9)
    add_topology_data<std::int32_t>(comm, grid_node, h5_id, path_prefix, mesh,
                                    tdim);
  else
    add_topology_data<std::int64_t>(comm, grid_node, h5_id, path_prefix, mesh,
                                    tdim);

  // Add geometry node and attributes (including writing data)
  add_geometry_data(comm, grid_node, h5_id, path_prefix, mesh);
}
//----------------------------------------------------------------------------
void XDMFFile::add_function(MPI_Comm mpi_comm, pugi::xml_node& xml_node,
                            hid_t h5_id, std::string h5_path,
                            const function::Function& u,
                            std::string function_name, const mesh::Mesh& mesh,
                            const std::string component)
{
  LOG(INFO) << "Adding function to node \"" << xml_node.path('/') << "\"";

  std::string element_family = u.function_space()->element()->family();
  const std::size_t element_degree = u.function_space()->element()->degree();
  const CellType element_cell_type
      = u.function_space()->element()->cell_shape();

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

  assert(u.function_space()->dofmap());
  const fem::GenericDofMap& dofmap = *u.function_space()->dofmap();

  const std::size_t tdim = mesh.topology().dim();
  std::vector<PetscInt> cell_dofs;
  std::vector<std::size_t> x_cell_dofs;
  const std::size_t n_cells = mesh.topology().ghost_offset(tdim);
  x_cell_dofs.reserve(n_cells);

  Eigen::Array<std::size_t, Eigen::Dynamic, 1> local_to_global_map
      = dofmap.tabulate_local_to_global_dofs();

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
  add_data_item(mpi_comm, fe_attribute_node, h5_id, h5_path + "/cell_dofs",
                cell_dofs, {num_cell_dofs_global, 1}, "UInt");

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

  add_data_item(mpi_comm, fe_attribute_node, h5_id, h5_path + "/vector",
                component_data_values, {(std::int64_t)u_vector.size(), 1},
                "Float");
#else
  add_data_item(mpi_comm, fe_attribute_node, h5_id, h5_path + "/vector",
                local_data, {(std::int64_t)u_vector.size(), 1}, "Float");
#endif

  if (MPI::rank(mpi_comm) == MPI::size(mpi_comm) - 1)
    x_cell_dofs.push_back(num_cell_dofs_global);

  const std::int64_t num_x_cell_dofs_global
      = mesh.num_entities_global(tdim) + 1;

  // Write number of dofs per cell
  add_data_item(mpi_comm, fe_attribute_node, h5_id, h5_path + "/x_cell_dofs",
                x_cell_dofs, {num_x_cell_dofs_global, 1}, "UInt");

  // Save cell ordering - copy to local vector and cut off ghosts
  std::vector<std::size_t> cells(mesh.topology().global_indices(tdim).begin(),
                                 mesh.topology().global_indices(tdim).begin()
                                     + n_cells);

  const std::int64_t num_cells_global = mesh.num_entities_global(tdim);

  add_data_item(mpi_comm, fe_attribute_node, h5_id, h5_path + "/cells", cells,
                {num_cells_global, 1}, "UInt");
}
//-----------------------------------------------------------------------------
mesh::Mesh XDMFFile::read_mesh(MPI_Comm comm,
                               const mesh::GhostMode ghost_mode) const
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
  const auto cell_type_str = get_cell_type(topology_node);

  const int degree = cell_type_str.second;
  if (degree == 2)
    LOG(WARNING) << "Caution: reading quadratic mesh";

  // Get toplogical dimensions
  std::unique_ptr<mesh::CellType> cell_type(
      mesh::CellType::create(cell_type_str.first));
  assert(cell_type);

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
  const std::vector<std::int64_t> gdims = get_dataset_shape(geometry_data_node);
  assert(gdims.size() == 2);
  assert(gdims[1] == gdim);

  // Geometry
  const auto geometry_data
      = get_dataset<double>(_mpi_comm.comm(), geometry_data_node, parent_path);
  const std::size_t num_local_points = geometry_data.size() / gdim;

  Eigen::Map<const EigenRowArrayXXd> points(geometry_data.data(),
                                            num_local_points, gdim);
  // Get topology dataset node
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  assert(topology_data_node);

  // Topology
  const std::vector<std::int64_t> tdims = get_dataset_shape(topology_data_node);
  const auto topology_data = get_dataset<std::int64_t>(
      _mpi_comm.comm(), topology_data_node, parent_path);
  const std::size_t npoint_per_cell = tdims[1];
  const std::size_t num_local_cells = topology_data.size() / npoint_per_cell;
  Eigen::Map<const EigenRowArrayXXi64> cells(topology_data.data(),
                                             num_local_cells, npoint_per_cell);

  // Set cell global indices by adding offset
  const std::int64_t cell_index_offset
      = MPI::global_offset(_mpi_comm.comm(), num_local_cells, true);
  std::vector<std::int64_t> global_cell_indices(num_local_cells);
  std::iota(global_cell_indices.begin(), global_cell_indices.end(),
            cell_index_offset);

  return mesh::Partitioning::build_distributed_mesh(
      _mpi_comm.comm(), cell_type->cell_type(), points, cells,
      global_cell_indices, ghost_mode);
}
//----------------------------------------------------------------------------
function::Function
XDMFFile::read_checkpoint(std::shared_ptr<const function::FunctionSpace> V,
                          std::string func_name, std::int64_t counter) const
{
  if (!name_same_on_all_procs(func_name))
  {
    throw std::runtime_error("Function name must be the same on all processes "
                             "when reading XDMF file.");
  }

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
  const fem::GenericDofMap& dofmap = *V->dofmap();

  // Read cell ordering
  std::vector<std::size_t> cells
      = get_dataset<std::size_t>(_mpi_comm.comm(), cells_dataitem, parent_path);

  const std::vector<std::int64_t> x_cell_dofs_shape
      = get_dataset_shape(cells_dataitem);

  // Divide cells equally between processes
  std::array<std::int64_t, 2> cell_range
      = MPI::local_range(_mpi_comm.comm(), x_cell_dofs_shape[0]);

  // Read number of dofs per cell
  std::vector<std::int64_t> x_cell_dofs = get_dataset<std::int64_t>(
      _mpi_comm.comm(), x_cell_dofs_dataitem, parent_path,
      {{cell_range[0], cell_range[1] + 1}});

  // Read cell dofmaps
  std::vector<PetscInt> cell_dofs
      = get_dataset<PetscInt>(_mpi_comm.comm(), cell_dofs_dataitem, parent_path,
                              {{x_cell_dofs.front(), x_cell_dofs.back()}});

  const std::vector<std::int64_t> vector_shape
      = get_dataset_shape(vector_dataitem);
  const std::size_t num_global_dofs = vector_shape[0];

  // Divide vector between processes
  const std::array<std::int64_t, 2> input_vector_range
      = MPI::local_range(_mpi_comm.comm(), num_global_dofs);

#ifdef PETSC_USE_COMPLEX
  // Read real component of function vector
  std::vector<double> real_vector = get_dataset<double>(
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
  std::vector<double> imag_vector = get_dataset<double>(
      _mpi_comm.comm(), imag_vector_dataitem, parent_path, input_vector_range);

  assert(real_vector.size() == imag_vector.size());

  // Compose complex function vector
  std::vector<PetscScalar> vector;
  vector.reserve(real_vector.size());
  std::transform(begin(real_vector), end(real_vector), begin(imag_vector),
                 std::back_inserter(vector),
                 [](double r, double i) { return r + i * PETSC_i; });
#else
  // Read function vector
  std::vector<double> vector = get_dataset<double>(
      _mpi_comm.comm(), vector_dataitem, parent_path, input_vector_range);
#endif

  function::Function u(V);
  HDF5Utility::set_local_vector_values(_mpi_comm.comm(), u.vector(), mesh,
                                       cells, cell_dofs, x_cell_dofs, vector,
                                       input_vector_range, dofmap);

  return u;
}
//----------------------------------------------------------------------------
template <typename T>
void XDMFFile::add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                 hid_t h5_id, const std::string path_prefix,
                                 const mesh::Mesh& mesh, int cell_dim)
{
  // Get number of cells (global) and vertices per cell from mesh
  const std::int64_t num_cells = mesh.topology().size_global(cell_dim);
  int num_nodes_per_cell = mesh.type().num_vertices(cell_dim);

  // Get VTK string for cell type and degree (linear or quadratic)
  const std::size_t degree = mesh.degree();
  const std::string vtk_cell_str
      = vtk_cell_type_str(mesh.type().entity_type(cell_dim), degree);

  pugi::xml_node topology_node = xml_node.append_child("Topology");
  assert(topology_node);
  topology_node.append_attribute("NumberOfElements")
      = std::to_string(num_cells).c_str();
  topology_node.append_attribute("TopologyType") = vtk_cell_str.c_str();

  // Compute packed topology data
  std::vector<T> topology_data;

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
        = mesh.coordinate_dofs().entity_points(tdim);

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
    topology_data = compute_topology_data<T>(mesh, cell_dim);

  topology_node.append_attribute("NodesPerElement") = num_nodes_per_cell;

  // Add topology DataItem node
  const std::string group_name = path_prefix + "/" + "mesh";
  const std::string h5_path = group_name + "/topology";
  const std::vector<std::int64_t> shape = {num_cells, num_nodes_per_cell};
  const std::string number_type = "UInt";

  add_data_item(comm, topology_node, h5_id, h5_path, topology_data, shape,
                number_type);
}
//-----------------------------------------------------------------------------
void XDMFFile::add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node,
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
  std::vector<double> x(_x.data(), _x.data() + _x.size());

  // XDMF does not support 1D, so handle as special case
  if (gdim == 1)
  {
    // Pad the coordinates with zeros for a dummy Y
    gdim = 2;
    std::vector<double> _x(2 * x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i)
      _x[2 * i] = x[i];
    std::swap(x, _x);
  }

  // Add geometry DataItem node
  const std::string group_name = path_prefix + "/" + "mesh";
  const std::string h5_path = group_name + "/geometry";
  const std::vector<std::int64_t> shape = {num_points, gdim};

  add_data_item(comm, geometry_node, h5_id, h5_path, x, shape);
}
//-----------------------------------------------------------------------------
template <typename T>
void XDMFFile::add_data_item(MPI_Comm comm, pugi::xml_node& xml_node,
                             hid_t h5_id, const std::string h5_path, const T& x,
                             const std::vector<std::int64_t> shape,
                             const std::string number_type)
{
  DLOG(INFO) << "Adding data item to node " << xml_node.path();

  // Add DataItem node
  assert(xml_node);
  pugi::xml_node data_item_node = xml_node.append_child("DataItem");
  assert(data_item_node);

  // Add dimensions attribute
  data_item_node.append_attribute("Dimensions")
      = common::container_to_string(shape, " ", 16).c_str();

  // Set type for topology data (needed by XDMF to prevent default to float)
  if (!number_type.empty())
    data_item_node.append_attribute("NumberType") = number_type.c_str();

  // Add format attribute
  if (h5_id < 0)
  {
    data_item_node.append_attribute("Format") = "XML";
    assert(shape.size() == 2);
    data_item_node.append_child(pugi::node_pcdata)
        .set_value(common::container_to_string(x, " ", 16, shape[1]).c_str());
  }
  else
  {
    data_item_node.append_attribute("Format") = "HDF";

    // Get name of HDF5 file
    const std::string hdf5_filename = HDF5Interface::get_filename(h5_id);
    const boost::filesystem::path p(hdf5_filename);

    // Add HDF5 filename and HDF5 internal path to XML file
    const std::string xdmf_path = p.filename().string() + ":" + h5_path;
    data_item_node.append_child(pugi::node_pcdata).set_value(xdmf_path.c_str());

    // Compute total number of items and check for consistency with shape
    assert(!shape.empty());
    std::int64_t num_items_total = 1;
    for (auto n : shape)
      num_items_total *= n;

    assert(num_items_total == (std::int64_t)MPI::sum(comm, x.size()));

    // Compute data offset and range of values
    std::int64_t local_shape0 = x.size();
    for (std::size_t i = 1; i < shape.size(); ++i)
    {
      assert(local_shape0 % shape[i] == 0);
      local_shape0 /= shape[i];
    }
    const std::int64_t offset = MPI::global_offset(comm, local_shape0, true);
    const std::array<std::int64_t, 2> local_range
        = {{offset, offset + local_shape0}};

    const bool use_mpi_io = (MPI::size(comm) > 1);
    HDF5Interface::write_dataset(h5_id, h5_path, x.data(), local_range, shape,
                                 use_mpi_io, false);

    // Add partitioning attribute to dataset
    std::vector<std::size_t> partitions;
    std::vector<std::size_t> offset_tmp(1, offset);
    MPI::gather(comm, offset_tmp, partitions);
    MPI::broadcast(comm, partitions);
    HDF5Interface::add_attribute(h5_id, h5_path, "partition", partitions);
  }
}
//----------------------------------------------------------------------------
std::set<std::uint32_t>
XDMFFile::compute_nonlocal_entities(const mesh::Mesh& mesh, int cell_dim)
{
  // If not already numbered, number entities of
  // order cell_dim so we can get shared_entities
  mesh::DistributedMeshTools::number_entities(mesh, cell_dim);

  const int mpi_rank = MPI::rank(mesh.mpi_comm());
  const std::map<std::int32_t, std::set<std::int32_t>>& shared_entities
      = mesh.topology().shared_entities(cell_dim);

  std::set<std::uint32_t> non_local_entities;

  const int tdim = mesh.topology().dim();
  bool ghosted
      = (mesh.topology().size(tdim) > mesh.topology().ghost_offset(tdim));

  if (!ghosted)
  {
    // No ghost cells - exclude shared entities
    // which are on lower rank processes
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
template <typename T>
std::vector<T> XDMFFile::compute_topology_data(const mesh::Mesh& mesh,
                                               int cell_dim)
{
  // Create vector to store topology data
  const int num_vertices_per_cell = mesh.type().num_vertices(cell_dim);
  std::vector<T> topology_data;
  topology_data.reserve(mesh.num_entities(cell_dim) * (num_vertices_per_cell));

  // Get mesh communicator
  MPI_Comm comm = mesh.mpi_comm();

  const std::vector<std::int8_t> perm = mesh.type().vtk_mapping();
  const int tdim = mesh.topology().dim();
  if (MPI::size(comm) == 1 or cell_dim == tdim)
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
        = compute_nonlocal_entities(mesh, cell_dim);

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
template <typename T>
std::vector<T>
XDMFFile::compute_value_data(const mesh::MeshFunction<T>& meshfunction)
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
              meshfunction.values() + meshfunction.size(), value_data.begin());
  }
  else
  {
    std::set<std::uint32_t> non_local_entities
        = compute_nonlocal_entities(*mesh, cell_dim);

    for (auto& e : mesh::MeshRange<mesh::MeshEntity>(*mesh, cell_dim))
    {
      if (non_local_entities.find(e.index()) == non_local_entities.end())
        value_data.push_back(meshfunction[e]);
    }
  }

  return value_data;
}
//----------------------------------------------------------------------------
std::pair<std::string, int>
XDMFFile::get_cell_type(const pugi::xml_node& topology_node)
{
  assert(topology_node);
  pugi::xml_attribute type_attr = topology_node.attribute("TopologyType");
  assert(type_attr);

  const std::map<std::string, std::pair<std::string, int>> xdmf_to_dolfin
      = {{"polyvertex", {"point", 1}},
         {"polyline", {"interval", 1}},
         {"edge_3", {"interval", 2}},
         {"triangle", {"triangle", 1}},
         {"triangle_6", {"triangle", 2}},
         {"tetrahedron", {"tetrahedron", 1}},
         {"tetrahedron_10", {"tetrahedron", 2}},
         {"quadrilateral", {"quadrilateral", 1}}};

  // Convert XDMF cell type string to DOLFIN cell type string
  std::string cell_type = type_attr.as_string();
  boost::algorithm::to_lower(cell_type);
  auto it = xdmf_to_dolfin.find(cell_type);
  if (it == xdmf_to_dolfin.end())
  {
    throw std::runtime_error("Cannot recognise cell type. Unknown value: "
                             + cell_type);
  }
  return it->second;
}
//----------------------------------------------------------------------------
std::vector<std::int64_t>
XDMFFile::get_dataset_shape(const pugi::xml_node& dataset_node)
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
std::int64_t XDMFFile::get_num_cells(const pugi::xml_node& topology_node)
{
  assert(topology_node);

  // Get number of cells from topology
  std::int64_t num_cells_topolgy = -1;
  pugi::xml_attribute num_cells_attr
      = topology_node.attribute("NumberOfElements");
  if (num_cells_attr)
    num_cells_topolgy = num_cells_attr.as_llong();

  // Get number of cells from topology dataset
  pugi::xml_node topology_dataset_node = topology_node.child("DataItem");
  assert(topology_dataset_node);
  const std::vector<std::int64_t> tdims
      = get_dataset_shape(topology_dataset_node);

  // Check that number of cells can be determined
  if (tdims.size() != 2 and num_cells_topolgy == -1)
  {
    throw std::runtime_error("Cannot determine number of cells in XMDF mesh");
  }

  // Check for consistency if number of cells appears in both the topology
  // and DataItem nodes
  if (num_cells_topolgy != -1 and tdims.size() == 2)
  {
    if (num_cells_topolgy != tdims[0])
    {
      throw std::runtime_error("Cannot determine number of cells in XMDF mesh");
    }
  }

  return std::max(num_cells_topolgy, tdims[0]);
}
//----------------------------------------------------------------------------
template <typename T>
std::vector<T> XDMFFile::get_dataset(MPI_Comm comm,
                                     const pugi::xml_node& dataset_node,
                                     const boost::filesystem::path& parent_path,
                                     std::array<std::int64_t, 2> range)
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
  const std::vector<std::int64_t> shape_xml = get_dataset_shape(dataset_node);

  const std::string format = format_attr.as_string();
  std::vector<T> data_vector;
  // Only read ASCII on process 0
  if (format == "XML")
  {
    if (MPI::rank(comm) == 0)
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
        range = MPI::local_range(comm, shape_hdf5[0]);
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
        range = MPI::local_range(comm, shape_xml[0]);
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
  {
    throw std::runtime_error("Storage format \"" + format + "\" is unknown");
  }

  // Get dimensions for consistency (if available in DataItem node)
  if (shape_xml.empty())
  {
    std::int64_t size = 1;
    for (auto dim : shape_xml)
      size *= dim;

    if (size != (std::int64_t)MPI::sum(comm, data_vector.size()))
    {
      throw std::runtime_error(
          "Data sizes in attribute and size of data read are inconsistent");
    }
  }

  return data_vector;
}
//----------------------------------------------------------------------------
std::array<std::string, 2>
XDMFFile::get_hdf5_paths(const pugi::xml_node& dataitem_node)
{
  // Check that node is a DataItem node
  assert(dataitem_node);
  const std::string dataitem_str = "DataItem";
  if (dataitem_node.name() != dataitem_str)
  {
    throw std::runtime_error("Node name is \""
                             + std::string(dataitem_node.name())
                             + "\", expecting \"DataItem\"");
  }

  // Check that format is HDF
  pugi::xml_attribute format_attr = dataitem_node.attribute("Format");
  assert(format_attr);
  const std::string format = format_attr.as_string();
  if (format.compare("HDF") != 0)
  {
    throw std::runtime_error("DataItem format \"" + format
                             + "\" is not \"HDF\"");
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
template <typename T>
mesh::MeshFunction<T>
XDMFFile::read_mesh_function(std::shared_ptr<const mesh::Mesh> mesh,
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

  // Check if a TimeSeries (old format), in which case the Grid will be down
  // one level
  if (!grid_node)
  {
    pugi::xml_node grid_node1 = domain_node.child("Grid");
    if (grid_node1)
    {
      for (pugi::xml_node node : grid_node1.children("Grid"))
      {
        pugi::xml_node attr_node = node.child("Attribute");
        if (attr_node
            and (name == "" or name == attr_node.attribute("Name").as_string()))
        {
          grid_node = node;
          value_node = attr_node;
          break;
        }
      }
    }
  }

  // Still can't find it
  if (!grid_node)
  {
    throw std::runtime_error("Mesh Grid with data Attribute not found in XDMF");
  }

  // Get topology node
  pugi::xml_node topology_node = grid_node.child("Topology");
  assert(topology_node);

  // Get cell type and topology of mesh::MeshFunction (may be different from
  // mesh::Mesh)
  const auto cell_type_str = get_cell_type(topology_node);
  assert(cell_type_str.second == 1);
  std::unique_ptr<mesh::CellType> cell_type(
      mesh::CellType::create(cell_type_str.first));
  assert(cell_type);
  const std::uint32_t num_vertices_per_cell = cell_type->num_entities(0);
  const std::uint32_t dim = cell_type->dim();

  const std::int64_t num_entities_global = get_num_cells(topology_node);

  // Ensure num_entities_global(cell_dim) is set and check dataset matches
  mesh::DistributedMeshTools::number_entities(*mesh, dim);
  assert(mesh->num_entities_global(dim) == num_entities_global);

  boost::filesystem::path xdmf_filename(_filename);
  const boost::filesystem::path parent_path = xdmf_filename.parent_path();

  // Get topology dataset
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  assert(topology_data_node);
  const auto topology_data = get_dataset<std::int64_t>(
      mesh->mpi_comm(), topology_data_node, parent_path);
  assert(topology_data.size() % num_vertices_per_cell == 0);

  // Get value dataset
  pugi::xml_node value_data_node = value_node.child("DataItem");
  assert(value_data_node);
  std::vector<T> value_data
      = get_dataset<T>(_mpi_comm.comm(), value_data_node, parent_path);

  // Create mesh function and scatter/gather data across processes
  mesh::MeshFunction<T> mf(mesh, dim, 0);
  remap_meshfunction_data(mf, topology_data, value_data);

  return mf;
}
//-----------------------------------------------------------------------------
template <typename T>
void XDMFFile::remap_meshfunction_data(
    mesh::MeshFunction<T>& meshfunction,
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
        = MPI::index_owner(comm, cell_topology.front(), max_vertex);

    send_topology[destination_process].insert(
        send_topology[destination_process].end(), cell_topology.begin(),
        cell_topology.end());
    send_values[destination_process].push_back(value_data[i]);
  }

  std::vector<std::vector<std::int64_t>> receive_topology(num_processes);
  std::vector<std::vector<T>> receive_values(num_processes);
  MPI::all_to_all(comm, send_topology, receive_topology);
  MPI::all_to_all(comm, send_values, receive_values);

  // Generate requests for data from remote processes, based on the
  // first vertex of the mesh::MeshEntities which belong on this process
  // Send our process number, and our local index, so it can come back
  // directly to the right place
  std::vector<std::vector<std::int64_t>> send_requests(num_processes);
  const std::size_t rank = MPI::rank(comm);
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
        = MPI::index_owner(comm, cell_topology.front(), max_vertex);
    // Map to this process and local index by appending to send data
    cell_topology.push_back(cell.index());
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

  MPI::all_to_all(comm, send_topology, receive_topology);
  MPI::all_to_all(comm, send_values, receive_values);

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
std::string XDMFFile::get_hdf5_filename(std::string xdmf_filename)
{
  boost::filesystem::path p(xdmf_filename);
  p.replace_extension(".h5");
  if (p.string() == xdmf_filename)
  {
    throw std::runtime_error(
        "Cannot deduce name of HDF5 file from XDMF filename. "
        "Filename clash. Check XDMF filename");
  }

  return p.string();
}
//----------------------------------------------------------------------------
template <typename T>
void XDMFFile::write_mesh_function(const mesh::MeshFunction<T>& meshfunction)
{
  // Check that encoding
  if (_encoding == Encoding::ASCII and _mpi_comm.size() != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }

  if (meshfunction.size() == 0)
  {
    throw std::runtime_error("No values in MeshFunction");
  }

  // Get mesh
  assert(meshfunction.mesh());
  std::shared_ptr<const mesh::Mesh> mesh = meshfunction.mesh();

  // Check if _xml_doc already has data. If not, create an outer structure
  // If it already has data, then we may append to it.

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
        mesh->mpi_comm(), get_hdf5_filename(_filename), hdf_filemode);
    assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }

  const std::string mf_name = "/MeshFunction/" + std::to_string(_counter);

  // If adding a mesh::MeshFunction of topology dimension dim() to an existing
  // mesh::Mesh,
  // do not rewrite mesh::Mesh
  // FIXME: do some checks on the existing mesh::Mesh to make sure it is the
  // same
  // as the meshfunction's mesh.
  pugi::xml_node grid_node = domain_node.child("Grid");
  const std::size_t cell_dim = meshfunction.dim();
  const std::size_t tdim = mesh->topology().dim();
  const bool grid_empty = grid_node.empty();

  // Check existing mesh::Mesh for compatibility.
  if (!grid_empty)
  {
    pugi::xml_node topology_node = grid_node.child("Topology");
    assert(topology_node);
    auto cell_type_str = get_cell_type(topology_node);
    if (mesh::CellType::type2string(mesh->type().cell_type())
        != cell_type_str.first)
    {
      throw std::runtime_error(
          "Incompatible Mesh type. Try writing the Mesh to XDMF first");
    }
  }

  if (grid_empty or cell_dim != tdim)
  {
    // Make new grid node
    grid_node = domain_node.append_child("Grid");
    assert(grid_node);
    grid_node.append_attribute("Name") = "mesh";
    grid_node.append_attribute("GridType") = "Uniform";

    // Make sure entities are numbered - only needed for  mesh::Edge in 3D in
    // parallel
    // FIXME: remove this once  mesh::Edge in 3D in parallel works properly
    mesh::DistributedMeshTools::number_entities(*mesh, cell_dim);

    const std::int64_t num_global_cells = mesh->num_entities_global(cell_dim);
    if (num_global_cells < 1e9)
      add_topology_data<std::int32_t>(_mpi_comm.comm(), grid_node, h5_id,
                                      mf_name, *mesh, cell_dim);
    else
      add_topology_data<std::int64_t>(_mpi_comm.comm(), grid_node, h5_id,
                                      mf_name, *mesh, cell_dim);

    // Add geometry node if none already, else link back to first existing
    // mesh::Mesh
    if (grid_empty)
      add_geometry_data(_mpi_comm.comm(), grid_node, h5_id, mf_name, *mesh);
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
  attribute_node.append_attribute("Name") = meshfunction.name().c_str();
  attribute_node.append_attribute("AttributeType") = "Scalar";
  attribute_node.append_attribute("Center") = "Cell";

  const std::int64_t num_values = mesh->num_entities_global(cell_dim);
  // Add attribute DataItem node and write data

  // Copy values to vector, removing duplicates
  std::vector<T> values = compute_value_data(meshfunction);

  add_data_item(_mpi_comm.comm(), attribute_node, h5_id, mf_name + "/values",
                values, {num_values, 1});

  // Save XML file (on process 0 only)
  if (_mpi_comm.rank() == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");

  // Increment the counter, so we can save multiple mesh::MeshFunctions in one
  // file
  ++_counter;
}
//-----------------------------------------------------------------------------
std::vector<PetscScalar>
XDMFFile::get_cell_data_values(const function::Function& u)
{
  assert(u.function_space()->dofmap());

  const auto mesh = u.function_space()->mesh();
  const std::size_t value_size = u.value_size();
  const std::size_t value_rank = u.value_rank();

  // Allocate memory for function values at cell centres
  const std::size_t tdim = mesh->topology().dim();
  const std::size_t num_local_cells = mesh->topology().ghost_offset(tdim);
  const std::size_t local_size = num_local_cells * value_size;

  // Build lists of dofs and create map
  std::vector<PetscInt> dof_set;
  dof_set.reserve(local_size);
  const auto dofmap = u.function_space()->dofmap();
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*mesh))
  {
    // Tabulate dofs
    auto dofs = dofmap->cell_dofs(cell.index());
    const std::size_t ndofs = dofmap->num_element_dofs(cell.index());
    assert(ndofs == value_size);
    for (std::size_t i = 0; i < ndofs; ++i)
      dof_set.push_back(dofs[i]);
  }

  // Get  values
  std::vector<PetscScalar> data_values(dof_set.size());
  {
    la::VecReadWrapper u_wrapper(u.vector().vec());
    Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x
        = u_wrapper.x;
    for (std::size_t i = 0; i < dof_set.size(); ++i)
      data_values[i] = x[dof_set[i]];
  }

  if (value_rank == 1 && value_size == 2)
  {
    // Pad out data for 2D vector to 3D
    data_values.resize(3 * num_local_cells);
    for (int j = (num_local_cells - 1); j >= 0; --j)
    {
      PetscScalar nd[3] = {data_values[j * 2], data_values[j * 2 + 1], 0};
      std::copy(nd, nd + 3, &data_values[j * 3]);
    }
  }
  else if (value_rank == 2 && value_size == 4)
  {
    data_values.resize(9 * num_local_cells);
    for (int j = (num_local_cells - 1); j >= 0; --j)
    {
      PetscScalar nd[9] = {data_values[j * 4],
                           data_values[j * 4 + 1],
                           0,
                           data_values[j * 4 + 2],
                           data_values[j * 4 + 3],
                           0,
                           0,
                           0,
                           0};
      std::copy(nd, nd + 9, &data_values[j * 9]);
    }
  }
  return data_values;
}
//-----------------------------------------------------------------------------
std::int64_t XDMFFile::get_padded_width(const function::Function& u)
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
bool XDMFFile::has_cell_centred_data(const function::Function& u)
{
  // Test for cell-centred data
  std::size_t cell_based_dim = 1;
  for (std::size_t i = 0; i < u.value_rank(); i++)
    cell_based_dim *= u.function_space()->mesh()->topology().dim();
  return (u.function_space()->dofmap()->max_element_dofs() == cell_based_dim);
}
//-----------------------------------------------------------------------------
std::vector<PetscScalar>
XDMFFile::get_point_data_values(const function::Function& u)
{
  auto mesh = u.function_space()->mesh();
  assert(mesh);
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      data_values = u.compute_point_values(*mesh);
  std::int64_t width = get_padded_width(u);

  // FIXME: Unpick the below code for the new layout of data from
  //        GenericFunction::compute_vertex_values
  const std::size_t num_local_points = mesh->geometry().num_points();
  std::vector<PetscScalar> _data_values(width * num_local_points, 0.0);

  const std::size_t value_rank = u.value_rank();
  if (value_rank > 0)
  {
    // Transpose vector/tensor data arrays
    const std::size_t value_size = u.value_size();
    for (std::size_t i = 0; i < num_local_points; i++)
    {
      for (std::size_t j = 0; j < value_size; j++)
      {
        std::size_t tensor_2d_offset
            = (j > 1 && value_rank == 2 && value_size == 4) ? 1 : 0;
        _data_values[i * width + j + tensor_2d_offset] = data_values(i, j);
      }
    }
  }
  else
  {
    _data_values = std::vector<PetscScalar>(
        data_values.data(),
        data_values.data() + data_values.rows() * data_values.cols());
  }

  // Reorder values by global point indices
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>>
      in_vals(_data_values.data(), _data_values.size() / width, width);

  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      vals = mesh::DistributedMeshTools::reorder_by_global_indices(
          mesh->mpi_comm(), in_vals, mesh->geometry().global_indices());

  _data_values
      = std::vector<PetscScalar>(vals.data(), vals.data() + vals.size());

  return _data_values;
}
//-----------------------------------------------------------------------------
std::vector<PetscScalar>
XDMFFile::get_p2_data_values(const function::Function& u)
{
  const auto mesh = u.function_space()->mesh();

  const std::size_t value_size = u.value_size();
  const std::size_t value_rank = u.value_rank();
  const std::size_t num_local_points
      = mesh->num_entities(0) + mesh->num_entities(1);
  const std::size_t width = get_padded_width(u);
  std::vector<PetscScalar> data_values(width * num_local_points);
  std::vector<PetscInt> data_dofs(data_values.size(), 0);

  assert(u.function_space()->dofmap());
  const auto dofmap = u.function_space()->dofmap();

  // function::Function can be P1 or P2
  if (dofmap->num_entity_dofs(1) == 0)
  {
    // P1
    for (auto& cell : mesh::MeshRange<mesh::Cell>(*mesh))
    {
      auto dofs = dofmap->cell_dofs(cell.index());
      std::size_t c = 0;
      for (std::size_t i = 0; i != value_size; ++i)
      {
        for (auto& v : mesh::EntityRange<mesh::Vertex>(cell))
        {
          const std::size_t v0 = v.index() * width;
          data_dofs[v0 + i] = dofs[c];
          ++c;
        }
      }
    }

    // Get the values at the vertex points
    {
      la::VecReadWrapper u_wrapper(u.vector().vec());
      Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x
          = u_wrapper.x;
      for (std::size_t i = 0; i < data_dofs.size(); ++i)
        data_values[i] = x[data_dofs[i]];
    }

    // Get midpoint values for  mesh::Edge points
    for (auto& e : mesh::MeshRange<mesh::Edge>(*mesh))
    {
      const std::size_t v0 = e.entities(0)[0];
      const std::size_t v1 = e.entities(0)[1];
      const std::size_t e0 = (e.index() + mesh->num_entities(0)) * width;
      for (std::size_t i = 0; i != value_size; ++i)
        data_values[e0 + i] = (data_values[v0 + i] + data_values[v1 + i]) / 2.0;
    }
  }
  else if (dofmap->num_entity_dofs(0) == dofmap->num_entity_dofs(1))
  {
    // P2
    // Go over all cells inserting values
    // FIXME: a lot of duplication here
    for (auto& cell : mesh::MeshRange<mesh::Cell>(*mesh))
    {
      auto dofs = dofmap->cell_dofs(cell.index());
      std::size_t c = 0;
      for (std::size_t i = 0; i != value_size; ++i)
      {
        for (auto& v : mesh::EntityRange<mesh::Vertex>(cell))
        {
          const std::size_t v0 = v.index() * width;
          data_dofs[v0 + i] = dofs[c];
          ++c;
        }
        for (auto& e : mesh::EntityRange<mesh::Edge>(cell))
        {
          const std::size_t e0 = (e.index() + mesh->num_entities(0)) * width;
          data_dofs[e0 + i] = dofs[c];
          ++c;
        }
      }
    }

    la::VecReadWrapper u_wrapper(u.vector().vec());
    Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x
        = u_wrapper.x;
    for (std::size_t i = 0; i < data_dofs.size(); ++i)
      data_values[i] = x[data_dofs[i]];
  }
  else
  {
    throw std::runtime_error(
        "Cannotget point values for function::Function. Function appears not "
        "to be defined on a P1 or P2 type function::FunctionSpace");
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
bool XDMFFile::name_same_on_all_procs(std::string name) const
{
  std::vector<std::string> names_received;
  MPI::all_gather(_mpi_comm.comm(), name, names_received);

  std::set<std::string> name_set(names_received.begin(), names_received.end());
  return (name_set.size() == 1);
}
//-----------------------------------------------------------------------------
std::string XDMFFile::vtk_cell_type_str(mesh::CellType::Type cell_type,
                                        int order)
{
  // FIXME: Move to CellType?
  switch (cell_type)
  {
  case mesh::CellType::Type::point:
    switch (order)
    {
    case 1:
      return "PolyVertex";
    }
  case mesh::CellType::Type::interval:
    switch (order)
    {
    case 1:
      return "PolyLine";
    case 2:
      return "Edge_3";
    }
  case mesh::CellType::Type::triangle:
    switch (order)
    {
    case 1:
      return "Triangle";
    case 2:
      return "Triangle_6";
    }
  case mesh::CellType::Type::quadrilateral:
    switch (order)
    {
    case 1:
      return "Quadrilateral";
    case 2:
      return "Quadrilateral_8";
    }
  case mesh::CellType::Type::tetrahedron:
    switch (order)
    {
    case 1:
      return "Tetrahedron";
    case 2:
      return "Tetrahedron_10";
    }
  case mesh::CellType::Type::hexahedron:
    switch (order)
    {
    case 1:
      return "Hexahedron";
    case 2:
      return "Hexahedron_20";
    }
  default:
    throw std::runtime_error("Invalid combination of cell type and order");
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
