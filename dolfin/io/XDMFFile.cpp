// Copyright (C) 2012-2015 Chris N. Richardson and Garth N. Wells
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
#include "XDMFxml.h"
#include "xmlutils.h"

using namespace dolfin;

//----------------------------------------------------------------------------
XDMFFile::XDMFFile(MPI_Comm comm, const std::string filename)
  : _mpi_comm(comm), _hdf5_filemode(""), _current_mesh_name(""),
    _filename(filename), _counter(0), _xml(new XDMFxml(filename))
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

  // Whether to save multi-dataset files as time series, or flat
  parameters.add("time_series", true);
}
//----------------------------------------------------------------------------
XDMFFile::~XDMFFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void XDMFFile::write(const Mesh& mesh, Encoding encoding)
{
  write_new(mesh, encoding);
  return;

  check_encoding(encoding);

  // Output data name
  const std::string name = mesh.name();

  // Topological and geometric dimensions
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t cell_dim = mesh.topology().dim();

  // Make sure entities are numbered
  DistributedMeshTools::number_entities(mesh, cell_dim);

  // Get number of global cells and points
  const std::size_t num_global_cells = mesh.size_global(cell_dim);
  std::size_t num_total_points = 0;
  for (std::size_t i = 0; i <= mesh.topology().dim(); ++i)
  {
    num_total_points +=
      mesh.geometry().num_entity_coordinates(i)*mesh.size_global(i);
  }

  const std::string group_name = "/Mesh/" + name;

  // Write hdf5 file on all processes
  if (encoding == Encoding::HDF5)
  {
#ifdef HAS_HDF5
    // Write mesh to HDF5 file
    // The XML below will obliterate any existing XDMF file
    if (_hdf5_filemode != "w")
    {
      // Create HDF5 file (truncate)
      _hdf5_file.reset(new HDF5File(mesh.mpi_comm(),
                       get_hdf5_filename(_filename), "w"));
      _hdf5_filemode = "w";
    }
    _hdf5_file->write(mesh, cell_dim, group_name);
#else
    dolfin_error("XDMFile.cpp", "write Mesh", "Need HDF5 support");
#endif
  }

  // Write the XML meta description on process zero
  if (MPI::rank(mesh.mpi_comm()) == 0)
  {
    dolfin_assert(_xml);
    _xml->init_mesh(name);

    // Write the topology
    std::string topology_xml_value;

    if (encoding == Encoding::ASCII)
      topology_xml_value = generate_xdmf_ascii_mesh_topology_data(mesh);
    else if (encoding == Encoding::HDF5)
    {
      const boost::filesystem::path p(get_hdf5_filename(_filename));
      topology_xml_value = p.filename().string() + ":" + group_name
        + "/topology";
    }

    // Describe topological connectivity
    _xml->mesh_topology(mesh.type().cell_type(), mesh.geometry().degree(),
                       num_global_cells, topology_xml_value,
                       xdmf_format_str(encoding));

    // Write the geometry
    std::string geometry_xml_value;

    if (encoding == Encoding::ASCII)
      geometry_xml_value = generate_xdmf_ascii_mesh_geometry_data(mesh);
    else if (encoding == Encoding::HDF5)
    {
      const boost::filesystem::path p(get_hdf5_filename(_filename));
      geometry_xml_value = p.filename().string() + ":" + group_name
        + "/coordinates";
    }

    // Describe geometric coordinates
    _xml->mesh_geometry(num_total_points, gdim, geometry_xml_value,
                        xdmf_format_str(encoding));

    _xml->write();
  }
}
//----------------------------------------------------------------------------
void XDMFFile::write(const Function& u, Encoding encoding)
{
  write(u, (double) _counter, encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const Function& u, double time_step, Encoding encoding)
{
  // FIXME: This function is too long. Breal up.

  check_encoding(encoding);

  // Conditions for starting a new HDF5 file
#ifdef HAS_HDF5
  if (encoding == Encoding::HDF5)
  {
    std::string hdf5_filename = get_hdf5_filename(_filename);
    const int mf_interval = parameters["multi_file"];
    if ((mf_interval != 0 and _counter % mf_interval == 0)
        or _hdf5_filemode != "w" )
    {
      if (mf_interval != 0)
      {
        std::stringstream s;
        s << std::setw(6) << std::setfill('0') << _counter;
        hdf5_filename += "_" + s.str();
      }

      // Create new HDF5 file (truncate),
      // closing any open file from a previous timestep
      _hdf5_file.reset(new HDF5File(_mpi_comm, hdf5_filename, "w"));
      _hdf5_filemode = "w";
    }

    dolfin_assert(_hdf5_file);
  }
#endif

  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();
  const std::size_t degree = mesh.geometry().degree();

  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();

  // Geometric and topological dimension
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t tdim = mesh.topology().dim();

  // Get some Function and cell information
  const std::size_t value_rank = u.value_rank();
  const std::size_t value_size = u.value_size();

  // For 2D vectors and tensors, pad out values with zeros to make 3D
  // (XDMF does not support 2D data)
  std::size_t padded_value_size = value_size;
  if (value_rank > 0)
  {
    if (value_size == 2)
      padded_value_size = 3;
    if (value_size == 4)
      padded_value_size = 9;
  }

  // Test for cell-centred data
  std::size_t cell_based_dim = 1;
  for (std::size_t i = 0; i < value_rank; i++)
    cell_based_dim *= tdim;

  const bool vertex_data = !(dofmap.max_element_dofs() == cell_based_dim);

  // Get Function data at vertices/cell centres
  std::vector<double> data_values;
  if (vertex_data)
    get_point_data_values(data_values, padded_value_size, u);
  else
  {
    dolfin_assert(u.function_space()->dofmap());
    dolfin_assert(u.vector());

    // Allocate memory for function values at cell centres
    const std::size_t num_local_cells = mesh.topology().ghost_offset(tdim);
    const std::size_t size = num_local_cells*value_size;

    // Build lists of dofs and create map
    std::vector<dolfin::la_index> dof_set;
    std::vector<std::size_t> offset(size + 1);
    std::vector<std::size_t>::iterator cell_offset = offset.begin();
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Tabulate dofs
      const ArrayView<const dolfin::la_index> dofs
        = dofmap.cell_dofs(cell->index());
      for (std::size_t i = 0; i < dofmap.num_element_dofs(cell->index()); ++i)
        dof_set.push_back(dofs[i]);

      // Add local dimension to cell offset and increment
      *(cell_offset + 1) = *(cell_offset)
        + dofmap.num_element_dofs(cell->index());
      ++cell_offset;
    }

    // Get  values
    data_values.resize(dof_set.size());
    dolfin_assert(u.vector());
    u.vector()->get_local(data_values.data(), dof_set.size(), dof_set.data());

    cell_offset = offset.begin();
    std::vector<double> _data_values(padded_value_size*num_local_cells, 0.0);
    std::size_t count = 0;
    if (value_rank == 1 && value_size == 2)
    {
      for (CellIterator cell(mesh); !cell.end(); ++cell)
      {
        _data_values[count++] = data_values[*cell_offset];
        _data_values[count++] = data_values[*cell_offset + 1];
        ++count;
        ++cell_offset;
      }
    }
    else if (value_rank == 2 && value_size == 4)
    {
      // Pad with 0.0 to 2D tensors to make them 3D
      for (CellIterator cell(mesh); !cell.end(); ++cell)
      {
        for (std::size_t i = 0; i < 2; i++)
        {
          _data_values[count++] = data_values[*cell_offset + 2*i];
          _data_values[count++] = data_values[*cell_offset + 2*i + 1];
          ++count;
        }

        count += 3;
        ++cell_offset;
      }
    }
    else
    {
      // Write all components
      for (CellIterator cell(mesh); !cell.end(); ++cell)
      {
        for (std::size_t i = 0; i < value_size; i++)
          _data_values[count++] = data_values[*cell_offset + i];

        ++cell_offset;
      }
    }
    data_values = _data_values;
  }

  // FIXME: Below is messy. Should query HDF5 file writer for existing
  //        mesh name
  // Write mesh to HDF5 file
  if (parameters["rewrite_function_mesh"] || _counter == 0)
  {
    if (encoding == Encoding::HDF5)
    {
#ifdef HAS_HDF5
      const std::string h5_mesh_name = "/Mesh/" + std::to_string(_counter);
      boost::filesystem::path p(get_hdf5_filename(_filename));
      _current_mesh_name = p.filename().string() + ":" + h5_mesh_name;
      _hdf5_file->write(mesh, h5_mesh_name);
#endif
    }
    else if (encoding == Encoding::ASCII)
      _current_mesh_name = "/Xdmf/Domain/Grid";
  }

  const bool mpi_io = MPI::size(mesh.mpi_comm()) > 1 ? true : false;
  std::vector<std::int64_t> global_size(2);
  global_size[1] = padded_value_size;

  std::size_t num_global_points = mesh.size_global(0);
  const std::size_t num_global_cells = mesh.size_global(tdim);
  if (vertex_data)
  {
    if (degree == 2)
    {
      dolfin_assert(!mpi_io);
      num_global_points = mesh.size(0) + mesh.size(1);
      global_size[0] = num_global_points;
    }
    else
    {
      // Remove duplicates for vertex-based data in parallel
      DistributedMeshTools::reorder_values_by_global_indices(mesh, data_values,
                                                             padded_value_size);
      global_size[0] = num_global_points;
    }
  }
  else
    global_size[0] = num_global_cells;

  // Save data values to HDF5 file.  Vertex/cell values are saved in
  // the hdf5 group /VisualisationVector as distinct from /Vector
  // which is used for solution vectors.

  const std::string dataset_name = "/VisualisationVector/"
                                   + std::to_string(_counter);

#ifdef HAS_HDF5
  if (encoding == Encoding::HDF5)
  {
    _hdf5_file->write_data(dataset_name, data_values, global_size, mpi_io);

    // Flush file. Improves chances of recovering data if
    // interrupted. Also makes file somewhat readable between writes.
    if (parameters["flush_output"])
      _hdf5_file->flush();
  }
#endif

  // Write the XML meta description (see http://www.xdmf.org) on
  // process zero

  if (MPI::rank(mesh.mpi_comm()) == 0)
  {
    dolfin_assert(_xml);
    _xml->init_timeseries(u.name(), time_step, _counter);

    if (encoding == Encoding::HDF5)
    {
      _xml->mesh_topology(mesh.type().cell_type(), degree,
                          num_global_cells, _current_mesh_name + "/topology",
                          xdmf_format_str(encoding));
      _xml->mesh_geometry(num_global_points, gdim,
                          _current_mesh_name + "/coordinates",
                          xdmf_format_str(encoding));

      boost::filesystem::path p(get_hdf5_filename(_filename));
      _xml->data_attribute(u.name(), value_rank, vertex_data,
                           num_global_points, num_global_cells,
                           padded_value_size,
                           p.filename().string() + ":" + dataset_name,
                           xdmf_format_str(encoding));
    }
    else if (encoding == Encoding::ASCII)
    {
      // Add the mesh topology and geometry to the xml data
      std::string topology_data = generate_xdmf_ascii_mesh_topology_data(mesh);
      std::string geometry_data = generate_xdmf_ascii_mesh_geometry_data(mesh);

      _xml->mesh_topology(mesh.type().cell_type(), degree,
                          num_global_cells, topology_data,
                          xdmf_format_str(encoding));
      _xml->mesh_geometry(num_global_points, gdim, geometry_data,
                          xdmf_format_str(encoding));

      // Add the Function vertex data to the xml data
      _xml->data_attribute(u.name(), value_rank, vertex_data,
                           num_global_points, num_global_cells,
                           padded_value_size,
                           generate_xdmf_ascii_data(data_values, "%.15e"),
                           xdmf_format_str(encoding));
    }
    _xml->write();
  }

  // Increment counter
  ++_counter;
}
//----------------------------------------------------------------------------
void XDMFFile::write(const MeshFunction<bool>& meshfunction, Encoding encoding)
{
  write_mesh_function(meshfunction, "%d", encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const MeshFunction<int>& meshfunction, Encoding encoding)
{
  write_mesh_function(meshfunction, "%d", encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const MeshFunction<std::size_t>& meshfunction,
                     Encoding encoding)
{
  write_mesh_function(meshfunction, "%d", encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const MeshFunction<double>& meshfunction,
                     Encoding encoding)
{
  write_mesh_function(meshfunction, "%.15e", encoding);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const MeshValueCollection<std::size_t>& mvc,
                     Encoding encoding)
{
  //FIXME: make work with ASCII encoding or call dolfin_not_implemented()
  check_encoding(encoding);

  // Provide some very basic functionality for saving
  // MeshValueCollections mainly for saving values on a boundary mesh

  dolfin_assert(mvc.mesh());
  std::shared_ptr<const Mesh> mesh = mvc.mesh();

#ifdef HAS_HDF5
  if (encoding == Encoding::HDF5)
  {
    if (_hdf5_filemode != "w")
    {
      // Append to existing HDF5 File
      _hdf5_file.reset(new HDF5File(mesh->mpi_comm(), get_hdf5_filename(_filename), "a"));
      _hdf5_filemode = "w";
    }
  }
#endif

  if (MPI::sum(mesh->mpi_comm(), mvc.size()) == 0)
  {
    dolfin_error("XDMFFile.cpp",
                 "save empty MeshValueCollection",
                 "No values in MeshValueCollection");
  }

  if (_current_mesh_name == "")
  {
    dolfin_error("XDMFFile.cpp",
                 "save MeshValueCollection",
                 "A Mesh must be saved first");
  }

  const std::size_t cell_dim = mvc.dim();
  CellType::Type cell_type = mesh->type().entity_type(cell_dim);

  const std::string dataset_name = "/MVC/" + mvc.name();

#ifdef HAS_HDF5
  if (encoding == Encoding::HDF5)
  {
    // Use HDF5 function to output MeshValueCollection
    _hdf5_file->write(mvc, dataset_name);
  }
#endif

  bool time_series = parameters["time_series"];

  if (MPI::rank(mesh->mpi_comm()) == 0)
  {
    dolfin_assert(_xml);
    if (time_series)
      _xml->init_timeseries(mvc.name(), (double) _counter, _counter);
    else
      _xml->init_mesh(mvc.name());

    if (encoding == Encoding::HDF5)
    {
      boost::filesystem::path p(get_hdf5_filename(_filename));
      const std::string dataset_ref = p.filename().string() + ":"
        + dataset_name;

      _xml->mesh_topology(cell_type, 1, mvc.size(), dataset_ref + "/topology",
                         xdmf_format_str(encoding));
      _xml->mesh_geometry(mesh->size_global(0), mesh->geometry().dim(),
                         _current_mesh_name + "/coordinates",
                         xdmf_format_str(encoding));
      _xml->data_attribute(mvc.name(), 0, false, mesh->size_global(0),
                          mvc.size(), 1, dataset_ref + "/values",
                          xdmf_format_str(encoding));
    }
    else if (encoding == Encoding::ASCII)
    {
      // FIXME: This will only reference the first mesh geometry
      _xml->mesh_geometry(mesh->size_global(0), mesh->geometry().dim(),
                         _current_mesh_name + "/Geometry/DataItem",
                         xdmf_format_str(encoding), true);
      write_ascii_mesh_value_collection(mvc, dataset_name);
    }

    _xml->write();
  }

  ++_counter;
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const std::vector<Point>& points, Encoding encoding)
{
  check_encoding(encoding);

  const std::string group_name = "/Points";
  // Get number of points (global)
  const std::size_t num_global_points = MPI::sum(_mpi_comm, points.size());

  // Initialise HDF5 file
  if (encoding == Encoding::HDF5)
  {
#ifdef HAS_HDF5
    if (_hdf5_filemode != "w")
    {
      // Create HDF5 file (truncate)
      _hdf5_file.reset(new HDF5File(_mpi_comm, get_hdf5_filename(_filename), "w"));
      _hdf5_filemode = "w";
    }

    // Write HDF5 file
    _hdf5_file->write(points, group_name);
#else
    dolfin_error("XDMFile.cpp", "write vector<Point>", "Need HDF5 support");
#endif
  }
  else if (encoding == Encoding::ASCII)
  {
    // FIXME: Make ASCII output work
    dolfin_not_implemented();
  }

  // The XML created below will obliterate any existing XDMF file
  write_point_xml(group_name, num_global_points, 0, encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const std::vector<Point>& points,
                     const std::vector<double>& values,
                     Encoding encoding)
{
  // FIXME: make ASCII output work
  check_encoding(encoding);

  // Write clouds of points to XDMF/HDF5 with values
  dolfin_assert(points.size() == values.size());

  // Get number of points (global)
  const std::size_t num_global_points = MPI::sum(_mpi_comm, points.size());
  const std::string group_name = "/Points";

  if (encoding == Encoding::HDF5)
  {
#ifdef HAS_HDF5
    // Initialise HDF5 file
    if (_hdf5_filemode != "w")
    {
      // Create HDF5 file (truncate)
      _hdf5_file.reset(new HDF5File(_mpi_comm, get_hdf5_filename(_filename), "w"));
      _hdf5_filemode = "w";
    }

    // Write HDF5 file
    _hdf5_file->write(points, group_name);

    const std::string values_name = group_name + "/values";
    _hdf5_file->write(values, values_name);
#else
    dolfin_error("XDMFile.cpp", "write vector<Point>", "Need HDF5 support");
#endif
  }

  // The XML created will obliterate any existing XDMF file
  write_point_xml(group_name, num_global_points, 1, encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::read(Mesh& mesh, UseFilePartition use_file_partition)
{
  read_new(mesh);
  return;

  // Parse XML file
  dolfin_assert(_xml);
  _xml->read();

  // Get topology and geometry data
  const auto topology = _xml->get_topology();
  const auto geometry = _xml->get_geometry();

  // Create a cell type string (we need to catch the case where
  // TopologyType == "PolyLine")
  const std::string cell_type_str
    = (topology.cell_type == "polyline") ? "interval" : topology.cell_type;

  // Create a DOLFIN cell type from string
  std::unique_ptr<CellType> cell_type(CellType::create(cell_type_str));
  dolfin_assert(cell_type);

  // Get geometric and toplogical dimensions
  const int gdim = geometry.dim;
  const int tdim = cell_type->dim();
  dolfin_assert(tdim <= gdim);

  if (topology.format == "HDF")
  {
#ifdef HAS_HDF5
    if (geometry.hdf5_filename != topology.hdf5_filename)
    {
      dolfin_error("XDMFFile.cpp",
                   "read XDMF mesh",
                   "Topology and geometry file names do not match");
    }

    // Close any associated HDF5 which may be open
    _hdf5_file.reset();

    // Prepend directory name of XDMF file...
    // FIXME: not robust - topo.hdf5_filename may already be an
    // absolute path
    boost::filesystem::path xdmf_path(_filename);
    boost::filesystem::path hdf5_path(topology.hdf5_filename);
    HDF5File mesh_file(_mpi_comm, (xdmf_path.parent_path() / hdf5_path).string(), "r");

    // Read the mesh from the associated HDF5 file
    mesh_file.read(mesh, topology.hdf5_dataset, geometry.hdf5_dataset,
                   gdim, *cell_type,
                   topology.num_cells,
                   geometry.num_points,
                   static_cast<bool>(use_file_partition));
#else
    dolfin_error("XDMFile.cpp", "open Mesh file", "Need HDF5 support");
#endif
  }
  else if (topology.format == "XML")
  {
    if (MPI::rank(mesh.mpi_comm()) == 0)
    {
      MeshEditor editor;
      editor.open(mesh, cell_type_str, tdim, gdim);

      // Read geometry
      editor.init_vertices_global(geometry.num_points, geometry.num_points);

      const auto& g_data = geometry.data;
      std::istringstream iss(g_data);
      std::string data_line;
      std::vector<std::string> coords(gdim);
      Point p({0.0, 0.0, 0.0});
      std::size_t index = 0;
      while(std::getline(iss, data_line))
      {
        boost::split(coords, data_line, boost::is_any_of(" "));
        for (int j = 0; j < gdim; ++j)
          p[j] = std::stod(coords[j]);
        editor.add_vertex(index, p);
        ++index;
      }

      if (geometry.num_points != index)
      {
        dolfin_error("XDMFFile.cpp",
                     "parse mesh geometry points",
                     (boost::format("number of points found in data (%d) does not match xdmf meta data (%d)")
                      % index % geometry.num_points).str());
      }

      // Read topology
      editor.init_cells_global(topology.num_cells, topology.num_cells);

      const auto& t_data = topology.data;
      iss.clear();
      iss.str(t_data);
      index = 0;
      std::vector<std::string> splt_str_indices(topology.points_per_cell);
      std::vector<std::size_t> point_indices(topology.points_per_cell);
      while(std::getline(iss, data_line))
      {
        boost::split(splt_str_indices, data_line, boost::is_any_of(" "));
        for (std::size_t j = 0; j < topology.points_per_cell; ++j)
          point_indices[j] = std::stol(splt_str_indices[j]);

        editor.add_cell(index, point_indices);
        ++index;
      }

      if (topology.num_cells != index)
      {
        dolfin_error("XDMFFile.cpp",
                     "parse mesh topology",
                     (boost::format("number of cells found in data (%d) does not match xdmf meta data (%d)")
                      % index % topology.num_cells).str());
      }

      editor.close();
    }
  }
}
//----------------------------------------------------------------------------
void XDMFFile::read(MeshFunction<bool>& meshfunction)
{
  const std::shared_ptr<const Mesh> mesh = meshfunction.mesh();
  dolfin_assert(mesh);

  const std::size_t cell_dim = meshfunction.dim();
  MeshFunction<std::size_t> mf(mesh, cell_dim);
  read_mesh_function(mf);

  for (MeshEntityIterator cell(*mesh, cell_dim); !cell.end(); ++cell)
    meshfunction[cell->index()] = (mf[cell->index()] == 1);
}
//----------------------------------------------------------------------------
void XDMFFile::read(MeshFunction<int>& meshfunction)
{
  read_mesh_function(meshfunction);
}
//----------------------------------------------------------------------------
void XDMFFile::read(MeshFunction<std::size_t>& meshfunction)
{
  read_mesh_function(meshfunction);
}
//----------------------------------------------------------------------------
void XDMFFile::read(MeshFunction<double>& meshfunction)
{
  read_mesh_function(meshfunction);
}
//----------------------------------------------------------------------------
void XDMFFile::write_new(const Mesh& mesh, Encoding encoding) const
{
  // Check that encoding is supported
  check_encoding(encoding);

  // Create pugi doc
  pugi::xml_document xml_doc;

  // Add XDMF node and version attribute
  pugi::xml_node xdmf_node = xml_doc.append_child("Xdmf");
  dolfin_assert(xdmf_node);
  xdmf_node.append_attribute("Version") = "3.0";

  // Add domain node and add name attribute
  pugi::xml_node domain_node = xdmf_node.append_child("Domain");
  dolfin_assert(domain_node);
  domain_node.append_attribute("Name") = "Mesh produced by DOLFIN";

  // Add grid node and attributes
  pugi::xml_node grid_node = domain_node.append_child("Grid");
  dolfin_assert(grid_node);
  grid_node.append_attribute("Name") = mesh.name().c_str();
  grid_node.append_attribute("GridType") = "Uniform";

  // Open a HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
  std::unique_ptr<HDF5File> h5_file;
  if (encoding == Encoding::HDF5)
  {
    // Open file
    h5_file.reset(new HDF5File(mesh.mpi_comm(), get_hdf5_filename(_filename), "w"));
    dolfin_assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }

  // Add topology node and attributes (including writa data)
  const int tdim = mesh.topology().dim();
  const std::int64_t num_global_cells = mesh.size_global(tdim) ;
  if (num_global_cells < 1e9)
    add_topology_data<std::int32_t>(_mpi_comm, grid_node, h5_id, "/Mesh", mesh);
  else
    add_topology_data<std::int64_t>(_mpi_comm, grid_node, h5_id, "/Mesh", mesh);

  // Add geometry node and attributes (including writa data)
  add_geometry_data(_mpi_comm, grid_node, h5_id, "/Mesh", mesh);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm) == 0)
    xml_doc.save_file(_filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
void XDMFFile::read_new(Mesh& mesh) const
{
  // FIXME: check for duplicated XML nodes

  // Extract parent filepath (required to HDF5 when XDMF stores  relative path
  // of the HDF5 files(s) and the XDMF is not opened from its own directory)
  boost::filesystem::path xdmf_filename(_filename);
  const boost::filesystem::path parent_path = xdmf_filename.parent_path();

  if (!boost::filesystem::exists(xdmf_filename))
  {
    dolfin_error("XDMFFile.cpp",
                 "opening check that XDMF file",
                 "XDMF file \"%s\" does not exist");
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
                                 const Mesh& mesh)
{
  // Get topological dimension
  const int tdim = mesh.topology().dim();

  // Get number of cells (global) and vertices per cell from mesh
  const std::int64_t num_cells = mesh.topology().size_global(tdim);
  const int num_vertices_per_cell = mesh.type().num_vertices(tdim);

  // Get VTK string for cell type
  const std::string vtk_cell_str
    = vtk_cell_type_str(mesh.type().entity_type(tdim), 1);

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
  add_data_item(comm, topology_node, h5_id, h5_path, topology_data, shape);
}
//----------------------------------------------------------------------------
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

  // Add geometry ad DataItem node
  const std::string group_name = path_prefix + "/" + mesh.name();
  const std::string h5_path = group_name + "/geometry";
  const std::vector<std::int64_t> shape = {num_points, gdim};
  add_data_item(comm, geometry_node, h5_id, h5_path, x, shape);
}
//----------------------------------------------------------------------------
template<typename T>
void XDMFFile::add_data_item(MPI_Comm comm, pugi::xml_node& xml_node,
                             hid_t h5_id, const std::string h5_path, const T& x,
                             const std::vector<std::int64_t> shape)
{
  // Add DataItem node
  dolfin_assert(xml_node);
  pugi::xml_node data_item_node = xml_node.append_child("DataItem");
  dolfin_assert(data_item_node);

  // Add dimensions attribute
  data_item_node.append_attribute("Dimensions")
    = container_to_string(shape, " ", 16).c_str();

  // Add format attribute
  if (h5_id < 0)
  {
    data_item_node.append_attribute("Format") = "XML";
    data_item_node.append_child(pugi::node_pcdata).set_value(container_to_string(x, " ", 16).c_str());
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
  }
}
//----------------------------------------------------------------------------
template<typename T>
std::vector<T> XDMFFile::compute_topology_data(const Mesh& mesh, int cell_dim)
{
  // Create vector to store topology data
  const int num_vertices_per_cell = mesh.type().num_vertices(cell_dim);
  std::vector<T> topology_data;
  topology_data.reserve(mesh.num_entities(cell_dim)*(num_vertices_per_cell));

  // Get mesh communicator
  MPI_Comm comm = mesh.mpi_comm();

  // Pack data
  const std::vector<std::int8_t> perm = mesh.type().vtk_mapping();
  if (MPI::size(comm) == 1)
  {
    // Simple case when mesh is not distributed
    const std::vector<size_t>& global_vertices = mesh.topology().global_indices(0);
    for (MeshEntityIterator c(mesh, cell_dim); !c.end(); ++c)
    {
      const unsigned int* entities = c->entities(0);
      for (unsigned int i = 0; i != c->num_entities(0); ++i)
        topology_data.push_back(global_vertices[entities[perm[i]]]);
    }
  }
  else
  {
    // If not already numbered, number entities of order tdim so
    // we can get shared_entities
    DistributedMeshTools::number_entities(mesh, cell_dim);

    // Check if mesh has ghost layers
    const MeshTopology& topology = mesh.topology();
    const bool ghosted = (topology.size(cell_dim) == topology.ghost_offset(cell_dim));

    // Get MPI rank
    const int mpi_rank = MPI::rank(comm);

    // Build set of entities that are on this process but which
    // should not be output by this process because they will be output
    // by another process.
    std::set<int> non_local_entities;
    if (!ghosted)
    {
      // Get shared entities
      const std::map<std::int32_t, std::set<unsigned int>>& shared_entities
        = mesh.topology().shared_entities(cell_dim);

      // Iterate over shared entities, and add to exlcusion set
      // if there is a lower rank process with the entity
      for (auto e = shared_entities.begin(); e != shared_entities.end(); ++e)
      {
        const int lowest_rank_owner = *(e->second.begin());
        if (lowest_rank_owner < mpi_rank)
          non_local_entities.insert(e->first);
      }
    }
    else
    {
      // Iterate through ghost cells, adding non-ghost entities
      // which are in lower rank process cells to a set for
      // exclusion from output
      for (MeshEntityIterator c(mesh, cell_dim, "ghost"); !c.end(); ++c)
      {
        const int cell_owner = c->owner();
        for (MeshEntityIterator e(*c, cell_dim); !e.end(); ++e)
        {
          if (!e->is_ghost() and cell_owner < mpi_rank)
            non_local_entities.insert(e->index());
        }
      }
    }

    // Generate topology data for this process, with special case
    // for mesh of points
    topology_data.reserve(mesh.num_entities(cell_dim)*(num_vertices_per_cell));
    if (cell_dim == 0)
    {
      for (VertexIterator v(mesh); !v.end(); ++v)
      {
        if (non_local_entities.find(v->index()) == non_local_entities.end())
          topology_data.push_back(v->global_index());
      }
    }
    else
    {
      // Local-to-global map for point indices
      const std::vector<std::size_t>& global_vertices
        = mesh.topology().global_indices(0);
      for (MeshEntityIterator e(mesh, cell_dim); !e.end(); ++e)
      {
        // If not excluded, add to topology
        if (non_local_entities.find(e->index()) == non_local_entities.end())
        {
          for (unsigned int i = 0; i != e->num_entities(0); ++i)
          {
            const int local_idx = e->entities(0)[perm[i]];
            topology_data.push_back(global_vertices[local_idx]);
          }
        }
      }
    }
  }

  return topology_data;
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
  // from the number of cells and the cell type (for topology, once
  // must supply cell type + (number of cells or dimensions).
  //
  // A geometry data item must have 'Dimensions' attribute

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
    // Get file and data path
    auto paths = XDMFxml::get_hdf5_paths(dataset_node);

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
void XDMFFile::read_mesh_function(MeshFunction<T>& meshfunction)
{
  dolfin_assert(_xml);
  _xml->read();

  Encoding encoding = get_file_encoding();
  check_encoding(encoding);

  if (encoding == Encoding::HDF5)
  {
#ifdef HAS_HDF5
    const std::string data_name = _xml->dataname();
    if (_hdf5_filemode != "r")
    {
      _hdf5_file.reset(new HDF5File(_mpi_comm, get_hdf5_filename(_filename), "r"));
      _hdf5_filemode = "r";
    }
    dolfin_assert(_hdf5_file);

    // Try to read the meshfunction from the associated HDF5 file
    _hdf5_file->read(meshfunction, "/Mesh/" + data_name);
#else
    dolfin_error("XDMFile.cpp", "open MeshFunction file", "Need HDF5 support");
#endif
  }
  else if (encoding == Encoding::ASCII)
  {
    std::vector<std::string> data_lines;
    const std::string data_set = _xml->get_first_data_set();
    boost::split(data_lines, data_set, boost::is_any_of("\n"));

    const std::size_t n_lines = data_lines.size();
    for (std::size_t j = 0; j < n_lines; ++j)
      meshfunction[j] = boost::lexical_cast<T>(data_lines[j]);
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
                                   std::string format, Encoding encoding)
{
  check_encoding(encoding);

  // Get mesh
  dolfin_assert(meshfunction.mesh());
  std::shared_ptr<const Mesh> mesh = meshfunction.mesh();

#ifdef HAS_HDF5
  if (encoding == Encoding::HDF5)
  {
    if (_hdf5_filemode != "w")
    {
      // Create HDF5 file (truncate)
      _hdf5_file.reset(new HDF5File(mesh->mpi_comm(), get_hdf5_filename(_filename), "w"));
      _hdf5_filemode = "w";
    }
  }
#endif

  if (meshfunction.size() == 0)
  {
    dolfin_error("XDMFFile.cpp",
                 "save empty MeshFunction",
                 "No values in MeshFunction");
  }

  const std::size_t cell_dim = meshfunction.dim();
  CellType::Type cell_type = mesh->type().entity_type(cell_dim);

  if (encoding == Encoding::HDF5)
  {
#ifdef HAS_HDF5
    // Use HDF5 function to output MeshFunction
    const std::string h5_mesh_name = "/Mesh/" + std::to_string(_counter);
    boost::filesystem::path p(get_hdf5_filename(_filename));
    _current_mesh_name = p.filename().string() + ":" + h5_mesh_name;
    _hdf5_file->write(meshfunction, h5_mesh_name);
#endif
  }
  else if (encoding == Encoding::ASCII)
    _current_mesh_name = "/Xdmf/Domain/Grid";

  // Saved MeshFunction values are in the /Mesh group
  const std::string dataset_name = _current_mesh_name + "/values";

  bool time_series = parameters["time_series"];

  if (MPI::rank(mesh->mpi_comm()) == 0)
  {
    dolfin_assert(_xml);
    const std::string meshfunction_name = meshfunction.name();
    if (time_series)
      _xml->init_timeseries(meshfunction_name, (double) _counter, _counter);
    else
      _xml->init_mesh(meshfunction_name);

    if (encoding == Encoding::HDF5)
    {
      _xml->mesh_topology(cell_type, 1, mesh->size_global(cell_dim),
                          _current_mesh_name + "/topology",
                          xdmf_format_str(encoding));
      _xml->mesh_geometry(mesh->size_global(0), mesh->geometry().dim(),
                          _current_mesh_name + "/coordinates",
                          xdmf_format_str(encoding));
      _xml->data_attribute(meshfunction_name, 0, false, mesh->size_global(0),
                           mesh->size_global(cell_dim), 1, dataset_name,
                           xdmf_format_str(encoding));
    }
    else if (encoding == Encoding::ASCII)
    {
      // Add the mesh topology and geometry to the xml data
      _xml->mesh_topology(cell_type, 1, mesh->size_global(cell_dim),
                          generate_xdmf_ascii_mesh_topology_data(*mesh,
                                                                 cell_dim),
                          xdmf_format_str(encoding));

      _xml->mesh_geometry(mesh->size_global(0), mesh->geometry().dim(),
                          generate_xdmf_ascii_mesh_geometry_data(*mesh),
                          xdmf_format_str(encoding));

      // Note: we use boost::container::vector rather than std::vector
      // to avoid well known problem with std::vector<bool>
      // No duplicates - ignore ghost cells if present
      boost::container::vector<T>
        data_values(meshfunction.values(), meshfunction.values()
                    + mesh->topology().ghost_offset(cell_dim));

      _xml->data_attribute(meshfunction_name, 0, false, mesh->size_global(0),
                           mesh->size_global(cell_dim), 1,
                           generate_xdmf_ascii_data(data_values, format),
                           xdmf_format_str(encoding));
    }

    _xml->write();
  }

  ++_counter;
}
//-----------------------------------------------------------------------------
void XDMFFile::write_point_xml(const std::string group_name,
                               const std::size_t num_global_points,
                               const unsigned int value_size,
                               Encoding encoding)
{
  // FIXME: move to XDMFxml.cpp

  // Write the XML meta description on process zero
  if (MPI::rank(_mpi_comm) == 0)
  {
    dolfin_assert(_xml);
    _xml->init_mesh("Point cloud");

    // Point topology, no connectivity data
    _xml->mesh_topology(CellType::Type::point, 0, num_global_points, "",
                        xdmf_format_str(encoding));

    // Describe geometric coordinates
    // FIXME: assumes 3D
    boost::filesystem::path p(get_hdf5_filename(_filename));
    _xml->mesh_geometry(num_global_points, 3,
                        p.filename().string() + ":/Points",
                        xdmf_format_str(encoding));

    if(value_size != 0)
    {
      dolfin_assert(value_size == 1 || value_size == 3);
      _xml->data_attribute("point_values", 0, true,
                           num_global_points, num_global_points, value_size,
                           p.filename().string() + ":" + group_name + "/values",
                           xdmf_format_str(encoding));
    }

    _xml->write();
  }
}
//----------------------------------------------------------------------------
void XDMFFile::get_point_data_values(std::vector<double>& data_values,
                                     std::size_t width,
                                     const Function& u) const
{
  const Mesh& mesh = *u.function_space()->mesh();
  const std::size_t value_size = u.value_size();
  const std::size_t value_rank = u.value_rank();

  if (mesh.geometry().degree() == 1)
  {
    u.compute_vertex_values(data_values, mesh);
    const std::size_t num_local_vertices = mesh.size(0);

    if (value_rank > 0)
    {
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
  }
  else if (mesh.geometry().degree() == 2)
  {
    const std::size_t num_local_points = mesh.size(0) + mesh.size(1);
    data_values.resize(width*num_local_points);
    std::vector<dolfin::la_index> data_dofs(data_values.size(), 0);

    dolfin_assert(u.function_space()->dofmap());
    const GenericDofMap& dofmap = *u.function_space()->dofmap();

    // Function can be P1 or P2
    if (dofmap.num_entity_dofs(1) == 0)
    {
      // P1
      for (CellIterator cell(mesh); !cell.end(); ++cell)
      {
        const ArrayView<const dolfin::la_index> dofs
          = dofmap.cell_dofs(cell->index());
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
      for (EdgeIterator e(mesh); !e.end(); ++e)
      {
        const std::size_t v0 = e->entities(0)[0];
        const std::size_t v1 = e->entities(0)[1];
        const std::size_t e0 = (e->index() + mesh.size(0))*width;
        for (std::size_t i = 0; i != value_size; ++i)
          data_values[e0 + i] = (data_values[v0 + i] + data_values[v1 + i])/2.0;
      }
    }
    else if (dofmap.num_entity_dofs(0) == dofmap.num_entity_dofs(1))
    {
      // P2
      // Go over all cells inserting values
      // FIXME: a lot of duplication here
      for (CellIterator cell(mesh); !cell.end(); ++cell)
      {
        const ArrayView<const dolfin::la_index> dofs
          = dofmap.cell_dofs(cell->index());
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
            const std::size_t e0 = (e->index() + mesh.size(0))*width;
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
  }
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
std::string XDMFFile::generate_xdmf_ascii_mesh_topology_data(const Mesh& mesh)
{
  return generate_xdmf_ascii_mesh_topology_data(mesh, mesh.geometry().dim());
}
//-----------------------------------------------------------------------------
std::string
XDMFFile::generate_xdmf_ascii_mesh_topology_data(const Mesh& mesh,
                                                 const std::size_t edim)
{
  std::unique_ptr<CellType>
    celltype(CellType::create(mesh.type().entity_type(edim)));

  // Permutation to VTK ordering
  const std::vector<std::int8_t> perm = celltype->vtk_mapping();

  std::string topology_xml_value;
  topology_xml_value += "\n";
  if (edim == 0)
  {
    for (VertexIterator v(mesh); !v.end(); ++v)
    {
      topology_xml_value
        += boost::str(boost::format("%d") % v->global_index()) + "\n";
    }
  }
  else
  {
    for (MeshEntityIterator c(mesh, edim); !c.end(); ++c)
    {
      for (unsigned int i = 0; i != c->num_entities(0); ++i)
      {
        const std::size_t local_idx = c->entities(0)[perm[i]];
        topology_xml_value
          += boost::str(boost::format("%d") % local_idx) + " ";
      }
      topology_xml_value += "\n";
    }
  }

  return topology_xml_value;
}
//-----------------------------------------------------------------------------
std::string XDMFFile::generate_xdmf_ascii_mesh_geometry_data(const Mesh& mesh)
{
  const std::size_t gdim = mesh.geometry().dim();
  std::string geometry_xml_value;
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    geometry_xml_value += "\n";
    const double* p = v->x();
    for (size_t i = 0; i < gdim; ++i)
      geometry_xml_value += boost::str(boost::format("%.15e") % p[i]) + " ";
  }
  geometry_xml_value += "\n";
  return geometry_xml_value;
}
//-----------------------------------------------------------------------------
template<typename T>
std::string XDMFFile::generate_xdmf_ascii_data(const T& data,
                                               std::string format)
{
  std::string data_str;
  data_str += "\n";
  for (std::size_t j = 0; j < data.size(); ++j)
    data_str += boost::str(boost::format(format) % data[j]) + "\n";
  return data_str;
}
//-----------------------------------------------------------------------------
template<typename T>
std::string XDMFFile::generate_xdmf_ascii_data(const T& data)
{
  std::string data_str;
  data_str += "\n";
  for (std::size_t j = 0; j < data.size(); ++j)
    data_str += boost::lexical_cast<std::string>(data[j]) + "\n";
  return data_str;
}
//-----------------------------------------------------------------------------
XDMFFile::Encoding XDMFFile::get_file_encoding() const
{
  dolfin_assert(_xml);
  _xml->read();
  const std::string xml_encoding_attrib = _xml->data_encoding();

  return get_file_encoding(xml_encoding_attrib);
}
//-----------------------------------------------------------------------------
XDMFFile::Encoding XDMFFile::get_file_encoding(std::string xml_encoding_attrib)
{
  return (xml_encoding_attrib == "XML") ? Encoding::ASCII : Encoding::HDF5;
}
//-----------------------------------------------------------------------------
template <typename T> void XDMFFile::write_ascii_mesh_value_collection(
  const MeshValueCollection<T>& mesh_values,
  const std::string name)
{
  const std::size_t dim = mesh_values.dim();
  std::shared_ptr<const Mesh> mesh = mesh_values.mesh();

  const std::map<std::pair<std::size_t, std::size_t>, T>& values
    = mesh_values.values();

  CellType::Type cell_type = mesh->type().entity_type(dim);
  std::unique_ptr<CellType> entity_type(CellType::create(cell_type));

  std::vector<T> value_data;
  value_data.reserve(values.size());
  std::string topology;
  topology += "\n";

  const std::size_t tdim = mesh->topology().dim();
  mesh->init(tdim, dim);
  for (auto &p : values)
  {
    MeshEntity cell = Cell(*mesh, p.first.first);
    if (dim != tdim)
    {
      const unsigned int entity_local_idx = cell.entities(dim)[p.first.second];
      cell = MeshEntity(*mesh, dim, entity_local_idx);
    }

    for (VertexIterator v(cell); !v.end(); ++v)
      topology += boost::str(boost::format("%d") % v->global_index()) + " ";

    topology += "\n";

    dolfin_assert(_xml);
    _xml->mesh_topology(cell_type, 1, mesh_values.size(), topology,
                        xdmf_format_str(Encoding::ASCII));

    _xml->data_attribute(mesh_values.name(), 0, false,
                         mesh->size_global(0), mesh_values.size(),
                         1, generate_xdmf_ascii_data(value_data),
                         xdmf_format_str(Encoding::ASCII));
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
      return "Qaud_8";
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
