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
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <dolfin/common/MPI.h>
#include <dolfin/common/defines.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/mesh/Vertex.h>
#include "HDF5File.h"
#include "HDF5Utility.h"
#include "XDMFFile.h"
#include "XDMFxml.h"

using namespace dolfin;

//----------------------------------------------------------------------------
XDMFFile::XDMFFile(MPI_Comm comm, const std::string filename)
  : _mpi_comm(comm), current_mesh_name(""),
    _filename(filename), counter(0)
{
  // Make name for HDF5 file (used to store data)
  boost::filesystem::path p(filename);
  p.replace_extension(".h5");
  hdf5_filename = p.string();

  // File mode will be set when reading or writing
  hdf5_filemode = "";

  // Rewrite the mesh at every time step in a time series. Should be
  // turned off if the mesh remains constant.
  parameters.add("rewrite_function_mesh", true);

  // Flush datasets to disk at each timestep. Allows inspection of the
  // HDF5 file whilst running, at some performance cost.
  parameters.add("flush_output", false);

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
void XDMFFile::get_point_data_values(std::vector<double>& data_values,
                                     std::size_t width,
                                     const Function& u)
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
      std::vector<double> _data_values(width*num_local_vertices,
                                       0.0);
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
      for (std::size_t i = 0; i < data_values.size(); i += 3)
        data_values[i + 2] = 0.0;
    else if (value_rank == 2 and value_size == 4)
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
//----------------------------------------------------------------------------
void XDMFFile::write(const Function& u, XDMFFile::Encoding encoding)
{
  write(u, (double) counter, encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const Function& u, double time_step, XDMFFile::Encoding encoding)
{

  // Conditions for starting a new HDF5 file
#ifdef HAS_HDF5
  if (encoding == XDMFFile::Encoding::HDF5)
  {
    const int mf_interval = parameters["multi_file"];
    if ( (mf_interval != 0 and counter%mf_interval == 0) or hdf5_filemode != "w" )
    {
      // Make name for HDF5 file (used to store data)
      boost::filesystem::path p(_filename);
      p.replace_extension(".h5");
      hdf5_filename = p.string();

      if (mf_interval != 0)
      {
        std::stringstream s;
        s << std::setw(6) << std::setfill('0') << counter;
        hdf5_filename += "_" + s.str();
      }

      // Create new HDF5 file (truncate),
      // closing any open file from a previous timestep
      hdf5_file.reset(new HDF5File(_mpi_comm, hdf5_filename, "w"));
      hdf5_filemode = "w";
    }

    dolfin_assert(hdf5_file);
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
  {
    get_point_data_values(data_values, padded_value_size, u);
  }
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
      *(cell_offset + 1)
        = *(cell_offset) + dofmap.num_element_dofs(cell->index());
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
  if (parameters["rewrite_function_mesh"] || counter == 0)
  {
#ifdef HAS_HDF5
    if (encoding == XDMFFile::Encoding::HDF5)
    {
      const std::string h5_mesh_name = "/Mesh/" + std::to_string(counter);
      boost::filesystem::path p(hdf5_filename);
      current_mesh_name = p.filename().string() + ":" + h5_mesh_name;
      hdf5_file->write(mesh, h5_mesh_name);
    }
#endif
  }

  const bool mpi_io = MPI::size(mesh.mpi_comm()) > 1 ? true : false;
  std::vector<std::size_t> global_size(2);
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
                                   + std::to_string(counter);

#ifdef HAS_HDF5
  if (encoding == XDMFFile::Encoding::HDF5)
  {
    hdf5_file->write_data(dataset_name, data_values, global_size, mpi_io);

    // Flush file. Improves chances of recovering data if
    // interrupted. Also makes file somewhat readable between writes.
    if (parameters["flush_output"])
      hdf5_file->flush();
  }
#endif

  // Write the XML meta description (see http://www.xdmf.org) on
  // process zero

  if (MPI::rank(mesh.mpi_comm()) == 0)
  {
    XDMFxml xml(_filename);
    xml.init_timeseries(u.name(), time_step, counter);

    if (encoding == XDMFFile::Encoding::HDF5)
    {
      xml.mesh_topology(mesh.type().cell_type(), degree,
                        num_global_cells, current_mesh_name, xdmf_format_str(encoding));
      xml.mesh_geometry(num_global_points, gdim, current_mesh_name, xdmf_format_str(encoding));

      boost::filesystem::path p(hdf5_filename);
      xml.data_attribute(u.name(), value_rank, vertex_data,
                         num_global_points, num_global_cells,
                         padded_value_size,
                         p.filename().string() + ":" + dataset_name,
                         xdmf_format_str(encoding));
    }
    else if (encoding == XDMFFile::Encoding::ASCII)
    {
      // Add the mesh topology and geometry to the xml data
      std::string topology_data = generate_xdmf_ascii_mesh_topology_data(mesh);
      std::string geometry_data = generate_xdmf_ascii_mesh_geometry_data(mesh);

      xml.mesh_topology(mesh.type().cell_type(), degree,
                        num_global_cells, topology_data, xdmf_format_str(encoding));
      xml.mesh_geometry(num_global_points, gdim, geometry_data, xdmf_format_str(encoding));

      // Add the Function vertex data to the xml data
      xml.data_attribute(u.name(), value_rank, vertex_data,
                         num_global_points, num_global_cells,
                         padded_value_size,
                         generate_xdmf_ascii_vertex_data(data_values, "%.15e"),
                         xdmf_format_str(encoding));
    }
    xml.write();
  }

  // Increment counter
  ++counter;
}
//----------------------------------------------------------------------------
void XDMFFile::read(Mesh& mesh, bool use_partition_from_file)
{

  XDMFxml xml(_filename);
  xml.read();

  const auto topo = xml.get_topology();
  const auto geom = xml.get_geometry();

  if (topo.format == "HDF")
  {
#ifdef HAS_HDF5
    if (geom.hdf5_filename != topo.hdf5_filename)
    {
      dolfin_error("XDMFFile.cpp",
                   "read XDMF mesh",
                   "Topology and geometry file names do not match");
    }
    // Close any associated HDF5 which may be open
    hdf5_file.reset();
    // Prepend directory name of XDMF file...
    // FIXME: not robust - topo.hdf5_filename may already be an absolute path
    boost::filesystem::path xdmf_path(_filename);
    HDF5File mesh_file(_mpi_comm, xdmf_path.parent_path().string()
                                  + "/" + topo.hdf5_filename, "r");
    // Try to read the mesh from the associated HDF5 file
    mesh_file.read(mesh, topo.hdf5_dataset, geom.hdf5_dataset,
                   topo.cell_type, use_partition_from_file);
#else
    dolfin_error("XDMFile.cpp", "open Mesh file", "Need HDF5 support");
#endif
  }
  else if (topo.format == "XML")
  {
    if (MPI::rank(mesh.mpi_comm()) == 0)
    {
      // Create mesh for editing
      std::unique_ptr<CellType> cell_type(CellType::create(topo.cell_type));
      std::size_t tdim = cell_type->dim();

      MeshEditor editor;
      editor.open(mesh, topo.cell_type, tdim, geom.dim);

      // Read geometry
      editor.init_vertices_global(geom.n_points, geom.n_points);

      const auto& g_data = geom.data;
      std::istringstream iss(g_data);
      std::string data_line;
      std::vector<std::string> coords(geom.dim);
      Point p;
      std::size_t index = 0;
      while(std::getline(iss, data_line))
      {
        boost::split(coords, data_line, boost::is_any_of(" "));
        for (int j = 0; j < geom.dim; ++j) {
          p[j] = std::stod(coords[j]);
        }
        editor.add_vertex(index, p);
        ++index;
      }

      if (geom.n_points != index)
        dolfin_error("XDMFFile.cpp",
                     "parse mesh geometry points",
                     (boost::format("number of points found in data (%d) does not match xdmf meta data (%d)")
                      % index % geom.n_points).str());

      // Read topology
      editor.init_cells_global(topo.n_cells, topo.n_cells);

      const auto& t_data = topo.data;
      iss.clear();
      iss.str(t_data);
      index = 0;
      std::vector<std::string> splt_str_indices(topo.points_per_cell);
      std::vector<std::size_t> point_indices(topo.points_per_cell);
      while(std::getline(iss, data_line))
      {
        boost::split(splt_str_indices, data_line, boost::is_any_of(" "));
        for (int j = 0; j < topo.points_per_cell; ++j)
          point_indices[j] = std::stol(splt_str_indices[j]);
        editor.add_cell(index, point_indices);
        ++index;
      }

      if (topo.n_cells != index)
        dolfin_error("XDMFFile.cpp",
                     "parse mesh topology",
                     (boost::format("number of cells found in data (%d) does not match xdmf meta data (%d)")
                      % index % topo.n_cells).str());

      editor.close();
    }
  }
}
//----------------------------------------------------------------------------
void XDMFFile::write(const Mesh& mesh, XDMFFile::Encoding encoding)
{
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
    num_total_points +=
        mesh.geometry().num_entity_coordinates(i)*mesh.size_global(i);

  const std::string group_name = "/Mesh/" + name;

  // Write hdf5 file on all processes
#ifdef HAS_HDF5
  if (encoding == XDMFFile::Encoding::HDF5)
  {
    // Write mesh to HDF5 file
    // The XML below will obliterate any existing XDMF file
    if (hdf5_filemode != "w")
    {
      // Create HDF5 file (truncate)
      hdf5_file.reset(new HDF5File(mesh.mpi_comm(), hdf5_filename, "w"));
      hdf5_filemode = "w";
    }
    hdf5_file->write(mesh, cell_dim, group_name);
  }
#endif

  // Write the XML meta description on process zero
  if (MPI::rank(mesh.mpi_comm()) == 0)
  {
    XDMFxml xml(_filename);
    xml.init_mesh(name);

    // Write the topology
    std::string topology_xml_value;

    if (encoding == XDMFFile::Encoding::ASCII)
    {
      topology_xml_value = generate_xdmf_ascii_mesh_topology_data(mesh);
    }
    else if (encoding == XDMFFile::Encoding::HDF5)
    {
      const boost::filesystem::path p(hdf5_filename);
      topology_xml_value = p.filename().string() + ":" + group_name + "/topology";
    }

    // Describe topological connectivity
    xml.mesh_topology(mesh.type().cell_type(), mesh.geometry().degree(),
                      num_global_cells, topology_xml_value, xdmf_format_str(encoding));

    // Write the geometry
    std::string geometry_xml_value;

    if (encoding == XDMFFile::Encoding::ASCII)
    {
      geometry_xml_value = generate_xdmf_ascii_mesh_geometry_data(mesh);
    }
    else if (encoding == XDMFFile::Encoding::HDF5)
    {
      const boost::filesystem::path p(hdf5_filename);
      geometry_xml_value = p.filename().string() + ":" + group_name + "/coordinates";
    }

    // Describe geometric coordinates
    xml.mesh_geometry(num_total_points, gdim,
                      geometry_xml_value, xdmf_format_str(encoding));

    xml.write();
  }
}
//----------------------------------------------------------------------------
void XDMFFile::write(const MeshFunction<bool>& meshfunction,
                     XDMFFile::Encoding encoding)
{
  write_mesh_function(meshfunction, "%d", encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const MeshFunction<int>& meshfunction,
                     XDMFFile::Encoding encoding)
{
  write_mesh_function(meshfunction, "%d", encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const MeshFunction<std::size_t>& meshfunction,
                     XDMFFile::Encoding encoding)
{
  write_mesh_function(meshfunction, "%d", encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const MeshFunction<double>& meshfunction,
                     XDMFFile::Encoding encoding)
{
  write_mesh_function(meshfunction, "%.15e", encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const std::vector<Point>& points,
                     XDMFFile::Encoding encoding)
{
  // FIXME: Make ASCII output work
  check_encoding(encoding);

  const std::string group_name = "/Points";
  // Get number of points (global)
  const std::size_t num_global_points = MPI::sum(_mpi_comm, points.size());

  // Initialise HDF5 file
#ifdef HAS_HDF5
  if (encoding == XDMFFile::Encoding::HDF5)
  {
    if (hdf5_filemode != "w")
    {
      // Create HDF5 file (truncate)
      hdf5_file.reset(new HDF5File(_mpi_comm, hdf5_filename, "w"));
      hdf5_filemode = "w";
    }

    // Write HDF5 file
    hdf5_file->write(points, group_name);
  }
#endif

  // The XML created below will obliterate any existing XDMF file
  write_point_xml(group_name, num_global_points, 0, encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const std::vector<Point>& points,
                     const std::vector<double>& values,
                     XDMFFile::Encoding encoding)
{
  // FIXME: make ASCII output work
  check_encoding(encoding);
  // Write clouds of points to XDMF/HDF5 with values
  dolfin_assert(points.size() == values.size());
  // Get number of points (global)
  const std::size_t num_global_points = MPI::sum(_mpi_comm, points.size());
  const std::string group_name = "/Points";

#ifdef HAS_HDF5
  if (encoding == XDMFFile::Encoding::HDF5)
  {
    // Initialise HDF5 file
    if (hdf5_filemode != "w")
    {
      // Create HDF5 file (truncate)
      hdf5_file.reset(new HDF5File(_mpi_comm, hdf5_filename, "w"));
      hdf5_filemode = "w";
    }

    // Write HDF5 file
    hdf5_file->write(points, group_name);

    const std::string values_name = group_name + "/values";
    hdf5_file->write(values, values_name);
  }
#endif

  // The XML created will obliterate any existing XDMF file
  write_point_xml(group_name, num_global_points, 1, encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::write_point_xml(const std::string group_name,
                               const std::size_t num_global_points,
                               const unsigned int value_size,
                               XDMFFile::Encoding encoding)
{
  // FIXME: move to XDMFxml.cpp

  // Write the XML meta description on process zero
  if (MPI::rank(_mpi_comm) == 0)
  {
    XDMFxml xml(_filename);
    xml.init_mesh("Point cloud");

    // Point topology, no connectivity data
    xml.mesh_topology(CellType::Type::point, 0, num_global_points, "", xdmf_format_str(encoding));

    // Describe geometric coordinates
    // FIXME: assumes 3D
    boost::filesystem::path p(hdf5_filename);
    xml.mesh_geometry(num_global_points, 3,
                      p.filename().string() + ":/Points", xdmf_format_str(encoding));

    if(value_size != 0)
    {
      dolfin_assert(value_size == 1 || value_size == 3);
      xml.data_attribute("point_values", 0, true,
                         num_global_points, num_global_points, value_size,
                         p.filename().string() + ":" + group_name + "/values",
                         xdmf_format_str(encoding));
    }

    xml.write();
  }
}
//----------------------------------------------------------------------------
void XDMFFile::write(const MeshValueCollection<std::size_t>& mvc,
                     XDMFFile::Encoding encoding)
{
  check_encoding(encoding);

  // Provide some very basic functionality for saving MeshValueCollections
  // mainly for saving values on a boundary mesh

  dolfin_assert(mvc.mesh());
  std::shared_ptr<const Mesh> mesh = mvc.mesh();

#ifdef HAS_HDF5
  if (encoding == XDMFFile::Encoding::HDF5)
  {
    if (hdf5_filemode != "w")
    {
      // Append to existing HDF5 File
      hdf5_file.reset(new HDF5File(mesh->mpi_comm(), hdf5_filename, "a"));
      hdf5_filemode = "w";
    }
  }
#endif

  if (mvc.size() == 0)
  {
    dolfin_error("XDMFFile.cpp",
                 "save empty MeshValueCollection",
                 "No values in MeshValueCollection");
  }

  if (current_mesh_name == "")
  {
    dolfin_error("XDMFFile.cpp",
                 "save MeshValueCollection",
                 "A Mesh must be saved first");
  }

  const std::size_t cell_dim = mvc.dim();
  CellType::Type cell_type = mesh->type().entity_type(cell_dim);

  const std::string dataset_name = "/MVC/" + mvc.name();

#ifdef HAS_HDF5
  if (encoding == XDMFFile::Encoding::HDF5)
  {
    // Use HDF5 function to output MeshValueCollection
    hdf5_file->write(mvc, dataset_name);
  }
#endif

  bool time_series = parameters["time_series"];

  if (MPI::rank(mesh->mpi_comm()) == 0)
  {
    XDMFxml xml(_filename);
    if (time_series)
      xml.init_timeseries(mvc.name(), (double)counter, counter);
    else
      xml.init_mesh(mvc.name());

    boost::filesystem::path p(hdf5_filename);
    const std::string dataset_ref
        = p.filename().string() + ":" + dataset_name;

    xml.mesh_topology(cell_type, 1, mvc.size(), dataset_ref, xdmf_format_str(encoding));
    xml.mesh_geometry(mesh->size_global(0), mesh->geometry().dim(),
                      current_mesh_name, xdmf_format_str(encoding));
    xml.data_attribute(mvc.name(), 0, false, mesh->size_global(0),
                       mvc.size(), 1, dataset_ref + "/values",
                       xdmf_format_str(encoding));
    xml.write();
  }

  ++counter;
}
//-----------------------------------------------------------------------------
template<typename T>
void XDMFFile::write(const MeshFunction<T>& meshfunction,
                     std::string format,
                     XDMFFile::Encoding encoding)
{
  write_mesh_function(meshfunction, format, encoding);
}
//----------------------------------------------------------------------------
void XDMFFile::read(MeshFunction<bool>& meshfunction)
{
  const Mesh& mesh = *meshfunction.mesh();
  const std::size_t cell_dim = meshfunction.dim();

  MeshFunction<std::size_t> mf(mesh, cell_dim);
  read_mesh_function(mf);

  for (MeshEntityIterator cell(mesh, cell_dim); !cell.end(); ++cell)
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
template<typename T>
void XDMFFile::read_mesh_function(MeshFunction<T>& meshfunction)
{
  XDMFxml xml(_filename);
  xml.read();

  Encoding encoding = get_file_encoding();

  if (encoding == Encoding::HDF5)
  {
#ifdef HAS_HDF5
    const std::string data_name = xml.dataname();

    if (hdf5_filemode != "r")
    {
      hdf5_file.reset(new HDF5File(_mpi_comm, hdf5_filename, "r"));
      hdf5_filemode = "r";
    }

    dolfin_assert(hdf5_file);

    // Try to read the meshfunction from the associated HDF5 file
    hdf5_file->read(meshfunction, "/Mesh/" + data_name);
#else
    dolfin_error("XDMFile.cpp", "open MeshFunction file", "Need HDF5 support");
#endif
  }
  else if (encoding == Encoding::ASCII)
  {
    std::vector<std::string> data_lines;
    const std::string data_set = xml.get_first_data_set();
    boost::split(data_lines, data_set, boost::is_any_of("\n"));

    const std::size_t n_lines = data_lines.size();
    for (std::size_t j = 0; j < n_lines; ++j)
    {
      meshfunction[j] = boost::lexical_cast<T>(data_lines[j]);
    }
  }
}
//----------------------------------------------------------------------------
void XDMFFile::check_encoding(XDMFFile::Encoding encoding)
{
  if (encoding == XDMFFile::Encoding::HDF5 and !has_hdf5())
  {
    dolfin_error("XDMFFile.cpp",
                 "write XDMF file",
                 "DOLFIN has not been compiled with HDF5 support");
  }
  if (encoding == XDMFFile::Encoding::ASCII and MPI::size(_mpi_comm) != 1)
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
std::string XDMFFile::generate_xdmf_ascii_mesh_topology_data(const Mesh& mesh,
                                                             const std::size_t edim)
{
  std::unique_ptr<CellType> celltype(CellType::create(mesh.type().entity_type(edim)));

  // Permutation to VTK ordering
  const std::vector<unsigned int> perm = celltype->vtk_mapping();

  std::string topology_xml_value;
  topology_xml_value += "\n";
  if (edim == 0)
  {
    for (VertexIterator v(mesh); !v.end(); ++v)
      topology_xml_value += boost::str(boost::format("%d") % v->global_index()) + "\n";
  }
  else
  {
    for (MeshEntityIterator c(mesh, edim); !c.end(); ++c)
    {
      for (unsigned int i = 0; i != c->num_entities(0); ++i)
      {
        const std::size_t local_idx = c->entities(0)[perm[i]];
        topology_xml_value += boost::str(boost::format("%d") % local_idx) + " ";
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
std::string XDMFFile::generate_xdmf_ascii_vertex_data(
    const std::vector<T>& data, std::string format)
{
  std::string data_str;
  data_str += "\n";
  for (std::size_t j = 0; j < data.size(); ++j)
    data_str += boost::str(boost::format(format) % data[j]) + "\n";
  return data_str;
}
//-----------------------------------------------------------------------------
template<typename T>
void XDMFFile::write_mesh_function(const MeshFunction<T>& meshfunction,
                                   std::string format,
                                   XDMFFile::Encoding encoding)
{
  check_encoding(encoding);

  // Get mesh
  dolfin_assert(meshfunction.mesh());
  const Mesh& mesh = *meshfunction.mesh();

#ifdef HAS_HDF5
  if (encoding == XDMFFile::Encoding::HDF5)
  {
    if (hdf5_filemode != "w")
    {
      // Create HDF5 file (truncate)
      hdf5_file.reset(new HDF5File(mesh.mpi_comm(), hdf5_filename, "w"));
      hdf5_filemode = "w";
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
  CellType::Type cell_type = mesh.type().entity_type(cell_dim);

#ifdef HAS_HDF5
  if (encoding == XDMFFile::Encoding::HDF5)
  {
    // Use HDF5 function to output MeshFunction
    const std::string h5_mesh_name = "/Mesh/" + std::to_string(counter);
    boost::filesystem::path p(hdf5_filename);
    current_mesh_name = p.filename().string() + ":" + h5_mesh_name;
    hdf5_file->write(meshfunction, h5_mesh_name);
  }
#endif

  // Saved MeshFunction values are in the /Mesh group
  const std::string dataset_name = current_mesh_name + "/values";

  bool time_series = parameters["time_series"];

  if (MPI::rank(mesh.mpi_comm()) == 0)
  {
    XDMFxml xml(_filename);
    const std::string meshfunction_name = meshfunction.name();
    if (time_series)
      xml.init_timeseries(meshfunction_name, (double)counter, counter);
    else
      xml.init_mesh(meshfunction_name);

    if (encoding == XDMFFile::Encoding::HDF5)
    {
      xml.mesh_topology(cell_type, 1, mesh.size_global(cell_dim),
                        current_mesh_name, xdmf_format_str(encoding));
      xml.mesh_geometry(mesh.size_global(0), mesh.geometry().dim(),
                        current_mesh_name, xdmf_format_str(encoding));
      xml.data_attribute(meshfunction_name, 0, false, mesh.size_global(0),
                         mesh.size_global(cell_dim), 1, dataset_name,
                         xdmf_format_str(encoding));
    }
    else if (encoding == XDMFFile::Encoding::ASCII)
    {
      // Add the mesh topology and geometry to the xml data
      xml.mesh_topology(cell_type, 1, mesh.size_global(cell_dim),
                        generate_xdmf_ascii_mesh_topology_data(mesh, cell_dim),
                        xdmf_format_str(encoding));
      xml.mesh_geometry(mesh.size_global(0), mesh.geometry().dim(),
                        generate_xdmf_ascii_mesh_geometry_data(mesh),
                        xdmf_format_str(encoding));

      std::vector<T> data_values;
      // No duplicates - ignore ghost cells if present
      data_values.assign(meshfunction.values(),
                         meshfunction.values() + mesh.topology().ghost_offset(cell_dim));
      xml.data_attribute(meshfunction_name, 0, false, mesh.size_global(0),
                         mesh.size_global(cell_dim), 1,
                         generate_xdmf_ascii_vertex_data(data_values, format),
                         xdmf_format_str(encoding));
    }
    xml.write();
  }

  ++counter;
}
//-----------------------------------------------------------------------------
XDMFFile::Encoding XDMFFile::get_file_encoding()
{
  XDMFxml xml(_filename);
  xml.read();
  const std::string xml_encoding_attrib = xml.data_encoding();
  return get_file_encoding(xml_encoding_attrib);
}
//-----------------------------------------------------------------------------
XDMFFile::Encoding XDMFFile::get_file_encoding(std::string xml_encoding_attrib)
{
  return (xml_encoding_attrib == "XML") ? XDMFFile::Encoding::ASCII
                                        : XDMFFile::Encoding::HDF5;
}
//-----------------------------------------------------------------------------
