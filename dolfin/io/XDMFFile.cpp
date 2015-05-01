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

#ifdef HAS_HDF5

#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <dolfin/common/MPI.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include "HDF5File.h"
#include "HDF5Utility.h"
#include "XDMFFile.h"
#include "XDMFxml.h"

using namespace dolfin;

//----------------------------------------------------------------------------
XDMFFile::XDMFFile(MPI_Comm comm, const std::string filename)
  : GenericFile(filename, "XDMF"), _mpi_comm(comm)
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

  // HDF5 file restart interval. Use 0 to collect all output in one file.
  parameters.add("multi_file", 0);

}
//----------------------------------------------------------------------------
XDMFFile::~XDMFFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void XDMFFile::write_quadratic(const Function& u_geom, const Function& u_val)
{
  // Experimental. For now, just work with one h5 file, cannot do
  // time series. Input two P2 Functions, one containing geometry, the
  // other the values. For a regular mesh,
  // can just interpolate ("x[0]", "x[1]", "x[2]") onto u_geom.

  boost::filesystem::path p(_filename);
  p.replace_extension(".h5");
  hdf5_filename = p.string();

  if (counter == 0)
    // Create new HDF5 file handle (truncate)
    hdf5_file.reset(new HDF5File(_mpi_comm, hdf5_filename, "w"));
  else
    // Create new HDF5 file handle (append)
    hdf5_file.reset(new HDF5File(_mpi_comm, hdf5_filename, "a"));

  hdf5_filemode = "w";

  // Get mesh and dofmap
  dolfin_assert(u_geom.function_space()->mesh());
  dolfin_assert(u_val.function_space()->mesh());
  dolfin_assert (u_val.function_space()->mesh()->id()
              == u_geom.function_space()->mesh()->id());
  const Mesh& mesh = *u_geom.function_space()->mesh();

  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  // FIMXE: Could work in 1D, but not yet tested
  dolfin_assert(tdim == 2 or tdim == 3);

  dolfin_assert(u_geom.function_space()->dofmap());
  const GenericDofMap& geom_dofmap = *u_geom.function_space()->dofmap();

  // Should be vector components on each edge and vertex
  dolfin_assert(geom_dofmap.num_entity_dofs(0) == gdim);
  dolfin_assert(geom_dofmap.num_entity_dofs(1) == gdim);

  // Number of local cells
  const std::size_t n_cells = mesh.topology().ghost_offset(tdim);
  std::vector<dolfin::la_index> cell_topology;

  std::vector<std::size_t> local_to_global_map;
  geom_dofmap.tabulate_local_to_global_dofs(local_to_global_map);

  // Mapping from dofs to XDMF Edge_3, Tri_6 and Tet_10 layout
  std::vector<std::size_t> node_mapping;
  if (tdim == 1)
    node_mapping = {0, 1, 2};
  else if (tdim == 2)
    node_mapping = {0, 1, 2, 5, 3, 4};
  else
    node_mapping = {0, 1, 2, 3, 9, 6, 8, 7, 5, 4};

  // NB relies on the x-component coming first, and ordering xyxy or xyzxyz etc.
  for (std::size_t i = 0; i != n_cells; ++i)
  {
    const dolfin::ArrayView<const dolfin::la_index>& cell_dofs_i = geom_dofmap.cell_dofs(i);
    dolfin_assert(cell_dofs_i.size() == node_mapping.size() * gdim);

    for (auto &node : node_mapping)
    {
      const dolfin::la_index idx = cell_dofs_i[node];
      dolfin_assert(idx < (dolfin::la_index)local_to_global_map.size());
      cell_topology.push_back(local_to_global_map[idx]/gdim);
    }
  }

  const bool mpi_io = MPI::size(_mpi_comm) > 1 ? true : false;

  // Save cell topologies
  std::vector<std::size_t> global_size(2);
  global_size[0] = MPI::sum(_mpi_comm, cell_topology.size()) / node_mapping.size();
  global_size[1] = node_mapping.size();

  const std::string h5_mesh_name = "/Mesh/" + boost::lexical_cast<std::string>(counter);
  current_mesh_name = p.filename().string() + ":" + h5_mesh_name;

  hdf5_file->write_data(h5_mesh_name + "/topology", cell_topology, global_size, mpi_io);

  // Save coordinates
  hdf5_file->write(*u_geom.vector(), h5_mesh_name + "/coordinates");

  // Save values
  const std::string dataset_name = "/Function/"
    + boost::lexical_cast<std::string>(counter) + "/values";
  hdf5_file->write(*u_val.vector(), dataset_name);

  const std::size_t value_rank = u_val.value_rank();
  const std::size_t value_size = u_val.value_size();

  if (MPI::rank(_mpi_comm) == 0)
  {
    XDMFxml xml(_filename);

    const std::size_t num_total_vertices = u_geom.vector()->size()/gdim;
    const std::size_t num_global_cells = global_size[0];

    std::string function_name = u_val.name();
    xml.init_timeseries(function_name, (double)counter, counter);

    boost::filesystem::path p(hdf5_filename);
    const std::string mesh_ref = p.filename().string() + ":" + h5_mesh_name;
    const std::string data_ref = p.filename().string() + ":" + dataset_name;

    xml.mesh_topology(mesh.type().cell_type(), 2, num_global_cells, mesh_ref);
    xml.mesh_geometry(num_total_vertices, gdim, mesh_ref);

    xml.data_attribute(function_name, value_rank, true,
                       num_total_vertices, num_global_cells,
                       value_size, data_ref);
    xml.write();
  }

  ++counter;
  hdf5_file->close();
}
//----------------------------------------------------------------------------
void XDMFFile::operator<< (const Function& u)
{
  std::pair<const Function*, double> ut(&u, (double) counter);
  *this << ut;
}
//----------------------------------------------------------------------------
void XDMFFile::operator<< (const std::pair<const Function*, double> ut)
{
  const int mf_interval = parameters["multi_file"];

  // Conditions for starting a new HDF5 file
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

  // Access Function, Mesh, dofmap  and time step
  dolfin_assert(ut.first);
  const Function& u = *(ut.first);

  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();

  const double time_step = ut.second;

  // Geometric and topological dimension
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t tdim = mesh.topology().dim();

  // Get some Function and cell information
  const std::size_t value_rank = u.value_rank();
  const std::size_t value_size = u.value_size();

  std::size_t padded_value_size = value_size;

  // Test for cell-centred data
  std::size_t cell_based_dim = 1;
  for (std::size_t i = 0; i < value_rank; i++)
    cell_based_dim *= tdim;
  const bool vertex_data = !(dofmap.max_element_dofs() == cell_based_dim);

  // Get number of local/global cells/vertices
  const std::size_t num_local_cells = mesh.topology().ghost_offset(tdim);
  const std::size_t num_local_vertices = mesh.num_vertices();
  const std::size_t num_global_cells = mesh.size_global(tdim);

  // Get Function data at vertices/cell centres
  std::vector<double> data_values;

  if (vertex_data)
  {
    u.compute_vertex_values(data_values, mesh);

    // Interleave the values for vector or tensor fields and pad 2D
    // vectors and tensors to 3D
    if (value_rank > 0)
    {
      if (value_size == 2)
        padded_value_size = 3;
      if (value_size == 4)
        padded_value_size = 9;

      std::vector<double> _data_values(padded_value_size*num_local_vertices,
                                       0.0);
      for (std::size_t i = 0; i < num_local_vertices; i++)
      {
        for (std::size_t j = 0; j < value_size; j++)
        {
          std::size_t tensor_2d_offset = (j > 1 && value_size == 4) ? 1 : 0;
          _data_values[i*padded_value_size + j + tensor_2d_offset]
              = data_values[i + j*num_local_vertices];
        }
      }
      data_values = _data_values;
    }
  }
  else
  {
    dolfin_assert(u.function_space()->dofmap());
    dolfin_assert(u.vector());

    // Allocate memory for function values at cell centres
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

    if (value_size == 2)
      padded_value_size = 3;
    if (value_size == 4)
      padded_value_size = 9;

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
    const std::string h5_mesh_name = "/Mesh/" + boost::lexical_cast<std::string>(counter);
    boost::filesystem::path p(hdf5_filename);
    current_mesh_name = p.filename().string() + ":" + h5_mesh_name;
    hdf5_file->write(mesh, h5_mesh_name);
  }

  // Remove duplicates for vertex-based data
  std::vector<std::size_t> global_size(2);
  global_size[1] = padded_value_size;
  if (vertex_data)
  {
    DistributedMeshTools::reorder_values_by_global_indices(mesh, data_values,
                                                           padded_value_size);
    global_size[0] = mesh.size_global(0);
  }
  else
    global_size[0] = mesh.size_global(tdim);

  // Save data values to HDF5 file.  Vertex/cell values are saved in
  // the hdf5 group /VisualisationVector as distinct from /Vector
  // which is used for solution vectors.
  const std::string dataset_name = "/VisualisationVector/"
    + boost::lexical_cast<std::string>(counter);

  const bool mpi_io = MPI::size(mesh.mpi_comm()) > 1 ? true : false;
  hdf5_file->write_data(dataset_name, data_values, global_size, mpi_io);

  // Flush file. Improves chances of recovering data if
  // interrupted. Also makes file somewhat readable between writes.
  if (parameters["flush_output"])
    hdf5_file->flush();

  // Write the XML meta description (see http://www.xdmf.org) on
  // process zero
  const std::size_t num_total_vertices = mesh.size_global(0);
  if (MPI::rank(mesh.mpi_comm()) == 0)
  {
    XDMFxml xml(_filename);
    xml.init_timeseries(u.name(), time_step, counter);
    xml.mesh_topology(mesh.type().cell_type(), 1, num_global_cells, current_mesh_name);
    xml.mesh_geometry(num_total_vertices, gdim, current_mesh_name);

    boost::filesystem::path p(hdf5_filename);
    xml.data_attribute(u.name(), value_rank, vertex_data,
                       num_total_vertices, num_global_cells,
                       padded_value_size,
                       p.filename().string() + ":" + dataset_name);
    xml.write();
  }

  // Increment counter
  counter++;
}
//----------------------------------------------------------------------------
void XDMFFile::operator>> (Mesh& mesh)
{
  read(mesh, false);
}
//-----------------------------------------------------------------------------
void XDMFFile::read(Mesh& mesh, bool use_partition_from_file)
{
  // Prepare HDF5 file
  if (hdf5_filemode != "r")
  {
    hdf5_file.reset(new HDF5File(_mpi_comm, hdf5_filename, "r"));
    hdf5_filemode = "r";
  }
  dolfin_assert(hdf5_file);

  XDMFxml xml(_filename);
  xml.read();

  // Try to read the mesh from the associated HDF5 file
  hdf5_file->read(mesh, "/Mesh/" + xml.meshname(), use_partition_from_file);
}
//----------------------------------------------------------------------------
void XDMFFile::operator<< (const Mesh& mesh)
{
  // Write Mesh to HDF5 file

  if (hdf5_filemode != "w")
  {
    // Create HDF5 file (truncate)
    hdf5_file.reset(new HDF5File(mesh.mpi_comm(), hdf5_filename, "w"));
    hdf5_filemode = "w";
  }

  // Output data name
  const std::string name = mesh.name();

  // Topological and geometric dimensions
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t cell_dim = mesh.topology().dim();

  // Make sure entities are numbered
  DistributedMeshTools::number_entities(mesh, cell_dim);

  // Get number of global cells and vertices
  const std::size_t num_global_cells   = mesh.size_global(cell_dim);
  const std::size_t num_total_vertices = mesh.size_global(0);

  // Write mesh to HDF5 file
  // The XML below will obliterate any existing XDMF file

  const std::string group_name = "/Mesh/" + name;
  hdf5_file->write(mesh, cell_dim, group_name);

  // Write the XML meta description on process zero
  if (MPI::rank(mesh.mpi_comm()) == 0)
  {
    XDMFxml xml(_filename);
    xml.init_mesh(name);

    boost::filesystem::path p(hdf5_filename);
    const std::string ref = p.filename().string() + ":" + group_name;

    // Describe topological connectivity
    xml.mesh_topology(mesh.type().cell_type(), 1, num_global_cells, ref);

    // Describe geometric coordinates
    xml.mesh_geometry(num_total_vertices, gdim, ref);

    xml.write();
  }
}
//----------------------------------------------------------------------------
void XDMFFile::operator<< (const MeshFunction<bool>& meshfunction)
{
  write_mesh_function(meshfunction);
}
//----------------------------------------------------------------------------
void XDMFFile::operator<< (const MeshFunction<int>& meshfunction)
{
  write_mesh_function(meshfunction);
}
//----------------------------------------------------------------------------
void XDMFFile::operator<< (const MeshFunction<std::size_t>& meshfunction)
{
  write_mesh_function(meshfunction);
}
//----------------------------------------------------------------------------
void XDMFFile::operator<< (const MeshFunction<double>& meshfunction)
{
  write_mesh_function(meshfunction);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const std::vector<Point>& points)
{
  // Initialise HDF5 file
  if (hdf5_filemode != "w")
  {
    // Create HDF5 file (truncate)
    hdf5_file.reset(new HDF5File(_mpi_comm, hdf5_filename, "w"));
    hdf5_filemode = "w";
  }

  // Get number of points (global)
  const std::size_t num_global_points = MPI::sum(_mpi_comm, points.size());

  // Write HDF5 file
  const std::string group_name = "/Points";
  hdf5_file->write(points, group_name);

  // The XML created below will obliterate any existing XDMF file
  write_point_xml(group_name, num_global_points, 0);
}
//----------------------------------------------------------------------------
void XDMFFile::write(const std::vector<Point>& points,
                     const std::vector<double>& values)
{
  // Write clouds of points to XDMF/HDF5 with values

  dolfin_assert(points.size() == values.size());

  // Initialise HDF5 file
  if (hdf5_filemode != "w")
  {
    // Create HDF5 file (truncate)
    hdf5_file.reset(new HDF5File(_mpi_comm, hdf5_filename, "w"));
    hdf5_filemode = "w";
  }

  // Get number of points (global)
  const std::size_t num_global_points = MPI::sum(_mpi_comm, points.size());

  // Write HDF5 file
  const std::string group_name = "/Points";
  hdf5_file->write(points, group_name);

  const std::string values_name = group_name + "/values";
  hdf5_file->write(values, values_name);

  // The XML created will obliterate any existing XDMF file
  write_point_xml(group_name, num_global_points, 1);
}
//----------------------------------------------------------------------------
void XDMFFile::write_point_xml(const std::string group_name,
                               const std::size_t num_global_points,
                               const unsigned int value_size)
{
  // Write the XML meta description on process zero
  if (MPI::rank(_mpi_comm) == 0)
  {
    XDMFxml xml(_filename);
    xml.init_mesh("Point cloud");

    // Point topology, no connectivity data
    xml.mesh_topology(CellType::Type::point, 0, num_global_points, "");

    // Describe geometric coordinates
    // FIXME: assumes 3D
    xml.mesh_geometry(num_global_points, 3, current_mesh_name);

    if(value_size != 0)
    {
      dolfin_assert(value_size == 1 || value_size == 3);

      boost::filesystem::path p(hdf5_filename);
      xml.data_attribute("point_values", 1, true,
                         num_global_points, num_global_points, value_size,
                         p.filename().string() + ":" + group_name + "/values");
    }

    xml.write();
  }
}
//----------------------------------------------------------------------------
template<typename T>
void XDMFFile::write_mesh_function(const MeshFunction<T>& meshfunction)
{
  // Get mesh
  dolfin_assert(meshfunction.mesh());
  const Mesh& mesh = *meshfunction.mesh();

  if (hdf5_filemode != "w")
  {
    // Create HDF5 file (truncate)
    hdf5_file.reset(new HDF5File(mesh.mpi_comm(), hdf5_filename, "w"));
    hdf5_filemode = "w";
  }

  if (meshfunction.size() == 0)
  {
    dolfin_error("XDMFFile.cpp",
                 "save empty MeshFunction",
                 "No values in MeshFunction");
  }

  const std::size_t cell_dim = meshfunction.dim();
  CellType::Type cell_type = mesh.type().entity_type(cell_dim);

  // Use HDF5 function to output MeshFunction
  const std::string h5_mesh_name = "/Mesh/" + boost::lexical_cast<std::string>(counter);
  boost::filesystem::path p(hdf5_filename);
  current_mesh_name = p.filename().string() + ":" + h5_mesh_name;
  hdf5_file->write(meshfunction, h5_mesh_name);

  // Saved MeshFunction values are in the /Mesh group
  const std::string dataset_name = current_mesh_name + "/values";

  if (MPI::rank(mesh.mpi_comm()) == 0)
  {
    XDMFxml xml(_filename);
    const std::string meshfunction_name = meshfunction.name();
    xml.init_timeseries(meshfunction_name, (double)counter, counter);
    xml.mesh_topology(cell_type, 1, mesh.size_global(cell_dim),
                      current_mesh_name);
    xml.mesh_geometry(mesh.size_global(0), mesh.geometry().dim(),
                      current_mesh_name);
    xml.data_attribute(meshfunction_name, 0, false, mesh.size_global(0),
                       mesh.size_global(cell_dim), 1, dataset_name);
    xml.write();
  }

  counter++;
}
//----------------------------------------------------------------------------
void XDMFFile::operator>> (MeshFunction<bool>& meshfunction)
{
  const Mesh& mesh = *meshfunction.mesh();
  const std::size_t cell_dim = meshfunction.dim();

  MeshFunction<std::size_t> mf(mesh, cell_dim);
  read_mesh_function(mf);

  for (MeshEntityIterator cell(mesh, cell_dim); !cell.end(); ++cell)
    meshfunction[cell->index()] = (mf[cell->index()] == 1);
}
//----------------------------------------------------------------------------
void XDMFFile::operator>> (MeshFunction<int>& meshfunction)
{
  read_mesh_function(meshfunction);
}
//----------------------------------------------------------------------------
void XDMFFile::operator>> (MeshFunction<std::size_t>& meshfunction)
{
  read_mesh_function(meshfunction);
}
//----------------------------------------------------------------------------
void XDMFFile::operator>> (MeshFunction<double>& meshfunction)
{
  read_mesh_function(meshfunction);
}
//----------------------------------------------------------------------------
template<typename T>
void XDMFFile::read_mesh_function(MeshFunction<T>& meshfunction)
{
  if (hdf5_filemode != "r")
  {
    hdf5_file.reset(new HDF5File(_mpi_comm, hdf5_filename, "r"));
    hdf5_filemode = "r";
  }

  dolfin_assert(hdf5_file);

  XDMFxml xml(_filename);
  xml.read();
  const std::string mesh_name = xml.meshname();
  const std::string data_name = xml.dataname();

  if (mesh_name != data_name)
    dolfin_error("XMDFFile.cpp",
                 "read MeshFunction",
                 "Data and Mesh names do not match in XDMF");

  // Try to read the meshfunction from the associated HDF5 file
  hdf5_file->read(meshfunction, "/Mesh/" + mesh_name);
}
//----------------------------------------------------------------------------
#endif
