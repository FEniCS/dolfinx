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
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <dolfin/common/MPI.h>
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
#include <dolfin/mesh/Vertex.h>
#include "HDF5File.h"
#include "HDF5Utility.h"
#include "XDMFFile.h"
#include "XDMFxml.h"

using namespace dolfin;

//----------------------------------------------------------------------------
XDMFFile::XDMFFile(MPI_Comm comm, const std::string filename, std::string encoding)
  : GenericFile(filename, "XDMF"), _mpi_comm(comm), _encoding(encoding)
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
  const std::size_t degree = mesh.geometry().degree();

  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();

  const double time_step = ut.second;

  // Geometric and topological dimension
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t tdim = mesh.topology().dim();

  // Get some Function and cell information
  const std::size_t value_rank = u.value_rank();
  const std::size_t value_size = u.value_size();

  // For 2D vectors and tensors, pad out values with zeros
  // to make 3D (XDMF does not support 2D data)
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
    const std::string h5_mesh_name = "/Mesh/" + std::to_string(counter);
    boost::filesystem::path p(hdf5_filename);
    current_mesh_name = p.filename().string() + ":" + h5_mesh_name;
    hdf5_file->write(mesh, h5_mesh_name);
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

  hdf5_file->write_data(dataset_name, data_values, global_size, mpi_io);

  // Flush file. Improves chances of recovering data if
  // interrupted. Also makes file somewhat readable between writes.
  if (parameters["flush_output"])
    hdf5_file->flush();

  // Write the XML meta description (see http://www.xdmf.org) on
  // process zero

  if (MPI::rank(mesh.mpi_comm()) == 0)
  {
    XDMFxml xml(_filename);
    xml.init_timeseries(u.name(), time_step, counter);
    xml.mesh_topology(mesh.type().cell_type(), degree,
                      num_global_cells, current_mesh_name);
    xml.mesh_geometry(num_global_points, gdim, current_mesh_name);

    boost::filesystem::path p(hdf5_filename);
    xml.data_attribute(u.name(), value_rank, vertex_data,
                       num_global_points, num_global_cells,
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

  const std::vector<std::string> topo_name = xml.topology_name();
  const std::vector<std::string> geom_name = xml.geometry_name();
  boost::filesystem::path topo_path(topo_name[0]);
  boost::filesystem::path hdf5_path(hdf5_filename);
  if (topo_path.filename() != hdf5_path.filename() or geom_name[0] != topo_name[0])
  {
    dolfin_error("XDMFFile.cpp",
                 "read XDMF mesh",
                 "Topology and geometry file names do not match");
  }

  // Try to read the mesh from the associated HDF5 file
  hdf5_file->read(mesh, topo_name[1],
                  geom_name[1],
                  topo_name[2], use_partition_from_file);
}
//----------------------------------------------------------------------------
void XDMFFile::write_ascii(const Mesh& mesh)
{
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

  // Write mesh to HDF5 file
  // The XML below will obliterate any existing XDMF file

  const std::string group_name = "/Mesh/" + name;

  // Write the XML meta description on process zero
  if (MPI::rank(mesh.mpi_comm()) == 0)
  {
    XDMFxml xml(_filename);
    xml.init_mesh(name);

    std::ostringstream oss_top;
    const std::size_t num_cell_entities = mesh.type().num_entities(0);
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      const unsigned int* vertices = c->entities(0);
      oss_top << std::endl;
      for (size_t i=0; i<num_cell_entities; ++i)
      {
        oss_top << vertices[i] << " ";
      }
    }
    oss_top << std::endl;

    // Describe topological connectivity
    xml.mesh_topology(mesh.type().cell_type(), mesh.geometry().degree(),
                      num_global_cells, oss_top.str());


    std::ostringstream oss_geo;
    for (VertexIterator v(mesh); !v.end(); ++v)
    {
      oss_geo << std::endl;
      const double* p = v->x();
      for (size_t i=0; i<gdim; ++i)
      {
        oss_geo << boost::str(boost::format("%.15e") % p[i]).c_str() << " ";
      }
    }
    oss_geo << std::endl;

    // Describe geometric coordinates
    xml.mesh_geometry(num_total_points, gdim, oss_geo.str());

    xml.write();
  }
}
//----------------------------------------------------------------------------
void XDMFFile::operator<< (const Mesh& mesh)
{
  if (_encoding == "ascii")
  {
    this->write_ascii(mesh);
    return;
  }

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

  // Get number of global cells and points
  const std::size_t num_global_cells = mesh.size_global(cell_dim);
  std::size_t num_total_points = 0;
  for (std::size_t i = 0; i <= mesh.topology().dim(); ++i)
    num_total_points +=
      mesh.geometry().num_entity_coordinates(i)*mesh.size_global(i);

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
    xml.mesh_topology(mesh.type().cell_type(), mesh.geometry().degree(),
                      num_global_cells, ref);

    // Describe geometric coordinates
    xml.mesh_geometry(num_total_points, gdim, ref);

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
    boost::filesystem::path p(hdf5_filename);
    xml.mesh_geometry(num_global_points, 3,
                      p.filename().string() + ":/Points");

    if(value_size != 0)
    {
      dolfin_assert(value_size == 1 || value_size == 3);
      xml.data_attribute("point_values", 0, true,
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
  const std::string h5_mesh_name = "/Mesh/" + std::to_string(counter);
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
  const std::string data_name = xml.dataname();

  // Try to read the meshfunction from the associated HDF5 file
  hdf5_file->read(meshfunction, "/Mesh/" + data_name);
}
//----------------------------------------------------------------------------
#endif
