// Copyright (C) 2012 Chris N. Richardson and Garth N. Wells
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
//
// First added:  2012-05-28
// Last changed: 2013-03-01

#ifdef HAS_HDF5

#include <ostream>
#include <sstream>
#include <vector>
#include <boost/assign.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include "pugixml.hpp"

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include "HDF5File.h"
#include "HDF5Interface.h"
#include "XDMFFile.h"

using namespace dolfin;

//----------------------------------------------------------------------------
XDMFFile::XDMFFile(const std::string filename) : GenericFile(filename, "XDMF")
{
  // Make name for HDF5 file (used to store data)
  boost::filesystem::path p(filename);
  p.replace_extension(".h5");
  hdf5_filename = p.string();

  // Create HDF5 file (truncate)
  hdf5_file.reset(new HDF5File(hdf5_filename, "w"));

  // Rewrite the mesh at every time step in a time series
  // Should be turned off if the mesh remains constant
  parameters.add("rewrite_function_mesh", true);

  // Flush datasets to disk at each timestep
  // Allows inspection of the HDF5 file whilst running, at some performance cost
  parameters.add("flush_output", false);

}
//----------------------------------------------------------------------------
XDMFFile::~XDMFFile()
{
  // Do nothing
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
  dolfin_assert(ut.first);
  const Function& u = *(ut.first);
  const double time_step = ut.second;

  // Update any ghost values
  u.update();

  // Get Mesh object
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  // Geometric dimension
  const std::size_t gdim = mesh.geometry().dim();

  // Get DOF map
  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();

  // Get some Function and cell information
  const std::size_t value_rank = u.value_rank();
  const std::size_t value_size = u.value_size();
  const std::size_t cell_dim = mesh.topology().dim();
  std::size_t padded_value_size = value_size;

  // Test for cell-centred data
  std::size_t cell_based_dim = 1;
  for (std::size_t i = 0; i < value_rank; i++)
    cell_based_dim *= cell_dim;
  const bool vertex_data = !(dofmap.max_cell_dimension() == cell_based_dim);

  // Get number of local/global cells/vertices

  const std::size_t num_local_cells = mesh.num_cells();
  const std::size_t num_local_vertices = mesh.num_vertices();
  const std::size_t num_global_cells = MPI::sum(num_local_cells);
  std::size_t num_total_vertices = MPI::sum(num_local_vertices);

  // Get Function data at vertices/cell centres
  std::vector<double> data_values;

  std::size_t num_local_entities = 0;
  std::size_t num_total_entities = 0;
  if (vertex_data)
  {
    num_local_entities = num_local_vertices; // includes duplicates
    num_total_entities = num_total_vertices;
    u.compute_vertex_values(data_values, mesh);

    // Interleave the values for vector or tensor fields and pad 2D
    // vectors and tensors to 3D
    if (value_rank > 0)
    {
      if (value_size == 2)
        padded_value_size = 3;
      if (value_size == 4)
        padded_value_size = 9;

      std::vector<double> _data_values(padded_value_size*num_local_entities, 0.0);
      for(std::size_t i = 0; i < num_local_entities; i++)
      {
        for (std::size_t j = 0; j < value_size; j++)
        {
          std::size_t tensor_2d_offset = (j > 1 && value_size == 4) ? 1 : 0;
          _data_values[i*padded_value_size + j + tensor_2d_offset]
              = data_values[i + j*num_local_entities];
        }
      }
      data_values = _data_values;
    }
  }
  else
  {
    num_local_entities = num_local_cells;
    num_total_entities = num_global_cells;
    dolfin_assert(u.function_space()->dofmap());
    const GenericDofMap& dofmap = *u.function_space()->dofmap();
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
      const std::vector<dolfin::la_index>& dofs = dofmap.cell_dofs(cell->index());
      for(std::size_t i = 0; i < dofmap.cell_dimension(cell->index()); ++i)
        dof_set.push_back(dofs[i]);

      // Add local dimension to cell offset and increment
      *(cell_offset + 1) = *(cell_offset) + dofmap.cell_dimension(cell->index());
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
    std::vector<double> _data_values(padded_value_size*num_local_entities, 0.0);
    std::size_t count = 0;
    if (value_rank == 1 && value_size == 2)
    {
      for (CellIterator cell(mesh); !cell.end(); ++cell)
      {
        _data_values[count++] = data_values[*cell_offset];
        _data_values[count++] = data_values[*cell_offset + 1];
        count++;
      }
      ++cell_offset;
    }
    else if (value_rank == 2 && value_size == 4)
    {
      // Pad with 0.0 to 2D tensors to make them 3D
      for (CellIterator cell(mesh); !cell.end(); ++cell)
      {
        for(std::size_t i = 0; i < 2; i++)
        {
          //          cout << "test: " << *cell_offset + 2*i << ", " << *cell_offset + 2*i + 1
          //   << ", " << data_values[*cell_offset + 2*i] << ", " << data_values[*cell_offset + 2*i + 1] << endl;
          _data_values[count++] = data_values[*cell_offset + 2*i];
          _data_values[count++] = data_values[*cell_offset + 2*i + 1];
          count++;
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
      current_mesh_name = boost::lexical_cast<std::string>(counter);
      hdf5_file->write(mesh, current_mesh_name);
  }

  // Vertex/cell values are saved in the hdf5 group /VisualisationVector
  // as distinct from /Vector which is used for solution vectors.

  // Save data values to HDF5 file

  std::vector<std::size_t> global_size(2);
  global_size[0] = num_total_entities;
  global_size[1] = padded_value_size;

  if(vertex_data)
  {
    hdf5_file->reorder_values_by_global_indices(mesh, data_values, global_size);
    num_total_vertices = global_size[0];
  }

  const std::string dataset_name = "/VisualisationVector/" 
    + boost::lexical_cast<std::string>(counter);

  hdf5_file->write_data(dataset_name, data_values, global_size);

  // Flush file. Improves chances of recovering data if interrupted. Also
  // makes file somewhat readable between writes.
  if(parameters["flush_output"])
    hdf5_file->flush();

  // Write the XML meta description (see http://www.xdmf.org) on process zero
  if (MPI::process_number() == 0)
  {
    output_xml(time_step, vertex_data,
               cell_dim, num_global_cells, gdim, num_total_vertices,
               value_rank, padded_value_size,
               u.name(), dataset_name);
  }

  // Increment counter
  counter++;
}
//----------------------------------------------------------------------------
void XDMFFile::operator<< (const Mesh& mesh)
{
  // Write Mesh to HDF5 file (use contiguous vertex indices for topology)
  dolfin_assert(hdf5_file);

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

  hdf5_file->write(mesh, cell_dim, name);

  // FIXME: Names should be returned by HDF5::write_mesh
  // Mesh data set names
  const std::string mesh_topology_name = "/Mesh/" + name + "/topology";
  const std::string mesh_coords_name = "/Mesh/" + name + "/coordinates";

  // Write the XML meta description on process zero
  if (MPI::process_number() == 0)
  {
    // Create XML document
    pugi::xml_document xml_doc;

    // XML headers
    xml_doc.append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf = xml_doc.append_child("Xdmf");
    xdmf.append_attribute("Version") = "2.0";
    xdmf.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";
    pugi::xml_node xdmf_domain = xdmf.append_child("Domain");
    pugi::xml_node xdmf_grid = xdmf_domain.append_child("Grid");
    xdmf_grid.append_attribute("Name") = "dolfin_mesh";
    xdmf_grid.append_attribute("GridType") = "Uniform";

    // Describe topological connectivity
    pugi::xml_node xdmf_topology = xdmf_grid.append_child("Topology");
    xml_mesh_topology(xdmf_topology, cell_dim, num_global_cells,
                      mesh_topology_name);

    // Describe geometric coordinates
    pugi::xml_node xdmf_geometry = xdmf_grid.append_child("Geometry");
    xml_mesh_geometry(xdmf_geometry, num_total_vertices, gdim,
                      mesh_coords_name);

    xml_doc.save_file(filename.c_str(), "  ");
  }

}
//----------------------------------------------------------------------------
void XDMFFile::operator<< (const MeshFunction<bool>& meshfunction)
{
  const Mesh& mesh = meshfunction.mesh();
  const std::size_t cell_dim = meshfunction.dim();

  // HDF5 does not support a boolean type,
  // so copy to a std::size_t with values 1 and 0
  MeshFunction<std::size_t> mf(mesh, cell_dim);
  for (MeshEntityIterator cell(mesh, cell_dim); !cell.end(); ++cell)
    mf[cell->index()] = (meshfunction[cell->index()] ? 1 : 0);

  write_mesh_function(mf);
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
template<typename T>
void XDMFFile::write_mesh_function(const MeshFunction<T>& meshfunction)
{
  dolfin_assert(hdf5_file);

  if (meshfunction.size() == 0)
  {
    dolfin_error("XDMFFile.cpp",
                 "save empty MeshFunction",
                 "No values in MeshFunction");
  }

  // Get mesh
  const Mesh& mesh = meshfunction.mesh();

  const std::size_t cell_dim = meshfunction.dim();
  dolfin_assert(cell_dim <= mesh.topology().dim());

  // // Collate data in a vector
  // std::vector<T> data_values;

  // // If not already numbered, number entities of order cell_dim
  // // so we can get shared_entities and correct size_global(cell_dim)
  // DistributedMeshTools::number_entities(mesh, cell_dim);


  current_mesh_name = boost::lexical_cast<std::string>(counter);
  hdf5_file->write_mesh_function(meshfunction, current_mesh_name);
  

  // hdf5_file->write(mesh, cell_dim, current_mesh_name);

  // if(cell_dim == mesh.topology().dim() || MPI::num_processes() == 1)
  // {
  //   // No duplicates
  //   data_values.assign(meshfunction.values(), meshfunction.values() + meshfunction.size());
  // }
  // else
  // {
  //   data_values.reserve(mesh.size(cell_dim));

  //   // Drop duplicate data
  //   const std::size_t my_rank = MPI::process_number();
  //   const std::map<unsigned int, std::set<unsigned int> >& shared_entities
  //     = mesh.topology().shared_entities(cell_dim);

  //   for(std::size_t i = 0; i < meshfunction.size(); ++i)
  //   {
  //     std::map<unsigned int, std::set<unsigned int> >::const_iterator sh
  //       = shared_entities.find(i);

  //     // If unshared, or shared and locally owned, append to vector
  //     if(sh == shared_entities.end())
  //       data_values.push_back(meshfunction[i]);
  //     else
  //     {
  //       std::set<unsigned int>::iterator lowest_proc = sh->second.begin();
  //       if(*lowest_proc > my_rank)
  //         data_values.push_back(meshfunction[i]);
  //     }
  //   }
  // }

  // // Write values to HDF5
  // std::vector<std::size_t> global_size(1, MPI::sum(data_values.size()));

  // Save MeshFunction values in the /Mesh group
  const std::string dataset_name = "/Mesh/" + current_mesh_name + "/values";
  
  // hdf5_file->write_data(dataset_name, data_values, global_size);

  // Write the XML meta description (see http://www.xdmf.org) on process zero
  if (MPI::process_number() == 0)
  {
    output_xml((double)counter, false,
               cell_dim, mesh.size_global(cell_dim),
               mesh.geometry().dim(), mesh.size_global(0),
               0, 1, meshfunction.name(), dataset_name);
  }

  counter++;
}
//----------------------------------------------------------------------------
void XDMFFile::xml_mesh_topology(pugi::xml_node &xdmf_topology,
                                 const std::size_t cell_dim,
                                 const std::size_t num_global_cells,
                                 const std::string topology_dataset_name) const
{
  xdmf_topology.append_attribute("NumberOfElements") = (unsigned int) num_global_cells;

  // Cell type
  if (cell_dim == 0)
  {
    xdmf_topology.append_attribute("TopologyType") = "PolyVertex";
    xdmf_topology.append_attribute("NodesPerElement") = "1";
  }
  else if (cell_dim == 1)
  {
    xdmf_topology.append_attribute("TopologyType") = "PolyLine";
    xdmf_topology.append_attribute("NodesPerElement") = "2";
  }
  else if (cell_dim == 2)
    xdmf_topology.append_attribute("TopologyType") = "Triangle";
  else if (cell_dim == 3)
    xdmf_topology.append_attribute("TopologyType") = "Tetrahedron";

  // Refer to all cells and dimensions
  pugi::xml_node xdmf_topology_data = xdmf_topology.append_child("DataItem");
  xdmf_topology_data.append_attribute("Format") = "HDF";
  const std::string cell_dims = boost::lexical_cast<std::string>(num_global_cells)
    + " " + boost::lexical_cast<std::string>(cell_dim + 1);
  xdmf_topology_data.append_attribute("Dimensions") = cell_dims.c_str();

  // For XDMF file need to remove path from filename so that xdmf
  // filenames such as "results/data.xdmf" correctly index h5 files in
  // the same directory
  boost::filesystem::path p(hdf5_filename);
  std::string topology_reference = p.filename().string() + ":" + topology_dataset_name;
  xdmf_topology_data.append_child(pugi::node_pcdata).set_value(topology_reference.c_str());
}
//----------------------------------------------------------------------------
void XDMFFile::xml_mesh_geometry(pugi::xml_node& xdmf_geometry,
                                 const std::size_t num_total_vertices,
                                 const std::size_t gdim,
                                 const std::string geometry_dataset_name) const
{
  dolfin_assert(0 < gdim && gdim <= 3);
  std::string geometry_type;
  if (gdim == 1)
  {
    //    dolfin_error("XDMFFile.cpp",
    //                 "write 1D mesh",
    //                 "One dimensional geometry not supported in XDMF");
    // FIXME: geometry "X" is not supported
    // This could be fixed by padding vertex coordinates to 2D and using "XY"
    geometry_type = "X_Y_Z";
  }
  else if (gdim == 2)
    geometry_type = "XY";
  else if (gdim == 3)
    geometry_type = "XYZ";

  xdmf_geometry.append_attribute("GeometryType") = geometry_type.c_str();
  pugi::xml_node xdmf_geom_data = xdmf_geometry.append_child("DataItem");

  xdmf_geom_data.append_attribute("Format") = "HDF";
  std::string geom_dim = boost::lexical_cast<std::string>(num_total_vertices)
    + " " + boost::lexical_cast<std::string>(gdim);
  xdmf_geom_data.append_attribute("Dimensions") = geom_dim.c_str();

  // FIXME: improve this workaround
  // When gdim==1, XDMF does not support a 1D geometry "X",
  // so need to provide some dummy Y and Z values.
  // Using the "X_Y_Z" geometry the Y and Z values can be supplied
  // as separate datasets, here in plain text (though it could be done in HDF5 too).
  
  if(gdim == 1)
  {
    std::string dummy_zeros;
    dummy_zeros.reserve(2*num_total_vertices);
    for(std::size_t i = 0; i < num_total_vertices; ++i)
      dummy_zeros += "0 ";

    pugi::xml_node xdmf_geom_1 = xdmf_geometry.append_child("DataItem");
    xdmf_geom_1.append_attribute("Format") = "XML";
    geom_dim = boost::lexical_cast<std::string>(num_total_vertices) + " 1" ;
    xdmf_geom_1.append_attribute("Dimensions") = geom_dim.c_str();
    xdmf_geom_1.append_child(pugi::node_pcdata).set_value(dummy_zeros.c_str());

    pugi::xml_node xdmf_geom_2 = xdmf_geometry.append_child("DataItem");
    xdmf_geom_2.append_attribute("Format") = "XML";
    geom_dim = boost::lexical_cast<std::string>(num_total_vertices) + " 1" ;
    xdmf_geom_2.append_attribute("Dimensions") = geom_dim.c_str();
    xdmf_geom_2.append_child(pugi::node_pcdata).set_value(dummy_zeros.c_str());

  }
  

  boost::filesystem::path p(hdf5_filename);
  const std::string geometry_reference
    = p.filename().string() + ":" + geometry_dataset_name;
  xdmf_geom_data.append_child(pugi::node_pcdata).set_value(geometry_reference.c_str());
}
//----------------------------------------------------------------------------
void XDMFFile::output_xml(const double time_step, const bool vertex_data,
                          const std::size_t cell_dim, const std::size_t num_global_cells,
                          const std::size_t gdim, const std::size_t num_total_vertices,
                          const std::size_t value_rank, const std::size_t padded_value_size,
                          const std::string name, const std::string dataset_name) const
{
  // Working data structure for formatting XML file
  std::string s;
  pugi::xml_document xml_doc;
  pugi::xml_node xdmf_timegrid;
  pugi::xml_node xdmf_timedata;

  if (counter == 0)
  {
    // First time step - create document template, adding a mesh and
    // an empty time-series
    xml_doc.append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf = xml_doc.append_child("Xdmf");
    xdmf.append_attribute("Version") = "2.0";
    xdmf.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";
    pugi::xml_node xdmf_domain = xdmf.append_child("Domain");

    //  /Xdmf/Domain/Grid - actually a TimeSeries, not a spatial grid
    xdmf_timegrid = xdmf_domain.append_child("Grid");
    xdmf_timegrid.append_attribute("Name") = "TimeSeries";
    xdmf_timegrid.append_attribute("GridType") = "Collection";
    xdmf_timegrid.append_attribute("CollectionType") = "Temporal";

    //  /Xdmf/Domain/Grid/Time
    pugi::xml_node xdmf_time = xdmf_timegrid.append_child("Time");
    xdmf_time.append_attribute("TimeType") = "List";
    xdmf_timedata = xdmf_time.append_child("DataItem");
    xdmf_timedata.append_attribute("Format") = "XML";
    xdmf_timedata.append_attribute("Dimensions") = "1";
    xdmf_timedata.append_child(pugi::node_pcdata);
  }
  else
  {
    // Subsequent timestep - read in existing XDMF file
    pugi::xml_parse_result result = xml_doc.load_file(filename.c_str());
    if (!result)
    {
      dolfin_error("XDMFFile.cpp",
                   "write data to XDMF file",
                   "XML parsing error when reading from existing file");
    }

    // Get data node
    xdmf_timegrid = xml_doc.child("Xdmf").child("Domain").child("Grid");
    dolfin_assert(xdmf_timegrid);

    // Get time series node
    xdmf_timedata = xdmf_timegrid.child("Time").child("DataItem");
    dolfin_assert(xdmf_timedata);
  }

  //  Add a time step to the TimeSeries List
  xdmf_timedata.attribute("Dimensions").set_value(static_cast<unsigned int>(counter + 1));
  s = boost::lexical_cast<std::string>(xdmf_timedata.first_child().value())
    + " " + boost::str((boost::format("%d") % time_step));
  xdmf_timedata.first_child().set_value(s.c_str());

  //   /Xdmf/Domain/Grid/Grid - the actual data for this timestep
  pugi::xml_node xdmf_grid = xdmf_timegrid.append_child("Grid");
  s = name + "_" + boost::lexical_cast<std::string>(counter);
  xdmf_grid.append_attribute("Name") = s.c_str();
  xdmf_grid.append_attribute("GridType") = "Uniform";

  // Grid/Topology
  pugi::xml_node xdmf_topology = xdmf_grid.append_child("Topology");
  xml_mesh_topology(xdmf_topology, cell_dim, num_global_cells,
                    "/Mesh/" + current_mesh_name + "/topology");

  // Grid/Geometry
  pugi::xml_node xdmf_geometry = xdmf_grid.append_child("Geometry");
  xml_mesh_geometry(xdmf_geometry, num_total_vertices, gdim,
                    "/Mesh/" + current_mesh_name + "/coordinates");

  // Grid/Attribute (Function value data)
  pugi::xml_node xdmf_values = xdmf_grid.append_child("Attribute");
  xdmf_values.append_attribute("Name") = name.c_str();

    if (value_rank == 0)
      xdmf_values.append_attribute("AttributeType") = "Scalar";
    else if (value_rank == 1)
      xdmf_values.append_attribute("AttributeType") = "Vector";
    else if (value_rank == 2)
      xdmf_values.append_attribute("AttributeType") = "Tensor";

    if (vertex_data)
      xdmf_values.append_attribute("Center") = "Node";
    else
      xdmf_values.append_attribute("Center") = "Cell";

    pugi::xml_node xdmf_data = xdmf_values.append_child("DataItem");
    xdmf_data.append_attribute("Format") = "HDF";

    const std::size_t num_total_entities = vertex_data ? num_total_vertices : num_global_cells;

    s = boost::lexical_cast<std::string>(num_total_entities) + " "
      + boost::lexical_cast<std::string>(padded_value_size);

    xdmf_data.append_attribute("Dimensions") = s.c_str();

    boost::filesystem::path p(hdf5_filename);
    s = p.filename().string() + ":" + dataset_name;
    xdmf_data.append_child(pugi::node_pcdata).set_value(s.c_str());

    // Write XML file
    xml_doc.save_file(filename.c_str(), "  ");
  }
//----------------------------------------------------------------------------
#endif
