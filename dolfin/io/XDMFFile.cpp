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
// Last changed: 2012-09-18

#ifdef HAS_HDF5

#include <ostream>
#include <sstream>
#include <vector>
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
#include <dolfin/common/Timer.h>
#include "HDF5File.h"
#include "XDMFFile.h"

using namespace dolfin;

//----------------------------------------------------------------------------
XDMFFile::XDMFFile(const std::string filename) : GenericFile(filename, "XDMF")
{
  // Do nothing

  // Name of HDF5 file
  boost::filesystem::path p(filename);
  p.replace_extension(".h5");

  // Create HDF5 file
  hdf5_file.reset(new HDF5File(p.string()));
  dolfin_assert(hdf5_file);
  hdf5_file->create();
}
//----------------------------------------------------------------------------
XDMFFile::~XDMFFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void XDMFFile::operator<<(const Function& u)
{
  std::pair<const Function*, double> ut(&u, (double) counter);
  *this << ut;
}
//----------------------------------------------------------------------------
void XDMFFile::operator<<(const std::pair<const Function*, double> ut)
{
  dolfin_assert(ut.first);
  const Function &u = *(ut.first);
  const double time_step = ut.second;

  Timer hdf5timer("Write XDMF (mesh + data) vertex");

  // Update any ghost values
  u.update();

  // Get Mesh object
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  // Get DOF map
  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();

  const uint value_rank = u.value_rank();
  const uint value_size = u.value_size();
  const uint cell_dim = mesh.topology().dim();
  uint value_size_io = value_size; // output size may be padded 2D -> 3D for paraview

  // Throw error for higher rank tensors
  if (value_rank > 2)
  {
    dolfin_error("XDMFFile.cpp",
                 "write data to XDMF file",
                 "Output of tensors with rank > 2 not yet supported");
  }

  // Throw error for 1D data sets
  if (cell_dim == 1)
  {
    dolfin_error("XDMFFile.cpp",
                 "write data to XDMF file",
                 "Output of 1D datasets not supported");
  }

  // Test for cell-centred data
  uint cell_based_dim = 1;
  for (uint i = 0; i < value_rank; i++)
    cell_based_dim *= cell_dim;
  const bool vertex_data = !(dofmap.max_cell_dimension() == cell_based_dim);

  // Get number of local/global cells/vertices

  // At present, each process saves its local vertices
  // sequentially in the HDF5 file, resulting in some duplication

  const uint num_local_cells = mesh.num_cells();
  const uint num_local_vertices = mesh.num_vertices();
  const uint num_all_local_vertices = MPI::sum(num_local_vertices);
  const uint num_global_cells = MPI::sum(num_local_cells);

  // Get Function data at vertices/cell centres
  std::vector<double> data_values;
  uint num_local_entities = 0;
  if (vertex_data)
  {
    num_local_entities = num_local_vertices;
    u.compute_vertex_values(data_values, mesh);
  }
  else
  {
    num_local_entities = num_local_cells;
    const GenericVector& v = *u.vector();
    v.get_local(data_values);
  }


  // Interleave the values for vector or tensor fields and pad 2D vectors
  // and tensors to 3D
  if (value_rank > 0)
  {
    if (value_size == 2)
      value_size_io = 3;
    if (value_size == 4)
      value_size_io = 9;

    std::vector<double> tmp;
    tmp.reserve(value_size*num_local_entities);
    for(uint i = 0; i < num_local_entities; i++)
    {
      for (uint j = 0; j < value_size; j++)
      {
        tmp.push_back(data_values[i + j*num_local_entities]);
        if (j == 1 && value_size == 4)
          tmp.push_back(0.0);
      }
      if (value_size == 2)    // 2D -> 3D vector
        tmp.push_back(0.0);
      if (value_size == 4)    // 2D -> 3D tensor
        tmp.insert(tmp.end(), 4, 0.0);
    }
    data_values.resize(tmp.size()); // 2D->3D padding increases size
    std::copy(tmp.begin(), tmp.end(), data_values.begin());
  }

  // Get names of mesh data sets used in the HDF5 file
  dolfin_assert(hdf5_file);
  const std::string mesh_coords_name = hdf5_file->mesh_coords_dataset_name(mesh);
  const std::string mesh_topology_name = hdf5_file->mesh_topo_dataset_name(mesh);

  // Write mesh to HDF5 file
  if (counter == 0 )
    hdf5_file->write_mesh(mesh, false);
  else if (!hdf5_file->dataset_exists(mesh_coords_name)
      || !hdf5_file->dataset_exists(mesh_topology_name))
  {
    hdf5_file->write_mesh(mesh, false);
  }

  // Working data structure for formatting XML file
  std::string s;

  // Vertex/cell values are saved in the hdf5 group /DataVector
  // as distinct from /Vector which is used for solution vectors.

  // Save data values to HDF5 file
  s = "/DataVector/" + boost::lexical_cast<std::string>(counter);
  hdf5_file->write(data_values, s.c_str(), value_size_io);

  // Write the XML meta description (see http://www.xdmf.org) on process 0
  if (MPI::process_number() == 0)
  {
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
      pugi::xml_node xdmf_domn = xdmf.append_child("Domain");

      //  /Xdmf/Domain/Grid - actually a TimeSeries, not a spatial grid
      xdmf_timegrid = xdmf_domn.append_child("Grid");
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
    xdmf_timedata.attribute("Dimensions").set_value(counter + 1);
    s = boost::lexical_cast<std::string>(xdmf_timedata.first_child().value()) + " "
          + boost::str((boost::format("%d") % time_step));
    xdmf_timedata.first_child().set_value(s.c_str());

    //   /Xdmf/Domain/Grid/Grid - the actual data for this timestep
    pugi::xml_node xdmf_grid = xdmf_timegrid.append_child("Grid");
    s = u.name() + "_" + boost::lexical_cast<std::string>(counter);
    xdmf_grid.append_attribute("Name") = s.c_str();
    xdmf_grid.append_attribute("GridType") = "Uniform";

    // Mesh is referenced in XDMF separately at each timestep.
    // Any changes to the mesh will be reflected in the hashes,
    // which will result in different dataset names, thus supporting
    // time-varying meshes.

    // Grid/Topology
    pugi::xml_node xdmf_topology = xdmf_grid.append_child("Topology");
    if (cell_dim == 2)
      xdmf_topology.append_attribute("TopologyType") = "Triangle";
    else if (cell_dim == 3)
      xdmf_topology.append_attribute("TopologyType") = "Tetrahedron";

    xdmf_topology.append_attribute("NumberOfElements") = num_global_cells;
    pugi::xml_node xdmf_topology_data = xdmf_topology.append_child("DataItem");

    xdmf_topology_data.append_attribute("Format") = "HDF";
    s = boost::lexical_cast<std::string>(num_global_cells) + " "
          + boost::lexical_cast<std::string>(cell_dim + 1);
    xdmf_topology_data.append_attribute("Dimensions") = s.c_str();

    // Need to remove path from filename
    // so that xdmf filenames such as "results/data.xdmf" correctly
    // index h5 files in the same directory
    boost::filesystem::path p(hdf5_file->filename);
    std::string hdf5_short_filename = p.filename().string();
    s = hdf5_short_filename + ":" + mesh_topology_name;
    xdmf_topology_data.append_child(pugi::node_pcdata).set_value(s.c_str());

    // Grid/Geometry
    pugi::xml_node xdmf_geom = xdmf_grid.append_child("Geometry");
    xdmf_geom.append_attribute("GeometryType") = "XYZ";
    pugi::xml_node xdmf_geom_data = xdmf_geom.append_child("DataItem");

    xdmf_geom_data.append_attribute("Format")="HDF";
    s = boost::lexical_cast<std::string>(num_all_local_vertices) + " 3";
    xdmf_geom_data.append_attribute("Dimensions") = s.c_str();

    s = hdf5_short_filename + ":" + mesh_coords_name;
    xdmf_geom_data.append_child(pugi::node_pcdata).set_value(s.c_str());

    // Grid/Attribute (Function value data)
    pugi::xml_node xdmf_vals = xdmf_grid.append_child("Attribute");
    xdmf_vals.append_attribute("Name") = u.name().c_str();

    if (value_rank == 0)
      xdmf_vals.append_attribute("AttributeType") = "Scalar";
    else if (value_rank == 1)
      xdmf_vals.append_attribute("AttributeType") = "Vector";
    else if (value_rank == 2)
      xdmf_vals.append_attribute("AttributeType") = "Tensor";

    if (vertex_data)
      xdmf_vals.append_attribute("Center") = "Node";
    else
      xdmf_vals.append_attribute("Center") = "Cell";

    pugi::xml_node xdmf_data = xdmf_vals.append_child("DataItem");
    xdmf_data.append_attribute("Format") = "HDF";
    if(vertex_data)
    {
      s = boost::lexical_cast<std::string>(num_all_local_vertices) + " "
          + boost::lexical_cast<std::string>(value_size_io);
    }
    else
    {
      s = boost::lexical_cast<std::string>(num_global_cells) + " "
          + boost::lexical_cast<std::string>(value_size_io);
    }
    xdmf_data.append_attribute("Dimensions") = s.c_str();

    s = hdf5_short_filename + ":/DataVector/" + boost::lexical_cast<std::string>(counter);
    xdmf_data.append_child(pugi::node_pcdata).set_value(s.c_str());

    // Write XML file
    xml_doc.save_file(filename.c_str(), "    ");
  }

  // Increment counter
  counter++;
}
//----------------------------------------------------------------------------
void XDMFFile::operator<<(const Mesh& mesh)
{
  Timer hdf5timer("HDF5 + XDMF Output (mesh)");

  // Write Mesh to HDF5 file (use contiguous vertex indices for topology)
  dolfin_assert(hdf5_file);
  hdf5_file->create();
  hdf5_file->write_mesh(mesh, false);

  // Get number of local/global cells/vertices
  const uint num_local_cells = mesh.num_cells();
  const uint num_local_vertices = mesh.num_vertices();
  const uint num_all_local_vertices = MPI::sum(num_local_vertices);
  const uint num_global_cells = MPI::sum(num_local_cells);

  // MPI collective calls
  boost::filesystem::path p(hdf5_file->filename);
  const std::string hdf5_short_filename = p.filename().string();
  const std::string topology_hash_name
      = hdf5_short_filename + ":" + hdf5_file->mesh_topo_dataset_name(mesh);
  const std::string coords_hash_name
      = hdf5_short_filename + ":" + hdf5_file->mesh_coords_dataset_name(mesh);

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
    pugi::xml_node xdmf_domn = xdmf.append_child("Domain");
    pugi::xml_node xdmf_grid = xdmf_domn.append_child("Grid");
    xdmf_grid.append_attribute("Name") = "dolfin_grid";
    xdmf_grid.append_attribute("GridType") = "Uniform";

    pugi::xml_node xdmf_topology = xdmf_grid.append_child("Topology");
    xdmf_topology.append_attribute("NumberOfElements") = num_global_cells;

    // Cell type
    const uint cell_dim = mesh.topology().dim();
    if (cell_dim == 2)
      xdmf_topology.append_attribute("TopologyType") = "Triangle";
    else if (cell_dim == 3)
      xdmf_topology.append_attribute("TopologyType") = "Tetrahedron";

    pugi::xml_node xdmf_topology_data = xdmf_topology.append_child("DataItem");

    xdmf_topology_data.append_attribute("Format") = "HDF";
    const std::string cell_dims = boost::lexical_cast<std::string>(num_global_cells)
          + " " + boost::lexical_cast<std::string>(cell_dim + 1);
    xdmf_topology_data.append_attribute("Dimensions") = cell_dims.c_str();

    xdmf_topology_data.append_child(pugi::node_pcdata).set_value(topology_hash_name.c_str());

    pugi::xml_node xdmf_geom = xdmf_grid.append_child("Geometry");
    xdmf_geom.append_attribute("GeometryType") = "XYZ";
    pugi::xml_node xdmf_geom_data = xdmf_geom.append_child("DataItem");

    xdmf_geom_data.append_attribute("Format") = "HDF";
    const std::string vertices_dims = boost::lexical_cast<std::string>(num_all_local_vertices) + " 3";
    xdmf_geom_data.append_attribute("Dimensions") = vertices_dims.c_str();

    xdmf_geom_data.append_child(pugi::node_pcdata).set_value(coords_hash_name.c_str());

    xml_doc.save_file(filename.c_str(), "    ");
  }
}
//----------------------------------------------------------------------------
#endif
