// Copyright (C) 2012 Chris N. Richardson
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
// Last changed: 2012-09-17

#ifdef HAS_HDF5

#include <ostream>
#include <sstream>
#include <vector>
#include <boost/filesystem.hpp>

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

  // FIXME: Open HDF5 storage file here
  // Generate .h5 from .xdmf filename

  // Name of HDF5 file
  boost::filesystem::path p(filename);
  p.replace_extension(".h5");
  //const std::string hdf5_filename = p.string();

  // Create HDF5 file and save mesh
  hdf5_file.reset(new HDF5File(p.string()));

}
//----------------------------------------------------------------------------
XDMFFile::~XDMFFile()
{
  // Do nothing

  // FIXME: Close HDF5  storage file here
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
  const GenericDofMap& dofmap= *u.function_space()->dofmap();

  const uint value_rank = u.value_rank();
  const uint value_size = u.value_size();
  const uint cell_dim = mesh.topology().dim();
  uint value_size_io = value_size; // output size may be padded 2D -> 3D for paraview

  // Test for cell-centred data
  uint cell_based_dim = 1;
  for (uint i = 0; i < value_rank; i++)
    cell_based_dim *= cell_dim;
  const bool vertex_data = !(dofmap.max_cell_dimension() == cell_based_dim);

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

  // Get nmber of local/global cells/vertices
  // FIXME: num_global_vertices is not correct
  const uint num_local_cells = mesh.num_cells();
  const uint num_local_vertices = mesh.num_vertices();
  const uint num_global_vertices = MPI::sum(num_local_vertices);
  const uint num_global_cells = MPI::sum(num_local_cells);

  // Get Function data
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

  // FIXME: Should this be in the HDF5 code
  const uint offset = MPI::global_offset(num_local_entities, true);
  const std::pair<uint,uint> data_range(offset, offset + num_local_entities);

  // Need to interleave the values (e.g. if vector or tensor field)
  // Also pad 2D vectors and tensors to 3D
  if (value_rank > 0)
  {
    std::vector<double> tmp;
    tmp.reserve(value_size*num_local_entities);
    for(uint i = 0; i < num_local_entities; i++)
    {
      for (uint j = 0; j < value_size; j++)
      {
        tmp.push_back(data_values[i + j*num_local_entities]);
        if(j == 1 && value_size == 4)
          tmp.push_back(0.0);
      }
      if (value_size == 2)    // 2D -> 3D vector
        tmp.push_back(0.0);
      if(value_size == 4) // 2D -> 3D tensor
        tmp.insert(tmp.end(), 4, 0.0);
    }
    data_values.resize(tmp.size()); // 2D->3D padding increases size
    std::copy(tmp.begin(), tmp.end(), data_values.begin());

    if (value_size == 2)
      value_size_io = 3;
    if (value_size == 4)
      value_size_io = 9;
  }

  dolfin_assert(hdf5_file);
  const std::string mc_name = hdf5_file->mesh_coords_dataset_name(mesh);
  const std::string mt_name = hdf5_file->mesh_topo_dataset_name(mesh);
  if (counter == 0)
  {
    hdf5_file->create();
    *(hdf5_file) << mesh;
  }
  else if (!hdf5_file->exists(mc_name) || !hdf5_file->exists(mt_name))
    *(hdf5_file) << mesh;

  // Vertex values are saved in the hdf5 'folder' /DataVector
  // as distinct from /Vector which is used for solution vectors.

  // Save actual data values to HDF5 file
  std::stringstream s("");
  s << "/DataVector/" << counter;

  hdf5_file->write(data_values[0], data_range, s.str().c_str(), value_size_io);

  // Write the XML meta description - see http://www.xdmf.org
  if (MPI::process_number() == 0)
  {
    pugi::xml_document xml_doc;
    pugi::xml_node xdmf_timegrid;
    pugi::xml_node xdmf_timedata;

    if (counter == 0)
    {
      // First time step - create document template, adding a mesh and an empty time-series.

      xml_doc.append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
      pugi::xml_node xdmf = xml_doc.append_child("Xdmf");
      xdmf.append_attribute("Version")="2.0";
      xdmf.append_attribute("xmlns:xi")="http://www.w3.org/2001/XInclude";

      pugi::xml_node xdmf_domn = xdmf.append_child("Domain");

      //     /Xdmf/Domain/Grid - actually a TimeSeries, not a spatial grid
      xdmf_timegrid = xdmf_domn.append_child("Grid");
      xdmf_timegrid.append_attribute("Name") = "TimeSeries";
      xdmf_timegrid.append_attribute("GridType") = "Collection";
      xdmf_timegrid.append_attribute("CollectionType") = "Temporal";

      //     /Xdmf/Domain/Grid/Time
      pugi::xml_node xdmf_time = xdmf_timegrid.append_child("Time");
      xdmf_time.append_attribute("TimeType") = "List";
      xdmf_timedata=xdmf_time.append_child("DataItem");
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

      // FIXME: Test that these exist

      // Get data node
      xdmf_timegrid = xml_doc.child("Xdmf").child("Domain").child("Grid");

      // Get time series node
      xdmf_timedata = xdmf_timegrid.child("Time").child("DataItem");
    }


    // FIXME: This is confusing. Why are string streams necessary?
    //        Can use Boost Lexical cast, if necessary
    // Add an extra timestep to the TimeSeries List
    s.str("");
    s << xdmf_timedata.first_child().value() << " " << time_step;
    xdmf_timedata.first_child().set_value(s.str().c_str());

    s.str("");
    s << (counter + 1);
    xdmf_timedata.attribute("Dimensions").set_value(s.str().c_str());

    //    /Xdmf/Domain/Grid/Grid - the actual data for this timestep
    pugi::xml_node xdmf_grid = xdmf_timegrid.append_child("Grid");
    s.str("");
    s << u.name() << "_" << counter;
    xdmf_grid.append_attribute("Name") = s.str().c_str();
    xdmf_grid.append_attribute("GridType") = "Uniform";

    // Grid/Topology
    pugi::xml_node xdmf_topology = xdmf_grid.append_child("Topology");
    if (cell_dim == 2)
      xdmf_topology.append_attribute("TopologyType") = "Triangle";
    else if (cell_dim == 3)
      xdmf_topology.append_attribute("TopologyType") = "Tetrahedron";

    xdmf_topology.append_attribute("NumberOfElements") = num_global_cells;
    pugi::xml_node xdmf_topology_data = xdmf_topology.append_child("DataItem");

    xdmf_topology_data.append_attribute("Format") = "HDF";
    std::stringstream s;
    s << num_global_cells << " " << (cell_dim + 1);
    xdmf_topology_data.append_attribute("Dimensions") = s.str().c_str();

    s.str("");
    s << hdf5_file->filename << ":" << mt_name;
    xdmf_topology_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());

    // Grid/Geometry
    pugi::xml_node xdmf_geom = xdmf_grid.append_child("Geometry");
    xdmf_geom.append_attribute("GeometryType") = "XYZ";
    pugi::xml_node xdmf_geom_data = xdmf_geom.append_child("DataItem");

    xdmf_geom_data.append_attribute("Format")="HDF";
    s.str("");
    s << num_global_vertices << " 3";
    xdmf_geom_data.append_attribute("Dimensions") = s.str().c_str();

    s.str("");
    s << hdf5_file->filename << ":" << mc_name;
    xdmf_geom_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());

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
    s.str("");
    if(vertex_data)
      s << num_global_vertices << " " << value_size_io;
    else
      s << num_global_cells << " " << value_size_io;

    xdmf_data.append_attribute("Dimensions") = s.str().c_str();
    s.str("");
    s<< hdf5_file->filename << ":/DataVector/" << counter;
    xdmf_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());
    xml_doc.save_file(filename.c_str(), "  ");
  }

  counter++;
}
//----------------------------------------------------------------------------
void XDMFFile::operator<<(const Mesh& mesh)
{
  Timer hdf5timer("HDF5+XDMF Output (mesh)");

  const uint cell_dim = mesh.topology().dim();
  const uint num_local_cells = mesh.num_cells();
  const uint num_local_vertices = mesh.num_vertices();
  const uint num_global_vertices = MPI::sum(num_local_vertices);
  const uint num_global_cells = MPI::sum(num_local_cells);

  std::string filename_data(HDF5Filename());

  // Create a new HDF5 file and save the mesh
  HDF5File h5file(filename_data);
  h5file.create();
  h5file << mesh;

  // Write the XML meta description
  if (MPI::process_number() == 0)
  {
    pugi::xml_document xml_doc;

    xml_doc.append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf = xml_doc.append_child("Xdmf");
    xdmf.append_attribute("Version") = "2.0";
    xdmf.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";
    pugi::xml_node xdmf_domn = xdmf.append_child("Domain");
    pugi::xml_node xdmf_grid = xdmf_domn.append_child("Grid");
    xdmf_grid.append_attribute("Name") = "dolfin_grid";
    xdmf_grid.append_attribute("GridType") = "Uniform";

    pugi::xml_node xdmf_topology = xdmf_grid.append_child("Topology");

    if (cell_dim == 2)
      xdmf_topology.append_attribute("TopologyType") = "Triangle";
    else if (cell_dim == 3)
      xdmf_topology.append_attribute("TopologyType") = "Tetrahedron";

    xdmf_topology.append_attribute("NumberOfElements") = num_global_cells;
    pugi::xml_node xdmf_topology_data = xdmf_topology.append_child("DataItem");

    xdmf_topology_data.append_attribute("Format") = "HDF";
    std::stringstream s;
    s << num_global_cells << " " << (cell_dim + 1);
    xdmf_topology_data.append_attribute("Dimensions") = s.str().c_str();

    s.str("");
    s<< filename_data << ":" << h5file.mesh_topo_dataset_name(mesh);
    xdmf_topology_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());

    pugi::xml_node xdmf_geom = xdmf_grid.append_child("Geometry");
    xdmf_geom.append_attribute("GeometryType") = "XYZ";
    pugi::xml_node xdmf_geom_data = xdmf_geom.append_child("DataItem");

    xdmf_geom_data.append_attribute("Format") = "HDF";
    s.str("");
    s << num_global_vertices << " 3";
    xdmf_geom_data.append_attribute("Dimensions") = s.str().c_str();

    s.str("");
    s << filename_data << ":" << h5file.mesh_coords_dataset_name(mesh);
    xdmf_geom_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());

    xml_doc.save_file(filename.c_str(), "  ");
  }
}
//----------------------------------------------------------------------------
std::string XDMFFile::HDF5Filename() const
{
  // Generate .h5 from .xdmf filename
  boost::filesystem::path p(filename);
  p.replace_extension(".h5");
  return p.string();
}
//----------------------------------------------------------------------------
#endif
