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
// Last changed: 2012-08-03

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
}
//----------------------------------------------------------------------------
XDMFFile::~XDMFFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void XDMFFile::operator<<(const Function& u)
{
  std::pair<const Function*, double> ut(&u, (double)counter);
  *this << ut;
}
//----------------------------------------------------------------------------
void XDMFFile::operator<<(const std::pair<const Function*, double> ut)
{
  // Save a Function to XDMF/HDF files for visualisation.
  // Downgrading may occur due to collecting the values
  // to vertices using compute_vertex_values()
  //
  // Creates an HDF5 file for storing Mesh and Vertex/Cell Values,
  // and an associated XDMF file for metadata.
  // Subsequent calls will store additional Vertex/Cell Values
  // in the same HDF5 file, and update the XDMF metadata
  // to represent a time series.

  dolfin_assert(ut.first);
  const Function &u = *(ut.first);
  const double time_step = ut.second;

  Timer hdf5timer("Write XDMF (mesh + data) vertex");

  // Update any ghost values
  u.update();

  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap= *u.function_space()->dofmap();

  const uint vrank = u.value_rank();
  const uint vsize = u.value_size();
  const uint cell_dim = mesh.topology().dim();
  uint vsize_io = vsize; // output size may be padded 2D -> 3D for paraview

  // Test for cell-centred data
  uint cell_based_dim = 1;
  for (uint i = 0; i < vrank; i++)
    cell_based_dim *= cell_dim;

  const bool vertex_data = !(dofmap.max_cell_dimension() == cell_based_dim);

  // Tensors of rank > 2 not supported
  if (vrank > 2)
  {
    dolfin_error("XDMFFile.cpp",
		             "write data to XDMF file",
		             "Output of tensors with rank > 2 not yet supported");
  }

  if (cell_dim == 1)
  {
    dolfin_error("XDMFFile.cpp",
		             "write data to XDMF file",
		             "Output of 1D datasets not supported");
  }

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
  // Also pad 2D vectors and tensors to 3D.
  if (vrank > 0)
  {
    std::vector<double> tmp;
    tmp.reserve(vsize*num_local_entities);
    for(uint i = 0; i < num_local_entities; i++)
    {
      for (uint j = 0; j < vsize; j++)
      {
        tmp.push_back(data_values[i + j*num_local_entities]);
        if(j==1 && vsize==4)
          tmp.push_back(0.0);
      }
      if (vsize == 2)    // 2D -> 3D vector
        tmp.push_back(0.0);
      if(vsize == 4) // 2D -> 3D tensor
        tmp.insert(tmp.end(), 4, 0.0);
    }
    data_values.resize(tmp.size()); // 2D->3D padding increases size
    std::copy(tmp.begin(), tmp.end(), data_values.begin());
    if (vsize == 2)
      vsize_io = 3;
    if (vsize == 4)
      vsize_io = 9;
  }

  // Create HDF5 file and save mesh
  std::string filename_data(HDF5Filename());
  HDF5File h5file(filename_data);
  std::string mc_name = h5file.mesh_coords_dataset_name(mesh);
  std::string mt_name = h5file.mesh_topo_dataset_name(mesh);

  if (counter == 0)
  {
    h5file.create();
    h5file << mesh;
  }
  else if (!h5file.exists(mc_name) || !h5file.exists(mt_name))
    h5file << mesh;

  // Vertex values are saved in the hdf5 'folder' /DataVector
  // as distinct from /Vector which is used for solution vectors.

  // Save actual data values to HDF5 file
  std::stringstream s("");
  s << "/DataVector/" << counter;
  h5file.write(data_values[0], data_range, s.str().c_str(), vsize_io);

  // FIXME: Use Boost filesystem to make this platform-independent
  // Remove path from filename_data
  std::size_t lastslash = filename_data.rfind('/');
  filename_data.erase(0, lastslash + 1);

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

      xdmf_timegrid = xml_doc.child("Xdmf").child("Domain").child("Grid");
      xdmf_timedata = xdmf_timegrid.child("Time").child("DataItem");
    }

    // Add an extra timestep to the TimeSeries List
    s.str("");
    s << xdmf_timedata.first_child().value() << " " << time_step;
    xdmf_timedata.first_child().set_value(s.str().c_str());
    s.str("");
    s << (counter+1);
    xdmf_timedata.attribute("Dimensions").set_value(s.str().c_str());

    //    /Xdmf/Domain/Grid/Grid - the actual data for this timestep
    pugi::xml_node xdmf_grid = xdmf_timegrid.append_child("Grid");
    s.str("");
    s << u.name() << "_" << counter;
    xdmf_grid.append_attribute("Name") = s.str().c_str();
    xdmf_grid.append_attribute("GridType") = "Uniform";

    // Grid/Topology

    pugi::xml_node xdmf_topology = xdmf_grid.append_child("Topology");
    if(cell_dim==2)
      xdmf_topology.append_attribute("TopologyType") = "Triangle";
    else if(cell_dim==3)
      xdmf_topology.append_attribute("TopologyType") = "Tetrahedron";

    xdmf_topology.append_attribute("NumberOfElements") = num_global_cells;
    pugi::xml_node xdmf_topology_data = xdmf_topology.append_child("DataItem");

    xdmf_topology_data.append_attribute("Format") = "HDF";
    std::stringstream s;
    s << num_global_cells << " " << (cell_dim + 1);
    xdmf_topology_data.append_attribute("Dimensions") = s.str().c_str();

    s.str("");
    s << filename_data << ":" << mt_name;
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
    s << filename_data << ":" << mc_name;
    xdmf_geom_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());

    // Grid/Attribute - actual data

    pugi::xml_node xdmf_vals=xdmf_grid.append_child("Attribute");
    xdmf_vals.append_attribute("Name")=u.name().c_str();

    if (vrank == 0)
      xdmf_vals.append_attribute("AttributeType")="Scalar";
    else if (vrank == 1)
      xdmf_vals.append_attribute("AttributeType")="Vector";
    else if (vrank ==2 )
      xdmf_vals.append_attribute("AttributeType")="Tensor";

    if (vertex_data)
      xdmf_vals.append_attribute("Center") = "Node";
    else
      xdmf_vals.append_attribute("Center") = "Cell";

    pugi::xml_node xdmf_data=xdmf_vals.append_child("DataItem");
    xdmf_data.append_attribute("Format")="HDF";
    s.str("");
    if(vertex_data)
      s << num_global_vertices << " " << vsize_io;
    else
      s << num_global_cells << " " << vsize_io;

    xdmf_data.append_attribute("Dimensions")=s.str().c_str();
    s.str("");
    s<< filename_data << ":/DataVector/" << counter;
    xdmf_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());

    xml_doc.save_file(filename.c_str(), "  ");
  }

  counter++;
}
//----------------------------------------------------------------------------
void XDMFFile::operator<<(const Mesh& mesh)
{
  // Save a mesh for visualisation, with e.g. ParaView. Creates a HDF5
  // file to store the mesh, and a related XDMF file with metadata.

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
    xdmf.append_attribute("xmlns:xi") = "\"http://www.w3.org/2001/XInclude\"";
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
  p.replace_extension("h5");
  return p.string();
}
//----------------------------------------------------------------------------
#endif
