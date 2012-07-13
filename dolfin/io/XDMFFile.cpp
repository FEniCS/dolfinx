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
// Last changed: 2012-07-13

#include <ostream>
#include <sstream>
#include <vector>

#include "pugixml.hpp"

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
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
  operator << (ut);
}
//----------------------------------------------------------------------------
void XDMFFile::operator<<(const std::pair<const Function*, double> ut)
{
  dolfin_assert(ut.first);
  const Function &u = *(ut.first);
  const double time_step = ut.second;

  // Save Function to XDMF/HDF file for visualisation
  // Can be read by paraview

  Timer hdf5timer("HDF5 + XDMF Output (mesh + data)");

  u.update();
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();
  const uint vrank = u.value_rank();
  const uint vsize = u.value_size();

  // Tensors of rank > 1 not yet supported
  if (vrank > 1)
  {
    dolfin_error("XDMFFile.cpp",
                "write data to XDMF file",
                "Outout of tensors with rank > 1 not yet supported");
  }

  const uint cell_dim = mesh.topology().dim();
  const uint num_local_cells = mesh.num_cells();
  const uint num_local_vertices = mesh.num_vertices();
  const uint num_global_vertices = MPI::sum(num_local_vertices);
  const uint num_global_cells = MPI::sum(num_local_cells);

  // Compute vertex values
  std::vector<double> vtx_values;
  u.compute_vertex_values(vtx_values, mesh);

  // Need to interleave the values (e.g. if not scalar field)
  if(vsize  > 1)
  {
    std::vector<double> tmp;
    tmp.reserve(num_local_vertices);
    for(uint i = 0; i < num_local_vertices; i++)
      for(uint j = 0; j < vsize; j++)
	      tmp.push_back(vtx_values[i + j*num_local_vertices]);
    std::copy(tmp.begin(), tmp.end(), vtx_values.begin());
  }

  // Get offset and size of local vertex usage in global terms
  const uint off = MPI::global_offset(num_local_vertices, true);
  std::pair<uint,uint> vertex_range(off, off + num_local_vertices);

  std::string filename_data(HDF5Filename());

  // Create HDF5 file and save data and coords
  HDF5File h5file(filename_data);

  // Presently, only save mesh on first timestep.
  // This is a practical consideration to rein in the size
  // of the file. Ideally, the mesh could have a UID or counter
  // which indicates when the mesh has changed.
  // Failing that, there could be a switch to choose between the
  // two behaviours (save mesh once, or save mesh every timestep)
  if(counter == 0)
  {
    h5file.create();
    h5file << mesh;
  }

  // Vertex values are saved in the hdf5 'folder' /VertexVector
  // as distinct from /Vector which is used for raw vectors.
  // For simple Lagrange elements, they are identical.
  std::stringstream s("");
  s << "/VertexVector/" << counter;
  h5file.write(vtx_values[0], vertex_range, s.str().c_str(), vsize); //values

  //Now go ahead and write the XML meta description
  if(MPI::process_number()==0)
  {
    pugi::xml_document xml_doc;
    pugi::xml_node xdmf_timegrid;
    pugi::xml_node xdmf_timedata;

    if(counter==0)
    {
      xml_doc.append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
      pugi::xml_node xdmf = xml_doc.append_child("Xdmf");
      xdmf.append_attribute("Version")="2.0";
      xdmf.append_attribute("xmlns:xi")="http://www.w3.org/2001/XInclude";

      pugi::xml_node xdmf_domn = xdmf.append_child("Domain");

      //         /Xdmf/Domain/Topology

      pugi::xml_node xdmf_topo = xdmf_domn.append_child("Topology");
      if(cell_dim==2)
        xdmf_topo.append_attribute("TopologyType") = "Triangle";
      else if(cell_dim==3)
        xdmf_topo.append_attribute("TopologyType") = "Tetrahedron";

      xdmf_topo.append_attribute("NumberOfElements") = num_global_cells;
      pugi::xml_node xdmf_topo_data = xdmf_topo.append_child("DataItem");

      xdmf_topo_data.append_attribute("Format") = "HDF";
      std::stringstream s;
      s << num_global_cells << " " << (cell_dim + 1);
      xdmf_topo_data.append_attribute("Dimensions") = s.str().c_str();

      s.str("");
      s << filename_data << ":/Mesh/Topology";
      xdmf_topo_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());

      //      /Xdmf/Domain/Geometry

      pugi::xml_node xdmf_geom = xdmf_domn.append_child("Geometry");
      xdmf_geom.append_attribute("GeometryType") = "XYZ";
      pugi::xml_node xdmf_geom_data = xdmf_geom.append_child("DataItem");

      xdmf_geom_data.append_attribute("Format")="HDF";
      s.str("");
      s << num_global_vertices << " 3";
      xdmf_geom_data.append_attribute("Dimensions") = s.str().c_str();

      s.str("");
      s << filename_data << ":/Mesh/Coordinates";
      xdmf_geom_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());


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
      xdmf_timedata.append_attribute("Dimensions") = "3";
      xdmf_timedata.append_child(pugi::node_pcdata);

    }
    else
    {
      // Read in existing XDMF file
      pugi::xml_parse_result result = xml_doc.load_file(filename.c_str());
      if (!result)
      {
        dolfin_error("XDMFFile.cpp",
                     "write data to XDMF file",
                     "XML parsing error when reading from existing file");
      }

      xdmf_timegrid = xml_doc.child("Xdmf").child("Domain").child("Grid");
      xdmf_timedata = xml_doc.child("Xdmf").child("Domain").child("Grid").child("Time").child("DataItem");
    }

    // Add an extra timestep to the TimeSeries List
    s.str("");
    s << xdmf_timedata.first_child().value() << " " << time_step;
    xdmf_timedata.first_child().set_value(s.str().c_str());

    //    /Xdmf/Domain/Grid/Grid - the actual data for this timestep
    pugi::xml_node xdmf_grid = xdmf_timegrid.append_child("Grid");
    s.str("");
    s << u.name() << "_" << counter;
    xdmf_grid.append_attribute("Name") = s.str().c_str();
    xdmf_grid.append_attribute("GridType") = "Uniform";
    pugi::xml_node xdmf_toporef = xdmf_grid.append_child("Topology");
    xdmf_toporef.append_attribute("Reference") = "/Xdmf/Domain/Topology[1]";
    pugi::xml_node xdmf_geomref = xdmf_grid.append_child("Geometry");
    xdmf_geomref.append_attribute("Reference") = "/Xdmf/Domain/Geometry[1]";

    pugi::xml_node xdmf_vals=xdmf_grid.append_child("Attribute"); //actual data
    xdmf_vals.append_attribute("Name")=u.name().c_str();
    if(vsize==1)
      xdmf_vals.append_attribute("AttributeType")="Scalar";
    else
      xdmf_vals.append_attribute("AttributeType")="Vector";
    xdmf_vals.append_attribute("Center")="Node";
    pugi::xml_node xdmf_data=xdmf_vals.append_child("DataItem");
    xdmf_data.append_attribute("Format")="HDF";
    s.str("");
    s << num_global_vertices << " " << vsize;
    xdmf_data.append_attribute("Dimensions")=s.str().c_str();
    s.str("");
    s<< filename_data << ":/VertexVector/" << counter;
    xdmf_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());

    xml_doc.save_file(filename.c_str(), "  ");
  }

  counter++;
}
//----------------------------------------------------------------------------
void XDMFFile::operator<<(const Mesh& mesh)
{
  // Mesh for visualisation, with e.g. ParaView
  Timer hdf5timer("HDF5+XDMF Output (mesh)");

  const uint cell_dim = mesh.topology().dim();
  const uint num_local_cells = mesh.num_cells();
  const uint num_local_vertices = mesh.num_vertices();
  const uint num_global_vertices = MPI::sum(num_local_vertices);
  const uint num_global_cells = MPI::sum(num_local_cells);

  std::string filename_data(HDF5Filename());

  HDF5File h5file(filename_data);
  // Truncate existing file. Could check, and add to it instead.
  // However, that is probably more suitable behaviour for
  // HDF5File::operator<<(const Mesh& mesh)
  h5file.create();
  h5file << mesh;

  // Go ahead and write the XML meta description
  if(MPI::process_number() == 0)
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

    pugi::xml_node xdmf_topo = xdmf_grid.append_child("Topology");

    if(cell_dim==2)
      xdmf_topo.append_attribute("TopologyType")="Triangle";
    else if(cell_dim==3)
      xdmf_topo.append_attribute("TopologyType")="Tetrahedron";

    xdmf_topo.append_attribute("NumberOfElements") = num_global_cells;
    pugi::xml_node xdmf_topo_data = xdmf_topo.append_child("DataItem");

    xdmf_topo_data.append_attribute("Format")="HDF";
    std::stringstream s;
    s << num_global_cells << " " << (cell_dim + 1);
    xdmf_topo_data.append_attribute("Dimensions") = s.str().c_str();

    s.str("");
    s<< filename_data << ":/Mesh/Topology";
    xdmf_topo_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());

    pugi::xml_node xdmf_geom = xdmf_grid.append_child("Geometry");
    xdmf_geom.append_attribute("GeometryType") = "XYZ";
    pugi::xml_node xdmf_geom_data = xdmf_geom.append_child("DataItem");

    xdmf_geom_data.append_attribute("Format") = "HDF";
    s.str("");
    s << num_global_vertices << " 3";
    xdmf_geom_data.append_attribute("Dimensions") = s.str().c_str();

    s.str("");
    s << filename_data << ":/Mesh/Coordinates";
    xdmf_geom_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());

    xml_doc.save_file(filename.c_str(), "  ");

    log(TRACE, "Saved mesh %s (%s) to file %s in XDMF format.",
    mesh.name().c_str(), mesh.label().c_str(), filename.c_str());
  }
}
//----------------------------------------------------------------------------
std::string XDMFFile::HDF5Filename() const
{
  // Generate .h5 from .xdmf filename
  std::string fname;
  fname.assign(filename, 0, filename.find_last_of("."));
  fname.append(".h5");
  return fname;
}
//----------------------------------------------------------------------------
