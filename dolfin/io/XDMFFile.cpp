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
//
// First added:  2012-05-28
// Last changed: 2012-05-28

#include <ostream>
#include <sstream>
#include <vector>
#include <boost/cstdint.hpp>
#include <boost/detail/endian.hpp>

#include "pugixml.hpp"

#include <dolfin/common/Array.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>

#include "XDMFFile.h"
#include "HDF5File.h"

using namespace dolfin;

//----------------------------------------------------------------------------

XDMFFile::XDMFFile(const std::string filename)
  : GenericFile(filename, "XDMF")
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
  //  if(MPI::num_processes()==1) // bork if not in parallel
  //    return;

  u.update();
  const FunctionSpace& V = *u.function_space(); 

  const Mesh& mesh = *V.mesh();
  const GenericDofMap& dofmap = *V.dofmap();
  const GenericVector& vec = *u.vector();

  const uint cell_dim = mesh.topology().dim();
  const uint num_global_cells = MPI::sum(mesh.num_cells());
  const uint num_global_vertices = dofmap.global_dimension();
  const std::pair<uint,uint> local_range = vec.local_range();
  const boost::unordered_map<uint, uint> off_process_owner=dofmap.off_process_owner();

  std::vector<uint> topo_data;
  std::map<uint,uint> local_map;

  // make a map of global to local vertices, ignoring any off process
  // also save some topological data for each cell
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const uint i = cell->index();

      std::vector<uint> gdof = dofmap.cell_dofs(i);
      std::vector<uint>::iterator igdof=gdof.begin();
      for (VertexIterator v(*cell); !v.end(); ++v){
	topo_data.push_back(*igdof);
       	if (off_process_owner.find(*igdof)==off_process_owner.end())    
	  //must be on this process
	  local_map.insert(std::pair<uint,uint>(*igdof,v->index()));
	++igdof;
      }
    }

  const double *m_coords = mesh.coordinates();
  std::vector<double>cvec;

  for(uint i=local_range.first;i<local_range.second;i++)
    {
      uint j=local_map[i];
      if(cell_dim==2){
	cvec.push_back(m_coords[j*2]);
	cvec.push_back(m_coords[j*2+1]);
	cvec.push_back(0.0);
      } else if(cell_dim==3){
	cvec.push_back(m_coords[j*3]);
	cvec.push_back(m_coords[j*3+1]);
	cvec.push_back(m_coords[j*3+2]);
      }
    }

  // these should all be in one HDF5 file as different datasets.

  std::string basename;
  std::ostringstream fname;

  basename.assign(filename, 0, filename.find_last_of("."));
  fname << basename << ".h5";
  std::string filename_data(fname.str());

  fname.str("");
  fname << basename << "_coords.h5";
  std::string filename_coords(fname.str());

  fname.str("");
  fname << basename << "_coords.h5";
  std::string filename_topo(fname.str());

  HDF5File h5coords(filename_coords);
  h5coords.write(cvec[0],local_range);

  // Want to write topo data as H5 file also, but need to 
  // know the 'local range' for cells, as opposed to vertices...
  // so meanwhile, use MPI::gather and XML
  fprintf(stderr,"topo data size = %d\n",topo_data.size());
  //  HDF5File h5topo("poisson_topo.h5");
  //  h5topo.write(topo_data[0],???cell_range???);
  
  std::vector<std::vector<uint> > topo_gather;
  MPI::gather(topo_data,topo_gather);

    if(MPI::process_number()==0){

    pugi::xml_document xml_doc;

    xml_doc.append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf = xml_doc.append_child("Xdmf");
    xdmf.append_attribute("Version")="2.0";
    //  xdmf.append_attribute("xmlns:xi")="[http://www.w3.org/2001/XInclude]";
    pugi::xml_node xdmf_domn = xdmf.append_child("Domain");
    pugi::xml_node xdmf_grid = xdmf_domn.append_child("Grid");
    xdmf_grid.append_attribute("Name")="dolfin_grid";
    xdmf_grid.append_attribute("GridType")="Uniform";
    
    pugi::xml_node xdmf_topo = xdmf_grid.append_child("Topology");

    if(cell_dim==2)
      xdmf_topo.append_attribute("TopologyType")="Triangle";
    else if(cell_dim==3)
      xdmf_topo.append_attribute("TopologyType")="Tetrahedron";

    xdmf_topo.append_attribute("NumberOfElements")=num_global_cells;
    pugi::xml_node xdmf_topo_data = xdmf_topo.append_child("DataItem");
    xdmf_topo_data.append_attribute("Format")="XML";
    // Soon:
    //    xdmf_topo_data.append_attribute("Format")="HDF"; 
    std::stringstream s;
        s << num_global_cells << " " << (cell_dim+1);
    xdmf_topo_data.append_attribute("Dimensions")=s.str().c_str();

    // Write out the gathered topo data
    s.str("\n");
    std::vector<std::vector<uint> >::iterator topo_n;
    for(topo_n=topo_gather.begin();topo_n!=topo_gather.end();++topo_n)
      for(std::vector<uint>::iterator topo_i=(*topo_n).begin();
	  topo_i!=(*topo_n).end();++topo_i)
	s << *topo_i << " ";
    s << std::endl;

    //    s.str("\npoisson_topo.h5:/dolfin_topo\n");
    xdmf_topo_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());
    
    pugi::xml_node xdmf_geom = xdmf_grid.append_child("Geometry");
    xdmf_geom.append_attribute("GeometryType")="XYZ";
    pugi::xml_node xdmf_geom_data = xdmf_geom.append_child("DataItem");

    xdmf_geom_data.append_attribute("Format")="HDF";
    s.str("");
    s << num_global_vertices << " 3";
    xdmf_geom_data.append_attribute("Dimensions")=s.str().c_str();

    s.str("");
    s << filename_coords << ":/dolfin_coords";
    xdmf_geom_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());
    
    pugi::xml_node xdmf_vals=xdmf_grid.append_child("Attribute"); //actual data
    xdmf_vals.append_attribute("Name")="Vertex Data";
    xdmf_vals.append_attribute("Center")="Node";
    pugi::xml_node xdmf_data=xdmf_vals.append_child("DataItem");
    xdmf_data.append_attribute("Format")="HDF";
    s.str("");
    s << num_global_vertices;
    xdmf_data.append_attribute("Dimensions")=s.str().c_str();
    s.str("");
    s<< filename_data << ":/dolfin_vector";
    xdmf_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());
    
    xml_doc.save_file(filename.c_str(), "  "); 
  }


}

void XDMFFile::operator<<(const Mesh& mesh)
{

  const uint cell_dim = mesh.topology().dim();
  const uint num_cells = mesh.num_cells();
  const uint num_vertices = mesh.num_vertices();

  pugi::xml_document xml_doc;
  xml_doc.append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
  pugi::xml_node xdmf = xml_doc.append_child("Xdmf");
  xdmf.append_attribute("Version")="2.0";
  xdmf.append_attribute("xmlns:xi")="\"http://www.w3.org/2001/XInclude\"";
  pugi::xml_node xdmf_domn = xdmf.append_child("Domain");
  pugi::xml_node xdmf_grid = xdmf_domn.append_child("Grid");
  xdmf_grid.append_attribute("Name")="dolfin_grid";
  xdmf_grid.append_attribute("GridType")="Uniform";

  pugi::xml_node xdmf_topo = xdmf_grid.append_child("Topology");
  xdmf_topo.append_attribute("TopologyType")="Triangle";
  xdmf_topo.append_attribute("NumberOfElements")=num_cells;
  pugi::xml_node xdmf_topo_data = xdmf_topo.append_child("DataItem");
  xdmf_topo_data.append_attribute("Format")="XML";
  std::stringstream s;
  s << num_cells << " " << (cell_dim+1);
  xdmf_topo_data.append_attribute("Dimensions")=s.str().c_str();
  s.str("");
  s << std::endl;
  for (MeshEntityIterator c(mesh, cell_dim); !c.end(); ++c)
    {
      for (VertexIterator v(*c); !v.end(); ++v)
	s << v->index() << " ";
      s << std::endl;
    }
  xdmf_topo_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());

  pugi::xml_node xdmf_geom = xdmf_grid.append_child("Geometry");
  xdmf_geom.append_attribute("GeometryType")="XYZ";
  pugi::xml_node xdmf_geom_data = xdmf_geom.append_child("DataItem");
  xdmf_geom_data.append_attribute("Format")="XML";
  s.str("");
  s << num_vertices << " 3 3";
  xdmf_geom_data.append_attribute("Dimensions")=s.str().c_str();
  s.str("");
  s << std::endl;
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    Point p = v->point();
    s << p.x() << " " << p.y() << " " <<  p.z() << "  " << std::endl;
  }
  xdmf_geom_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());

  xml_doc.save_file(filename.c_str(), "  ");

  log(TRACE, "Saved mesh %s (%s) to file %s in XDMF format.",
      mesh.name().c_str(), mesh.label().c_str(), filename.c_str());
}
