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


//-----------------------------------------------------------------------------
void XDMFFile::build_global_to_cell_dof(
  std::vector<std::vector<std::pair<uint, uint> > >& global_dof_to_cell_dof,
  const FunctionSpace& V)
{
  // Get mesh and dofmap
  dolfin_assert(V.mesh());
  dolfin_assert(V.dofmap());
  const Mesh& mesh = *V.mesh();
  const GenericDofMap& dofmap = *V.dofmap();

  std::vector<std::vector<std::vector<uint > > > gathered_dofmap;
  std::vector<std::vector<uint > > local_dofmap(mesh.num_cells());

  if (MPI::num_processes() > 1)
  {
    // Get local-to-global cell numbering
    dolfin_assert(mesh.parallel_data().have_global_entity_indices(mesh.topology().dim()));
    const MeshFunction<uint>& global_cell_indices
      = mesh.parallel_data().global_entity_indices(mesh.topology().dim());

    // Build dof map data with global cell indices
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const uint local_cell_index = cell->index();
      const uint global_cell_index = global_cell_indices[*cell];
      local_dofmap[local_cell_index] = dofmap.cell_dofs(local_cell_index);
      local_dofmap[local_cell_index].push_back(global_cell_index);
    }
  }
  else
  {
    // Build dof map data
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const uint local_cell_index = cell->index();
      local_dofmap[local_cell_index] = dofmap.cell_dofs(local_cell_index);
      local_dofmap[local_cell_index].push_back(local_cell_index);
    }
  }

  // Gather dof map data on root process
  MPI::gather(local_dofmap, gathered_dofmap);

  // Build global dof - (global cell, local dof) map on root process
  if (MPI::process_number() == 0)
  {
    global_dof_to_cell_dof.resize(dofmap.global_dimension());

    std::vector<std::vector<std::vector<uint > > > ::const_iterator proc_dofmap;
    for (proc_dofmap = gathered_dofmap.begin(); proc_dofmap != gathered_dofmap.end(); ++proc_dofmap)
    {
      std::vector<std::vector<uint> >::const_iterator cell_dofmap;
      for (cell_dofmap = proc_dofmap->begin(); cell_dofmap != proc_dofmap->end(); ++cell_dofmap)
      {
        const std::vector<uint>& cell_dofs = *cell_dofmap;
        const uint global_cell_index = cell_dofs.back();
        for (uint i = 0; i < cell_dofs.size() - 1; ++i)
          global_dof_to_cell_dof[cell_dofs[i]].push_back(std::make_pair(global_cell_index, i));
      }
    }
  }
}


void XDMFFile::operator<<(const Function& u)
{
  if(MPI::num_processes()==1) // bork if not in parallel
    return;

  u.update();
  const FunctionSpace& V = *u.function_space(); 

  const Mesh& mesh = *V.mesh();
  const GenericDofMap& dofmap = *V.dofmap();
  const GenericVector& vec = *u.vector();

  const uint cell_dim = mesh.topology().dim();
  const uint num_global_cells = MPI::sum(mesh.num_cells());
  const uint num_local_cells = mesh.num_cells();
  const uint num_global_vertices = dofmap.global_dimension();
  const uint num_local_vertices = vec.local_size();
  const std::pair<uint,uint> local_range = vec.local_range();
  const std::pair<uint,uint> ownership_range=dofmap.ownership_range();
  const boost::unordered_map<uint, uint> off_process_owner=dofmap.off_process_owner();

  std::stringstream s;

  fprintf(stderr,"%d %d %d %d\n",num_global_cells,num_local_cells,
	  num_global_vertices,num_local_vertices);

  fprintf(stderr,"DOF Range=%d,%d\n",ownership_range.first,ownership_range.second);
  fprintf(stderr,"Vec Range=%d,%d\n",local_range.first,local_range.second);

  std::map<uint,uint> local_map;

  std::stringstream tstr;

  // make a map of global to local vertices, ignoring any off process
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const uint i = cell->index();
      std::vector<uint> vv = dofmap.cell_dofs(i);
      std::vector<uint>::iterator vvc=vv.begin();
      for (VertexIterator v(*cell); !v.end(); ++v){
	tstr << *vvc << " ";
       	if (off_process_owner.find(*vvc)==off_process_owner.end())
	  {
	    //must be on this process
	    //	    fprintf(stderr,"G%d=L%d\n",*vvc,v->index());
	    local_map.insert(std::pair<uint,uint>(*vvc,v->index()));
	  }
	++vvc;
      }
      tstr << std::endl;
    }


  s.str("");

  // written for 2D only!!!
  const double *m_coords = mesh.coordinates();
  
  for(uint i=local_range.first;i<local_range.second;i++)
    {
      uint j=local_map[i];
      //      fprintf(stderr,"G%d=L%d\n",i,j);
      //      fprintf(stderr,"%5f %5f\n",m_coords[j*2],m_coords[j*2+1]);
      s << m_coords[j*2] << " " << m_coords[j*2+1] << " 0" <<  std::endl;
    }

  // probably better to save as a hdf5 field
  std::vector<std::string> ss,tss;
  
  MPI::gather(s.str(),ss);
  MPI::gather(tstr.str(),tss);

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
    xdmf_topo.append_attribute("TopologyType")="Triangle";
    xdmf_topo.append_attribute("NumberOfElements")=num_global_cells;
    pugi::xml_node xdmf_topo_data = xdmf_topo.append_child("DataItem");
    xdmf_topo_data.append_attribute("Format")="XML";

    s.str("");
    s << num_global_cells << " " << (cell_dim+1);
    xdmf_topo_data.append_attribute("Dimensions")=s.str().c_str();
    s.str("");
    s << std::endl;
    
    std::vector<std::string>::iterator ss_it;
    for (ss_it=tss.begin(); ss_it!=tss.end(); ++ss_it)
      s << *ss_it;


    xdmf_topo_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());
    
    pugi::xml_node xdmf_geom = xdmf_grid.append_child("Geometry");
    xdmf_geom.append_attribute("GeometryType")="XYZ";
    pugi::xml_node xdmf_geom_data = xdmf_geom.append_child("DataItem");
    xdmf_geom_data.append_attribute("Format")="XML";
    s.str("");
    s << num_global_vertices << " 3 3";
    xdmf_geom_data.append_attribute("Dimensions")=s.str().c_str();

    s.str("");

    for (ss_it=ss.begin(); ss_it!=ss.end(); ++ss_it)
      s << *ss_it;

    xdmf_geom_data.append_child(pugi::node_pcdata).set_value(s.str().c_str());
    
    pugi::xml_node xdmf_vals=xdmf_grid.append_child("Attribute"); //actual data
    xdmf_vals.append_attribute("Name")="Vertex Data";
    xdmf_vals.append_attribute("Center")="Node";
    pugi::xml_node xdmf_data=xdmf_vals.append_child("DataItem");
    xdmf_data.append_attribute("Format")="HDF";
    s.str("");
    s << num_global_vertices;
    xdmf_data.append_attribute("Dimensions")=s.str().c_str();
    s.str("\npoisson.h5:/dolfin_vector\n");
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
