// Copyright (C) 2009-2011 Anders Logg
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
// First added:  2009-02-11
// Last changed: 2013-02-11

#include <map>
#include <vector>

#include "Cell.h"
#include "Mesh.h"
#include "MeshEditor.h"
#include "MeshFunction.h"
#include "SubDomain.h"
#include "SubMesh.h"
#include "Vertex.h"
#include "MeshValueCollection.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SubMesh::SubMesh(const Mesh& mesh, const SubDomain& sub_domain)
{
  // Create mesh function and mark sub domain
  MeshFunction<std::size_t> sub_domains(mesh, mesh.topology().dim());
  sub_domains = 0;
  sub_domain.mark(sub_domains, 1);

  // Create sub mesh
  init(mesh, sub_domains, 1);
}
//-----------------------------------------------------------------------------
SubMesh::SubMesh(const Mesh& mesh,
                 const MeshFunction<std::size_t>& sub_domains, std::size_t sub_domain)
{
  // Create sub mesh
  init(mesh, sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
SubMesh::SubMesh(const Mesh& mesh, std::size_t sub_domain)
{
  
  if (mesh.domains().num_marked(mesh.topology().dim())==0)
  {
    dolfin_error("SubMesh.cpp",
                 "construct SubMesh",
                 "Mesh does not include a MeshValueCollection the cell dimension of the mesh");
  } 
  
  // Get MeshFunction from MeshValueCollection
  boost::shared_ptr<const MeshFunction<std::size_t> > sub_domains = \
    mesh.domains().cell_domains();
  
  // Create sub mesh
  init(mesh, *sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
SubMesh::~SubMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SubMesh::init(const Mesh& mesh,
                   const MeshFunction<std::size_t>& sub_domains, std::size_t sub_domain)
{
  // Open mesh for editing
  MeshEditor editor;
  const std::size_t cell_dim_t = mesh.topology().dim();
  editor.open(*this, mesh.type().cell_type(),
              cell_dim_t, mesh.geometry().dim());

  // Extract cells
  std::set<std::size_t> cells;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if (sub_domains[*cell] == sub_domain)
      cells.insert(cell->index());
  }

  // Map to keep track of new local indices for vertices and cells
  std::map<std::size_t, std::size_t> local_vertex_indices;
  std::vector<std::size_t> local_cell_indices;
  local_cell_indices.reserve(cells.size());
  MeshFunction<std::size_t> parent_to_local_cell_indices(mesh, cell_dim_t);

  // Add cells
  editor.init_cells(cells.size());
  std::size_t current_cell = 0;
  std::size_t current_vertex = 0;
  for (std::set<std::size_t>::iterator cell_it = cells.begin();
       cell_it != cells.end(); ++cell_it)
  {
    std::vector<std::size_t> cell_vertices;
    Cell cell(mesh, *cell_it);
    for (VertexIterator vertex(cell); !vertex.end(); ++vertex)
    {
      const std::size_t parent_vertex_index = vertex->index();
      std::size_t local_vertex_index = 0;
      std::map<std::size_t, std::size_t>::iterator vertex_it
        = local_vertex_indices.find(parent_vertex_index);
      if (vertex_it != local_vertex_indices.end())
        local_vertex_index = vertex_it->second;
      else
      {
        local_vertex_index = current_vertex++;
        local_vertex_indices[parent_vertex_index] = local_vertex_index;
      }
      cell_vertices.push_back(local_vertex_index);
    }
    local_cell_indices.push_back(cell.index());
    parent_to_local_cell_indices[*cell_it] = current_cell;
    editor.add_cell(current_cell++, cell_vertices);
  }

  // Add vertices
  editor.init_vertices(local_vertex_indices.size());
  for (std::map<std::size_t, std::size_t>::iterator it = local_vertex_indices.begin();
       it != local_vertex_indices.end(); ++it)
  {
    Vertex vertex(mesh, it->first);
    if (MPI::num_processes() > 1)
      error("SubMesh::init not working in parallel");

    // FIXME: Get global vertex index
    editor.add_vertex(it->second, vertex.point());
  }

  // Close editor
  editor.close();

  // Build local-to-parent mapping for vertices
  boost::shared_ptr<MeshFunction<std::size_t> > parent_vertex_indices
    = data().create_mesh_function("parent_vertex_indices", 0);
  for (std::map<std::size_t, std::size_t>::iterator it = local_vertex_indices.begin();
       it != local_vertex_indices.end(); ++it)
  {
    (*parent_vertex_indices)[it->second] = it->first;
  }

  // Build local-to-parent mapping for cells
  current_cell = 0;
  boost::shared_ptr<MeshFunction<std::size_t> > parent_cell_indices
    = data().create_mesh_function("parent_cell_indices", cell_dim_t);
  for (std::vector<std::size_t>::iterator it = local_cell_indices.begin();
       it != local_cell_indices.end(); ++it)
  {
    (*parent_cell_indices)[current_cell++] = *it;
  }
  
  // Init present MeshValueCollection
  const MeshDomains& parent_domains = mesh.domains();
  this->domains().init(parent_domains.max_dim());

  // Collect MeshValueCollections from parent mesh
  for (std::size_t dim_t=0; dim_t <= parent_domains.max_dim(); dim_t++)
  {
    
    // If parent mesh does not has a MeshValueCollection for dim_t
    if (parent_domains.num_marked(dim_t)==0)
      continue;
    
    // Special case when local entity does not map to vertex number
    if (cell_dim_t==3 && dim_t==1)
      continue;

    // Get local markers
    boost::shared_ptr<MeshValueCollection<std::size_t> > local_markers = \
    	this->domains().markers(dim_t);
    
    // Get values map from parent MeshValueCollection
    const std::map<std::pair<std::size_t, std::size_t>, std::size_t>& values = \
    	parent_domains.markers(dim_t)->values();

    // Parent connectivity to map local entity to vertex value
    const MeshConnectivity& parent_conn = mesh.topology()(cell_dim_t, 0);

    // Iterate over all values in parent MeshValueCollection
    std::map<std::pair<std::size_t, std::size_t>, std::size_t>::const_iterator itt;
    for (itt = values.begin(); itt != values.end(); itt++)
    {
    	
      // Check if the cell is included in the submesh
      if (sub_domains[itt->first.first] == sub_domain)
      {

	// If Cell MeshValueCollection
	if (dim_t == cell_dim_t)
	{
	  // Set new value based on local cell
	  local_markers->set_value(parent_to_local_cell_indices[itt->first.first], 
				   0, itt->second);
	}

	// Facet or Vertex MeshValueCollection
	else
	{
	  
	  // Local Cell
	  const Cell local_cell(*this, parent_to_local_cell_indices[itt->first.first]);

	  // Find what parent vertex the local entity of the MeshValueCollection map to
	  // and the equivalent local vertex index
	  const std::size_t parent_vertex_index = parent_conn(itt->first.first) \
	    [itt->first.second];
	  std::map<std::size_t, std::size_t>::iterator vertex_it \
	    = local_vertex_indices.find(parent_vertex_index);
	  const std::size_t local_vertex_index = vertex_it->second;
    	    
	  // Find what new local entity index the local vertex maps to
	  std::size_t local_entity_index;
	  for (local_entity_index=0; local_entity_index<local_cell.num_entities(0); 
	       local_entity_index++)
	    if (local_cell.entities(0)[local_entity_index] == local_vertex_index)
	      break;

	  // Set new value based on local cell and new local entity index
	  local_markers->set_value(local_cell.index(), local_entity_index, itt->second);

	}
	
      }
    
    }

  }
  
}
//-----------------------------------------------------------------------------
