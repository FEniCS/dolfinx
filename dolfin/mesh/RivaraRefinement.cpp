// Copyright (C) 2008 Johan Jansson
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Bartosz Sawicki, 2009.

#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include "RivaraRefinement.h"


using namespace dolfin;

//-----------------------------------------------------------------------------
void RivaraRefinement::refine(Mesh& mesh, 
			      MeshFunction<bool>& cell_marker,
			      MeshFunction<uint>& cell_map,
			      Array<int>& facet_map)
{
  message("Refining simplicial mesh by recursive Rivara bisection.");

  int dim = mesh.topology().dim();

  // Create dynamic mesh and import data
  DMesh dmesh;
  dmesh.importMesh(mesh);

  // Rewrite MeshFunction into vector
  std::vector<bool> dmarked(mesh.numCells());
  for (CellIterator ci(mesh); !ci.end(); ++ci)
  {
    if(cell_marker.get(ci->index()) == true)
    {
      dmarked[ci->index()] = true;
    }  
    else
    {
      dmarked[ci->index()] = false;
    }
  }

  // Main refinement algorithm
  dmesh.bisectMarked(dmarked);

  // Remove deleted cells from global list
  for(std::list<DCell* >::iterator it = dmesh.cells.begin();
      it != dmesh.cells.end(); )
  {  
    DCell* dc = *it;
    if(dc->deleted)
      it = dmesh.cells.erase(it);
    else
      it++;
  }  

  // Vector for cell and facet mappings
  std::vector<int> new2old_cell_arr;
  std::vector<int> new2old_facet_arr;

  Mesh omesh;

  dmesh.exportMesh(omesh, new2old_cell_arr, new2old_facet_arr);

  mesh = omesh;

  // Generate cell mesh function map
  cell_map.init(mesh, dim);
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    cell_map.set(c->index(), new2old_cell_arr[c->index()]);
  }

  //Generate facet map array 
  Array<int> new_facet_map(new2old_facet_arr.size());
  facet_map = new_facet_map;
  for (uint i=0; i<new2old_facet_arr.size(); i++ )
  {
    facet_map[i] = new2old_facet_arr[i];
  }

}
//-----------------------------------------------------------------------------

RivaraRefinement::DVertex::DVertex() : id(0), cells(0), p(0.0, 0.0, 0.0)
{
}
//-----------------------------------------------------------------------------
RivaraRefinement::DCell::DCell() : id(0), parent_id(0), vertices(0), deleted(false), facets(0)
{
}
//-----------------------------------------------------------------------------
RivaraRefinement::DMesh::DMesh() : vertices(0), cells(0)
{
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::importMesh(Mesh& mesh)
{
  cell_type = &(mesh.type());
  dim = mesh.topology().dim();
  vertices.clear();
  cells.clear();

  // Import vertices
  std::vector<DVertex *> vertexvec;
  for (VertexIterator vi(mesh); !vi.end(); ++vi)
  {
    DVertex* dv = new DVertex;
    dv->p = vi->point();
    
    addVertex(dv);
    vertexvec.push_back(dv);
  }

  // Import cells
  for (CellIterator ci(mesh); !ci.end(); ++ci)
  {
    DCell* dc = new DCell;

    std::vector<DVertex*> vs(ci->numEntities(0));
    uint i = 0;
    for (VertexIterator vi(*ci); !vi.end(); ++vi)
    {
      DVertex* dv = vertexvec[vi->index()];
      vs[i] = dv;
      i++;
    }
    
    // Initialize facets
    for (uint i=0; i < cell_type->numEntities(0); i++)
    {
      dc->facets.push_back(i);
    }
    
    addCell(dc, vs, ci->index());

    // Define the same cell numbering
    dc->id = ci->index();
  }
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::exportMesh(Mesh& mesh, 
     std::vector<int>& new2old_cell, std::vector<int>& new2old_facet)
{
  number();

  new2old_cell.resize(cells.size());
  new2old_facet.resize(cells.size()*cell_type->numEntities(0));

  MeshEditor editor;
  editor.open(mesh, cell_type->cellType(), dim, dim);
  
  editor.initVertices(vertices.size());
  editor.initCells(cells.size());

  // Add vertices
  uint current_vertex = 0;
  for(std::list<DVertex* >::iterator it = vertices.begin();
      it != vertices.end(); ++it)
  {
    DVertex* dv = *it;
    editor.addVertex(current_vertex++, dv->p);
  }

  Array<uint> cell_vertices(cell_type->numEntities(0));
  uint current_cell = 0;
  for(std::list<DCell* >::iterator it = cells.begin();
      it != cells.end(); ++it)
  {
    DCell* dc = *it;

    for(uint j = 0; j < dc->vertices.size(); j++)
    {
      DVertex* dv = dc->vertices[j];
      cell_vertices[j] = dv->id;
    }
    editor.addCell(current_cell, cell_vertices);
    new2old_cell[current_cell] = dc->parent_id;
  
    for(uint j = 0; j < dc->facets.size(); j++)
    {
      uint index = cell_type->numEntities(0)*current_cell + j;
      new2old_facet[ index ] = dc->facets[j];
    }  

    current_cell++;
  }
  editor.close();
   
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::number()
{
  uint i = 0;
  for(std::list<DVertex* >::iterator it = vertices.begin();
      it != vertices.end(); ++it)
  {
    DVertex* dv = *it;
    dv->id = i;
    i++;
  }

  i = 0;
  for(std::list<DCell* >::iterator it = cells.begin();
      it != cells.end(); ++it)
  {
    DCell* dc = *it;
    dc->id = i;
    i++;   
  }
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::bisect(DCell* dcell, DVertex* hangv,
		   DVertex* hv0, DVertex* hv1)
{
  //cout << "Refining cell: " << dcell->id << endl;

  bool closing = false;

  // Find longest edge
  real lmax = 0.0;
  uint ii = 0;
  uint jj = 0;
  for(uint i = 0; i < dcell->vertices.size(); i++)
  {
    for(uint j = 0; j < dcell->vertices.size(); j++)
    {
      if(i != j)
      {
	real l = dcell->vertices[i]->p.distance(dcell->vertices[j]->p);
	if(l >= lmax)
	{
	  ii = i;
	  jj = j;
	  lmax = l;
	}
      }
    }
  }

  DVertex* v0 = dcell->vertices[ii];
  DVertex* v1 = dcell->vertices[jj];

  DVertex* mv = 0;

  // Check if no hanging vertices remain, otherwise create hanging
  // vertex and continue refinement
  if((v0 == hv0 || v0 == hv1) && (v1 == hv0 || v1 == hv1))
  {
    mv = hangv;
    closing = true;
  }
  else
  {
    mv = new DVertex;
    mv->p = (dcell->vertices[ii]->p + dcell->vertices[jj]->p) / 2.0;
    addVertex(mv);
    closing = false;
  }

  if(ii>jj){
    uint tmp = ii;
    ii = jj;
    jj = tmp;
  }

  // Create new cells
  DCell* c0 = new DCell;
  DCell* c1 = new DCell;
  std::vector<DVertex*> vs0(0);
  std::vector<DVertex*> vs1(0);
  for(uint i = 0; i < dcell->vertices.size(); i++)
  {
    if(i != ii)
    {
      vs1.push_back(dcell->vertices[i]);
    }
    if(i != jj)
    {
      vs0.push_back(dcell->vertices[i]);
    }
  }  
  vs0.push_back(mv);
  vs1.push_back(mv);
  
  propagateFacets(dcell, c0, c1, ii, jj);
  
  addCell(c0, vs0, dcell->parent_id);
  addCell(c1, vs1, dcell->parent_id);
  removeCell(dcell);

  // Continue refinement
  if(!closing)
  {
    // Bisect opposite cell of edge with hanging node
    for(;;)
    {
      DCell* copp = opposite(dcell, v0, v1);
      if(copp != 0)
      {
	    bisect(copp, mv, v0, v1);
      }
      else
      {
        break;
      }
    }
  }
}
//-----------------------------------------------------------------------------
RivaraRefinement::DCell* RivaraRefinement::DMesh::opposite(DCell* dcell, 
                  DVertex* v1, DVertex* v2)
{
  for(std::list<DCell* >::iterator it = v1->cells.begin();
      it != v1->cells.end(); ++it)
  {
    DCell* c = *it;

    if(c != dcell)
    {
      int matches = 0;
      for(uint i = 0; i < c->vertices.size(); i++)
      {
        if(c->vertices[i] == v1 || c->vertices[i] == v2)
	{
          matches++;
        }
      }

      if(matches == 2)
      {
        return c;
      }
    }
  }  
  return 0;
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::addVertex(DVertex* v)
{
  vertices.push_back(v);
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::addCell(DCell* c, std::vector<DVertex*> vs, 
                                      int parent_id)
{
  for(uint i = 0; i < vs.size(); i++)
  {
    DVertex* v = vs[i];
    c->vertices.push_back(v);
    v->cells.push_back(c);
  }

  cells.push_back(c);
  c->parent_id = parent_id;
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::removeCell(DCell* c)
{
  for(uint i = 0; i < c->vertices.size(); ++i)
  {
    DVertex* v = c->vertices[i];
    v->cells.remove(c);
  }  
  c->deleted = true;
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::bisectMarked(std::vector<bool> marked_ids)
{
  std::list<DCell*> marked_cells;
  for(std::list<DCell* >::iterator it = cells.begin();
      it != cells.end(); ++it)
  {
    DCell* c = *it;

    if(marked_ids[c->id])
    {
      marked_cells.push_back(c);
    }
  }
  for(std::list<DCell* >::iterator it = marked_cells.begin();
      it != marked_cells.end(); ++it)
  {
    DCell* c = *it;
    if(!c->deleted)
    {
      bisect(c, 0, 0, 0);
    }
  }
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::propagateFacets(DCell* dcell, DCell* c0, 
                       DCell* c1, uint ii, uint jj)
{
  std::vector<int> facets0(dim+1);
  std::vector<int> facets1(dim+1);
  
  // Last facet has always highest number
  facets0[dim] = jj;
  facets1[dim] = ii;
   
  // New facet
  facets0[ii] = -1;
  facets1[jj-1] = -1;

  // Untouched facets
  std::vector<int> rest;
  for(uint i = 0; i < dim+1; i++)
    if(i != ii && i != jj) {
      rest.push_back(i);
    }
  int j=0, k=0;
  for(uint i = 0; i < dim; i++){
    if(i != ii)
      facets0[i] = rest[j++];
    if(i != (jj-1))
      facets1[i] = rest[k++];
  }

  // Rewrite facets whenever different that -1
  //   ( -1 for new, internal facets )
  for(uint i = 0; i < dim+1; i++)
  {
    if(facets0[i] != -1)
    {
      c0->facets.push_back( dcell->facets[facets0[i]] );
    } 
    else 
    {
      c0->facets.push_back( -1 );
    }
    if(facets1[i] != -1)
    {
      c1->facets.push_back( dcell->facets[facets1[i]] );
    } 
    else 
    {
      c1->facets.push_back( -1 );
    }
  }

}



