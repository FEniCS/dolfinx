// Copyright (C) 2008 Johan Jansson
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
// Modified by Bartosz Sawicki, 2009.
// Modified by Garth N. Wells, 2010.
// Modified by Anders Logg, 2010.
//
// First added:  2008
// Last changed: 2011-03-17

#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include "RivaraRefinement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void RivaraRefinement::refine(Mesh& refined_mesh,
                              const Mesh& mesh,
                              const MeshFunction<bool>& cell_marker,
                              MeshFunction<std::size_t>& cell_map,
                              std::vector<int>& facet_map)
{
  log(TRACE, "Refining simplicial mesh by recursive Rivara bisection.");

  // Create dynamic mesh and import data
  DMesh dmesh;
  dmesh.import_mesh(mesh);

  // Rewrite MeshFunction into vector
  std::vector<bool> dmarked(mesh.num_cells());
  for (CellIterator ci(mesh); !ci.end(); ++ci)
  {
    if(cell_marker[*ci] == true)
      dmarked[ci->index()] = true;
    else
      dmarked[ci->index()] = false;
  }

  // Main refinement algorithm
  dmesh.bisect_marked(dmarked);

  // Remove deleted cells from global list
  for(std::list<DCell* >::iterator it = dmesh.cells.begin(); it != dmesh.cells.end(); )
  {
    DCell* dc = *it;
    if(dc->deleted)
    {
      delete *it;
      it = dmesh.cells.erase(it);
    }
    else
      it++;
  }

  // Vector for cell and facet mappings
  std::vector<int> new2old_cell_arr;
  std::vector<int> new2old_facet_arr;

  // Refine mesh
  dmesh.export_mesh(refined_mesh, new2old_cell_arr, new2old_facet_arr);

  // Generate cell mesh function map
  cell_map.init(refined_mesh, mesh.topology().dim());
  for (CellIterator c(refined_mesh); !c.end(); ++c)
    cell_map[*c] = new2old_cell_arr[c->index()];

  // Generate facet map array
  std::vector<int> new_facet_map(new2old_facet_arr.size());
  facet_map = new_facet_map;
  for (std::size_t i = 0; i < new2old_facet_arr.size(); i++)
  {
    facet_map[i] = new2old_facet_arr[i];
  }

}
//-----------------------------------------------------------------------------
RivaraRefinement::DVertex::DVertex() : id(0), cells(0), p(0.0, 0.0, 0.0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
RivaraRefinement::DCell::DCell() : id(0), parent_id(0), vertices(0),
                                   deleted(false), facets(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
RivaraRefinement::DMesh::DMesh() : vertices(0), cells(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
RivaraRefinement::DMesh::~DMesh()
{
  // Delete allocated DVertices
  for (std::list<DVertex* >::iterator it = vertices.begin();
      it != vertices.end(); ++it)
  {
    delete *it;
  }
  // Delete allocated DCells
  for (std::list<DCell* >::iterator it = cells.begin();
      it != cells.end(); ++it)
  {
    delete *it;
  }
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::import_mesh(const Mesh& mesh)
{
  cell_type = &(mesh.type());
  tdim = mesh.topology().dim();
  gdim = mesh.geometry().dim();
  vertices.clear();
  cells.clear();

  // Import vertices
  std::vector<DVertex *> vertexvec;
  for (VertexIterator vi(mesh); !vi.end(); ++vi)
  {
    DVertex* dv = new DVertex;
    dv->p = vi->point();
    dv->id = vi->index();

    add_vertex(dv);
    vertexvec.push_back(dv);
  }

  // Import cells
  for (CellIterator ci(mesh); !ci.end(); ++ci)
  {
    DCell* dc = new DCell;
    std::vector<DVertex*> vs(ci->num_entities(0));
    std::size_t counter = 0;
    for (VertexIterator vi(*ci); !vi.end(); ++vi)
    {
      DVertex* dv = vertexvec[vi->index()];
      vs[counter] = dv;
      counter++;
    }

    // Initialize facets
    for (std::size_t i = 0; i < cell_type->num_entities(0); i++)
      dc->facets.push_back(i);

    add_cell(dc, vs, ci->index());

    // Define the same cell numbering
    dc->id = ci->index();
  }
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::export_mesh(Mesh& mesh,
     std::vector<int>& new2old_cell, std::vector<int>& new2old_facet)
{
  number();

  new2old_cell.resize(cells.size());
  new2old_facet.resize(cells.size()*cell_type->num_entities(0));

  MeshEditor editor;
  editor.open(mesh, cell_type->cell_type(), tdim, gdim);

  editor.init_vertices(vertices.size());
  editor.init_cells(cells.size());

  // Add vertices
  std::size_t current_vertex = 0;
  for (std::list<DVertex*>::iterator it = vertices.begin();
        it != vertices.end(); ++it)
  {
    DVertex* dv = *it;
    editor.add_vertex(current_vertex, dv->p);
    current_vertex++;
  }

  std::vector<std::size_t> cell_vertices(cell_type->num_entities(0));
  std::size_t current_cell = 0;
  for (std::list<DCell*>::iterator it = cells.begin();
        it != cells.end(); ++it)
  {
    const DCell* dc = *it;

    for (std::size_t j = 0; j < dc->vertices.size(); j++)
    {
      const DVertex* dv = dc->vertices[j];
      cell_vertices[j] = dv->id;
    }
    editor.add_cell(current_cell, cell_vertices);
    new2old_cell[current_cell] = dc->parent_id;

    for (std::size_t j = 0; j < dc->facets.size(); j++)
    {
      std::size_t index = cell_type->num_entities(0)*current_cell + j;
      new2old_facet[ index ] = dc->facets[j];
    }

    current_cell++;
  }
  editor.close();
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::number()
{
  std::size_t i = 0;
  for (std::list<DCell*>::iterator it = cells.begin();
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
  bool closing = false;

  // Find longest edge
  double lmax = 0.0;
  std::size_t ii = 0;
  std::size_t jj = 0;
  for(std::size_t i = 0; i < dcell->vertices.size(); i++)
  {
    for(std::size_t j = 0; j < dcell->vertices.size(); j++)
    {
      if(i != j)
      {
        const double l = dcell->vertices[i]->p.distance(dcell->vertices[j]->p);
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
    mv->id = vertices.size();
    add_vertex(mv);
    closing = false;
  }

  if(ii>jj)
  {
    std::size_t tmp = ii;
    ii = jj;
    jj = tmp;
  }

  // Create new cells & keep them ordered
  DCell* c0 = new DCell;
  DCell* c1 = new DCell;
  std::vector<DVertex*> vs0(0);
  std::vector<DVertex*> vs1(0);
  bool pushed0 = false;
  bool pushed1 = false;
  for(std::size_t i = 0; i < dcell->vertices.size(); i++)
  {
    if (i != ii)
    {
      if (mv->id < dcell->vertices[i]->id && !pushed1)
      {
        vs1.push_back(mv);
        pushed1 =  true;
      }
      vs1.push_back(dcell->vertices[i]);
    }
    if(i != jj)
    {
      if( (mv->id < dcell->vertices[i]->id) && !pushed0 )
      {
        vs0.push_back(mv);
        pushed0 = true;
      }
      vs0.push_back(dcell->vertices[i]);
    }
  }
  if( !pushed0 )
    vs0.push_back(mv);
  if( !pushed1 )
    vs1.push_back(mv);
  add_cell(c0, vs0, dcell->parent_id);
  add_cell(c1, vs1, dcell->parent_id);

  propagate_facets(dcell, c0, c1, ii, jj, mv);

  remove_cell(dcell);

  // Continue refinement
  if (!closing)
  {
    // Bisect opposite cell of edge with hanging node
    for (;;)
    {
      DCell* copp = opposite(dcell, v0, v1);
      if (copp != 0)
        bisect(copp, mv, v0, v1);
      else
        break;
    }
  }
}
//-----------------------------------------------------------------------------
RivaraRefinement::DCell* RivaraRefinement::DMesh::opposite(DCell* dcell,
                  DVertex* v1, DVertex* v2)
{
  for (std::list<DCell*>::iterator it = v1->cells.begin();
      it != v1->cells.end(); ++it)
  {
    DCell* c = *it;
    if (c != dcell)
    {
      std::size_t matches = 0;
      for (std::size_t i = 0; i < c->vertices.size(); i++)
      {
        if (c->vertices[i] == v1 || c->vertices[i] == v2)
          matches++;
      }
      if (matches == 2)
        return c;
    }
  }
  return 0;
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::add_vertex(DVertex* v)
{
  vertices.push_back(v);
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::add_cell(DCell* c, std::vector<DVertex*> vs,
                                       int parent_id)
{
  for (std::size_t i = 0; i < vs.size(); i++)
  {
    DVertex* v = vs[i];
    c->vertices.push_back(v);
    v->cells.push_back(c);
  }

  cells.push_back(c);
  c->parent_id = parent_id;
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::remove_cell(DCell* c)
{
  for (std::size_t i = 0; i < c->vertices.size(); ++i)
  {
    DVertex* v = c->vertices[i];
    v->cells.remove(c);
  }
  c->deleted = true;
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::bisect_marked(std::vector<bool> marked_ids)
{
  std::list<DCell*> marked_cells;
  for (std::list<DCell*>::iterator it = cells.begin();
      it != cells.end(); ++it)
  {
    DCell* c = *it;
    if (marked_ids[c->id])
      marked_cells.push_back(c);
  }
  for (std::list<DCell*>::iterator it = marked_cells.begin();
      it != marked_cells.end(); ++it)
  {
    DCell* c = *it;
    if(!c->deleted)
      bisect(c, 0, 0, 0);
  }
}
//-----------------------------------------------------------------------------
void RivaraRefinement::DMesh::propagate_facets(DCell* dcell, DCell* c0,
                       DCell* c1, std::size_t ii, std::size_t jj, DVertex* mv)
{
  // Initialize local facets
  std::vector<int> facets0(tdim + 1);
  std::vector<int> facets1(tdim + 1);
  for(std::size_t i = 0; i < tdim + 1; i++)
  {
    facets0[i] = -2;
    facets1[i] = -2;
  }

  // New facets
  if (mv->id < dcell->vertices[ii]->id)
    facets0[ii+1] = -1;
  else
    facets0[ii] = -1;
  if (mv->id < dcell->vertices[jj]->id)
    facets1[jj] = -1;
  else
    facets1[jj-1] = -1;

  // Changed facets
  int c0i = 0;
  int c1i = 0;
  for (std::size_t i = 0; i < tdim + 1; i++)
  {
    if ( mv->id > c0->vertices[i]->id )
      c0i++;
    if ( mv->id > c1->vertices[i]->id )
      c1i++;
  }
  facets0[c0i] = jj;
  facets1[c1i] = ii;

  // Untouched facets
  std::vector<int> rest;
  for (std::size_t i = 0; i < tdim + 1; i++)
  {
    if(i != ii && i != jj)
      rest.push_back(i);
  }
  int j = 0, k = 0;
  for(std::size_t i = 0; i < tdim + 1; i++)
  {
    if(facets0[i] == -2)
      facets0[i] = rest[j++];
    if(facets1[i] == -2)
      facets1[i] = rest[k++];
  }

  // Rewrite facets whenever different that -1
  //   ( -1 for new, internal facets )
  for (std::size_t i = 0; i < tdim + 1; i++)
  {
    if (facets0[i] != -1)
      c0->facets.push_back(dcell->facets[facets0[i]]);
    else
      c0->facets.push_back(-1);

    if (facets1[i] != -1)
      c1->facets.push_back(dcell->facets[facets1[i]]);
    else
      c1->facets.push_back(-1);
  }
}
//-----------------------------------------------------------------------------
