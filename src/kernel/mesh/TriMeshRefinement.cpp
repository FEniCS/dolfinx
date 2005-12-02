// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2005-12-01

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/Cell.h>
#include <dolfin/Vertex.h>
#include <dolfin/TriMeshRefinement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
bool TriMeshRefinement::checkRule(Cell& cell, int no_marked_edges)
{
  dolfin_assert(cell.type() == Cell::triangle);

  // Choose refinement rule

  if ( checkRuleRegular(cell, no_marked_edges) )
    return true;

  if ( checkRuleIrregular1(cell, no_marked_edges) )
    return true;

  if ( checkRuleIrregular2(cell, no_marked_edges) )
    return true;

  // We didn't find a matching rule for refinement
  return false;
}
//-----------------------------------------------------------------------------
void TriMeshRefinement::refine(Cell& cell, Mesh& mesh)
{
  // Refine cell according to marker
  switch ( cell.marker() ) {
  case Cell::marked_for_no_ref:
    refineNoRefine(cell, mesh);
    break;
  case Cell::marked_for_reg_ref:
    refineRegular(cell, mesh);
    break;
  case Cell::marked_for_irr_ref_1:
    refineIrregular1(cell, mesh);
    break;
  case Cell::marked_for_irr_ref_2:
    refineIrregular2(cell, mesh);
    break;
  default:
    // We should not rearch this case, cell cannot be
    // marked_for_coarsening or marked_according_to_ref
    dolfin_error("Inconsistent cell markers.");
  }
}
//-----------------------------------------------------------------------------
bool TriMeshRefinement::checkRuleRegular(Cell& cell, int no_marked_edges)
{
  // A triangle is refined regularly if all 4 edges are marked.

  if ( no_marked_edges != 4 )
    return false;

  cell.marker() = Cell::marked_for_reg_ref;
  return true;
}
//-----------------------------------------------------------------------------
bool TriMeshRefinement::checkRuleIrregular1(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 1

  if ( no_marked_edges != 1 )
    return false;

  cell.marker() = Cell::marked_for_irr_ref_1;
  return true;
}
//-----------------------------------------------------------------------------
bool TriMeshRefinement::checkRuleIrregular2(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 2

  if ( no_marked_edges != 2 )
    return false;

  cell.marker() = Cell::marked_for_irr_ref_2;
  return true;
}
//-----------------------------------------------------------------------------
void TriMeshRefinement::refineNoRefine(Cell& cell, Mesh& mesh)
{
  // Don't refine the triangle and create a copy in the new mesh.

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_no_ref);
  
  // Create new vertices with the same coordinates as existing vertices
  Vertex& n0 = createVertex(cell.vertex(0), mesh, cell);
  Vertex& n1 = createVertex(cell.vertex(1), mesh, cell);
  Vertex& n2 = createVertex(cell.vertex(2), mesh, cell);

  // Create a new cell
  cell.initChildren(1);
  createCell(n0, n1, n2, mesh, cell);

  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::unref;
}
//-----------------------------------------------------------------------------
void TriMeshRefinement::refineRegular(Cell& cell, Mesh& mesh)
{
  // Refine one triangle into four new ones, introducing new vertices 
  // at the midpoints of the edges. 

  // Check that cell's parent is not refined irregularly, 
  // since then it should be further refined
  dolfin_assert(okToRefine(cell));

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_reg_ref);

  // Create new vertices with the same coordinates as the previous vertices in cell  
  Vertex& n0 = createVertex(cell.vertex(0), mesh, cell);
  Vertex& n1 = createVertex(cell.vertex(1), mesh, cell);
  Vertex& n2 = createVertex(cell.vertex(2), mesh, cell);

  // Create new vertices with the new coordinates 
  Vertex& n01 = createVertex(cell.vertex(0).midpoint(cell.vertex(1)), mesh, cell);
  Vertex& n02 = createVertex(cell.vertex(0).midpoint(cell.vertex(2)), mesh, cell);
  Vertex& n12 = createVertex(cell.vertex(1).midpoint(cell.vertex(2)), mesh, cell);

  // Create new cells 
  cell.initChildren(4);
  createCell(n0,  n01, n02, mesh, cell);
  createCell(n01, n1,  n12, mesh, cell);
  createCell(n02, n12, n2,  mesh, cell);
  createCell(n01, n12, n02, mesh, cell);
  
  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::ref_reg;
}
//-----------------------------------------------------------------------------
void TriMeshRefinement::refineIrregular1(Cell& cell, Mesh& mesh)
{
  // One edge is marked. Insert one new vertex at the midpoint of the
  // marked edge, then connect this new vertex to the vertex not on
  // the marked edge. This gives 2 new triangles.

  // Check that cell's parent is not refined irregularly, 
  // since then it should be further refined
  dolfin_assert(okToRefine(cell));

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_irr_ref_1);

  // Sort vertices by the number of marked edges
  PArray<Vertex*> vertices;
  sortVertices(cell, vertices);

  // Create new vertices with the same coordinates as the old vertices
  Vertex& n0 = createVertex(*vertices(0), mesh, cell);
  Vertex& n1 = createVertex(*vertices(1), mesh, cell);
  Vertex& nn = createVertex(*vertices(2), mesh, cell); // Not marked

  // Find edge
  Edge* e = cell.findEdge(*vertices(0), *vertices(1));
  dolfin_assert(e);

  // Create new vertex on marked edge 
  Vertex& ne = createVertex(e->midpoint(), mesh, cell);
  
  // Create new cells
  cell.initChildren(2); 
  createCell(ne, nn, n0, mesh, cell);
  createCell(ne, nn, n1, mesh, cell);
  
  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::ref_irr;
}
//-----------------------------------------------------------------------------
void TriMeshRefinement::refineIrregular2(Cell& cell, Mesh& mesh)
{
  // Two edges are marked. Insert two new vertices at the midpoints of the
  // marked edges, then connect these new vertices to each other and one 
  // of the vertices on the unmarked edge. This gives 3 new triangles.

  // Check that cell's parent is not refined irregularly, 
  // since then it should be further refined
  dolfin_assert(okToRefine(cell));

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_irr_ref_2);

  // Sort vertices by the number of marked edges
  PArray<Vertex*> vertices;
  sortVertices(cell, vertices);

  // Create new vertices with the same coordinates as the old vertices
  Vertex& n_dm = createVertex(*vertices(0), mesh, cell);
  Vertex& n_m0 = createVertex(*vertices(1), mesh, cell);
  Vertex& n_m1 = createVertex(*vertices(2), mesh, cell);

  // Find the edges
  Edge* e0 = cell.findEdge(*vertices(0), *vertices(1));
  Edge* e1 = cell.findEdge(*vertices(0), *vertices(2));
  dolfin_assert(e0);
  dolfin_assert(e1);

  // Create new vertices on marked edges 
  Vertex& n_e0 = createVertex(e0->midpoint(), mesh, cell);
  Vertex& n_e1 = createVertex(e1->midpoint(), mesh, cell);

  // Create new cells
  cell.initChildren(3); 
  createCell(n_dm, n_e0, n_e1, mesh, cell);
  createCell(n_m0, n_e0, n_e1, mesh, cell);
  createCell(n_e1, n_m0, n_m1, mesh, cell);
  
  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::ref_irr;
}
//-----------------------------------------------------------------------------
Cell& TriMeshRefinement::createCell(Vertex& n0, Vertex& n1, Vertex& n2,
				    Mesh& mesh, Cell& cell)
{
  Cell& c = mesh.createCell(n0, n1, n2);
  c.setParent(cell);
  cell.addChild(c);

  return c;
}
//-----------------------------------------------------------------------------
