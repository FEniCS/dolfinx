// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/Cell.h>
#include <dolfin/Node.h>
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
  
  // Create new nodes with the same coordinates as existing nodes
  Node& n0 = createNode(cell.node(0), mesh, cell);
  Node& n1 = createNode(cell.node(1), mesh, cell);
  Node& n2 = createNode(cell.node(2), mesh, cell);

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
  // Refine one triangle into four new ones, introducing new nodes 
  // at the midpoints of the edges. 

  // Check that cell's parent is not refined irregularly, 
  // since then it should be further refined
  dolfin_assert(okToRefine(cell));

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_reg_ref);

  // Create new nodes with the same coordinates as the previous nodes in cell  
  Node& n0 = createNode(cell.node(0), mesh, cell);
  Node& n1 = createNode(cell.node(1), mesh, cell);
  Node& n2 = createNode(cell.node(2), mesh, cell);

  // Create new nodes with the new coordinates 
  Node& n01 = createNode(cell.node(0).midpoint(cell.node(1)), mesh, cell);
  Node& n02 = createNode(cell.node(0).midpoint(cell.node(2)), mesh, cell);
  Node& n12 = createNode(cell.node(1).midpoint(cell.node(2)), mesh, cell);

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
  // One edge is marked. Insert one new node at the midpoint of the
  // marked edge, then connect this new node to the node not on
  // the marked edge. This gives 2 new triangles.

  // Check that cell's parent is not refined irregularly, 
  // since then it should be further refined
  dolfin_assert(okToRefine(cell));

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_irr_ref_1);

  // Sort nodes by the number of marked edges
  PArray<Node*> nodes;
  sortNodes(cell, nodes);

  // Create new nodes with the same coordinates as the old nodes
  Node& n0 = createNode(*nodes(0), mesh, cell);
  Node& n1 = createNode(*nodes(1), mesh, cell);
  Node& nn = createNode(*nodes(2), mesh, cell); // Not marked

  // Find edge
  Edge* e = cell.findEdge(*nodes(0), *nodes(1));
  dolfin_assert(e);

  // Create new node on marked edge 
  Node& ne = createNode(e->midpoint(), mesh, cell);
  
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
  // Two edges are marked. Insert two new nodes at the midpoints of the
  // marked edges, then connect these new nodes to each other and one 
  // of the nodes on the unmarked edge. This gives 3 new triangles.

  // Check that cell's parent is not refined irregularly, 
  // since then it should be further refined
  dolfin_assert(okToRefine(cell));

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_irr_ref_2);

  // Sort nodes by the number of marked edges
  PArray<Node*> nodes;
  sortNodes(cell, nodes);

  // Create new nodes with the same coordinates as the old nodes
  Node& n_dm = createNode(*nodes(0), mesh, cell);
  Node& n_m0 = createNode(*nodes(1), mesh, cell);
  Node& n_m1 = createNode(*nodes(2), mesh, cell);

  // Find the edges
  Edge* e0 = cell.findEdge(*nodes(0), *nodes(1));
  Edge* e1 = cell.findEdge(*nodes(0), *nodes(2));
  dolfin_assert(e0);
  dolfin_assert(e1);

  // Create new nodes on marked edges 
  Node& n_e0 = createNode(e0->midpoint(), mesh, cell);
  Node& n_e1 = createNode(e1->midpoint(), mesh, cell);

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
Cell& TriMeshRefinement::createCell(Node& n0, Node& n1, Node& n2,
				    Mesh& mesh, Cell& cell)
{
  Cell& c = mesh.createCell(n0, n1, n2);
  c.setParent(cell);
  cell.addChild(c);

  return c;
}
//-----------------------------------------------------------------------------
