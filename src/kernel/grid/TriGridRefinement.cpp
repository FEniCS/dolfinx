// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Grid.h>
#include <dolfin/Cell.h>
#include <dolfin/Node.h>
#include <dolfin/TriGridRefinement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
bool TriGridRefinement::checkRule(Cell& cell, int no_marked_edges)
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
void TriGridRefinement::refine(Cell& cell, Grid& grid)
{
  // Refine cell according to marker
  switch ( cell.marker() ) {
  case Cell::marked_for_no_ref:
    refineNoRefine(cell, grid);
    break;
  case Cell::marked_for_reg_ref:
    refineRegular(cell, grid);
    break;
  case Cell::marked_for_irr_ref_1:
    refineIrregular1(cell, grid);
    break;
  case Cell::marked_for_irr_ref_2:
    refineIrregular1(cell, grid);
    break;
  default:
    // We should not rearch this case, cell cannot be
    // marked_for_coarsening or marked_according_to_ref
    dolfin_error("Inconsistent cell markers.");
  }
}
//-----------------------------------------------------------------------------
bool TriGridRefinement::checkRuleRegular(Cell& cell, int no_marked_edges)
{
  // A triangle is refined regularly if all 4 edges are marked.

  if ( no_marked_edges != 4 )
    return false;

  cell.marker() = Cell::marked_for_reg_ref;
  return true;
}
//-----------------------------------------------------------------------------
bool TriGridRefinement::checkRuleIrregular1(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 1

  if ( no_marked_edges != 1 )
    return false;

  cell.marker() = Cell::marked_for_irr_ref_1;
  return true;
}
//-----------------------------------------------------------------------------
bool TriGridRefinement::checkRuleIrregular2(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 2

  if ( no_marked_edges != 2 )
    return false;

  cell.marker() = Cell::marked_for_irr_ref_2;
  return true;
}
//-----------------------------------------------------------------------------
void TriGridRefinement::refineNoRefine(Cell& cell, Grid& grid)
{
  // Don't refine the triangle and create a copy in the new grid.

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == Cell::marked_for_no_ref);
  
  // Create new nodes with the same coordinates as existing nodes
  Node& n0 = createNode(cell.node(0), grid, cell);
  Node& n1 = createNode(cell.node(1), grid, cell);
  Node& n2 = createNode(cell.node(2), grid, cell);

  // Create a new cell
  grid.createCell(n0, n1, n2);

  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::unref;
}
//-----------------------------------------------------------------------------
void TriGridRefinement::refineRegular(Cell& cell, Grid& grid)
{
  // Refine one triangle into four new ones, introducing new nodes 
  // at the midpoints of the edges. 

  // Create new nodes with the same coordinates as the previous nodes in cell  
  Node& n0 = grid.createNode(cell.coord(0));
  Node& n1 = grid.createNode(cell.coord(1));
  Node& n2 = grid.createNode(cell.coord(2));

  // Create new nodes with the new coordinates 
  Node& n01 = grid.createNode(cell.node(0).midpoint(cell.node(1)));
  Node& n02 = grid.createNode(cell.node(0).midpoint(cell.node(2)));
  Node& n12 = grid.createNode(cell.node(1).midpoint(cell.node(2)));

  // Create new cells 
  grid.createCell(n0,  n01, n02);
  grid.createCell(n01, n1,  n12);
  grid.createCell(n02, n12, n2 );
  grid.createCell(n01, n12, n02);
  
  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::ref_reg;
}
//-----------------------------------------------------------------------------
void TriGridRefinement::refineIrregular1(Cell& cell, Grid& grid)
{
  // One edge is marked. Insert one new node at the midpoint of the
  // marked edge, then connect this new node to the node not on
  // the marked edge. This gives 2 new triangles.

  // Sort nodes by the number of marked edges
  Array<Node*> nodes;
  sortNodes(cell, nodes);

  // Create new nodes with the same coordinates as the old nodes
  Node& n0 = grid.createNode(nodes(0)->coord());
  Node& n1 = grid.createNode(nodes(1)->coord());
  Node& nn = grid.createNode(nodes(2)->coord()); // Not marked

  // Find edge
  Edge* e = cell.findEdge(*nodes(0), *nodes(1));
  dolfin_assert(e);

  // Create new node on marked edge 
  Node& ne = grid.createNode(e->midpoint());
  
  // Create new cells 
  grid.createCell(ne, nn, n0);
  grid.createCell(ne, nn, n1);
  
  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::ref_irr;
}
//-----------------------------------------------------------------------------
void TriGridRefinement::refineIrregular2(Cell& cell, Grid& grid)
{
  // Two edges are marked. Insert two new nodes at the midpoints of the
  // marked edges, then connect these new nodes to each other and one 
  // of the nodes on the unmarked edge. This gives 3 new triangles.

  // Sort nodes by the number of marked edges
  Array<Node*> nodes;
  sortNodes(cell, nodes);

  // Create new nodes with the same coordinates as the old nodes
  Node& n_dm = grid.createNode(nodes(0)->coord());
  Node& n_m0 = grid.createNode(nodes(1)->coord());
  Node& n_m1 = grid.createNode(nodes(2)->coord());

  // Find the edges
  Edge* e0 = cell.findEdge(*nodes(0), *nodes(1));
  Edge* e1 = cell.findEdge(*nodes(0), *nodes(2));
  dolfin_assert(e0);
  dolfin_assert(e1);

  // Create new nodes on marked edges 
  Node& n_e0 = grid.createNode(e0->midpoint());
  Node& n_e1 = grid.createNode(e1->midpoint());

  // Create new cells 
  grid.createCell(n_dm, n_e0, n_e1);
  grid.createCell(n_m0, n_e0, n_e1);
  grid.createCell(n_e1, n_m0, n_m1);
  
  // Set marker of cell
  cell.marker() = Cell::marked_according_to_ref;

  // Set status of cell
  cell.status() = Cell::ref_irr;
}
//-----------------------------------------------------------------------------
