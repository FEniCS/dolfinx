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
  case marked_for_reg_ref:
    refineRegular(cell, grid);
    break;
  case marked_for_irr_ref_1:
    refineIrregular1(cell, grid);
    break;
  case marked_for_irr_ref_2:
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

  cell.marker() = marked_for_reg_ref;
  return true;
}
//-----------------------------------------------------------------------------
bool TriGridRefinement::checkRuleIrregular1(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 1

  if ( no_marked_edges != 1 )
    return false;

  cell.marker() = marked_for_irr_ref_1;
  return true;
}
//-----------------------------------------------------------------------------
bool TriGridRefinement::checkRuleIrregular2(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 2

  if ( no_marked_edges != 2 )
    return false;

  cell.marker() = marked_for_irr_ref_2;
  return true;
}
//-----------------------------------------------------------------------------
void TriGridRefinement::refineRegular(Cell& cell, Grid& grid)
{
  // Refine one triangle into four new ones, introducing new nodes 
  // at the midpoints of the edges. 

  // Create new nodes with the same coordinates as the previous nodes in cell  
  Node* n0 = grid.createNode(cell.coord(0));
  Node* n1 = grid.createNode(cell.coord(1));
  Node* n2 = grid.createNode(cell.coord(2));

  // Update parent-child info 
  n0->setParent(cell.node(0));
  n1->setParent(cell.node(1));
  n2->setParent(cell.node(2));

  cell.node(0)->setParent(n0);
  cell.node(1)->setParent(n1);
  cell.node(2)->setParent(n2);
  
  // Create new nodes with the new coordinates 
  Node* n01 = grid.createNode(cell.node(0)->midpoint(*cell.node(1)));
  Node* n02 = grid.createNode(cell.node(0)->midpoint(*cell.node(2)));
  Node* n12 = grid.createNode(cell.node(1)->midpoint(*cell.node(2)));

  // Create new cells 
  Cell* t1 = grid.createCell(n0,  n01, n02);
  Cell* t2 = grid.createCell(n01, n1,  n12);
  Cell* t3 = grid.createCell(n02, n12, n2 );
  Cell* t4 = grid.createCell(n01, n12, n02);

  // Update parent-child info 
  t1->setParent(&cell);
  t2->setParent(&cell);
  t3->setParent(&cell);
  t4->setParent(&cell);

  cell.addChild(t1);
  cell.addChild(t2);
  cell.addChild(t3);
  cell.addChild(t4);
}
//-----------------------------------------------------------------------------
void TriGridRefinement::refineIrregular1(Cell& cell, Grid& grid)
{
  // One edge is marked. Insert one new node at the midpoint of the
  // marked edge, then connect this new node to the node not on
  // the marked edge. This gives 2 new triangles.

  // Sort nodes by the number of marked edges
  Array<Node*> nodes(3);
  sortNodes(cell, nodes);

  // Create new nodes with the same coordinates as the old nodes
  Node* n_m1 = grid.createNode(nodes(0)->coord());
  Node* n_m2 = grid.createNode(nodes(1)->coord());
  Node* n_nm = grid.createNode(nodes(2)->coord());

  // Update parent-child info
  n_m1->setParent(nodes(0));
  n_m2->setParent(nodes(1));
  n_nm->setParent(nodes(2));

  // Create new node on marked edge 
  Node* n_e = grid.createNode(cell.findEdge(n_m1,n_m2)->midpoint());
  
  // Create new cells 
  Cell* t1 = grid.createCell(n_e, n_nm, n_m1);
  Cell* t2 = grid.createCell(n_e, n_nm, n_m2);
  
  // Update parent-child info
  t1->setParent(&cell);
  t2->setParent(&cell);
  
  cell.addChild(t1);
  cell.addChild(t2);
}
//-----------------------------------------------------------------------------
void TriGridRefinement::refineIrregular2(Cell& cell, Grid& grid)
{
  // Two edges are marked. Insert two new nodes at the midpoints of the
  // marked edges, then connect these new nodes to each other and one 
  // of the nodes on the unmarked edge. This gives 3 new triangles.

  // Sort nodes by the number of marked edges
  Array<Node*> nodes(3);
  sortNodes(cell, nodes);

  // Create new nodes with the same coordinates as the old nodes
  Node* n_dm = grid.createNode(nodes(0)->coord());
  Node* n_m1 = grid.createNode(nodes(1)->coord());
  Node* n_m2 = grid.createNode(nodes(2)->coord());

  // Update parent-child info
  n_dm->setParent(nodes(0));
  n_m1->setParent(nodes(1));
  n_m2->setParent(nodes(2));

  // Create new nodes on marked edges 
  Node* n_e1 = grid.createNode(cell.findEdge(n_dm,n_m1)->midpoint());
  Node* n_e2 = grid.createNode(cell.findEdge(n_dm,n_m2)->midpoint());

  // Create new cells 
  Cell* t1 = grid.createCell(n_dm, n_e1, n_e2);
  Cell* t2 = grid.createCell(n_m1, n_e1, n_e2);
  Cell* t3 = grid.createCell(n_e2, n_m1, n_m2);
  
  // Update parent-child info
  t1->setParent(&cell);
  t2->setParent(&cell);
  t3->setParent(&cell);
  
  cell.addChild(t1);
  cell.addChild(t2);
  cell.addChild(t3);
}
//-----------------------------------------------------------------------------
