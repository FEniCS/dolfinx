// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Grid.h>
#include <dolfin/Cell.h>
#include <dolfin/Node.h>
#include <dolfin/TetGridRefinement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
bool TetGridRefinement::checkRule(Cell& cell, int no_marked_edges)
{
  dolfin_assert(cell.type() == Cell::triangle);

  // Choose refinement rule
  
  if ( checkRuleRegular(cell, no_marked_edges) )
    return true;

  if ( checkRuleIrregular1(cell, no_marked_edges) )
    return true;

  if ( checkRuleIrregular2(cell, no_marked_edges) )
    return true;

  if ( checkRuleIrregular3(cell, no_marked_edges) )
    return true;

  if ( checkRuleIrregular4(cell, no_marked_edges) )
    return true;

  // We didn't find a matching rule for refinement
  return false;
}
//-----------------------------------------------------------------------------
void TetGridRefinement::refine(Cell& cell, Grid& grid)
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
    refineIrregular2(cell, grid);
    break;
  case marked_for_irr_ref_3:
    refineIrregular3(cell, grid);
    break;
  case marked_for_irr_ref_4:
    refineIrregular4(cell, grid);
    break;
  case marked_for_no_ref:
    // Do nothing
    break;
  default:
    // We should not rearch this case, cell cannot be
    // marked_for_coarsening or marked_according_to_ref
    dolfin_error("Inconsistent cell markers.");
  }
}
//-----------------------------------------------------------------------------
bool TetGridRefinement::checkRuleRegular(Cell& cell, int no_marked_edges)
{
  // Check if cell should be regularly refined.
  // A cell is refined regularly if all edges are marked.

  if ( no_marked_edges != 6 )
    return false;

  cell.marker() = marked_for_reg_ref;
  return true;
}
//-----------------------------------------------------------------------------
bool TetGridRefinement::checkRuleIrregular1(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 1.
  // 

  if ( no_marked_edges != 3 )
    return false;

  if ( !markedEdgesOnSameFace(cell) )
    return false;

  cell.marker() = marked_for_irr_ref_1;
  return true;
}
//-----------------------------------------------------------------------------
bool TetGridRefinement::checkRuleIrregular2(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 2

  if ( no_marked_edges != 1 )
    return false;

  cell.marker() = marked_for_irr_ref_2;
  return true;
}
//-----------------------------------------------------------------------------
bool TetGridRefinement::checkRuleIrregular3(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 3

  if ( no_marked_edges != 2 )
    return false;

  if ( !markedEdgesOnSameFace(cell) )
    return false;

  cell.marker() = marked_for_irr_ref_3;
  return true;
}
//-----------------------------------------------------------------------------
bool TetGridRefinement::checkRuleIrregular4(Cell& cell, int no_marked_edges)
{
  // Check if cell matches irregular refinement rule 4

  if ( no_marked_edges != 3 )
    return false;

  // Note that this has already been checked by checkRule3(), but this
  // way the algorithm is a little cleaner.
  if ( markedEdgesOnSameFace(cell) )
    return false;

  cell.marker() = marked_for_irr_ref_4;
  return true;
}
//-----------------------------------------------------------------------------
void TetGridRefinement::refineRegular(Cell& cell, Grid& grid)
{
  // Refine 1 tetrahedron into 8 new ones, introducing new nodes 
  // at the midpoints of the edges.

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == marked_for_reg_ref);
  
  // Create new nodes with the same coordinates as existing nodes
  Node *n0 = grid.createNode(cell.coord(0));
  Node *n1 = grid.createNode(cell.coord(1));
  Node *n2 = grid.createNode(cell.coord(2));
  Node *n3 = grid.createNode(cell.coord(3));

  // Update parent-child info 
  n0->setParent(cell.node(0));
  n1->setParent(cell.node(1));
  n2->setParent(cell.node(2));
  n3->setParent(cell.node(3));

  // Create new nodes with the new coordinates 
  Node *n01 = grid.createNode(cell.node(0)->midpoint(*cell.node(1)));
  Node *n02 = grid.createNode(cell.node(0)->midpoint(*cell.node(2)));
  Node *n03 = grid.createNode(cell.node(0)->midpoint(*cell.node(3)));
  Node *n12 = grid.createNode(cell.node(1)->midpoint(*cell.node(2)));
  Node *n13 = grid.createNode(cell.node(1)->midpoint(*cell.node(3)));
  Node *n23 = grid.createNode(cell.node(2)->midpoint(*cell.node(3)));

  // Create new cells 
  Cell *t1 = grid.createCell(n0,  n01, n02, n03);
  Cell *t2 = grid.createCell(n01, n1,  n12, n13);
  Cell *t3 = grid.createCell(n02, n12, n2,  n23);
  Cell *t4 = grid.createCell(n03, n13, n23, n3 );
  Cell *t5 = grid.createCell(n01, n02, n03, n13);
  Cell *t6 = grid.createCell(n01, n02, n12, n13);
  Cell *t7 = grid.createCell(n02, n03, n13, n23);
  Cell *t8 = grid.createCell(n02, n12, n13, n23);

  // Update parent-child info 
  t1->setParent(&cell);
  t2->setParent(&cell);
  t3->setParent(&cell);
  t4->setParent(&cell);
  t5->setParent(&cell);
  t6->setParent(&cell);
  t7->setParent(&cell);
  t8->setParent(&cell);
  
  cell.addChild(t1);
  cell.addChild(t2);
  cell.addChild(t3);
  cell.addChild(t4);
  cell.addChild(t5);
  cell.addChild(t6);
  cell.addChild(t7);
  cell.addChild(t8);
}
//-----------------------------------------------------------------------------
void TetGridRefinement::refineIrregular1(Cell& cell, Grid& grid)
{
  // Three edges are marked on the same face. Insert three new nodes
  // at the midpoints on the marked edges, connect the new nodes to
  // each other, as well as to the node that is not on the marked
  // face. This gives 4 new tetrahedrons.

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == marked_for_irr_ref_1);
  
  // Sort nodes by the number of marked edges
  Array<Node*> nodes;
  sortNodes(cell, nodes);

  // Create new nodes with the same coordinates as the old nodes
  Node *n_m1 = grid.createNode(cell.markedNode(0).coord());
  Node *n_m2 = grid.createNode(cell.markedNode(1).coord());
  Node *n_m3 = grid.createNode(cell.markedNode(2).coord());
  Node *n_nm = grid.createNode(cell.nonMarkedNode(0).coord());
         
  // Update parent-child info
  n_m1->setParent(cell.markedNode(0));
  n_m2->setParent(cell.markedNode(1));
  n_m3->setParent(cell.markedNode(2));
  n_nm->setParent(cell.nonMarkedNode(0));

  // Create new nodes on the edges of the marked face
  Node *n_e12 = grid.createNode(cell.findEdge(n_m1,n_m2).midpoint());
  Node *n_e13 = grid.createNode(cell.findEdge(n_m1,n_m3).midpoint());
  Node *n_e23 = grid.createNode(cell.findEdge(n_m2,n_m3).midpoint());
  
  // Create new cells 
  Cell *t1 = grid.createCell(n_nm,n_e12,n_e13,n_e23);
  Cell *t2 = grid.createCell(n_nm,n_e12,n_e13,n_m1);
  Cell *t3 = grid.createCell(n_nm,n_e12,n_e23,n_m2);
  Cell *t4 = grid.createCell(n_nm,n_e13,n_e23,n_m3);
  
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
void TetGridRefinement::refineIrregular2(Cell& cell, Grid& grid)
{
  // One edge is marked. Insert one new node at the midpoint of the
  // marked edge, then connect this new node to the two nodes not on
  // the marked edge. This gives 2 new tetrahedrons.

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == marked_for_irr_ref_2);

  // Create new nodes with the same coordinates as the old nodes
  Node *n_m1  = grid.createNode(cell.markedNode(0).coord());
  Node *n_m2  = grid.createNode(cell.markedNode(1).coord());
  Node *n_nm1 = grid.createNode(cell.nonMarkedNode(0).coord());
  Node *n_nm2 = grid.createNode(cell.nonMarkedNode(1).coord());

  // Update parent-child info
  n_m1->setParent(cell.markedNode(0));
  n_m2->setParent(cell.markedNode(1));
  n_nm1->setParent(cell.nonMarkedNode(0));
  n_nm2->setParent(cell.nonMarkedNode(2));

  // Create new node on marked edge 
  Node *n_e = grid.createNode(cell.findEdge(n_m1,n_m2).midpoint());
  
  // Create new cells 
  Cell *t1 = grid.createCell(n_e,n_nm1,n_nm2,n_m1);
  Cell *t2 = grid.createCell(n_e,n_nm1,n_nm2,n_m2);
  
  // Update parent-child info
  t1->setParent(&cell);
  t2->setParent(&cell);
  
  cell.addChild(t1);
  cell.addChild(t2);
}
//-----------------------------------------------------------------------------
void TetGridRefinement::refineIrregular3(Cell& cell, Grid& grid)
{
  // Two edges are marked, both on the same face. There are two
  // possibilities, and the chosen alternative must match the
  // corresponding face of the neighbor tetrahedron. We insert two new
  // nodes at the midpoints of the marked edges. Three new edges are
  // created by connecting the two new nodes to each other and to the
  // node opposite the face of the two marked edges. Finally, an edge
  // is created by either
  // 
  //   (1) connecting new node 1 with the endnode of marked edge 2,
  //       that is not common with marked edge 1, or
  //
  //   (2) connecting new node 2 with the endnode of marked edge 1, 
  //       that is not common with marked edge 2.

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == marked_for_irr_ref_3);

  // Create new nodes with the same coordinates as the old nodes
  Node *n_nm = grid.createNode(cell.nonMarkedNode(0).coord());
  Node *n_dm = grid.createNode(cell.markedNode(0).coord()); 
  // (assuming that the node that is marked by two edges has the lowest index) 
  Node *n_m1 = grid.createNode(cell.markedNode(1).coord());
  Node *n_m2 = grid.createNode(cell.markedNode(2).coord());

  // Update parent-child info
  n_nm->setParent(cell.nonMarkedNode(0).coord());
  n_dm->setParent(cell.markedNode(0).coord()); 
  n_m1->setParent(cell.markedNode(1).coord()); 
  n_m2->setParent(cell.markedNode(2).coord()); 

  // Create new node on marked edge 
  Node *n_e1 = grid.createNode(cell.markedEdge(0).midpoint());
  Node *n_e2 = grid.createNode(cell.markedEdge(1).midpoint());

  // Create new cells 
  Cell *t1;
  Cell *t2;
  Cell *t3;

  // Check neighbor face to marked face to refine correctly 

  //    t1 = grid.createCell(n_e1,n_e2,n_e11,n_e21);
  //    t2 = grid.createCell(n_e1,n_e2,n_e11,n_e22);
  //    t3 = grid.createCell(n_e1,n_e2,n_e12,n_e21);

  // Update parent-child info
  t1->setParent(&cell);
  t2->setParent(&cell);
  t3->setParent(&cell);
  
  cell.addChild(t1);
  cell.addChild(t2);
  cell.addChild(t3);
}
//-----------------------------------------------------------------------------
void TetGridRefinement::refineIrregular4(Cell& cell, Grid& grid)
{
  // Two edges are marked, opposite to each other. We insert two new
  // nodes at the midpoints of the marked edges, insert a new edge
  // between the two nodes, and insert four new edges by connecting
  // the new nodes to the endpoints of the opposite edges.

  // Check that the cell is marked correctly 
  dolfin_assert(cell.marker() == marked_for_irr_ref_4);

  // Create new nodes with the same coordinates as the old nodes
  Node *n_e11 = grid.createNode(cell.markedEdge(0).node(0).coord());
  Node *n_e12 = grid.createNode(cell.markedEdge(0).node(1).coord());
  Node *n_e21 = grid.createNode(cell.markedEdge(1).node(0).coord());
  Node *n_e22 = grid.createNode(cell.markedEdge(1).node(1).coord());

  // Update parent-child info
  n_e11->setParent(cell.markedEdge(0).node(0).coord());
  n_e12->setParent(cell.markedEdge(0).node(1).coord());
  n_e21->setParent(cell.markedEdge(1).node(0).coord());
  n_e22->setParent(cell.markedEdge(1).node(1).coord());

  // Create new node on marked edge 
  Node *n_e1 = grid.createNode(cell.markedEdge(0).midpoint());
  Node *n_e2 = grid.createNode(cell.markedEdge(1).midpoint());

  // Create new cells 
  Cell *t1 = grid.createCell(n_e1,n_e2,n_e11,n_e21);
  Cell *t2 = grid.createCell(n_e1,n_e2,n_e11,n_e22);
  Cell *t3 = grid.createCell(n_e1,n_e2,n_e12,n_e21);
  Cell *t4 = grid.createCell(n_e1,n_e2,n_e12,n_e22);

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
bool TetGridRefinement::markedEdgesOnSameFace(Cell& cell)
{
  // FIXME: not implemented

  return true;
}
//-----------------------------------------------------------------------------
