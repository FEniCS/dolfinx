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
  if ( checkRuleRegular(cell, no_marked_edge) )
    return true;

  // Which other rules do we have for triangles?

  return false;
}
//-----------------------------------------------------------------------------
void TriGridRefinement::refine(Cell& cell, Grid& grid)
{
  // Refine cell according to marker
  switch ( cell.marker() ) {
  case marked_for_reg_ref:
    regularRefinement(cell, grid);
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

  cell.mark() = marked_for_reg_ref;
  return true;
}
//-----------------------------------------------------------------------------
bool TriGridRefinement::checkRuleIrregular1(Cell& cell, int no_marked_edges)
{
  // Do we have this rule for triangles?
  
  return true;
}
//-----------------------------------------------------------------------------
bool TriGridRefinement::checkRuleIrregular2(Cell& cell, int no_marked_edges)
{
  // Do we have this rule for triangles?
  
  return true;
}
//-----------------------------------------------------------------------------
bool TriGridRefinement::checkRuleIrregular3(Cell& cell, int no_marked_edges)
{
  // Do we have this rule for triangles?
  
  return true;
}
//-----------------------------------------------------------------------------
bool TriGridRefinement::checkRuleIrregular4(Cell& cell, int no_marked_edges)
{
  // Do we have this rule for triangles?
  
  return true;
}
//-----------------------------------------------------------------------------
void TriGridRefinement::refineRegular(Cell& cell, Grid& grid)
{
  // Refine one triangle into four new ones, introducing new nodes 
  // at the midpoints of the edges. 

  // Create new nodes with the same coordinates as the previous nodes in cell  
  Node *n0 = grid.createNode(cell.coord(0));
  Node *n1 = grid.createNode(cell.coord(1));
  Node *n2 = grid.createNode(cell.coord(2));

  // Update parent-child info 
  n0->setParent(cell.node(0));
  n1->setParent(cell.node(1));
  n2->setParent(cell.node(2));

  cell.node(0)->setParent(n0);
  cell.node(1)->setParent(n1);
  cell.node(2)->setParent(n2);
  
  // Create new nodes with the new coordinates 
  Node *n01 = grid.createNode(cell.node(0)->coord().midpoint(cell.node(1)->coord()));
  Node *n02 = grid.createNode(cell.node(0)->coord().midpoint(cell.node(2)->coord()));
  Node *n12 = grid.createNode(cell.node(1)->coord().midpoint(cell.node(2)->coord()));

  // Create new cells 
  Cell *t1 = grid.createCell(n0, n01,n02);
  Cell *t2 = grid.createCell(n01,n1, n12);
  Cell *t3 = grid.createCell(n02,n12,n2 );
  Cell *t4 = grid.createCell(n01,n12,n02);

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
void TriGridRefinement::refineIrregular1(Cell& c, Grid& grid)
{
  // What should we do in this rule for triangles?
}
//-----------------------------------------------------------------------------
void TriGridRefinement::refineIrregular2(Cell& c, Grid& grid)
{
  // What should we do in this rule for triangles?
}
//-----------------------------------------------------------------------------
void TriGridRefinement::refineIrregular3(Cell& c, Grid& grid)
{
  // What should we do in this rule for triangles?
}
//-----------------------------------------------------------------------------
void TriGridRefinement::refineIrregular4(Cell& c, Grid& grid)
{
  // What should we do in this rule for triangles?
}
//-----------------------------------------------------------------------------
