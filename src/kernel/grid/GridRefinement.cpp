// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/Grid.h>
#include <dolfin/Cell.h>
#include <dolfin/Edge.h>
#include <dolfin/GridHierarchy.h>
#include <dolfin/GridIterator.h>
#include <dolfin/TriGridRefinement.h>
#include <dolfin/TetGridRefinement.h>
#include <dolfin/GridRefinement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void GridRefinement::refine(GridHierarchy& grids)
{
  // Write a message
  dolfin_start("Refining grid:");
  cout << grids.fine().rd->noMarkedCells()
       << " cells marked for refinement." << endl;
  
  // Init marks for the finest grid (others already exist)
  initMarks(grids.fine());

  // Refine grid hierarchy
  globalRefinement(grids);

  // Write a message
  cout << "Grid hierarchy consists of " << grids.size() << " grids." << endl;

  dolfin_end();
}
//-----------------------------------------------------------------------------
void GridRefinement::createFineGrid(GridHierarchy& grids)
{
  // Create the new grid
  //Grid* grid = new Grid(grids.fine());

  // Add the grid to the grid hierarchy
  //grids.add(*grid);
}
//-----------------------------------------------------------------------------
void GridRefinement::globalRefinement(GridHierarchy& grids)
{
  // The global grid refinement algorithm working on the whole grid hierarchy.
  // This is algorithm GlobalRefinement() in Beys paper.

  // Phase I: Visit all grids top-down
  for (GridIterator grid(grids,last); !grid.end(); --grid) {
    evaluateMarks(*grid);
    closeGrid(*grid);
  }
  
  // Phase II: Vist all grids bottom-up
  for (GridIterator grid(grids); !grid.end(); ++grid) {
    if (grid.index() > 0)
      closeGrid(*grid);
    unrefineGrid(*grid);
    refineGrid(*grid);
  }

  // FIXME: k = k + 1 ?

}
//-----------------------------------------------------------------------------
void GridRefinement::initMarks(Grid& grid)
{
  // Make sure that all cells have markers
  for (CellIterator c(grid); !c.end(); ++c)
    c->initMarker();

  // Make sure that all edges have markers
  for (EdgeIterator e(grid); !e.end(); ++e)
    e->initMarker();

  // Set markers for cells and edges
  for (List<Cell*>::Iterator c(grid.rd->marked_cells); !c.end(); ++c) {

    // Mark cell for regular refinement
    (*c)->marker() = marked_for_reg_ref;

    // Mark edges of the cell
    for (EdgeIterator e(**c); !e.end(); ++e)
      e->mark(**c);

  }
}
//-----------------------------------------------------------------------------
void GridRefinement::evaluateMarks(Grid& grid)
{
  // Evaluate and adjust marks for a grid.
  // This is algorithm EvaluateMarks() in Beys paper.

  for (CellIterator c(grid); !c.end(); ++c) {

    if ( c->status() == ref_reg && childrenMarkedForCoarsening(*c) )
      c->marker() = marked_for_no_ref;
    
    if ( c->status() == ref_irr ) {
      if ( edgeOfChildMarkedForRefinement(*c) )
	c->marker() = marked_for_reg_ref;
      else
	c->marker() = marked_for_no_ref;
    }
   
  }
}
//-----------------------------------------------------------------------------
void GridRefinement::closeGrid(Grid& grid)
{
  // Perform the green closer on a grid.
  // This is algorithm CloseGrid() in Bey's paper.

  // Create a list of all elements that need to be closed
  List<Cell*> cells;
  for (CellIterator c(grid); !c.end(); ++c)
    if ( edgeMarkedByOther(*c) ) {
      cells.add(c);
      c->closed() = false;
    }

  // Repeat until the list of elements is empty
  while ( !cells.empty() ) {

    // Get first cell and remove it from the list
    Cell* cell = cells.pop();

    // Close cell
    closeCell(*cell, cells);

  }
}
//-----------------------------------------------------------------------------
void GridRefinement::refineGrid(Grid& grid)
{
  // Refine a grid according to marks.
  // This is algorithm RefineGrid() in Bey's paper.

  // Change markers from marked_for_coarsening to marked_for_no_ref
  for (CellIterator c(grid); c.end(); ++c)
    if ( c->marker() == marked_for_coarsening )
      c->marker() = marked_for_no_ref;
  
  // Refine other cells
  for (CellIterator c(grid); !c.end(); ++c) {
    
    // Skip cells which are marked_according_to_ref
    if ( c->marker() == marked_according_to_ref )
      continue;

    // Refine according to refinement rule
    refine(*c, grid);
    
  }
}
//-----------------------------------------------------------------------------
void GridRefinement::unrefineGrid(Grid& grid)
{
  // Unrefine a grid according to marks.
  // This is algorithm UnrefineGrid() in Bey's paper.

  
}
//-----------------------------------------------------------------------------
void GridRefinement::closeCell(Cell& cell, List<Cell*>& cells)
{
  // Close a cell, either by regular or irregular refinement. We check all
  // edges of the cell and try to find a matching refinement rule. This rule
  // is then assigned to the cell's marker for later refinement of the cell.
  // This is algorithm CloseElement() in Bey's paper.
  
  // First count the number of marked edges in the cell
  int no_marked_edges = 0;
  for (EdgeIterator e(cell); !e.end(); ++e)
    if ( e->marked() )
      no_marked_edges++;

  // Check which rule should be applied
  if ( checkRule(cell, no_marked_edges) )
    return;

  // If we didn't find a matching refinement rule, mark cell for regular
  // refinement and add cells containing the previously unmarked edges
  // to the list of cells that need to be closed.

  for (EdgeIterator e(cell); !e.end(); ++e) {

    // Skip marked edges
    if ( e->marked() )
      continue;
    
    // Mark edge by this cell
    e->mark(cell);

    // Add neighbors to the list of cells that need to be closed
    for (CellIterator c(cell); !c.end(); ++c)
      if ( c->haveEdge(*e) && c->closed() && c->status() == ref_reg && c != cell )
	  cells.add(c);
  }

  // Remember that the cell has been closed, important since we don't want
  // to add cells which are not yet closed (and are already in the list).
  cell.closed() = true;
}
//-----------------------------------------------------------------------------
bool GridRefinement::checkRule(Cell& cell, int no_marked_edges)
{
  switch ( cell.type() ) {
  case Cell::triangle:
    return TriGridRefinement::checkRule(cell, no_marked_edges);
    break;
  case Cell::tetrahedron:
    return TetGridRefinement::checkRule(cell, no_marked_edges);
    break;
  default:
    dolfin_error("Unknown cell type.");
  }

  return false;
}
//-----------------------------------------------------------------------------
void GridRefinement::refine(Cell& cell, Grid& grid)
{
  switch ( cell.type() ) {
  case Cell::triangle:
    TriGridRefinement::refine(cell, grid);
    break;
  case Cell::tetrahedron:
    TetGridRefinement::refine(cell, grid);
    break;
  default:
    dolfin_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
bool GridRefinement::childrenMarkedForCoarsening(Cell& cell)
{
  for (int i = 0; i < cell.noChildren(); i++)
    if ( cell.child(i)->marker() != marked_for_coarsening )
      return false;
    
  return true;
}
//-----------------------------------------------------------------------------
bool GridRefinement::edgeOfChildMarkedForRefinement(Cell& cell)
{
  for (int i = 0; i < cell.noChildren(); i++)
    for (EdgeIterator e(*cell.child(i)); !e.end(); ++e)
      if ( e->marked() )
	return true;

  return false;
}
//-----------------------------------------------------------------------------
bool GridRefinement::edgeMarkedByOther(Cell& cell)
{
  // FIXME: Doesn't seem to be correct

  for (EdgeIterator e(cell); !e.end(); ++e)
    if ( e->marked() )
      return true;

  return false;
}
//-----------------------------------------------------------------------------
