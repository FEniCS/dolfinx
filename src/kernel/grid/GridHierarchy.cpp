// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Grid.h>
#include <dolfin/GridHierarchy.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GridHierarchy::GridHierarchy()
{
  clear();
}
//-----------------------------------------------------------------------------
GridHierarchy::GridHierarchy(Grid& grid)
{
  clear();
  init(grid);
}
//-----------------------------------------------------------------------------
GridHierarchy::~GridHierarchy()
{
  clear();
}
//-----------------------------------------------------------------------------
void GridHierarchy::init(Grid& grid)
{
  // Clear previous grid hierarchy
  clear();

  // Find top grid (level 0)
  Grid* top = &grid;
  for (; top->_parent; top = top->_parent);
  
  // Count the number of grids
  int count = 0;
  for (Grid* g = top; g; g = g->_child)
    count++;

  // Allocate memory for the grids
  grids.init(count);

  // Put the grids in the list
  int pos = 0;
  for (Grid* g = top; g; g = g->_child)
    grids(pos++) = g;

  // Write a message
  cout << "Creating grid hierarchy: found " << count << " grid(s)." << endl;
}
//-----------------------------------------------------------------------------
void GridHierarchy::clear()
{
  grids.clear();
}
//-----------------------------------------------------------------------------
void GridHierarchy::add(Grid& grid)
{
  

}
//-----------------------------------------------------------------------------
Grid& GridHierarchy::operator() (int level) const
{
  if ( empty() )
    dolfin_error("Grid hierarchy is empty.");
  
  if ( level < 0 || level >= grids.size() )
    dolfin_error1("No grid at level %d.", level);
  
  return *(grids(level));
}
//-----------------------------------------------------------------------------
Grid& GridHierarchy::coarse() const
{
  if ( empty() )
    dolfin_error("Grid hierarchy is empty.");
  
  return *(grids(0));
}
//-----------------------------------------------------------------------------
Grid& GridHierarchy::fine() const
{
  if ( empty() )
    dolfin_error("Grid hierarchy is empty.");

  return *(grids(grids.size() - 1));
}
//-----------------------------------------------------------------------------
int GridHierarchy::size() const
{
  return grids.size();
}
//-----------------------------------------------------------------------------
bool GridHierarchy::empty() const
{
  return grids.empty();
}
//-----------------------------------------------------------------------------
