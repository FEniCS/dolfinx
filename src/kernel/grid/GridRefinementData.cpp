// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/Grid.h>
#include <dolfin/GridRefinementData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GridRefinementData::GridRefinementData(Grid& grid)
{
  this->grid = &grid;
}
//-----------------------------------------------------------------------------
GridRefinementData::~GridRefinementData()
{
  clear();
}
//-----------------------------------------------------------------------------
void GridRefinementData::clear()
{
  marked_cells.clear();
}
//-----------------------------------------------------------------------------
void GridRefinementData::mark(Cell& cell)
{
  marked_cells.add(&cell);
}
//-----------------------------------------------------------------------------
int GridRefinementData::noMarkedCells() const
{
  return marked_cells.size();

}
//-----------------------------------------------------------------------------
void GridRefinementData::setGrid(Grid& grid)
{
  this->grid = &grid;
}
//-----------------------------------------------------------------------------
