// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/Grid.h>
#include <dolfin/GridRefinementData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GridRefinementData::GridRefinementData(Grid* grid)
{
  this->grid = grid;
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
  cell_markers.clear();
  edge_markers.clear();
}
//-----------------------------------------------------------------------------
void GridRefinementData::mark(Cell* cell)
{
  marked_cells.add(cell);
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
void GridRefinementData::initMarkers()
{
  cell_markers.init(grid->noCells());
  edge_markers.init(grid->noEdges());
}
//-----------------------------------------------------------------------------
Cell::Marker& GridRefinementData::cellMarker(int id)
{
  return cell_markers(id).marker;
}
//-----------------------------------------------------------------------------
void GridRefinementData::edgeMark(int id, Cell& cell)
{
  edge_markers(id).mark(cell);
}
//-----------------------------------------------------------------------------
bool GridRefinementData::edgeMarked(int id) const
{
  return edge_markers(id).marked();
}
//-----------------------------------------------------------------------------
bool GridRefinementData::edgeMarked(int id, Cell& cell) const
{
  return edge_markers(id).marked(cell);
}
//-----------------------------------------------------------------------------
