#include <dolfin/Grid.h>
#include <dolfin/Cell.h>
#include <dolfin/GenericCell.h>
#include <dolfin/CellIterator.h>
#include "GridData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// CellIterator
//-----------------------------------------------------------------------------
CellIterator::CellIterator(Grid &grid)
{
  c = new GridCellIterator(grid);
}
//-----------------------------------------------------------------------------
CellIterator::CellIterator(Grid *grid)
{
  c = new GridCellIterator(*grid);
}
//-----------------------------------------------------------------------------
CellIterator::CellIterator(Cell &cell)
{
  c = new CellCellIterator(cell);
}
//-----------------------------------------------------------------------------
CellIterator::CellIterator(CellIterator &cellIterator)
{
  c = new CellCellIterator(*cellIterator);
}
//-----------------------------------------------------------------------------
CellIterator::CellIterator(Node &node)
{
  c = new NodeCellIterator(node);
}
//-----------------------------------------------------------------------------
CellIterator::CellIterator(NodeIterator &nodeIterator)
{
  c = new NodeCellIterator(*nodeIterator);
}
//-----------------------------------------------------------------------------
CellIterator::~CellIterator()
{
  delete c;
}
//-----------------------------------------------------------------------------
CellIterator::operator CellPointer() const
{
  return c->pointer();
}
//-----------------------------------------------------------------------------
CellIterator& CellIterator::operator++()
{
  ++(*c);

  return *this;
}
//-----------------------------------------------------------------------------
bool CellIterator::end()
{
  return c->end();
}
//-----------------------------------------------------------------------------
Cell& CellIterator::operator*() const
{
  return *(*c);
}
//-----------------------------------------------------------------------------
Cell* CellIterator::operator->() const
{
  return c->pointer();
}
//-----------------------------------------------------------------------------
// CellIterator::GridCellIterator
//-----------------------------------------------------------------------------
CellIterator::GridCellIterator::GridCellIterator(Grid &grid)
{
  cell_iterator = grid.grid_data->cells.begin();
  at_end = grid.grid_data->cells.end();
}
//-----------------------------------------------------------------------------
void CellIterator::GridCellIterator::operator++()
{
  ++cell_iterator;
}
//-----------------------------------------------------------------------------
bool CellIterator::GridCellIterator::end()
{
  return cell_iterator == at_end;
}
//-----------------------------------------------------------------------------
Cell& CellIterator::GridCellIterator::operator*() const
{
  return *cell_iterator;
}
//-----------------------------------------------------------------------------
Cell* CellIterator::GridCellIterator::operator->() const
{
  return cell_iterator.pointer();
}
//-----------------------------------------------------------------------------
Cell* CellIterator::GridCellIterator::pointer() const
{
  return cell_iterator.pointer();
}
//-----------------------------------------------------------------------------
// CellIterator::NodeCellIterator
//-----------------------------------------------------------------------------
CellIterator::NodeCellIterator::NodeCellIterator(Node &node)
{
  cell_iterator = node.nc.begin();
}
//-----------------------------------------------------------------------------
void CellIterator::NodeCellIterator::operator++()
{
  ++cell_iterator;
}
//-----------------------------------------------------------------------------
bool CellIterator::NodeCellIterator::end()
{
  return cell_iterator.end();
}
//-----------------------------------------------------------------------------
Cell& CellIterator::NodeCellIterator::operator*() const
{
  return **cell_iterator;
}
//-----------------------------------------------------------------------------
Cell* CellIterator::NodeCellIterator::operator->() const
{
  return *cell_iterator;
}
//-----------------------------------------------------------------------------
Cell* CellIterator::NodeCellIterator::pointer() const
{
  return *cell_iterator;
}
//-----------------------------------------------------------------------------
// CellIterator::CellCellIterator
//-----------------------------------------------------------------------------
CellIterator::CellCellIterator::CellCellIterator(Cell &cell)
{
  cell_iterator = cell.cc.begin();
}
//-----------------------------------------------------------------------------
void CellIterator::CellCellIterator::operator++()
{
  ++cell_iterator;
}
//-----------------------------------------------------------------------------
bool CellIterator::CellCellIterator::end()
{
  return cell_iterator.end();
}
//-----------------------------------------------------------------------------
Cell& CellIterator::CellCellIterator::operator*() const
{
  return **cell_iterator;
}
//-----------------------------------------------------------------------------
Cell* CellIterator::CellCellIterator::operator->() const
{
  return *cell_iterator;
}
//-----------------------------------------------------------------------------
Cell* CellIterator::CellCellIterator::pointer() const
{
  return *cell_iterator;
}
//-----------------------------------------------------------------------------
