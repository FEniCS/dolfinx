#include <dolfin/Grid.h>
#include <dolfin/Node.h>
#include <dolfin/Cell.h>
#include <dolfin/GridIterators.h>
#include "GridData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// NodeIterator
//-----------------------------------------------------------------------------
NodeIterator::NodeIterator(Grid& grid)
{
  n = new GridNodeIterator(grid);
}
//-----------------------------------------------------------------------------
NodeIterator::~NodeIterator()
{
  delete n;
}
//-----------------------------------------------------------------------------
void NodeIterator::operator++()
{
  ++(*n);
}
//-----------------------------------------------------------------------------
bool NodeIterator::end()
{
  return n->end();
}
//-----------------------------------------------------------------------------
Node& NodeIterator::operator*() const
{
  return *(*n);
}
//-----------------------------------------------------------------------------
Node* NodeIterator::operator->() const
{
  return n->pointer();
}
//-----------------------------------------------------------------------------
// GridNodeIterator
//-----------------------------------------------------------------------------
GridNodeIterator::GridNodeIterator(Grid& grid)
{
  node_iterator = grid.grid_data->nodes.begin();
  at_end = grid.grid_data->nodes.end();
}
//-----------------------------------------------------------------------------
void GridNodeIterator::operator++()
{
  ++node_iterator;
}
//-----------------------------------------------------------------------------
bool GridNodeIterator::end()
{
  return node_iterator == at_end;
}
//-----------------------------------------------------------------------------
Node& GridNodeIterator::operator*() const
{
  return *node_iterator;
}
//-----------------------------------------------------------------------------
Node* GridNodeIterator::operator->() const
{
  return node_iterator.pointer();
}
//-----------------------------------------------------------------------------
Node* GridNodeIterator::pointer() const
{
  return node_iterator.pointer();
}
//-----------------------------------------------------------------------------
// GridCellIterator
//-----------------------------------------------------------------------------
GridCellIterator::GridCellIterator(Grid& grid)
{
  cell_iterator = grid.grid_data->cells.begin();
  at_end = grid.grid_data->cells.end();
}
//-----------------------------------------------------------------------------
void GridCellIterator::operator++()
{
  ++cell_iterator;
}
//-----------------------------------------------------------------------------
bool GridCellIterator::end()
{
  return cell_iterator == at_end;
}
//-----------------------------------------------------------------------------
Cell& GridCellIterator::operator*() const
{
  return *cell_iterator;
}
//-----------------------------------------------------------------------------
Cell* GridCellIterator::operator->() const
{
  return cell_iterator.pointer();
}
//-----------------------------------------------------------------------------
Cell* GridCellIterator::pointer() const
{
  return cell_iterator.pointer();
}
//-----------------------------------------------------------------------------
