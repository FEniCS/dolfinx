#include <dolfin/Grid.h>
#include <dolfin/Edge.h>
#include <dolfin/EdgeIterator.h>
#include <dolfin/GenericCell.h>
#include <dolfin/GridData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
// EdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Grid &grid)
{
  n = new GridEdgeIterator(grid);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Grid *grid)
{
  n = new GridEdgeIterator(*grid);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Cell &cell)
{
  n = new CellEdgeIterator(cell);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const CellIterator &cellIterator)
{
  n = new CellEdgeIterator(*cellIterator);
}
//-----------------------------------------------------------------------------
EdgeIterator::operator EdgePointer() const
{
  return e->pointer();
}
//-----------------------------------------------------------------------------
EdgeIterator::~EdgeIterator()
{
  delete e;
}
//-----------------------------------------------------------------------------
EdgeIterator& EdgeIterator::operator++()
{
  ++(*e);

  return *this;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::end()
{
  return e->end();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::last()
{
  return e->last();
}
//-----------------------------------------------------------------------------
int EdgeIterator::index()
{
  return e->index();
}
//-----------------------------------------------------------------------------
Edge& EdgeIterator::operator*() const
{
  return *(*e);
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::operator->() const
{
  return e->pointer();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::operator==(const EdgeIterator& n) const
{
  return this->e->pointer() == e.e->pointer();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::operator!=(const EdgeIterator& n) const
{
  return this->e->pointer() != e.e->pointer();
}
//-----------------------------------------------------------------------------
// EdgeIterator::GridEdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::GridEdgeIterator::GridEdgeIterator(const Grid& grid)
{
  edge_iterator = grid.gd->edges.begin();
  at_end = grid.gd->edges.end();
}
//-----------------------------------------------------------------------------
void EdgeIterator::GridEdgeIterator::operator++()
{
  ++edge_iterator;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::GridEdgeIterator::end()
{
  return edge_iterator == at_end;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::GridEdgeIterator::last()
{
  return edge_iterator.last();
}
//-----------------------------------------------------------------------------
int EdgeIterator::GridEdgeIterator::index()
{
  return edge_iterator.index();
}
//-----------------------------------------------------------------------------
Edge& EdgeIterator::GridEdgeIterator::operator*() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::GridEdgeIterator::operator->() const
{
  return edge_iterator.pointer();
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::GridEdgeIterator::pointer() const
{
  return edge_iterator.pointer();
}
//-----------------------------------------------------------------------------
// EdgeIterator::CellEdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::CellEdgeIterator::CellEdgeIterator(const Cell &cell)
{
  edge_iterator = cell.ce.begin();
}
//-----------------------------------------------------------------------------
void EdgeIterator::CellEdgeIterator::operator++()
{
  ++edge_iterator;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::CellEdgeIterator::end()
{
  return edge_iterator.end();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::CellEdgeIterator::last()
{
  return edge_iterator.last();
}
//-----------------------------------------------------------------------------
int EdgeIterator::CellEdgeIterator::index()
{
  return edge_iterator.index();
}
//-----------------------------------------------------------------------------
Edge& EdgeIterator::CellEdgeIterator::operator*() const
{
  return **edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::CellEdgeIterator::operator->() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::CellEdgeIterator::pointer() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
