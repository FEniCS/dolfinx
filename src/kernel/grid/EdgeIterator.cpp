// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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
  e = new GridEdgeIterator(grid);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Grid *grid)
{
  e = new GridEdgeIterator(*grid);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Cell &cell)
{
  e = new CellEdgeIterator(cell);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const CellIterator &cellIterator)
{
  e = new CellEdgeIterator(*cellIterator);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Face& face)
{
  e = new FaceEdgeIterator(face);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const FaceIterator& faceIterator)
{
  e = new FaceEdgeIterator(*faceIterator);
}
//-----------------------------------------------------------------------------
EdgeIterator::operator EdgePointer() const
{
  return e->pointer();
}
//-----------------------------------------------------------------------------
EdgeIterator::~EdgeIterator()
{
  if ( e )
    delete e;
  e = 0;
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
bool EdgeIterator::operator==(const EdgeIterator& e) const
{
  return this->e->pointer() == e.e->pointer();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::operator!=(const EdgeIterator& e) const
{
  return this->e->pointer() != e.e->pointer();
}
//-----------------------------------------------------------------------------
// EdgeIterator::GridEdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::GridEdgeIterator::GridEdgeIterator(const Grid& grid)
{
  edge_iterator = grid.gd.edges.begin();
  at_end = grid.gd.edges.end();
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
// EdgeIterator::NodeEdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::NodeEdgeIterator::NodeEdgeIterator(const Node &node)
{
  edge_iterator = node.ne.begin();
}
//-----------------------------------------------------------------------------
void EdgeIterator::NodeEdgeIterator::operator++()
{
  ++edge_iterator;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::NodeEdgeIterator::end()
{
  return edge_iterator.end();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::NodeEdgeIterator::last()
{
  return edge_iterator.last();
}
//-----------------------------------------------------------------------------
int EdgeIterator::NodeEdgeIterator::index()
{
  return edge_iterator.index();
}
//-----------------------------------------------------------------------------
Edge& EdgeIterator::NodeEdgeIterator::operator*() const
{
  return **edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::NodeEdgeIterator::operator->() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::NodeEdgeIterator::pointer() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
// EdgeIterator::CellEdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::CellEdgeIterator::CellEdgeIterator(const Cell &cell)
{
  edge_iterator = cell.c->ce.begin();
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
// EdgeIterator::FaceEdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::FaceEdgeIterator::FaceEdgeIterator(const Face &face)
{
  edge_iterator = face.fe.begin();
}
//-----------------------------------------------------------------------------
void EdgeIterator::FaceEdgeIterator::operator++()
{
  ++edge_iterator;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::FaceEdgeIterator::end()
{
  return edge_iterator.end();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::FaceEdgeIterator::last()
{
  return edge_iterator.last();
}
//-----------------------------------------------------------------------------
int EdgeIterator::FaceEdgeIterator::index()
{
  return edge_iterator.index();
}
//-----------------------------------------------------------------------------
Edge& EdgeIterator::FaceEdgeIterator::operator*() const
{
  return **edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::FaceEdgeIterator::operator->() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::FaceEdgeIterator::pointer() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
