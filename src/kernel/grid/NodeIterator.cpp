// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Grid.h>
#include <dolfin/Boundary.h>
#include <dolfin/Node.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/GenericCell.h>
#include <dolfin/GridData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
// NodeIterator
//-----------------------------------------------------------------------------
NodeIterator::NodeIterator(const Grid &grid)
{
  n = new GridNodeIterator(grid);
}
//-----------------------------------------------------------------------------
NodeIterator::NodeIterator(const Grid *grid)
{
  n = new GridNodeIterator(*grid);
}
//-----------------------------------------------------------------------------
NodeIterator::NodeIterator(const Boundary& boundary)
{
  n = new BoundaryNodeIterator(boundary);
}
//-----------------------------------------------------------------------------
NodeIterator::NodeIterator(const Node &node)
{
  n = new NodeNodeIterator(node);
}
//-----------------------------------------------------------------------------
NodeIterator::NodeIterator(const NodeIterator &nodeIterator)
{
  n = new NodeNodeIterator(*nodeIterator);
}
//-----------------------------------------------------------------------------
NodeIterator::NodeIterator(const Cell &cell)
{
  n = new CellNodeIterator(cell);
}
//-----------------------------------------------------------------------------
NodeIterator::NodeIterator(const CellIterator &cellIterator)
{
  n = new CellNodeIterator(*cellIterator);
}
//-----------------------------------------------------------------------------
NodeIterator::operator NodePointer() const
{
  return n->pointer();
}
//-----------------------------------------------------------------------------
NodeIterator::~NodeIterator()
{
  delete n;
}
//-----------------------------------------------------------------------------
NodeIterator& NodeIterator::operator++()
{
  ++(*n);

  return *this;
}
//-----------------------------------------------------------------------------
bool NodeIterator::end()
{
  return n->end();
}
//-----------------------------------------------------------------------------
bool NodeIterator::last()
{
  return n->last();
}
//-----------------------------------------------------------------------------
int NodeIterator::index()
{
  return n->index();
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
bool NodeIterator::operator==(const NodeIterator& n) const
{
  return this->n->pointer() == n.n->pointer();
}
//-----------------------------------------------------------------------------
bool NodeIterator::operator!=(const NodeIterator& n) const
{
  return this->n->pointer() != n.n->pointer();
}
//-----------------------------------------------------------------------------
// NodeIterator::GridNodeIterator
//-----------------------------------------------------------------------------
NodeIterator::GridNodeIterator::GridNodeIterator(const Grid& grid)
{
  node_iterator = grid.gd->nodes.begin();
  at_end = grid.gd->nodes.end();
}
//-----------------------------------------------------------------------------
void NodeIterator::GridNodeIterator::operator++()
{
  ++node_iterator;
}
//-----------------------------------------------------------------------------
bool NodeIterator::GridNodeIterator::end()
{
  return node_iterator == at_end;
}
//-----------------------------------------------------------------------------
bool NodeIterator::GridNodeIterator::last()
{
  return node_iterator.last();
}
//-----------------------------------------------------------------------------
int NodeIterator::GridNodeIterator::index()
{
  return node_iterator.index();
}
//-----------------------------------------------------------------------------
Node& NodeIterator::GridNodeIterator::operator*() const
{
  return *node_iterator;
}
//-----------------------------------------------------------------------------
Node* NodeIterator::GridNodeIterator::operator->() const
{
  return node_iterator.pointer();
}
//-----------------------------------------------------------------------------
Node* NodeIterator::GridNodeIterator::pointer() const
{
  return node_iterator.pointer();
}
//-----------------------------------------------------------------------------
// NodeIterator::BoundaryNodeIterator
//-----------------------------------------------------------------------------
NodeIterator::BoundaryNodeIterator::BoundaryNodeIterator
(const Boundary& boundary) : node_iterator(boundary.grid->bd->nodes)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NodeIterator::BoundaryNodeIterator::operator++()
{
  ++node_iterator;
}
//-----------------------------------------------------------------------------
bool NodeIterator::BoundaryNodeIterator::end()
{
  return node_iterator.end();
}
//-----------------------------------------------------------------------------
bool NodeIterator::BoundaryNodeIterator::last()
{
  return node_iterator.last();
}
//-----------------------------------------------------------------------------
int NodeIterator::BoundaryNodeIterator::index()
{
  return node_iterator.index();
}
//-----------------------------------------------------------------------------
Node& NodeIterator::BoundaryNodeIterator::operator*() const
{
  return **node_iterator;
}
//-----------------------------------------------------------------------------
Node* NodeIterator::BoundaryNodeIterator::operator->() const
{
  return *node_iterator;
}
//-----------------------------------------------------------------------------
Node* NodeIterator::BoundaryNodeIterator::pointer() const
{
  return *node_iterator;
}
//-----------------------------------------------------------------------------
// NodeIterator::CellNodeIterator
//-----------------------------------------------------------------------------
NodeIterator::CellNodeIterator::CellNodeIterator(const Cell &cell)
{
  node_iterator = cell.c->cn.begin();
}
//-----------------------------------------------------------------------------
void NodeIterator::CellNodeIterator::operator++()
{
  ++node_iterator;
}
//-----------------------------------------------------------------------------
bool NodeIterator::CellNodeIterator::end()
{
  return node_iterator.end();
}
//-----------------------------------------------------------------------------
bool NodeIterator::CellNodeIterator::last()
{
  return node_iterator.last();
}
//-----------------------------------------------------------------------------
int NodeIterator::CellNodeIterator::index()
{
  return node_iterator.index();
}
//-----------------------------------------------------------------------------
Node& NodeIterator::CellNodeIterator::operator*() const
{
  return **node_iterator;
}
//-----------------------------------------------------------------------------
Node* NodeIterator::CellNodeIterator::operator->() const
{
  return *node_iterator;
}
//-----------------------------------------------------------------------------
Node* NodeIterator::CellNodeIterator::pointer() const
{
  return *node_iterator;
}
//-----------------------------------------------------------------------------
// NodeIterator::NodeNodeIterator
//-----------------------------------------------------------------------------
NodeIterator::NodeNodeIterator::NodeNodeIterator(const Node &node)
{
  node_iterator = node.nn.begin();
}
//-----------------------------------------------------------------------------
void NodeIterator::NodeNodeIterator::operator++()
{
  ++node_iterator;
}
//-----------------------------------------------------------------------------
bool NodeIterator::NodeNodeIterator::end()
{
  return node_iterator.end();
}
//-----------------------------------------------------------------------------
bool NodeIterator::NodeNodeIterator::last()
{
  return node_iterator.last();
}
//-----------------------------------------------------------------------------
int NodeIterator::NodeNodeIterator::index()
{
  return node_iterator.index();
}
//-----------------------------------------------------------------------------
Node& NodeIterator::NodeNodeIterator::operator*() const
{
  return **node_iterator;
}
//-----------------------------------------------------------------------------
Node* NodeIterator::NodeNodeIterator::operator->() const
{
  return *node_iterator;
}
//-----------------------------------------------------------------------------
Node* NodeIterator::NodeNodeIterator::pointer() const
{
  return *node_iterator;
}
//-----------------------------------------------------------------------------
