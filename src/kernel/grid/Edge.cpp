// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/Grid.h>
#include <dolfin/Node.h>
#include <dolfin/EdgeRefData.h>
#include <dolfin/Edge.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Edge::Edge()
{
  rd = 0;
  clear();
}
//-----------------------------------------------------------------------------
Edge::Edge(Node& n0, Node& n1)
{
  rd = 0;
  clear();
}
//-----------------------------------------------------------------------------
Edge::~Edge()
{
  clear();
}
//-----------------------------------------------------------------------------
void Edge::clear()
{
  grid = 0;
  _id = -1;
  n0 = 0;
  n1 = 0;

  if ( rd )
    delete rd;
  rd = 0;
}
//-----------------------------------------------------------------------------
int Edge::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
Node& Edge::node(int i) const
{
  if ( i == 0 )
    return *n0;

  if ( i == 1 )
    return *n1;

  dolfin_error("Node number must 0 or 1.");
  return *n0;
}
//-----------------------------------------------------------------------------
Point& Edge::coord(int i) const
{
  if ( i == 0 )
    return n0->coord();

  if ( i == 1 )
    return n1->coord();

  dolfin_error("Node number must 0 or 1.");
  return n0->coord();
}
//-----------------------------------------------------------------------------
real Edge::length() const
{
  return n0->dist(*n1);
}
//-----------------------------------------------------------------------------
Point Edge::midpoint() const
{
  Point p = n0->coord();
  p += n1->coord();
  p /= 2.0;

  return p;
}
//-----------------------------------------------------------------------------
bool Edge::equals(const Node& n0, const Node& n1) const
{
  if ( this->n0 == &n0 && this->n1 == &n1 )
    return true;

  if ( this->n0 == &n1 && this->n1 == &n0 )
    return true;

  return false;
}
//-----------------------------------------------------------------------------
bool Edge::contains(const Node& n) const
{
  if ( this->n0 == &n || this->n1 == &n )
    return true;

  return false;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const Edge& edge)
{
  stream << "[ Edge: id = " << edge.id()
	 << " n0 = " << edge.node(0).id()
	 << " n1 = " << edge.node(1).id() << " ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
int Edge::setID(int id, Grid& grid)
{
  this->grid = &grid;
  return _id = id;
}
//-----------------------------------------------------------------------------
void Edge::setGrid(Grid& grid)
{
  this->grid = &grid;
}
//-----------------------------------------------------------------------------
void Edge::set(Node& n0, Node& n1)
{
  this->n0 = &n0;
  this->n1 = &n1;
}
//-----------------------------------------------------------------------------
void Edge::initMarker()
{
  if ( !rd )
    rd = new EdgeRefData();
  dolfin_assert(rd);
}
//-----------------------------------------------------------------------------
void Edge::mark(Cell& cell)
{
  initMarker();
  rd->mark(cell);
}
//-----------------------------------------------------------------------------
bool Edge::marked()
{
  initMarker();
  return rd->marked();
}
//-----------------------------------------------------------------------------
bool Edge::marked(Cell& cell)
{
  initMarker();
  return rd->marked(cell);
}
//-----------------------------------------------------------------------------
