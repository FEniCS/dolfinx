// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>

#include <dolfin/Edge.h>
#include <dolfin/Node.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Edge::Edge()
{
  grid = 0;
  _id = -1;

  n0 = 0;
  n1 = 0;

  // FIXME: Remove?
  marked_for_refinement = false;
}
//-----------------------------------------------------------------------------
Edge::Edge(Node* n0, Node* n1)
{
  grid = 0;
  _id = -1;

  this->n0 = n0;
  this->n1 = n1;

  // FIXME: Remove?
  marked_for_refinement = false;
}
//-----------------------------------------------------------------------------
Edge::~Edge()
{

}
//-----------------------------------------------------------------------------
int Edge::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
Node* Edge::node(int i) const
{
  if ( i == 0 )
    return n0;

  if ( i == 1 )
    return n1;

  dolfin_error("Node number must 0 or 1.");

  return 0;
}
//-----------------------------------------------------------------------------
Point Edge::coord(int i) const
{
  if ( i == 0 )
    return n0->coord();

  if ( i == 1 )
    return n1->coord();

  dolfin_error("Node number must 0 or 1.");

  return 0;
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
bool Edge::equals(Node* n0, Node* n1) const
{
  if ( this->n0 == n0 && this->n1 == n1 )
    return true;

  if ( this->n0 == n1 && this->n1 == n0 )
    return true;

  return false;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const Edge& edge)
{
  stream << "[ Edge: id = " << edge.id()
	 << " n0 = " << edge.node(0)->id()
	 << " n1 = " << edge.node(1)->id() << " ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
int Edge::setID(int id, Grid* grid)
{
  this->grid = grid;
  return _id = id;
}
//-----------------------------------------------------------------------------
void Edge::set(Node* n0, Node* n1)
{
  this->n0 = n0;
  this->n1 = n1;
}
//-----------------------------------------------------------------------------



// FIXME: Remove?

void Edge::setMarkedForReUse(bool re_use)
{
  _marked_for_re_use = re_use;
}
//-----------------------------------------------------------------------------
bool Edge::markedForReUse()
{
  return _marked_for_re_use;
}
//-----------------------------------------------------------------------------
int Edge::level() const
{
  return _level;
}
//-----------------------------------------------------------------------------
void Edge::setLevel(int level)
{
  _level = level;
}
//-----------------------------------------------------------------------------
int Edge::refinedByCells() 
{
  return _no_cells_refined;
}
//-----------------------------------------------------------------------------
Cell* Edge::refinedByCell(int i) 
{
  return refined_by_cell(i);
}
//-----------------------------------------------------------------------------
void Edge::setRefinedByCell(Cell* c) 
{
  refined_by_cell.add(c);
  _no_cells_refined++;
}
//-----------------------------------------------------------------------------
bool Edge::marked()
{
  return marked_for_refinement;
}
//-----------------------------------------------------------------------------
void Edge::mark()
{
  marked_for_refinement = true;
}
//-----------------------------------------------------------------------------
void Edge::unmark()
{
  marked_for_refinement = false;
}
//-----------------------------------------------------------------------------
