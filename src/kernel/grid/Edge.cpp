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
  _id = -1;

  _en0 = NULL;
  _en1 = NULL;

  marked_for_refinement = false;
}
//-----------------------------------------------------------------------------
Edge::Edge(Node *en0, Node *en1)
{
  _id = -1;

  _en0 = en0;
  _en1 = en1;

  marked_for_refinement = false;
}
//-----------------------------------------------------------------------------
Edge::~Edge()
{
}
//-----------------------------------------------------------------------------
void Edge::set(Node *en0, Node *en1)
{
  _en0 = en0;
  _en1 = en1;
}
//-----------------------------------------------------------------------------
int Edge::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
void Edge::setID(int id)
{
  _id = id;
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
Node* Edge::node(int node)
{
  if ( (node<0) || (node>=2) )
    dolfin_error("Illegal node.");

  if ( node == 0 ) return _en0;
  if ( node == 1 ) return _en1;
  dolfin_error("edge end node must be 0 or 1");
}
//-----------------------------------------------------------------------------
Point Edge::coord(int node)
{
  if ( (node<0) || (node>=2) )
    dolfin_error("Illegal node.");
  
  if ( node == 0 ) return _en0->coord();
  if ( node == 1 ) return _en1->coord();
  dolfin_error("edge end node must be 0 or 1");
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
real Edge::length()
{
  // Get the coordinates
  Point p1 = _en0->coord();
  Point p2 = _en1->coord();

  return p1.dist(p2);
}
//-----------------------------------------------------------------------------
Point Edge::midpoint()
{
  // Get the coordinates
  Point p1 = _en0->coord();
  Point p2 = _en1->coord();

  // Make sure we get full precision
  real x1, x2, y1, y2, z1, z2;

  x1 = real(p1.x); y1 = real(p1.y); z1 = real(p1.z);
  x2 = real(p2.x); y2 = real(p2.y); z2 = real(p2.z);

  // The midpoint of the edge 
  Point mp;
  mp.x = 0.5*(x1+x2);
  mp.y = 0.5*(y1+y2);
  mp.z = 0.5*(z1+z2);
  
  return ( mp );
}
//-----------------------------------------------------------------------------
