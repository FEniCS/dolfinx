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
  end_nodes.init(2);

  end_nodes(0) = 0;
  end_nodes(1) = 0;

  marked_for_refinement = false;
}
//-----------------------------------------------------------------------------
Edge::Edge(Node *en1, Node *en2)
{
  end_nodes.init(2);

  end_nodes(0) = en1;
  end_nodes(1) = en2;

  marked_for_refinement = false;
}
//-----------------------------------------------------------------------------
void Edge::set(Node *en1, Node *en2)
{
  end_nodes.init(2);

  end_nodes(0) = en1;
  end_nodes(1) = en2;
}
//-----------------------------------------------------------------------------
Node* Edge::node(int node)
{
  if ( (node<0) || (node>=2) )
    dolfin_error("Illegal node.");
  
  return end_nodes(node);
}
//-----------------------------------------------------------------------------
Point Edge::coord(int node)
{
  if ( (node<0) || (node>=2) )
    dolfin_error("Illegal node.");
  
  return end_nodes(node)->coord();
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
real Edge::computeLength()
{
  // Get the coordinates
  Point p1 = end_nodes(0)->coord();
  Point p2 = end_nodes(1)->coord();

  return p1.dist(p2);
}
//-----------------------------------------------------------------------------
Point Edge::computeMidpoint()
{
  // Get the coordinates
  Point p1 = end_nodes(0)->coord();
  Point p2 = end_nodes(1)->coord();

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
