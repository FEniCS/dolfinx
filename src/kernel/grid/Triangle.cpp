// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Node.h>
#include <dolfin/Point.h>
#include <dolfin/Cell.h>
#include <dolfin/Triangle.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Triangle::Triangle(Node* n0, Node* n1, Node* n2) : GenericCell()
{
  cn.init(noNodes());

  cn(0) = n0;
  cn(1) = n1;
  cn(2) = n2;
}
//-----------------------------------------------------------------------------
int Triangle::noNodes() const
{
  return 3;
}
//-----------------------------------------------------------------------------
int Triangle::noEdges() const
{
  return 3;
}
//-----------------------------------------------------------------------------
int Triangle::noFaces() const
{
  return 1;
}
//-----------------------------------------------------------------------------
int Triangle::noBoundaries() const
{
  return noEdges();
}
//-----------------------------------------------------------------------------
Cell::Type Triangle::type() const
{
  return Cell::triangle;
}
//-----------------------------------------------------------------------------
real Triangle::volume() const
{
  // The volume of a triangle is the same as its area
   
  // Get the coordinates
  real x1 = coord(0).x; real y1 = coord(0).y; real z1 = coord(0).z;
  real x2 = coord(1).x; real y2 = coord(1).y; real z2 = coord(1).z;
  real x3 = coord(2).x; real y3 = coord(2).y; real z3 = coord(2).z;
                                                                                                                             
  // Formula for volume from http://mathworld.wolfram.com
  real v1 = (y1*z2 + z1*y3 + y2*z3) - ( y3*z2 + z3*y1 + y2*z1 );
  real v2 = (z1*x2 + x1*z3 + z2*x3) - ( z3*x2 + x3*z1 + z2*x1 );
  real v3 = (x1*y2 + y1*x3 + x2*y3) - ( x3*y2 + y3*x1 + x2*y1 );
                                                                                                                             
  real v = 0.5 * sqrt( v1*v1 + v2*v2 + v3*v3 );
                                                                                                                             
  return v;
}
//-----------------------------------------------------------------------------
real Triangle::diameter() const
{
  // Compute side lengths
  real a  = coord(1).dist(coord(2));
  real b  = coord(0).dist(coord(2));
  real c  = coord(0).dist(coord(1));
                                                                                                                             
  // Formula for diameter (2*circumradius) from http://mathworld.wolfram.com
  real d = 0.5 * a*b*c / volume();
                                                                                                                             
  return d;
}
//-----------------------------------------------------------------------------
void Triangle::createEdges()
{
  ce.init(3);
  ce.reset();

  createEdge(cn(0), cn(1));
  createEdge(cn(1), cn(2));
  createEdge(cn(2), cn(0));
}
//-----------------------------------------------------------------------------
void Triangle::createFaces()
{
  // A triangle has no faces
}
//-----------------------------------------------------------------------------
