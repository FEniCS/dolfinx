// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Node.h>
#include <dolfin/Point.h>
#include <dolfin/Cell.h>
#include <dolfin/Tetrahedron.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Tetrahedron::Tetrahedron(Node* n0, Node* n1, Node* n2, Node* n3) : GenericCell()
{
  cn.init(noNodes());

  cn(0) = n0;
  cn(1) = n1;
  cn(2) = n2;
  cn(3) = n3;
}
//-----------------------------------------------------------------------------
int Tetrahedron::noNodes() const
{
  return 4;
}
//-----------------------------------------------------------------------------
int Tetrahedron::noEdges() const
{
  return 6;
}
//-----------------------------------------------------------------------------
int Tetrahedron::noFaces() const
{
  return 4;
}
//-----------------------------------------------------------------------------
int Tetrahedron::noBoundaries() const
{
  return noFaces();
}
//-----------------------------------------------------------------------------
Cell::Type Tetrahedron::type() const
{
  return Cell::tetrahedron;
}
//-----------------------------------------------------------------------------
real Tetrahedron::volume() const
{
  // Get the coordinates
  real x1 = coord(0).x; real y1 = coord(0).y; real z1 = coord(0).z;
  real x2 = coord(1).x; real y2 = coord(1).y; real z2 = coord(1).z;
  real x3 = coord(2).x; real y3 = coord(2).y; real z3 = coord(2).z;
  real x4 = coord(3).x; real y4 = coord(3).y; real z4 = coord(3).z;
                                                                                                                             
  // Formula for volume from http://mathworld.wolfram.com
  real v = ( x1 * ( y2*z3 + y4*z2 + y3*z4 - y3*z2 - y2*z4 - y4*z3 ) -
	     x2 * ( y1*z3 + y4*z1 + y3*z4 - y3*z1 - y1*z4 - y4*z3 ) +
	     x3 * ( y1*z2 + y4*z1 + y2*z4 - y2*z1 - y1*z4 - y4*z2 ) -
	     x4 * ( y1*z2 + y2*z3 + y3*z1 - y2*z1 - y3*z2 - y1*z3 ) );

  return fabs(v);
}
//-----------------------------------------------------------------------------
real Tetrahedron::diameter() const
{
  // Compute side lengths
  real a  = coord(1).dist(coord(2));
  real b  = coord(0).dist(coord(2));
  real c  = coord(0).dist(coord(1));
  real aa = coord(0).dist(coord(3));
  real bb = coord(1).dist(coord(3));
  real cc = coord(2).dist(coord(3));
                                                                                                                             
  // Compute "area" of triangle with strange side lengths
  real l1   = a*aa;
  real l2   = b*bb;
  real l3   = c*cc;
  real s    = 0.5*(l1+l2+l3);
  real area = sqrt(s*(s-l1)*(s-l2)*(s-l3));
                                                                                                                             
  // Formula for diameter (2*circumradius) from http://mathworld.wolfram.com
  real d = area / ( 3.0*volume() );
                                                                                                                             
  return d;
}
//-----------------------------------------------------------------------------
void Tetrahedron::createEdges()
{
  ce.init(6);
  ce.reset();

  createEdge(cn(0), cn(1));
  createEdge(cn(1), cn(2));
  createEdge(cn(2), cn(0));
  createEdge(cn(0), cn(3));
  createEdge(cn(1), cn(3));
  createEdge(cn(2), cn(3));
}
//-----------------------------------------------------------------------------
void Tetrahedron::createFaces()
{
  cf.init(4);
  cf.reset();

  createFace(ce(0), ce(1), ce(2));
  createFace(ce(0), ce(4), ce(3));
  createFace(ce(1), ce(5), ce(4));
  createFace(ce(2), ce(5), ce(3));
}
//-----------------------------------------------------------------------------
