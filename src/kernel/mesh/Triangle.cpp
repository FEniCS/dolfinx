// Copyright (C) 2002-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2006-02-20

#include <cmath>

#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
#include <dolfin/Vertex.h>
#include <dolfin/Point.h>
#include <dolfin/Edge.h>
#include <dolfin/Cell.h>
#include <dolfin/Triangle.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Triangle::Triangle(Vertex& n0, Vertex& n1, Vertex& n2) : GenericCell()
{
  cn.init(numVertices());

  cn(0) = &n0;
  cn(1) = &n1;
  cn(2) = &n2;
}
//-----------------------------------------------------------------------------
int Triangle::numVertices() const
{
  return 3;
}
//-----------------------------------------------------------------------------
int Triangle::numEdges() const
{
  return 3;
}
//-----------------------------------------------------------------------------
int Triangle::numFaces() const
{
  return 1;
}
//-----------------------------------------------------------------------------
int Triangle::numBoundaries() const
{
  return numEdges();
}
//-----------------------------------------------------------------------------
Cell::Type Triangle::type() const
{
  return Cell::triangle;
}
//-----------------------------------------------------------------------------
Cell::Orientation Triangle::orientation() const
{
  Point v01 = cn(1)->coord() - cn(0)->coord();
  Point v02 = cn(2)->coord() - cn(0)->coord();
  Point n(-v01.y, v01.x);

  return ( n * v02 < 0.0 ? Cell::left : Cell::right );
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
dolfin::uint Triangle::edgeAlignment(uint i) const
{
  // Check alignment with convention in FFC manual
  return ( cn((i + 1) % 3) == ce(i)->n0 ? 0 : 1 );
}
//-----------------------------------------------------------------------------
dolfin::uint Triangle::faceAlignment(uint i) const
{
  dolfin_error("A triangle has no faces.");
  return 0;
}
//-----------------------------------------------------------------------------
void Triangle::createEdges()
{
  ce.init(3);
  ce.reset();

  createEdge(*cn(1), *cn(2));
  createEdge(*cn(2), *cn(0));
  createEdge(*cn(0), *cn(1));
}
//-----------------------------------------------------------------------------
void Triangle::createFaces()
{
  // A triangle has no faces
}
//-----------------------------------------------------------------------------
void Triangle::sort()
{
  // Sort local mesh entities according to ordering in FFC manual

  // Sort the vertices counter-clockwise
  if ( orientation() == Cell::left )
  {
    Vertex* tmp = cn(1);
    cn(1) = cn(2);
    cn(2) = tmp;
  }
  
  // Sort the edges to have edge i opposite to vertex i
  Array<Edge*> edges(3);
  edges = 0;
  for (uint i = 0; i < 3; i++)
  {
    Vertex* n = cn(i);
    for (uint j = 0; j < 3; j++)
    {
      Edge* e = ce(j);
      if ( !(e->contains(*n)) )
      {
	edges[i] = e;
	break;
      }
    }
  }
  for (uint i = 0; i < 3; i++)
  {
    dolfin_assert(edges[i]);
    ce(i) = edges[i];
  }
}
//-----------------------------------------------------------------------------
