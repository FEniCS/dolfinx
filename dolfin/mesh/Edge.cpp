// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman 2006.
//
// First added:  2006-06-02
// Last changed: 2007-01-10

#include <cmath>
#include <dolfin/common/types.h>
#include "Vertex.h"
#include "Edge.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
double Edge::length()
{
  uint* vertices = entities(0);
  dolfin_assert(vertices);

  const Vertex v0(_mesh, vertices[0]);
  const Vertex v1(_mesh, vertices[1]);
  
  const Point p0 = v0.point();
  const Point p1 = v1.point();

  double length(sqrt((p1.x()-p0.x())*(p1.x()-p0.x()) + 
		   (p1.y()-p0.y())*(p1.y()-p0.y()) + 
		   (p1.z()-p0.z())*(p1.z()-p0.z())));

  return length;
}
//-----------------------------------------------------------------------------
Point Edge::midpoint()
{
  uint* vertices = entities(0);
  dolfin_assert(vertices);

  const Vertex v0(_mesh, vertices[0]);
  const Vertex v1(_mesh, vertices[1]);
  
  const Point p0 = v0.point();
  const Point p1 = v1.point();

  Point p(0.5*(p0.x() + p1.x()),
	  0.5*(p0.y() + p1.y()),
	  0.5*(p0.z() + p1.z()));

  return p;
}
//-----------------------------------------------------------------------------
