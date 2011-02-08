// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman 2006.
//
// First added:  2006-06-02
// Last changed: 2011-02-08

#include <cmath>
#include <dolfin/common/types.h>
#include "Vertex.h"
#include "Edge.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
double Edge::length() const
{
  const uint* vertices = entities(0);
  assert(vertices);

  const Vertex v0(*_mesh, vertices[0]);
  const Vertex v1(*_mesh, vertices[1]);

  const Point p0 = v0.point();
  const Point p1 = v1.point();

  double length(sqrt((p1.x()-p0.x())*(p1.x()-p0.x())
               + (p1.y()-p0.y())*(p1.y()-p0.y())
               + (p1.z()-p0.z())*(p1.z()-p0.z())));

  return length;
}
//-----------------------------------------------------------------------------
double Edge::inner(const Edge& edge) const
{
  const uint* v0 = entities(0);
  const uint* v1 = edge.entities(0);
  assert(v0);
  assert(v1);

  const MeshGeometry& g = _mesh->geometry();
  const double* x00 = g.x(v0[0]);
  const double* x01 = g.x(v0[1]);
  const double* x10 = g.x(v1[0]);
  const double* x11 = g.x(v1[1]);

  double sum = 0.0;
  const uint gdim = g.dim();
  for (uint i = 0; i < gdim; i++)
    sum += (x01[i] - x00[i]) * (x11[i] - x10[i]);

  return sum;
}
//-----------------------------------------------------------------------------
