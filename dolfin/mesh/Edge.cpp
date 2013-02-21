// Copyright (C) 2006 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Johan Hoffman 2006.
//
// First added:  2006-06-02
// Last changed: 2011-02-08

#include <cmath>
#include "Vertex.h"
#include "Edge.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
double Edge::length() const
{
  const unsigned int* vertices = entities(0);
  dolfin_assert(vertices);

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
double Edge::dot(const Edge& edge) const
{
  const unsigned int* v0 = entities(0);
  const unsigned int* v1 = edge.entities(0);
  dolfin_assert(v0);
  dolfin_assert(v1);

  const MeshGeometry& g = _mesh->geometry();
  const double* x00 = g.x(v0[0]);
  const double* x01 = g.x(v0[1]);
  const double* x10 = g.x(v1[0]);
  const double* x11 = g.x(v1[1]);

  double sum = 0.0;
  const std::size_t gdim = g.dim();
  for (std::size_t i = 0; i < gdim; i++)
    sum += (x01[i] - x00[i]) * (x11[i] - x10[i]);

  return sum;
}
//-----------------------------------------------------------------------------
