// Copyright (C) 2006 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Edge.h"
#include "Vertex.h"
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
double Edge::length() const
{
  const std::int32_t* vertices = entities(0);
  dolfin_assert(vertices);

  const Vertex v0(*_mesh, vertices[0]);
  const Vertex v1(*_mesh, vertices[1]);

  const Point p0 = v0.point();
  const Point p1 = v1.point();

  double length(sqrt((p1.x() - p0.x()) * (p1.x() - p0.x())
                     + (p1.y() - p0.y()) * (p1.y() - p0.y())
                     + (p1.z() - p0.z()) * (p1.z() - p0.z())));

  return length;
}
//-----------------------------------------------------------------------------
double Edge::dot(const Edge& edge) const
{
  const std::int32_t* v0 = entities(0);
  const std::int32_t* v1 = edge.entities(0);
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
