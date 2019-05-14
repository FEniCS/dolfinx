// Copyright (C) 2006 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Edge.h"
#include "Vertex.h"
#include <cmath>
#include <dolfin/mesh/Geometry.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
double Edge::length() const
{
  const std::int32_t* vertices = entities(0);
  assert(vertices);

  const Vertex v0(*_mesh, vertices[0]);
  const Vertex v1(*_mesh, vertices[1]);

  const Eigen::Vector3d p0 = v0.x();
  const Eigen::Vector3d p1 = v1.x();

  return (p0 - p1).norm();
}
//-----------------------------------------------------------------------------
double Edge::dot(const Edge& edge) const
{
  const std::int32_t* v0 = entities(0);
  const std::int32_t* v1 = edge.entities(0);
  assert(v0);
  assert(v1);

  const Geometry& g = _mesh->geometry();
  const Eigen::Ref<const Eigen::Vector3d> x00 = g.x(v0[0]);
  const Eigen::Ref<const Eigen::Vector3d> x01 = g.x(v0[1]);
  const Eigen::Ref<const Eigen::Vector3d> x10 = g.x(v1[0]);
  const Eigen::Ref<const Eigen::Vector3d> x11 = g.x(v1[1]);

  double sum = 0.0;
  const std::size_t gdim = g.dim();
  for (std::size_t i = 0; i < gdim; i++)
    sum += (x01[i] - x00[i]) * (x11[i] - x10[i]);

  return sum;
}
//-----------------------------------------------------------------------------
