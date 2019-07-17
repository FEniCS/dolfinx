// Copyright (C) 2006-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Facet.h"
#include "Cell.h"
#include "IntervalCell.h"
#include "TriangleCell.h"
#include <dolfin/geometry/utils.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
double Facet::squared_distance(const Eigen::Vector3d& point) const
{
  if (_dim == 1)
  {
    // Extract vertices
    const Geometry& geometry = _mesh->geometry();
    const std::int32_t* vertices = entities(0);
    const Eigen::Vector3d a = geometry.x(vertices[0]);
    const Eigen::Vector3d b = geometry.x(vertices[1]);

    // Compute squared distance
    return IntervalCell::squared_distance(point, a, b);
  }
  else if (_dim == 2)
  {
    // Extract vertices
    const Geometry& geometry = _mesh->geometry();
    const std::int32_t* vertices = entities(0);
    const Eigen::Vector3d a = geometry.x(vertices[0]);
    const Eigen::Vector3d b = geometry.x(vertices[1]);
    const Eigen::Vector3d c = geometry.x(vertices[2]);

    // Compute squared distance
    return geometry::squared_distance_triangle(point, a, b, c);
  }

  throw std::runtime_error(" Compute (squared) distance to facet not "
                           "implemented for facets of dimension "
                           + std::to_string(_dim));
  return 0.0;
}
//-----------------------------------------------------------------------------
bool Facet::exterior() const
{
  const std::size_t D = _mesh->topology().dim();
  if (this->num_global_entities(D) == 1)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
