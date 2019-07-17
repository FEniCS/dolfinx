// Copyright (C) 2006-2014 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IntervalCell.h"
#include "Cell.h"
#include "Geometry.h"
#include "MeshEntity.h"
#include <algorithm>
#include <stdexcept>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
double IntervalCell::squared_distance(const Cell& cell,
                                      const Eigen::Vector3d& point) const
{
  // Get the vertices as points
  const Geometry& geometry = cell.mesh().geometry();
  const std::int32_t* vertices = cell.entities(0);
  const Eigen::Vector3d a = geometry.x(vertices[0]);
  const Eigen::Vector3d b = geometry.x(vertices[1]);

  // Call function to compute squared distance
  return squared_distance(point, a, b);
}
//-----------------------------------------------------------------------------
double IntervalCell::squared_distance(const Eigen::Vector3d& point,
                                      const Eigen::Vector3d& a,
                                      const Eigen::Vector3d& b)
{
  // Compute vector
  const Eigen::Vector3d v0 = point - a;
  const Eigen::Vector3d v1 = point - b;
  const Eigen::Vector3d v01 = b - a;

  // Check if a is closest point (outside of interval)
  const double a0 = v0.dot(v01);
  if (a0 < 0.0)
    return v0.dot(v0);

  // Check if b is closest point (outside the interval)
  const double a1 = -v1.dot(v01);
  if (a1 < 0.0)
    return v1.dot(v1);

  // Inside interval, so use Pythagoras to subtract length of projection
  return std::max(v0.dot(v0) - a0 * a0 / v01.dot(v01), 0.0);
}
//-----------------------------------------------------------------------------
Eigen::Vector3d IntervalCell::normal(const Cell& cell, std::size_t facet) const
{
  // Get mesh geometry
  const Geometry& geometry = cell.mesh().geometry();

  // Get the two vertices as points
  const std::int32_t* vertices = cell.entities(0);
  Eigen::Vector3d p0 = geometry.x(vertices[0]);
  Eigen::Vector3d p1 = geometry.x(vertices[1]);

  // Compute normal
  Eigen::Vector3d n = p0 - p1;
  if (facet == 1)
    n *= -1.0;

  // Normalize
  n /= n.norm();

  return n;
}
//-----------------------------------------------------------------------------
