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
int IntervalCell::num_entities(int dim) const
{
  switch (dim)
  {
  case 0:
    return 2; // vertices
  case 1:
    return 1; // cells
  default:
    throw std::invalid_argument("Illegal dimension");
  }

  return 0;
}
//-----------------------------------------------------------------------------
int IntervalCell::num_vertices(int dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // cells
  default:
    throw std::invalid_argument("Illegal dimension");
  }

  return 0;
}
//-----------------------------------------------------------------------------
void IntervalCell::create_entities(
    Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        e,
    std::size_t dim, const std::int32_t* v) const
{
  // For completeness, IntervalCell has two 'edges'
  assert(dim == 0);

  // Resize data structure
  e.resize(2, 1);
  // Create the three edges
  e(0, 0) = v[0];
  e(1, 0) = v[1];
}
//-----------------------------------------------------------------------------
double IntervalCell::volume(const MeshEntity& interval) const
{
  // Check that we get an interval
  if (interval.dim() != 1)
    throw std::invalid_argument("Illegal dimension");

  // Get mesh geometry
  const Geometry& geometry = interval.mesh().geometry();

  // Get the coordinates of the two vertices
  const std::int32_t* vertices = interval.entities(0);
  const Eigen::Vector3d x0 = geometry.x(vertices[0]);
  const Eigen::Vector3d x1 = geometry.x(vertices[1]);

  return (x1 - x0).norm();
}
//-----------------------------------------------------------------------------
double IntervalCell::circumradius(const MeshEntity& interval) const
{
  // Check that we get an interval
  if (interval.dim() != 1)
    throw std::invalid_argument("Illegal dimension");

  // Circumradius is half the volume for an interval (line segment)
  return volume(interval) / 2.0;
}
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
double IntervalCell::normal(const Cell& cell, std::size_t facet,
                            std::size_t i) const
{
  return normal(cell, facet)[i];
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
Eigen::Vector3d IntervalCell::cell_normal(const Cell& cell) const
{
  // Get mesh geometry
  const Geometry& geometry = cell.mesh().geometry();

  // Cell_normal only defined for gdim = 1, 2 for now
  const std::size_t gdim = geometry.dim();
  if (gdim > 2)
  {
    throw std::invalid_argument("Illegal dimension");
  }

  // Get the two vertices as points
  const std::int32_t* vertices = cell.entities(0);
  Eigen::Vector3d p0 = geometry.x(vertices[0]);
  Eigen::Vector3d p1 = geometry.x(vertices[1]);

  // Define normal by rotating tangent counterclockwise
  Eigen::Vector3d t = p1 - p0;
  Eigen::Vector3d n(-t[1], t[0], 0.0);

  // Normalize
  n /= n.norm();

  return n;
}
//-----------------------------------------------------------------------------
double IntervalCell::facet_area(const Cell& cell, std::size_t facet) const
{
  return 1.0;
}
//-----------------------------------------------------------------------------
