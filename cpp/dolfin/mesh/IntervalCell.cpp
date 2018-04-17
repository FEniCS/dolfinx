// Copyright (C) 2006-2014 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IntervalCell.h"
#include "Cell.h"
#include "MeshEntity.h"
#include "MeshGeometry.h"
#include <algorithm>
#include <dolfin/log/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
std::size_t IntervalCell::dim() const { return 1; }
//-----------------------------------------------------------------------------
std::size_t IntervalCell::num_entities(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 2; // vertices
  case 1:
    return 1; // cells
  default:
    log::dolfin_error("IntervalCell.cpp",
                      "access number of entities of interval cell",
                      "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t IntervalCell::num_vertices(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // cells
  default:
    log::dolfin_error(
        "IntervalCell.cpp",
        "access number of vertices for subsimplex of interval cell",
        "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
void IntervalCell::create_entities(boost::multi_array<std::int32_t, 2>& e,
                                   std::size_t dim, const std::int32_t* v) const
{
  // For completeness, IntervalCell has two 'edges'
  assert(dim == 0);

  // Resize data structure
  e.resize(boost::extents[2][1]);
  // Create the three edges
  e[0][0] = v[0];
  e[1][0] = v[1];
}
//-----------------------------------------------------------------------------
double IntervalCell::volume(const MeshEntity& interval) const
{
  // Check that we get an interval
  if (interval.dim() != 1)
  {
    log::dolfin_error("IntervalCell.cpp",
                      "compute volume (length) of interval cell",
                      "Illegal mesh entity, not an interval");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = interval.mesh().geometry();

  // Get the coordinates of the two vertices
  const std::int32_t* vertices = interval.entities(0);
  const geometry::Point x0 = geometry.point(vertices[0]);
  const geometry::Point x1 = geometry.point(vertices[1]);

  return x1.distance(x0);
}
//-----------------------------------------------------------------------------
double IntervalCell::circumradius(const MeshEntity& interval) const
{
  // Check that we get an interval
  if (interval.dim() != 1)
  {
    log::dolfin_error("IntervalCell.cpp", "compute diameter of interval cell",
                      "Illegal mesh entity, not an interval");
  }

  // Circumradius is half the volume for an interval (line segment)
  return volume(interval) / 2.0;
}
//-----------------------------------------------------------------------------
double IntervalCell::squared_distance(const Cell& cell,
                                      const geometry::Point& point) const
{
  // Get the vertices as points
  const MeshGeometry& geometry = cell.mesh().geometry();
  const std::int32_t* vertices = cell.entities(0);
  const geometry::Point a = geometry.point(vertices[0]);
  const geometry::Point b = geometry.point(vertices[1]);

  // Call function to compute squared distance
  return squared_distance(point, a, b);
}
//-----------------------------------------------------------------------------
double IntervalCell::squared_distance(const geometry::Point& point,
                                      const geometry::Point& a,
                                      const geometry::Point& b)
{
  // Compute vector
  const geometry::Point v0 = point - a;
  const geometry::Point v1 = point - b;
  const geometry::Point v01 = b - a;

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
geometry::Point IntervalCell::normal(const Cell& cell, std::size_t facet) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the two vertices as points
  const std::int32_t* vertices = cell.entities(0);
  geometry::Point p0 = geometry.point(vertices[0]);
  geometry::Point p1 = geometry.point(vertices[1]);

  // Compute normal
  geometry::Point n = p0 - p1;
  if (facet == 1)
    n *= -1.0;

  // Normalize
  n /= n.norm();

  return n;
}
//-----------------------------------------------------------------------------
geometry::Point IntervalCell::cell_normal(const Cell& cell) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Cell_normal only defined for gdim = 1, 2 for now
  const std::size_t gdim = geometry.dim();
  if (gdim > 2)
    log::dolfin_error("IntervalCell.cpp", "compute cell normal",
                      "Illegal geometric dimension (%d)", gdim);

  // Get the two vertices as points
  const std::int32_t* vertices = cell.entities(0);
  geometry::Point p0 = geometry.point(vertices[0]);
  geometry::Point p1 = geometry.point(vertices[1]);

  // Define normal by rotating tangent counterclockwise
  geometry::Point t = p1 - p0;
  geometry::Point n(-t[1], t[0]);

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
std::string IntervalCell::description(bool plural) const
{
  if (plural)
    return "intervals";
  else
    return "interval";
}
//-----------------------------------------------------------------------------
