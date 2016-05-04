// Copyright (C) 2006-2014 Anders Logg
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
// Modified by Kristian Oelgaard 2007
// Modified by Kristoffer Selim 2008
// Modified by Marie E. Rognes 2011
// Modified by August Johansson 2014
//
// First added:  2006-06-05
// Last changed: 2016-05-04

#include <algorithm>
#include <dolfin/log/log.h>
#include "Cell.h"
#include "MeshEditor.h"
#include "MeshEntity.h"
#include "MeshGeometry.h"
#include "IntervalCell.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::size_t IntervalCell::dim() const
{
  return 1;
}
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
    dolfin_error("IntervalCell.cpp",
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
    dolfin_error("IntervalCell.cpp",
                 "access number of vertices for subsimplex of interval cell",
                 "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t IntervalCell::orientation(const Cell& cell) const
{
  const Point up(0.0, 1.0);
  return cell.orientation(up);
}
//-----------------------------------------------------------------------------
void IntervalCell::create_entities(boost::multi_array<unsigned int, 2>& e,
                                   std::size_t dim, const unsigned int* v) const
{
  // For completeness, IntervalCell has two 'edges'
  dolfin_assert(dim == 0);

  // Resize data structure
  e.resize(boost::extents[2][1]);
  // Create the three edges
  e[0][0] = v[0]; e[1][0] = v[1];
}
//-----------------------------------------------------------------------------
double IntervalCell::volume(const MeshEntity& interval) const
{
  // Check that we get an interval
  if (interval.dim() != 1)
  {
    dolfin_error("IntervalCell.cpp",
                 "compute volume (length) of interval cell",
                 "Illegal mesh entity, not an interval");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = interval.mesh().geometry();

  // Get the coordinates of the two vertices
  const unsigned int* vertices = interval.entities(0);
  const Point x0 = geometry.point(vertices[0]);
  const Point x1 = geometry.point(vertices[1]);

  return x1.distance(x0);
}
//-----------------------------------------------------------------------------
double IntervalCell::circumradius(const MeshEntity& interval) const
{
  // Check that we get an interval
  if (interval.dim() != 1)
  {
    dolfin_error("IntervalCell.cpp",
                 "compute diameter of interval cell",
                 "Illegal mesh entity, not an interval");
  }

  // Circumradius is half the volume for an interval (line segment)
  return volume(interval)/2.0;
}
//-----------------------------------------------------------------------------
double IntervalCell::squared_distance(const Cell& cell,
                                      const Point& point) const
{
  // Get the vertices as points
  const MeshGeometry& geometry = cell.mesh().geometry();
  const unsigned int* vertices = cell.entities(0);
  const Point a = geometry.point(vertices[0]);
  const Point b = geometry.point(vertices[1]);

  // Call function to compute squared distance
  return squared_distance(point, a, b);
}
//-----------------------------------------------------------------------------
double IntervalCell::squared_distance(const Point& point,
                                      const Point& a,
                                      const Point& b)
{
  // Compute vector
  const Point v0  = point - a;
  const Point v1  = point - b;
  const Point v01 = b - a;

  // Check if a is closest point (outside of interval)
  const double a0 = v0.dot(v01);
  if (a0 < 0.0)
    return v0.dot(v0);

  // Check if b is closest point (outside the interval)
  const double a1 = - v1.dot(v01);
  if (a1 < 0.0)
    return v1.dot(v1);

  // Inside interval, so use Pythagoras to subtract length of projection
  return std::max(v0.dot(v0) - a0*a0 / v01.dot(v01), 0.0);
}
//-----------------------------------------------------------------------------
double IntervalCell::normal(const Cell& cell, std::size_t facet,
                            std::size_t i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
Point IntervalCell::normal(const Cell& cell, std::size_t facet) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the two vertices as points
  const unsigned int* vertices = cell.entities(0);
  Point p0 = geometry.point(vertices[0]);
  Point p1 = geometry.point(vertices[1]);

  // Compute normal
  Point n = p0 - p1;
  if (facet == 1)
    n *= -1.0;

  // Normalize
  n /= n.norm();

  return n;
}
//-----------------------------------------------------------------------------
Point IntervalCell::cell_normal(const Cell& cell) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Cell_normal only defined for gdim = 1, 2 for now
  const std::size_t gdim = geometry.dim();
  if (gdim > 2)
    dolfin_error("IntervalCell.cpp",
                 "compute cell normal",
                 "Illegal geometric dimension (%d)", gdim);

  // Get the two vertices as points
  const unsigned int* vertices = cell.entities(0);
  Point p0 = geometry.point(vertices[0]);
  Point p1 = geometry.point(vertices[1]);

  // Define normal by rotating tangent counterclockwise
  Point t = p1 - p0;
  Point n(-t.y(), t.x());

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
void IntervalCell::order(Cell& cell,
                         const std::vector<std::size_t>&
                         local_to_global_vertex_indices) const
{
  // Sort i - j for i > j: 1 - 0

  // Get mesh topology
  MeshTopology& topology = const_cast<MeshTopology&>(cell.mesh().topology());

  // Sort local vertices in ascending order, connectivity 1 - 0
  if (!topology(1, 0).empty())
  {
    unsigned int* cell_vertices = const_cast<unsigned int*>(cell.entities(0));
    sort_entities(2, cell_vertices, local_to_global_vertex_indices);
  }
}
//-----------------------------------------------------------------------------
bool IntervalCell::collides(const Cell& cell, const Point& point) const
{
  return CollisionDetection::collides(cell, point);
}
//-----------------------------------------------------------------------------
bool IntervalCell::collides(const Cell& cell, const MeshEntity& entity) const
{
  return CollisionDetection::collides(cell, entity);
}
//-----------------------------------------------------------------------------
std::vector<double>
IntervalCell::triangulate_intersection(const Cell& c0, const Cell& c1) const
{
  return IntersectionTriangulation::triangulate(c0, c1);
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
