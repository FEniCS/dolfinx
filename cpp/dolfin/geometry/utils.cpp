// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/Geometry.h>
#include <dolfin/mesh/Mesh.h>

using namespace dolfin;

namespace
{
// Check whether point is outside region defined by facet ABC.
// The fourth vertex is needed to define the orientation.
bool point_outside_of_plane(const Eigen::Vector3d& point,
                            const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                            const Eigen::Vector3d& c, const Eigen::Vector3d& d)
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // PointOutsideOfPlane on page 144, Section 5.1.6.
  const Eigen::Vector3d v = (b - a).cross(c - a);
  const double signp = v.dot(point - a);
  const double signd = v.dot(d - a);
  return signp * signd < 0.0;
}

} // namespace

//-----------------------------------------------------------------------------
double geometry::squared_distance(const mesh::Cell& cell,
                                  const Eigen::Vector3d& point)
{
  const mesh::CellType type = cell.mesh().cell_type;
  const mesh::Geometry& geometry = cell.mesh().geometry();
  switch (type)
  {
  case (mesh::CellType::interval):
  {
    const std::int32_t* vertices = cell.entities(0);
    const Eigen::Vector3d a = geometry.x(vertices[0]);
    const Eigen::Vector3d b = geometry.x(vertices[1]);
    return geometry::squared_distance_interval(point, a, b);
  }
  case (mesh::CellType::triangle):
  {
    const std::int32_t* vertices = cell.entities(0);
    const Eigen::Vector3d a = geometry.x(vertices[0]);
    const Eigen::Vector3d b = geometry.x(vertices[1]);
    const Eigen::Vector3d c = geometry.x(vertices[2]);
    return geometry::squared_distance_triangle(point, a, b, c);
  }
  case (mesh::CellType::tetrahedron):
  {
    // Algorithm from Real-time collision detection by Christer Ericson:
    // ClosestPtPointTetrahedron on page 143, Section 5.1.6.
    //
    // Note: This algorithm actually computes the closest point but we
    // only return the distance to that point.

    // Get the vertices as points
    const std::int32_t* vertices = cell.entities(0);
    const Eigen::Vector3d a = geometry.x(vertices[0]);
    const Eigen::Vector3d b = geometry.x(vertices[1]);
    const Eigen::Vector3d c = geometry.x(vertices[2]);
    const Eigen::Vector3d d = geometry.x(vertices[3]);

    // Initialize squared distance
    double r2 = std::numeric_limits<double>::max();

    // Check face ABC
    if (point_outside_of_plane(point, a, b, c, d))
      r2 = std::min(r2, geometry::squared_distance_triangle(point, a, b, c));

    // Check face ACD
    if (point_outside_of_plane(point, a, c, d, b))
      r2 = std::min(r2, geometry::squared_distance_triangle(point, a, c, d));

    // Check face ADB
    if (point_outside_of_plane(point, a, d, b, c))
      r2 = std::min(r2, geometry::squared_distance_triangle(point, a, d, b));

    // Check facet BDC
    if (point_outside_of_plane(point, b, d, c, a))
      r2 = std::min(r2, geometry::squared_distance_triangle(point, b, d, c));

    // Point is inside tetrahedron so distance is zero
    if (r2 == std::numeric_limits<double>::max())
      r2 = 0.0;

    return r2;
  }
  default:
    throw std::invalid_argument(
        "cell_normal not supported for this cell type.");
  }
  return 0.0;
}
//-----------------------------------------------------------------------------
double geometry::squared_distance_triangle(const Eigen::Vector3d& point,
                                           const Eigen::Vector3d& a,
                                           const Eigen::Vector3d& b,
                                           const Eigen::Vector3d& c)
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // Closest Pt Point Triangle on page 141, Section 5.1.5.
  //
  // Algorithm modified to handle triangles embedded in 3D.
  //
  // Note: This algorithm actually computes the closest point but we
  // only return the distance to that point.

  // Compute normal to plane defined by triangle
  const Eigen::Vector3d ab = b - a;
  const Eigen::Vector3d ac = c - a;
  Eigen::Vector3d n = ab.cross(ac);
  n /= n.norm();

  // Subtract projection onto plane
  const double pn = (point - a).dot(n);
  const Eigen::Vector3d p = point - n * pn;

  // Check if point is in vertex region outside A
  const Eigen::Vector3d ap = p - a;
  const double d1 = ab.dot(ap);
  const double d2 = ac.dot(ap);
  if (d1 <= 0.0 && d2 <= 0.0)
    return ap.squaredNorm() + pn * pn;

  // Check if point is in vertex region outside B
  const Eigen::Vector3d bp = p - b;
  const double d3 = ab.dot(bp);
  const double d4 = ac.dot(bp);
  if (d3 >= 0.0 && d4 <= d3)
    return bp.squaredNorm() + pn * pn;

  // Check if point is in edge region of AB and if so compute projection
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
  {
    const double v = d1 / (d1 - d3);
    return (a + ab * v - p).squaredNorm() + pn * pn;
  }

  // Check if point is in vertex region outside C
  const Eigen::Vector3d cp = p - c;
  const double d5 = ab.dot(cp);
  const double d6 = ac.dot(cp);
  if (d6 >= 0.0 && d5 <= d6)
    return cp.squaredNorm() + pn * pn;

  // Check if point is in edge region of AC and if so compute projection
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
  {
    const double w = d2 / (d2 - d6);
    return (a + ac * w - p).squaredNorm() + pn * pn;
  }

  // Check if point is in edge region of BC and if so compute projection
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
  {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return (b + (c - b) * w - p).squaredNorm() + pn * pn;
  }

  // Point is inside triangle so return distance to plane
  return pn * pn;
}
//-----------------------------------------------------------------------------
double geometry::squared_distance_interval(const Eigen::Vector3d& point,
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
