// Copyright (C) 2006-2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TetrahedronCell.h"
#include "Cell.h"
#include "Facet.h"
#include "Geometry.h"
#include "TriangleCell.h"
#include "Vertex.h"
#include <algorithm>
#include <cmath>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
double TetrahedronCell::squared_distance(const Cell& cell,
                                         const Eigen::Vector3d& point) const
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // ClosestPtPointTetrahedron on page 143, Section 5.1.6.
  //
  // Note: This algorithm actually computes the closest point but we
  // only return the distance to that point.

  // Get the vertices as points
  const Geometry& geometry = cell.mesh().geometry();
  const std::int32_t* vertices = cell.entities(0);
  const Eigen::Vector3d a = geometry.x(vertices[0]);
  const Eigen::Vector3d b = geometry.x(vertices[1]);
  const Eigen::Vector3d c = geometry.x(vertices[2]);
  const Eigen::Vector3d d = geometry.x(vertices[3]);

  // Initialize squared distance
  double r2 = std::numeric_limits<double>::max();

  // Check face ABC
  if (point_outside_of_plane(point, a, b, c, d))
    r2 = std::min(r2, TriangleCell::squared_distance(point, a, b, c));

  // Check face ACD
  if (point_outside_of_plane(point, a, c, d, b))
    r2 = std::min(r2, TriangleCell::squared_distance(point, a, c, d));

  // Check face ADB
  if (point_outside_of_plane(point, a, d, b, c))
    r2 = std::min(r2, TriangleCell::squared_distance(point, a, d, b));

  // Check facet BDC
  if (point_outside_of_plane(point, b, d, c, a))
    r2 = std::min(r2, TriangleCell::squared_distance(point, b, d, c));

  // Point is inside tetrahedron so distance is zero
  if (r2 == std::numeric_limits<double>::max())
    r2 = 0.0;

  return r2;
}
//-----------------------------------------------------------------------------
Eigen::Vector3d TetrahedronCell::normal(const Cell& cell,
                                        std::size_t facet) const
{
  // Make sure we have facets
  cell.mesh().create_connectivity(3, 2);

  // Create facet from the mesh and local facet number
  Facet f(cell.mesh(), cell.entities(2)[facet]);

  // Get global index of opposite vertex
  const std::size_t v0 = cell.entities(0)[facet];

  // Get global index of vertices on the facet
  std::size_t v1 = f.entities(0)[0];
  std::size_t v2 = f.entities(0)[1];
  std::size_t v3 = f.entities(0)[2];

  // Get mesh geometry
  const Geometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the four vertices
  const Eigen::Vector3d P0 = geometry.x(v0);
  const Eigen::Vector3d P1 = geometry.x(v1);
  const Eigen::Vector3d P2 = geometry.x(v2);
  const Eigen::Vector3d P3 = geometry.x(v3);

  // Create vectors
  Eigen::Vector3d V0 = P0 - P1;
  Eigen::Vector3d V1 = P2 - P1;
  Eigen::Vector3d V2 = P3 - P1;

  // Compute normal vector
  Eigen::Vector3d n = V1.cross(V2);

  // Normalize
  n /= n.norm();

  // Flip direction of normal so it points outward
  if (n.dot(V0) > 0)
    n *= -1.0;

  return n;
}
//-----------------------------------------------------------------------------
std::size_t TetrahedronCell::find_edge(std::size_t i, const Cell& cell) const
{
  // Get vertices and edges
  const std::int32_t* v = cell.entities(0);
  const std::int32_t* e = cell.entities(1);
  assert(v);
  assert(e);

  // Ordering convention for edges (order of non-incident vertices)
  static const std::size_t EV[6][2]
      = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

  // Look for edge satisfying ordering convention
  auto connectivity = cell.mesh().topology().connectivity(1, 0);
  assert(connectivity);
  for (std::size_t j = 0; j < 6; j++)
  {
    const std::int32_t* ev = connectivity->connections(e[j]);
    assert(ev);
    const std::int32_t v0 = v[EV[i][0]];
    const std::int32_t v1 = v[EV[i][1]];
    if (ev[0] != v0 && ev[0] != v1 && ev[1] != v0 && ev[1] != v1)
      return j;
  }

  // We should not reach this
  throw std::runtime_error("Edge not found");
  return 0;
}
//-----------------------------------------------------------------------------
bool TetrahedronCell::point_outside_of_plane(const Eigen::Vector3d& point,
                                             const Eigen::Vector3d& a,
                                             const Eigen::Vector3d& b,
                                             const Eigen::Vector3d& c,
                                             const Eigen::Vector3d& d) const
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // PointOutsideOfPlane on page 144, Section 5.1.6.

  const Eigen::Vector3d v = (b - a).cross(c - a);
  const double signp = v.dot(point - a);
  const double signd = v.dot(d - a);

  return signp * signd < 0.0;
}
//-----------------------------------------------------------------------------
