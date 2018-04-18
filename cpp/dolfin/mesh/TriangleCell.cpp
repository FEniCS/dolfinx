// Copyright (C) 2006-2014 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TriangleCell.h"
#include "Cell.h"
#include "Facet.h"
#include "MeshEntity.h"
#include "Vertex.h"
#include <algorithm>
#include <cmath>
#include <dolfin/log/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
std::size_t TriangleCell::dim() const { return 2; }
//-----------------------------------------------------------------------------
std::size_t TriangleCell::num_entities(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 3; // vertices
  case 1:
    return 3; // edges
  case 2:
    return 1; // cells
  default:
    log::dolfin_error("TriangleCell.cpp",
                      "access number of entities of triangle cell",
                      "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t TriangleCell::num_vertices(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // edges
  case 2:
    return 3; // cells
  default:
    log::dolfin_error(
        "TriangleCell.cpp",
        "access number of vertices for subsimplex of triangle cell",
        "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
void TriangleCell::create_entities(boost::multi_array<std::int32_t, 2>& e,
                                   std::size_t dim, const std::int32_t* v) const
{
  // We only need to know how to create edges
  if (dim != 1)
  {
    log::dolfin_error(
        "TriangleCell.cpp", "create entities of triangle cell",
        "Don't know how to create entities of topological dimension %d", dim);
  }

  // Resize data structure
  e.resize(boost::extents[3][2]);

  // Create the three edges
  e[0][0] = v[1];
  e[0][1] = v[2];
  e[1][0] = v[0];
  e[1][1] = v[2];
  e[2][0] = v[0];
  e[2][1] = v[1];
}
//-----------------------------------------------------------------------------
double TriangleCell::volume(const MeshEntity& triangle) const
{
  // Check that we get a triangle
  if (triangle.dim() != 2)
  {
    log::dolfin_error("TriangleCell.cpp",
                      "compute volume (area) of triangle cell",
                      "Illegal mesh entity, not a triangle");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = triangle.mesh().geometry();

  // Get the coordinates of the three vertices
  const std::int32_t* vertices = triangle.entities(0);
  const geometry::Point x0 = geometry.point(vertices[0]);
  const geometry::Point x1 = geometry.point(vertices[1]);
  const geometry::Point x2 = geometry.point(vertices[2]);

  if (geometry.dim() == 2)
  {
    // Compute area of triangle embedded in R^2
    double v2 = (x0[0] * x1[1] + x0[1] * x2[0] + x1[0] * x2[1])
                - (x2[0] * x1[1] + x2[1] * x0[0] + x1[0] * x0[1]);

    // Formula for volume from http://mathworld.wolfram.com
    return 0.5 * std::abs(v2);
  }
  else if (geometry.dim() == 3)
  {
    // Compute area of triangle embedded in R^3
    const double v0 = (x0[1] * x1[2] + x0[2] * x2[1] + x1[1] * x2[2])
                      - (x2[1] * x1[2] + x2[2] * x0[1] + x1[1] * x0[2]);
    const double v1 = (x0[2] * x1[0] + x0[0] * x2[2] + x1[2] * x2[0])
                      - (x2[2] * x1[0] + x2[0] * x0[2] + x1[2] * x0[0]);
    const double v2 = (x0[0] * x1[1] + x0[1] * x2[0] + x1[0] * x2[1])
                      - (x2[0] * x1[1] + x2[1] * x0[0] + x1[0] * x0[1]);

    // Formula for volume from http://mathworld.wolfram.com
    return 0.5 * sqrt(v0 * v0 + v1 * v1 + v2 * v2);
  }
  else
  {
    log::dolfin_error(
        "TriangleCell.cpp", "compute volume of triangle",
        "Only know how to compute volume when embedded in R^2 or R^3");
  }

  return 0.0;
}
//-----------------------------------------------------------------------------
double TriangleCell::circumradius(const MeshEntity& triangle) const
{
  // Check that we get a triangle
  if (triangle.dim() != 2)
  {
    log::dolfin_error("TriangleCell.cpp", "compute diameter of triangle cell",
                      "Illegal mesh entity, not a triangle");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = triangle.mesh().geometry();

  // Only know how to compute the diameter when embedded in R^2 or R^3
  if (geometry.dim() != 2 && geometry.dim() != 3)
    log::dolfin_error(
        "TriangleCell.cpp", "compute diameter of triangle",
        "Only know how to compute diameter when embedded in R^2 or R^3");

  // Get the coordinates of the three vertices
  const std::int32_t* vertices = triangle.entities(0);
  const geometry::Point p0 = geometry.point(vertices[0]);
  const geometry::Point p1 = geometry.point(vertices[1]);
  const geometry::Point p2 = geometry.point(vertices[2]);

  // FIXME: Assuming 3D coordinates, could be more efficient if
  // FIXME: if we assumed 2D coordinates in 2D

  // Compute side lengths
  const double a = p1.distance(p2);
  const double b = p0.distance(p2);
  const double c = p0.distance(p1);

  // Formula for circumradius from
  // http://mathworld.wolfram.com/Triangle.html
  return a * b * c / (4.0 * volume(triangle));
}
//-----------------------------------------------------------------------------
double TriangleCell::squared_distance(const Cell& cell,
                                      const geometry::Point& point) const
{
  // Get the vertices as points
  const MeshGeometry& geometry = cell.mesh().geometry();
  const std::int32_t* vertices = cell.entities(0);
  const geometry::Point a = geometry.point(vertices[0]);
  const geometry::Point b = geometry.point(vertices[1]);
  const geometry::Point c = geometry.point(vertices[2]);

  // Call function to compute squared distance
  return squared_distance(point, a, b, c);
}
//-----------------------------------------------------------------------------
double TriangleCell::squared_distance(const geometry::Point& point,
                                      const geometry::Point& a,
                                      const geometry::Point& b,
                                      const geometry::Point& c)
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // ClosestPtgeometry::PointTriangle on page 141, Section 5.1.5.
  //
  // Algorithm modified to handle triangles embedded in 3D.
  //
  // Note: This algorithm actually computes the closest point but we
  // only return the distance to that point.

  // Compute normal to plane defined by triangle
  const geometry::Point ab = b - a;
  const geometry::Point ac = c - a;
  geometry::Point n = ab.cross(ac);
  n /= n.norm();

  // Subtract projection onto plane
  const double pn = (point - a).dot(n);
  const geometry::Point p = point - n * pn;

  // Check if point is in vertex region outside A
  const geometry::Point ap = p - a;
  const double d1 = ab.dot(ap);
  const double d2 = ac.dot(ap);
  if (d1 <= 0.0 && d2 <= 0.0)
    return p.squared_distance(a) + pn * pn;

  // Check if point is in vertex region outside B
  const geometry::Point bp = p - b;
  const double d3 = ab.dot(bp);
  const double d4 = ac.dot(bp);
  if (d3 >= 0.0 && d4 <= d3)
    return p.squared_distance(b) + pn * pn;

  // Check if point is in edge region of AB and if so compute projection
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
  {
    const double v = d1 / (d1 - d3);
    return p.squared_distance(a + ab * v) + pn * pn;
  }

  // Check if point is in vertex region outside C
  const geometry::Point cp = p - c;
  const double d5 = ab.dot(cp);
  const double d6 = ac.dot(cp);
  if (d6 >= 0.0 && d5 <= d6)
    return p.squared_distance(c) + pn * pn;

  // Check if point is in edge region of AC and if so compute
  // projection
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
  {
    const double w = d2 / (d2 - d6);
    return p.squared_distance(a + ac * w) + pn * pn;
  }

  // Check if point is in edge region of BC and if so compute
  // projection
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
  {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return p.squared_distance(b + (c - b) * w) + pn * pn;
  }

  // Point is inside triangle so return distance to plane
  return pn * pn;
}
//-----------------------------------------------------------------------------
double TriangleCell::normal(const Cell& cell, std::size_t facet,
                            std::size_t i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
geometry::Point TriangleCell::normal(const Cell& cell, std::size_t facet) const
{
  // Make sure we have facets
  cell.mesh().init(2, 1);

  // Create facet from the mesh and local facet number
  Facet f(cell.mesh(), cell.entities(1)[facet]);

  // The normal vector is currently only defined for a triangle in R^2
  // MER: This code is super for a triangle in R^3 too, this error
  // could be removed, unless it is here for some other reason.
  if (cell.mesh().geometry().dim() != 2)
  {
    log::dolfin_error(
        "TriangleCell.cpp", "find normal",
        "Normal vector is not defined in dimension %d (only defined "
        "when the triangle is in R^2",
        cell.mesh().geometry().dim());
  }

  // Get global index of opposite vertex
  const std::size_t v0 = cell.entities(0)[facet];

  // Get global index of vertices on the facet
  const std::size_t v1 = f.entities(0)[0];
  const std::size_t v2 = f.entities(0)[1];

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the three vertices
  const geometry::Point p0 = geometry.point(v0);
  const geometry::Point p1 = geometry.point(v1);
  const geometry::Point p2 = geometry.point(v2);

  // Subtract projection of p2 - p0 onto p2 - p1
  geometry::Point t = p2 - p1;
  t /= t.norm();
  geometry::Point n = p2 - p0;
  n -= t * n.dot(t);

  // Normalize
  n /= n.norm();

  return n;
}
//-----------------------------------------------------------------------------
geometry::Point TriangleCell::cell_normal(const Cell& cell) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Cell_normal only defined for gdim = 2, 3:
  const std::size_t gdim = geometry.dim();
  if (gdim > 3)
  {
    log::dolfin_error("TriangleCell.cpp", "compute cell normal",
                      "Illegal geometric dimension (%d)", gdim);
  }

  // Get the three vertices as points
  const std::int32_t* vertices = cell.entities(0);
  const geometry::Point p0 = geometry.point(vertices[0]);
  const geometry::Point p1 = geometry.point(vertices[1]);
  const geometry::Point p2 = geometry.point(vertices[2]);

  // Defined cell normal via cross product of first two edges:
  const geometry::Point v01 = p1 - p0;
  const geometry::Point v02 = p2 - p0;
  geometry::Point n = v01.cross(v02);

  // Normalize
  n /= n.norm();

  return n;
}
//-----------------------------------------------------------------------------
double TriangleCell::facet_area(const Cell& cell, std::size_t facet) const
{
  // Create facet from the mesh and local facet number
  const Facet f(cell.mesh(), cell.entities(1)[facet]);

  // Get global index of vertices on the facet
  const std::size_t v0 = f.entities(0)[0];
  const std::size_t v1 = f.entities(0)[1];

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the two vertices
  const geometry::Point p0 = geometry.point(v0);
  const geometry::Point p1 = geometry.point(v1);

  return p1.distance(p0);
}
//-----------------------------------------------------------------------------
std::string TriangleCell::description(bool plural) const
{
  if (plural)
    return "triangles";
  return "triangle";
}
//-----------------------------------------------------------------------------
std::size_t TriangleCell::find_edge(std::size_t i, const Cell& cell) const
{
  // Get vertices and edges
  const std::int32_t* v = cell.entities(0);
  const std::int32_t* e = cell.entities(1);
  assert(v);
  assert(e);

  // Look for edge satisfying ordering convention
  for (std::size_t j = 0; j < 3; j++)
  {
    const std::int32_t* ev = cell.mesh().topology().connectivity(1, 0)(e[j]);
    assert(ev);
    if (ev[0] != v[i] && ev[1] != v[i])
      return j;
  }

  // We should not reach this
  log::dolfin_error("TriangleCell.cpp", "find specified edge in cell",
                    "Edge really not found");
  return 0;
}
//-----------------------------------------------------------------------------
