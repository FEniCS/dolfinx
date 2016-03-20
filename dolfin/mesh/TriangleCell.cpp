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
// Modified by Garth N. Wells 2006
// Modified by Kristian Oelgaard 2006-2007
// Modified by Dag Lindbo 2008
// Modified by Kristoffer Selim 2008
// Modified by Jan Blechta 2013
// Modified by August Johansson 2014
//
// First added:  2006-06-05
// Last changed: 2014-05-22

#include <algorithm>
#include <cmath>
#include <dolfin/log/log.h>
#include "Cell.h"
#include "MeshEditor.h"
#include "MeshEntity.h"
#include "Facet.h"
#include "TriangleCell.h"
#include "Vertex.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::size_t TriangleCell::dim() const
{
  return 2;
}
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
    dolfin_error("TriangleCell.cpp",
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
    dolfin_error("TriangleCell.cpp",
                 "access number of vertices for subsimplex of triangle cell",
                 "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t TriangleCell::orientation(const Cell& cell) const
{
  const Point up(0.0, 0.0, 1.0);
  return cell.orientation(up);
}
//-----------------------------------------------------------------------------
void TriangleCell::create_entities(boost::multi_array<unsigned int, 2>&  e,
                                   std::size_t dim, const unsigned int* v) const
{
  // We only need to know how to create edges
  if (dim != 1)
  {
    dolfin_error("TriangleCell.cpp",
                 "create entities of triangle cell",
                 "Don't know how to create entities of topological dimension %d", dim);
  }

  // Resize data structure
  e.resize(boost::extents[3][2]);

  // Create the three edges
  e[0][0] = v[1]; e[0][1] = v[2];
  e[1][0] = v[0]; e[1][1] = v[2];
  e[2][0] = v[0]; e[2][1] = v[1];
}
//-----------------------------------------------------------------------------
double TriangleCell::volume(const MeshEntity& triangle) const
{
  // Check that we get a triangle
  if (triangle.dim() != 2)
  {
    dolfin_error("TriangleCell.cpp",
                 "compute volume (area) of triangle cell",
                 "Illegal mesh entity, not a triangle");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = triangle.mesh().geometry();

  // Get the coordinates of the three vertices
  const unsigned int* vertices = triangle.entities(0);
  const Point x0 = geometry.point(vertices[0]);
  const Point x1 = geometry.point(vertices[1]);
  const Point x2 = geometry.point(vertices[2]);

  if (geometry.dim() == 2)
  {
    // Compute area of triangle embedded in R^2
    double v2 = (x0[0]*x1[1] + x0[1]*x2[0] + x1[0]*x2[1])
      - (x2[0]*x1[1] + x2[1]*x0[0] + x1[0]*x0[1]);

    // Formula for volume from http://mathworld.wolfram.com
    return 0.5 * std::abs(v2);
  }
  else if (geometry.dim() == 3)
  {
    // Compute area of triangle embedded in R^3
    const double v0 = (x0[1]*x1[2] + x0[2]*x2[1] + x1[1]*x2[2])
      - (x2[1]*x1[2] + x2[2]*x0[1] + x1[1]*x0[2]);
    const double v1 = (x0[2]*x1[0] + x0[0]*x2[2] + x1[2]*x2[0])
      - (x2[2]*x1[0] + x2[0]*x0[2] + x1[2]*x0[0]);
    const double v2 = (x0[0]*x1[1] + x0[1]*x2[0] + x1[0]*x2[1])
      - (x2[0]*x1[1] + x2[1]*x0[0] + x1[0]*x0[1]);

    // Formula for volume from http://mathworld.wolfram.com
    return  0.5*sqrt(v0*v0 + v1*v1 + v2*v2);
  }
  else
  {
    dolfin_error("TriangleCell.cpp",
                 "compute volume of triangle",
                 "Only know how to compute volume when embedded in R^2 or R^3");
  }

  return 0.0;
}
//-----------------------------------------------------------------------------
double TriangleCell::diameter(const MeshEntity& triangle) const
{
  // Check that we get a triangle
  if (triangle.dim() != 2)
  {
    dolfin_error("TriangleCell.cpp",
                 "compute diameter of triangle cell",
                 "Illegal mesh entity, not a triangle");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = triangle.mesh().geometry();

  // Only know how to compute the diameter when embedded in R^2 or R^3
  if (geometry.dim() != 2 && geometry.dim() != 3)
    dolfin_error("TriangleCell.cpp",
                 "compute diameter of triangle",
                 "Only know how to compute diameter when embedded in R^2 or R^3");

  // Get the coordinates of the three vertices
  const unsigned int* vertices = triangle.entities(0);
  const Point p0 = geometry.point(vertices[0]);
  const Point p1 = geometry.point(vertices[1]);
  const Point p2 = geometry.point(vertices[2]);

  // FIXME: Assuming 3D coordinates, could be more efficient if
  // FIXME: if we assumed 2D coordinates in 2D

  // Compute side lengths
  const double a  = p1.distance(p2);
  const double b  = p0.distance(p2);
  const double c  = p0.distance(p1);

  // Formula for diameter (2*circumradius) from http://mathworld.wolfram.com
  return 0.5*a*b*c / volume(triangle);
}
//-----------------------------------------------------------------------------
double TriangleCell::squared_distance(const Cell& cell,
                                      const Point& point) const
{
  // Get the vertices as points
  const MeshGeometry& geometry = cell.mesh().geometry();
  const unsigned int* vertices = cell.entities(0);
  const Point a = geometry.point(vertices[0]);
  const Point b = geometry.point(vertices[1]);
  const Point c = geometry.point(vertices[2]);

  // Call function to compute squared distance
  return squared_distance(point, a, b, c);
}
//-----------------------------------------------------------------------------
double TriangleCell::squared_distance(const Point& point,
                                      const Point& a,
                                      const Point& b,
                                      const Point& c)
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // ClosestPtPointTriangle on page 141, Section 5.1.5.
  //
  // Algorithm modified to handle triangles embedded in 3D.
  //
  // Note: This algorithm actually computes the closest point but we
  // only return the distance to that point.

  // Compute normal to plane defined by triangle
  const Point ab = b - a;
  const Point ac = c - a;
  Point n = ab.cross(ac);
  n /= n.norm();

  // Subtract projection onto plane
  const double pn = (point - a).dot(n);
  const Point p = point - pn*n;

  // Check if point is in vertex region outside A
  const Point ap = p - a;
  const double d1 = ab.dot(ap);
  const double d2 = ac.dot(ap);
  if (d1 <= 0.0 && d2 <= 0.0)
    return p.squared_distance(a) + pn*pn;

  // Check if point is in vertex region outside B
  const Point bp = p - b;
  const double d3 = ab.dot(bp);
  const double d4 = ac.dot(bp);
  if (d3 >= 0.0 && d4 <= d3)
    return p.squared_distance(b) + pn*pn;

  // Check if point is in edge region of AB and if so compute projection
  const double vc = d1*d4 - d3*d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
  {
    const double v = d1 / (d1 - d3);
    return p.squared_distance(a + v*ab) + pn*pn;
  }

  // Check if point is in vertex region outside C
  const Point cp = p - c;
  const double d5 = ab.dot(cp);
  const double d6 = ac.dot(cp);
  if (d6 >= 0.0 && d5 <= d6)
    return p.squared_distance(c) + pn*pn;

  // Check if point is in edge region of AC and if so compute
  // projection
  const double vb = d5*d2 - d1*d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
  {
    const double w = d2 / (d2 - d6);
    return p.squared_distance(a + w*ac) + pn*pn;
  }

  // Check if point is in edge region of BC and if so compute
  // projection
  const double va = d3*d6 - d5*d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
  {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return p.squared_distance(b + w*(c - b)) + pn*pn;
  }

  // Point is inside triangle so return distance to plane
  return pn*pn;
}
//-----------------------------------------------------------------------------
double TriangleCell::normal(const Cell& cell, std::size_t facet, std::size_t i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
Point TriangleCell::normal(const Cell& cell, std::size_t facet) const
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
    dolfin_error("TriangleCell.cpp",
                 "find normal",
                 "Normal vector is not defined in dimension %d (only defined when the triangle is in R^2", cell.mesh().geometry().dim());
  }

  // Get global index of opposite vertex
  const std::size_t v0 = cell.entities(0)[facet];

  // Get global index of vertices on the facet
  const std::size_t v1 = f.entities(0)[0];
  const std::size_t v2 = f.entities(0)[1];

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the three vertices
  const Point p0 = geometry.point(v0);
  const Point p1 = geometry.point(v1);
  const Point p2 = geometry.point(v2);

  // Subtract projection of p2 - p0 onto p2 - p1
  Point t = p2 - p1;
  t /= t.norm();
  Point n = p2 - p0;
  n -= n.dot(t)*t;

  // Normalize
  n /= n.norm();

  return n;
}
//-----------------------------------------------------------------------------
Point TriangleCell::cell_normal(const Cell& cell) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Cell_normal only defined for gdim = 2, 3:
  const std::size_t gdim = geometry.dim();
  if (gdim > 3)
  {
    dolfin_error("TriangleCell.cpp",
                 "compute cell normal",
                 "Illegal geometric dimension (%d)", gdim);
  }

  // Get the three vertices as points
  const unsigned int* vertices = cell.entities(0);
  const Point p0 = geometry.point(vertices[0]);
  const Point p1 = geometry.point(vertices[1]);
  const Point p2 = geometry.point(vertices[2]);

  // Defined cell normal via cross product of first two edges:
  const Point v01 = p1 - p0;
  const Point v02 = p2 - p0;
  Point n = v01.cross(v02);

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
  const Point p0 = geometry.point(v0);
  const Point p1 = geometry.point(v1);

  return p1.distance(p0);
}
//-----------------------------------------------------------------------------
void TriangleCell::order(
  Cell& cell,
  const std::vector<std::size_t>& local_to_global_vertex_indices) const
{
  // Sort i - j for i > j: 1 - 0, 2 - 0, 2 - 1

  // Get mesh topology
  const MeshTopology& topology = cell.mesh().topology();

  // Sort local vertices on edges in ascending order, connectivity 1 - 0
  if (!topology(1, 0).empty())
  {
    dolfin_assert(!topology(2, 1).empty());

    // Get edge indices (local)
    const unsigned int* cell_edges = cell.entities(1);

    // Sort vertices on each edge
    for (std::size_t i = 0; i < 3; i++)
    {
      unsigned int* edge_vertices = const_cast<unsigned int*>(topology(1, 0)(cell_edges[i]));
      sort_entities(2, edge_vertices, local_to_global_vertex_indices);
    }
  }

  // Sort local vertices on cell in ascending order, connectivity 2 - 0
  if (!topology(2, 0).empty())
  {
    unsigned int* cell_vertices = const_cast<unsigned int*>(cell.entities(0));
    sort_entities(3, cell_vertices, local_to_global_vertex_indices);
  }

  // Sort local edges on cell after non-incident vertex, connectivity 2 - 1
  if (!topology(2, 1).empty())
  {
    dolfin_assert(!topology(2, 1).empty());

    // Get cell vertex and edge indices (local)
    const unsigned int* cell_vertices = cell.entities(0);
    unsigned int* cell_edges = const_cast<unsigned int*>(cell.entities(1));

    // Loop over vertices on cell
    for (std::size_t i = 0; i < 3; i++)
    {
      // Loop over edges on cell
      for (std::size_t j = i; j < 3; j++)
      {
        const unsigned int* edge_vertices = topology(1, 0)(cell_edges[j]);

        // Check if the ith vertex of the cell is non-incident with edge j
        if (std::count(edge_vertices, edge_vertices + 2, cell_vertices[i]) == 0)
        {
          // Swap edge numbers
          std::size_t tmp = cell_edges[i];
          cell_edges[i] = cell_edges[j];
          cell_edges[j] = tmp;
          break;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
bool TriangleCell::collides(const Cell& cell, const Point& point) const
{
  return CollisionDetection::collides(cell, point);
}
//-----------------------------------------------------------------------------
bool TriangleCell::collides(const Cell& cell, const MeshEntity& entity) const
{
  return CollisionDetection::collides(cell, entity);
}
//-----------------------------------------------------------------------------
std::vector<double>
TriangleCell::triangulate_intersection(const Cell& c0, const Cell& c1) const
{
  return IntersectionTriangulation::triangulate_intersection(c0, c1);
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
  const unsigned int* v = cell.entities(0);
  const unsigned int* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Look for edge satisfying ordering convention
  for (std::size_t j = 0; j < 3; j++)
  {
    const unsigned int* ev = cell.mesh().topology()(1, 0)(e[j]);
    dolfin_assert(ev);
    if (ev[0] != v[i] && ev[1] != v[i])
      return j;
  }

  // We should not reach this
  dolfin_error("TriangleCell.cpp",
               "find specified edge in cell",
               "Edge really not found");
  return 0;
}
//-----------------------------------------------------------------------------
