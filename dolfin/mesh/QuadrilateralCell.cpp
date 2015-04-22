// Copyright (C) 2015 Chris Richardson
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

#include <algorithm>
#include <dolfin/log/log.h>
#include <Eigen/Dense>
#include "Cell.h"
#include "MeshEditor.h"
#include "MeshEntity.h"
#include "Facet.h"
#include "QuadrilateralCell.h"
#include "Vertex.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::size_t QuadrilateralCell::dim() const
{
  return 2;
}
//-----------------------------------------------------------------------------
std::size_t QuadrilateralCell::num_entities(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 4; // vertices
  case 1:
    return 4; // edges
  case 2:
    return 1; // cells
  default:
    dolfin_error("QuadrilateralCell.cpp",
                 "access number of entities of quadrilateral cell",
                 "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t QuadrilateralCell::num_vertices(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // edges
  case 2:
    return 4; // cells
  default:
    dolfin_error("QuadrilateralCell.cpp",
                 "access number of vertices for subsimplex of quadrilateral cell",
                 "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t QuadrilateralCell::orientation(const Cell& cell) const
{
  const Point up(0.0, 0.0, 1.0);
  return cell.orientation(up);
}
//-----------------------------------------------------------------------------
void QuadrilateralCell::create_entities(boost::multi_array<unsigned int, 2>& e,
                                        std::size_t dim, const unsigned int* v) const
{
  // We only need to know how to create edges
  if (dim != 1)
  {
    dolfin_error("QuadrilateralCell.cpp",
                 "create entities of quadrilateral cell",
                 "Don't know how to create entities of topological dimension %d", dim);
  }

  // Resize data structure
  e.resize(boost::extents[4][2]);

  // Create the four edges
  e[0][0] = v[2]; e[0][1] = v[3];
  e[1][0] = v[1]; e[1][1] = v[2];
  e[2][0] = v[0]; e[2][1] = v[3];
  e[3][0] = v[0]; e[3][1] = v[1];
}
//-----------------------------------------------------------------------------
void QuadrilateralCell::refine_cell(Cell& cell, MeshEditor& editor,
                               std::size_t& current_cell) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
double QuadrilateralCell::volume(const MeshEntity& cell) const
{
  if (cell.dim() != 2)
  {
    dolfin_error("QuadrilateralCell.cpp",
                 "compute volume (area) of cell",
                 "Illegal mesh entity");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the four vertices
  const unsigned int* vertices = cell.entities(0);
  const double* x0 = geometry.x(vertices[0]);
  const double* x1 = geometry.x(vertices[1]);
  const double* x2 = geometry.x(vertices[2]);
  const double* x3 = geometry.x(vertices[3]);


  if (geometry.dim() == 2)
  {
    Eigen::Vector3d a(x0[0] - x2[0], x0[1] - x2[1], 0.0);
    Eigen::Vector3d b(x1[0] - x3[0], x1[1] - x3[1], 0.0);
    Eigen::Vector3d c = a.cross(b);

    return 0.5 * std::abs(c[2]);
  }
  else if (geometry.dim() == 3)
  {
    Eigen::Vector3d a(x0[0] - x2[0], x0[1] - x2[1], x0[2] - x2[2]);
    Eigen::Vector3d b(x0[0] - x3[0], x0[1] - x3[1], x0[2] - x3[2]);
    Eigen::Vector3d c(x0[0] - x1[0], x0[1] - x1[1], x0[2] - x1[2]);
    double d0 = a.cross(b).norm();
    d0 += a.cross(c).norm();

    warning("Calculating area of quadrilateral in 3D by subdividing into triangles");
    return 0.5*d0;
  }
  else
    dolfin_error("QuadrilateralCell.cpp",
                 "compute volume of quadrilateral",
                 "Only know how to compute volume when embedded in R^2 or R^3");

  return 0.0;
}
//-----------------------------------------------------------------------------
double QuadrilateralCell::diameter(const MeshEntity& cell) const
{
  // Check that we get a cell
  if (cell.dim() != 2)
  {
    dolfin_error("QuadrilateralCell.cpp",
                 "compute diameter of quadrilateral cell",
                 "Illegal mesh entity");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Only know how to compute the diameter when embedded in R^2 or R^3
  if (geometry.dim() != 2 && geometry.dim() != 3)
    dolfin_error("QuadrilateralCell.cpp",
                 "compute diameter of quadrilateral",
                 "Only know how to compute diameter when embedded in R^2 or R^3");

  // Get the coordinates of the three vertices
  const unsigned int* vertices = cell.entities(0);
  const Point p0 = geometry.point(vertices[0]);
  const Point p1 = geometry.point(vertices[1]);
  const Point p2 = geometry.point(vertices[2]);
  const Point p3 = geometry.point(vertices[2]);

  dolfin_not_implemented();
  return 0.0;
}
//-----------------------------------------------------------------------------
double QuadrilateralCell::squared_distance(const Cell& cell,
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
double QuadrilateralCell::squared_distance(const Point& point,
                                      const Point& a,
                                      const Point& b,
                                      const Point& c)
{
  dolfin_not_implemented();
  return 0.0;
}
//-----------------------------------------------------------------------------
double QuadrilateralCell::normal(const Cell& cell, std::size_t facet, std::size_t i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
Point QuadrilateralCell::normal(const Cell& cell, std::size_t facet) const
{
  // Make sure we have facets
  cell.mesh().init(2, 1);

  // Create facet from the mesh and local facet number
  Facet f(cell.mesh(), cell.entities(1)[facet]);

  // The normal vector is currently only defined for a triangle in R^2
  // MER: This code is super for a triangle in R^3 too, this error
  // could be removed, unless it is here for some other reason.
  if (cell.mesh().geometry().dim() != 2)
    dolfin_error("QuadrilateralCell.cpp",
                 "find normal",
                 "Normal vector is not defined in dimension %d (only defined when the triangle is in R^2", cell.mesh().geometry().dim());

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
Point QuadrilateralCell::cell_normal(const Cell& cell) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Cell_normal only defined for gdim = 2, 3:
  const std::size_t gdim = geometry.dim();
  if (gdim > 3)
    dolfin_error("QuadrilateralCell.cpp",
                 "compute cell normal",
                 "Illegal geometric dimension (%d)", gdim);

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
double QuadrilateralCell::facet_area(const Cell& cell, std::size_t facet) const
{
  // Create facet from the mesh and local facet number
  const Facet f(cell.mesh(), cell.entities(1)[facet]);

  // Get global index of vertices on the facet
  const std::size_t v0 = f.entities(0)[0];
  const std::size_t v1 = f.entities(0)[1];

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the two vertices
  const double* p0 = geometry.x(v0);
  const double* p1 = geometry.x(v1);

  // Compute distance between vertices
  double d = 0.0;
  for (std::size_t i = 0; i < geometry.dim(); i++)
  {
    const double dp = p0[i] - p1[i];
    d += dp*dp;
  }

  return std::sqrt(d);
}
//-----------------------------------------------------------------------------
void QuadrilateralCell::order(Cell& cell,
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
bool QuadrilateralCell::collides(const Cell& cell, const Point& point) const
{
  return CollisionDetection::collides(cell, point);
}
//-----------------------------------------------------------------------------
bool QuadrilateralCell::collides(const Cell& cell, const MeshEntity& entity) const
{
  return CollisionDetection::collides(cell, entity);
}
//-----------------------------------------------------------------------------
std::vector<double>
QuadrilateralCell::triangulate_intersection(const Cell& c0, const Cell& c1) const
{
  return IntersectionTriangulation::triangulate_intersection(c0, c1);
}
//-----------------------------------------------------------------------------
std::string QuadrilateralCell::description(bool plural) const
{
  if (plural)
    return "quadrilaterals";
  return "quadrilateral";
}
//-----------------------------------------------------------------------------
std::size_t QuadrilateralCell::find_edge(std::size_t i, const Cell& cell) const
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
  dolfin_error("QuadrilateralCell.cpp",
               "find specified edge in cell",
               "Edge really not found");
  return 0;
}
//-----------------------------------------------------------------------------
