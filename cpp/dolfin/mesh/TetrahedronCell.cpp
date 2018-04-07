// Copyright (C) 2006-2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TetrahedronCell.h"
#include "Cell.h"
#include "Facet.h"
#include "MeshGeometry.h"
#include "TriangleCell.h"
#include "Vertex.h"
#include <algorithm>
#include <boost/multi_array.hpp>
#include <cmath>
#include <dolfin/log/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
std::size_t TetrahedronCell::dim() const { return 3; }
//-----------------------------------------------------------------------------
std::size_t TetrahedronCell::num_entities(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 4; // vertices
  case 1:
    return 6; // edges
  case 2:
    return 4; // faces
  case 3:
    return 1; // cells
  default:
    log::dolfin_error("TetrahedronCell.cpp",
                      "access number of entities of tetrahedron cell",
                      "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t TetrahedronCell::num_vertices(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // edges
  case 2:
    return 3; // faces
  case 3:
    return 4; // cells
  default:
    log::dolfin_error(
        "TetrahedronCell.cpp",
        "access number of vertices for subsimplex of tetrahedron cell",
        "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
void TetrahedronCell::create_entities(boost::multi_array<std::int32_t, 2>& e,
                                      std::size_t dim,
                                      const std::int32_t* v) const
{
  // We only need to know how to create edges and faces
  switch (dim)
  {
  case 1:
    // Resize data structure
    e.resize(boost::extents[6][2]);

    // Create the six edges
    e[0][0] = v[2];
    e[0][1] = v[3];
    e[1][0] = v[1];
    e[1][1] = v[3];
    e[2][0] = v[1];
    e[2][1] = v[2];
    e[3][0] = v[0];
    e[3][1] = v[3];
    e[4][0] = v[0];
    e[4][1] = v[2];
    e[5][0] = v[0];
    e[5][1] = v[1];
    break;
  case 2:
    // Resize data structure
    e.resize(boost::extents[4][3]);

    // Create the four faces
    e[0][0] = v[1];
    e[0][1] = v[2];
    e[0][2] = v[3];
    e[1][0] = v[0];
    e[1][1] = v[2];
    e[1][2] = v[3];
    e[2][0] = v[0];
    e[2][1] = v[1];
    e[2][2] = v[3];
    e[3][0] = v[0];
    e[3][1] = v[1];
    e[3][2] = v[2];
    break;
  default:
    log::dolfin_error(
        "TetrahedronCell.cpp", "create entities of tetrahedron cell",
        "Don't know how to create entities of topological dimension %d", dim);
  }
}
//-----------------------------------------------------------------------------
double TetrahedronCell::volume(const MeshEntity& tetrahedron) const
{
  // Check that we get a tetrahedron
  if (tetrahedron.dim() != 3)
  {
    log::dolfin_error("TetrahedronCell.cpp",
                      "compute volume of tetrahedron cell",
                      "Illegal mesh entity, not a tetrahedron");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = tetrahedron.mesh().geometry();

  // Only know how to compute the volume when embedded in R^3
  if (geometry.dim() != 3)
  {
    log::dolfin_error("TetrahedronCell.cpp", "compute volume of tetrahedron",
                      "Only know how to compute volume when embedded in R^3");
  }

  // Get the coordinates of the four vertices
  const std::int32_t* vertices = tetrahedron.entities(0);
  const geometry::Point x0 = geometry.point(vertices[0]);
  const geometry::Point x1 = geometry.point(vertices[1]);
  const geometry::Point x2 = geometry.point(vertices[2]);
  const geometry::Point x3 = geometry.point(vertices[3]);

  // Formula for volume from http://mathworld.wolfram.com
  const double v
      = (x0[0] * (x1[1] * x2[2] + x3[1] * x1[2] + x2[1] * x3[2] - x2[1] * x1[2]
                  - x1[1] * x3[2] - x3[1] * x2[2])
         - x1[0] * (x0[1] * x2[2] + x3[1] * x0[2] + x2[1] * x3[2]
                    - x2[1] * x0[2] - x0[1] * x3[2] - x3[1] * x2[2])
         + x2[0] * (x0[1] * x1[2] + x3[1] * x0[2] + x1[1] * x3[2]
                    - x1[1] * x0[2] - x0[1] * x3[2] - x3[1] * x1[2])
         - x3[0] * (x0[1] * x1[2] + x1[1] * x2[2] + x2[1] * x0[2]
                    - x1[1] * x0[2] - x2[1] * x1[2] - x0[1] * x2[2]));

  return std::abs(v) / 6.0;
}
//-----------------------------------------------------------------------------
double TetrahedronCell::circumradius(const MeshEntity& tetrahedron) const
{
  // Check that we get a tetrahedron
  if (tetrahedron.dim() != 3)
  {
    log::dolfin_error("TetrahedronCell.cpp",
                      "compute diameter of tetrahedron cell",
                      "Illegal mesh entity, not a tetrahedron");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = tetrahedron.mesh().geometry();

  // Only know how to compute the volume when embedded in R^3
  if (geometry.dim() != 3)
  {
    log::dolfin_error(
        "TetrahedronCell.cpp", "compute diameter",
        "Tetrahedron is not embedded in R^3, only know how to compute "
        "diameter in that case");
  }

  // Get the coordinates of the four vertices
  const std::int32_t* vertices = tetrahedron.entities(0);
  const geometry::Point p0 = geometry.point(vertices[0]);
  const geometry::Point p1 = geometry.point(vertices[1]);
  const geometry::Point p2 = geometry.point(vertices[2]);
  const geometry::Point p3 = geometry.point(vertices[3]);

  // Compute side lengths
  const double a = p1.distance(p2);
  const double b = p0.distance(p2);
  const double c = p0.distance(p1);
  const double aa = p0.distance(p3);
  const double bb = p1.distance(p3);
  const double cc = p2.distance(p3);

  // Compute "area" of triangle with strange side lengths
  const double la = a * aa;
  const double lb = b * bb;
  const double lc = c * cc;
  const double s = 0.5 * (la + lb + lc);
  const double area = sqrt(s * (s - la) * (s - lb) * (s - lc));

  // Formula for circumradius from
  // http://mathworld.wolfram.com/Tetrahedron.html
  return area / (6.0 * volume(tetrahedron));
}
//-----------------------------------------------------------------------------
double TetrahedronCell::squared_distance(const Cell& cell,
                                         const geometry::Point& point) const
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // ClosestPtPointTetrahedron on page 143, Section 5.1.6.
  //
  // Note: This algorithm actually computes the closest point but we
  // only return the distance to that point.

  // Get the vertices as points
  const MeshGeometry& geometry = cell.mesh().geometry();
  const std::int32_t* vertices = cell.entities(0);
  const geometry::Point a = geometry.point(vertices[0]);
  const geometry::Point b = geometry.point(vertices[1]);
  const geometry::Point c = geometry.point(vertices[2]);
  const geometry::Point d = geometry.point(vertices[3]);

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
double TetrahedronCell::normal(const Cell& cell, std::size_t facet,
                               std::size_t i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
geometry::Point TetrahedronCell::normal(const Cell& cell,
                                        std::size_t facet) const
{
  // Make sure we have facets
  cell.mesh().init(3, 2);

  // Create facet from the mesh and local facet number
  Facet f(cell.mesh(), cell.entities(2)[facet]);

  // Get global index of opposite vertex
  const std::size_t v0 = cell.entities(0)[facet];

  // Get global index of vertices on the facet
  std::size_t v1 = f.entities(0)[0];
  std::size_t v2 = f.entities(0)[1];
  std::size_t v3 = f.entities(0)[2];

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the four vertices
  const geometry::Point P0 = geometry.point(v0);
  const geometry::Point P1 = geometry.point(v1);
  const geometry::Point P2 = geometry.point(v2);
  const geometry::Point P3 = geometry.point(v3);

  // Create vectors
  geometry::Point V0 = P0 - P1;
  geometry::Point V1 = P2 - P1;
  geometry::Point V2 = P3 - P1;

  // Compute normal vector
  geometry::Point n = V1.cross(V2);

  // Normalize
  n /= n.norm();

  // Flip direction of normal so it points outward
  if (n.dot(V0) > 0)
    n *= -1.0;

  return n;
}
//-----------------------------------------------------------------------------
geometry::Point TetrahedronCell::cell_normal(const Cell& cell) const
{
  log::dolfin_error("TetrahedronCell.cpp", "compute cell normal",
                    "cell_normal not implemented for TetrahedronCell");

  return geometry::Point();
}
//-----------------------------------------------------------------------------
double TetrahedronCell::facet_area(const Cell& cell, std::size_t facet) const
{
  assert(cell.mesh().topology().dim() == 3);
  assert(cell.mesh().geometry().dim() == 3);

  // Create facet from the mesh and local facet number
  Facet f(cell.mesh(), cell.entities(2)[facet]);

  // Get mesh geometry
  const MeshGeometry& geometry = f.mesh().geometry();

  // Get the coordinates of the three vertices
  const std::int32_t* vertices = f.entities(0);
  const geometry::Point x0 = geometry.point(vertices[0]);
  const geometry::Point x1 = geometry.point(vertices[1]);
  const geometry::Point x2 = geometry.point(vertices[2]);

  // Compute area of triangle embedded in R^3
  double v0 = (x0[1] * x1[2] + x0[2] * x2[1] + x1[1] * x2[2])
              - (x2[1] * x1[2] + x2[2] * x0[1] + x1[1] * x0[2]);
  double v1 = (x0[2] * x1[0] + x0[0] * x2[2] + x1[2] * x2[0])
              - (x2[2] * x1[0] + x2[0] * x0[2] + x1[2] * x0[0]);
  double v2 = (x0[0] * x1[1] + x0[1] * x2[0] + x1[0] * x2[1])
              - (x2[0] * x1[1] + x2[1] * x0[0] + x1[0] * x0[1]);

  // Formula for area from http://mathworld.wolfram.com
  return 0.5 * sqrt(v0 * v0 + v1 * v1 + v2 * v2);
}
//-----------------------------------------------------------------------------
void TetrahedronCell::order(
    Cell& cell,
    const std::vector<std::int64_t>& local_to_global_vertex_indices) const
{
  // Sort i - j for i > j: 1 - 0, 2 - 0, 2 - 1, 3 - 0, 3 - 1, 3 - 2

  // Get mesh topology
  const MeshTopology& topology = cell.mesh().topology();

  // Sort local vertices on edges in ascending order, connectivity 1 - 0
  if (!topology.connectivity(1, 0).empty())
  {
    assert(!topology.connectivity(3, 1).empty());

    // Get edges
    const std::int32_t* cell_edges = cell.entities(1);

    // Sort vertices on each edge
    for (std::size_t i = 0; i < 6; i++)
    {
      std::int32_t* edge_vertices = const_cast<std::int32_t*>(
          topology.connectivity(1, 0)(cell_edges[i]));
      sort_entities(2, edge_vertices, local_to_global_vertex_indices);
    }
  }

  // Sort local vertices on facets in ascending order, connectivity 2 - 0
  if (!topology.connectivity(2, 0).empty())
  {
    assert(!topology.connectivity(3, 2).empty());

    // Get facets
    const std::int32_t* cell_facets = cell.entities(2);

    // Sort vertices on each facet
    for (std::size_t i = 0; i < 4; i++)
    {
      std::int32_t* facet_vertices = const_cast<std::int32_t*>(
          topology.connectivity(2, 0)(cell_facets[i]));
      sort_entities(3, facet_vertices, local_to_global_vertex_indices);
    }
  }

  // Sort local edges on local facets after non-incident vertex,
  // connectivity 2 - 1
  if (!topology.connectivity(2, 1).empty())
  {
    assert(!topology.connectivity(3, 2).empty());
    assert(!topology.connectivity(2, 0).empty());
    assert(!topology.connectivity(1, 0).empty());

    // Get facet numbers
    const std::int32_t* cell_facets = cell.entities(2);

    // Loop over facets on cell
    for (std::size_t i = 0; i < 4; i++)
    {
      // For each facet number get the global vertex numbers
      const std::int32_t* facet_vertices
          = topology.connectivity(2, 0)(cell_facets[i]);

      // For each facet number get the global edge number
      std::int32_t* cell_edges = const_cast<std::int32_t*>(
          topology.connectivity(2, 1)(cell_facets[i]));

      // Loop over vertices on facet
      std::size_t m = 0;
      for (std::size_t j = 0; j < 3; j++)
      {
        // Loop edges on facet
        for (std::size_t k(m); k < 3; k++)
        {
          // For each edge number get the global vertex numbers
          const std::int32_t* edge_vertices
              = topology.connectivity(1, 0)(cell_edges[k]);

          // Check if the jth vertex of facet i is non-incident on edge k
          if (!std::count(edge_vertices, edge_vertices + 2, facet_vertices[j]))
          {
            // Swap facet numbers
            std::size_t tmp = cell_edges[m];
            cell_edges[m] = cell_edges[k];
            cell_edges[k] = tmp;
            m++;
            break;
          }
        }
      }
    }
  }

  // Sort local vertices on cell in ascending order, connectivity 3 - 0
  if (!topology.connectivity(3, 0).empty())
  {
    std::int32_t* cell_vertices = const_cast<std::int32_t*>(cell.entities(0));
    sort_entities(4, cell_vertices, local_to_global_vertex_indices);
  }

  // Sort local edges on cell after non-incident vertex tuble, connectivity 3-1
  if (!topology.connectivity(3, 1).empty())
  {
    assert(!topology.connectivity(1, 0).empty());

    // Get cell vertices and edge numbers
    const std::int32_t* cell_vertices = cell.entities(0);
    std::int32_t* cell_edges = const_cast<std::int32_t*>(cell.entities(1));

    // Loop two vertices on cell as a lexicographical tuple
    // (i, j): (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    std::size_t m = 0;
    for (std::size_t i = 0; i < 3; i++)
    {
      for (std::size_t j = i + 1; j < 4; j++)
      {
        // Loop edge numbers
        for (std::size_t k = m; k < 6; k++)
        {
          // Get local vertices on edge
          const std::int32_t* edge_vertices
              = topology.connectivity(1, 0)(cell_edges[k]);

          // Check if the ith and jth vertex of the cell are
          // non-incident on edge k
          if (!std::count(edge_vertices, edge_vertices + 2, cell_vertices[i])
              && !std::count(edge_vertices, edge_vertices + 2,
                             cell_vertices[j]))
          {
            // Swap edge numbers
            std::size_t tmp = cell_edges[m];
            cell_edges[m] = cell_edges[k];
            cell_edges[k] = tmp;
            m++;
            break;
          }
        }
      }
    }
  }

  // Sort local facets on cell after non-incident vertex, connectivity
  // 3 - 2
  if (!topology.connectivity(3, 2).empty())
  {
    assert(!topology.connectivity(2, 0).empty());

    // Get cell vertices and facet numbers
    const std::int32_t* cell_vertices = cell.entities(0);
    std::int32_t* cell_facets = const_cast<std::int32_t*>(cell.entities(2));

    // Loop vertices on cell
    for (std::size_t i = 0; i < 4; i++)
    {
      // Loop facets on cell
      for (std::size_t j = i; j < 4; j++)
      {
        std::int32_t* facet_vertices = const_cast<std::int32_t*>(
            topology.connectivity(2, 0)(cell_facets[j]));

        // Check if the ith vertex of the cell is non-incident on facet j
        if (!std::count(facet_vertices, facet_vertices + 3, cell_vertices[i]))
        {
          // Swap facet numbers
          std::size_t tmp = cell_facets[i];
          cell_facets[i] = cell_facets[j];
          cell_facets[j] = tmp;
          break;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
std::string TetrahedronCell::description(bool plural) const
{
  if (plural)
    return "tetrahedra";
  return "tetrahedron";
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
  for (std::size_t j = 0; j < 6; j++)
  {
    const std::int32_t* ev = cell.mesh().topology().connectivity(1, 0)(e[j]);
    assert(ev);
    const std::int32_t v0 = v[EV[i][0]];
    const std::int32_t v1 = v[EV[i][1]];
    if (ev[0] != v0 && ev[0] != v1 && ev[1] != v0 && ev[1] != v1)
      return j;
  }

  // We should not reach this
  log::dolfin_error("TetrahedronCell.cpp", "find specified edge in cell",
                    "Edge really not found");
  return 0;
}
//-----------------------------------------------------------------------------
bool TetrahedronCell::point_outside_of_plane(const geometry::Point& point,
                                             const geometry::Point& a,
                                             const geometry::Point& b,
                                             const geometry::Point& c,
                                             const geometry::Point& d) const
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // PointOutsideOfPlane on page 144, Section 5.1.6.

  const geometry::Point v = (b - a).cross(c - a);
  const double signp = v.dot(point - a);
  const double signd = v.dot(d - a);

  return signp * signd < 0.0;
}
//-----------------------------------------------------------------------------
