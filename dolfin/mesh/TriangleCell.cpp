// Copyright (C) 2006-2011 Anders Logg
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
//
// First added:  2006-06-05
// Last changed: 2011-11-14

#include <algorithm>
#include <dolfin/log/dolfin_log.h>
#include "Cell.h"
#include "MeshEditor.h"
#include "Facet.h"
#include "TriangleCell.h"
#include "Vertex.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint TriangleCell::dim() const
{
  return 2;
}
//-----------------------------------------------------------------------------
dolfin::uint TriangleCell::num_entities(uint dim) const
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
dolfin::uint TriangleCell::num_vertices(uint dim) const
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
dolfin::uint TriangleCell::orientation(const Cell& cell) const
{
  const Vertex v0(cell.mesh(), cell.entities(0)[0]);
  const Vertex v1(cell.mesh(), cell.entities(0)[1]);
  const Vertex v2(cell.mesh(), cell.entities(0)[2]);

  const Point p01 = v1.point() - v0.point();
  const Point p02 = v2.point() - v0.point();
  const Point n(-p01.y(), p01.x());

  return (n.dot(p02) < 0.0 ? 1 : 0);
}
//-----------------------------------------------------------------------------
void TriangleCell::create_entities(std::vector<std::vector<uint> >& e,
                                   uint dim, const uint* v) const
{
  // We only need to know how to create edges
  if (dim != 1)
  {
    dolfin_error("TriangleCell.cpp",
                 "create entities of triangle cell",
                 "Don't know how to create entities of topological dimension %d", dim);
  }

  // Create the three edges
  e[0][0] = v[1]; e[0][1] = v[2];
  e[1][0] = v[0]; e[1][1] = v[2];
  e[2][0] = v[0]; e[2][1] = v[1];
}
//-----------------------------------------------------------------------------
void TriangleCell::refine_cell(Cell& cell, MeshEditor& editor,
                               uint& current_cell) const
{
  // Get vertices and edges
  const uint* v = cell.entities(0);
  const uint* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Get offset for new vertex indices
  const uint offset = cell.mesh().num_vertices();

  // Compute indices for the six new vertices
  const uint v0 = v[0];
  const uint v1 = v[1];
  const uint v2 = v[2];
  const uint e0 = offset + e[find_edge(0, cell)];
  const uint e1 = offset + e[find_edge(1, cell)];
  const uint e2 = offset + e[find_edge(2, cell)];

  // Add the four new cells
  editor.add_cell(current_cell++, v0, e2, e1);
  editor.add_cell(current_cell++, v1, e0, e2);
  editor.add_cell(current_cell++, v2, e1, e0);
  editor.add_cell(current_cell++, e0, e1, e2);
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
  const uint* vertices = triangle.entities(0);
  const double* x0 = geometry.x(vertices[0]);
  const double* x1 = geometry.x(vertices[1]);
  const double* x2 = geometry.x(vertices[2]);

  if (geometry.dim() == 2)
  {
    // Compute area of triangle embedded in R^2
    double v2 = (x0[0]*x1[1] + x0[1]*x2[0] + x1[0]*x2[1]) - (x2[0]*x1[1] + x2[1]*x0[0] + x1[0]*x0[1]);

    // Formula for volume from http://mathworld.wolfram.com
    return v2 = 0.5 * std::abs(v2);
  }
  else if (geometry.dim() == 3)
  {
    // Compute area of triangle embedded in R^3
    const double v0 = (x0[1]*x1[2] + x0[2]*x2[1] + x1[1]*x2[2]) - (x2[1]*x1[2] + x2[2]*x0[1] + x1[1]*x0[2]);
    const double v1 = (x0[2]*x1[0] + x0[0]*x2[2] + x1[2]*x2[0]) - (x2[2]*x1[0] + x2[0]*x0[2] + x1[2]*x0[0]);
    const double v2 = (x0[0]*x1[1] + x0[1]*x2[0] + x1[0]*x2[1]) - (x2[0]*x1[1] + x2[1]*x0[0] + x1[0]*x0[1]);

    // Formula for volume from http://mathworld.wolfram.com
    return  0.5*sqrt(v0*v0 + v1*v1 + v2*v2);
  }
  else
    dolfin_error("TriangleCell.cpp",
                 "compute volume of triangle",
                 "Only know how to compute volume when embedded in R^2 or R^3");

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
  const uint* vertices = triangle.entities(0);
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
  return 0.5*a*b*c/volume(triangle);
}
//-----------------------------------------------------------------------------
double TriangleCell::normal(const Cell& cell, uint facet, uint i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
Point TriangleCell::normal(const Cell& cell, uint facet) const
{
  // Make sure we have facets
  cell.mesh().init(2, 1);

  // Create facet from the mesh and local facet number
  Facet f(cell.mesh(), cell.entities(1)[facet]);

  // The normal vector is currently only defined for a triangle in R^2
  if (cell.mesh().geometry().dim() != 2)
    dolfin_error("TriangleCell.cpp",
                 "find normal",
                 "Normal vector is not defined in dimension %d (only defined when the triangle is in R^2", cell.mesh().geometry().dim());

  // Get global index of opposite vertex
  const uint v0 = cell.entities(0)[facet];

  // Get global index of vertices on the facet
  const uint v1 = f.entities(0)[0];
  const uint v2 = f.entities(0)[1];

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
double TriangleCell::facet_area(const Cell& cell, uint facet) const
{
  // Create facet from the mesh and local facet number
  const Facet f(cell.mesh(), cell.entities(1)[facet]);

  // Get global index of vertices on the facet
  const uint v0 = f.entities(0)[0];
  const uint v1 = f.entities(0)[1];

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the two vertices
  const double* p0 = geometry.x(v0);
  const double* p1 = geometry.x(v1);

  // Compute distance between vertices
  double d = 0.0;
  for (uint i = 0; i < geometry.dim(); i++)
  {
    const double dp = p0[i] - p1[i];
    d += dp*dp;
  }

  return std::sqrt(d);
}
//-----------------------------------------------------------------------------
void TriangleCell::order(Cell& cell,
                         const MeshFunction<uint>* global_vertex_indices) const
{
  // Sort i - j for i > j: 1 - 0, 2 - 0, 2 - 1

  // Get mesh topology
  const MeshTopology& topology = cell.mesh().topology();

  // Sort local vertices on edges in ascending order, connectivity 1 - 0
  if (topology(1, 0).size() > 0)
  {
    dolfin_assert(topology(2, 1).size() > 0);

    // Get edges
    const uint* cell_edges = cell.entities(1);

    // Sort vertices on each edge
    for (uint i = 0; i < 3; i++)
    {
      uint* edge_vertices = const_cast<uint*>(topology(1, 0)(cell_edges[i]));
      sort_entities(2, edge_vertices, global_vertex_indices);
    }
  }

  // Sort local vertices on cell in ascending order, connectivity 2 - 0
  if (topology(2, 0).size() > 0)
  {
    uint* cell_vertices = const_cast<uint*>(cell.entities(0));
    sort_entities(3, cell_vertices, global_vertex_indices);
  }

  // Sort local edges on cell after non-incident vertex, connectivity 2 - 1
  if (topology(2, 1).size() > 0)
  {
    dolfin_assert(topology(2, 1).size() > 0);

    // Get cell vertices and edges
    const uint* cell_vertices = cell.entities(0);
    uint* cell_edges = const_cast<uint*>(cell.entities(1));

    // Loop over vertices on cell
    for (uint i = 0; i < 3; i++)
    {
      // Loop over edges on cell
      for (uint j = i; j < 3; j++)
      {
        const uint* edge_vertices = topology(1, 0)(cell_edges[j]);

        // Check if the ith vertex of the cell is non-incident with edge j
        if (std::count(edge_vertices, edge_vertices + 2, cell_vertices[i]) == 0)
        {
          // Swap edge numbers
          uint tmp = cell_edges[i];
          cell_edges[i] = cell_edges[j];
          cell_edges[j] = tmp;
          break;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
std::string TriangleCell::description(bool plural) const
{
  if (plural)
    return "triangles";
  return "triangle";
}
//-----------------------------------------------------------------------------
dolfin::uint TriangleCell::find_edge(uint i, const Cell& cell) const
{
  // Get vertices and edges
  const uint* v = cell.entities(0);
  const uint* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Look for edge satisfying ordering convention
  for (uint j = 0; j < 3; j++)
  {
    const uint* ev = cell.mesh().topology()(1, 0)(e[j]);
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
