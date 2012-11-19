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
// Modified by Johan Hoffman, 2006.
// Modified by Garth N. Wells, 2006.
// Modified by Kristian Oelgaard, 2006.
// Modified by Kristoffer Selim, 2008.
//
// First added:  2006-06-05
// Last changed: 2011-11-21

#include <algorithm>
#include <dolfin/log/dolfin_log.h>
#include "Cell.h"
#include "Facet.h"
#include "MeshEditor.h"
#include "MeshGeometry.h"
#include "Vertex.h"
#include "TetrahedronCell.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint TetrahedronCell::dim() const
{
  return 3;
}
//-----------------------------------------------------------------------------
dolfin::uint TetrahedronCell::num_entities(uint dim) const
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
    dolfin_error("TetrahedronCell.cpp",
                 "access number of entities of tetrahedron cell",
                 "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint TetrahedronCell::num_vertices(uint dim) const
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
    dolfin_error("TetrahedronCell.cpp",
                 "access number of vertices for subsimplex of tetrahedron cell",
                 "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint TetrahedronCell::orientation(const Cell& cell) const
{
  const Vertex v0(cell.mesh(), cell.entities(0)[0]);
  const Vertex v1(cell.mesh(), cell.entities(0)[1]);
  const Vertex v2(cell.mesh(), cell.entities(0)[2]);
  const Vertex v3(cell.mesh(), cell.entities(0)[3]);

  const Point p01 = v1.point() - v0.point();
  const Point p02 = v2.point() - v0.point();
  const Point p03 = v3.point() - v0.point();

  const Point n = p01.cross(p02);

  return (n.dot(p03) < 0.0 ? 1 : 0);
}
//-----------------------------------------------------------------------------
void TetrahedronCell::create_entities(std::vector<std::vector<std::size_t> >& e,
                                      uint dim, const std::size_t* v) const
{
  // We only need to know how to create edges and faces
  switch (dim)
  {
  case 1:
    // Create the six edges
    e[0][0] = v[2]; e[0][1] = v[3];
    e[1][0] = v[1]; e[1][1] = v[3];
    e[2][0] = v[1]; e[2][1] = v[2];
    e[3][0] = v[0]; e[3][1] = v[3];
    e[4][0] = v[0]; e[4][1] = v[2];
    e[5][0] = v[0]; e[5][1] = v[1];
    break;
  case 2:
    // Create the four faces
    e[0][0] = v[1]; e[0][1] = v[2]; e[0][2] = v[3];
    e[1][0] = v[0]; e[1][1] = v[2]; e[1][2] = v[3];
    e[2][0] = v[0]; e[2][1] = v[1]; e[2][2] = v[3];
    e[3][0] = v[0]; e[3][1] = v[1]; e[3][2] = v[2];
    break;
  default:
    dolfin_error("TetrahedronCell.cpp",
                 "create entities of tetrahedron cell",
                 "Don't know how to create entities of topological dimension %d", dim);
  }
}
//-----------------------------------------------------------------------------
void TetrahedronCell::refine_cell(Cell& cell, MeshEditor& editor,
                                  std::size_t& current_cell) const
{
  // Get vertices and edges
  const std::size_t* v = cell.entities(0);
  const std::size_t* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Get offset for new vertex indices
  const std::size_t offset = cell.mesh().num_vertices();

  // Compute indices for the ten new vertices
  const std::size_t v0 = v[0];
  const std::size_t v1 = v[1];
  const std::size_t v2 = v[2];
  const std::size_t v3 = v[3];
  const std::size_t e0 = offset + e[find_edge(0, cell)];
  const std::size_t e1 = offset + e[find_edge(1, cell)];
  const std::size_t e2 = offset + e[find_edge(2, cell)];
  const std::size_t e3 = offset + e[find_edge(3, cell)];
  const std::size_t e4 = offset + e[find_edge(4, cell)];
  const std::size_t e5 = offset + e[find_edge(5, cell)];

  // Regular refinement creates 8 new cells but we need to be careful
  // to make the partition in a way that does not make the aspect
  // ratio worse in each refinement. We do this by cutting the middle
  // octahedron along the shortest of three possible paths.
  dolfin_assert(editor.mesh);
  const Point p0 = editor.mesh->geometry().point(e0);
  const Point p1 = editor.mesh->geometry().point(e1);
  const Point p2 = editor.mesh->geometry().point(e2);
  const Point p3 = editor.mesh->geometry().point(e3);
  const Point p4 = editor.mesh->geometry().point(e4);
  const Point p5 = editor.mesh->geometry().point(e5);
  const double d05 = p0.distance(p5);
  const double d14 = p1.distance(p4);
  const double d23 = p2.distance(p3);

  // Data structure to hold cells
  std::vector<std::vector<std::size_t> > cells(8, std::vector<std::size_t>(4));

  // First create the 4 congruent tetrahedra at the corners
  cells[0][0] = v0; cells[0][1] = e3; cells[0][2] = e4; cells[0][3] = e5;
  cells[1][0] = v1; cells[1][1] = e1; cells[1][2] = e2; cells[1][3] = e5;
  cells[2][0] = v2; cells[2][1] = e0; cells[2][2] = e2; cells[2][3] = e4;
  cells[3][0] = v3; cells[3][1] = e0; cells[3][2] = e1; cells[3][3] = e3;

  // Then divide the remaining octahedron into 4 tetrahedra
  if (d05 <= d14 && d14 <= d23)
  {
    cells[4][0] = e0; cells[4][1] = e1; cells[4][2] = e2; cells[4][3] = e5;
    cells[5][0] = e0; cells[5][1] = e1; cells[5][2] = e3; cells[5][3] = e5;
    cells[6][0] = e0; cells[6][1] = e2; cells[6][2] = e4; cells[6][3] = e5;
    cells[7][0] = e0; cells[7][1] = e3; cells[7][2] = e4; cells[7][3] = e5;
  }
  else if (d14 <= d23)
  {
    cells[4][0] = e0; cells[4][1] = e1; cells[4][2] = e2; cells[4][3] = e4;
    cells[5][0] = e0; cells[5][1] = e1; cells[5][2] = e3; cells[5][3] = e4;
    cells[6][0] = e1; cells[6][1] = e2; cells[6][2] = e4; cells[6][3] = e5;
    cells[7][0] = e1; cells[7][1] = e3; cells[7][2] = e4; cells[7][3] = e5;
  }
  else
  {
    cells[4][0] = e0; cells[4][1] = e1; cells[4][2] = e2; cells[4][3] = e3;
    cells[5][0] = e0; cells[5][1] = e2; cells[5][2] = e3; cells[5][3] = e4;
    cells[6][0] = e1; cells[6][1] = e2; cells[6][2] = e3; cells[6][3] = e5;
    cells[7][0] = e2; cells[7][1] = e3; cells[7][2] = e4; cells[7][3] = e5;
  }

  // Add cells
  std::vector<std::vector<std::size_t> >::const_iterator _cell;
  for (_cell = cells.begin(); _cell != cells.end(); ++_cell)
    editor.add_cell(current_cell++, *_cell);
}
//-----------------------------------------------------------------------------
void TetrahedronCell::refine_cellIrregular(Cell& cell, MeshEditor& editor,
  std::size_t& current_cell, uint refinement_rule, std::size_t* marked_edges) const
{
  dolfin_not_implemented();

  /*
  // Get vertices and edges
  const std::size_t* v = cell.entities(0);
  const std::size_t* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Get offset for new vertex indices
  const std::size_t offset = cell.mesh().num_vertices();

  // Compute indices for the ten new vertices
  const std::size_t v0 = v[0];
  const std::size_t v1 = v[1];
  const std::size_t v2 = v[2];
  const std::size_t v3 = v[3];
  const std::size_t e0 = offset + e[0];
  const std::size_t e1 = offset + e[1];
  const std::size_t e2 = offset + e[2];
  const std::size_t e3 = offset + e[3];
  const std::size_t e4 = offset + e[4];
  const std::size_t e5 = offset + e[5];

  // Refine according to refinement rule
  // The rules are numbered according to the paper:
  // J. Bey, "Tetrahedral Grid Refinement", 1995.
  switch (refinement_rule)
  {
  case 1:
    // Rule 1: 4 new cells
    editor.add_cell(current_cell++, v0, e1, e3, e2);
    editor.add_cell(current_cell++, v1, e2, e4, e0);
    editor.add_cell(current_cell++, v2, e0, e5, e1);
    editor.add_cell(current_cell++, v3, e5, e4, e3);
    break;
  case 2:
    // Rule 2: 2 new cells
    editor.add_cell(current_cell++, v0, e1, e3, e2);
    editor.add_cell(current_cell++, v1, e2, e4, e0);
    break;
  case 3:
    // Rule 3: 3 new cells
    editor.add_cell(current_cell++, v0, e1, e3, e2);
    editor.add_cell(current_cell++, v1, e2, e4, e0);
    editor.add_cell(current_cell++, v2, e0, e5, e1);
    break;
  case 4:
    // Rule 4: 4 new cells
    editor.add_cell(current_cell++, v0, e1, e3, e2);
    editor.add_cell(current_cell++, v1, e2, e4, e0);
    editor.add_cell(current_cell++, v2, e0, e5, e1);
    editor.add_cell(current_cell++, v3, e5, e4, e3);
    break;
  default:
    dolfin_error("TetrahedronCell.cpp",
                 "perform regular cut refinement of tetrahedron",
                 "Illegal rule (%d) for irregular refinement of tetrahedron",
                 refinement_rule);
  }
  */
}
//-----------------------------------------------------------------------------
double TetrahedronCell::volume(const MeshEntity& tetrahedron) const
{
  // Check that we get a tetrahedron
  if (tetrahedron.dim() != 3)
  {
    dolfin_error("TetrahedronCell.cpp",
                 "compute volume of tetrahedron cell",
                 "Illegal mesh entity, not a tetrahedron");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = tetrahedron.mesh().geometry();

  // Only know how to compute the volume when embedded in R^3
  if (geometry.dim() != 3)
  {
    dolfin_error("TetrahedronCell.cpp",
                 "compute volume of tetrahedron",
                 "Only know how to compute volume when embedded in R^3");
  }

  // Get the coordinates of the four vertices
  const std::size_t* vertices = tetrahedron.entities(0);
  const double* x0 = geometry.x(vertices[0]);
  const double* x1 = geometry.x(vertices[1]);
  const double* x2 = geometry.x(vertices[2]);
  const double* x3 = geometry.x(vertices[3]);

  // Formula for volume from http://mathworld.wolfram.com
  const double v = (x0[0]*(x1[1]*x2[2] + x3[1]*x1[2] + x2[1]*x3[2] - x2[1]*x1[2] - x1[1]*x3[2] - x3[1]*x2[2]) -
                    x1[0]*(x0[1]*x2[2] + x3[1]*x0[2] + x2[1]*x3[2] - x2[1]*x0[2] - x0[1]*x3[2] - x3[1]*x2[2]) +
                    x2[0]*(x0[1]*x1[2] + x3[1]*x0[2] + x1[1]*x3[2] - x1[1]*x0[2] - x0[1]*x3[2] - x3[1]*x1[2]) -
                    x3[0]*(x0[1]*x1[2] + x1[1]*x2[2] + x2[1]*x0[2] - x1[1]*x0[2] - x2[1]*x1[2] - x0[1]*x2[2]));

  return std::abs(v)/6.0;
}
//-----------------------------------------------------------------------------
double TetrahedronCell::diameter(const MeshEntity& tetrahedron) const
{
  // Check that we get a tetrahedron
  if (tetrahedron.dim() != 3)
  {
    dolfin_error("TetrahedronCell.cpp",
                 "compute diameter of tetrahedron cell",
                 "Illegal mesh entity, not a tetrahedron");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = tetrahedron.mesh().geometry();

  // Only know how to compute the volume when embedded in R^3
  if (geometry.dim() != 3)
  {
    dolfin_error("TetrahedronCell.cpp",
                 "compute diameter",
                 "Tetrahedron is not embedded in R^3, only know how to compute diameter in that case");
  }

  // Get the coordinates of the four vertices
  const std::size_t* vertices = tetrahedron.entities(0);
  const Point p0 = geometry.point(vertices[0]);
  const Point p1 = geometry.point(vertices[1]);
  const Point p2 = geometry.point(vertices[2]);
  const Point p3 = geometry.point(vertices[3]);

  // Compute side lengths
  const double a  = p1.distance(p2);
  const double b  = p0.distance(p2);
  const double c  = p0.distance(p1);
  const double aa = p0.distance(p3);
  const double bb = p1.distance(p3);
  const double cc = p2.distance(p3);

  // Compute "area" of triangle with strange side lengths
  const double la   = a*aa;
  const double lb   = b*bb;
  const double lc   = c*cc;
  const double s    = 0.5*(la+lb+lc);
  const double area = sqrt(s*(s-la)*(s-lb)*(s-lc));

  // Formula for diameter (2*circumradius) from http://mathworld.wolfram.com
  return area/(3.0*volume(tetrahedron));
}
//-----------------------------------------------------------------------------
double TetrahedronCell::normal(const Cell& cell, uint facet, uint i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
Point TetrahedronCell::normal(const Cell& cell, uint facet) const
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
  const double* p0 = geometry.x(v0);
  const double* p1 = geometry.x(v1);
  const double* p2 = geometry.x(v2);
  const double* p3 = geometry.x(v3);

  // Create points from vertex coordinates
  Point P0(p0[0], p0[1], p0[2]);
  Point P1(p1[0], p1[1], p1[2]);
  Point P2(p2[0], p2[1], p2[2]);
  Point P3(p3[0], p3[1], p3[2]);

  // Create vectors
  Point V0 = P0 - P1;
  Point V1 = P2 - P1;
  Point V2 = P3 - P1;

  // Compute normal vector
  Point n = V1.cross(V2);

  // Normalize
  n /= n.norm();

  // Flip direction of normal so it points outward
  if (n.dot(V0) > 0)
    n *= -1.0;

  return n;
}
//-----------------------------------------------------------------------------
double TetrahedronCell::facet_area(const Cell& cell, dolfin::uint facet) const
{
  dolfin_assert(cell.mesh().topology().dim() == 3);
  dolfin_assert(cell.mesh().geometry().dim() == 3);

  // Create facet from the mesh and local facet number
  Facet f(cell.mesh(), cell.entities(2)[facet]);

  // Get mesh geometry
  const MeshGeometry& geometry = f.mesh().geometry();

  // Get the coordinates of the three vertices
  const std::size_t* vertices = f.entities(0);
  const double* x0 = geometry.x(vertices[0]);
  const double* x1 = geometry.x(vertices[1]);
  const double* x2 = geometry.x(vertices[2]);

  // Compute area of triangle embedded in R^3
  double v0 = (x0[1]*x1[2] + x0[2]*x2[1] + x1[1]*x2[2]) - (x2[1]*x1[2] + x2[2]*x0[1] + x1[1]*x0[2]);
  double v1 = (x0[2]*x1[0] + x0[0]*x2[2] + x1[2]*x2[0]) - (x2[2]*x1[0] + x2[0]*x0[2] + x1[2]*x0[0]);
  double v2 = (x0[0]*x1[1] + x0[1]*x2[0] + x1[0]*x2[1]) - (x2[0]*x1[1] + x2[1]*x0[0] + x1[0]*x0[1]);

  // Formula for area from http://mathworld.wolfram.com
  return  0.5*sqrt(v0*v0 + v1*v1 + v2*v2);
}
//-----------------------------------------------------------------------------
void TetrahedronCell::order(Cell& cell,
                 const std::vector<std::size_t>& local_to_global_vertex_indices) const
{
  // Sort i - j for i > j: 1 - 0, 2 - 0, 2 - 1, 3 - 0, 3 - 1, 3 - 2

  // Get mesh topology
  const MeshTopology& topology = cell.mesh().topology();

  // Sort local vertices on edges in ascending order, connectivity 1 - 0
  if (!topology(1, 0).empty())
  {
    dolfin_assert(!topology(3, 1).empty());

    // Get edges
    const std::size_t* cell_edges = cell.entities(1);

    // Sort vertices on each edge
    for (uint i = 0; i < 6; i++)
    {
      std::size_t* edge_vertices = const_cast<std::size_t*>(topology(1, 0)(cell_edges[i]));
      sort_entities(2, edge_vertices, local_to_global_vertex_indices);
    }
  }

  // Sort local vertices on facets in ascending order, connectivity 2 - 0
  if (!topology(2, 0).empty())
  {
    dolfin_assert(!topology(3, 2).empty());

    // Get facets
    const std::size_t* cell_facets = cell.entities(2);

    // Sort vertices on each facet
    for (uint i = 0; i < 4; i++)
    {
      std::size_t* facet_vertices = const_cast<std::size_t*>(topology(2, 0)(cell_facets[i]));
      sort_entities(3, facet_vertices, local_to_global_vertex_indices);
    }
  }

  // Sort local edges on local facets after non-incident vertex, connectivity 2 - 1
  if (!topology(2, 1).empty())
  {
    dolfin_assert(!topology(3, 2).empty());
    dolfin_assert(!topology(2, 0).empty());
    dolfin_assert(!topology(1, 0).empty());

    // Get facet numbers
    const std::size_t* cell_facets = cell.entities(2);

    // Loop over facets on cell
    for (uint i = 0; i < 4; i++)
    {
      // For each facet number get the global vertex numbers
      const std::size_t* facet_vertices = topology(2, 0)(cell_facets[i]);

      // For each facet number get the global edge number
      std::size_t* cell_edges = const_cast<std::size_t*>(topology(2, 1)(cell_facets[i]));

      // Loop over vertices on facet
      uint m = 0;
      for (uint j = 0; j < 3; j++)
      {
        // Loop edges on facet
        for (uint k(m); k < 3; k++)
        {
          // For each edge number get the global vertex numbers
          const std::size_t* edge_vertices = topology(1, 0)(cell_edges[k]);

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
  if (!topology(3, 0).empty())
  {
    std::size_t* cell_vertices = const_cast<std::size_t*>(cell.entities(0));
    sort_entities(4, cell_vertices, local_to_global_vertex_indices);
  }

  // Sort local edges on cell after non-incident vertex tuble, connectivity 3-1
  if (!topology(3, 1).empty())
  {
    dolfin_assert(!topology(1, 0).empty());

    // Get cell vertices and edge numbers
    const std::size_t* cell_vertices = cell.entities(0);
    std::size_t* cell_edges = const_cast<std::size_t*>(cell.entities(1));

    // Loop two vertices on cell as a lexicographical tuple
    // (i, j): (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    uint m = 0;
    for (uint i = 0; i < 3; i++)
    {
      for (uint j = i+1; j < 4; j++)
      {
        // Loop edge numbers
        for (uint k = m; k < 6; k++)
        {
          // Get local vertices on edge
          const std::size_t* edge_vertices = topology(1, 0)(cell_edges[k]);

          // Check if the ith and jth vertex of the cell are non-incident on edge k
          if (!std::count(edge_vertices, edge_vertices+2, cell_vertices[i]) && \
              !std::count(edge_vertices, edge_vertices+2, cell_vertices[j]) )
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

  // Sort local facets on cell after non-incident vertex, connectivity 3 - 2
  if (!topology(3, 2).empty())
  {
    dolfin_assert(!topology(2, 0).empty());

    // Get cell vertices and facet numbers
    const std::size_t* cell_vertices = cell.entities(0);
    std::size_t* cell_facets   = const_cast<std::size_t*>(cell.entities(2));

    // Loop vertices on cell
    for (uint i = 0; i < 4; i++)
    {
      // Loop facets on cell
      for (uint j = i; j < 4; j++)
      {
        std::size_t* facet_vertices = const_cast<std::size_t*>(topology(2, 0)(cell_facets[j]));

        // Check if the ith vertex of the cell is non-incident on facet j
        if (!std::count(facet_vertices, facet_vertices+3, cell_vertices[i]))
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
std::size_t TetrahedronCell::find_edge(uint i, const Cell& cell) const
{
  // Get vertices and edges
  const std::size_t* v = cell.entities(0);
  const std::size_t* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Ordering convention for edges (order of non-incident vertices)
  static uint EV[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

  // Look for edge satisfying ordering convention
  for (uint j = 0; j < 6; j++)
  {
    const std::size_t* ev = cell.mesh().topology()(1, 0)(e[j]);
    dolfin_assert(ev);
    const std::size_t v0 = v[EV[i][0]];
    const std::size_t v1 = v[EV[i][1]];
    if (ev[0] != v0 && ev[0] != v1 && ev[1] != v0 && ev[1] != v1)
      return j;
  }

  // We should not reach this
  dolfin_error("TetrahedronCell.cpp",
               "find specified edge in cell",
               "Edge really not found");
  return 0;
}
//-----------------------------------------------------------------------------
