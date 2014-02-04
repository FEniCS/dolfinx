// Copyright (C) 2006-2013 Anders Logg
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
// Modified by Johan Hoffman 2006
// Modified by Garth N. Wells 2006
// Modified by Kristian Oelgaard 2006
// Modified by Kristoffer Selim 2008
//
// First added:  2006-06-05
// Last changed: 2014-01-31

#include <algorithm>
#include <dolfin/log/log.h>
#include "Cell.h"
#include "Facet.h"
#include "MeshEditor.h"
#include "MeshGeometry.h"
#include "Vertex.h"
#include "TriangleCell.h"
#include "TetrahedronCell.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::size_t TetrahedronCell::dim() const
{
  return 3;
}
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
    dolfin_error("TetrahedronCell.cpp",
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
    dolfin_error("TetrahedronCell.cpp",
		 "access number of vertices for subsimplex of tetrahedron cell",
		 "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t TetrahedronCell::orientation(const Cell& cell) const
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
void
TetrahedronCell::create_entities(std::vector<std::vector<unsigned int> >& e,
                                 std::size_t dim, const unsigned int* v) const
{
  // We only need to know how to create edges and faces
  switch (dim)
  {
  case 1:
    // Resize data structure
    e.resize(6);
    for (int i = 0; i < 6; ++i)
      e[i] .resize(2);
    
    // Create the six edges
    e[0][0] = v[2]; e[0][1] = v[3];
    e[1][0] = v[1]; e[1][1] = v[3];
    e[2][0] = v[1]; e[2][1] = v[2];
    e[3][0] = v[0]; e[3][1] = v[3];
    e[4][0] = v[0]; e[4][1] = v[2];
    e[5][0] = v[0]; e[5][1] = v[1];
    break;
  case 2:
    // Resize data structure
    e.resize(4);
    for (int i = 0; i < 4; ++i)
      e[i] .resize(3);

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
  const unsigned int* v = cell.entities(0);
  const unsigned int* e = cell.entities(1);
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
  dolfin_assert(editor._mesh);
  const Point p0 = editor._mesh->geometry().point(e0);
  const Point p1 = editor._mesh->geometry().point(e1);
  const Point p2 = editor._mesh->geometry().point(e2);
  const Point p3 = editor._mesh->geometry().point(e3);
  const Point p4 = editor._mesh->geometry().point(e4);
  const Point p5 = editor._mesh->geometry().point(e5);
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
					   std::size_t& current_cell, std::size_t refinement_rule, std::size_t* marked_edges) const
{
  dolfin_not_implemented();

  /*
  // Get vertices and edges
  const unsigned int* v = cell.entities(0);
  const unsigned int* e = cell.entities(1);
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
  const unsigned int* vertices = tetrahedron.entities(0);
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
  const unsigned int* vertices = tetrahedron.entities(0);
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
double TetrahedronCell::squared_distance(const Cell& cell, const Point& point) const
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // ClosestPtPointTetrahedron on page 143, Section 5.1.6.
  //
  // Note: This algorithm actually computes the closest point but we
  // only return the distance to that point.

  // Get the vertices as points
  const MeshGeometry& geometry = cell.mesh().geometry();
  const unsigned int* vertices = cell.entities(0);
  const Point a = geometry.point(vertices[0]);
  const Point b = geometry.point(vertices[1]);
  const Point c = geometry.point(vertices[2]);
  const Point d = geometry.point(vertices[3]);

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
double TetrahedronCell::normal(const Cell& cell, std::size_t facet, std::size_t i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
Point TetrahedronCell::normal(const Cell& cell, std::size_t facet) const
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
Point TetrahedronCell::cell_normal(const Cell& cell) const
{
  dolfin_error("TetrahedronCell.cpp",
               "compute cell normal",
               "cell_normal not implemented for TetrahedronCell");

  return Point();
}
//-----------------------------------------------------------------------------
double TetrahedronCell::facet_area(const Cell& cell, std::size_t facet) const
{
  dolfin_assert(cell.mesh().topology().dim() == 3);
  dolfin_assert(cell.mesh().geometry().dim() == 3);

  // Create facet from the mesh and local facet number
  Facet f(cell.mesh(), cell.entities(2)[facet]);

  // Get mesh geometry
  const MeshGeometry& geometry = f.mesh().geometry();

  // Get the coordinates of the three vertices
  const unsigned int* vertices = f.entities(0);
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
    const unsigned int* cell_edges = cell.entities(1);

    // Sort vertices on each edge
    for (std::size_t i = 0; i < 6; i++)
    {
      unsigned int* edge_vertices = const_cast<unsigned int*>(topology(1, 0)(cell_edges[i]));
      sort_entities(2, edge_vertices, local_to_global_vertex_indices);
    }
  }

  // Sort local vertices on facets in ascending order, connectivity 2 - 0
  if (!topology(2, 0).empty())
  {
    dolfin_assert(!topology(3, 2).empty());

    // Get facets
    const unsigned int* cell_facets = cell.entities(2);

    // Sort vertices on each facet
    for (std::size_t i = 0; i < 4; i++)
    {
      unsigned int* facet_vertices = const_cast<unsigned int*>(topology(2, 0)(cell_facets[i]));
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
    const unsigned int* cell_facets = cell.entities(2);

    // Loop over facets on cell
    for (std::size_t i = 0; i < 4; i++)
    {
      // For each facet number get the global vertex numbers
      const unsigned int* facet_vertices = topology(2, 0)(cell_facets[i]);

      // For each facet number get the global edge number
      unsigned int* cell_edges = const_cast<unsigned int*>(topology(2, 1)(cell_facets[i]));

      // Loop over vertices on facet
      std::size_t m = 0;
      for (std::size_t j = 0; j < 3; j++)
      {
	// Loop edges on facet
	for (std::size_t k(m); k < 3; k++)
	{
	  // For each edge number get the global vertex numbers
	  const unsigned int* edge_vertices = topology(1, 0)(cell_edges[k]);

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
    unsigned int* cell_vertices = const_cast<unsigned int*>(cell.entities(0));
    sort_entities(4, cell_vertices, local_to_global_vertex_indices);
  }

  // Sort local edges on cell after non-incident vertex tuble, connectivity 3-1
  if (!topology(3, 1).empty())
  {
    dolfin_assert(!topology(1, 0).empty());

    // Get cell vertices and edge numbers
    const unsigned int* cell_vertices = cell.entities(0);
    unsigned int* cell_edges = const_cast<unsigned int*>(cell.entities(1));

    // Loop two vertices on cell as a lexicographical tuple
    // (i, j): (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    std::size_t m = 0;
    for (std::size_t i = 0; i < 3; i++)
    {
      for (std::size_t j = i+1; j < 4; j++)
      {
	// Loop edge numbers
	for (std::size_t k = m; k < 6; k++)
	{
	  // Get local vertices on edge
	  const unsigned int* edge_vertices = topology(1, 0)(cell_edges[k]);

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
    const unsigned int* cell_vertices = cell.entities(0);
    unsigned int* cell_facets   = const_cast<unsigned int*>(cell.entities(2));

    // Loop vertices on cell
    for (std::size_t i = 0; i < 4; i++)
    {
      // Loop facets on cell
      for (std::size_t j = i; j < 4; j++)
      {
	unsigned int* facet_vertices = const_cast<unsigned int*>(topology(2, 0)(cell_facets[j]));

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
bool TetrahedronCell::collides(const Cell& cell, const Point& point) const
{
  // Algorithm from http://www.blackpawn.com/texts/pointinpoly/
  // See also "Real-Time Collision Detection" by Christer Ericson.
  //
  // We express AP as a linear combination of the vectors AB, AC and
  // AD. Point is inside triangle iff AP is a convex combination.

  // Get the vertices as points
  const MeshGeometry& geometry = cell.mesh().geometry();
  const unsigned int* vertices = cell.entities(0);
  Point p0 = geometry.point(vertices[0]);
  Point p1 = geometry.point(vertices[1]);
  Point p2 = geometry.point(vertices[2]);
  Point p3 = geometry.point(vertices[3]);

  // Compute vectors
  Point v1 = p1 - p0;
  Point v2 = p2 - p0;
  Point v3 = p3 - p0;
  Point v = point - p0;

  // Compute entries of linear system
  const double a11 = v1.dot(v1);
  const double a12 = v1.dot(v2);
  const double a13 = v1.dot(v3);
  const double a22 = v2.dot(v2);
  const double a23 = v2.dot(v3);
  const double a33 = v3.dot(v3);
  const double b1 = v.dot(v1);
  const double b2 = v.dot(v2);
  const double b3 = v.dot(v3);

  // Compute subdeterminants
  const double d11 = a22*a33 - a23*a23;
  const double d12 = a12*a33 - a23*a13;
  const double d13 = a12*a23 - a22*a13;
  const double d22 = a11*a33 - a13*a13;
  const double d23 = a11*a23 - a12*a13;
  const double d33 = a11*a22 - a12*a12;

  // Compute inverse of determinant determinant
  const double inv_det = 1.0 / (a11*d11 - a12*d12 + a13*d13);

  // Solve linear system
  const double x1 = inv_det*( d11*b1 - d12*b2 + d13*b3);
  const double x2 = inv_det*(-d12*b1 + d22*b2 - d23*b3);
  const double x3 = inv_det*( d13*b1 - d23*b2 + d33*b3);

  // Tolerance for numeric test (using vector v1)
  const double dx = std::abs(v1.x());
  const double dy = std::abs(v1.y());
  const double dz = std::abs(v1.z());
  const double eps = std::max(DOLFIN_EPS_LARGE, DOLFIN_EPS_LARGE*std::max(dx, std::max(dy, dz)));

  // Check if point is inside cell
  return x1 >= -eps && x2 >= -eps && x3 >= -eps && x1 + x2 + x3 <= 1.0 + eps;
}
//-----------------------------------------------------------------------------
bool TetrahedronCell::collides(const Cell& cell, const MeshEntity& entity) const
{
  switch (entity.dim())
  {
  case 0:
    dolfin_not_implemented();
    break;
  case 1:
    dolfin_not_implemented();
    break;
  case 2:
    return collides_triangle(cell, entity);
  case 3:
    return collides_tetrahedron(cell, entity);
  default:
    dolfin_error("TetrahedronCell.cpp",
		 "collides",
		 "unknown dimension of entity");
  }

  return false;
}

// //-----------------------------------------------------------------------------
// std::vector<double>
// TetrahedronCell::triangulate_intersection(const Cell& cell_0, const Cell& cell_1) const
// {
//   // This algorithm computes the intersection of cell_0 and cell_1 by
//   // returning a vector<double> with points describing a tetrahedral
//   // mesh of the intersection. We will use the fact that the
//   // intersection is a convex polyhedron. The algorithm works by first
//   // identifying intersection points: vertex points inside a cell,
//   // edge-face collision points and edge-edge collision points (the
//   // edge-edge is a rare occurance). Having the intersection points,
//   // we identify points that are coplanar and thus form a facet of the
//   // polyhedron. These points are then used to form a tesselation of
//   // triangles, which are used to form tetrahedra by the use of the
//   // center point of the polyhedron. This center point is thus an
//   // additional point not found on the polyhedron facets.

//   // Tolerance for coplanar points
//   const double coplanar_tol=1e-11;

//   // Tolerance for the tetrahedron determinant (otherwise problems
//   // with warped tets)
//   const double tet_det_tol=1e-15; 

//   // Tolerance for duplicate points (p and q are the same if
//   // (p-q).norm() < same_point_tol)
//   const double same_point_tol=1e-15;

//   // Tolerance for small triangle (could be improved by identifying
//   // sliver and small triangles)
//   const double tri_det_tol=1e-11; 

//   // Points in the triangulation (unique)
//   std::vector<Point> points;

//   // Get the vertices as points
//   const MeshGeometry& geom_0 = cell_0.mesh().geometry();
//   const unsigned int* vert_0 = cell_0.entities(0);
//   const MeshGeometry& geom_1 = cell_1.mesh().geometry();
//   const unsigned int* vert_1 = cell_1.entities(0);

//   // Node intersection
//   for (int i=0; i<4; ++i) 
//   {
//     if (collides(cell_0, geom_1.point(vert_1[i]))) 
//       points.push_back(geom_1.point(vert_1[i]));

//     if (collides(cell_1, geom_0.point(vert_0[i]))) 
//       points.push_back(geom_0.point(vert_0[i]));
//   }

//   // Edge face intersections 
//   std::vector<std::vector<std::size_t> > edges_0(6,std::vector<std::size_t>(2));
//   std::vector<std::vector<std::size_t> > edges_1(6,std::vector<std::size_t>(2));
//   create_entities(edges_0, 1,vert_0);
//   create_entities(edges_1, 1,vert_1);

//   std::vector<std::vector<std::size_t> > faces_0(4,std::vector<std::size_t>(3));
//   std::vector<std::vector<std::size_t> > faces_1(4,std::vector<std::size_t>(3));
//   create_entities(faces_0, 2,vert_0);
//   create_entities(faces_1, 2,vert_1);

//   // Loop over edges e and faces f
//   for (int e=0; e<6; ++e) 
//   { 
//     for (int f=0; f<4; ++f) 
//     {
//       Point pta;
//       if (edge_face_collision(geom_0.point(faces_0[f][0]),
// 			      geom_0.point(faces_0[f][1]),
// 			      geom_0.point(faces_0[f][2]),
// 			      geom_1.point(edges_1[e][0]),
// 			      geom_1.point(edges_1[e][1]),
// 			      pta)) 
// 	points.push_back(pta);
	
//       Point ptb;
//       if (edge_face_collision(geom_1.point(faces_1[f][0]),
// 			      geom_1.point(faces_1[f][1]),
// 			      geom_1.point(faces_1[f][2]),
// 			      geom_0.point(edges_0[e][0]),
// 			      geom_0.point(edges_0[e][1]),
// 			      ptb)) 
// 	points.push_back(ptb);
//     }
//   }    
  
//   // Edge edge intersection (only needed in very few cases)
//   for (int i=0; i<6; ++i) 
//   {
//     for (int j=0; j<6; ++j) 
//     {
//       Point pt;
//       if (edge_edge_collision(geom_0.point(edges_0[i][0]),
// 			      geom_0.point(edges_0[i][1]),
// 			      geom_1.point(edges_1[j][0]),
// 			      geom_1.point(edges_1[j][1]),
// 			      pt)) 
// 	points.push_back(pt);
//     }
//   }
  
//   // Remove duplicate nodes
//   std::vector<Point> tmp; 
//   tmp.reserve(points.size());
//   for (std::size_t i=0; i<points.size(); ++i) 
//   {
//     bool different=true;
//     for (std::size_t j=i+1; j<points.size(); ++j) 
//     {
//       if ((points[i]-points[j]).norm()<same_point_tol) {
// 	different=false;
// 	break;
//       }
//     }

//     if (different) 
//     {
//       tmp.push_back(points[i]);
//     }
//   }
//   points=tmp;

//   // We didn't find sufficiently many points: can't form any
//   // tetrahedra.
//   if (points.size()<4) return std::vector<double>();

//   // Points forming the tetrahedral partitioning of the polyhedron. We
//   // have 4 points per tetrahedron in three dimensions => 12 doubles
//   // per tetrahedron.
//   std::vector<double> triangulation;

//   // Start forming a tesselation
//   if (points.size()==4) 
//   {
//     // Include if determinant is sufficiently large. The determinant
//     // can possibly be computed in a more stable way if needed.
//     const double det=(points[3]-points[0]).dot((points[1]-points[0]).cross(points[2]-points[0]));
//     if (std::abs(det)>tet_det_tol) 
//     {
//       if (det<-tet_det_tol) std::swap(points[0],points[1]);
	  
//       // One tet with four vertices in 3D gives 12 doubles
//       triangulation.resize(12); 
//       for (std::size_t m=0,idx=0; m<4; ++m) 
// 	for (std::size_t d=0; d<3; ++d,++idx) 
// 	  triangulation[idx]=points[m][d];
	  
//     }
//     // Note: this can be empty if the tetrahedron was not sufficiently
//     // large
//     return triangulation;
//   }

//   // Tetrahedra are created using the facet points and a center point.
//   Point polyhedroncenter=points[0];
//   for (std::size_t i=1; i<points.size(); ++i) 
//     polyhedroncenter+=points[i];
//   polyhedroncenter/=points.size();

//   // Data structure for storing checked triangle indices (do this
//   // better with some fancy stl structure?)
//   const int N=points.size(), N2=points.size()*points.size();
//   std::vector<int> checked(N*N2+N2+N,0);

//   // Find coplanar points
//   for (std::size_t i=0; i<points.size(); ++i) 
//   {
//     for (std::size_t j=i+1; j<points.size(); ++j) 
//     {
//       for (std::size_t k=0; k<points.size(); ++k) 
//       {
// 	if (checked[i*N2+j*N+k]==0 and k!=i and k!=j) 
// 	{
// 	  // Check that triangle area is sufficiently large
// 	  Point n=(points[j]-points[i]).cross(points[k]-points[i]);
// 	  const double tridet=n.norm();
// 	  if (tridet<tri_det_tol) { break; }
		  
// 	  // Normalize normal
// 	  n/=tridet; 
		  
// 	  // Compute triangle center
// 	  const Point tricenter=(points[i]+points[j]+points[k])/3.;
		  
// 	  // Check whether all other points are on one side of thus
// 	  // facet. Initialize as true for the case of only three
// 	  // coplanar points.
// 	  bool on_convex_hull=true; 
		  
// 	  // Compute dot products to check which side of the plane
// 	  // (i,j,k) we're on. Note: it seems to be better to compute
// 	  // n.dot(points[m]-n.dot(tricenter) rather than
// 	  // n.dot(points[m]-tricenter).
// 	  std::vector<double> ip(points.size(),-(n.dot(tricenter)));
// 	  for (std::size_t m=0; m<points.size(); ++m) 
// 	    ip[m]+=n.dot(points[m]);
		  
// 	  // Check inner products range by finding max & min (this
// 	  // seemed possibly more numerically stable than checking all
// 	  // vs all and then break).
// 	  double minip=9e99, maxip=-9e99;
// 	  for (size_t m=0; m<points.size(); ++m) 
// 	    if (m!=i and m!=j and m!=k)
// 	    {
// 	      minip = (minip>ip[m]) ? ip[m] : minip;
// 	      maxip = (maxip<ip[m]) ? ip[m] : maxip;
// 	    }

// 	  // Different sign => triangle is not on the convex hull
// 	  if (minip*maxip<-DOLFIN_EPS) 
// 	    on_convex_hull=false;
		  
// 	  if (on_convex_hull) 
// 	  {
// 	    // Find all coplanar points on this facet given the
// 	    // tolerance coplanar_tol
// 	    std::vector<std::size_t> coplanar; 
// 	    for (std::size_t m=0; m<points.size(); ++m) 
// 	      if (std::abs(ip[m])<coplanar_tol) 
// 		coplanar.push_back(m);
		      
// 	    // Mark this plane (how to do this better?)
// 	    for (std::size_t m=0; m<coplanar.size(); ++m) 
// 	      for (std::size_t n=m+1; n<coplanar.size(); ++n) 
// 		for (std::size_t o=n+1; o<coplanar.size(); ++o) 
// 		  checked[coplanar[m]*N2+coplanar[n]*N+coplanar[o]]=
// 		    checked[coplanar[m]*N2+coplanar[o]*N+coplanar[n]]=
// 		    checked[coplanar[n]*N2+coplanar[m]*N+coplanar[o]]=
// 		    checked[coplanar[n]*N2+coplanar[o]*N+coplanar[m]]=
// 		    checked[coplanar[o]*N2+coplanar[n]*N+coplanar[m]]=
// 		    checked[coplanar[o]*N2+coplanar[m]*N+coplanar[n]]=1;
		      
// 	    // Do the actual tesselation using the coplanar points and
// 	    // a center point
// 	    if (coplanar.size()==3) 
// 	    { 
// 	      // Form one tetrahedron
// 	      std::vector<Point> cand(4);
// 	      cand[0]=points[coplanar[0]];
// 	      cand[1]=points[coplanar[1]];
// 	      cand[2]=points[coplanar[2]];
// 	      cand[3]=polyhedroncenter;
			  
// 	      // Include if determinant is sufficiently large
// 	      const double det=(cand[3]-cand[0]).dot((cand[1]-cand[0]).cross(cand[2]-cand[0]));
// 	      if (std::abs(det)>tet_det_tol) 
// 	      {
// 		if (det<-tet_det_tol) 
// 		  std::swap(cand[0],cand[1]);

// 		for (std::size_t m=0; m<4; ++m) 
// 		  for (std::size_t d=0; d<3; ++d) 
// 		    triangulation.push_back(cand[m][d]);
// 	      }

// 	    }
// 	    else 
// 	    {
// 	      // Tesselate as in the triangle-triangle intersection
// 	      // case: First sort points using a Graham scan, then
// 	      // connect to form triangles. Finally form tetrahedra
// 	      // using the center of the polyhedron.
	      
// 	      // Use the center of the coplanar points and point no 0
// 	      // as reference for the angle calculation
// 	      Point pointscenter=points[coplanar[0]];
// 	      for (std::size_t m=1; m<coplanar.size(); ++m) 
// 		pointscenter+=points[coplanar[m]];
// 	      pointscenter/=coplanar.size();
			  
// 	      std::vector<std::pair<double, std::size_t> > order;
// 	      Point ref=points[coplanar[0]]-pointscenter;
// 	      ref/=ref.norm();
// 	      const Point orthref=n.cross(ref);
			  
// 	      // Calculate and store angles
// 	      for (std::size_t m=1; m<coplanar.size(); ++m) 
// 	      {		
// 		const Point v=points[coplanar[m]]-pointscenter;
// 		const double frac=ref.dot(v)/v.norm();
// 		double alpha;
// 		if (frac<=-1) alpha=DOLFIN_PI;
// 		else if (frac>=1) alpha=0;
// 		else { 
// 		  alpha=acos(frac);
// 		  if (v.dot(orthref)<0) 
// 		    alpha=2*DOLFIN_PI-alpha; 
// 		}
// 		order.push_back(std::make_pair(alpha,m));
// 	      }

// 	      // Sort angles
// 	      std::sort(order.begin(),order.end());
			  
// 	      // Tesselate
// 	      for (std::size_t m=0; m<coplanar.size()-2; ++m) 
// 	      {
// 		// Candidate tetrahedron:
// 		std::vector<Point> cand(4);
// 		cand[0]=points[coplanar[0]];
// 		cand[1]=points[coplanar[order[m].second]];
// 		cand[2]=points[coplanar[order[m+1].second]];
// 		cand[3]=polyhedroncenter;

// 		// Include tetrahedron if determinant is "large"
// 		const double det=(cand[3]-cand[0]).dot((cand[1]-cand[0]).cross(cand[2]-cand[0]));
// 		if (std::abs(det)>tet_det_tol) 
// 		{
// 		  if (det<-tet_det_tol) 
// 		    std::swap(cand[0],cand[1]);
// 		  for (std::size_t n=0; n<4; ++n) 
// 		    for (std::size_t d=0; d<3; ++d) 
// 		      triangulation.push_back(cand[n][d]);
// 		}
// 	      }
// 	    }
// 	  }
// 	}
//       }
//     }
//   }

//   return triangulation;
// }
// //-----------------------------------------------------------------------------
// std::vector<double> 
// TetrahedronCell::triangulate_intersection_triangle(const Cell& cell,
// 						   const MeshEntity& entity) const
// {
//   // Tesselate the intersection of a tetrahedron c and a triangle
//   // entity. This gives a convex polyhedron.

//   // Tolerance for duplicate points (p and q are the same if
//   // (p-q).norm() < same_point_tol)
//   const double same_point_tol=1e-15;

  
//   const double Tritol=1e-13;

  
//   // Get the vertices of the tet as points
//   const MeshGeometry& geometry_tet = cell.mesh().geometry();
//   const unsigned int* vertices_tet = cell.entities(0);

//   // Get the vertices of the triangle as points
//   const MeshGeometry& geometry_tri = entity.mesh().geometry();
//   const unsigned int* vertices_tri = entity.entities(0);
//   const Point q0 = geometry_tri.point(vertices_tri[0]);
//   const Point q1 = geometry_tri.point(vertices_tri[1]);
//   const Point q2 = geometry_tri.point(vertices_tri[2]);

//   std::vector<Point> points;

//   // Triangle node in tetrahedron intersection
//   if (collides(cell,q0)) points.push_back(q0);
//   if (collides(cell,q1)) points.push_back(q1);
//   if (collides(cell,q2)) points.push_back(q2);


//   // Check if a tetrahedron edge intersects the triangle
//   std::vector<std::vector<std::size_t> > tetedges(6,std::vector<std::size_t>(2));  
//   create_entities(tetedges, 1,vertices_tet);

//   Point pt;
//   for (int e=0; e<6; ++e) {
//     if (edge_face_collision(q0,q1,q2,
// 			    geometry_tet.point(tetedges[e][0]),
// 			    geometry_tet.point(tetedges[e][1]),
// 			    pt)) {
//       points.push_back(pt);
//     }
//   }			  

  
//   // Check if a triangle edge intersects a tetrahedron face
//   std::vector<std::vector<std::size_t> > tetfaces(4,std::vector<std::size_t>(3));
//   create_entities(tetfaces, 2,vertices_tet);

//   for (int f=0; f<4; ++f) {
//     if (edge_face_collision(geometry_tet.point(tetfaces[f][0]),
// 			    geometry_tet.point(tetfaces[f][1]),
// 			    geometry_tet.point(tetfaces[f][2]),
// 			    q0,q1, pt)) {
//       points.push_back(pt);
//     }
//     if (edge_face_collision(geometry_tet.point(tetfaces[f][0]),
// 			    geometry_tet.point(tetfaces[f][1]),
// 			    geometry_tet.point(tetfaces[f][2]),
// 			    q0,q2, pt)) {
//       points.push_back(pt);
//     }
//     if (edge_face_collision(geometry_tet.point(tetfaces[f][0]),
// 			    geometry_tet.point(tetfaces[f][1]),
// 			    geometry_tet.point(tetfaces[f][2]),
// 			    q1,q2, pt)) {
//       points.push_back(pt);
//     }
//   }

//   // Remove duplicate nodes
//   std::vector<Point> tmp; 
//   tmp.reserve(points.size());
  
//   for (std::size_t i=0; i<points.size(); ++i) {
//     bool different=true;
//     for (std::size_t j=i+1; j<points.size(); ++j) {
//       if ((points[i]-points[j]).norm()<same_point_tol) {
// 	different=false;
// 	break;
//       }
//     }
//     if (different) {
//       tmp.push_back(points[i]);
//     }
//   }
//   points=tmp;

  
//   // We didn't find sufficiently many points
//   if (points.size()<3) {
//     return std::vector<double>();
//   }

//   std::vector<double> triangulation;

//   Point n=(points[2]-points[0]).cross(points[1]-points[0]);
//   const double det=n.norm();
//   n/=det;

//   if (points.size()==3) { 
//     // Include if determinant is sufficiently large
//     if (std::abs(det)>Tritol) {
//       if (det<-Tritol) {
// 	std::swap(points[0],points[1]);
//       }
//       // One triangle with three vertices in 3D gives 9 doubles
//       triangulation.resize(9);
//       for (std::size_t m=0,idx=0; m<4; ++m) {
// 	for (std::size_t d=0; d<3; ++d) {
// 	  triangulation[idx]=points[m][d];
// 	}
//       }
//     }
    
//     return triangulation;
//   }
   
//   // Tesselate as in the triangle-triangle intesection case: First
//   // sort points using a Graham scan, then connect to form triangles.
  
//   // Use the center of the points and point no 0 as reference for the
//   // angle calculation
//   Point pointscenter=points[0];
//   for (std::size_t m=1; m<points.size(); ++m) {
//     pointscenter+=points[m];
//   }
//   pointscenter/=points.size();

//   std::vector<std::pair<double, std::size_t> > order;
//   Point ref=points[0]-pointscenter;
//   ref/=ref.norm();
//   const Point orthref=n.cross(ref);
	  
//   // Calculate and store angles
//   for (std::size_t m=1; m<points.size(); ++m) {		
//     const Point v=points[m]-pointscenter;
//     const double frac=ref.dot(v)/v.norm();
//     double alpha;
//     if (frac<=-1) alpha=DOLFIN_PI;
//     else if (frac>=1) alpha=0;
//     else { 
//       alpha=acos(frac);
//       if (v.dot(orthref)<0) { alpha=2*DOLFIN_PI-alpha; }
//     }
//     order.push_back(std::make_pair(alpha,m));
//   }

//   // Sort angles
//   std::sort(order.begin(),order.end());

//   // Tesselate
//   for (std::size_t m=0; m<order.size()-2; ++m) {
//     // Candidate triangle
//     std::vector<Point> cand(3);
//     cand[0]=points[0];
//     cand[1]=points[order[m].second];
//     cand[2]=points[order[m+1].second];
    
//     // Include triangle if determinant is sufficiently large
//     const double det=((cand[2]-cand[1]).cross(cand[1]-cand[0])).norm();
//     if (std::abs(det)>Tritol) {
//       if (det<-Tritol) {
// 	std::swap(cand[0],cand[1]);
// 	for (std::size_t n=0; n<3; ++n) {
// 	  for (std::size_t d=0; d<3; ++d) {
// 	    triangulation.push_back(cand[n][d]);
// 	  }
// 	}
//       }
//     }
//   }

//   return triangulation;
// }

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
  const unsigned int* v = cell.entities(0);
  const unsigned int* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Ordering convention for edges (order of non-incident vertices)
  static std::size_t EV[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

  // Look for edge satisfying ordering convention
  for (std::size_t j = 0; j < 6; j++)
  {
    const unsigned int* ev = cell.mesh().topology()(1, 0)(e[j]);
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
bool TetrahedronCell::point_outside_of_plane(const Point& point,
                                             const Point& a,
                                             const Point& b,
                                             const Point& c,
                                             const Point& d) const
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // PointOutsideOfPlane on page 144, Section 5.1.6.

  const Point v = (b - a).cross(c - a);
  const double signp = v.dot(point - a);
  const double signd = v.dot(d - a);

  return signp * signd < 0.0;
}
//-----------------------------------------------------------------------------
bool TetrahedronCell::collides_triangle(const Cell& cell,
                                        const MeshEntity& entity) const
{
  // This algorithm checks whether a triangle and a tetrahedron
  // intersects. It's not fast.

  // Get the vertices of the tet as points
  const MeshGeometry& geometry_tet = cell.mesh().geometry();
  const unsigned int* vertices_tet = cell.entities(0);

  // Get the vertices of the triangle as points
  const MeshGeometry& geometry_tri = entity.mesh().geometry();
  const unsigned int* vertices_tri = entity.entities(0);
  const Point q0 = geometry_tri.point(vertices_tri[0]);
  const Point q1 = geometry_tri.point(vertices_tri[1]);
  const Point q2 = geometry_tri.point(vertices_tri[2]);

  // Check if triangle vertices intersects
  if (collides(cell,q0)) return true;
  if (collides(cell,q1)) return true;
  if (collides(cell,q2)) return true;
  
  // Check if a tetrahedron edge intersects the triangle
  std::vector<std::vector<unsigned int> > tetedges(6,std::vector<unsigned int>(2));  
  create_entities(tetedges, 1,vertices_tet);

  Point pt;
  for (int e=0; e<6; ++e) {
    if (edge_face_collision(q0,q1,q2,
			    geometry_tet.point(tetedges[e][0]),
			    geometry_tet.point(tetedges[e][1]),
			    pt)) {
      return true;
    }
  }			  

  // Check if a triangle edge intersects a tetrahedron face
  std::vector<std::vector<unsigned int> > tetfaces(4,std::vector<unsigned int>(3));
  create_entities(tetfaces, 2,vertices_tet);

  for (int f=0; f<4; ++f) {
    if (edge_face_collision(geometry_tet.point(tetfaces[f][0]),
			    geometry_tet.point(tetfaces[f][1]),
			    geometry_tet.point(tetfaces[f][2]),
			    q0,q1, pt)) {
      return true;
    }
    if (edge_face_collision(geometry_tet.point(tetfaces[f][0]),
			    geometry_tet.point(tetfaces[f][1]),
			    geometry_tet.point(tetfaces[f][2]),
			    q0,q2, pt)) {
      return true;
    }
    if (edge_face_collision(geometry_tet.point(tetfaces[f][0]),
			    geometry_tet.point(tetfaces[f][1]),
			    geometry_tet.point(tetfaces[f][2]),
			    q1,q2, pt)) {
      return true;
    }
  }

  // No intersection found
  return false;
}
// //-----------------------------------------------------------------------------
// bool TetrahedronCell::collides_tetrahedron(const Cell& cell,
//                                            const MeshEntity& entity) const
// {
//   // This algorithm checks whether two tetrahedra intersects.

//   // Algorithm and source code from Fabio Ganovelli, Federico Ponchio
//   // and Claudio Rocchini: Fast Tetrahedron-Tetrahedron Overlap
//   // Algorithm. DOI: 10.1080/10867651.2002.10487557.

//   // Get the vertices as points
//   const MeshGeometry& geometry = cell.mesh().geometry();
//   const unsigned int* vertices = cell.entities(0);
//   const MeshGeometry& geometry_q = entity.mesh().geometry();
//   const unsigned int* vertices_q = entity.entities(0);
//   std::vector<Point> V1(4),V2(4);
//   for (int i=0; i<4; ++i) {
//     V1[i]=geometry.point(vertices[i]);
//     V2[i]=geometry_q.point(vertices_q[i]);
//   }
 
//   // Get the vectors between V2 and V1[0]
//   std::vector<Point> P_V1(4); 
//   for (int i=0; i<4; ++i) {
//     P_V1[i] = V2[i]-V1[0];
//   }

//   // Data structure for edges of V1 and V2
//   std::vector<Point> e_v1(5), e_v2(5);
//   e_v1[0]=V1[1]-V1[0];
//   e_v1[1]=V1[2]-V1[0];
//   e_v1[2]=V1[3]-V1[0];

//   Point n;
//   n=e_v1[1].cross(e_v1[0]);
//   // Maybe flip normal. Normal should be outward.
//   if (n.dot(e_v1[2])>0) n*=-1;

//   std::vector<int> masks(4);
//   std::vector<std::vector<double> > Coord_1(4,std::vector<double>(4));

//   if (separating_plane_face_A_1(P_V1,n, Coord_1[0],masks[0])) return false;

//   n=e_v1[0].cross(e_v1[2]);
//   // Maybe flip normal
//   if (n.dot(e_v1[1])>0) n*=-1;

//   if (separating_plane_face_A_1(P_V1,n, Coord_1[1],masks[1])) return false;

//   if (separating_plane_edge_A(Coord_1,masks, 0,1)) return false;

//   n=e_v1[2].cross(e_v1[1]);
//   // Mmaybe flip normal
//   if (n.dot(e_v1[0])>0) n*=-1;

//   if (separating_plane_face_A_1(P_V1,n, Coord_1[2],masks[2])) return false;

//   if (separating_plane_edge_A(Coord_1,masks, 0,2)) return false;

//   if (separating_plane_edge_A(Coord_1,masks, 1,2)) return false;

//   e_v1[4]=V1[3]-V1[1];
//   e_v1[3]=V1[2]-V1[1];
//   n=e_v1[3].cross(e_v1[4]);
//   // Maybe flip normal. Note the < since e_v1[0]=v1-v0.
//   if (n.dot(e_v1[0])<0) n*=-1; 

//   if (separating_plane_face_A_2(V1,V2,n, Coord_1[3],masks[3])) return false;

//   if (separating_plane_edge_A(Coord_1,masks, 0,3)) return false;

//   if (separating_plane_edge_A(Coord_1,masks, 1,3)) return false;

//   if (separating_plane_edge_A(Coord_1,masks, 2,3)) return false;

//   if ((masks[0] | masks[1] | masks[2] | masks[3] )!= 15) return true;


//   // From now on, if there is a separating plane it is parallel to a
//   // face of b.

//   std::vector<Point> P_V2(4);
//   for (int i=0; i<4; ++i) {
//     P_V2[i] = V1[i]-V2[0];
//   }

//   e_v2[0]=V2[1]-V2[0];
//   e_v2[1]=V2[2]-V2[0];
//   e_v2[2]=V2[3]-V2[0];

//   n=e_v2[1].cross(e_v2[0]);
//   // Maybe flip normal
//   if (n.dot(e_v2[2])>0) n*=-1;

//   if (separating_plane_face_B_1(P_V2,n)) return false;

//   n=e_v2[0].cross(e_v2[2]);
//   // Maybe flip normal
//   if (n.dot(e_v2[1])>0) n*=-1;

//   if (separating_plane_face_B_1(P_V2,n)) return false;

//   n=e_v2[2].cross(e_v2[1]);
//   // Maybe flip normal
//   if (n.dot(e_v2[0])>0) n*=-1;

//   if (separating_plane_face_B_1(P_V2,n)) return false;

//   e_v2[4]=V2[3]-V2[1];
//   e_v2[3]=V2[2]-V2[1];

//   n=e_v2[3].cross(e_v2[4]);
//   // Maybe flip normal. Note the < since e_v2[0]=V2[1]-V2[0].
//   if (n.dot(e_v2[0])<0) n*=-1; 

//   if (separating_plane_face_B_2(V1,V2,n)) return false;

//   return true;

// }
// //-----------------------------------------------------------------------------
// bool TetrahedronCell::separating_plane_face_A_1(const std::vector<Point>& P_V1,
//                                                 const Point& n,
//                                                 std::vector<double>& Coord, 
//                                                 int&  maskEdges) const
// {
//   // Helper function for collides_tetrahedron: checks if plane pv1 is
//   // a separating plane. Stores local coordinates bc and the mask bit
//   // maskEdges.

//   maskEdges = 0;
//   if ((Coord[0] = P_V1[0].dot(n)) > 0) maskEdges = 1;
//   if ((Coord[1] = P_V1[1].dot(n)) > 0) maskEdges |= 2;
//   if ((Coord[2] = P_V1[2].dot(n)) > 0) maskEdges |= 4;
//   if ((Coord[3] = P_V1[3].dot(n)) > 0) maskEdges |= 8;
//   return (maskEdges == 15);
// }

// //-----------------------------------------------------------------------------
// bool TetrahedronCell::separating_plane_face_A_2(const std::vector<Point>& V1,
//                                                 const std::vector<Point>& V2,
//                                                 const Point& n,
//                                                 std::vector<double>& Coord, 
//                                                 int&  maskEdges) const 
// {
//   // Helper function for collides_tetrahedron: checks if plane v1,v2
//   // is a separating plane. Stores local coordinates bc and the mask
//   // bit maskEdges.

//   maskEdges = 0;
//   if ((Coord[0] = (V2[0]-V1[1]).dot(n)) > 0) maskEdges = 1;
//   if ((Coord[1] = (V2[1]-V1[1]).dot(n)) > 0) maskEdges |= 2;
//   if ((Coord[2] = (V2[2]-V1[1]).dot(n)) > 0) maskEdges |= 4;
//   if ((Coord[3] = (V2[3]-V1[1]).dot(n)) > 0) maskEdges |= 8;
//   return (maskEdges == 15);
// }
// //-----------------------------------------------------------------------------
// bool TetrahedronCell::separating_plane_edge_A(const std::vector<std::vector<double> >& Coord_1,
//                                               const std::vector<int>& masks,
//                                               int f0 , 
//                                               int f1) const
// {
//   // Helper function for collides_tetrahedron: checks if edge is in
//   // the plane separating faces f0 and f1.

//   const std::vector<double>& coord_f0=Coord_1[f0];
//   const std::vector<double>& coord_f1=Coord_1[f1];

//   int maskf0 = masks[f0];
//   int maskf1 = masks[f1];

//   if ((maskf0 | maskf1) != 15) // if there is a vertex of b
//     return false; // included in (-,-) return false

//   maskf0 &= (maskf0 ^ maskf1); // exclude the vertices in (+,+)
//   maskf1 &= (maskf0 ^ maskf1);

//   // edge 0: 0--1
//   if (((maskf0 & 1) && // the vertex 0 of b is in (-,+)
//        (maskf1 & 2)) && // the vertex 1 of b is in (+,-)
//       (((coord_f0[1] * coord_f1[0]) -
//         (coord_f0[0] * coord_f1[1])) >0 ))
//     // the edge of b (0,1) intersect (-,-) (see the paper)
//     return false;

//   if (((maskf0 & 2) && (maskf1 & 1)) && (((coord_f0[1] * coord_f1[0]) - (coord_f0[0] * coord_f1[1])) < 0))
//     return false;

//   // edge 1: 0--2
//   if (((maskf0 & 1) && (maskf1 & 4)) && (((coord_f0[2] * coord_f1[0]) - (coord_f0[0] * coord_f1[2])) > 0))
//     return false;

//   if (((maskf0 & 4) && (maskf1 & 1)) && (((coord_f0[2] * coord_f1[0]) - (coord_f0[0] * coord_f1[2])) < 0))
//     return false;

//   // edge 2: 0--3
//   if (((maskf0 & 1) &&(maskf1 & 8)) && (((coord_f0[3] * coord_f1[0]) - (coord_f0[0] * coord_f1[3])) > 0))
//     return false;

//   if (((maskf0 & 8) && (maskf1 & 1)) && (((coord_f0[3] * coord_f1[0]) - (coord_f0[0] * coord_f1[3])) < 0))
//     return false;

//   // edge 3: 1--2
//   if (((maskf0 & 2) && (maskf1 & 4)) && (((coord_f0[2] * coord_f1[1]) - (coord_f0[1] * coord_f1[2])) > 0))
//     return false;

//   if (((maskf0 & 4) && (maskf1 & 2)) && (((coord_f0[2] * coord_f1[1]) - (coord_f0[1] * coord_f1[2])) < 0))
//     return false;

//   // edge 4: 1--3
//   if (((maskf0 & 2) && (maskf1 & 8)) && (((coord_f0[3] * coord_f1[1]) - (coord_f0[1] * coord_f1[3])) > 0))
//     return false;

//   if (((maskf0 & 8) && (maskf1 & 2)) && (((coord_f0[3] * coord_f1[1]) - (coord_f0[1] * coord_f1[3])) < 0))
//     return false;

//   // edge 5: 2--3
//   if (((maskf0 & 4) && (maskf1 & 8)) && (((coord_f0[3] * coord_f1[2]) - (coord_f0[2] * coord_f1[3])) > 0))
//     return false;

//   if (((maskf0 & 8) && (maskf1 & 4)) && (((coord_f0[3] * coord_f1[2]) - (coord_f0[2] * coord_f1[3])) < 0))
//     return false;

//   // Now there exists a separating plane supported by the edge shared
//   // by f0 and f1.
//   return true; 
// }
//-----------------------------------------------------------------------------
bool TetrahedronCell::edge_face_collision(const Point& r,
                                          const Point& s,
                                          const Point& t,
                                          const Point& a,
                                          const Point& b,
                                          Point& pt) const
{
  // This standard edge face intersection test is as follows:
  // - Check if end points of the edge (a,b) on opposite side of plane
  // given by the face (r,s,t)
  // - If we have sign change, compute intersection with plane.
  // - Check if computed point is on triangle given by face.

  // If the edge and the face are in the same plane, we return false
  // and leave this to the edge-edge intersection test.

  // Tolerance for edga and face in plane (topologically 2D problem)
  const double Top2dtol=1e-15; 
  
  // Compute normal
  const Point rs=s-r;
  const Point rt=t-r;
  Point n=rs.cross(rt);
  n/=n.norm();

  // Check sign change (note that if either dot product is zero it's
  // orthogonal)
  const double da=n.dot(a-r);
  const double db=n.dot(b-r);

  // Note: if da and db we may have edge intersection (detected in
  // other routine)
  if (da*db>0) return false; 

  // Face and edge are in topological 2d: taken care of in edge-edge
  // intersection.
  if (std::abs(da)+std::abs(db) < Top2dtol) return false;

  // Calculate intersection
  pt=a+std::abs(da)/(std::abs(da)+std::abs(db))*(b-a);

  // Check if point is in triangle by calculating and checking
  // barycentric coords.
  const double d00=rs.squared_norm();
  const double d01=rs.dot(rt);
  const double d11=rt.squared_norm();
  const Point e2=pt-r;
  const double d20=e2.dot(rs);
  const double d21=e2.dot(rt);
  const double invdet=1./(d00*d11-d01*d01);
  const double v=(d11*d20-d01*d21)*invdet;
  if (v<0) return false;
  const double w=(d00*d21-d01*d20)*invdet;
  if (w<0) return false;
  if (v+w>1) return false;

  return true;
}
//-----------------------------------------------------------------------------
bool TetrahedronCell::edge_edge_collision(const Point& a,
                                          const Point& b,
                                          const Point& c,
                                          const Point& d,
                                          Point& pt) const
{
  // This algorithm determines if the edge (a,b) intersects the edge
  // (c,d). If so, the intersection point is pt. It doesn't allow for
  // edges that are the same. 

  // Tolerance for same point.
  const double same_point_tol=1e-13;

  if ((a-c).norm()<same_point_tol and (b-d).norm()<same_point_tol) {
    return false; 
  }

  if ((a-d).norm()<same_point_tol and (b-c).norm()<same_point_tol) {
    return false;
  }

  // Tolerance for orthogonality
  const double Orthtol=1e-15;

  // Tolerance for coplanarity of the edges
  const double coplanar_tol=1e-15;

  // Form edges and normal
  const Point L1=b-a, L2=d-c;
  const Point ca=c-a;
  const Point n=L1.cross(L2);

  // Check if L1 and L2 are coplanar
  if (std::abs(ca.dot(n))>coplanar_tol) return false;

  // Find orthogonal plane with normal n1
  const Point n1=n.cross(L1);
  const double n1dotL2=n1.dot(L2);
  if (std::abs(n1dotL2)>Orthtol) { 
    const double t=n1.dot(a-c)/n1dotL2;

    // Find orthogonal plane with normal n2
    const Point n2=n.cross(L2);
    const double n2dotL1=n2.dot(L1);

    if (t>=0 and t<=1 and std::abs(n2dotL1)>Orthtol) {
      const double s=n2.dot(c-a)/n2dotL1;
      if (s>=0 and s<=1) {
        pt=a+s*L1;
        return true;
      }
    }
  }

  return false;
}  
//-----------------------------------------------------------------------------
