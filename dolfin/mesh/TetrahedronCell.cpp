// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman 2006.
// Modified by Garth N. Wells 2006.
// Modified by Kristian Oelgaard 2006.
//
// First added:  2006-06-05
// Last changed: 2008-05-08

#include <algorithm>
#include <dolfin/log/dolfin_log.h>
#include "Cell.h"
#include "MeshEditor.h"
#include "MeshGeometry.h"
#include "Facet.h"
#include "TetrahedronCell.h"
#include "Vertex.h"
#include "GeometricPredicates.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint TetrahedronCell::dim() const
{
  return 3;
}
//-----------------------------------------------------------------------------
dolfin::uint TetrahedronCell::numEntities(uint dim) const
{
  switch ( dim )
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
    error("Illegal topological dimension %d for tetrahedron.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint TetrahedronCell::numVertices(uint dim) const
{
  switch ( dim )
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
    error("Illegal topological dimension %d for tetrahedron.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint TetrahedronCell::orientation(const Cell& cell) const
{
  // This is a trick to be allowed to initialize mesh entities from cell
  Cell& c = const_cast<Cell&>(cell);

  Vertex v0(c.mesh(), c.entities(0)[0]);
  Vertex v1(c.mesh(), c.entities(0)[1]);
  Vertex v2(c.mesh(), c.entities(0)[2]);
  Vertex v3(c.mesh(), c.entities(0)[3]);

  Point p01 = v1.point() - v0.point();
  Point p02 = v2.point() - v0.point();
  Point p03 = v3.point() - v0.point();

  Point n = p01.cross(p02);

  return ( n.dot(p03) < 0.0 ? 1 : 0 );
}
//-----------------------------------------------------------------------------
void TetrahedronCell::createEntities(uint** e, uint dim, const uint* v) const
{
  // We only need to know how to create edges and faces
  switch ( dim )
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
    error("Don't know how to create entities of topological dimension %d.", dim);
  }
}
//-----------------------------------------------------------------------------
void TetrahedronCell::orderEntities(Cell& cell) const
{
  // Sort i - j for i > j: 1 - 0, 2 - 0, 2 - 1, 3 - 0, 3 - 1, 3 - 2

  // Get mesh topology
  MeshTopology& topology = cell.mesh().topology();

  // Sort local vertices on edges in ascending order, connectivity 1 - 0
  if ( topology(1, 0).size() > 0 )
  {
    dolfin_assert(topology(3, 1).size() > 0);

    // Get edges
    uint* cell_edges = cell.entities(1);

    // Sort vertices on each edge
    for (uint i = 0; i < 6; i++)
    {
      uint* edge_vertices = topology(1, 0)(cell_edges[i]);
      std::sort(edge_vertices, edge_vertices + 2);
    }
  }

  // Sort local vertices on facets in ascending order, connectivity 2 - 0
  if ( topology(2, 0).size() > 0 )
  {
    dolfin_assert(topology(3, 2).size() > 0);

    // Get facets
    uint* cell_facets = cell.entities(2);

    // Sort vertices on each facet
    for (uint i = 0; i < 4; i++)
    {
      uint* facet_vertices = topology(2, 0)(cell_facets[i]);
      std::sort(facet_vertices, facet_vertices + 3);
    }
  }

  // Sort local edges on local facets after non-incident vertex, connectivity 2 - 1
  if ( topology(2, 1).size() > 0 )
  {
    dolfin_assert(topology(3, 2).size() > 0);
    dolfin_assert(topology(2, 0).size() > 0);
    dolfin_assert(topology(1, 0).size() > 0);

    // Get facet numbers
    uint* cell_facets = cell.entities(2);

    // Loop over facets on cell
    for (uint i = 0; i < 4; i++)
    {
      // For each facet number get the global vertex numbers
      uint* facet_vertices = topology(2, 0)(cell_facets[i]);

      // For each facet number get the global edge number
      uint* cell_edges = topology(2, 1)(cell_facets[i]);

      // Loop over vertices on facet
      uint m = 0;
      for (uint j = 0; j < 3; j++)
      {
        // Loop edges on facet
        for (uint k(m); k < 3; k++)
        {
          // For each edge number get the global vertex numbers
          uint* edge_vertices = topology(1, 0)(cell_edges[k]);

          // Check if the jth vertex of facet i is non-incident on edge k
          if (!std::count(edge_vertices, edge_vertices+2, facet_vertices[j]))
          {
            // Swap facet numbers
            uint tmp = cell_edges[m];
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
  if ( topology(3, 0).size() > 0 )
  {
    uint* cell_vertices = cell.entities(0);
    std::sort(cell_vertices, cell_vertices+4);
  }

  // Sort local edges on cell after non-incident vertex tuble, connectivity 3-1
  if ( topology(3, 1).size() > 0 )
  {
    dolfin_assert(topology(1, 0).size() > 0);

    // Get cell vertices and edge numbers
    uint* cell_vertices = cell.entities(0);
    uint* cell_edges = cell.entities(1);

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
          uint* edge_vertices = topology(1, 0)(cell_edges[k]);

          // Check if the ith and jth vertex of the cell are non-incident on edge k
          if (!std::count(edge_vertices, edge_vertices+2, cell_vertices[i]) && \
              !std::count(edge_vertices, edge_vertices+2, cell_vertices[j]) )
          {
            // Swap edge numbers
            uint tmp = cell_edges[m];
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
  if ( topology(3, 2).size() > 0 )
  {
    dolfin_assert(topology(2, 0).size() > 0);

    // Get cell vertices and facet numbers
    uint* cell_vertices = cell.entities(0);
    uint* cell_facets = cell.entities(2);

    // Loop vertices on cell
    for (uint i = 0; i < 4; i++)
    {
      // Loop facets on cell
      for (uint j = i; j < 4; j++)
      {
        uint* facet_vertices = topology(2, 0)(cell_facets[j]);

        // Check if the ith vertex of the cell is non-incident on facet j
        if (!std::count(facet_vertices, facet_vertices+3, cell_vertices[i]))
        {
          // Swap facet numbers
          uint tmp = cell_facets[i];
          cell_facets[i] = cell_facets[j];
          cell_facets[j] = tmp;
          break;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void TetrahedronCell::refineCell(Cell& cell, MeshEditor& editor,
			     uint& current_cell) const
{
  // Get vertices and edges
  const uint* v = cell.entities(0);
  const uint* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Get offset for new vertex indices
  const uint offset = cell.mesh().numVertices();

  // Compute indices for the ten new vertices
  const uint v0 = v[0];
  const uint v1 = v[1];
  const uint v2 = v[2];
  const uint v3 = v[3];
  const uint e0 = offset + e[findEdge(0, cell)];
  const uint e1 = offset + e[findEdge(1, cell)];
  const uint e2 = offset + e[findEdge(2, cell)];
  const uint e3 = offset + e[findEdge(3, cell)];
  const uint e4 = offset + e[findEdge(4, cell)];
  const uint e5 = offset + e[findEdge(5, cell)];

  // Regular refinement: 8 new cells
  editor.addCell(current_cell++, v0, e3, e4, e5);
  editor.addCell(current_cell++, v1, e1, e2, e5);
  editor.addCell(current_cell++, v2, e0, e2, e4);
  editor.addCell(current_cell++, v3, e0, e1, e3);
  editor.addCell(current_cell++, e0, e1, e2, e5);
  editor.addCell(current_cell++, e0, e1, e3, e5);
  editor.addCell(current_cell++, e0, e2, e4, e5);
  editor.addCell(current_cell++, e0, e3, e4, e5);
}
//-----------------------------------------------------------------------------
void TetrahedronCell::refineCellIrregular(Cell& cell, MeshEditor& editor, 
				      uint& current_cell, uint refinement_rule, 
				      uint* marked_edges) const
{
  warning("Not implemented yet.");

  /*
  // Get vertices and edges
  const uint* v = cell.entities(0);
  const uint* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Get offset for new vertex indices
  const uint offset = cell.mesh().numVertices();

  // Compute indices for the ten new vertices
  const uint v0 = v[0];
  const uint v1 = v[1];
  const uint v2 = v[2];
  const uint v3 = v[3];
  const uint e0 = offset + e[0];
  const uint e1 = offset + e[1];
  const uint e2 = offset + e[2];
  const uint e3 = offset + e[3];
  const uint e4 = offset + e[4];
  const uint e5 = offset + e[5];

  // Refine according to refinement rule
  // The rules are numbered according to the paper: 
  // J. Bey, "Tetrahedral Grid Refinement", 1995.   
  switch ( refinement_rule )
  {
  case 1:
    // Rule 1: 4 new cells
    editor.addCell(current_cell++, v0, e1, e3, e2);
    editor.addCell(current_cell++, v1, e2, e4, e0);
    editor.addCell(current_cell++, v2, e0, e5, e1);
    editor.addCell(current_cell++, v3, e5, e4, e3);
    break;
  case 2:
    // Rule 2: 2 new cells
    editor.addCell(current_cell++, v0, e1, e3, e2);
    editor.addCell(current_cell++, v1, e2, e4, e0);
    break;
  case 3:
    // Rule 3: 3 new cells
    editor.addCell(current_cell++, v0, e1, e3, e2);
    editor.addCell(current_cell++, v1, e2, e4, e0);
    editor.addCell(current_cell++, v2, e0, e5, e1);
    break;
  case 4:
    // Rule 4: 4 new cells
    editor.addCell(current_cell++, v0, e1, e3, e2);
    editor.addCell(current_cell++, v1, e2, e4, e0);
    editor.addCell(current_cell++, v2, e0, e5, e1);
    editor.addCell(current_cell++, v3, e5, e4, e3);
    break;
  default:
    error("Illegal rule for irregular refinement of tetrahedron.");
  }
  */
}
//-----------------------------------------------------------------------------
real TetrahedronCell::volume(const MeshEntity& tetrahedron) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = tetrahedron.mesh().geometry();

  // Only know how to compute the volume when embedded in R^3
  if ( geometry.dim() != 3 )
    error("Only know how to compute the volume of a tetrahedron when embedded in R^3.");

  // Get the coordinates of the four vertices
  const uint* vertices = tetrahedron.entities(0);
  const real* x0 = geometry.x(vertices[0]);
  const real* x1 = geometry.x(vertices[1]);
  const real* x2 = geometry.x(vertices[2]);
  const real* x3 = geometry.x(vertices[3]);

  // Formula for volume from http://mathworld.wolfram.com
  real v = ( x0[0] * ( x1[1]*x2[2] + x3[1]*x1[2] + x2[1]*x3[2] - x2[1]*x1[2] - x1[1]*x3[2] - x3[1]*x2[2] ) -
             x1[0] * ( x0[1]*x2[2] + x3[1]*x0[2] + x2[1]*x3[2] - x2[1]*x0[2] - x0[1]*x3[2] - x3[1]*x2[2] ) +
             x2[0] * ( x0[1]*x1[2] + x3[1]*x0[2] + x1[1]*x3[2] - x1[1]*x0[2] - x0[1]*x3[2] - x3[1]*x1[2] ) -
             x3[0] * ( x0[1]*x1[2] + x1[1]*x2[2] + x2[1]*x0[2] - x1[1]*x0[2] - x2[1]*x1[2] - x0[1]*x2[2] ) );
  
  return std::abs(v) / 6.0;
}
//-----------------------------------------------------------------------------
real TetrahedronCell::diameter(const MeshEntity& tetrahedron) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = tetrahedron.mesh().geometry();

  // Only know how to compute the volume when embedded in R^3
  if ( geometry.dim() != 3 )
    error("Only know how to compute the diameter of a tetrahedron when embedded in R^3.");
  
  // Get the coordinates of the four vertices
  const uint* vertices = tetrahedron.entities(0);
  Point p0 = geometry.point(vertices[0]);
  Point p1 = geometry.point(vertices[1]);
  Point p2 = geometry.point(vertices[2]);
  Point p3 = geometry.point(vertices[3]);

  // Compute side lengths
  real a  = p1.distance(p2);
  real b  = p0.distance(p2);
  real c  = p0.distance(p1);
  real aa = p0.distance(p3);
  real bb = p1.distance(p3);
  real cc = p2.distance(p3);
                                
  // Compute "area" of triangle with strange side lengths
  real la   = a*aa;
  real lb   = b*bb;
  real lc   = c*cc;
  real s    = 0.5*(la+lb+lc);
  real area = sqrt(s*(s-la)*(s-lb)*(s-lc));
                                
  // Formula for diameter (2*circumradius) from http://mathworld.wolfram.com
  return area / ( 3.0*volume(tetrahedron) );
}
//-----------------------------------------------------------------------------
real TetrahedronCell::normal(const Cell& cell, uint facet, uint i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
Point TetrahedronCell::normal(const Cell& cell, uint facet) const
{
  // This is a trick to be allowed to initialize a facet from the cell
  Cell& c = const_cast<Cell&>(cell);
  
  // Create facet from the mesh and local facet number
  Facet f(c.mesh(), c.entities(2)[facet]);

  // Get global index of opposite vertex
  const uint v0 = cell.entities(0)[facet];
  
  // Get global index of vertices on the facet
  uint v1 = f.entities(0)[0];
  uint v2 = f.entities(0)[1];
  uint v3 = f.entities(0)[2];

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();
  
  // Get the coordinates of the four vertices
  const real* p0 = geometry.x(v0);
  const real* p1 = geometry.x(v1);
  const real* p2 = geometry.x(v2);
  const real* p3 = geometry.x(v3);

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
bool TetrahedronCell::intersects(const MeshEntity& tetrahedron,
                             const Point& p) const
{
  // Adapted from gts_point_is_in_triangle from GTS

  // Get mesh geometry
  const MeshGeometry& geometry = tetrahedron.mesh().geometry();

  // Get global index of vertices of the tetrahedron
  uint v0 = tetrahedron.entities(0)[0];
  uint v1 = tetrahedron.entities(0)[1];
  uint v2 = tetrahedron.entities(0)[2];
  uint v3 = tetrahedron.entities(0)[3];

  // Check orientation
  dolfin::uint vtmp;
  if(orientation((Cell&)tetrahedron) == 1)
  {
    vtmp = v3;
    v3 = v2;
    v2 = vtmp;
  }

  // Get the coordinates of the four vertices
  const real* x0 = geometry.x(v0);
  const real* x1 = geometry.x(v1);
  const real* x2 = geometry.x(v2);
  const real* x3 = geometry.x(v3);

  real xcoordinates[3];
  real* x = xcoordinates;

  x[0] = p[0];
  x[1] = p[1];
  x[2] = p[2];

  real d1, d2, d3, d4;

  // Test orientation of p w.r.t. each face
  d1 = orient3d((double *)x2, (double *)x1, (double *)x0, x);
  d2 = orient3d((double *)x0, (double *)x3, (double *)x2, x);
  d3 = orient3d((double *)x0, (double *)x1, (double *)x3, x);
  d4 = orient3d((double *)x1, (double *)x2, (double *)x3, x);

  // FIXME: Need to check the predicates for correctness
//   if(fabs(d1) == DOLFIN_EPS ||
//      fabs(d2) == DOLFIN_EPS ||
//      fabs(d3) == DOLFIN_EPS ||
//      fabs(d4) == DOLFIN_EPS)
//   {
//     return true;
//   }
  if(d1 < 0.0)
    return false;
  if(d2 < 0.0)
    return false;
  if(d3 < 0.0)
    return false;
  if(d4 < 0.0)
    return false;

  return true;
}
//-----------------------------------------------------------------------------
bool TetrahedronCell::intersects(const MeshEntity& interval, const Point& p1, const Point& p2) const
{
  // FIXME: Not implemented
  error("Interval::intersects() not implemented");

  return false;
}
//-----------------------------------------------------------------------------
std::string TetrahedronCell::description() const
{
  std::string s = "tetrahedron (simplex of topological dimension 3)";
  return s;
}
//-----------------------------------------------------------------------------
dolfin::uint TetrahedronCell::findEdge(uint i, const Cell& cell) const
{
  // Get vertices and edges
  const uint* v = cell.entities(0);
  const uint* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);
  
  // Ordering convention for edges (order of non-incident vertices)
  static uint EV[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

  // Look for edge satisfying ordering convention
  for (uint j = 0; j < 6; j++)
  {
    const uint* ev = cell.mesh().topology()(1, 0)(e[j]);
    dolfin_assert(ev);
    const uint v0 = v[EV[i][0]];
    const uint v1 = v[EV[i][1]];
    if (ev[0] != v0 && ev[0] != v1 && ev[1] != v0 && ev[1] != v1)
      return j;
  }

  // We should not reach this
  error("Unable to find edge.");

  return 0;
}
//-----------------------------------------------------------------------------
