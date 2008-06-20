// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
// Modified by Kristian Oelgaard, 2006-2007.
// Modified by Dag Lindbo, 2008
// 
// First added:  2006-06-05
// Last changed: 2008-06-20
//
// Rename of the former Triangle.cpp
//

#include <algorithm>
#include <dolfin/log/dolfin_log.h>
#include "Cell.h"
#include "MeshEditor.h"
#include "Facet.h"
#include "TriangleCell.h"
#include "Vertex.h"
#include "GeometricPredicates.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint TriangleCell::dim() const
{
  return 2;
}
//-----------------------------------------------------------------------------
dolfin::uint TriangleCell::numEntities(uint dim) const
{
  switch ( dim )
    {
    case 0:
      return 3; // vertices
    case 1:
      return 3; // edges
    case 2:
      return 1; // cells
    default:
      error("Illegal topological dimension %d for triangle.", dim);
    }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint TriangleCell::numVertices(uint dim) const
{
  switch ( dim )
    {
    case 0:
      return 1; // vertices
    case 1:
      return 2; // edges
    case 2:
      return 3; // cells
    default:
      error("Illegal topological dimension %d for triangle.", dim);
    }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint TriangleCell::orientation(const Cell& cell) const
{
  // This is a trick to be allowed to initialize mesh entities from cell
  Cell& c = const_cast<Cell&>(cell);

  Vertex v0(c.mesh(), c.entities(0)[0]);
  Vertex v1(c.mesh(), c.entities(0)[1]);
  Vertex v2(c.mesh(), c.entities(0)[2]);

  Point p01 = v1.point() - v0.point();
  Point p02 = v2.point() - v0.point();
  Point n(-p01.y(), p01.x());

  return ( n.dot(p02) < 0.0 ? 1 : 0 );
}
//-----------------------------------------------------------------------------
void TriangleCell::createEntities(uint** e, uint dim, const uint* v) const
{
  // We only need to know how to create edges
  if ( dim != 1 )
    error("Don't know how to create entities of topological dimension %d.", dim);

  // Create the three edges
  e[0][0] = v[1]; e[0][1] = v[2];
  e[1][0] = v[0]; e[1][1] = v[2];
  e[2][0] = v[0]; e[2][1] = v[1];
}
//-----------------------------------------------------------------------------
void TriangleCell::orderEntities(Cell& cell) const
{
  // Sort i - j for i > j: 1 - 0, 2 - 0, 2 - 1

  // Get mesh topology
  MeshTopology& topology = cell.mesh().topology();

  // Sort local vertices on edges in ascending order, connectivity 1 - 0
  if ( topology(1, 0).size() > 0 )
    {
      dolfin_assert(topology(2, 1).size() > 0);

      // Get edges
      uint* cell_edges = cell.entities(1);

      // Sort vertices on each edge
      for (uint i = 0; i < 3; i++)
	{
	  uint* edge_vertices = topology(1, 0)(cell_edges[i]);
	  std::sort(edge_vertices, edge_vertices + 2);
	}
    }

  // Sort local vertices on cell in ascending order, connectivity 2 - 0
  if ( topology(2, 0).size() > 0 )
    {
      uint* cell_vertices = cell.entities(0);
      std::sort(cell_vertices, cell_vertices + 3);
    }

  // Sort local edges on cell after non-incident vertex, connectivity 2 - 1
  if ( topology(2, 1).size() > 0 )
    {
      dolfin_assert(topology(2, 1).size() > 0);

      // Get cell vertices and edges
      uint* cell_vertices = cell.entities(0);
      uint* cell_edges = cell.entities(1);

      // Loop over vertices on cell
      for (uint i = 0; i < 3; i++)
	{
	  // Loop over edges on cell
	  for (uint j = i; j < 3; j++)
	    {
	      uint* edge_vertices = topology(1, 0)(cell_edges[j]);

	      // Check if the ith vertex of the cell is non-incident with edge j
	      if ( std::count(edge_vertices, edge_vertices + 2, cell_vertices[i]) == 0 )
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
void TriangleCell::refineCell(Cell& cell, MeshEditor& editor,
			      uint& current_cell) const
{
  // Get vertices and edges
  const uint* v = cell.entities(0);
  const uint* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Get offset for new vertex indices
  const uint offset = cell.mesh().numVertices();

  // Compute indices for the six new vertices
  const uint v0 = v[0];
  const uint v1 = v[1];
  const uint v2 = v[2];
  const uint e0 = offset + e[findEdge(0, cell)];
  const uint e1 = offset + e[findEdge(1, cell)];
  const uint e2 = offset + e[findEdge(2, cell)];
  
  // Add the four new cells
  editor.addCell(current_cell++, v0, e2, e1);
  editor.addCell(current_cell++, v1, e0, e2);
  editor.addCell(current_cell++, v2, e1, e0);
  editor.addCell(current_cell++, e0, e1, e2);
}
//-----------------------------------------------------------------------------
real TriangleCell::volume(const MeshEntity& triangle) const
{
  // Check that we get a triangle
  if ( triangle.dim() != 2 )
    error("Illegal mesh entity for computation of triangle volume (area). Not a triangle.");

  // Get mesh geometry
  const MeshGeometry& geometry = triangle.mesh().geometry();

  // Get the coordinates of the three vertices
  const uint* vertices = triangle.entities(0);
  const real* x0 = geometry.x(vertices[0]);
  const real* x1 = geometry.x(vertices[1]);
  const real* x2 = geometry.x(vertices[2]);
  
  if ( geometry.dim() == 2 )
    {
      // Compute area of triangle embedded in R^2
      real v2 = (x0[0]*x1[1] + x0[1]*x2[0] + x1[0]*x2[1]) - (x2[0]*x1[1] + x2[1]*x0[0] + x1[0]*x0[1]);
    
      // Formula for volume from http://mathworld.wolfram.com 
      return v2 = 0.5 * std::abs(v2);
    }
  else if ( geometry.dim() == 3 )
    { 
      // Compute area of triangle embedded in R^3
      real v0 = (x0[1]*x1[2] + x0[2]*x2[1] + x1[1]*x2[2]) - (x2[1]*x1[2] + x2[2]*x0[1] + x1[1]*x0[2]);
      real v1 = (x0[2]*x1[0] + x0[0]*x2[2] + x1[2]*x2[0]) - (x2[2]*x1[0] + x2[0]*x0[2] + x1[2]*x0[0]);
      real v2 = (x0[0]*x1[1] + x0[1]*x2[0] + x1[0]*x2[1]) - (x2[0]*x1[1] + x2[1]*x0[0] + x1[0]*x0[1]);
  
      // Formula for volume from http://mathworld.wolfram.com 
      return  0.5 * sqrt(v0*v0 + v1*v1 + v2*v2);
    }
  else
    error("Only know how to volume (area) of a triangle when embedded in R^2 or R^3.");
 
  return 0.0;
}
//-----------------------------------------------------------------------------
real TriangleCell::diameter(const MeshEntity& triangle) const
{
  // Check that we get a triangle
  if ( triangle.dim() != 2 )
    error("Illegal mesh entity for computation of triangle diameter. Not a triangle.");

  // Get mesh geometry
  const MeshGeometry& geometry = triangle.mesh().geometry();

  // Only know how to compute the diameter when embedded in R^2 or R^3
  if ( geometry.dim() != 2 && geometry.dim() != 3 )
    error("Only know how to volume (area) of a triangle when embedded in R^2 or R^3.");

  // Get the coordinates of the three vertices
  const uint* vertices = triangle.entities(0);
  Point p0 = geometry.point(vertices[0]);
  Point p1 = geometry.point(vertices[1]);
  Point p2 = geometry.point(vertices[2]);

  // FIXME: Assuming 3D coordinates, could be more efficient if
  // FIXME: if we assumed 2D coordinates in 2D

  // Compute side lengths
  real a  = p1.distance(p2);
  real b  = p0.distance(p2);
  real c  = p0.distance(p1);

  // Formula for diameter (2*circumradius) from http://mathworld.wolfram.com
  return 0.5 * a*b*c / volume(triangle);
}
//-----------------------------------------------------------------------------
real TriangleCell::normal(const Cell& cell, uint facet, uint i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
Point TriangleCell::normal(const Cell& cell, uint facet) const
{
  // This is a trick to be allowed to initialize a facet from the cell
  Cell& c = const_cast<Cell&>(cell);
  
  // Create facet from the mesh and local facet number
  Facet f(c.mesh(), c.entities(1)[facet]);
  
  // The normal vector is currently only defined for a triangle in R^2
  if (c.mesh().geometry().dim() != 2)
    error("The normal vector is only defined when the triangle is in R^2");
  
  // Get global index of opposite vertex
  const uint v0 = cell.entities(0)[facet];
  
  // Get global index of vertices on the facet
  const uint v1 = f.entities(0)[0];
  const uint v2 = f.entities(0)[1];
  
  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();
  
  // Get the coordinates of the three vertices
  const real* p0 = geometry.x(v0);
  const real* p1 = geometry.x(v1);
  const real* p2 = geometry.x(v2);

  // Vector normal to facet
  Point n;
  n[0] = (p2[1] - p1[1]);
  n[1] = -(p2[0] - p1[0]);

  // Normalize
  n /= std::sqrt(n[0]*n[0] + n[1]*n[1]);

  // Flip direction of normal so it points outward
  if ( (n[0]*(p0[0] - p1[0]) + n[1]*(p0[1] - p1[1])) > 0 )
    n *= -1.0;

  return n;
}
//-----------------------------------------------------------------------------
dolfin::real TriangleCell::facetArea(const Cell& cell, uint facet) const
{
  // This is a trick to be allowed to initialize a facet from the cell
  Cell& c = const_cast<Cell&>(cell);
  
  // Create facet from the mesh and local facet number
  Facet f(c.mesh(), c.entities(1)[facet]);
  
  // Get global index of vertices on the facet
  const uint v0 = f.entities(0)[0];
  const uint v1 = f.entities(0)[1];
  
  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the two vertices
  const real* p0 = geometry.x(v0);
  const real* p1 = geometry.x(v1);

  // Compute distance between vertices
  real d = 0.0;
  for (uint i = 0; i < geometry.dim(); i++)
  {
    const real dp = p0[i] - p1[i];
    d += dp*dp;
  }
  
  return std::sqrt(d);
}
//-----------------------------------------------------------------------------
bool TriangleCell::intersects(const MeshEntity& triangle, const Point& p) const
{
  // Adapted from gts_point_is_in_triangle from GTS

  // Get mesh geometry
  const MeshGeometry& geometry = triangle.mesh().geometry();

  // Get global index of vertices of the triangle
  uint v0 = triangle.entities(0)[0];
  uint v1 = triangle.entities(0)[1];
  uint v2 = triangle.entities(0)[2];

  // Check orientation
  dolfin::uint vtmp;
  if(orientation((Cell&)triangle) == 1)
    {
      vtmp = v2;
      v2 = v1;
      v1 = vtmp;
    }

  // Get the coordinates of the three vertices
  const real* x0 = geometry.x(v0);
  const real* x1 = geometry.x(v1);
  const real* x2 = geometry.x(v2);

  real xcoordinates[3];
  real* x = xcoordinates;

  x[0] = p[0];
  x[1] = p[1];
  x[2] = p[2];

  real d1, d2, d3;

  // Test orientation of p w.r.t. each edge
  d1 = orient2d((double *)x0, (double *)x1, x);
  d2 = orient2d((double *)x1, (double *)x2, x);
  d3 = orient2d((double *)x2, (double *)x0, x);

  // FIXME: Need to check the predicates for correctness
  //   if(fabs(d1) == DOLFIN_EPS ||
  //      fabs(d2) == DOLFIN_EPS ||
  //      fabs(d3) == DOLFIN_EPS)
  //   {
  //     return true;
  //   }
  if(d1 < 0.0)
    return false;
  if(d2 < 0.0)
    return false;
  if(d3 < 0.0)
    return false;

  return true;
}
//-----------------------------------------------------------------------------
bool TriangleCell::intersects(const MeshEntity& tri,const Point& p1,const Point& p2) const
{
  // Adapted from gts_point_is_in_triangle from GTS

  // Get mesh geometry
  const MeshGeometry& geometry = tri.mesh().geometry();

  // Get global index of vertices of the triangle
  uint v0 = tri.entities(0)[0];
  uint v1 = tri.entities(0)[1];
  uint v2 = tri.entities(0)[2];

  // Check orientation
  dolfin::uint vtmp;
  if(orientation((Cell&)tri) == 1)
    {
      vtmp = v2;
      v2 = v1;
      v1 = vtmp;
    }

  // Get the coordinates of the three vertices
  const real* x0 = geometry.x(v0);
  const real* x1 = geometry.x(v1);
  const real* x2 = geometry.x(v2);

  // point a
  real p1coordinates[3];
  real* pa = p1coordinates;

  pa[0] = p1[0];
  pa[1] = p1[1];
  pa[2] = p1[2];

  // point b
  real p2coordinates[3];
  real* pb = p2coordinates;

  pb[0] = p2[0];
  pb[1] = p2[1];
  pb[2] = p2[2];

  real d1, d2, d3;

  // Test orientation of each vertex w.r.t. pa-pb
  d1 = orient2d((double *)pa, (double *)pb, (double*) x0);
  d2 = orient2d((double *)pa, (double *)pb, (double*) x1);
  d3 = orient2d((double *)pa, (double *)pb, (double*) x2);

  if( d1<0 && d2<0 && d3<0)
    return false;
  if( d1>0 && d2>0 && d3>0)
    return false;

  // Line pa-pb intersects triangle but both pa and pb are
  // on the negative side of x0-x1:
  d1 = orient2d((double*)x0, (double*)x1, (double*) pa);
  d2 = orient2d((double*)x0, (double*)x1, (double*) pb);

  if( d1<0 && d2<0)
    return false;

  // Line pa-pb intersects triangle but both pa and pb are
  // on the negative side of x1-x2:
  d1 = orient2d((double*)x1, (double*)x2, (double*) pa);
  d2 = orient2d((double*)x1, (double*)x2, (double*) pb);

  if( d1<0 && d2<0)
    return false;
  
  // Line pa-pb intersects triangle but both pa and pb are
  // on the negative side of x2-x0:
  d1 = orient2d((double*)x2, (double*)x0, (double*) pa);
  d2 = orient2d((double*)x2, (double*)x0, (double*) pb);

  if( d1<0 && d2<0)
    return false;
  
  return true;
}
//-----------------------------------------------------------------------------
std::string TriangleCell::description() const
{
  std::string s = "triangle (simplex of topological dimension 2)";
  return s;
}
//-----------------------------------------------------------------------------
dolfin::uint TriangleCell::findEdge(uint i, const Cell& cell) const
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
  error("Unable to find edge.");

  return 0;
}
//-----------------------------------------------------------------------------
