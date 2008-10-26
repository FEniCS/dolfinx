// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian Oelgaard, 2007.
// Modified by Kristoffer Selim, 2008.
//
// First added:  2006-06-05
// Last changed: 2008-10-08

#include <algorithm>
#include <dolfin/log/dolfin_log.h>
#include "Cell.h"
#include "MeshEditor.h"
#include "MeshGeometry.h"
#include "IntervalCell.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint IntervalCell::dim() const
{
  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint IntervalCell::numEntities(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 2; // vertices
  case 1:
    return 1; // cells
  default:
    error("Illegal topological dimension %d for interval.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint IntervalCell::numVertices(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // cells
  default:
    error("Illegal topological dimension %d for interval.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint IntervalCell::orientation(const Cell& cell) const
{
  Point v01 = Point(cell.entities(0)[1]) - Point(cell.entities(0)[0]);
  Point n(-v01.y(), v01.x());

  return ( n.dot(v01) < 0.0 ? 1 : 0 );
}
//-----------------------------------------------------------------------------
void IntervalCell::createEntities(uint** e, uint dim, const uint* v) const
{
  // We don't need to create any entities
  error("Don't know how to create entities of topological dimension %d.", dim);
}
//-----------------------------------------------------------------------------
void IntervalCell::orderEntities(Cell& cell) const
{
  // Sort i - j for i > j: 1 - 0

  // Get mesh topology
  MeshTopology& topology = const_cast<MeshTopology&>(cell.mesh().topology());

  // Sort local vertices in ascending order, connectivity 1 - 0
  if ( topology(1, 0).size() > 0 )
  {
    uint* cell_vertices = const_cast<uint*>(cell.entities(0));
    std::sort(cell_vertices, cell_vertices + 2);
  }
}
//-----------------------------------------------------------------------------
void IntervalCell::refineCell(Cell& cell, MeshEditor& editor,
			  uint& current_cell) const
{
  // Get vertices and edges
  const uint* v = cell.entities(0);
  const uint* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Get offset for new vertex indices
  const uint offset = cell.mesh().numVertices();

  // Compute indices for the three new vertices
  const uint v0 = v[0];
  const uint v1 = v[1];
  const uint e0 = offset + e[0];
  
  // Add the two new cells
  editor.addCell(current_cell++, v0, e0);
  editor.addCell(current_cell++, e0, v1);
}
//-----------------------------------------------------------------------------
double IntervalCell::volume(const MeshEntity& interval) const
{
  // Check that we get an interval
  if ( interval.dim() != 1 )
    error("Illegal mesh entity for computation of interval volume (length). Not an interval.");

  // Get mesh geometry
  const MeshGeometry& geometry = interval.mesh().geometry();

  // Get the coordinates of the two vertices
  const uint* vertices = interval.entities(0);
  const double* x0 = geometry.x(vertices[0]);
  const double* x1 = geometry.x(vertices[1]);
  
  // Compute length of interval (line segment)
  double sum = 0.0;
  for (uint i = 0; i < geometry.dim(); ++i)
  {
    const double dx = x1[i] - x0[i];
    sum += dx*dx;
  }

  return std::sqrt(sum);
}
//-----------------------------------------------------------------------------
double IntervalCell::diameter(const MeshEntity& interval) const
{
  // Check that we get an interval
  if ( interval.dim() != 1 )
    error("Illegal mesh entity for computation of interval diameter. Not an interval.");

  // Diameter is same as volume for interval (line segment)
  return volume(interval);
}
//-----------------------------------------------------------------------------
double IntervalCell::normal(const Cell& cell, uint facet, uint i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
Point IntervalCell::normal(const Cell& cell, uint facet) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // The normal vector is currently only defined for an interval in R^1
  if ( geometry.dim() != 1 )
    error("The normal vector is only defined when the interval is in R^1");

  // Get the two vertices as points
  const uint* vertices = cell.entities(0);
  Point p0 = geometry.point(vertices[0]);
  Point p1 = geometry.point(vertices[1]);

  // Compute normal
  Point n = p0 - p1;
  if (facet == 1)
    n *= -1.0;

  // Normalize
  n /= n.norm();

  return n;
}
//-----------------------------------------------------------------------------
double IntervalCell::facetArea(const Cell& cell, uint facet) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
bool IntervalCell::intersects(const MeshEntity& entity, const Point& p) const
{
  error("Not implemented.");
  return false;
}
//-----------------------------------------------------------------------------
bool IntervalCell::intersects(const MeshEntity& entity,
                              const Point& p0, const Point& p1) const
{
  error("Not implemented.");
  return false;
}
//-----------------------------------------------------------------------------
bool IntervalCell::intersects(const MeshEntity& entity, const Cell& cell) const

{
  error("Not implemented.");
  return false;
} 
//-----------------------------------------------------------------------------
std::string IntervalCell::description() const
{
  std::string s = "interval (simplex of topological dimension 1)";
  return s;
}
//-----------------------------------------------------------------------------
