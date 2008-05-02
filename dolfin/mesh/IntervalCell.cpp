// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-05
// Last changed: 2007-07-20
//
// Modified by Kristian Oelgaard 2007.
//
// Rename of the former Interval.cpp
//

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
  MeshTopology& topology = cell.mesh().topology();

  // Sort local vertices in ascending order, connectivity 1 - 0
  if ( topology(1, 0).size() > 0 )
  {
    uint* cell_vertices = cell.entities(0);
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
real IntervalCell::volume(const MeshEntity& interval) const
{
  // Check that we get an interval
  if ( interval.dim() != 1 )
    error("Illegal mesh entity for computation of interval volume (length). Not an interval.");

  // Get mesh geometry
  const MeshGeometry& geometry = interval.mesh().geometry();

  // Get the coordinates of the two vertices
  const uint* vertices = interval.entities(0);
  const real* x0 = geometry.x(vertices[0]);
  const real* x1 = geometry.x(vertices[1]);
  
  // Compute length of interval (line segment)
  real sum = 0.0;
  for (uint i = 0; i < geometry.dim(); ++i)
  {
    const real dx = x1[i] - x0[i];
    sum += dx*dx;
  }

  return std::sqrt(sum);
}
//-----------------------------------------------------------------------------
real IntervalCell::diameter(const MeshEntity& interval) const
{
  // Check that we get an interval
  if ( interval.dim() != 1 )
    error("Illegal mesh entity for computation of interval diameter. Not an interval.");

  // Diameter is same as volume for interval (line segment)
  return volume(interval);
}
//-----------------------------------------------------------------------------
real IntervalCell::normal(const Cell& cell, uint facet, uint i) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // The normal vector is currently only defined for an interval in R^1
  if ( geometry.dim() != 1 )
    error("The normal vector is only defined when the interval is in R^1");

  // Currently only the x coordinate can be returned
  if ( i != 0 )
    error("IntervalCell::normal() can currently only return the x-component.");

  // Get the two vertices as points
  const uint* vertices = cell.entities(0);
  Point p0 = geometry.point(vertices[0]);
  Point p1 = geometry.point(vertices[1]);

  // Compute normal for the two facet
  if (facet == 0)
  {
    // Represent interval as a vector
    Point iv = p0-p1;

    // Divide by norm of vector and get component
    return (iv/iv.norm()).x();
  }
  else if (facet == 1)
  {
    // Represent interval as a vector
    Point iv = p1-p0;

    // Divide by norm of vector and get component
    return (iv/iv.norm()).x();
  }
  else
    error("Local facet number must be either 0 or 1");

  return 0.0;
}
//-----------------------------------------------------------------------------
bool IntervalCell::intersects(const MeshEntity& interval, const Point& p) const
{
  // FIXME: Not implemented
  error("Interval::intersects() not implemented");

  return false;
}
//-----------------------------------------------------------------------------
bool IntervalCell::intersects(const MeshEntity& interval, const Point& p1, const Point& p2) const
{
  // FIXME: Not implemented
  error("Interval::intersects() not implemented");

  return false;
}
//-----------------------------------------------------------------------------
std::string IntervalCell::description() const
{
  std::string s = "interval (simplex of topological dimension 1)";
  return s;
}
//-----------------------------------------------------------------------------
