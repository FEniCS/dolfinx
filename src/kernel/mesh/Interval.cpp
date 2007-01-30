// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-12-06

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/MeshEditor.h>
#include <dolfin/MeshGeometry.h>
#include <dolfin/Interval.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint Interval::dim() const
{
  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint Interval::numEntities(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 2; // vertices
  case 1:
    return 1; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for interval.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint Interval::numVertices(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for interval.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint Interval::alignment(const Cell& cell, uint dim, uint e) const
{
  dolfin_error("Unable to compute alignment for entity of dimension %d for interval.");
  return 0;
}
//-----------------------------------------------------------------------------
void Interval::createEntities(uint** e, uint dim, const uint v[]) const
{
  // We don't need to create any entities
  dolfin_error1("Don't know how to create entities of topological dimension %d.", dim);
}
//-----------------------------------------------------------------------------
void Interval::orderEntities(Cell& cell) const
{
  // FIXME: Implement
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
void Interval::refineCell(Cell& cell, MeshEditor& editor,
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
real Interval::volume(const MeshEntity& interval) const
{
  // Check that we get an interval
  if ( interval.dim() != 1 )
    dolfin_error("Illegal mesh entity for computation of interval volume (length). Not an interval.");

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
real Interval::diameter(const MeshEntity& interval) const
{
  // Check that we get an interval
  if ( interval.dim() != 1 )
    dolfin_error("Illegal mesh entity for computation of interval diameter. Not an interval.");

  // Diameter is same as volume for interval (line segment)
  return volume(interval);
}
//-----------------------------------------------------------------------------
real Interval::normal(const Cell& cell, uint facet, uint i) const
{
  dolfin_error("Not implemented. Please fix this Kristian. ;-)");
  return 0.0;
}
//-----------------------------------------------------------------------------
bool Interval::intersects(const MeshEntity& interval, const Point& p) const
{
  // FIXME: Not implemented
  dolfin_error("Interval::intersects() not implemented");

  return false;
}
//-----------------------------------------------------------------------------
std::string Interval::description() const
{
  std::string s = "interval (simplex of topological dimension 1)";
  return s;
}
//-----------------------------------------------------------------------------
