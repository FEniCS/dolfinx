// Copyright (C) 2006-2008 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Kristian Oelgaard, 2007.
// Modified by Kristoffer Selim, 2008.
//
// First added:  2006-06-05
// Last changed: 2010-03-02

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
dolfin::uint IntervalCell::num_entities(uint dim) const
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
dolfin::uint IntervalCell::num_vertices(uint dim) const
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
void IntervalCell::create_entities(uint** e, uint dim, const uint* v) const
{
  // We don't need to create any entities
  error("Don't know how to create entities of topological dimension %d.", dim);
}
//-----------------------------------------------------------------------------
void IntervalCell::refine_cell(Cell& cell, MeshEditor& editor,
                              uint& current_cell) const
{
  // Get vertices and edges
  const uint* v = cell.entities(0);
  const uint* e = cell.entities(1);
  assert(v);
  assert(e);

  // Get offset for new vertex indices
  const uint offset = cell.mesh().num_vertices();

  // Compute indices for the three new vertices
  const uint v0 = v[0];
  const uint v1 = v[1];
  const uint e0 = offset + e[0];

  // Add the two new cells
  editor.add_cell(current_cell++, v0, e0);
  editor.add_cell(current_cell++, e0, v1);
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
double IntervalCell::facet_area(const Cell& cell, uint facet) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
void IntervalCell::order(Cell& cell,
                         const MeshFunction<uint>* global_vertex_indices) const
{
  // Sort i - j for i > j: 1 - 0

  // Get mesh topology
  MeshTopology& topology = const_cast<MeshTopology&>(cell.mesh().topology());

  // Sort local vertices in ascending order, connectivity 1 - 0
  if (topology(1, 0).size() > 0)
  {
    uint* cell_vertices = const_cast<uint*>(cell.entities(0));
    sort_entities(2, cell_vertices, global_vertex_indices);
  }
}
//-----------------------------------------------------------------------------
std::string IntervalCell::description(bool plural) const
{
  if (plural)
    return "intervals";
  return "interval";
}
//-----------------------------------------------------------------------------
