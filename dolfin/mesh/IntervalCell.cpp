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
// Modified by Kristian Oelgaard, 2007.
// Modified by Kristoffer Selim, 2008.
// Modified by Marie E. Rognes, 2011.
//
// First added:  2006-06-05
// Last changed: 2011-11-14

#include <algorithm>
#include <dolfin/log/dolfin_log.h>
#include "Cell.h"
#include "MeshEditor.h"
#include "MeshEntity.h"
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
  switch (dim)
  {
  case 0:
    return 2; // vertices
  case 1:
    return 1; // cells
  default:
    dolfin_error("IntervalCell.cpp",
                 "access number of entities of interval cell",
                 "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint IntervalCell::num_vertices(uint dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // cells
  default:
    dolfin_error("IntervalCell.cpp",
                 "access number of vertices for subsimplex of interval cell",
                 "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint IntervalCell::orientation(const Cell& cell) const
{
  Point v01 = Point(cell.entities(0)[1]) - Point(cell.entities(0)[0]);
  Point n(-v01.y(), v01.x());

  return (n.dot(v01) < 0.0 ? 1 : 0);
}
//-----------------------------------------------------------------------------
void IntervalCell::create_entities(std::vector<std::vector<uint> >& e,
                                   uint dim, const uint* v) const
{
  // We don't need to create any entities
  dolfin_error("IntervalCell.cpp",
               "create entities of interval cell",
               "Don't know how to create entities of topological dimension %d", dim);
}
//-----------------------------------------------------------------------------
void IntervalCell::refine_cell(Cell& cell, MeshEditor& editor,
                              uint& current_cell) const
{
  // Get vertices
  const uint* v = cell.entities(0);
  dolfin_assert(v);

  // Get offset for new vertex indices
  const uint offset = cell.mesh().num_vertices();

  // Compute indices for the three new vertices
  const uint v0 = v[0];
  const uint v1 = v[1];
  const uint e0 = offset + cell.index();

  // Add the two new cells
  std::vector<uint> new_cell(2);

  new_cell[0] = v0; new_cell[1] = e0;
  editor.add_cell(current_cell++, new_cell);

  new_cell[0] = e0; new_cell[1] = v1;
  editor.add_cell(current_cell++, new_cell);
}
//-----------------------------------------------------------------------------
double IntervalCell::volume(const MeshEntity& interval) const
{
  // Check that we get an interval
  if (interval.dim() != 1)
  {
    dolfin_error("IntervalCell.cpp",
                 "compute volume (length) of interval cell",
                 "Illegal mesh entity, not an interval");
  }

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
  if (interval.dim() != 1)
  {
    dolfin_error("IntervalCell.cpp",
                 "compute diameter of interval cell",
                 "Illegal mesh entity, not an interval");
  }

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
  return 1.0;
}
//-----------------------------------------------------------------------------
void IntervalCell::order(Cell& cell,
                         const MeshFunction<uint>* global_vertex_indices) const
{
  // Sort i - j for i > j: 1 - 0

  // Get mesh topology
  MeshTopology& topology = const_cast<MeshTopology&>(cell.mesh().topology());

  // Sort local vertices in ascending order, connectivity 1 - 0
  if (!topology(1, 0).empty())
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
