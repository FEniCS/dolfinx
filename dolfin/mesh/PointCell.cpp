// Copyright (C) 2007-2008 Kristian B. Oelgaard
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
// Modified by Anders Logg, 2008.
// Modified by Kristoffer Sleim, 2008.
//
// First added:  2007-12-12
// Last changed: 2010-01-19

#include <dolfin/log/dolfin_log.h>
#include "Cell.h"
#include "Facet.h"
#include "MeshEditor.h"
#include "Vertex.h"
#include "PointCell.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint PointCell::dim() const
{
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint PointCell::num_entities(uint dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  default:
    error("Illegal topological dimension %d for point.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint PointCell::num_vertices(uint dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  default:
    error("Illegal topological dimension %d for point.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint PointCell::orientation(const Cell& cell) const
{
  error("PointCell::orientation() not defined.");
  return 0;
}
//-----------------------------------------------------------------------------
void PointCell::create_entities(uint** e, uint dim, const uint* v) const
{
  error("PointCell::create_entities() don't know how to create entities on a point.");
}
//-----------------------------------------------------------------------------
void PointCell::refine_cell(Cell& cell, MeshEditor& editor,
                          uint& current_cell) const
{
  error("PointCell::refine_cell() not defined.");
}
//-----------------------------------------------------------------------------
double PointCell::volume(const MeshEntity& triangle) const
{
  error("PointCell::volume() not defined.");
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::diameter(const MeshEntity& triangle) const
{
  error("PointCell::diameter() not defined.");
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::normal(const Cell& cell, uint facet, uint i) const
{
  error("PointCell::normal() not defined.");
  return 0.0;
}
//-----------------------------------------------------------------------------
Point PointCell::normal(const Cell& cell, uint facet) const
{
  error("PointCell::normal() not defined.");
  Point p;
  return p;
}
//-----------------------------------------------------------------------------
double PointCell::facet_area(const Cell& cell, uint facet) const
{
  error("PointCell::facet_aread() not defined.");
  return 0.0;
}
//-----------------------------------------------------------------------------
void PointCell::order(Cell& cell,
                      const MeshFunction<uint>* global_vertex_indices) const
{
  error("PointCell::order() not defined.");
}
//-----------------------------------------------------------------------------
std::string PointCell::description(bool plural) const
{
  if (plural)
    return "points";
  return "points";
}
//-----------------------------------------------------------------------------
dolfin::uint PointCell::find_edge(uint i, const Cell& cell) const
{
  error("PointCell::find_edge() not defined.");
  return 0;
}
//-----------------------------------------------------------------------------
