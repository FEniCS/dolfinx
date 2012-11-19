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
#include "MeshEntity.h"
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
    dolfin_error("PointCell.cpp",
                 "extract number of entities of given dimension in cell",
                 "Illegal topological dimension %d for point", dim);
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
    dolfin_error("PointCell.cpp",
                 "extract number of vertices of given dimension in cell",
                 "Illegal topological dimension %d for point", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint PointCell::orientation(const Cell& cell) const
{
  dolfin_error("PointCell.cpp",
               "find orientation",
               "Orientation not defined for point cell");
  return 0;
}
//-----------------------------------------------------------------------------
void PointCell::create_entities(std::vector<std::vector<std::size_t> >& e,
                                uint dim,
                                const std::size_t* v) const
{
  dolfin_error("PointCell.cpp",
               "create entities",
               "Entities on a point cell are not defined");
}
//-----------------------------------------------------------------------------
void PointCell::refine_cell(Cell& cell, MeshEditor& editor,
                          std::size_t& current_cell) const
{
  dolfin_error("PointCell.cpp",
               "refine cell",
               "Refinement of a point cell is not defined");
}
//-----------------------------------------------------------------------------
double PointCell::volume(const MeshEntity& triangle) const
{
  dolfin_error("PointCell.cpp",
               "compute volume of cell",
               "Volume of a point cell is not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::diameter(const MeshEntity& triangle) const
{
  dolfin_error("PointCell.cpp",
               "find diameter of cell",
               "Diameter of a point cell is not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::normal(const Cell& cell, uint facet, uint i) const
{
  dolfin_error("PointCell.cpp",
               "find component of normal vector of cell",
               "Component %d of normal of a point cell is not defined", i);
  return 0.0;
}
//-----------------------------------------------------------------------------
Point PointCell::normal(const Cell& cell, uint facet) const
{
  dolfin_error("PointCell.cpp",
               "find normal vector of cell",
               "Normal vector of a point cell is not defined");
  Point p;
  return p;
}
//-----------------------------------------------------------------------------
double PointCell::facet_area(const Cell& cell, uint facet) const
{
  dolfin_error("PointCell.cpp",
               "find facet area of cell",
               "Facet area of a point cell is not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
void PointCell::order(Cell& cell,
                 const std::vector<std::size_t>& local_to_global_vertex_indices) const
{
  dolfin_error("PointCell.cpp",
               "order cell",
               "Ordering of a point cell is not defined");
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
  dolfin_error("PointCell.cpp",
               "find edge",
               "Edges are not defined for a point cell");
  return 0;
}
//-----------------------------------------------------------------------------
