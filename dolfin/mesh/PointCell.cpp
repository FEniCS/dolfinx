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
// Modified by Anders Logg 2008-2014
// Modified by Kristoffer Sleim 2008
// Modified by August Johansson 2014
//
// First added:  2007-12-12
// Last changed: 2016-05-05

#include <dolfin/log/log.h>
#include <dolfin/geometry/CollisionDetection.h>
#include <dolfin/geometry/IntersectionTriangulation.h>
#include "Cell.h"
#include "Facet.h"
#include "MeshEditor.h"
#include "MeshEntity.h"
#include "Vertex.h"
#include "PointCell.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::size_t PointCell::dim() const
{
  return 0;
}
//-----------------------------------------------------------------------------
std::size_t PointCell::num_entities(std::size_t dim) const
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
std::size_t PointCell::num_vertices(std::size_t dim) const
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
std::size_t PointCell::orientation(const Cell& cell) const
{
  dolfin_error("PointCell.cpp",
               "find orientation",
               "Orientation not defined for point cell");
  return 0;
}
//-----------------------------------------------------------------------------
void PointCell::create_entities(boost::multi_array<unsigned int, 2>& e,
                                std::size_t dim,
                                const unsigned int* v) const
{
  dolfin_error("PointCell.cpp",
               "create entities",
               "Entities on a point cell are not defined");
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

double PointCell::circumradius(const MeshEntity& point) const
{
  dolfin_error("PointCell.cpp",
               "find circumradious of cell",
               "Circumradius of a point cell is not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::squared_distance(const Cell& cell, const Point& point) const
{
  dolfin_not_implemented();
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::normal(const Cell& cell, std::size_t facet,
                         std::size_t i) const
{
  dolfin_error("PointCell.cpp",
               "find component of normal vector of cell",
               "Component %d of normal of a point cell is not defined", i);
  return 0.0;
}
//-----------------------------------------------------------------------------
Point PointCell::normal(const Cell& cell, std::size_t facet) const
{
  dolfin_error("PointCell.cpp",
               "find normal vector of cell",
               "Normal vector of a point cell is not defined");
  Point p;
  return p;
}
//-----------------------------------------------------------------------------
Point PointCell::cell_normal(const Cell& cell) const
{
  dolfin_error("PointCell.cpp",
               "compute cell normal",
               "Normal vector of a point cell is not defined");
  Point p;
  return p;
}
//-----------------------------------------------------------------------------
double PointCell::facet_area(const Cell& cell, std::size_t facet) const
{
  dolfin_error("PointCell.cpp",
               "find facet area of cell",
               "Facet area of a point cell is not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
void PointCell::order(
  Cell& cell,
  const std::vector<std::size_t>& local_to_global_vertex_indices) const
{
  dolfin_error("PointCell.cpp",
               "order cell",
               "Ordering of a point cell is not defined");
}
//-----------------------------------------------------------------------------
bool PointCell::collides(const Cell& cell, const Point& point) const
{
  return CollisionDetection::collides(cell, point);
}
//-----------------------------------------------------------------------------
bool PointCell::collides(const Cell& cell, const MeshEntity& entity) const
{
  return CollisionDetection::collides(cell, entity);
}
//-----------------------------------------------------------------------------
std::string PointCell::description(bool plural) const
{
  if (plural)
    return "points";
  return "points";
}
//-----------------------------------------------------------------------------
std::size_t PointCell::find_edge(std::size_t i, const Cell& cell) const
{
  dolfin_error("PointCell.cpp",
               "find edge",
               "Edges are not defined for a point cell");
  return 0;
}
//-----------------------------------------------------------------------------
