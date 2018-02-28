// Copyright (C) 2007-2008 Kristian B. Oelgaard
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PointCell.h"
#include "Cell.h"
#include "Facet.h"
#include "MeshEntity.h"
#include "Vertex.h"
#include <dolfin/geometry/CollisionPredicates.h>
#include <dolfin/log/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
std::size_t PointCell::dim() const { return 0; }
//-----------------------------------------------------------------------------
std::size_t PointCell::num_entities(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  default:
    log::dolfin_error("PointCell.cpp",
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
    log::dolfin_error("PointCell.cpp",
                 "extract number of vertices of given dimension in cell",
                 "Illegal topological dimension %d for point", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
void PointCell::create_entities(boost::multi_array<std::int32_t, 2>& e,
                                std::size_t dim, const std::int32_t* v) const
{
  log::dolfin_error("PointCell.cpp", "create entities",
               "Entities on a point cell are not defined");
}
//-----------------------------------------------------------------------------
double PointCell::volume(const MeshEntity& triangle) const
{
  log::dolfin_error("PointCell.cpp", "compute volume of cell",
               "Volume of a point cell is not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------

double PointCell::circumradius(const MeshEntity& point) const
{
  log::dolfin_error("PointCell.cpp", "find circumradious of cell",
               "Circumradius of a point cell is not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::squared_distance(const Cell& cell, const geometry::Point& point) const
{
  dolfin_not_implemented();
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::normal(const Cell& cell, std::size_t facet,
                         std::size_t i) const
{
  log::dolfin_error("PointCell.cpp", "find component of normal vector of cell",
               "Component %d of normal of a point cell is not defined", i);
  return 0.0;
}
//-----------------------------------------------------------------------------
geometry::Point PointCell::normal(const Cell& cell, std::size_t facet) const
{
  log::dolfin_error("PointCell.cpp", "find normal vector of cell",
               "Normal vector of a point cell is not defined");
  return geometry::Point();
}
//-----------------------------------------------------------------------------
geometry::Point PointCell::cell_normal(const Cell& cell) const
{
  log::dolfin_error("PointCell.cpp", "compute cell normal",
               "Normal vector of a point cell is not defined");
  return geometry::Point();
}
//-----------------------------------------------------------------------------
double PointCell::facet_area(const Cell& cell, std::size_t facet) const
{
  log::dolfin_error("PointCell.cpp", "find facet area of cell",
               "Facet area of a point cell is not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
void PointCell::order(
    Cell& cell,
    const std::vector<std::int64_t>& local_to_global_vertex_indices) const
{
  log::dolfin_error("PointCell.cpp", "order cell",
               "Ordering of a point cell is not defined");
}
//-----------------------------------------------------------------------------
bool PointCell::collides(const Cell& cell, const geometry::Point& point) const
{
  return geometry::CollisionPredicates::collides(cell, point);
}
//-----------------------------------------------------------------------------
bool PointCell::collides(const Cell& cell, const MeshEntity& entity) const
{
  return geometry::CollisionPredicates::collides(cell, entity);
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
  log::dolfin_error("PointCell.cpp", "find edge",
               "Edges are not defined for a point cell");
  return 0;
}
//-----------------------------------------------------------------------------
