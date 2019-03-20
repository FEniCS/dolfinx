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
#include <spdlog/spdlog.h>

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
    spdlog::error("PointCell.cpp",
                  "extract number of entities of given dimension in cell",
                  "Illegal topological dimension %d for point", dim);
    throw std::runtime_error("Illegal dimension");
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
    spdlog::error("PointCell.cpp",
                  "extract number of vertices of given dimension in cell",
                  "Illegal topological dimension %d for point", dim);
    throw std::runtime_error("Illegal dimension");
  }

  return 0;
}
//-----------------------------------------------------------------------------
void PointCell::create_entities(boost::multi_array<std::int32_t, 2>& e,
                                std::size_t dim, const std::int32_t* v) const
{
  spdlog::error("PointCell.cpp", "create entities",
                "Entities on a point cell are not defined");
  throw std::runtime_error("Not defined");
}
//-----------------------------------------------------------------------------
double PointCell::volume(const MeshEntity& triangle) const
{
  spdlog::error("PointCell.cpp", "compute volume of cell",
                "Volume of a point cell is not defined");
  throw std::runtime_error("Not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------

double PointCell::circumradius(const MeshEntity& point) const
{
  spdlog::error("PointCell.cpp", "find circumradious of cell",
                "Circumradius of a point cell is not defined");
  throw std::runtime_error("Not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::squared_distance(const Cell& cell,
                                   const geometry::Point& point) const
{
  spdlog::error("PointCell.cpp",
                "Squared distance of a point cell is not defined");
  throw std::runtime_error("Not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::normal(const Cell& cell, std::size_t facet,
                         std::size_t i) const
{
  spdlog::error("PointCell.cpp", "find component of normal vector of cell",
                "Component %d of normal of a point cell is not defined", i);
  throw std::runtime_error("Not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
geometry::Point PointCell::normal(const Cell& cell, std::size_t facet) const
{
  spdlog::error("PointCell.cpp", "find normal vector of cell",
                "Normal vector of a point cell is not defined");
  throw std::runtime_error("Not defined");
  return geometry::Point();
}
//-----------------------------------------------------------------------------
geometry::Point PointCell::cell_normal(const Cell& cell) const
{
  spdlog::error("PointCell.cpp", "compute cell normal",
                "Normal vector of a point cell is not defined");
  throw std::runtime_error("Not defined");
  return geometry::Point();
}
//-----------------------------------------------------------------------------
double PointCell::facet_area(const Cell& cell, std::size_t facet) const
{
  spdlog::error("PointCell.cpp", "find facet area of cell",
                "Facet area of a point cell is not defined");
  throw std::runtime_error("Not defined");
  return 0.0;
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
  spdlog::error("PointCell.cpp", "find edge",
                "Edges are not defined for a point cell");
  throw std::runtime_error("Not defined");
  return 0;
}
//-----------------------------------------------------------------------------
