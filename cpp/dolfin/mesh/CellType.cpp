// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CellType.h"
#include "Cell.h"
#include "HexahedronCell.h"
#include "IntervalCell.h"
#include "PointCell.h"
#include "QuadrilateralCell.h"
#include "TetrahedronCell.h"
#include "TriangleCell.h"
#include "Vertex.h"
#include <algorithm>
#include <array>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
CellType::CellType(Type cell_type, Type facet_type)
    : type(cell_type), facet_type(facet_type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CellType* CellType::create(Type type)
{
  switch (type)
  {
  case Type::point:
    return new PointCell();
  case Type::interval:
    return new IntervalCell();
  case Type::triangle:
    return new TriangleCell();
  case Type::tetrahedron:
    return new TetrahedronCell();
  case Type::quadrilateral:
    return new QuadrilateralCell();
  case Type::hexahedron:
    return new HexahedronCell();
  default:
    throw std::runtime_error("Unknown cell type");
  }

  return nullptr;
}
//-----------------------------------------------------------------------------
CellType::Type CellType::string2type(std::string type)
{
  if (type == "point")
    return Type::point;
  else if (type == "interval")
    return Type::interval;
  else if (type == "triangle")
    return Type::triangle;
  else if (type == "tetrahedron")
    return Type::tetrahedron;
  else if (type == "quadrilateral")
    return Type::quadrilateral;
  else if (type == "hexahedron")
    return Type::hexahedron;
  else
    throw std::runtime_error("Unknown cell type (" + type + ")");

  // Should no reach this point
  return Type::interval;
}
//-----------------------------------------------------------------------------
std::string CellType::type2string(Type type)
{
  switch (type)
  {
  case Type::point:
    return "point";
  case Type::interval:
    return "interval";
  case Type::triangle:
    return "triangle";
  case Type::tetrahedron:
    return "tetrahedron";
  case Type::quadrilateral:
    return "quadrilateral";
  case Type::hexahedron:
    return "hexahedron";
  default:
    throw std::runtime_error("Unknown cell type");
  }

  return "";
}
//-----------------------------------------------------------------------------
CellType::Type CellType::entity_type(std::size_t i) const
{
  if (i == dim())
    return type;
  else if (i == dim() - 1)
    return facet_type;
  else if (i == 1)
    return Type::interval;
  return Type::point;
}
//-----------------------------------------------------------------------------
double CellType::h(const MeshEntity& entity) const
{
  // Get mesh geometry
  const Geometry& geometry = entity.mesh().geometry();

  // Get number of cell vertices
  const int num_vertices = entity.num_entities(0);

  // Get the coordinates (Points) of the vertices
  const std::int32_t* vertices = entity.entities(0);
  assert(vertices);
  std::array<Eigen::Vector3d, 8> points;
  assert(num_vertices <= 8);
  for (int i = 0; i < num_vertices; ++i)
    points[i] = geometry.x(vertices[i]);

  // Get maximum edge length
  double h = 0.0;
  for (int i = 0; i < num_vertices; ++i)
  {
    for (int j = i + 1; j < num_vertices; ++j)
      h = std::max(h, (points[i] - points[j]).norm());
  }

  return h;
}
//-----------------------------------------------------------------------------
double CellType::inradius(const Cell& cell) const
{
  // Check cell type
  if (type != Type::interval and type != Type::triangle
      and type != Type::tetrahedron)
  {
    throw std::runtime_error(
        "inradius function not implemented for non-simplicial cells");
  }

  // Pick dim
  const size_t d = dim();

  // Compute volume
  const double V = volume(cell);

  // Handle degenerate case
  if (V == 0.0)
    return 0.0;

  // Compute total area of facets
  double A = 0;
  for (std::size_t i = 0; i <= d; i++)
    A += facet_area(cell, i);

  // See Jonathan Richard Shewchuk: What Is a Good Linear Finite
  // Element?, online:
  // http://www.cs.berkeley.edu/~jrs/papers/elemj.pdf
  return d * V / A;
}
//-----------------------------------------------------------------------------
double CellType::radius_ratio(const Cell& cell) const
{
  const double r = inradius(cell);

  // Handle degenerate case
  if (r == 0.0)
    return 0.0;
  else
    return dim() * r / circumradius(cell);
}
//-----------------------------------------------------------------------------