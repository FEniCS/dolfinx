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
CellTypeOld::CellTypeOld(CellType cell_type, CellType facet_type)
    : type(cell_type), facet_type(facet_type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CellTypeOld* CellTypeOld::create(CellType type)
{
  switch (type)
  {
  case CellType::point:
    return new PointCell();
  case CellType::interval:
    return new IntervalCell();
  case CellType::triangle:
    return new TriangleCell();
  case CellType::tetrahedron:
    return new TetrahedronCell();
  case CellType::quadrilateral:
    return new QuadrilateralCell();
  case CellType::hexahedron:
    return new HexahedronCell();
  default:
    throw std::runtime_error("Unknown cell type");
  }

  return nullptr;
}
//-----------------------------------------------------------------------------
CellType CellTypeOld::entity_type(int i) const
{
  const int dim = mesh::cell_dim(this->type);
  if (i == dim)
    return type;
  else if (i == dim - 1)
    return facet_type;
  else if (i == 1)
    return CellType::interval;

  return CellType::point;
}
//-----------------------------------------------------------------------------
double CellTypeOld::h(const MeshEntity& entity) const
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
double CellTypeOld::inradius(const Cell& cell) const
{
  // Check cell type
  if (type != CellType::interval and type != CellType::triangle
      and type != CellType::tetrahedron)
  {
    throw std::runtime_error(
        "inradius function not implemented for non-simplicial cells");
  }

  // Pick dim
  const int d = mesh::cell_dim(this->type);

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
double CellTypeOld::radius_ratio(const Cell& cell) const
{
  const double r = inradius(cell);

  // Handle degenerate case
  if (r == 0.0)
    return 0.0;
  else
    return mesh::cell_dim(this->type) * r / circumradius(cell);
}
//-----------------------------------------------------------------------------