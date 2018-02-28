// Copyright (C) 2006-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Facet.h"
#include "Cell.h"
#include "IntervalCell.h"
#include "TriangleCell.h"
#include <dolfin/geometry/Point.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
geometry::Point Facet::normal() const
{
  const std::size_t D = _mesh->topology().dim();
  _mesh->init(D - 1);
  _mesh->init(D - 1, D);
  dolfin_assert(_mesh->ordered());

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(D)[0]);

  // Get local index of facet with respect to the cell
  const std::size_t local_facet = cell.index(*this);

  return cell.normal(local_facet);
}
//-----------------------------------------------------------------------------
double Facet::squared_distance(const geometry::Point& point) const
{
  if (_dim == 1)
  {
    // Extract vertices
    const MeshGeometry& geometry = _mesh->geometry();
    const std::int32_t* vertices = entities(0);
    const geometry::Point a = geometry.point(vertices[0]);
    const geometry::Point b = geometry.point(vertices[1]);

    // Compute squared distance
    return IntervalCell::squared_distance(point, a, b);
  }
  else if (_dim == 2)
  {
    // Extract vertices
    const MeshGeometry& geometry = _mesh->geometry();
    const std::int32_t* vertices = entities(0);
    const geometry::Point a = geometry.point(vertices[0]);
    const geometry::Point b = geometry.point(vertices[1]);
    const geometry::Point c = geometry.point(vertices[2]);

    // Compute squared distance
    return TriangleCell::squared_distance(point, a, b, c);
  }

  dolfin_error("Facet.cpp", "compute (squared) distance to facet",
               "Not implemented for facets of dimension %d", _dim);

  return 0.0;
}
//-----------------------------------------------------------------------------
bool Facet::exterior() const
{
  const std::size_t D = _mesh->topology().dim();
  if (this->num_global_entities(D) == 1)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
