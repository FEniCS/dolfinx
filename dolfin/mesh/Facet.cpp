// Copyright (C) 2006-2015 Anders Logg
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
// Modified by Garth N. Wells 2011
// Modified by Martin Alnaes, 2015

#include <dolfin/geometry/Point.h>
#include "IntervalCell.h"
#include "TriangleCell.h"
#include "Cell.h"
#include "MeshTopology.h"
#include "Facet.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
double Facet::normal(std::size_t i) const
{
  const std::size_t D = _mesh->topology().dim();
  _mesh->init(D - 1);
  _mesh->init(D - 1, D);
  dolfin_assert(_mesh->ordered());

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(D)[0]);

  // Get local index of facet with respect to the cell
  const std::size_t local_facet = cell.index(*this);

  return cell.normal(local_facet, i);
}
//-----------------------------------------------------------------------------
Point Facet::normal() const
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
double Facet::squared_distance(const Point& point) const
{
  if (_dim == 1)
  {
    // Extract vertices
    const MeshGeometry& geometry = _mesh->geometry();
    const unsigned int* vertices = entities(0);
    const Point a = geometry.point(vertices[0]);
    const Point b = geometry.point(vertices[1]);

    // Compute squared distance
    return IntervalCell::squared_distance(point, a, b);
  }
  else if (_dim == 2)
  {
    // Extract vertices
    const MeshGeometry& geometry = _mesh->geometry();
    const unsigned int* vertices = entities(0);
    const Point a = geometry.point(vertices[0]);
    const Point b = geometry.point(vertices[1]);
    const Point c = geometry.point(vertices[2]);

    // Compute squared distance
    return TriangleCell::squared_distance(point, a, b, c);
  }

  dolfin_error("Facet.cpp",
               "compute (squared) distance to facet",
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
