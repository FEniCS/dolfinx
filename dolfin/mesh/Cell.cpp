// Copyright (C) 2016 Anders Logg
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

#include <dolfin/geometry/IntersectionConstruction.h>
#include <dolfin/geometry/CollisionPredicates.h>
#include "Cell.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
bool Cell::contains(const Point& point) const
{
  return CollisionPredicates::collides(*this, point);
}
//-----------------------------------------------------------------------------
bool Cell::collides(const Point& point) const
{
  return CollisionPredicates::collides(*this, point);
}
//-----------------------------------------------------------------------------
bool Cell::collides(const MeshEntity& entity) const
{
  return CollisionPredicates::collides(*this, entity);
}
//-----------------------------------------------------------------------------
std::vector<Point>
Cell::intersection(const MeshEntity& entity) const
{
  return IntersectionConstruction::intersection(*this, entity);
}
//-----------------------------------------------------------------------------
