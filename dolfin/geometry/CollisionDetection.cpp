// Copyright (C) 2014 Anders Logg
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
// First added:  2014-02-03
// Last changed: 2014-02-03

#include <dolfin/mesh/MeshEntity.h>
#include "Point.h"
#include "CollisionDetection.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
bool
CollisionDetection::collides_tetrahedron_point(const MeshEntity& tetrahedron,
                                               const Point& point)
{
  dolfin_assert(tetrahedron.mesh().topology().dim() == 3);

  dolfin_not_implemented();
  return false;
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::collides_tetrahedron_triangle(const MeshEntity& tetrahedron,
                                                  const MeshEntity& triangle)
{
  dolfin_assert(tetrahedron.mesh().topology().dim() == 3);
  dolfin_assert(triangle.mesh().topology().dim() == 2);

  dolfin_not_implemented();
  return false;
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::collides_tetrahedron_tetrahedron(const MeshEntity& tetrahedron_0,
                                                     const MeshEntity& tetrahedron_1)
{
  dolfin_assert(tetrahedron_0.mesh().topology().dim() == 3);
  dolfin_assert(tetrahedron_1.mesh().topology().dim() == 3);

  dolfin_not_implemented();
  return false;
}
//-----------------------------------------------------------------------------
