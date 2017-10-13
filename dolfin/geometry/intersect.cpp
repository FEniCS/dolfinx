// Copyright (C) 2013 Anders Logg
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
// First added:  2013-05-30
// Last changed: 2017-09-25

#include "MeshPointIntersection.h"
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include "intersect.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::shared_ptr<const MeshPointIntersection>
dolfin::intersect(const Mesh& mesh, const Point& point)
{
  // Intersection is only implemented for simplex meshes
  if (!mesh.type().is_simplex())
  {
    dolfin_error("intersect.cpp",
		 "intersect mesh and point",
		 "Intersection is only implemented for simplex meshes");
  }

  return std::shared_ptr<const MeshPointIntersection>
    (new MeshPointIntersection(mesh, point));
}
//-----------------------------------------------------------------------------
