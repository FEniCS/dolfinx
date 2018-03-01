// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "intersect.h"
#include "MeshPointIntersection.h"
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>

using namespace dolfin;
using namespace dolfin::geometry;

//-----------------------------------------------------------------------------
std::shared_ptr<const MeshPointIntersection>
dolfin::geometry::intersect(const mesh::Mesh& mesh, const Point& point)
{
  // Intersection is only implemented for simplex meshes
  if (!mesh.type().is_simplex())
  {
    log::dolfin_error("intersect.cpp", "intersect mesh and point",
                 "Intersection is only implemented for simplex meshes");
  }

  return std::make_shared<const MeshPointIntersection>(mesh, point);
}
//-----------------------------------------------------------------------------
