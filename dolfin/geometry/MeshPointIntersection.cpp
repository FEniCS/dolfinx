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
// First added:  2013-04-18
// Last changed: 2013-08-28

#include <dolfin/log/LogStream.h>
#include <dolfin/mesh/Cell.h>
#include "BoundingBoxTree.h"
#include "MeshPointIntersection.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshPointIntersection::MeshPointIntersection(const Mesh& mesh,
                                             const Point& point)
{
  // Build bounding box tree
  BoundingBoxTree tree;
  tree.build(mesh);

  // Compute intersection
  _intersected_cells = tree.compute_entity_collisions(point);
}
//-----------------------------------------------------------------------------
MeshPointIntersection::~MeshPointIntersection()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
