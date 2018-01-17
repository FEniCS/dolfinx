// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshPointIntersection.h"
#include "BoundingBoxTree.h"
#include <dolfin/log/LogStream.h>
#include <dolfin/mesh/Cell.h>

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
