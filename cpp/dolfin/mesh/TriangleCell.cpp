// Copyright (C) 2006-2014 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TriangleCell.h"
#include "Cell.h"
#include "Facet.h"
#include "MeshEntity.h"
#include "Vertex.h"
#include <algorithm>
#include <cmath>
#include <dolfin/geometry/utils.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
double TriangleCell::squared_distance(const Cell& cell,
                                      const Eigen::Vector3d& point) const
{
  // Get the vertices as points
  const Geometry& geometry = cell.mesh().geometry();
  const std::int32_t* vertices = cell.entities(0);
  const Eigen::Vector3d a = geometry.x(vertices[0]);
  const Eigen::Vector3d b = geometry.x(vertices[1]);
  const Eigen::Vector3d c = geometry.x(vertices[2]);

  // Call function to compute squared distance
  return geometry::squared_distance_triangle(point, a, b, c);
}
//-----------------------------------------------------------------------------
