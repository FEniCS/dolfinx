// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "QuadrilateralCell.h"
#include "Cell.h"
#include "Facet.h"
#include "MeshEntity.h"
#include "Vertex.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cfloat>
#include <cmath>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
double QuadrilateralCell::squared_distance(const Cell& cell,
                                           const Eigen::Vector3d& point) const
{
  throw std::runtime_error("Not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
Eigen::Vector3d QuadrilateralCell::normal(const Cell& cell,
                                          std::size_t facet) const
{
  // Make sure we have facets
  cell.mesh().create_connectivity(2, 1);

  // Create facet from the mesh and local facet number
  Facet f(cell.mesh(), cell.entities(1)[facet]);

  if (cell.mesh().geometry().dim() != 2)
  {
    throw std::runtime_error("Illegal geometric dimension");
  }

  // Get global index of opposite vertex
  const std::size_t v0 = cell.entities(0)[facet];

  // Get global index of vertices on the facet
  const std::size_t v1 = f.entities(0)[0];
  const std::size_t v2 = f.entities(0)[1];

  // Get mesh geometry
  const Geometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the three vertices
  const Eigen::Vector3d p0 = geometry.x(v0);
  const Eigen::Vector3d p1 = geometry.x(v1);
  const Eigen::Vector3d p2 = geometry.x(v2);

  // Subtract projection of p2 - p0 onto p2 - p1
  Eigen::Vector3d t = p2 - p1;
  t /= t.norm();
  Eigen::Vector3d n = p2 - p0;
  n -= t * n.dot(t);

  // Normalize
  n /= n.norm();

  return n;
}
//-----------------------------------------------------------------------------
