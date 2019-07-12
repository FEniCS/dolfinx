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
int QuadrilateralCell::num_vertices(int dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // edges
  case 2:
    return 4; // cells
  default:
    throw std::runtime_error("Illegal topological dimension");
  }

  return 0;
}
//-----------------------------------------------------------------------------
void QuadrilateralCell::create_entities(
    Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        e,
    std::size_t dim, const std::int32_t* v) const
{
  // We only need to know how to create edges
  if (dim != 1)
  {
    throw std::runtime_error("Illegal topological dimension");
  }

  // Resize data structure
  e.resize(4, 2);

  // Create the four edges
  e(0, 0) = v[0];
  e(0, 1) = v[1];
  e(1, 0) = v[2];
  e(1, 1) = v[3];
  e(2, 0) = v[0];
  e(2, 1) = v[2];
  e(3, 0) = v[1];
  e(3, 1) = v[3];
}
//-----------------------------------------------------------------------------
double QuadrilateralCell::circumradius(const MeshEntity& cell) const
{
  // Check that we get a cell
  if (cell.dim() != 2)
  {
    throw std::runtime_error("Illegal topological dimension");
  }

  throw std::runtime_error("Not supported");

  return 0.0;
}
//-----------------------------------------------------------------------------
double QuadrilateralCell::squared_distance(const Cell& cell,
                                           const Eigen::Vector3d& point) const
{
  throw std::runtime_error("Not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
double QuadrilateralCell::normal(const Cell& cell, std::size_t facet,
                                 std::size_t i) const
{
  return normal(cell, facet)[i];
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
Eigen::Vector3d QuadrilateralCell::cell_normal(const Cell& cell) const
{
  // Get mesh geometry
  const Geometry& geometry = cell.mesh().geometry();

  // Cell_normal only defined for gdim = 2, 3:
  const std::size_t gdim = geometry.dim();
  if (gdim > 3)
  {
    throw std::runtime_error("Illegal geometric dimension");
  }

  // Get the three vertices as points
  const std::int32_t* vertices = cell.entities(0);
  const Eigen::Vector3d p0 = geometry.x(vertices[0]);
  const Eigen::Vector3d p1 = geometry.x(vertices[1]);
  const Eigen::Vector3d p2 = geometry.x(vertices[2]);

  // Defined cell normal via cross product of first two edges:
  const Eigen::Vector3d v01 = p1 - p0;
  const Eigen::Vector3d v02 = p2 - p0;
  Eigen::Vector3d n = v01.cross(v02);

  // Normalize
  n /= n.norm();

  return n;
}
//-----------------------------------------------------------------------------
double QuadrilateralCell::facet_area(const Cell& cell, std::size_t facet) const
{
  // Create facet from the mesh and local facet number
  const Facet f(cell.mesh(), cell.entities(1)[facet]);

  // Get global index of vertices on the facet
  const std::size_t v0 = f.entities(0)[0];
  const std::size_t v1 = f.entities(0)[1];

  // Get mesh geometry
  const Geometry& geometry = cell.mesh().geometry();

  const Eigen::Vector3d p0 = geometry.x(v0);
  const Eigen::Vector3d p1 = geometry.x(v1);

  return (p0 - p1).norm();
}
//-----------------------------------------------------------------------------
