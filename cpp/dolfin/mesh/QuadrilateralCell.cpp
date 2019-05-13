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
std::size_t QuadrilateralCell::dim() const { return 2; }
//-----------------------------------------------------------------------------
std::size_t QuadrilateralCell::num_entities(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 4; // vertices
  case 1:
    return 4; // edges
  case 2:
    return 1; // cells
  default:
    throw std::runtime_error("Illegal topological dimension");
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t QuadrilateralCell::num_vertices(std::size_t dim) const
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
double QuadrilateralCell::volume(const MeshEntity& cell) const
{
  if (cell.dim() != 2)
  {
    throw std::runtime_error("Illegal topological dimension");
  }

  // Get mesh geometry
  const Geometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the four vertices
  const std::int32_t* vertices = cell.entities(0);
  const Eigen::Vector3d p0 = geometry.x(vertices[0]);
  const Eigen::Vector3d p1 = geometry.x(vertices[1]);
  const Eigen::Vector3d p2 = geometry.x(vertices[2]);
  const Eigen::Vector3d p3 = geometry.x(vertices[3]);

  if (geometry.dim() != 2 && geometry.dim() != 3)
  {
    throw std::runtime_error("Illegal geometric dimension");
  }

  const Eigen::Vector3d c = (p0 - p3).cross(p1 - p2);
  const double volume = 0.5 * c.norm();

  if (geometry.dim() == 3)
  {
    // Vertices are coplanar if det(p1-p0 | p3-p0 | p2-p0) is zero
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> m;
    m.row(0) = (p1 - p0).transpose();
    m.row(1) = (p3 - p0).transpose();
    m.row(2) = (p2 - p0).transpose();

    const double copl = m.determinant();
    const double h = std::min(1.0, std::pow(volume, 1.5));
    // Check for coplanarity
    if (std::abs(copl) > h * DBL_EPSILON)
    {
      throw std::runtime_error("Not coplanar");
    }
  }

  return volume;
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
geometry::Point QuadrilateralCell::cell_normal(const Cell& cell) const
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
  const geometry::Point p0 = geometry.point(vertices[0]);
  const geometry::Point p1 = geometry.point(vertices[1]);
  const geometry::Point p2 = geometry.point(vertices[2]);

  // Defined cell normal via cross product of first two edges:
  const geometry::Point v01 = p1 - p0;
  const geometry::Point v02 = p2 - p0;
  geometry::Point n = v01.cross(v02);

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

  const geometry::Point p0 = geometry.point(v0);
  const geometry::Point p1 = geometry.point(v1);

  return (p0 - p1).norm();
}
//-----------------------------------------------------------------------------
std::string QuadrilateralCell::description(bool plural) const
{
  if (plural)
    return "quadrilaterals";
  return "quadrilateral";
}
//-----------------------------------------------------------------------------
