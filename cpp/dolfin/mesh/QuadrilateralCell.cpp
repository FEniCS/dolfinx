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
#include <cmath>
#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>

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
    log::dolfin_error("QuadrilateralCell.cpp",
                      "access number of entities of quadrilateral cell",
                      "Illegal topological dimension (%d)", dim);
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
    log::dolfin_error(
        "QuadrilateralCell.cpp",
        "access number of vertices for subsimplex of quadrilateral cell",
        "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
void QuadrilateralCell::create_entities(boost::multi_array<std::int32_t, 2>& e,
                                        std::size_t dim,
                                        const std::int32_t* v) const
{
  // We only need to know how to create edges
  if (dim != 1)
  {
    log::dolfin_error(
        "QuadrilateralCell.cpp", "create entities of quadrilateral cell",
        "Don't know how to create entities of topological dimension %d", dim);
  }

  // Resize data structure
  e.resize(boost::extents[4][2]);

  // Create the four edges
  e[0][0] = v[0];
  e[0][1] = v[1];
  e[1][0] = v[2];
  e[1][1] = v[3];
  e[2][0] = v[0];
  e[2][1] = v[2];
  e[3][0] = v[1];
  e[3][1] = v[3];
}
//-----------------------------------------------------------------------------
double QuadrilateralCell::volume(const MeshEntity& cell) const
{
  if (cell.dim() != 2)
  {
    log::dolfin_error("QuadrilateralCell.cpp", "compute volume (area) of cell",
                      "Illegal mesh entity");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the four vertices
  const std::int32_t* vertices = cell.entities(0);
  const geometry::Point p0 = geometry.point(vertices[0]);
  const geometry::Point p1 = geometry.point(vertices[1]);
  const geometry::Point p2 = geometry.point(vertices[2]);
  const geometry::Point p3 = geometry.point(vertices[3]);

  if (geometry.dim() != 2 && geometry.dim() != 3)
  {
    log::dolfin_error("QuadrilateralCell.cpp",
                      "compute volume of quadrilateral",
                      "Only know how to compute volume in R^2 or R^3");
  }

  const geometry::Point c = (p0 - p3).cross(p1 - p2);
  const double volume = 0.5 * c.norm();

  if (geometry.dim() == 3)
  {
    // Vertices are coplanar if det(p1-p0 | p3-p0 | p2-p0) is zero
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> m;
    m.row(0) << (p1 - p0)[0], (p1 - p0)[1], (p1 - p0)[2];
    m.row(1) << (p3 - p0)[0], (p3 - p0)[1], (p3 - p0)[2];
    m.row(2) << (p2 - p0)[0], (p2 - p0)[1], (p2 - p0)[2];
    const double copl = m.determinant();
    const double h = std::min(1.0, std::pow(volume, 1.5));
    // Check for coplanarity
    if (std::abs(copl) > h * DOLFIN_EPS)
    {
      log::dolfin_error("QuadrilateralCell.cpp",
                        "compute volume of quadrilateral",
                        "Vertices of the quadrilateral are not coplanar");
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
    log::dolfin_error("QuadrilateralCell.cpp",
                      "compute circumradius of quadrilateral cell",
                      "Illegal mesh entity");
  }

  log::dolfin_error("QuadrilateralCell.cpp",
                    "compute cirumradius of quadrilateral cell",
                    "Don't know how to compute circumradius");

  dolfin_not_implemented();
  return 0.0;
}
//-----------------------------------------------------------------------------
double QuadrilateralCell::squared_distance(const Cell& cell,
                                           const geometry::Point& point) const
{
  dolfin_not_implemented();
  return 0.0;
}
//-----------------------------------------------------------------------------
double QuadrilateralCell::normal(const Cell& cell, std::size_t facet,
                                 std::size_t i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
geometry::Point QuadrilateralCell::normal(const Cell& cell,
                                          std::size_t facet) const
{

  // Make sure we have facets
  cell.mesh().init(2, 1);

  // Create facet from the mesh and local facet number
  Facet f(cell.mesh(), cell.entities(1)[facet]);

  if (cell.mesh().geometry().dim() != 2)
    log::dolfin_error(
        "QuadrilateralCell.cpp", "find normal",
        "Normal vector is not defined in dimension %d (only defined "
        "when the triangle is in R^2",
        cell.mesh().geometry().dim());

  // Get global index of opposite vertex
  const std::size_t v0 = cell.entities(0)[facet];

  // Get global index of vertices on the facet
  const std::size_t v1 = f.entities(0)[0];
  const std::size_t v2 = f.entities(0)[1];

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the three vertices
  const geometry::Point p0 = geometry.point(v0);
  const geometry::Point p1 = geometry.point(v1);
  const geometry::Point p2 = geometry.point(v2);

  // Subtract projection of p2 - p0 onto p2 - p1
  geometry::Point t = p2 - p1;
  t /= t.norm();
  geometry::Point n = p2 - p0;
  n -= t * n.dot(t);

  // Normalize
  n /= n.norm();

  return n;
}
//-----------------------------------------------------------------------------
geometry::Point QuadrilateralCell::cell_normal(const Cell& cell) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Cell_normal only defined for gdim = 2, 3:
  const std::size_t gdim = geometry.dim();
  if (gdim > 3)
    log::dolfin_error("QuadrilateralCell.cpp", "compute cell normal",
                      "Illegal geometric dimension (%d)", gdim);

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
  const MeshGeometry& geometry = cell.mesh().geometry();

  const geometry::Point p0 = geometry.point(v0);
  const geometry::Point p1 = geometry.point(v1);

  return (p0 - p1).norm();
}
//-----------------------------------------------------------------------------
void QuadrilateralCell::order(
    Cell& cell,
    const std::vector<std::int64_t>& local_to_global_vertex_indices) const
{
  // Not implemented
  // FIXME - probably not appropriate for quad cells.
}
//-----------------------------------------------------------------------------
std::string QuadrilateralCell::description(bool plural) const
{
  if (plural)
    return "quadrilaterals";
  return "quadrilateral";
}
//-----------------------------------------------------------------------------
