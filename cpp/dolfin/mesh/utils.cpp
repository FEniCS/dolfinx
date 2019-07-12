// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Geometry.h"
#include "MeshEntity.h"
#include <Eigen/Dense>
#include <cfloat>
#include <cstdlib>
#include <stdexcept>

using namespace dolfin;

//-----------------------------------------------------------------------------
std::string mesh::to_string(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return "point";
  case mesh::CellType::interval:
    return "interval";
  case mesh::CellType::triangle:
    return "triangle";
  case mesh::CellType::tetrahedron:
    return "tetrahedron";
  case mesh::CellType::quadrilateral:
    return "quadrilateral";
  case mesh::CellType::hexahedron:
    return "hexahedron";
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return "";
}
//-----------------------------------------------------------------------------
mesh::CellType mesh::to_type(std::string type)
{
  if (type == "point")
    return mesh::CellType::point;
  else if (type == "interval")
    return mesh::CellType::interval;
  else if (type == "triangle")
    return mesh::CellType::triangle;
  else if (type == "tetrahedron")
    return mesh::CellType::tetrahedron;
  else if (type == "quadrilateral")
    return mesh::CellType::quadrilateral;
  else if (type == "hexahedron")
    return mesh::CellType::hexahedron;
  else
    throw std::runtime_error("Unknown cell type (" + type + ")");

  // Should no reach this point
  return mesh::CellType::interval;
}
//-----------------------------------------------------------------------------
int mesh::cell_dim(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return 0;
  case mesh::CellType::interval:
    return 1;
  case mesh::CellType::triangle:
    return 2;
  case mesh::CellType::tetrahedron:
    return 3;
  case mesh::CellType::quadrilateral:
    return 2;
  case mesh::CellType::hexahedron:
    return 3;
  default:
    throw std::runtime_error("Unknown cell type.");
    return -1;
  }
}
//-----------------------------------------------------------------------------
int mesh::cell_num_entities(mesh::CellType type, int dim)
{
  switch (type)
  {
  case mesh::CellType::point:
    switch (dim)
    {
    case 0:
      return 1; // vertices
    }
  case mesh::CellType::interval:
    switch (dim)
    {
    case 0:
      return 2; // vertices
    case 1:
      return 1; // cells
    }
  case mesh::CellType::triangle:
    switch (dim)
    {
    case 0:
      return 3; // vertices
    case 1:
      return 3; // edges
    case 2:
      return 1; // cells
    }
  case mesh::CellType::tetrahedron:
    switch (dim)
    {
    case 0:
      return 4; // vertices
    case 1:
      return 6; // edges
    case 2:
      return 4; // faces
    case 3:
      return 1; // cells
    }
  case mesh::CellType::quadrilateral:
    switch (dim)
    {
    case 0:
      return 4; // vertices
    case 1:
      return 4; // edges
    case 2:
      return 1; // cells
    }
  case mesh::CellType::hexahedron:
    switch (dim)
    {
    case 0:
      return 8; // vertices
    case 1:
      return 12; // edges
    case 2:
      return 6; // faces
    case 3:
      return 1; // cells
    }
  default:
    throw std::runtime_error("Unknown cell type.");
    return -1;
  }
}
//-----------------------------------------------------------------------------
bool mesh::is_simplex(mesh::CellType type)
{
  return static_cast<int>(type) > 0;
}
//-----------------------------------------------------------------------------
int mesh::num_cell_vertices(mesh::CellType type)
{
  return std::abs(static_cast<int>(type));
}
//-----------------------------------------------------------------------------
double mesh::volume_interval(const mesh::MeshEntity& interval)
{
  assert(interval.mesh().type().type == mesh::CellType::interval);

  // Get mesh geometry
  const Geometry& geometry = interval.mesh().geometry();

  // Get the coordinates of the two vertices
  const std::int32_t* vertices = interval.entities(0);
  const Eigen::Vector3d x0 = geometry.x(vertices[0]);
  const Eigen::Vector3d x1 = geometry.x(vertices[1]);
  return (x1 - x0).norm();
}
//-----------------------------------------------------------------------------
double mesh::volume(const mesh::MeshEntity& entity)
{
  mesh::CellType type = entity.mesh().type().type;
  switch (type)
  {
  case mesh::CellType::point:
    return 0.0;
  case mesh::CellType::interval:
    return mesh::volume_interval(entity);
  case mesh::CellType::triangle:
    return mesh::volume_triangle(entity);
  case mesh::CellType::tetrahedron:
    return mesh::volume_tetrahedron(entity);
  case mesh::CellType::quadrilateral:
    return mesh::volume_quadrilateral(entity);
  case mesh::CellType::hexahedron:
    throw std::runtime_error(
        "Volume computation for hexahedral cell not supported.");
  default:
    throw std::runtime_error("Unknown cell type.");
    return 0.0;
  }
}
//-----------------------------------------------------------------------------
double mesh::volume_triangle(const mesh::MeshEntity& triangle)
{
  assert(triangle.mesh().type().type == mesh::CellType::triangle);

  // Get mesh geometry
  const Geometry& geometry = triangle.mesh().geometry();

  // Get the coordinates of the three vertices
  const std::int32_t* vertices = triangle.entities(0);
  const Eigen::Vector3d x0 = geometry.x(vertices[0]);
  const Eigen::Vector3d x1 = geometry.x(vertices[1]);
  const Eigen::Vector3d x2 = geometry.x(vertices[2]);

  if (geometry.dim() == 2)
  {
    // Compute area of triangle embedded in R^2
    double v2 = (x0[0] * x1[1] + x0[1] * x2[0] + x1[0] * x2[1])
                - (x2[0] * x1[1] + x2[1] * x0[0] + x1[0] * x0[1]);

    // Formula for volume from http://mathworld.wolfram.com
    return 0.5 * std::abs(v2);
  }
  else if (geometry.dim() == 3)
  {
    // Compute area of triangle embedded in R^3
    const double v0 = (x0[1] * x1[2] + x0[2] * x2[1] + x1[1] * x2[2])
                      - (x2[1] * x1[2] + x2[2] * x0[1] + x1[1] * x0[2]);
    const double v1 = (x0[2] * x1[0] + x0[0] * x2[2] + x1[2] * x2[0])
                      - (x2[2] * x1[0] + x2[0] * x0[2] + x1[2] * x0[0]);
    const double v2 = (x0[0] * x1[1] + x0[1] * x2[0] + x1[0] * x2[1])
                      - (x2[0] * x1[1] + x2[1] * x0[0] + x1[0] * x0[1]);

    // Formula for volume from http://mathworld.wolfram.com
    return 0.5 * sqrt(v0 * v0 + v1 * v1 + v2 * v2);
  }
  else
    throw std::runtime_error("Illegal geometric dimension");

  return 0.0;
}
//-----------------------------------------------------------------------------
double mesh::volume_quadrilateral(const mesh::MeshEntity& quadrilateral)
{
  assert(quadrilateral.mesh().type().type == mesh::CellType::quadrilateral);

  // Get mesh geometry
  const Geometry& geometry = quadrilateral.mesh().geometry();

  // Get the coordinates of the four vertices
  const std::int32_t* vertices = quadrilateral.entities(0);
  const Eigen::Vector3d p0 = geometry.x(vertices[0]);
  const Eigen::Vector3d p1 = geometry.x(vertices[1]);
  const Eigen::Vector3d p2 = geometry.x(vertices[2]);
  const Eigen::Vector3d p3 = geometry.x(vertices[3]);

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
      throw std::runtime_error("Not coplanar");
  }

  return volume;
}
//-----------------------------------------------------------------------------
double mesh::volume_tetrahedron(const mesh::MeshEntity& tetrahedron)
{
  assert(tetrahedron.mesh().type().type == mesh::CellType::tetrahedron);

  // Get mesh geometry
  const Geometry& geometry = tetrahedron.mesh().geometry();

  // Only know how to compute the volume when embedded in R^3
  assert(geometry.dim() == 3);

  // Get the coordinates of the four vertices
  const std::int32_t* vertices = tetrahedron.entities(0);
  const Eigen::Vector3d x0 = geometry.x(vertices[0]);
  const Eigen::Vector3d x1 = geometry.x(vertices[1]);
  const Eigen::Vector3d x2 = geometry.x(vertices[2]);
  const Eigen::Vector3d x3 = geometry.x(vertices[3]);

  // Formula for volume from http://mathworld.wolfram.com
  const double v = (x0[0]
                        * (x1[1] * x2[2] + x3[1] * x1[2] + x2[1] * x3[2]
                           - x2[1] * x1[2] - x1[1] * x3[2] - x3[1] * x2[2])
                    - x1[0]
                          * (x0[1] * x2[2] + x3[1] * x0[2] + x2[1] * x3[2]
                             - x2[1] * x0[2] - x0[1] * x3[2] - x3[1] * x2[2])
                    + x2[0]
                          * (x0[1] * x1[2] + x3[1] * x0[2] + x1[1] * x3[2]
                             - x1[1] * x0[2] - x0[1] * x3[2] - x3[1] * x1[2])
                    - x3[0]
                          * (x0[1] * x1[2] + x1[1] * x2[2] + x2[1] * x0[2]
                             - x1[1] * x0[2] - x2[1] * x1[2] - x0[1] * x2[2]));

  return std::abs(v) / 6.0;
}
//-----------------------------------------------------------------------------
// double mesh::inradius(const mesh::Cell& cell)
// {

// }
//-----------------------------------------------------------------------------
std::vector<std::int8_t> mesh::vtk_mapping(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return {0};
  case mesh::CellType::interval:
    return {0, 1};
  case mesh::CellType::triangle:
    return {0, 1, 2};
  case mesh::CellType::tetrahedron:
    return {0, 1, 2, 3};
  case mesh::CellType::quadrilateral:
    return {0, 1, 3, 2};
  case mesh::CellType::hexahedron:
    return {0, 1, 3, 2, 4, 5, 7, 6};
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return std::vector<std::int8_t>();
}
//-----------------------------------------------------------------------------
