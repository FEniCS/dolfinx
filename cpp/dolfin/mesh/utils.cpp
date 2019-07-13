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

namespace
{
//-----------------------------------------------------------------------------
Eigen::ArrayXd volume_interval(const mesh::Mesh& mesh,
                               const Eigen::Ref<const Eigen::ArrayXi> entities)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();
  assert(topology.connectivity(1, 0));
  const mesh::Connectivity& connectivity = *topology.connectivity(1, 0);

  Eigen::ArrayXd v(entities.rows());
  for (Eigen::Index i = 0; i < entities.rows(); ++i)
  {
    // Get the coordinates of the two vertices
    const std::int32_t* vertices = connectivity.connections(entities[i]);
    const Eigen::Vector3d x0 = geometry.x(vertices[0]);
    const Eigen::Vector3d x1 = geometry.x(vertices[1]);
    v[i] = (x1 - x0).norm();
  }

  return v;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd volume_triangle(const mesh::Mesh& mesh,
                               const Eigen::Ref<const Eigen::ArrayXi> entities)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();
  assert(topology.connectivity(2, 0));
  const mesh::Connectivity& connectivity = *topology.connectivity(2, 0);

  const int gdim = geometry.dim();
  assert(gdim == 2 or gdim == 3);
  Eigen::ArrayXd v(entities.rows());
  if (gdim == 2)
  {
    for (Eigen::Index i = 0; i < entities.rows(); ++i)
    {
      const std::int32_t* vertices = connectivity.connections(entities[i]);
      const Eigen::Vector3d x0 = geometry.x(vertices[0]);
      const Eigen::Vector3d x1 = geometry.x(vertices[1]);
      const Eigen::Vector3d x2 = geometry.x(vertices[2]);

      // Compute area of triangle embedded in R^2
      double v2 = (x0[0] * x1[1] + x0[1] * x2[0] + x1[0] * x2[1])
                  - (x2[0] * x1[1] + x2[1] * x0[0] + x1[0] * x0[1]);

      // Formula for volume from http://mathworld.wolfram.com
      v[i] = 0.5 * std::abs(v2);
    }
  }
  else if (gdim == 3)
  {
    for (Eigen::Index i = 0; i < entities.rows(); ++i)
    {
      const std::int32_t* vertices = connectivity.connections(entities[i]);
      const Eigen::Vector3d x0 = geometry.x(vertices[0]);
      const Eigen::Vector3d x1 = geometry.x(vertices[1]);
      const Eigen::Vector3d x2 = geometry.x(vertices[2]);

      // Compute area of triangle embedded in R^3
      const double v0 = (x0[1] * x1[2] + x0[2] * x2[1] + x1[1] * x2[2])
                        - (x2[1] * x1[2] + x2[2] * x0[1] + x1[1] * x0[2]);
      const double v1 = (x0[2] * x1[0] + x0[0] * x2[2] + x1[2] * x2[0])
                        - (x2[2] * x1[0] + x2[0] * x0[2] + x1[2] * x0[0]);
      const double v2 = (x0[0] * x1[1] + x0[1] * x2[0] + x1[0] * x2[1])
                        - (x2[0] * x1[1] + x2[1] * x0[0] + x1[0] * x0[1]);

      // Formula for volume from http://mathworld.wolfram.com
      v[i] = 0.5 * sqrt(v0 * v0 + v1 * v1 + v2 * v2);
    }
  }
  else
    throw std::runtime_error("Unexpected geometric dimension.");

  return v;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd
volume_tetrahedron(const mesh::Mesh& mesh,
                   const Eigen::Ref<const Eigen::ArrayXi> entities)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();
  assert(topology.connectivity(3, 0));
  const mesh::Connectivity& connectivity = *topology.connectivity(3, 0);

  Eigen::ArrayXd v(entities.rows());
  for (Eigen::Index i = 0; i < entities.rows(); ++i)
  {
    // Get the coordinates of the four vertices
    const std::int32_t* vertices = connectivity.connections(entities[i]);
    const Eigen::Vector3d x0 = geometry.x(vertices[0]);
    const Eigen::Vector3d x1 = geometry.x(vertices[1]);
    const Eigen::Vector3d x2 = geometry.x(vertices[2]);
    const Eigen::Vector3d x3 = geometry.x(vertices[3]);

    // Formula for volume from http://mathworld.wolfram.com
    const double v_tmp
        = (x0[0]
               * (x1[1] * x2[2] + x3[1] * x1[2] + x2[1] * x3[2] - x2[1] * x1[2]
                  - x1[1] * x3[2] - x3[1] * x2[2])
           - x1[0]
                 * (x0[1] * x2[2] + x3[1] * x0[2] + x2[1] * x3[2]
                    - x2[1] * x0[2] - x0[1] * x3[2] - x3[1] * x2[2])
           + x2[0]
                 * (x0[1] * x1[2] + x3[1] * x0[2] + x1[1] * x3[2]
                    - x1[1] * x0[2] - x0[1] * x3[2] - x3[1] * x1[2])
           - x3[0]
                 * (x0[1] * x1[2] + x1[1] * x2[2] + x2[1] * x0[2]
                    - x1[1] * x0[2] - x2[1] * x1[2] - x0[1] * x2[2]));

    v[i] = std::abs(v_tmp) / 6.0;
  }

  return v;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd
volume_quadrilateral(const mesh::Mesh& mesh,
                     const Eigen::Ref<const Eigen::ArrayXi> entities)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();
  assert(topology.connectivity(2, 0));
  const mesh::Connectivity& connectivity = *topology.connectivity(2, 0);

  const int gdim = geometry.dim();
  Eigen::ArrayXd v(entities.rows());
  for (Eigen::Index i = 0; i < entities.rows(); ++i)
  {
    // Get the coordinates of the four vertices
    const std::int32_t* vertices = connectivity.connections(entities[i]);
    const Eigen::Vector3d p0 = geometry.x(vertices[0]);
    const Eigen::Vector3d p1 = geometry.x(vertices[1]);
    const Eigen::Vector3d p2 = geometry.x(vertices[2]);
    const Eigen::Vector3d p3 = geometry.x(vertices[3]);

    const Eigen::Vector3d c = (p0 - p3).cross(p1 - p2);
    const double volume = 0.5 * c.norm();

    if (gdim == 3)
    {
      // Vertices are coplanar if det(p1-p0 | p3-p0 | p2-p0) is zero
      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> m;
      m.row(0) = (p1 - p0).transpose();
      m.row(1) = (p3 - p0).transpose();
      m.row(2) = (p2 - p0).transpose();

      // Check for coplanarity
      const double copl = m.determinant();
      const double h = std::min(1.0, std::pow(volume, 1.5));
      if (std::abs(copl) > h * DBL_EPSILON)
        throw std::runtime_error("Not coplanar");
    }

    v[i] = volume;
  }
  return v;
}
//-----------------------------------------------------------------------------

} // namespace

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
    return std::string();
  }
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
Eigen::ArrayXd
mesh::volume_cells(const mesh::Mesh& mesh,
                   const Eigen::Ref<const Eigen::ArrayXi> entities)
{
  const int dim = mesh::cell_dim(mesh.type().type);
  return mesh::volume_entities(mesh, entities, dim);
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd
mesh::volume_entities(const mesh::Mesh& mesh,
                      const Eigen::Ref<const Eigen::ArrayXi> entities, int dim)
{
  // if (entities.rows() > mesh.num_entities(dim))
  // {
  //   throw std::runtime_error(
  //       "Too many entities requested for volume computation.");
  // }

  const mesh::CellTypeOld& cell_type_obj = mesh.type();
  const mesh::CellType type = cell_type_obj.entity_type(dim);
  switch (type)
  {
  case mesh::CellType::point:
    return Eigen::ArrayXd::Zero(entities.rows());
  case mesh::CellType::interval:
    return volume_interval(mesh, entities);
  case mesh::CellType::triangle:
    return volume_triangle(mesh, entities);
  case mesh::CellType::tetrahedron:
    return volume_tetrahedron(mesh, entities);
  case mesh::CellType::quadrilateral:
    return volume_quadrilateral(mesh, entities);
  case mesh::CellType::hexahedron:
    throw std::runtime_error(
        "Volume computation for hexahedral cell not supported.");
  default:
    throw std::runtime_error("Unknown cell type.");
    return Eigen::ArrayXd();
  }
}
//-----------------------------------------------------------------------------
double mesh::volume(const mesh::MeshEntity& e)
{
  Eigen::ArrayXi index(1);
  index = e.index();
  Eigen::ArrayXd v = volume_entities(e.mesh(), index, e.dim());
  assert(v.rows() == 1);
  return v[0];
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
