// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Cell.h"
#include "Facet.h"
#include "Geometry.h"
#include "MeshEntity.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <stdexcept>

using namespace dolfin;

namespace
{
//-----------------------------------------------------------------------------
template <typename T>
T volume_interval(const mesh::Mesh& mesh,
                  const Eigen::Ref<const Eigen::ArrayXi> entities)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();
  assert(topology.connectivity(1, 0));
  const mesh::Connectivity& connectivity = *topology.connectivity(1, 0);

  T v(entities.rows());
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
template <typename T>
T volume_triangle(const mesh::Mesh& mesh,
                  const Eigen::Ref<const Eigen::ArrayXi> entities)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();
  assert(topology.connectivity(2, 0));
  const mesh::Connectivity& connectivity = *topology.connectivity(2, 0);

  const int gdim = geometry.dim();
  assert(gdim == 2 or gdim == 3);
  T v(entities.rows());
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
template <typename T>
T volume_tetrahedron(const mesh::Mesh& mesh,
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
template <typename T>
T volume_quadrilateral(const mesh::Mesh& mesh,
                       const Eigen::Ref<const Eigen::ArrayXi> entities)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();
  assert(topology.connectivity(2, 0));
  const mesh::Connectivity& connectivity = *topology.connectivity(2, 0);

  const int gdim = geometry.dim();
  T v(entities.rows());
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
/// Compute (generalized) volume of mesh entities of given dimension.
/// This templated versions allows for fixed size (statically allocated)
/// return arrays, which can be important for performance when computing
/// for a small number of entities.
template <typename T>
T volume_entities_tmpl(const mesh::Mesh& mesh,
                       const Eigen::Ref<const Eigen::ArrayXi> entities, int dim)
{
  const mesh::CellType type = cell_entity_type(mesh.cell_type, dim);
  switch (type)
  {
  case mesh::CellType::point:
  {
    T v(entities.rows());
    v.setOnes();
    return v;
  }
  case mesh::CellType::interval:
    return volume_interval<T>(mesh, entities);
  case mesh::CellType::triangle:
    return volume_triangle<T>(mesh, entities);
  case mesh::CellType::tetrahedron:
    return volume_tetrahedron<T>(mesh, entities);
  case mesh::CellType::quadrilateral:
    return volume_quadrilateral<T>(mesh, entities);
  case mesh::CellType::hexahedron:
    throw std::runtime_error(
        "Volume computation for hexahedral cell not supported.");
  default:
    throw std::runtime_error("Unknown cell type.");
    return T();
  }
}
//-----------------------------------------------------------------------------
template <typename T>
T circumradius_triangle(const mesh::Mesh& mesh,
                        const Eigen::Ref<const Eigen::ArrayXi> entities)
{
  // Get mesh geometry
  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();
  assert(topology.connectivity(2, 0));
  const mesh::Connectivity& connectivity = *topology.connectivity(2, 0);

  T volumes = volume_entities_tmpl<T>(mesh, entities, 2);

  T cr(entities.rows());
  for (Eigen::Index e = 0; e < entities.rows(); ++e)
  {
    const std::int32_t* vertices = connectivity.connections(entities[e]);
    const Eigen::Vector3d p0 = geometry.x(vertices[0]);
    const Eigen::Vector3d p1 = geometry.x(vertices[1]);
    const Eigen::Vector3d p2 = geometry.x(vertices[2]);

    // Compute side lengths
    const double a = (p1 - p2).norm();
    const double b = (p0 - p2).norm();
    const double c = (p0 - p1).norm();

    // Formula for circumradius from
    // http://mathworld.wolfram.com/Triangle.html
    cr[e] = a * b * c / (4.0 * volumes[e]);
  }
  return cr;
}
//-----------------------------------------------------------------------------
template <typename T>
T circumradius_tetrahedron(const mesh::Mesh& mesh,
                           const Eigen::Ref<const Eigen::ArrayXi> entities)
{
  // Get mesh geometry
  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();
  assert(topology.connectivity(3, 0));
  const mesh::Connectivity& connectivity = *topology.connectivity(3, 0);

  T volumes = volume_entities_tmpl<T>(mesh, entities, 3);

  T cr(entities.rows());
  for (Eigen::Index e = 0; e < entities.rows(); ++e)
  {
    const std::int32_t* vertices = connectivity.connections(entities[e]);
    const Eigen::Vector3d p0 = geometry.x(vertices[0]);
    const Eigen::Vector3d p1 = geometry.x(vertices[1]);
    const Eigen::Vector3d p2 = geometry.x(vertices[2]);
    const Eigen::Vector3d p3 = geometry.x(vertices[3]);

    // Compute side lengths
    const double a = (p1 - p2).norm();
    const double b = (p0 - p2).norm();
    const double c = (p0 - p1).norm();
    const double aa = (p0 - p3).norm();
    const double bb = (p1 - p3).norm();
    const double cc = (p2 - p3).norm();

    // Compute "area" of triangle with strange side lengths
    const double la = a * aa;
    const double lb = b * bb;
    const double lc = c * cc;
    const double s = 0.5 * (la + lb + lc);
    const double area = sqrt(s * (s - la) * (s - lb) * (s - lc));

    // Formula for circumradius from
    // http://mathworld.wolfram.com/Tetrahedron.html
    cr[e] = area / (6.0 * volumes[e]);
  }
  return cr;
}
//-----------------------------------------------------------------------------
template <typename T>
T circumradius_tmpl(const mesh::Mesh& mesh,
                    const Eigen::Ref<const Eigen::ArrayXi> entities, int dim)
{
  const mesh::CellType type = cell_entity_type(mesh.cell_type, dim);
  switch (type)
  {
  case mesh::CellType::point:
  {
    T cr(entities.rows());
    cr.setZero();
    return cr;
  }
  case mesh::CellType::interval:
    return volume_interval<T>(mesh, entities) / 2;
  case mesh::CellType::triangle:
    return circumradius_triangle<T>(mesh, entities);
  case mesh::CellType::tetrahedron:
    return circumradius_tetrahedron<T>(mesh, entities);
  // case mesh::CellType::quadrilateral:
  //   // continue;
  // case mesh::CellType::hexahedron:
  //   // continue;
  default:
    throw std::runtime_error(
        "Unsupported cell type for circumradius computation.");
    return T();
  }
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_entities_interval(int dim)
{
  assert(dim == 0);
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> e(2, 1);
  e(0, 0) = 0;
  e(1, 0) = 1;
  return e;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_entities_triangle(int dim)
{
  // We only need to know how to create edges
  assert(dim == 1);

  // Create the three edges
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> e(3, 2);
  e(0, 0) = 1;
  e(0, 1) = 2;
  e(1, 0) = 0;
  e(1, 1) = 2;
  e(2, 0) = 0;
  e(2, 1) = 1;

  return e;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_entities_quadrilateral(int dim)
{
  assert(dim == 1);

  // Create the four edges
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> e(4, 2);
  e(0, 0) = 0;
  e(0, 1) = 1;
  e(1, 0) = 2;
  e(1, 1) = 3;
  e(2, 0) = 0;
  e(2, 1) = 2;
  e(3, 0) = 1;
  e(3, 1) = 3;

  return e;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_entities_tetrahedron(int dim)
{
  // We only need to know how to create edges and faces
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> e;
  switch (dim)
  {
  case 1:
    // Resize data structure
    e.resize(6, 2);

    // Create the six edges
    e(0, 0) = 2;
    e(0, 1) = 3;
    e(1, 0) = 1;
    e(1, 1) = 3;
    e(2, 0) = 1;
    e(2, 1) = 2;
    e(3, 0) = 0;
    e(3, 1) = 3;
    e(4, 0) = 0;
    e(4, 1) = 2;
    e(5, 0) = 0;
    e(5, 1) = 1;
    break;
  case 2:
    // Resize data structure
    e.resize(4, 3);

    // Create the four faces
    e(0, 0) = 1;
    e(0, 1) = 2;
    e(0, 2) = 3;
    e(1, 0) = 0;
    e(1, 1) = 2;
    e(1, 2) = 3;
    e(2, 0) = 0;
    e(2, 1) = 1;
    e(2, 2) = 3;
    e(3, 0) = 0;
    e(3, 1) = 1;
    e(3, 2) = 2;
    break;
  default:
    throw std::runtime_error("Illegal topological dimension");
  }

  return e;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_entities_hexahedron(int dim)
{
  // We need to know how to create edges and faces

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> e;
  switch (dim)
  {
  case 1:
    // Resize data structure
    e.resize(12, 2);

    // Create the 12 edges
    e(0, 0) = 0;
    e(0, 1) = 1;
    e(1, 0) = 2;
    e(1, 1) = 3;
    e(2, 0) = 4;
    e(2, 1) = 5;
    e(3, 0) = 6;
    e(3, 1) = 7;
    e(4, 0) = 0;
    e(4, 1) = 2;
    e(5, 0) = 1;
    e(5, 1) = 3;
    e(6, 0) = 4;
    e(6, 1) = 6;
    e(7, 0) = 5;
    e(7, 1) = 7;
    e(8, 0) = 0;
    e(8, 1) = 4;
    e(9, 0) = 1;
    e(9, 1) = 5;
    e(10, 0) = 2;
    e(10, 1) = 6;
    e(11, 0) = 3;
    e(11, 1) = 7;
    break;
  case 2:
    // Resize data structure
    e.resize(6, 4);

    // Create the 6 faces
    e(0, 0) = 0;
    e(0, 1) = 1;
    e(0, 2) = 2;
    e(0, 3) = 3;
    e(1, 0) = 4;
    e(1, 1) = 5;
    e(1, 2) = 6;
    e(1, 3) = 7;
    e(2, 0) = 0;
    e(2, 1) = 1;
    e(2, 2) = 4;
    e(2, 3) = 5;
    e(3, 0) = 2;
    e(3, 1) = 3;
    e(3, 2) = 6;
    e(3, 3) = 7;
    e(4, 0) = 0;
    e(4, 1) = 2;
    e(4, 2) = 4;
    e(4, 3) = 6;
    e(5, 0) = 1;
    e(5, 1) = 3;
    e(5, 2) = 5;
    e(5, 3) = 7;
    break;
  default:
    throw std::runtime_error("Illegal topological dimension. Must be 1 or 2.");
  }

  return e;
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
mesh::CellType mesh::cell_entity_type(mesh::CellType type, int d)
{
  const int dim = mesh::cell_dim(type);
  if (d == dim)
    return type;
  else if (d == 1)
    return CellType::interval;
  else if (d == (dim - 1))
    return mesh::cell_facet_type(type);

  return CellType::point;
}
//-----------------------------------------------------------------------------
mesh::CellType mesh::cell_facet_type(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return mesh::CellType::point;
  case mesh::CellType::interval:
    return mesh::CellType::point;
  case mesh::CellType::triangle:
    return mesh::CellType::interval;
  case mesh::CellType::tetrahedron:
    return mesh::CellType::triangle;
  case mesh::CellType::quadrilateral:
    return mesh::CellType::interval;
  case mesh::CellType::hexahedron:
    return mesh::CellType::quadrilateral;
  default:
    throw std::runtime_error("Unknown cell type.");
    return mesh::CellType::point;
  }
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
mesh::create_entities(mesh::CellType type, int dim)
{
  switch (type)
  {
  // case mesh::CellType::point:
  //   return create_entities_point(e, v);
  case mesh::CellType::interval:
    return create_entities_interval(dim);
  case mesh::CellType::triangle:
    return create_entities_triangle(dim);
  case mesh::CellType::tetrahedron:
    return create_entities_tetrahedron(dim);
  case mesh::CellType::quadrilateral:
    return create_entities_quadrilateral(dim);
  case mesh::CellType::hexahedron:
    return create_entities_hexahedron(dim);
  default:
    throw std::runtime_error("Unsupported cell type.");
    return Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();
  }
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
  const int dim = mesh::cell_dim(mesh.cell_type);
  return volume_entities_tmpl<Eigen::ArrayXd>(mesh, entities, dim);
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd
mesh::volume_entities(const mesh::Mesh& mesh,
                      const Eigen::Ref<const Eigen::ArrayXi> entities, int dim)
{
  return volume_entities_tmpl<Eigen::ArrayXd>(mesh, entities, dim);
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
Eigen::ArrayXd
mesh::circumradius(const mesh::Mesh& mesh,
                   const Eigen::Ref<const Eigen::ArrayXi> entities, int dim)
{
  return circumradius_tmpl<Eigen::ArrayXd>(mesh, entities, dim);
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd mesh::h(const Mesh& mesh,
                       const Eigen::Ref<const Eigen::ArrayXi> entities, int dim)
{
  // Get number of cell vertices
  const mesh::CellType type = cell_entity_type(mesh.cell_type, dim);
  const int num_vertices = num_cell_vertices(type);

  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();
  assert(topology.connectivity(dim, 0));
  const mesh::Connectivity& connectivity = *topology.connectivity(dim, 0);

  Eigen::ArrayXd h_cells = Eigen::ArrayXd::Zero(entities.rows());
  assert(num_vertices <= 8);
  std::array<Eigen::Vector3d, 8> points;
  for (Eigen::Index e = 0; e < entities.rows(); ++e)
  {
    // Get the coordinates  of the vertices
    const std::int32_t* vertices = connectivity.connections(entities[e]);
    for (int i = 0; i < num_vertices; ++i)
      points[i] = geometry.x(vertices[i]);

    // Get maximum edge length
    for (int i = 0; i < num_vertices; ++i)
    {
      for (int j = i + 1; j < num_vertices; ++j)
        h_cells[e] = std::max(h_cells[e], (points[i] - points[j]).norm());
    }
  }

  return h_cells;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd mesh::inradius(const mesh::Mesh& mesh,
                              const Eigen::Ref<const Eigen::ArrayXi> entities)
// double mesh::inradius(const mesh::Cell& cell)
{
  // Cell type
  const mesh::CellType type = mesh.cell_type;

  // Check cell type
  if (!mesh::is_simplex(type))
  {
    throw std::runtime_error(
        "inradius function not implemented for non-simplicial cells");
  }

  // Get cell dimension
  const int d = mesh::cell_dim(type);
  const mesh::Topology& topology = mesh.topology();
  assert(topology.connectivity(d, d - 1));
  const mesh::Connectivity& connectivity = *topology.connectivity(d, d - 1);

  const Eigen::ArrayXd volumes = mesh::volume_cells(mesh, entities);

  // // Build list of facets
  // Eigen::ArrayXi facet_list = ;
  // for (Eigen::Index c = 0; c < entities.rows(); ++c)
  //   const std::int32_t* facets = connectivity.connections(entities[c]);
  // for (int i = 0; i <= d; i++)
  //   facet_list[i] = facets[i];

  // FIXME: This computes all facets
  // Eigen::ArrayXi facet_list(mesh.num_entities(d - 1));
  // std::iota(facet_list.data(), facet_list.data() + facet_list.size(), 0);
  // const Eigen::ArrayXd facet_areas
  //     = mesh::volume_entities(mesh, facet_list, d - 1);

  Eigen::ArrayXd r(entities.rows());
  Eigen::ArrayXi facet_list(d + 1);
  for (Eigen::Index c = 0; c < entities.rows(); ++c)
  {
    if (volumes[c] == 0.0)
    {
      r[c] = 0.0;
      continue;
    }

    const std::int32_t* facets = connectivity.connections(entities[c]);
    for (int i = 0; i <= d; i++)
      facet_list[i] = facets[i];
    const double A = volume_entities_tmpl<Eigen::Array<double, 4, 1>>(
                         mesh, facet_list, d - 1)
                         .head(d + 1)
                         .sum();

    // See Jonathan Richard Shewchuk: What Is a Good Linear Finite
    // Element?, online:
    // http://www.cs.berkeley.edu/~jrs/papers/elemj.pdf
    // return d * V / A;
    r[c] = d * volumes[c] / A;
  }

  return r;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd
mesh::radius_ratio(const mesh::Mesh& mesh,
                   const Eigen::Ref<const Eigen::ArrayXi> entities)
{
  const mesh::CellType type = mesh.cell_type;
  const int dim = mesh::cell_dim(type);
  Eigen::ArrayXd r = mesh::inradius(mesh, entities);
  Eigen::ArrayXd cr = mesh::circumradius(mesh, entities, dim);
  return mesh::cell_dim(mesh.cell_type) * r / cr;
}
//-----------------------------------------------------------------------------
Eigen::Vector3d mesh::cell_normal(const mesh::Cell& cell)
{
  const int gdim = cell.mesh().geometry().dim();
  const mesh::CellType type = cell.type();
  const mesh::Geometry& geometry = cell.mesh().geometry();

  switch (type)
  {
  case (mesh::CellType::interval):
  {
    if (gdim > 2)
      throw std::invalid_argument("Interval cell normal undefined in 3D");

    // Get the two vertices as points
    const std::int32_t* vertices = cell.entities(0);
    Eigen::Vector3d p0 = geometry.x(vertices[0]);
    Eigen::Vector3d p1 = geometry.x(vertices[1]);

    // Define normal by rotating tangent counter-clockwise
    Eigen::Vector3d t = p1 - p0;
    Eigen::Vector3d n(-t[1], t[0], 0.0);
    return n.normalized();
  }
  case (mesh::CellType::triangle):
  {
    // Get the three vertices as points
    const std::int32_t* vertices = cell.entities(0);
    const Eigen::Vector3d p0 = geometry.x(vertices[0]);
    const Eigen::Vector3d p1 = geometry.x(vertices[1]);
    const Eigen::Vector3d p2 = geometry.x(vertices[2]);

    // Define cell normal via cross product of first two edges
    return ((p1 - p0).cross(p2 - p0)).normalized();
  }
  case (mesh::CellType::quadrilateral):
  {
    // TODO: check

    // Get three vertices as points
    const std::int32_t* vertices = cell.entities(0);
    const Eigen::Vector3d p0 = geometry.x(vertices[0]);
    const Eigen::Vector3d p1 = geometry.x(vertices[1]);
    const Eigen::Vector3d p2 = geometry.x(vertices[2]);

    // Defined cell normal via cross product of first two edges:
    return ((p1 - p0).cross(p2 - p0)).normalized();
  }
  default:
    throw std::invalid_argument(
        "cell_normal not supported for this cell type.");
  }
  return Eigen::Vector3d();
}
//-----------------------------------------------------------------------------
Eigen::Vector3d mesh::normal(const mesh::Cell& cell, int facet)
{
  const mesh::Geometry& geometry = cell.mesh().geometry();
  const mesh::CellType type = cell.type();

  switch (type)
  {
  case (mesh::CellType::interval):
  {
    const std::int32_t* vertices = cell.entities(0);
    Eigen::Vector3d n = geometry.x(vertices[0]) - geometry.x(vertices[1]);
    n.normalize();
    if (facet == 1)
      return -1.0 * n;
    else
      return n;
  }
  case (mesh::CellType::triangle):
  {
    // The normal vector is currently only defined for a triangle in R^2
    // MER: This code is super for a triangle in R^3 too, this error
    // could be removed, unless it is here for some other reason.
    if (geometry.dim() != 2)
      throw std::runtime_error("Illegal geometric dimension");

    cell.mesh().create_connectivity(2, 1);
    mesh::Facet f(cell.mesh(), cell.entities(1)[facet]);

    // Get global index of opposite vertex
    const int v0 = cell.entities(0)[facet];

    // Get global index of vertices on the facet
    const int v1 = f.entities(0)[0];
    const int v2 = f.entities(0)[1];

    // Get the coordinates of the three vertices
    const Eigen::Vector3d p0 = geometry.x(v0);
    const Eigen::Vector3d p1 = geometry.x(v1);
    const Eigen::Vector3d p2 = geometry.x(v2);

    // Subtract projection of p2 - p0 onto p2 - p1
    Eigen::Vector3d t = p2 - p1;
    Eigen::Vector3d n = p2 - p0;
    t.normalize();
    n.normalize();
    n -= t * n.dot(t);
    n.normalize();
    return n;
  }
  case (mesh::CellType::quadrilateral):
  {
    // Make sure we have facets
    cell.mesh().create_connectivity(2, 1);

    // Create facet from the mesh and local facet number
    Facet f(cell.mesh(), cell.entities(1)[facet]);

    if (cell.mesh().geometry().dim() != 2)
      throw std::runtime_error("Illegal geometric dimension");

    // Get global index of opposite vertex
    const std::size_t v0 = cell.entities(0)[facet];

    // Get global index of vertices on the facet
    const std::size_t v1 = f.entities(0)[0];
    const std::size_t v2 = f.entities(0)[1];

    // Get the coordinates of the three vertices
    const Eigen::Vector3d p0 = geometry.x(v0);
    const Eigen::Vector3d p1 = geometry.x(v1);
    const Eigen::Vector3d p2 = geometry.x(v2);

    // Subtract projection of p2 - p0 onto p2 - p1
    Eigen::Vector3d t = p2 - p1;
    t.normalize();
    Eigen::Vector3d n = p2 - p0;
    n -= t * n.dot(t);
    n.normalize();
    return n;
  }
  case (mesh::CellType::tetrahedron):
  {
    // Make sure we have facets
    cell.mesh().create_connectivity(3, 2);

    // Create facet from the mesh and local facet number
    Facet f(cell.mesh(), cell.entities(2)[facet]);

    // Get global index of opposite vertex
    const std::size_t v0 = cell.entities(0)[facet];

    // Get global index of vertices on the facet
    std::size_t v1 = f.entities(0)[0];
    std::size_t v2 = f.entities(0)[1];
    std::size_t v3 = f.entities(0)[2];

    // Get the coordinates of the four vertices
    const Eigen::Vector3d P0 = geometry.x(v0);
    const Eigen::Vector3d P1 = geometry.x(v1);
    const Eigen::Vector3d P2 = geometry.x(v2);
    const Eigen::Vector3d P3 = geometry.x(v3);

    // Create vectors
    Eigen::Vector3d V0 = P0 - P1;
    Eigen::Vector3d V1 = P2 - P1;
    Eigen::Vector3d V2 = P3 - P1;

    // Compute normal vector
    Eigen::Vector3d n = V1.cross(V2);

    // Normalize
    n.normalize();

    // Flip direction of normal so it points outward
    if (n.dot(V0) > 0)
      n *= -1.0;

    return n;
  }
  // case (mesh::CellType::hexahedron):
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return Eigen::Vector3d();
}
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
