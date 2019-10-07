// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Geometry.h"
#include "MeshEntity.h"
#include "MeshIterator.h"
#include "cell_types.h"
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
                  const Eigen::Ref<const Eigen::ArrayXi>& entities)
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
                  const Eigen::Ref<const Eigen::ArrayXi>& entities)
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
                     const Eigen::Ref<const Eigen::ArrayXi>& entities)
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
                       const Eigen::Ref<const Eigen::ArrayXi>& entities)
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
                       const Eigen::Ref<const Eigen::ArrayXi>& entities,
                       int dim)
{
  const mesh::CellType type = cell_entity_type(mesh.cell_type(), dim);
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
                        const Eigen::Ref<const Eigen::ArrayXi>& entities)
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
                           const Eigen::Ref<const Eigen::ArrayXi>& entities)
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
                    const Eigen::Ref<const Eigen::ArrayXi>& entities, int dim)
{
  const mesh::CellType type = cell_entity_type(mesh.cell_type(), dim);
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

} // namespace

//-----------------------------------------------------------------------------
Eigen::ArrayXd
mesh::volume_entities(const mesh::Mesh& mesh,
                      const Eigen::Ref<const Eigen::ArrayXi>& entities, int dim)
{
  return volume_entities_tmpl<Eigen::ArrayXd>(mesh, entities, dim);
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd mesh::h(const Mesh& mesh,
                       const Eigen::Ref<const Eigen::ArrayXi>& entities,
                       int dim)
{
  // Get number of cell vertices
  const mesh::CellType type = cell_entity_type(mesh.cell_type(), dim);
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
Eigen::ArrayXd
mesh::circumradius(const mesh::Mesh& mesh,
                   const Eigen::Ref<const Eigen::ArrayXi>& entities, int dim)
{
  return circumradius_tmpl<Eigen::ArrayXd>(mesh, entities, dim);
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd mesh::inradius(const mesh::Mesh& mesh,
                              const Eigen::Ref<const Eigen::ArrayXi>& entities)
{
  // Cell type
  const mesh::CellType type = mesh.cell_type();

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

  const Eigen::ArrayXd volumes = mesh::volume_entities(mesh, entities, d);

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
                   const Eigen::Ref<const Eigen::ArrayXi>& entities)
{
  const mesh::CellType type = mesh.cell_type();
  const int dim = mesh::cell_dim(type);
  Eigen::ArrayXd r = mesh::inradius(mesh, entities);
  Eigen::ArrayXd cr = mesh::circumradius(mesh, entities, dim);
  return mesh::cell_dim(mesh.cell_type()) * r / cr;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
mesh::cell_normals(const mesh::Mesh& mesh, int dim)
{
  const int gdim = mesh.geometry().dim();
  const mesh::CellType type = mesh::cell_entity_type(mesh.cell_type(), dim);
  const mesh::Geometry& geometry = mesh.geometry();

  switch (type)
  {
  case (mesh::CellType::interval):
  {
    if (gdim > 2)
      throw std::invalid_argument("Interval cell normal undefined in 3D");

    Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> n(
        mesh.num_entities(1), 3);
    for (int i = 0; i < mesh.num_entities(1); ++i)
    {
      // Get the two vertices as points
      const mesh::MeshEntity e(mesh, 1, i);
      const std::int32_t* vertices = e.entities(0);
      Eigen::Vector3d p0 = geometry.x(vertices[0]);
      Eigen::Vector3d p1 = geometry.x(vertices[1]);

      // Define normal by rotating tangent counter-clockwise
      Eigen::Vector3d t = p1 - p0;
      n.row(i) = Eigen::Vector3d(-t[1], t[0], 0.0).normalized();
    }
    return n;
  }
  case (mesh::CellType::triangle):
  {
    Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> n(
        mesh.num_entities(2), 3);
    for (int i = 0; i < mesh.num_entities(2); ++i)
    {
      // Get the three vertices as points
      const mesh::MeshEntity e(mesh, 2, i);
      const std::int32_t* vertices = e.entities(0);
      const Eigen::Vector3d p0 = geometry.x(vertices[0]);
      const Eigen::Vector3d p1 = geometry.x(vertices[1]);
      const Eigen::Vector3d p2 = geometry.x(vertices[2]);

      // Define cell normal via cross product of first two edges
      n.row(i) = ((p1 - p0).cross(p2 - p0)).normalized();
    }
    return n;
  }
  case (mesh::CellType::quadrilateral):
  {
    // TODO: check
    Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> n(
        mesh.num_entities(2), 3);
    for (int i = 0; i < mesh.num_entities(2); ++i)
    {
      // Get three vertices as points
      const mesh::MeshEntity e(mesh, 2, i);
      const std::int32_t* vertices = e.entities(0);
      const Eigen::Vector3d p0 = geometry.x(vertices[0]);
      const Eigen::Vector3d p1 = geometry.x(vertices[1]);
      const Eigen::Vector3d p2 = geometry.x(vertices[2]);

      // Defined cell normal via cross product of first two edges:
      n.row(i) = ((p1 - p0).cross(p2 - p0)).normalized();
    }
    return n;
  }
  default:
    throw std::invalid_argument(
        "cell_normal not supported for this cell type.");
  }
  return Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>();
}
//-----------------------------------------------------------------------------
Eigen::Vector3d mesh::normal(const mesh::MeshEntity& cell, int facet_local)
{
  const mesh::Geometry& geometry = cell.mesh().geometry();
  const mesh::CellType type = cell.mesh().cell_type();
  const int tdim = cell.mesh().topology().dim();
  assert(cell.dim() == tdim);

  switch (type)
  {
  case (mesh::CellType::interval):
  {
    const std::int32_t* vertices = cell.entities(0);
    Eigen::Vector3d n = geometry.x(vertices[0]) - geometry.x(vertices[1]);
    n.normalize();
    if (facet_local == 1)
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
    mesh::MeshEntity f(cell.mesh(), tdim - 1, cell.entities(1)[facet_local]);

    // Get global index of opposite vertex
    const std::int32_t v0 = cell.entities(0)[facet_local];

    // Get global index of vertices on the facet
    const std::int32_t v1 = f.entities(0)[0];
    const std::int32_t v2 = f.entities(0)[1];

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
    if (cell.mesh().geometry().dim() != 2)
      throw std::runtime_error("Illegal geometric dimension");

    // Make sure we have facets
    cell.mesh().create_connectivity(2, 1);

    // Create facet from the mesh and local facet number
    MeshEntity f(cell.mesh(), tdim - 1, cell.entities(1)[facet_local]);

    // Get global index of opposite vertex
    const std::int32_t v0 = cell.entities(0)[facet_local];

    // Get global index of vertices on the facet
    const std::int32_t v1 = f.entities(0)[0];
    const std::int32_t v2 = f.entities(0)[1];

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
    MeshEntity f(cell.mesh(), tdim - 1, cell.entities(2)[facet_local]);

    // Get global index of opposite vertex
    const std::int32_t v0 = cell.entities(0)[facet_local];

    // Get global index of vertices on the facet
    const std::int32_t v1 = f.entities(0)[0];
    const std::int32_t v2 = f.entities(0)[1];
    const std::int32_t v3 = f.entities(0)[2];

    // Get the coordinates of the four vertices
    const Eigen::Vector3d p0 = geometry.x(v0);
    const Eigen::Vector3d p1 = geometry.x(v1);
    const Eigen::Vector3d p2 = geometry.x(v2);
    const Eigen::Vector3d p3 = geometry.x(v3);

    // Compute normal vector
    Eigen::Vector3d n = (p2 - p1).cross(p3 - p1);

    // Normalize
    n.normalize();

    // Flip direction of normal so it points outward
    if (n.dot(p0 - p1) > 0)
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
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> mesh::midpoints(
    const mesh::Mesh& mesh, int dim,
    const Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, 1>>& entities)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points
      = geometry.points();

  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> x(entities.rows(),
                                                             3);

  // Special case: a vertex is its own midpoint
  if (dim == 0)
  {
    for (Eigen::Index e = 0; e < entities.rows(); ++e)
      x.row(e) = points.row(entities[e]);
  }
  else
  {
    const mesh::Topology& topology = mesh.topology();
    assert(topology.connectivity(dim, 0));
    std::shared_ptr<const mesh::Connectivity> connectivity
        = topology.connectivity(dim, 0);
    const int num_vertices
        = mesh::cell_num_entities(cell_entity_type(mesh.cell_type(), dim), 0);
    for (Eigen::Index e = 0; e < entities.rows(); ++e)
    {
      const std::int32_t* vertices = connectivity->connections(entities[e]);
      x.row(e) = 0.0;
      for (int i = 0; i < num_vertices; ++i)
        x.row(e) += points.row(vertices[i]);
    }
    x /= num_vertices;
  }

  return x;
}
//-----------------------------------------------------------------------------
