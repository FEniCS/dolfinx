// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Geometry.h"
#include "MeshTags.h"
#include "cell_types.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <stdexcept>
#include <unordered_set>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
template <typename T>
T volume_interval(const mesh::Mesh& mesh,
                  const Eigen::Ref<const Eigen::ArrayXi>& entities)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();

  T v(entities.rows());
  for (Eigen::Index i = 0; i < entities.rows(); ++i)
  {
    // Get the coordinates of the two vertices
    auto dofs = x_dofs.links(entities[i]);
    const Eigen::Vector3d x0 = geometry.node(dofs[0]);
    const Eigen::Vector3d x1 = geometry.node(dofs[1]);
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
  const int gdim = geometry.dim();
  assert(gdim == 2 or gdim == 3);
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();

  T v(entities.rows());
  if (gdim == 2)
  {
    for (Eigen::Index i = 0; i < entities.rows(); ++i)
    {
      auto dofs = x_dofs.links(entities[i]);
      const Eigen::Vector3d x0 = geometry.node(dofs[0]);
      const Eigen::Vector3d x1 = geometry.node(dofs[1]);
      const Eigen::Vector3d x2 = geometry.node(dofs[2]);

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
      auto dofs = x_dofs.links(entities[i]);
      const Eigen::Vector3d x0 = geometry.node(dofs[0]);
      const Eigen::Vector3d x1 = geometry.node(dofs[1]);
      const Eigen::Vector3d x2 = geometry.node(dofs[2]);

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
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();

  Eigen::ArrayXd v(entities.rows());
  for (Eigen::Index i = 0; i < entities.rows(); ++i)
  {
    auto dofs = x_dofs.links(entities[i]);
    const Eigen::Vector3d x0 = geometry.node(dofs[0]);
    const Eigen::Vector3d x1 = geometry.node(dofs[1]);
    const Eigen::Vector3d x2 = geometry.node(dofs[2]);
    const Eigen::Vector3d x3 = geometry.node(dofs[3]);

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
  const int gdim = geometry.dim();
  T v(entities.rows());
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();

  for (Eigen::Index e = 0; e < entities.rows(); ++e)
  {
    // Get the coordinates of the four vertices
    auto dofs = x_dofs.links(entities[e]);
    const Eigen::Vector3d p0 = geometry.node(dofs[0]);
    const Eigen::Vector3d p1 = geometry.node(dofs[1]);
    const Eigen::Vector3d p2 = geometry.node(dofs[2]);
    const Eigen::Vector3d p3 = geometry.node(dofs[3]);

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

    v[e] = volume;
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
  const mesh::CellType type
      = cell_entity_type(mesh.topology().cell_type(), dim);
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
    assert(mesh.topology().dim() == dim);
    return volume_triangle<T>(mesh, entities);
  case mesh::CellType::tetrahedron:
    return volume_tetrahedron<T>(mesh, entities);
  case mesh::CellType::quadrilateral:
    assert(mesh.topology().dim() == dim);
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
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();

  T volumes = volume_entities_tmpl<T>(mesh, entities, 2);
  T cr(entities.rows());
  for (Eigen::Index e = 0; e < entities.rows(); ++e)
  {
    auto dofs = x_dofs.links(entities[e]);
    const Eigen::Vector3d p0 = geometry.node(dofs[0]);
    const Eigen::Vector3d p1 = geometry.node(dofs[1]);
    const Eigen::Vector3d p2 = geometry.node(dofs[2]);

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
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();
  T volumes = volume_entities_tmpl<T>(mesh, entities, 3);

  T cr(entities.rows());
  for (Eigen::Index e = 0; e < entities.rows(); ++e)
  {
    auto dofs = x_dofs.links(entities[e]);
    const Eigen::Vector3d p0 = geometry.node(dofs[0]);
    const Eigen::Vector3d p1 = geometry.node(dofs[1]);
    const Eigen::Vector3d p2 = geometry.node(dofs[2]);
    const Eigen::Vector3d p3 = geometry.node(dofs[3]);

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
  const mesh::CellType type
      = cell_entity_type(mesh.topology().cell_type(), dim);
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
graph::AdjacencyList<std::int64_t>
mesh::extract_topology(const CellType& cell_type,
                       const fem::ElementDofLayout& layout,
                       const graph::AdjacencyList<std::int64_t>& cells)
{
  // Use ElementDofLayout to get vertex dof indices (local to a cell)
  const int num_vertices_per_cell = num_cell_vertices(cell_type);
  std::vector<int> local_vertices(num_vertices_per_cell);
  for (int i = 0; i < num_vertices_per_cell; ++i)
  {
    const Eigen::Array<int, Eigen::Dynamic, 1> local_index
        = layout.entity_dofs(0, i);
    assert(local_index.rows() == 1);
    local_vertices[i] = local_index[0];
  }

  // Extract vertices
  Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      topology(cells.num_nodes(), num_vertices_per_cell);
  for (int i = 0; i < cells.num_nodes(); ++i)
  {
    auto p = cells.links(i);
    for (int j = 0; j < num_vertices_per_cell; ++j)
      topology(i, j) = p(local_vertices[j]);
  }

  return graph::AdjacencyList<std::int64_t>(topology);
}
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
  if (dim != mesh.topology().dim())
    throw std::runtime_error("Cell size when dim ne tdim  requires updating.");

  // Get number of cell vertices
  const mesh::CellType type
      = cell_entity_type(mesh.topology().cell_type(), dim);
  const int num_vertices = num_cell_vertices(type);

  const mesh::Geometry& geometry = mesh.geometry();
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();

  Eigen::ArrayXd h_cells = Eigen::ArrayXd::Zero(entities.rows());
  assert(num_vertices <= 8);
  std::array<Eigen::Vector3d, 8> points;
  for (Eigen::Index e = 0; e < entities.rows(); ++e)
  {
    // Get the coordinates  of the vertices
    auto dofs = x_dofs.links(entities[e]);
    for (int i = 0; i < num_vertices; ++i)
      points[i] = geometry.node(dofs[i]);

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
  const mesh::CellType type = mesh.topology().cell_type();

  // Check cell type
  if (!mesh::is_simplex(type))
  {
    throw std::runtime_error(
        "inradius function not implemented for non-simplicial cells");
  }

  // Get cell dimension
  const int d = mesh::cell_dim(type);
  const mesh::Topology& topology = mesh.topology();
  // FIXME: cleanup these calls as part of topology storage management rework.
  mesh.topology_mutable().create_entities(d - 1);
  auto connectivity = topology.connectivity(d, d - 1);
  assert(connectivity);

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

    auto facets = connectivity->links(entities[c]);
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
  const mesh::CellType type = mesh.topology().cell_type();
  const int dim = mesh::cell_dim(type);
  Eigen::ArrayXd r = mesh::inradius(mesh, entities);
  Eigen::ArrayXd cr = mesh::circumradius(mesh, entities, dim);
  return mesh::cell_dim(mesh.topology().cell_type()) * r / cr;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
mesh::cell_normals(const mesh::Mesh& mesh, int dim)
{
  const int gdim = mesh.geometry().dim();
  const mesh::CellType type
      = mesh::cell_entity_type(mesh.topology().cell_type(), dim);
  const mesh::Geometry& geometry = mesh.geometry();

  switch (type)
  {
  case (mesh::CellType::interval):
  {
    if (gdim > 2)
      throw std::invalid_argument("Interval cell normal undefined in 3D");
    auto map = mesh.topology().index_map(1);
    assert(map);
    const std::int32_t num_cells = map->size_local() + map->num_ghosts();
    Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> n(num_cells, 3);
    for (int i = 0; i < num_cells; ++i)
    {
      // Get the two vertices as points
      auto vertices = mesh.topology().connectivity(1, 0)->links(i);
      Eigen::Vector3d p0 = geometry.node(vertices[0]);
      Eigen::Vector3d p1 = geometry.node(vertices[1]);

      // Define normal by rotating tangent counter-clockwise
      Eigen::Vector3d t = p1 - p0;
      n.row(i) = Eigen::Vector3d(-t[1], t[0], 0.0).normalized();
    }
    return n;
  }
  case (mesh::CellType::triangle):
  {
    auto map = mesh.topology().index_map(2);
    assert(map);
    const std::int32_t num_cells = map->size_local() + map->num_ghosts();
    Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> n(num_cells, 3);
    for (int i = 0; i < num_cells; ++i)
    {
      // Get the three vertices as points
      auto vertices = mesh.topology().connectivity(2, 0)->links(i);
      const Eigen::Vector3d p0 = geometry.node(vertices[0]);
      const Eigen::Vector3d p1 = geometry.node(vertices[1]);
      const Eigen::Vector3d p2 = geometry.node(vertices[2]);

      // Define cell normal via cross product of first two edges
      n.row(i) = ((p1 - p0).cross(p2 - p0)).normalized();
    }
    return n;
  }
  case (mesh::CellType::quadrilateral):
  {
    // TODO: check
    auto map = mesh.topology().index_map(2);
    assert(map);
    const std::int32_t num_cells = map->size_local() + map->num_ghosts();
    Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> n(num_cells, 3);
    for (int i = 0; i < num_cells; ++i)
    {
      // Get three vertices as points
      auto vertices = mesh.topology().connectivity(2, 0)->links(i);
      const Eigen::Vector3d p0 = geometry.node(vertices[0]);
      const Eigen::Vector3d p1 = geometry.node(vertices[1]);
      const Eigen::Vector3d p2 = geometry.node(vertices[2]);

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
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> mesh::midpoints(
    const mesh::Mesh& mesh, int dim,
    const Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, 1>>& entities)
{
  const mesh::Topology& topology = mesh.topology();
  const mesh::Geometry& geometry = mesh.geometry();
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x
      = geometry.x();

  const int tdim = topology.dim();

  // Get geometry dofmap
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // Build map from vertex -> geometry dof
  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  auto map_v = topology.index_map(0);
  assert(map_v);
  const std::int32_t num_vertices = map_v->size_local() + map_v->num_ghosts();
  std::vector<std::int32_t> vertex_to_x(num_vertices);
  auto map_c = topology.index_map(tdim);
  assert(map_c);
  for (int c = 0; c < map_c->size_local() + map_c->num_ghosts(); ++c)
  {
    auto vertices = c_to_v->links(c);
    auto dofs = x_dofmap.links(c);
    for (int i = 0; i < vertices.rows(); ++i)
    {
      // FIXME: We are making an assumption here on the
      // ElementDofLayout. We should use an ElementDofLayout to map
      // between local vertex index an x dof index.
      vertex_to_x[vertices[i]] = dofs(i);
    }
  }

  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> x_mid(
      entities.rows(), 3);

  // Special case: a vertex is its own midpoint
  if (dim == 0)
  {
    for (Eigen::Index e = 0; e < entities.rows(); ++e)
      x_mid.row(e) = x.row(vertex_to_x[e]);
  }
  else
  {
    // FIXME: This assumes a linear geometry.
    auto e_to_v = topology.connectivity(dim, 0);
    assert(e_to_v);
    for (Eigen::Index e = 0; e < entities.rows(); ++e)
    {
      auto vertices = e_to_v->links(entities[e]);
      x_mid.row(e) = 0.0;
      for (int i = 0; i < vertices.rows(); ++i)
        x_mid.row(e) += x.row(vertex_to_x[vertices[i]]);
      x_mid.row(e) /= vertices.rows();
    }
  }

  return x_mid;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1> mesh::locate_entities(
    const mesh::Mesh& mesh, const int dim,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker)
{
  const mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();

  // Create entities and connectivities
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(tdim, 0);
  if (dim < tdim)
    mesh.topology_mutable().create_connectivity(dim, 0);

  // Get all vertex 'node' indices
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const std::int32_t num_vertices = topology.index_map(0)->size_local()
                                    + topology.index_map(0)->num_ghosts();
  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  std::vector<std::int32_t> vertex_to_node(num_vertices);
  for (int c = 0; c < c_to_v->num_nodes(); ++c)
  {
    auto x_dofs = x_dofmap.links(c);
    auto vertices = c_to_v->links(c);
    for (int i = 0; i < vertices.size(); ++i)
      vertex_to_node[vertices[i]] = x_dofs[i];
  }

  // Pack coordinates of vertices
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_nodes
      = mesh.geometry().x();
  Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor> x_vertices(
      3, vertex_to_node.size());
  for (std::size_t i = 0; i < vertex_to_node.size(); ++i)
    x_vertices.col(i) = x_nodes.row(vertex_to_node[i]);

  // Run marker function on vertex coordinates
  const Eigen::Array<bool, Eigen::Dynamic, 1> marked = marker(x_vertices);
  if (marked.rows() != x_vertices.cols())
    throw std::runtime_error("Length of array of markers is wrong.");

  // Iterate over entities to build vector of marked entities
  auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);
  std::vector<std::int32_t> entities;
  for (int e = 0; e < e_to_v->num_nodes(); ++e)
  {
    // Iterate over entity vertices
    bool all_vertices_marked = true;
    auto vertices = e_to_v->links(e);
    for (int i = 0; i < vertices.rows(); ++i)
    {
      if (!marked[vertices[i]])
      {
        all_vertices_marked = false;
        break;
      }
    }

    if (all_vertices_marked)
      entities.push_back(e);
  }

  return Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(
      entities.data(), entities.size());
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1> mesh::locate_entities_boundary(
    const mesh::Mesh& mesh, const int dim,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker)
{
  const mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();
  if (dim == tdim)
  {
    throw std::runtime_error(
        "Cannot use mesh::locate_entities_boundary (boundary) for cells.");
  }

  // Compute marker for boundary facets
  mesh.topology_mutable().create_entities(tdim - 1);
  mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
  const std::vector boundary_facet = mesh::compute_boundary_facets(topology);

  // Create entities and connectivities
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(tdim - 1, dim);
  mesh.topology_mutable().create_connectivity(tdim - 1, 0);
  mesh.topology_mutable().create_connectivity(0, tdim);
  mesh.topology_mutable().create_connectivity(tdim, 0);

  // Build set of vertices on boundary and set of boundary entities
  auto f_to_v = topology.connectivity(tdim - 1, 0);
  assert(f_to_v);
  auto f_to_e = topology.connectivity(tdim - 1, dim);
  assert(f_to_e);
  std::unordered_set<std::int32_t> boundary_vertices;
  std::unordered_set<std::int32_t> facet_entities;
  for (std::size_t f = 0; f < boundary_facet.size(); ++f)
  {
    if (boundary_facet[f])
    {
      auto entities = f_to_e->links(f);
      for (int i = 0; i < entities.size(); ++i)
        facet_entities.insert(entities[i]);

      auto vertices = f_to_v->links(f);
      for (int i = 0; i < vertices.size(); ++i)
        boundary_vertices.insert(vertices[i]);
    }
  }

  // Get geometry data
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_nodes
      = mesh.geometry().x();

  // Build vector of boundary vertices
  const std::vector<std::int32_t> vertices(boundary_vertices.begin(),
                                           boundary_vertices.end());

  // Get all vertex 'node' indices
  auto v_to_c = topology.connectivity(0, tdim);
  assert(v_to_c);
  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor> x_vertices(
      3, vertices.size());
  std::vector<std::int32_t> vertex_to_pos(v_to_c->num_nodes(), -1);
  for (std::size_t i = 0; i < vertices.size(); ++i)
  {
    const std::int32_t v = vertices[i];

    // Get first cell and find position
    const int c = v_to_c->links(v)[0];
    auto vertices = c_to_v->links(c);
    const auto* it
        = std::find(vertices.data(), vertices.data() + vertices.rows(), v);
    assert(it != (vertices.data() + vertices.rows()));
    const int local_pos = std::distance(vertices.data(), it);

    auto dofs = x_dofmap.links(c);
    x_vertices.col(i) = x_nodes.row(dofs[local_pos]);

    vertex_to_pos[v] = i;
  }

  // Run marker function on the vertex coordinates
  const Eigen::Array<bool, Eigen::Dynamic, 1> marked = marker(x_vertices);
  if (marked.size() != x_vertices.cols())
    throw std::runtime_error("Length of array of markers is wrong.");

  // Loop over entities and check vertex markers
  auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);
  std::vector<std::int32_t> entities;
  for (auto e : facet_entities)
  {
    // Assume all vertices on this entity are marked
    bool all_vertices_marked = true;

    // Iterate over entity vertices
    auto vertices = e_to_v->links(e);
    for (int i = 0; i < vertices.rows(); ++i)
    {
      const std::int32_t idx = vertices[i];
      const std::int32_t pos = vertex_to_pos[idx];
      if (!marked[pos])
      {
        all_vertices_marked = false;
        break;
      }
    }

    // Mark facet with all vertices marked
    if (all_vertices_marked)
      entities.push_back(e);
  }

  return Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(
      entities.data(), entities.size());
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
mesh::entities_to_geometry(
    const mesh::Mesh& mesh, const int dim,
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& entity_list,
    bool orient)
{
  dolfinx::mesh::CellType cell_type = mesh.topology().cell_type();
  const int num_entity_vertices
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, dim));
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      entity_geometry(entity_list.size(), num_entity_vertices);

  if (orient
      and (cell_type != dolfinx::mesh::CellType::tetrahedron or dim != 2))
    throw std::runtime_error("Can only orient facets of a tetrahedral mesh");

  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();

  const int tdim = topology.dim();
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(dim, tdim);
  mesh.topology_mutable().create_connectivity(dim, 0);
  mesh.topology_mutable().create_connectivity(tdim, 0);

  const graph::AdjacencyList<std::int32_t>& xdofs = geometry.dofmap();
  const auto e_to_c = topology.connectivity(dim, tdim);
  assert(e_to_c);
  const auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);
  const auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  for (int i = 0; i < entity_list.size(); ++i)
  {
    const std::int32_t idx = entity_list[i];
    const std::int32_t cell = e_to_c->links(idx)[0];
    const auto ev = e_to_v->links(idx);
    assert(ev.size() == num_entity_vertices);
    const auto cv = c_to_v->links(cell);
    const auto xc = xdofs.links(cell);
    for (int j = 0; j < num_entity_vertices; ++j)
    {
      int k = std::distance(cv.data(),
                            std::find(cv.data(), cv.data() + cv.size(), ev[j]));
      assert(k < cv.size());
      entity_geometry(i, j) = xc[k];
    }

    if (orient)
    {
      // Compute cell midpoint
      Eigen::Vector3d midpoint(0.0, 0.0, 0.0);
      for (int j = 0; j < xc.size(); ++j)
        midpoint += geometry.node(xc[j]);
      midpoint /= xc.size();
      // Compute vector triple product of two edges and vector to midpoint
      Eigen::Vector3d p0 = geometry.node(entity_geometry(i, 0));
      Eigen::Matrix3d a;
      a.row(0) = midpoint - p0;
      a.row(1) = geometry.node(entity_geometry(i, 1)) - p0;
      a.row(2) = geometry.node(entity_geometry(i, 2)) - p0;
      // Midpoint direction should be opposite to normal, hence this should be
      // negative. Switch points if not.
      if (a.determinant() > 0.0)
        std::swap(entity_geometry(i, 1), entity_geometry(i, 2));
    }
  }

  return entity_geometry;
}
//------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
mesh::exterior_facet_indices(const Mesh& mesh)
{
  // Note: Possible duplication of mesh::Topology::compute_boundary_facets

  const mesh::Topology& topology = mesh.topology();
  std::vector<std::int32_t> surface_facets;

  // Get number of facets owned by this process
  const int tdim = topology.dim();
  mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
  auto f_to_c = topology.connectivity(tdim - 1, tdim);
  assert(topology.index_map(tdim - 1));
  std::set<std::int32_t> fwd_shared_facets;

  // Only need to consider shared facets when there are no ghost cells
  if (topology.index_map(tdim)->num_ghosts() == 0)
  {
    fwd_shared_facets.insert(
        topology.index_map(tdim - 1)->shared_indices().begin(),
        topology.index_map(tdim - 1)->shared_indices().end());
  }

  // Find all owned facets (not ghost) with only one attached cell, which are
  // also not shared forward (ghost on another process)
  const int num_facets = topology.index_map(tdim - 1)->size_local();
  for (int f = 0; f < num_facets; ++f)
  {
    if (f_to_c->num_links(f) == 1
        and fwd_shared_facets.find(f) == fwd_shared_facets.end())
      surface_facets.push_back(f);
  }

  // Copy over to Eigen::Array
  return Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(
      surface_facets.data(), surface_facets.size());
}
//------------------------------------------------------------------------------