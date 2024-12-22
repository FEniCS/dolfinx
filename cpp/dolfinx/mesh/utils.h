// Copyright (C) 2019-2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "Topology.h"
#include "graphbuild.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <concepts>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <functional>
#include <mpi.h>
#include <numeric>
#include <span>

/// @file utils.h
/// @brief Functions supporting mesh operations

namespace dolfinx::fem
{
class ElementDofLayout;
}

namespace dolfinx::mesh
{
enum class CellType;

/// Enum for different partitioning ghost modes
enum class GhostMode : int
{
  none,
  shared_facet,
  shared_vertex
};

namespace impl
{
/// @brief Re-order an adjacency list of fixed degree.
/// @param[in,out] list Fixed-degree adjacency list stored row-major.
/// Degree is equal to `list.size() / nodemap.size()`.
/// @param[in] nodemap Map from old to new index, i.e. fro old index `i`
/// its new index is `nodemap[i]`.
template <typename T>
void reorder_list(std::span<T> list, std::span<const std::int32_t> nodemap)
{
  if (nodemap.empty())
    return;

  assert(list.size() % nodemap.size() == 0);
  std::size_t degree = list.size() / nodemap.size();
  const std::vector<T> orig(list.begin(), list.end());
  for (std::size_t n = 0; n < nodemap.size(); ++n)
  {
    std::span links_old(orig.data() + n * degree, degree);
    auto links_new = list.subspan(nodemap[n] * degree, degree);
    std::ranges::copy(links_old, links_new.begin());
  }
}

/// @brief Compute the coordinates of 'vertices' for entities of a given
/// dimension that are attached to specified facets.
///
/// @pre The provided facets must be on the boundary of the mesh.
///
/// @param[in] mesh Mesh to compute the vertex coordinates for.
/// @param[in] dim Topological dimension of the entities.
/// @param[in] facets List of facets (must be on the mesh boundary).
/// @return (0) Entities attached to the boundary facets (sorted), (1)
/// vertex coordinates (shape is `(3, num_vertices)`) and (2) map from
/// vertex in the full mesh to the position in the vertex coordinates
/// array (set to -1 if vertex in full mesh is not in the coordinate
/// array).
template <std::floating_point T>
std::tuple<std::vector<std::int32_t>, std::vector<T>, std::vector<std::int32_t>>
compute_vertex_coords_boundary(const mesh::Mesh<T>& mesh, int dim,
                               std::span<const std::int32_t> facets)
{
  auto topology = mesh.topology();
  assert(topology);
  const int tdim = topology->dim();
  if (dim == tdim)
  {
    throw std::runtime_error(
        "Cannot use mesh::locate_entities_boundary (boundary) for cells.");
  }

  // Build set of vertices on boundary and set of boundary entities
  mesh.topology_mutable()->create_connectivity(tdim - 1, 0);
  mesh.topology_mutable()->create_connectivity(tdim - 1, dim);
  std::vector<std::int32_t> vertices, entities;
  {
    auto f_to_v = topology->connectivity(tdim - 1, 0);
    assert(f_to_v);
    auto f_to_e = topology->connectivity(tdim - 1, dim);
    assert(f_to_e);
    for (auto f : facets)
    {
      auto v = f_to_v->links(f);
      vertices.insert(vertices.end(), v.begin(), v.end());
      auto e = f_to_e->links(f);
      entities.insert(entities.end(), e.begin(), e.end());
    }

    // Build vector of boundary vertices
    {
      std::ranges::sort(vertices);
      auto [unique_end, range_end] = std::ranges::unique(vertices);
      vertices.erase(unique_end, range_end);
    }

    {
      std::ranges::sort(entities);
      auto [unique_end, range_end] = std::ranges::unique(entities);
      entities.erase(unique_end, range_end);
    }
  }

  // Get geometry data
  auto x_dofmap = mesh.geometry().dofmap();
  std::span<const T> x_nodes = mesh.geometry().x();

  // Get all vertex 'node' indices
  mesh.topology_mutable()->create_connectivity(0, tdim);
  mesh.topology_mutable()->create_connectivity(tdim, 0);
  auto v_to_c = topology->connectivity(0, tdim);
  assert(v_to_c);
  auto c_to_v = topology->connectivity(tdim, 0);
  assert(c_to_v);
  std::vector<T> x_vertices(3 * vertices.size(), -1.0);
  std::vector<std::int32_t> vertex_to_pos(v_to_c->num_nodes(), -1);
  for (std::size_t i = 0; i < vertices.size(); ++i)
  {
    const std::int32_t v = vertices[i];

    // Get first cell and find position
    const std::int32_t c = v_to_c->links(v).front();
    auto cell_vertices = c_to_v->links(c);
    auto it = std::find(cell_vertices.begin(), cell_vertices.end(), v);
    assert(it != cell_vertices.end());
    const std::size_t local_pos = std::distance(cell_vertices.begin(), it);

    auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t j = 0; j < 3; ++j)
      x_vertices[j * vertices.size() + i] = x_nodes[3 * dofs[local_pos] + j];
    vertex_to_pos[v] = i;
  }

  return {std::move(entities), std::move(x_vertices), std::move(vertex_to_pos)};
}

} // namespace impl

/// @brief Compute the indices of all exterior facets that are owned by
/// the caller.
///
/// An exterior facet (co-dimension 1) is one that is connected globally
/// to only one cell of co-dimension 0).
///
/// @note Collective.
///
/// @param[in] topology Mesh topology.
/// @return Sorted list of owned facet indices that are exterior facets
/// of the mesh.
std::vector<std::int32_t> exterior_facet_indices(const Topology& topology);

/// @brief Signature for the cell partitioning function. Function that
/// implement this interface compute the destination rank for cells
/// currently on this rank.
///
/// @param[in] comm MPI Communicator.
/// @param[in] nparts Number of partitions.
/// @param[in] cell_types Cell types in the mesh.
/// @param[in] cells Lists of cells of each cell type. `cells[i]` is a
/// flattened row major 2D array of shape (num_cells, num_cell_vertices)
/// for `cell_types[i]` on this process, containing the global indices
/// for the cell vertices. Each cell can appear only once across all
/// processes. The cell vertex indices are not necessarily contiguous
/// globally, i.e. the maximum index across all processes can be greater
/// than the number of vertices. High-order 'nodes', e.g. mid-side
/// points, should not be included.
/// @return Destination ranks for each cell on this process.
/// @note Cells can have multiple destination ranks, when ghosted.
using CellPartitionFunction = std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm comm, int nparts, const std::vector<CellType>& cell_types,
    const std::vector<std::span<const std::int64_t>>& cells)>;

/// @brief Extract topology from cell data, i.e. extract cell vertices.
/// @param[in] cell_type The cell shape
/// @param[in] layout The layout of geometry 'degrees-of-freedom' on the
/// reference cell.
/// @param[in] cells List of 'nodes' for each cell using global indices.
/// The layout must be consistent with `layout`.
/// @return Cell topology. The global indices will, in general, have
/// 'gaps' due to mid-side and other higher-order nodes being removed
/// from the input `cell`.
std::vector<std::int64_t> extract_topology(CellType cell_type,
                                           const fem::ElementDofLayout& layout,
                                           std::span<const std::int64_t> cells);

/// @brief Compute greatest distance between any two vertices of the
/// mesh entities (`h`).
/// @param[in] mesh Mesh that the entities belong to.
/// @param[in] entities Indices (local to process) of entities to
/// compute `h` for.
/// @param[in] dim Topological dimension of the entities.
/// @returns Greatest distance between any two vertices, `h[i]`
/// corresponds to the entity `entities[i]`.
template <std::floating_point T>
std::vector<T> h(const Mesh<T>& mesh, std::span<const std::int32_t> entities,
                 int dim)
{
  if (entities.empty())
    return std::vector<T>();
  if (dim == 0)
    return std::vector<T>(entities.size(), 0);

  // Get the geometry dofs for the vertices of each entity
  const std::vector<std::int32_t> vertex_xdofs
      = entities_to_geometry(mesh, dim, entities, false);
  assert(!entities.empty());
  const std::size_t num_vertices = vertex_xdofs.size() / entities.size();

  // Get the  geometry coordinate
  std::span<const T> x = mesh.geometry().x();

  // Function to compute the length of (p0 - p1)
  auto delta_norm = [](auto&& p0, auto&& p1)
  {
    T norm = 0;
    for (std::size_t i = 0; i < 3; ++i)
      norm += (p0[i] - p1[i]) * (p0[i] - p1[i]);
    return std::sqrt(norm);
  };

  // Compute greatest distance between any to vertices
  assert(dim > 0);
  std::vector<T> h(entities.size(), 0);
  for (std::size_t e = 0; e < entities.size(); ++e)
  {
    // Get geometry 'dof' for each vertex of entity e
    std::span<const std::int32_t> e_vertices(
        vertex_xdofs.data() + e * num_vertices, num_vertices);

    // Compute maximum distance between any two vertices
    for (std::size_t i = 0; i < e_vertices.size(); ++i)
    {
      std::span<const T, 3> p0(x.data() + 3 * e_vertices[i], 3);
      for (std::size_t j = i + 1; j < e_vertices.size(); ++j)
      {
        std::span<const T, 3> p1(x.data() + 3 * e_vertices[j], 3);
        h[e] = std::max(h[e], delta_norm(p0, p1));
      }
    }
  }

  return h;
}

/// @brief Compute normal to given cell (viewed as embedded in 3D)
/// @returns The entity normals. The shape is `(entities.size(), 3)` and
/// the storage is row-major.
template <std::floating_point T>
std::vector<T> cell_normals(const Mesh<T>& mesh, int dim,
                            std::span<const std::int32_t> entities)
{
  auto topology = mesh.topology();
  assert(topology);

  if (entities.empty())
    return std::vector<T>();

  if (topology->cell_type() == CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cell");

  const int gdim = mesh.geometry().dim();
  const CellType type = cell_entity_type(topology->cell_type(), dim, 0);

  // Find geometry nodes for topology entities
  std::span<const T> x = mesh.geometry().x();
  std::vector<std::int32_t> geometry_entities
      = entities_to_geometry(mesh, dim, entities, false);

  const std::size_t shape1 = geometry_entities.size() / entities.size();
  std::vector<T> n(entities.size() * 3);
  switch (type)
  {
  case CellType::interval:
  {
    if (gdim > 2)
      throw std::invalid_argument("Interval cell normal undefined in 3D");
    for (std::size_t i = 0; i < entities.size(); ++i)
    {
      // Get the two vertices as points
      std::array vertices{geometry_entities[i * shape1],
                          geometry_entities[i * shape1 + 1]};
      std::array p = {std::span<const T, 3>(x.data() + 3 * vertices[0], 3),
                      std::span<const T, 3>(x.data() + 3 * vertices[1], 3)};

      // Define normal by rotating tangent counter-clockwise
      std::array<T, 3> t;
      std::ranges::transform(p[1], p[0], t.begin(),
                             [](auto x, auto y) { return x - y; });

      T norm = std::sqrt(t[0] * t[0] + t[1] * t[1]);
      std::span<T, 3> ni(n.data() + 3 * i, 3);
      ni[0] = -t[1] / norm;
      ni[1] = t[0] / norm;
      ni[2] = 0.0;
    }
    return n;
  }
  case CellType::triangle:
  {
    for (std::size_t i = 0; i < entities.size(); ++i)
    {
      // Get the three vertices as points
      std::array vertices = {geometry_entities[i * shape1 + 0],
                             geometry_entities[i * shape1 + 1],
                             geometry_entities[i * shape1 + 2]};
      std::array p = {std::span<const T, 3>(x.data() + 3 * vertices[0], 3),
                      std::span<const T, 3>(x.data() + 3 * vertices[1], 3),
                      std::span<const T, 3>(x.data() + 3 * vertices[2], 3)};

      // Compute (p1 - p0) and (p2 - p0)
      std::array<T, 3> dp1, dp2;
      std::ranges::transform(p[1], p[0], dp1.begin(),
                             [](auto x, auto y) { return x - y; });
      std::ranges::transform(p[2], p[0], dp2.begin(),
                             [](auto x, auto y) { return x - y; });

      // Define cell normal via cross product of first two edges
      std::array<T, 3> ni = math::cross(dp1, dp2);
      T norm = std::sqrt(ni[0] * ni[0] + ni[1] * ni[1] + ni[2] * ni[2]);
      std::ranges::transform(ni, std::next(n.begin(), 3 * i),
                             [norm](auto x) { return x / norm; });
    }

    return n;
  }
  case CellType::quadrilateral:
  {
    // TODO: check
    for (std::size_t i = 0; i < entities.size(); ++i)
    {
      // Get the three vertices as points
      std::array vertices = {geometry_entities[i * shape1 + 0],
                             geometry_entities[i * shape1 + 1],
                             geometry_entities[i * shape1 + 2]};
      std::array p = {std::span<const T, 3>(x.data() + 3 * vertices[0], 3),
                      std::span<const T, 3>(x.data() + 3 * vertices[1], 3),
                      std::span<const T, 3>(x.data() + 3 * vertices[2], 3)};

      // Compute (p1 - p0) and (p2 - p0)
      std::array<T, 3> dp1, dp2;
      std::ranges::transform(p[1], p[0], dp1.begin(),
                             [](auto x, auto y) { return x - y; });
      std::ranges::transform(p[2], p[0], dp2.begin(),
                             [](auto x, auto y) { return x - y; });

      // Define cell normal via cross product of first two edges
      std::array<T, 3> ni = math::cross(dp1, dp2);
      T norm = std::sqrt(ni[0] * ni[0] + ni[1] * ni[1] + ni[2] * ni[2]);
      std::ranges::transform(ni, std::next(n.begin(), 3 * i),
                             [norm](auto x) { return x / norm; });
    }

    return n;
  }
  default:
    throw std::invalid_argument(
        "cell_normal not supported for this cell type.");
  }
}

/// @brief Compute the midpoints for mesh entities of a given dimension.
/// @returns The entity midpoints. The shape is `(entities.size(), 3)`
/// and the storage is row-major.
template <std::floating_point T>
std::vector<T> compute_midpoints(const Mesh<T>& mesh, int dim,
                                 std::span<const std::int32_t> entities)
{
  if (entities.empty())
    return std::vector<T>();

  std::span<const T> x = mesh.geometry().x();

  // Build map from entity -> geometry dof
  const std::vector<std::int32_t> e_to_g
      = entities_to_geometry(mesh, dim, entities, false);
  std::size_t shape1 = e_to_g.size() / entities.size();

  std::vector<T> x_mid(entities.size() * 3, 0);
  for (std::size_t e = 0; e < entities.size(); ++e)
  {
    std::span<T, 3> p(x_mid.data() + 3 * e, 3);
    std::span<const std::int32_t> rows(e_to_g.data() + e * shape1, shape1);
    for (auto row : rows)
    {
      std::span<const T, 3> xg(x.data() + 3 * row, 3);
      std::ranges::transform(p, xg, p.begin(),
                             [size = rows.size()](auto x, auto y)
                             { return x + y / size; });
    }
  }

  return x_mid;
}

namespace impl
{
/// The coordinates for all 'vertices' in the mesh
/// @param[in] mesh Mesh to compute the vertex coordinates for
/// @return The vertex coordinates. The shape is `(3, num_vertices)` and
/// the jth column hold the coordinates of vertex j.
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
compute_vertex_coords(const mesh::Mesh<T>& mesh)
{
  auto topology = mesh.topology();
  assert(topology);
  const int tdim = topology->dim();

  // Create entities and connectivities
  mesh.topology_mutable()->create_connectivity(tdim, 0);

  // Get all vertex 'node' indices
  auto x_dofmap = mesh.geometry().dofmap();
  const std::int32_t num_vertices = topology->index_map(0)->size_local()
                                    + topology->index_map(0)->num_ghosts();
  auto c_to_v = topology->connectivity(tdim, 0);
  assert(c_to_v);
  std::vector<std::int32_t> vertex_to_node(num_vertices);
  for (int c = 0; c < c_to_v->num_nodes(); ++c)
  {
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto vertices = c_to_v->links(c);
    for (std::size_t i = 0; i < vertices.size(); ++i)
      vertex_to_node[vertices[i]] = x_dofs[i];
  }

  // Pack coordinates of vertices
  std::span<const T> x_nodes = mesh.geometry().x();
  std::vector<T> x_vertices(3 * vertex_to_node.size(), 0.0);
  for (std::size_t i = 0; i < vertex_to_node.size(); ++i)
  {
    std::int32_t pos = 3 * vertex_to_node[i];
    for (std::size_t j = 0; j < 3; ++j)
      x_vertices[j * vertex_to_node.size() + i] = x_nodes[pos + j];
  }

  return {std::move(x_vertices), {3, vertex_to_node.size()}};
}

} // namespace impl

/// Requirements on function for geometry marking
template <typename Fn, typename T>
concept MarkerFn = std::is_invocable_r<
    std::vector<std::int8_t>, Fn,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
                     std::size_t, 3,
                     MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>>::value;

/// @brief Compute indices of all mesh entities that evaluate to true
/// for the provided geometric marking function.
///
/// An entity is considered marked if the marker function evaluates to true
/// for all of its vertices.
///
/// @param[in] mesh Mesh to mark entities on.
/// @param[in] dim Topological dimension of the entities to be
/// considered.
/// @param[in] marker Marking function, returns `true` for a point that
/// is 'marked', and `false` otherwise.
/// @returns List of marked entity indices, including any ghost indices
/// (indices local to the process).
template <std::floating_point T, MarkerFn<T> U>
std::vector<std::int32_t> locate_entities(const Mesh<T>& mesh, int dim,
                                          U marker)
{
  using cmdspan3x_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T,
      MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
          std::size_t, 3, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>;

  // Run marker function on vertex coordinates
  const auto [xdata, xshape] = impl::compute_vertex_coords(mesh);
  cmdspan3x_t x(xdata.data(), xshape);
  const std::vector<std::int8_t> marked = marker(x);
  if (marked.size() != x.extent(1))
    throw std::runtime_error("Length of array of markers is wrong.");

  auto topology = mesh.topology();
  assert(topology);
  const int tdim = topology->dim();

  mesh.topology_mutable()->create_entities(dim);
  mesh.topology_mutable()->create_connectivity(tdim, 0);
  if (dim < tdim)
    mesh.topology_mutable()->create_connectivity(dim, 0);

  // Iterate over entities of dimension 'dim' to build vector of marked
  // entities
  auto e_to_v = topology->connectivity(dim, 0);
  assert(e_to_v);
  std::vector<std::int32_t> entities;
  for (int e = 0; e < e_to_v->num_nodes(); ++e)
  {
    // Iterate over entity vertices
    bool all_vertices_marked = true;
    for (std::int32_t v : e_to_v->links(e))
    {
      if (!marked[v])
      {
        all_vertices_marked = false;
        break;
      }
    }

    if (all_vertices_marked)
      entities.push_back(e);
  }

  return entities;
}

/// @brief Compute indices of all mesh entities that are attached to an
/// owned boundary facet and evaluate to true for the provided geometric
/// marking function.
///
/// An entity is considered marked if the marker function evaluates to
/// true for all of its vertices.
///
/// @note For vertices and edges, in parallel this function will not
/// necessarily mark all entities that are on the exterior boundary. For
/// example, it is possible for a process to have a vertex that lies on
/// the boundary without any of the attached facets being a boundary
/// facet. When used to find degrees-of-freedom, e.g. using
/// fem::locate_dofs_topological, the function that uses the data
/// returned by this function must typically perform some parallel
/// communication.
///
/// @param[in] mesh Mesh to mark entities on.
/// @param[in] dim Topological dimension of the entities to be
/// considered. Must be less than the topological dimension of the mesh.
/// @param[in] marker Marking function, returns `true` for a point that
/// is 'marked', and `false` otherwise.
/// @returns List of marked entity indices (indices local to the
/// process).
template <std::floating_point T, MarkerFn<T> U>
std::vector<std::int32_t> locate_entities_boundary(const Mesh<T>& mesh, int dim,
                                                   U marker)
{
  auto topology = mesh.topology();
  assert(topology);
  int tdim = topology->dim();
  if (dim == tdim)
  {
    throw std::runtime_error(
        "Cannot use mesh::locate_entities_boundary (boundary) for cells.");
  }

  // Compute list of boundary facets
  mesh.topology_mutable()->create_entities(tdim - 1);
  mesh.topology_mutable()->create_connectivity(tdim - 1, tdim);
  std::vector<std::int32_t> boundary_facets = exterior_facet_indices(*topology);

  using cmdspan3x_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T,
      MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
          std::size_t, 3, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>;

  // Run marker function on the vertex coordinates
  auto [facet_entities, xdata, vertex_to_pos]
      = impl::compute_vertex_coords_boundary(mesh, dim, boundary_facets);
  cmdspan3x_t x(xdata.data(), 3, xdata.size() / 3);
  std::vector<std::int8_t> marked = marker(x);
  if (marked.size() != x.extent(1))
    throw std::runtime_error("Length of array of markers is wrong.");

  // Loop over entities and check vertex markers
  mesh.topology_mutable()->create_entities(dim);
  auto e_to_v = topology->connectivity(dim, 0);
  assert(e_to_v);
  std::vector<std::int32_t> entities;
  for (auto e : facet_entities)
  {
    // Iterate over entity vertices
    bool all_vertices_marked = true;
    for (auto v : e_to_v->links(e))
    {
      const std::int32_t pos = vertex_to_pos[v];
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

  return entities;
}

/// @brief Compute the geometry degrees of freedom associated with
/// the closure of a given set of cell entities.
///
/// @param[in] mesh The mesh.
/// @param[in] dim Topological dimension of the entities of interest.
/// @param[in] entities Entity indices (local to process).
/// @param[in] permute If `true`, permute the DOFs such that they are
/// consistent with the orientation of `dim`-dimensional mesh entities.
/// This requires `create_entity_permutations` to be called first.
/// @return The geometry DOFs associated with the closure of each entity
/// in `entities`. The shape is `(num_entities, num_xdofs_per_entity)`
/// and the storage is row-major. The index `indices[i, j]` is the
/// position in the geometry array of the `j`-th vertex of the
/// `entity[i]`.
///
/// @pre The mesh connectivities `dim -> mesh.topology().dim()` and
/// `mesh.topology().dim() -> dim` must have been computed. Otherwise an
/// exception is thrown.
template <std::floating_point T>
std::vector<std::int32_t>
entities_to_geometry(const Mesh<T>& mesh, int dim,
                     std::span<const std::int32_t> entities,
                     bool permute = false)
{
  auto topology = mesh.topology();
  assert(topology);
  CellType cell_type = topology->cell_type();
  if (cell_type == CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cells");

  const int tdim = topology->dim();
  const Geometry<T>& geometry = mesh.geometry();
  auto xdofs = geometry.dofmap();

  // Get the DOF layout and the number of DOFs per entity
  const fem::CoordinateElement<T>& coord_ele = geometry.cmap();
  const fem::ElementDofLayout layout = coord_ele.create_dof_layout();
  const std::size_t num_entity_dofs = layout.num_entity_closure_dofs(dim);
  std::vector<std::int32_t> entity_xdofs;
  entity_xdofs.reserve(entities.size() * num_entity_dofs);

  // Get the element's closure DOFs
  const std::vector<std::vector<std::vector<int>>>& closure_dofs_all
      = layout.entity_closure_dofs_all();

  // Special case when dim == tdim (cells)
  if (dim == tdim)
  {
    for (std::size_t i = 0; i < entities.size(); ++i)
    {
      const std::int32_t c = entities[i];
      // Extract degrees of freedom
      auto x_c = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          xdofs, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::int32_t entity_dof : closure_dofs_all[tdim][0])
        entity_xdofs.push_back(x_c[entity_dof]);
    }
    return entity_xdofs;
  }

  assert(dim != tdim);

  auto e_to_c = topology->connectivity(dim, tdim);
  if (!e_to_c)
  {
    throw std::runtime_error(
        "Entity-to-cell connectivity has not been computed. Missing dims "
        + std::to_string(dim) + "->" + std::to_string(tdim));
  }

  auto c_to_e = topology->connectivity(tdim, dim);
  if (!c_to_e)
  {
    throw std::runtime_error(
        "Cell-to-entity connectivity has not been computed. Missing dims "
        + std::to_string(tdim) + "->" + std::to_string(dim));
  }

  // Get the cell info, which is needed to permute the closure dofs
  std::span<const std::uint32_t> cell_info;
  if (permute)
    cell_info = std::span(mesh.topology()->get_cell_permutation_info());

  for (std::size_t i = 0; i < entities.size(); ++i)
  {
    const std::int32_t e = entities[i];

    // Get a cell connected to the entity
    assert(!e_to_c->links(e).empty());
    std::int32_t c = e_to_c->links(e).front();

    // Get the local index of the entity
    std::span<const std::int32_t> cell_entities = c_to_e->links(c);
    auto it = std::find(cell_entities.begin(), cell_entities.end(), e);
    assert(it != cell_entities.end());
    std::size_t local_entity = std::distance(cell_entities.begin(), it);

    std::vector<std::int32_t> closure_dofs(closure_dofs_all[dim][local_entity]);

    // Cell sub-entities must be permuted so that their local orientation agrees
    // with their global orientation
    if (permute)
    {
      mesh::CellType entity_type
          = mesh::cell_entity_type(cell_type, dim, local_entity);
      coord_ele.permute_subentity_closure(closure_dofs, cell_info[c],
                                          entity_type, local_entity);
    }

    // Extract degrees of freedom
    auto x_c = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        xdofs, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::int32_t entity_dof : closure_dofs)
      entity_xdofs.push_back(x_c[entity_dof]);
  }

  return entity_xdofs;
}

/// @brief Create a function that computes destination rank for mesh
/// cells in this rank by applying the default graph partitioner to the
/// dual graph of the mesh.
/// @return Function that computes the destination ranks for each cell.
CellPartitionFunction create_cell_partitioner(mesh::GhostMode ghost_mode
                                              = mesh::GhostMode::none,
                                              const graph::partition_fn& partfn
                                              = &graph::partition_graph);

/// @brief Compute incident indices.
/// @param[in] topology The topology.
/// @param[in] entities List of indices of topological dimension `d0`.
/// @param[in] d0 Topological dimension.
/// @param[in] d1 Topological dimension.
/// @return List of entities of topological dimension `d1` that are
/// incident to entities in `entities` (topological dimension `d0`).
std::vector<std::int32_t>
compute_incident_entities(const Topology& topology,
                          std::span<const std::int32_t> entities, int d0,
                          int d1);

/// @brief Create a distributed mesh::Mesh from mesh data and using the
/// provided graph partitioning function for determining the parallel
/// distribution of the mesh.
///
/// The input cells and geometry data can be distributed across the
/// calling ranks, but must be not duplicated across ranks.
///
/// The function `partitioner` computes the parallel distribution, i.e.
/// the destination rank for each cell passed to the constructor. If
/// `partitioner`  is not callable, i.e. it does not store a callable
/// function, no parallel re-distribution of cells is performed.
///
/// @note Collective.
///
/// @param[in] comm Communicator to build the mesh on.
/// @param[in] commt Communicator that the topology data (`cells`) is
/// distributed on. This should be `MPI_COMM_NULL` for ranks that should
/// not participate in computing the topology partitioning.
/// @param[in] cells Cells, grouped by cell type with `cells[i]` being
/// the cells of the same type. Cells are defined by their 'nodes'
/// (using global indices) following the Basix ordering, and for each
/// cell type concatenated to form a flattened list. For lowest-order
/// cells this will be just the cell vertices. For higher-order geometry
/// cells, other cell 'nodes' will be included. See io::cells for
/// examples of the Basix ordering.
/// @param[in] elements Coordinate elements for the cells, where
/// `elements[i]` is the coordinate element for the cells in `cells[i]`.
/// **The list of elements must be the same on all calling parallel
/// ranks.**
/// @param[in] commg Communicator for geometry.
/// @param[in] x Geometry data ('node' coordinates). Row-major storage.
/// The global index of the `i`th node (row) in `x` is taken as `i` plus
/// the paralell rank offset (on `comm`), where rhe offset is the sum of
/// `x` rows on all lower ranks than the caller.
/// @param[in] xshape Shape of the `x` data.
/// @param[in] partitioner Graph partitioner that computes the owning
/// rank for each cell in `cells`. If not callable, cells are not
/// redistributed.
/// @return A mesh distributed on the communicator `comm`.
template <typename U>
Mesh<typename std::remove_reference_t<typename U::value_type>> create_mesh(
    MPI_Comm comm, MPI_Comm commt,
    std::vector<std::span<const std::int64_t>> cells,
    const std::vector<fem::CoordinateElement<
        typename std::remove_reference_t<typename U::value_type>>>& elements,
    MPI_Comm commg, const U& x, std::array<std::size_t, 2> xshape,
    const CellPartitionFunction& partitioner)
{
  assert(cells.size() == elements.size());
  std::vector<CellType> celltypes;
  std::ranges::transform(elements, std::back_inserter(celltypes),
                         [](auto e) { return e.cell_shape(); });
  std::vector<fem::ElementDofLayout> doflayouts;
  std::ranges::transform(elements, std::back_inserter(doflayouts),
                         [](auto e) { return e.create_dof_layout(); });

  // Note: `extract_topology` extracts topology data, i.e. just the
  // vertices. For P1 geometry this should just be the identity
  // operator. For other elements the filtered lists may have 'gaps',
  // i.e. the indices might not be contiguous.
  //
  // `extract_topology` could be skipped for 'P1 geometry' elements

  std::int32_t num_cell_types = cells.size();

  // -- Partition topology across ranks of comm
  std::vector<std::vector<std::int64_t>> cells1(num_cell_types);
  std::vector<std::vector<std::int64_t>> original_idx1(num_cell_types);
  std::vector<std::vector<int>> ghost_owners(num_cell_types);
  if (partitioner)
  {
    spdlog::info("Using partitioner with cell data ({} cell types)",
                 num_cell_types);
    graph::AdjacencyList<std::int32_t> dest(0);
    if (commt != MPI_COMM_NULL)
    {
      int size = dolfinx::MPI::size(comm);
      std::vector<std::vector<std::int64_t>> t(num_cell_types);
      std::vector<std::span<const std::int64_t>> tspan(num_cell_types);
      for (std::int32_t i = 0; i < num_cell_types; ++i)
      {
        t[i] = extract_topology(celltypes[i], doflayouts[i], cells[i]);
        tspan[i] = std::span(t[i]);
      }
      dest = partitioner(commt, size, celltypes, tspan);
    }

    std::int32_t cell_offset = 0;
    for (std::int32_t i = 0; i < num_cell_types; ++i)
    {
      std::size_t num_cell_nodes = doflayouts[i].num_dofs();
      assert(cells[i].size() % num_cell_nodes == 0);
      std::size_t num_cells = cells[i].size() / num_cell_nodes;

      // Extract destination AdjacencyList for this cell type
      std::vector<std::int32_t> offsets_i(
          std::next(dest.offsets().begin(), cell_offset),
          std::next(dest.offsets().begin(), cell_offset + num_cells + 1));
      std::vector<std::int32_t> data_i(
          std::next(dest.array().begin(), offsets_i.front()),
          std::next(dest.array().begin(), offsets_i.back()));
      std::int32_t offset_0 = offsets_i.front();
      std::ranges::for_each(offsets_i,
                            [&offset_0](std::int32_t& j) { j -= offset_0; });
      graph::AdjacencyList<std::int32_t> dest_i(data_i, offsets_i);
      cell_offset += num_cells;

      // Distribute cells (topology, includes higher-order 'nodes') to
      // destination rank
      std::vector<int> src_ranks;
      std::tie(cells1[i], src_ranks, original_idx1[i], ghost_owners[i])
          = graph::build::distribute(comm, cells[i],
                                     {num_cells, num_cell_nodes}, dest_i);
      spdlog::debug("Got {} cells from distribution", cells1[i].size());
    }
  }
  else
  {
    // No partitioning, construct a global index
    std::int64_t num_owned = 0;
    for (std::int32_t i = 0; i < num_cell_types; ++i)
    {
      cells1[i] = std::vector<std::int64_t>(cells[i].begin(), cells[i].end());
      std::int32_t num_cell_nodes = doflayouts[i].num_dofs();
      assert(cells1[i].size() % num_cell_nodes == 0);
      original_idx1[i].resize(cells1[i].size() / num_cell_nodes);
      num_owned += original_idx1[i].size();
    }

    // Add on global offset
    std::int64_t global_offset = 0;
    MPI_Exscan(&num_owned, &global_offset, 1, MPI_INT64_T, MPI_SUM, comm);
    for (std::int32_t i = 0; i < num_cell_types; ++i)
    {
      std::iota(original_idx1[i].begin(), original_idx1[i].end(),
                global_offset);
      global_offset += original_idx1[i].size();
    }
  }

  // Extract cell 'topology', i.e. extract the vertices for each cell
  // and discard any 'higher-order' nodes
  std::vector<std::vector<std::int64_t>> cells1_v(num_cell_types);
  for (std::int32_t i = 0; i < num_cell_types; ++i)
  {
    cells1_v[i] = extract_topology(celltypes[i], doflayouts[i], cells1[i]);
    spdlog::info("Extract basic topology: {}->{}", cells1[i].size(),
                 cells1_v[i].size());
  }

  // Build local dual graph for owned cells to (i) get list of vertices
  // on the process boundary and (ii) apply re-ordering to cells for
  // locality
  auto boundary_v_fn = [](const std::vector<CellType>& celltypes,
                          const std::vector<fem::ElementDofLayout>& doflayouts,
                          const std::vector<std::vector<int>>& ghost_owners,
                          std::vector<std::vector<std::int64_t>>& cells1,
                          std::vector<std::vector<std::int64_t>>& cells1_v,
                          std::vector<std::vector<std::int64_t>>& original_idx1)
  {
    std::vector<std::pair<std::vector<std::int64_t>, int>> boundary_v_data;

    // Build lists of cells (by cell type) that excludes ghosts
    std::vector<std::span<const std::int64_t>> cells1_v_local;
    for (std::size_t i = 0; i < celltypes.size(); ++i)
    {
      int num_cell_vertices = mesh::num_cell_vertices(celltypes[i]);
      std::size_t num_owned_cells
          = cells1_v[i].size() / num_cell_vertices - ghost_owners[i].size();
      cells1_v_local.emplace_back(cells1_v[i].data(),
                                  num_owned_cells * num_cell_vertices);

      // Build local dual graph for cell type
      auto [graph, unmatched_facets, max_v, _facet_attached_cells]
          = build_local_dual_graph(std::vector{celltypes[i]},
                                   std::vector{cells1_v_local.back()});

      // Store unmatched_facets for current cell type
      boundary_v_data.emplace_back(std::move(unmatched_facets), max_v);

      // Compute re-ordering of graph
      const std::vector<std::int32_t> remap = graph::reorder_gps(graph);

      // Update 'original' indices
      const std::vector<std::int64_t>& orig_idx = original_idx1[i];
      std::vector<std::int64_t> _original_idx(orig_idx.size());
      std::copy_n(orig_idx.rbegin(), ghost_owners[i].size(),
                  _original_idx.rbegin());
      {
        for (std::size_t j = 0; j < remap.size(); ++j)
          _original_idx[remap[j]] = orig_idx[j];
      }
      original_idx1[i] = _original_idx;

      // Reorder cells
      impl::reorder_list(
          std::span(cells1_v[i].data(), remap.size() * num_cell_vertices),
          remap);
      impl::reorder_list(
          std::span(cells1[i].data(), remap.size() * doflayouts[i].num_dofs()),
          remap);
    }

    // Pack 'unmatched' facets for all cell types into one array
    std::vector<std::int64_t> boundary_v_all;
    boundary_v_all.reserve(std::accumulate(
        boundary_v_data.begin(), boundary_v_data.end(), std::size_t(0),
        [](std::size_t x, auto& y) { return x + y.first.size(); }));
    int max_v = std::ranges::max_element(boundary_v_data, [](auto& a, auto& b)
                                         { return a.second < b.second; })
                    ->second;
    // for (std::pair<std::vector<std::int64_t>, int>& v_data : boundary_v_data)
    for (const auto& [v_data, num_v] : boundary_v_data)
    {
      std::size_t num_entties = v_data.size();
      boundary_v_all.insert(boundary_v_all.end(), num_entties, -1);

      auto it1 = std::prev(boundary_v_all.end(), num_entties);
      for (auto it = v_data.begin(); it != v_data.end(); it += num_v)
      {
        std::copy_n(it, num_v, it1);
        it1 += max_v;
      }
    }

    // Sort rows
    const std::vector<std::int32_t> perm = dolfinx::sort_by_perm(
        std::span<const std::int64_t>(boundary_v_all), max_v);

    // For facets in boundary_v_all that appear only once, store the
    // facet vertices
    std::vector<std::int64_t> boundary_v_all_new;
    {
      auto it = perm.begin();
      while (it != perm.end())
      {
        std::span f0(boundary_v_all.data() + (*it) * max_v, max_v);

        // Insert f0 vertices
        boundary_v_all_new.insert(boundary_v_all_new.end(), f0.begin(),
                                  f0.end());

        // Find iterator to next facet different from f0
        auto it1 = std::find_if_not(
            it, perm.end(),
            [f0, max_v, &boundary_v_all](auto idx) -> bool
            {
              return std::equal(f0.begin(), f0.end(),
                                std::next(boundary_v_all.begin(), idx * max_v));
            });

        // Advance iterator and increment entity
        it = it1;
      }

      std::ranges::sort(boundary_v_all_new);
      auto [unique_end, range_end] = std::ranges::unique(boundary_v_all_new);
      boundary_v_all_new.erase(unique_end, range_end);
    }

    // spdlog::info("Build local dual graph");
    auto [xgraph, boundary_v, xmax_v, xfacet_attached_cells]
        = build_local_dual_graph(celltypes, cells1_v_local);

    // Boundary vertices are marked as 'unknown'
    {
      std::ranges::sort(boundary_v);
      auto [unique_end, range_end] = std::ranges::unique(boundary_v);
      boundary_v.erase(unique_end, range_end);
    }

    if (boundary_v_all_new != boundary_v)
      throw std::runtime_error("Boundary vertex mis-match.");

    // Remove -1 if it occurs in boundary vertices (may occur in mixed
    // topology)
    if (!boundary_v.empty() and boundary_v[0] == -1)
      boundary_v.erase(boundary_v.begin());

    return boundary_v;
  };

  const std::vector<std::int64_t> boundary_v = boundary_v_fn(
      celltypes, doflayouts, ghost_owners, cells1, cells1_v, original_idx1);

  spdlog::debug("Got {} boundary vertices", boundary_v.size());

  // Create Topology

  std::vector<std::span<const std::int64_t>> cells1_v_span;
  std::ranges::transform(cells1_v, std::back_inserter(cells1_v_span),
                         [](auto& c) { return std::span(c); });
  std::vector<std::span<const std::int64_t>> original_idx1_span;
  std::ranges::transform(original_idx1, std::back_inserter(original_idx1_span),
                         [](auto& c) { return std::span(c); });
  std::vector<std::span<const int>> ghost_owners_span;
  std::ranges::transform(ghost_owners, std::back_inserter(ghost_owners_span),
                         [](auto& c) { return std::span(c); });

  Topology topology
      = create_topology(comm, celltypes, cells1_v_span, original_idx1_span,
                        ghost_owners_span, boundary_v);

  // Create connectivities required higher-order geometries for creating
  // a Geometry object
  for (int i = 0; i < num_cell_types; ++i)
  {
    for (int e = 1; e < topology.dim(); ++e)
    {
      if (doflayouts[i].num_entity_dofs(e) > 0)
        topology.create_entities(e);
    }

    if (elements[i].needs_dof_permutations())
      topology.create_entity_permutations();
  }

  // Build list of unique (global) node indices from cells1 and
  // distribute coordinate data
  std::vector<std::int64_t> nodes1, nodes2;
  for (std::vector<std::int64_t>& c : cells1)
    nodes1.insert(nodes1.end(), c.begin(), c.end());
  for (std::vector<std::int64_t>& c : cells1)
    nodes2.insert(nodes2.end(), c.begin(), c.end());

  dolfinx::radix_sort(nodes1);
  auto [unique_end, range_end] = std::ranges::unique(nodes1);
  nodes1.erase(unique_end, range_end);

  std::vector coords
      = dolfinx::MPI::distribute_data(comm, nodes1, commg, x, xshape[1]);

  // Create geometry object
  Geometry geometry
      = create_geometry(topology, elements, nodes1, nodes2, coords, xshape[1]);

  return Mesh(comm, std::make_shared<Topology>(std::move(topology)),
              std::move(geometry));
}

/// @brief Create a distributed mesh from mesh data using a provided
/// graph partitioning function for determining the parallel
/// distribution of the mesh.
///
/// From mesh input data that is distributed across processes, a
/// distributed mesh::Mesh is created. If the partitioning function is
/// not callable, i.e. it does not store a callable function, no
/// re-distribution of cells is done.
///
/// @param[in] comm Communicator to build the mesh on.
/// @param[in] commt Communicator that the topology data (`cells`) is
/// distributed on. This should be `MPI_COMM_NULL` for ranks that should
/// not participate in computing the topology partitioning.
/// @param[in] cells Cells on the calling process. Each cell (node in
/// the `AdjacencyList`) is defined by its 'nodes' (using global
/// indices) following the Basix ordering. For lowest order cells this
/// will be just the cell vertices. For higher-order cells, other cells
/// 'nodes' will be included. See dolfinx::io::cells for examples of the
/// Basix ordering.
/// @param[in] element Coordinate element for the cells.
/// @param[in] commg Communicator for geometry.
/// @param[in] x Geometry data ('node' coordinates). Row-major storage.
/// The global index of the `i`th node (row) in `x` is taken as `i` plus
/// the process offset  on`comm`, The offset  is the sum of `x` rows on
/// all processed with a lower rank than the caller.
/// @param[in] xshape Shape of the `x` data.
/// @param[in] partitioner Graph partitioner that computes the owning
/// rank for each cell. If not callable, cells are not redistributed.
/// @return A mesh distributed on the communicator `comm`.
template <typename U>
Mesh<typename std::remove_reference_t<typename U::value_type>> create_mesh(
    MPI_Comm comm, MPI_Comm commt, std::span<const std::int64_t> cells,
    const fem::CoordinateElement<
        typename std::remove_reference_t<typename U::value_type>>& element,
    MPI_Comm commg, const U& x, std::array<std::size_t, 2> xshape,
    const CellPartitionFunction& partitioner)
{
  return create_mesh(comm, commt, std::vector{cells}, std::vector{element},
                     commg, x, xshape, partitioner);
}

/// @brief Create a distributed mesh from mesh data using the default
/// graph partitioner to determine the parallel distribution of the
/// mesh.
///
/// This function takes mesh input data that is distributed across
/// processes and creates a mesh::Mesh, with the mesh cell distribution
/// determined by the default cell partitioner. The default partitioner
/// is based on graph partitioning.
///
/// @param[in] comm MPI communicator to build the mesh on.
/// @param[in] cells Cells on the calling process. See ::create_mesh for
/// a detailed description.
/// @param[in] elements Coordinate elements for the cells.
/// @param[in] x Geometry data ('node' coordinates). See ::create_mesh
/// for a detailed description.
/// @param[in] xshape The shape of `x`. It should be `(num_points, gdim)`.
/// @param[in] ghost_mode The requested type of cell ghosting/overlap
/// @return A mesh distributed on the communicator `comm`.
template <typename U>
Mesh<typename std::remove_reference_t<typename U::value_type>>
create_mesh(MPI_Comm comm, std::span<const std::int64_t> cells,
            const fem::CoordinateElement<
                std::remove_reference_t<typename U::value_type>>& elements,
            const U& x, std::array<std::size_t, 2> xshape, GhostMode ghost_mode)
{
  if (dolfinx::MPI::size(comm) == 1)
    return create_mesh(comm, comm, cells, elements, comm, x, xshape, nullptr);
  else
  {
    return create_mesh(comm, comm, cells, elements, comm, x, xshape,
                       create_cell_partitioner(ghost_mode));
  }
}

/// @brief Create a sub-geometry from a mesh and a subset of mesh entities to
/// be included.
///
/// A sub-geometry is simply a mesh::Geometry object containing only the
/// geometric information for the subset of entities. The entities may
/// differ in topological dimension from the original mesh.
///
/// @param[in] mesh The full mesh.
/// @param[in] dim Topological dimension of the sub-topology.
/// @param[in] subentity_to_entity Map from sub-topology entity to the
/// entity in the parent topology.
/// @return A sub-geometry and a map from sub-geometry coordinate
/// degree-of-freedom to the coordinate degree-of-freedom in `geometry`.
template <std::floating_point T>
std::pair<Geometry<T>, std::vector<int32_t>>
create_subgeometry(const Mesh<T>& mesh, int dim,
                   std::span<const std::int32_t> subentity_to_entity)
{
  const Geometry<T>& geometry = mesh.geometry();

  // Get the geometry dofs in the sub-geometry based on the entities in
  // sub-geometry
  const fem::ElementDofLayout layout = geometry.cmap().create_dof_layout();

  std::vector<std::int32_t> x_indices
      = entities_to_geometry(mesh, dim, subentity_to_entity, true);

  std::vector<std::int32_t> sub_x_dofs = x_indices;
  std::ranges::sort(sub_x_dofs);
  auto [unique_end, range_end] = std::ranges::unique(sub_x_dofs);
  sub_x_dofs.erase(unique_end, range_end);

  // Get the sub-geometry dofs owned by this process
  auto x_index_map = geometry.index_map();
  assert(x_index_map);

  std::shared_ptr<common::IndexMap> sub_x_dof_index_map;
  std::vector<std::int32_t> subx_to_x_dofmap;
  {
    auto [map, new_to_old] = common::create_sub_index_map(
        *x_index_map, sub_x_dofs, common::IndexMapOrder::any, true);
    sub_x_dof_index_map = std::make_shared<common::IndexMap>(std::move(map));
    subx_to_x_dofmap = std::move(new_to_old);
  }

  // Create sub-geometry coordinates
  std::span<const T> x = geometry.x();
  std::int32_t sub_num_x_dofs = subx_to_x_dofmap.size();
  std::vector<T> sub_x(3 * sub_num_x_dofs);
  for (std::int32_t i = 0; i < sub_num_x_dofs; ++i)
  {
    std::copy_n(std::next(x.begin(), 3 * subx_to_x_dofmap[i]), 3,
                std::next(sub_x.begin(), 3 * i));
  }

  // Create geometry to sub-geometry  map
  std::vector<std::int32_t> x_to_subx_dof_map(
      x_index_map->size_local() + x_index_map->num_ghosts(), -1);
  for (std::size_t i = 0; i < subx_to_x_dofmap.size(); ++i)
    x_to_subx_dof_map[subx_to_x_dofmap[i]] = i;

  // Create sub-geometry dofmap
  std::vector<std::int32_t> sub_x_dofmap;
  sub_x_dofmap.reserve(x_indices.size());
  std::ranges::transform(x_indices, std::back_inserter(sub_x_dofmap),
                         [&x_to_subx_dof_map](auto x_dof)
                         {
                           assert(x_to_subx_dof_map[x_dof] != -1);
                           return x_to_subx_dof_map[x_dof];
                         });

  // Create sub-geometry coordinate element
  CellType sub_coord_cell
      = cell_entity_type(geometry.cmap().cell_shape(), dim, 0);
  // Special handling if point meshes, as they only support constant basis
  // functions
  int degree = geometry.cmap().degree();
  if (sub_coord_cell == CellType::point)
    degree = 0;
  fem::CoordinateElement<T> sub_cmap(sub_coord_cell, degree,
                                     geometry.cmap().variant());

  // Sub-geometry input_global_indices
  const std::vector<std::int64_t>& igi = geometry.input_global_indices();
  std::vector<std::int64_t> sub_igi;
  sub_igi.reserve(subx_to_x_dofmap.size());
  std::ranges::transform(subx_to_x_dofmap, std::back_inserter(sub_igi),
                         [&igi](auto sub_x_dof) { return igi[sub_x_dof]; });

  // Create geometry
  return {Geometry(sub_x_dof_index_map, std::move(sub_x_dofmap), {sub_cmap},
                   std::move(sub_x), geometry.dim(), std::move(sub_igi)),
          std::move(subx_to_x_dofmap)};
}

/// @brief Create a new mesh consisting of a subset of entities in a
/// mesh.
/// @param[in] mesh The mesh.
/// @param[in] dim Dimension entities in `mesh` that will be cells in
/// the sub-mesh.
/// @param[in] entities Indices of entites in `mesh` to include in the
/// sub-mesh.
/// @return A new mesh, and maps from the new mesh entities, vertices,
/// and geometry to the input mesh entities, vertices, and geometry.
template <std::floating_point T>
std::tuple<Mesh<T>, std::vector<std::int32_t>, std::vector<std::int32_t>,
           std::vector<std::int32_t>>
create_submesh(const Mesh<T>& mesh, int dim,
               std::span<const std::int32_t> entities)
{
  // Create sub-topology
  mesh.topology_mutable()->create_connectivity(dim, 0);
  auto [topology, subentity_to_entity, subvertex_to_vertex]
      = mesh::create_subtopology(*mesh.topology(), dim, entities);

  // Create sub-geometry
  const int tdim = mesh.topology()->dim();
  mesh.topology_mutable()->create_entities(dim);
  mesh.topology_mutable()->create_connectivity(dim, tdim);
  mesh.topology_mutable()->create_connectivity(tdim, dim);
  mesh.topology_mutable()->create_entity_permutations();
  auto [geometry, subx_to_x_dofmap]
      = mesh::create_subgeometry(mesh, dim, subentity_to_entity);

  return {Mesh(mesh.comm(), std::make_shared<Topology>(std::move(topology)),
               std::move(geometry)),
          std::move(subentity_to_entity), std::move(subvertex_to_vertex),
          std::move(subx_to_x_dofmap)};
}

} // namespace dolfinx::mesh
