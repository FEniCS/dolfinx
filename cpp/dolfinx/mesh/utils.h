// Copyright (C) 2019-2026 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "EntityMap.h"
#include "Mesh.h"
#include "MeshTags.h"
#include "Topology.h"
#include "graphbuild.h"
#include "types.h"
#include <algorithm>
#include <array>
#include <basix/mdspan.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <exception>
#include <format>
#include <mpi.h>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

/// @file utils.h
/// @brief Functions supporting mesh operations

namespace dolfinx::fem
{
class ElementDofLayout;
}

namespace dolfinx::mesh
{
enum class CellType : std::int8_t;

namespace impl
{
/// @brief Re-order the nodes of a fixed-degree adjacency list.
/// @param[in,out] list Fixed-degree adjacency list stored row-major.
/// Degree is equal to `list.size() / nodemap.size()`.
/// @param[in] nodemap Map from old to new index, i.e. for an old index
/// `i` the new index is `nodemap[i]`.
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
  auto x_dofmap = mesh.geometry().dofmaps().front();
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
    auto it = std::ranges::find(cell_vertices, v);
    assert(it != cell_vertices.end());
    const std::size_t local_pos
        = std::ranges::distance(cell_vertices.begin(), it);

    auto dofs = md::submdspan(x_dofmap, c, md::full_extent);
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
/// @param[in] facet_type_idx The index of the facet type in
/// Topology::entity_types(facet_dim)
/// @return Sorted list of owned facet indices that are exterior facets
/// of the mesh.
std::vector<std::int32_t> exterior_facet_indices(const Topology& topology,
                                                 int facet_type_idx);

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

namespace impl
{
/// @brief Find a mesh's process boundary vertices, reordering cells for
/// locality as a side effect.
///
/// Logically this function only finds the vertices that may be shared
/// with another process (from the facets unmatched by another local
/// cell), which requires building the local dual graph. Since that is
/// the same graph a cell reorder (e.g. for cache locality) is computed
/// from, this function also applies `reorder_fn` to it and reorders
/// `cells`, `cells_v` and `original_idx` **in place**, to avoid
/// building the graph twice. `reorder_fn` is therefore required even
/// though it plays no part in finding boundary vertices -- it is
/// bundled in because the two computations share the same dual graph.
///
/// @param[in] reorder_fn Cell reordering function. A graph reordering
/// function is applied to the local dual graph; a geometric reordering
/// function is applied to cell centroids.
/// @param[in] cell_centroids Centroids of cells, grouped by cell type.
/// @param[in] gdim Number of coordinate components in cell_centroids.
/// @param[in] max_facet_to_cell_links Maximum number of cells a facet can be
/// connected to.
/// @param[in] celltypes List of celltypes in mesh.
/// @param[in] num_vertices_per_cell Number of vertices per cell, one
/// entry per cell type, as returned by ::num_vertices_per_cell_type.
/// @param[in] cells_v_owned View of the owned (non-ghost) cell vertices
/// of each cell type, as returned by ::owned_cell_vertices for `cells_v`
/// and `ghost_owners`.
/// @param[in] doflayouts List of DOF layouts in mesh.
/// @param[in] ghost_owners List of ghost owner per cell per celltype.
/// @param[in,out] cells List of cells per celltype. Reordered during the
/// call.
/// @param[in,out] cells_v List of vertices (no higher order nodes) of
/// cell per celltype. Reordered during the call. `cells_v[i]` may alias
/// `cells[i]` ('P1 geometry'), in which case it is reordered once only.
/// @param[in,out] original_idx Contains the permutation applied to the
/// cells per celltype.
/// @param[in] num_threads Number of threads to use when building the
/// local dual graph. Must be >= 1.
/// @return Boundary vertices (for all cell types).
std::vector<std::int64_t> reorder_cells(
    const graph::Reorder& reorder_fn, std::span<const double> cell_centroids,
    int gdim, std::optional<std::int32_t> max_facet_to_cell_links,
    const std::vector<CellType>& celltypes,
    std::span<const int> num_vertices_per_cell,
    const std::vector<std::span<const std::int64_t>>& cells_v_owned,
    const std::vector<fem::ElementDofLayout>& doflayouts,
    const std::vector<std::vector<int>>& ghost_owners,
    std::vector<std::vector<std::int64_t>>& cells,
    std::vector<std::span<std::int64_t>>& cells_v,
    std::vector<std::vector<std::int64_t>>& original_idx, int num_threads);
} // namespace impl

/// @brief Extract topology from cell data, i.e. extract cell vertices.
/// @param[in] cell_type Cell shape.
/// @param[in] layout Layout of geometry 'degrees-of-freedom' on the
/// reference cell.
/// @param[in] cells List of 'nodes' for each cell using global indices.
/// The layout must be consistent with `layout`.
/// @return Cell topology. The global indices will, in general, have
/// 'gaps' due to mid-side and other higher-order nodes being removed
/// from the input `cell`.
std::vector<std::int64_t> extract_topology(CellType cell_type,
                                           const fem::ElementDofLayout& layout,
                                           std::span<const std::int64_t> cells);

/// @brief Check if ::extract_topology is the identity operation for a
/// dof layout, i.e. the cell 'nodes' are exactly the cell vertices, in
/// vertex order ('P1 geometry').
///
/// When this holds, cell node data can be used directly as cell
/// topology, without the copy that ::extract_topology performs.
///
/// @param[in] cell_type Cell shape.
/// @param[in] layout Layout of geometry 'degrees-of-freedom' on the
/// reference cell.
/// @return `true` if the cell 'nodes' are the cell vertices.
bool is_vertex_dof_layout(CellType cell_type,
                          const fem::ElementDofLayout& layout);

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
  const auto [vertex_xdofs, xdof_shape]
      = entities_to_geometry(mesh, dim, entities, false);

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
        vertex_xdofs.data() + e * xdof_shape[1], xdof_shape[1]);

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

/// @brief Compute normal to given cell (viewed as embedded in 3D).
/// @returns The entity normals. The shape is `(entities.size(), 3)` and
/// the storage is row-major.
template <std::floating_point T>
std::vector<T> cell_normals(const Mesh<T>& mesh, int dim,
                            std::span<const std::int32_t> entities)
{
  if (entities.empty())
    return std::vector<T>();

  auto topology = mesh.topology();
  assert(topology);
  if (topology->cell_type() == CellType::prism and dim == 2)
  {
    throw std::runtime_error(
        "Cell normal computation for prism cells not yet supported.");
  }

  const int gdim = mesh.geometry().dim();
  const CellType type = cell_entity_type(topology->cell_type(), dim, 0);

  // Find geometry nodes for topology entities
  std::span<const T> x = mesh.geometry().x();
  const auto [geometry_entities, eshape]
      = entities_to_geometry(mesh, dim, entities, false);

  std::vector<T> n(entities.size() * 3);
  switch (type)
  {
  case CellType::interval:
  {
    if (gdim > 2)
      throw std::invalid_argument("Interval cell normal undefined in 3D.");
    for (std::size_t i = 0; i < entities.size(); ++i)
    {
      // Get the two vertices as points
      std::array vertices{geometry_entities[i * eshape[1]],
                          geometry_entities[i * eshape[1] + 1]};
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
      std::array vertices = {geometry_entities[i * eshape[1] + 0],
                             geometry_entities[i * eshape[1] + 1],
                             geometry_entities[i * eshape[1] + 2]};
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
      std::array vertices = {geometry_entities[i * eshape[1] + 0],
                             geometry_entities[i * eshape[1] + 1],
                             geometry_entities[i * eshape[1] + 2]};
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
  const auto [e_to_g, eshape]
      = entities_to_geometry(mesh, dim, entities, false);

  std::vector<T> x_mid(entities.size() * 3, 0);
  for (std::size_t e = 0; e < entities.size(); ++e)
  {
    std::span<T, 3> p(x_mid.data() + 3 * e, 3);
    std::span<const std::int32_t> rows(e_to_g.data() + e * eshape[1],
                                       eshape[1]);
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
/// @brief The coordinates for all 'vertices' in the mesh.
/// @param[in] mesh Mesh to compute the vertex coordinates for.
/// @return The vertex coordinates. The shape is `(3, num_vertices)` and
/// the `jth` column hold the coordinates of vertex `j`.
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
compute_vertex_coords(const mesh::Mesh<T>& mesh)
{
  auto topology = mesh.topology();
  assert(topology);
  const int tdim = topology->dim();

  // Create entities and connectivities

  // Get all vertex 'node' indices
  const std::int32_t num_vertices = topology->index_map(0)->size_local()
                                    + topology->index_map(0)->num_ghosts();

  std::vector<std::int32_t> vertex_to_node(num_vertices);
  for (int cell_type_idx = 0,
           num_cell_types = topology->entity_types(tdim).size();
       cell_type_idx < num_cell_types; ++cell_type_idx)
  {
    auto x_dofmap = mesh.geometry().dofmaps().at(cell_type_idx);
    auto c_to_v = topology->connectivity({tdim, cell_type_idx}, {0, 0});
    assert(c_to_v);
    for (int c = 0; c < c_to_v->num_nodes(); ++c)
    {
      auto x_dofs = md::submdspan(x_dofmap, c, md::full_extent);
      auto vertices = c_to_v->links(c);
      for (std::size_t i = 0; i < vertices.size(); ++i)
        vertex_to_node[vertices[i]] = x_dofs[i];
    }
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
    md::mdspan<const T,
               md::extents<std::size_t, 3, md::dynamic_extent>>>::value;

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
/// @param[in] entity_type_idx The index of the entity type in
/// Topology::entity_types(dim)
/// @returns List of marked entity indices, including any ghost indices
/// (indices local to the process).
template <std::floating_point T, MarkerFn<T> U>
std::vector<std::int32_t> locate_entities(const Mesh<T>& mesh, int dim,
                                          U marker, int entity_type_idx)
{

  using cmdspan3x_t
      = md::mdspan<const T, md::extents<std::size_t, 3, md::dynamic_extent>>;

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
  if (dim < tdim)
    mesh.topology_mutable()->create_connectivity(dim, 0);

  // Iterate over entities of dimension 'dim' to build vector of marked
  // entities
  auto e_to_v = topology->connectivity({dim, entity_type_idx}, {0, 0});
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
  const int num_entity_types = mesh.topology()->entity_types(dim).size();
  if (num_entity_types > 1)
  {
    throw std::runtime_error(
        "Multiple entity types of this dimension. Specify entity type index");
  }
  return locate_entities(mesh, dim, marker, 0);
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
  // TODO Rewrite this function, it should be possible to simplify considerably
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

  using cmdspan3x_t
      = md::mdspan<const T, md::extents<std::size_t, 3, md::dynamic_extent>>;

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
/// @return Geometry DOFs associated with the closure of each entity in
/// `entities` and the shape. The shape is `(num_entities,
/// num_xdofs_per_entity)` and the storage is row-major. The index
/// `indices[i, j]` is the position in the geometry array of the `j`-th
/// vertex of the `entity[i]`.
///
/// @pre Mesh connectivities `dim -> mesh.topology().dim()` and
/// `mesh.topology().dim() -> dim` must have been computed. Otherwise an
/// exception is thrown.
template <std::floating_point T>
std::pair<std::vector<std::int32_t>, std::array<std::size_t, 2>>
entities_to_geometry(const Mesh<T>& mesh, int dim,
                     std::span<const std::int32_t> entities,
                     bool permute = false)
{
  auto topology = mesh.topology();
  assert(topology);
  CellType cell_type = topology->cell_type();
  if ((cell_type == CellType::prism or cell_type == CellType::pyramid)
      and dim == 2)
  {
    throw std::runtime_error("mesh::entities_to_geometry for prism/pyramid "
                             "cell facets not yet supported.");
  }

  const int tdim = topology->dim();
  const Geometry<T>& geometry = mesh.geometry();
  auto xdofs = geometry.dofmaps().front();

  // Get the DOF layout and the number of DOFs per entity
  const fem::CoordinateElement<T>& coord_ele = geometry.cmaps().front();
  const fem::ElementDofLayout layout = coord_ele.create_dof_layout();
  const std::size_t num_entity_dofs = layout.entity_closure_dofs(dim, 0).size();
  std::vector<std::int32_t> entity_xdofs;
  entity_xdofs.reserve(entities.size() * num_entity_dofs);
  std::array<std::size_t, 2> eshape{entities.size(), num_entity_dofs};

  // Get the element's closure DOFs
  const std::vector<std::vector<std::vector<int>>>& closure_dofs_all
      = layout.entity_closure_dofs_all();

  // Special case when dim == tdim (cells)
  if (dim == tdim)
  {
    for (std::int32_t c : entities)
    {
      // Extract degrees of freedom
      auto x_c = md::submdspan(xdofs, c, md::full_extent);
      for (std::int32_t entity_dof : closure_dofs_all[tdim][0])
        entity_xdofs.push_back(x_c[entity_dof]);
    }

    return {std::move(entity_xdofs), eshape};
  }

  assert(dim != tdim);

  auto e_to_c = topology->connectivity(dim, tdim);
  if (!e_to_c)
  {
    throw std::runtime_error(std::format(
        "Entity-to-cell connectivity has not been computed. Missing dims "
        "{}->{}",
        dim, tdim));
  }

  auto c_to_e = topology->connectivity(tdim, dim);
  if (!c_to_e)
  {
    throw std::runtime_error(std::format(
        "Cell-to-entity connectivity has not been computed. Missing dims "
        "{}->{}",
        tdim, dim));
  }

  // Get the cell info, which is needed to permute the closure dofs
  std::span<const std::uint32_t> cell_info;
  if (permute)
    cell_info = std::span(mesh.topology()->get_cell_permutation_info());

  for (std::int32_t e : entities)
  {
    // Get a cell connected to the entity
    assert(!e_to_c->links(e).empty());
    std::int32_t c = e_to_c->links(e).front();

    // Get the local index of the entity
    std::span<const std::int32_t> cell_entities = c_to_e->links(c);
    auto it = std::find(cell_entities.begin(), cell_entities.end(), e);
    assert(it != cell_entities.end());
    std::size_t local_entity = std::ranges::distance(cell_entities.begin(), it);

    // Cell sub-entities must be permuted so that their local
    // orientation agrees with their global orientation
    std::vector<std::int32_t> closure_dofs(closure_dofs_all[dim][local_entity]);
    if (permute)
    {
      mesh::CellType entity_type
          = mesh::cell_entity_type(cell_type, dim, local_entity);
      coord_ele.permute_subentity_closure(closure_dofs, cell_info[c],
                                          entity_type, local_entity);
    }

    // Extract degrees of freedom
    auto x_c = md::submdspan(xdofs, c, md::full_extent);
    for (std::int32_t entity_dof : closure_dofs)
      entity_xdofs.push_back(x_c[entity_dof]);
  }

  return {std::move(entity_xdofs), eshape};
}

/// @brief Compute incident entities.
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

namespace impl
{
/// @brief Number of vertices for each cell type in `celltypes`.
/// @param[in] celltypes Cell types.
/// @return Number of vertices per cell, one entry per entry of
/// `celltypes`, in the same order.
inline std::vector<int>
num_vertices_per_cell_type(const std::vector<CellType>& celltypes)
{
  std::vector<int> n;
  n.reserve(celltypes.size());
  std::ranges::transform(celltypes, std::back_inserter(n),
                         [](CellType c) { return mesh::num_cell_vertices(c); });
  return n;
}

/// @brief View of the owned (non-ghost) prefix of each cell type's
/// vertex array.
///
/// `cells_v[i]` must have any ghost cells appended after the cells it
/// owns, with `ghost_owners[i].size()` trailing ghost entries, as
/// produced by ::partition_cells.
///
/// @param[in] num_vertices_per_cell Number of vertices per cell, one
/// entry per cell type.
/// @param[in] cells_v Cell vertices for each cell type, ghosts
/// included.
/// @param[in] ghost_owners Owning rank of the ghost cells (the
/// trailing entries) of each cell type in `cells_v`.
/// @return View of the owned-cell prefix of `cells_v[i]`, for each `i`.
inline std::vector<std::span<const std::int64_t>>
owned_cell_vertices(std::span<const int> num_vertices_per_cell,
                    const std::vector<std::span<std::int64_t>>& cells_v,
                    const std::vector<std::vector<int>>& ghost_owners)
{
  std::vector<std::span<const std::int64_t>> views;
  views.reserve(cells_v.size());
  for (std::size_t i = 0; i < cells_v.size(); ++i)
  {
    std::size_t num_owned
        = cells_v[i].size() / num_vertices_per_cell[i] - ghost_owners[i].size();
    views.emplace_back(cells_v[i].data(), num_owned * num_vertices_per_cell[i]);
  }
  return views;
}

/// @brief Run `fn` locally, recording any exception it throws instead
/// of letting it propagate.
///
/// Meant to be used together with ::mpi_check. `fn` may be called
/// conditionally, e.g. depending on which alternative a
/// `graph::Partitioner`/`graph::Reorder` variant holds, and so in
/// general only by a subset of the ranks of some communicator -- unlike
/// ::mpi_check, this call is therefore never itself collective. `failed`
/// and `error_msg` may be reused across several ::try_locally calls, so
/// that a later call does not need to re-check whether an earlier one
/// on the same rank already failed.
///
/// @param[in] fn Nullary callable to run locally.
/// @param[in,out] failed Set to 1 if `fn` throws; left unmodified
/// otherwise.
/// @param[in,out] error_msg Set to the exception message if `fn`
/// throws; left unmodified otherwise.
template <typename F>
void try_locally(F&& fn, int& failed, std::string& error_msg)
{
  try
  {
    fn();
  }
  catch (const std::exception& e)
  {
    failed = 1;
    error_msg = e.what();
  }
  catch (...)
  {
    failed = 1;
    error_msg = "unknown exception";
  }
}

/// @brief Collectively propagate a local failure recorded by
/// ::try_locally to every rank of `comm`, throwing the same exception on
/// every rank if any rank failed.
///
/// A rank-local exception cannot simply propagate from a collective
/// operation: some ranks may already be blocked in a later collective
/// by the time another rank throws, which would deadlock rather than
/// fail. Funnelling the failure through an `MPI_Allreduce` first ensures
/// every rank either continues normally or throws together.
///
/// @note Collective.
///
/// @param[in] comm Communicator to propagate the failure across.
/// @param[in] op_name Name of the operation, used in the exception
/// message on failure.
/// @param[in] failed 1 if this rank's ::try_locally call(s) failed, 0
/// otherwise.
/// @param[in] error_msg This rank's failure message, if `failed`.
inline void mpi_check(MPI_Comm comm, std::string_view op_name, int failed,
                      const std::string& error_msg)
{
  int any_failed = 0;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm);
  if (any_failed)
  {
    throw std::runtime_error(
        failed ? std::format("{} failed: {}", op_name, error_msg)
               : std::format("{} failed on another rank.", op_name));
  }
}

/// @brief Compute the centroid of each cell from vertex coordinates
/// already available locally.
///
/// Serial -- performs no communication. Used by ::compute_cell_centroids
/// once the vertex coordinates it needs have been gathered.
///
/// @note The returned centroids are always `double`, regardless of `T`,
/// for the same reason as graph::geom_partition_fn: partition quality
/// is insensitive to position precision.
///
/// @tparam T Scalar type of `coords`.
/// @param[in] num_vertices_per_cell Number of vertices per cell, one
/// entry per cell type of `cells`.
/// @param[in] cells Cells of each cell type, using the same global
/// vertex indices as the keys of `node_to_pos`.
/// @param[in] node_to_pos Map from a global vertex index appearing in
/// `cells` to its row in `coords`.
/// @param[in] coords Vertex coordinates, row-major with `gdim` columns,
/// one row per entry of `node_to_pos`.
/// @param[in] gdim Number of coordinate components per node.
/// @return Cell centroids, row-major with `gdim` columns, one row per
/// cell, with the cells of each cell type concatenated in the order
/// they appear in `cells`.
template <std::floating_point T>
std::vector<double> cell_centroids_local(
    std::span<const int> num_vertices_per_cell,
    const std::vector<std::span<const std::int64_t>>& cells,
    const boost::unordered_flat_map<std::int64_t, std::size_t>& node_to_pos,
    std::span<const T> coords, int gdim)
{
  std::size_t num_cells = 0;
  for (std::size_t i = 0; i < cells.size(); ++i)
    num_cells += cells[i].size() / num_vertices_per_cell[i];
  std::vector<double> centroid(gdim * num_cells, 0);

  std::size_t c0 = 0;
  for (std::size_t i = 0; i < cells.size(); ++i)
  {
    const int nv = num_vertices_per_cell[i];
    const double w = 1.0 / nv;
    for (std::size_t c = 0; c < cells[i].size() / nv; ++c)
    {
      for (int v = 0; v < nv; ++v)
      {
        auto it = node_to_pos.find(cells[i][nv * c + v]);
        assert(it != node_to_pos.end());
        std::size_t pos = it->second;
        for (int d = 0; d < gdim; ++d)
          centroid[gdim * (c0 + c) + d] += w * coords[gdim * pos + d];
      }
    }

    c0 += cells[i].size() / nv;
  }

  return centroid;
}

/// @brief Compute the centroid of each cell from its vertex positions.
///
/// @note Collective. The `MPI::distribute_data` call this makes can
/// throw on a rank-local condition (e.g. non-empty `x` on a null
/// `commg`), so a caller invoking this alongside other fallible
/// collectives should run it, too, inside a ::try_locally /::mpi_check
/// guarded region.
///
/// @tparam T Scalar type of `x`.
/// @param[in] comm Communicator that `cells` is distributed across.
/// @param[in] num_vertices_per_cell Number of vertices per cell, one
/// entry per cell type of `cells`.
/// @param[in] cells Cells of each cell type, using global vertex
/// indices (no higher-order 'nodes'). `cells[i]` is a flattened
/// row-major array of shape `(num_cells_i, num_vertices_per_cell[i])`
/// for cell type `i`, where `num_cells_i` is however many cells of
/// that type are on this rank.
/// @param[in] commg Communicator that `x` is distributed across.
/// @param[in] x Geometry ('node') coordinates, row-major with `gdim`
/// columns, distributed over `commg`. Rows are addressed by the
/// global vertex indices used in `cells`; only the rows for vertices
/// referenced by `cells` on this rank are gathered from `commg`.
/// @param[in] gdim Number of coordinate components per node.
/// @return Cell centroids, row-major with `gdim` columns, one row per
/// cell, with the cells of each cell type concatenated in the order
/// they appear in `cells`.
template <std::floating_point T>
std::vector<double>
compute_cell_centroids(MPI_Comm comm,
                       std::span<const int> num_vertices_per_cell,
                       const std::vector<std::span<const std::int64_t>>& cells,
                       MPI_Comm commg, std::span<const T> x, int gdim)
{
  // Vertices of the cells on this rank, sorted and with duplicates
  // removed, and the coordinates for them
  std::vector<std::int64_t> nodes;
  {
    std::size_t size = 0;
    for (std::span<const std::int64_t> c : cells)
      size += c.size();
    nodes.reserve(size);
    for (std::span<const std::int64_t> c : cells)
      nodes.insert(nodes.end(), c.begin(), c.end());
    dolfinx::radix_sort(nodes);
    auto [unique_end, range_end] = std::ranges::unique(nodes);
    nodes.erase(unique_end, range_end);
  }
  const std::vector<T> coords
      = dolfinx::MPI::distribute_data(comm, nodes, commg, x, gdim);

  // Hash map from global vertex index to its position in `nodes` (and
  // so its row in `coords`), turning the many repeated cell-vertex
  // lookups below into an O(1) average lookup rather than an
  // O(log(nodes.size())) binary search each time -- most vertices are
  // shared by several cells, so the same key is looked up repeatedly.
  boost::unordered_flat_map<std::int64_t, std::size_t> node_to_pos;
  node_to_pos.reserve(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i)
    node_to_pos.emplace(nodes[i], i);

  return cell_centroids_local(num_vertices_per_cell, cells, node_to_pos,
                              std::span<const T>(coords), gdim);
}

/// @brief Partition cells across ranks of `comm`, or, if `partitioner`
/// does not hold a callable function, assign each cell (which stays on
/// its current rank) a globally unique index.
///
/// @tparam T Scalar type of `x`.
/// @param[in] comm Communicator to distribute cells on.
/// @param[in] commt Communicator that `cells` is distributed on. Must
/// be `MPI_COMM_NULL` on ranks that should not participate in computing
/// the partition.
/// @param[in] cells Cells, grouped by cell type, as for ::create_mesh.
/// @param[in] celltypes Cell type, one entry per entry of `cells`.
/// @param[in] doflayouts Element dof layout, one entry per entry of
/// `cells`.
/// @param[in] p1_geometry True if every layout in `doflayouts` is a
/// vertex-only dof layout, so that a cell's 'nodes' are already exactly
/// its vertices and extracting the topology is unnecessary.
/// @param[in] partitioner Partitioner, as for ::create_mesh, together
/// with the node weights it is called with (one entry per cell in
/// `cells`, flattened across cell types in the same order as `cells`;
/// if `std::nullopt`, cells are treated as having equal weight). Used
/// only if `partitioner.fn` holds a graph::partition_fn or a
/// graph::hybrid_partition_fn.
/// @param[in] ghosting Flag to enable ghosting of the output cell
/// distribution. Passed on to `partitioner` if it holds a
/// graph::partition_fn or a graph::hybrid_partition_fn; has no
/// effect if it holds a graph::geom_partition_fn, which can never
/// ghost.
/// @param[in] max_facet_to_cell_links Bound on the number of cells a
/// facet must be connected to for it to be considered *matched* (not
/// on boundary for non-branching meshes). Used to build the mesh dual
/// graph if `partitioner` holds a graph::partition_fn or a
/// graph::hybrid_partition_fn; has no effect if it holds a
/// graph::geom_partition_fn, which never needs the dual graph.
/// @param[in] num_threads Number of threads to use when building the
/// mesh dual graph. Must be >= 1. Used only if `partitioner` holds a
/// graph::partition_fn or a graph::hybrid_partition_fn.
/// @param[in] commg Communicator that `x` is distributed on. Used only
/// if `partitioner` holds a graph::geom_partition_fn or a
/// graph::hybrid_partition_fn.
/// @param[in] x Geometry ('node') coordinates. Used only if
/// `partitioner` holds a graph::geom_partition_fn or a
/// graph::hybrid_partition_fn.
/// @param[in] xshape Shape of `x`.
/// @return
/// 1. Cells assigned to this rank, by cell type, with all 'nodes' (not
///    just vertices) and, if ghosted, any ghost cells appended.
/// 2. The original global index of each cell in (1).
/// 3. The owning rank of the ghost cells (the trailing entries) in (1).
template <std::floating_point T>
std::tuple<std::vector<std::vector<std::int64_t>>,
           std::vector<std::vector<std::int64_t>>,
           std::vector<std::vector<int>>>
partition_cells(MPI_Comm comm, MPI_Comm commt,
                const std::vector<std::span<const std::int64_t>>& cells,
                const std::vector<CellType>& celltypes,
                const std::vector<fem::ElementDofLayout>& doflayouts,
                bool p1_geometry, const graph::Partitioner& partitioner,
                bool ghosting,
                std::optional<std::int32_t> max_facet_to_cell_links,
                int num_threads, MPI_Comm commg, std::span<const T> x,
                std::array<std::size_t, 2> xshape)
{
  const std::int32_t num_cell_types = cells.size();
  std::vector<std::vector<std::int64_t>> cells1(num_cell_types);
  std::vector<std::vector<std::int64_t>> original_idx1(num_cell_types);
  std::vector<std::vector<int>> ghost_owners(num_cell_types);
  if (graph::has_partitioner(partitioner.fn))
  {
    spdlog::info("Using partitioner with cell data ({} cell types)",
                 num_cell_types);
    graph::AdjacencyList<std::int32_t> dest(0);
    int failed = 0;
    std::string error_msg;

    // Geometric data can be distributed on ranks that do not participate in
    // topology partitioning. Gather cell centroids collectively over `comm`
    // so that every rank in `commg` participates in the coordinate exchange.
    std::vector<double> centroid;
    const bool needs_centroids
        = std::holds_alternative<graph::geom_partition_fn>(partitioner.fn)
          or std::holds_alternative<graph::hybrid_partition_fn>(partitioner.fn);
    std::vector<std::vector<std::int64_t>> topology(num_cell_types);
    std::vector<std::span<const std::int64_t>> topology_view(num_cell_types);
    if (needs_centroids or commt != MPI_COMM_NULL)
    {
      for (std::int32_t i = 0; i < num_cell_types; ++i)
      {
        if (p1_geometry)
          topology_view[i] = cells[i];
        else
        {
          topology[i] = extract_topology(celltypes[i], doflayouts[i], cells[i]);
          topology_view[i] = topology[i];
        }
      }
    }

    if (needs_centroids)
    {
      // compute_cell_centroids is collective (MPI::distribute_data) and
      // can throw on a rank-local condition, so it is run inside the
      // same try_locally/mpi_check guarded region as the partitioner
      // calls below.
      try_locally(
          [&]
          {
            std::vector<int> num_vertices_per_cell
                = num_vertices_per_cell_type(celltypes);
            centroid
                = compute_cell_centroids(comm, num_vertices_per_cell,
                                         topology_view, commg, x, xshape[1]);
          },
          failed, error_msg);
    }

    if (std::holds_alternative<graph::geom_partition_fn>(partitioner.fn))
    {
      try_locally(
          [&]
          {
            int size = dolfinx::MPI::size(comm);
            const auto& p = std::get<graph::geom_partition_fn>(partitioner.fn);
            dest = graph::regular_adjacency_list(
                p(comm, size, std::span<const double>(centroid), xshape[1],
                  partitioner.node_weights),
                1);
          },
          failed, error_msg);
    }

    if (commt != MPI_COMM_NULL)
    {
      // A partitioner such as graph::parmetis::geom_partitioner (which
      // requires nparts to equal the number of ranks calling it) can
      // throw only on the ranks with commt != MPI_COMM_NULL, which may
      // be a strict subset of comm (e.g. cells built on rank 0 only).
      // mpi_check below turns that into a comm-wide decision before any
      // rank reaches the graph::build::distribute collective, or a throw
      // here would leave the rest of comm blocked on it forever.
      try_locally(
          [&]
          {
            int size = dolfinx::MPI::size(comm);
            // Shared by the graph::partition_fn and
            // graph::hybrid_partition_fn alternatives below: neither has any
            // other way to obtain the mesh dual graph.
            auto dual_graph = [&]() -> graph::AdjacencyList<std::int64_t>
            {
              return build_dual_graph(commt, celltypes, topology_view,
                                      max_facet_to_cell_links, num_threads);
            };

            dest = std::visit(
                [&](const auto& p) -> graph::AdjacencyList<std::int32_t>
                {
                  using P = std::decay_t<decltype(p)>;
                  if constexpr (std::is_same_v<P, graph::hybrid_partition_fn>)
                  {
                    return p(commt, size, dual_graph(),
                             std::span<const double>(centroid),
                             partitioner.node_weights, std::nullopt, ghosting);
                  }
                  else if constexpr (std::is_same_v<P, graph::partition_fn>)
                  {
                    return p(commt, size, dual_graph(),
                             partitioner.node_weights, std::nullopt, ghosting);
                  }
                  else
                    return dest;
                },
                partitioner.fn);
          },
          failed, error_msg);
    }

    mpi_check(comm, "Cell partitioning", failed, error_msg);

    std::int32_t cell_offset = 0;
    for (std::int32_t i = 0; i < num_cell_types; ++i)
    {
      std::size_t num_cell_nodes = doflayouts[i].num_dofs();
      if (cells[i].size() % num_cell_nodes != 0)
      {
        throw std::runtime_error("Cell array size is not a multiple of the "
                                 "number of nodes per cell.");
      }
      std::size_t num_cells = cells[i].size() / num_cell_nodes;

      // Extract destination AdjacencyList for this cell type
      std::vector<std::int32_t> offsets_i(
          std::next(dest.offsets().begin(), cell_offset),
          std::next(dest.offsets().begin(), cell_offset + num_cells + 1));
      std::vector<std::int32_t> data_i(
          std::next(dest.array().begin(), offsets_i.front()),
          std::next(dest.array().begin(), offsets_i.back()));
      const std::int32_t offset_0 = offsets_i.front();
      std::ranges::transform(offsets_i, offsets_i.begin(),
                             [offset_0](std::int32_t j)
                             { return j - offset_0; });
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
  else // No partitioning: keep cells on their current rank
  {
    // Count cells of each type on this rank. Each cell still needs a
    // globally unique index (assigned below), even though it is not
    // being redistributed, and the counts are needed first to size
    // `original_idx1` and to determine this rank's share via the
    // exclusive scan that follows.
    std::int64_t num_owned = 0;
    for (std::int32_t i = 0; i < num_cell_types; ++i)
    {
      cells1[i] = std::vector<std::int64_t>(cells[i].begin(), cells[i].end());
      std::int32_t num_cell_nodes = doflayouts[i].num_dofs();
      if (cells1[i].size() % num_cell_nodes != 0)
      {
        throw std::runtime_error("Cell array size is not a multiple of the "
                                 "number of nodes per cell.");
      }
      original_idx1[i].resize(cells1[i].size() / num_cell_nodes);
      num_owned += original_idx1[i].size();
    }

    // Assign a globally unique index to each cell. `global_offset`
    // starts as the number of cells owned by lower-ranked processes
    // (from the exclusive scan), and is advanced by each cell type's
    // count in turn so that the numbering is contiguous across cell
    // types too.
    std::int64_t global_offset = 0;
    MPI_Exscan(&num_owned, &global_offset, 1, MPI_INT64_T, MPI_SUM, comm);
    for (std::int32_t i = 0; i < num_cell_types; ++i)
    {
      std::iota(original_idx1[i].begin(), original_idx1[i].end(),
                global_offset);
      global_offset += original_idx1[i].size();
    }
  }

  return {std::move(cells1), std::move(original_idx1), std::move(ghost_owners)};
}
} // namespace impl

/// @brief Create a distributed mesh::Mesh from mesh data and using the
/// provided graph partitioning function for determining the parallel
/// distribution of the mesh.
///
/// The input cells and geometry data can be distributed across the
/// calling ranks, but must be not duplicated across ranks.
///
/// The function `partitioner` computes the parallel distribution, i.e.
/// the destination rank for each cell. If it is not callable, no
/// redistribution is performed.
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
/// the parallel rank offset (on `comm`), where the offset is the sum of
/// `x` rows on all lower ranks than the caller.
/// @param[in] xshape Shape of the `x` data.
/// @param[in] partitioner Partitioner that computes the owning rank for
/// each cell in `cells`, together with the node weights it is called
/// with (one entry per cell in `cells`, flattened across cell types in
/// the same order as `cells`; if `std::nullopt`, cells are treated as
/// having equal weight). If `partitioner.fn` is not callable, cells are
/// not redistributed. If it holds a graph::geom_partition_fn or a
/// graph::hybrid_partition_fn, this function computes the centroid
/// of each cell in `cells` (from `x`) and supplies them, see
/// graph::AnyPartitionFunction.
/// @param[in] ghost_mode Ghost mode of the created mesh, passed to
/// `partitioner` if it holds a graph::partition_fn or a
/// graph::hybrid_partition_fn. Has no effect if it holds a
/// graph::geom_partition_fn, which can never ghost.
/// @param[in] max_facet_to_cell_links Bound on the number of cells a
/// facet can be connected to.
/// @param[in] num_threads Number threads to use in mesh construction.
/// Must be >= 1.
/// @param[in] reorder_fn Function that reorders (locally) cells that
/// are owned by this process.
/// @return A mesh distributed on the communicator `comm`.
template <typename U>
Mesh<typename std::remove_reference_t<typename U::value_type>> create_mesh(
    MPI_Comm comm, MPI_Comm commt,
    std::vector<std::span<const std::int64_t>> cells,
    const std::vector<fem::CoordinateElement<
        typename std::remove_reference_t<typename U::value_type>>>& elements,
    MPI_Comm commg, const U& x, std::array<std::size_t, 2> xshape,
    const graph::Partitioner& partitioner, GhostMode ghost_mode,
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads,
    const graph::Reorder& reorder_fn = graph::Reorder{})
{
  using T = typename std::remove_reference_t<typename U::value_type>;

  if (cells.size() != elements.size())
    throw std::runtime_error("Number of cell arrays and elements must match.");
  std::vector<CellType> celltypes;
  std::ranges::transform(elements, std::back_inserter(celltypes),
                         [](auto& e) { return e.cell_shape(); });
  std::vector<fem::ElementDofLayout> doflayouts;
  std::ranges::transform(elements, std::back_inserter(doflayouts),
                         [](auto& e) { return e.create_dof_layout(); });

  // Note: `extract_topology` extracts topology data, i.e. just the
  // vertices. For other elements the filtered lists may have 'gaps',
  // i.e. the indices might not be contiguous.
  //
  // For 'P1 geometry' the extraction is the identity operator, and cell
  // node data is used directly as cell topology. This avoids copies of
  // the (large) cell array, and lets the geometry node indices be taken
  // from the topology vertices rather than re-derived by sorting the
  // cell array (see below).
  const bool p1_geometry = std::ranges::all_of(
      std::views::iota(std::size_t(0), elements.size()),
      [&celltypes, &doflayouts](std::size_t i)
      { return is_vertex_dof_layout(celltypes[i], doflayouts[i]); });

  const std::int32_t num_cell_types = cells.size();

  // Partition cells across ranks of `comm` (or, if `partitioner` is not
  // callable, keep them on their current rank and just assign each a
  // globally unique index)
  const bool ghosting = (ghost_mode != GhostMode::none);
  auto [cells1, original_idx1, ghost_owners] = impl::partition_cells(
      comm, commt, cells, celltypes, doflayouts, p1_geometry, partitioner,
      ghosting, max_facet_to_cell_links, num_threads, commg,
      std::span<const T>(x), xshape);

  // Extract cell 'topology', i.e. extract the vertices for each cell
  // and discard any 'higher-order' nodes. `cells1_v_storage` is empty
  // for 'P1 geometry', where `cells1_v` views `cells1` directly.
  std::vector<std::vector<std::int64_t>> cells1_v_storage(num_cell_types);
  std::vector<std::span<std::int64_t>> cells1_v(num_cell_types);
  for (std::int32_t i = 0; i < num_cell_types; ++i)
  {
    if (p1_geometry)
      cells1_v[i] = cells1[i];
    else
    {
      cells1_v_storage[i]
          = extract_topology(celltypes[i], doflayouts[i], cells1[i]);
      cells1_v[i] = cells1_v_storage[i];
    }

    spdlog::info("Extract basic topology: {}->{}", cells1[i].size(),
                 cells1_v[i].size());
  }

  const std::vector<int> num_cell_vertices
      = impl::num_vertices_per_cell_type(celltypes);
  const std::vector<std::span<const std::int64_t>> cells1_v_owned
      = impl::owned_cell_vertices(num_cell_vertices, cells1_v, ghost_owners);

  // Re-order cells and get boundary vertices. The re-ordering is done
  // on the cell topology, i.e. the vertex indices, and the higher-order
  // nodes are re-ordered accordingly. Centroid computation (needed only
  // for a geometric reorder_fn) is itself collective, so it is run
  // inside the same guarded region as impl::reorder_cells rather than
  // ahead of it, so that a failure in either is propagated to every
  // rank before any rank moves on to a later collective.
  std::vector<std::int64_t> boundary_v;
  int failed = 0;
  std::string error_message;
  impl::try_locally(
      [&]
      {
        std::vector<double> cell_centroids;
        if (std::holds_alternative<graph::reorder_geom_fn>(reorder_fn))
        {
          cell_centroids = impl::compute_cell_centroids(
              comm, num_cell_vertices, cells1_v_owned, commg,
              std::span<const T>(x), xshape[1]);
        }

        boundary_v = impl::reorder_cells(
            reorder_fn, cell_centroids, xshape[1], max_facet_to_cell_links,
            celltypes, num_cell_vertices, cells1_v_owned, doflayouts,
            ghost_owners, cells1, cells1_v, original_idx1, num_threads);
      },
      failed, error_message);
  impl::mpi_check(comm, "Cell reordering", failed, error_message);

  spdlog::debug("Got {} boundary vertices", boundary_v.size());

  // Create Topology
  std::vector<std::span<const std::int64_t>> cells1_v_span(cells1_v.begin(),
                                                           cells1_v.end());
  std::vector<std::span<const std::int64_t>> original_idx1_span;
  std::ranges::transform(original_idx1, std::back_inserter(original_idx1_span),
                         [](auto& c) { return std::span(c); });
  std::vector<std::span<const int>> ghost_owners_span;
  std::ranges::transform(ghost_owners, std::back_inserter(ghost_owners_span),
                         [](auto& c) { return std::span(c); });

  // Note: `vertex_index` holds the sorted input global indices of the
  // topology vertices, which for 'P1 geometry' are exactly the geometry
  // node indices required below.
  auto [topology, vertex_index] = mesh::impl::create_topology(
      comm, celltypes, cells1_v_span, original_idx1_span, ghost_owners_span,
      boundary_v, num_threads);

  // Create connectivities required higher-order geometries for creating
  // a Geometry object
  for (int i = 0; i < num_cell_types; ++i)
  {
    const auto& entity_dofs = doflayouts[i].entity_dofs_all();
    for (int dim = 1; dim < topology.dim(); ++dim)
    {
      // Accumulate count of all dofs on this dimension
      int dim_sum
          = std::accumulate(entity_dofs[dim].begin(), entity_dofs[dim].end(), 0,
                            [](int c, auto v) { return c + v.size(); });

      spdlog::debug("Counting entity dofs, dim={}: {}", dim, dim_sum);
      if (dim_sum > 0)
        topology.create_entities(dim);
    }

    if (elements[i].needs_dof_permutations())
      topology.create_entity_permutations();
  }

  // Cell 'node' indices (global), as a single flat array. This is
  // `cells1` for a single cell type, and concatenated otherwise.
  std::vector<std::int64_t> nodes2_storage;
  std::span<const std::int64_t> nodes2;
  if (num_cell_types == 1)
    nodes2 = cells1.front();
  else
  {
    std::size_t size = 0;
    for (const std::vector<std::int64_t>& c : cells1)
      size += c.size();
    nodes2_storage.reserve(size);
    for (const std::vector<std::int64_t>& c : cells1)
      nodes2_storage.insert(nodes2_storage.end(), c.begin(), c.end());
    nodes2 = nodes2_storage;
  }

  // Sorted list of unique (global) node indices. For 'P1 geometry' the
  // nodes are the vertices, which `create_topology` has already sorted
  // and made unique, so re-deriving them from the (much larger) cell
  // array is avoided.
  std::vector<std::int64_t> nodes1;
  if (p1_geometry)
    nodes1 = std::move(vertex_index);
  else
  {
    nodes1.assign(nodes2.begin(), nodes2.end());
    dolfinx::radix_sort(nodes1);
    auto [unique_end, range_end] = std::ranges::unique(nodes1);
    nodes1.erase(unique_end, range_end);
  }

  std::vector coords
      = dolfinx::MPI::distribute_data(comm, nodes1, commg, x, xshape[1]);

  // Create geometry object
  Geometry geometry
      = create_geometry(topology, elements, nodes1, nodes2, coords, xshape[1]);

  return Mesh(comm, std::make_shared<Topology>(std::move(topology)),
              std::move(geometry));
}

/// @brief Create a distributed mesh with a single cell type from mesh
/// data and using a provided graph partitioning function for
/// determining the parallel distribution of the mesh.
///
/// From mesh input data that is distributed across processes, a
/// distributed mesh::Mesh is created. If the partitioning function is
/// not callable, i.e. it does not store a callable function, no
/// re-distribution of cells is done.
///
/// This constructor provides a simplified interface to the more general
/// ::create_mesh constructor, which supports meshes with more than one
/// cell type.
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
/// the process offset on `comm`. The offset is the sum of `x` rows on
/// all processes with a lower rank than the caller.
/// @param[in] xshape Shape of the `x` data.
/// @param[in] partitioner Partitioner that computes the owning rank for
/// each cell, together with the node weights it is called with (one
/// entry per cell in `cells`; if `std::nullopt`, cells are treated as
/// having equal weight). If `partitioner.fn` is not callable, cells are
/// not redistributed. See the more general ::create_mesh for the
/// graph::geom_partition_fn and graph::hybrid_partition_fn
/// alternatives.
/// @param[in] ghost_mode Ghost mode of the created mesh, as for the
/// more general ::create_mesh.
/// @param[in] max_facet_to_cell_links Bound on the number of cells a
/// facet can be connected to.
/// @param[in] num_threads Number threads to use in mesh construction.
/// Must be >= 1.
/// @param[in] reorder_fn Function that reorders (locally) cells that
/// are owned by this process.
/// @return A mesh distributed on the communicator `comm`.
template <typename U>
Mesh<typename std::remove_reference_t<typename U::value_type>> create_mesh(
    MPI_Comm comm, MPI_Comm commt, std::span<const std::int64_t> cells,
    const fem::CoordinateElement<
        typename std::remove_reference_t<typename U::value_type>>& element,
    MPI_Comm commg, const U& x, std::array<std::size_t, 2> xshape,
    const graph::Partitioner& partitioner, GhostMode ghost_mode,
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads,
    const graph::Reorder& reorder_fn = graph::Reorder{})
{
  return create_mesh(comm, commt, std::vector{cells}, std::vector{element},
                     commg, x, xshape, partitioner, ghost_mode,
                     max_facet_to_cell_links, num_threads, reorder_fn);
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
/// @param[in] xshape Shape of `x`. It should be `(num_points, gdim)`.
/// @param[in] ghost_mode Required type of cell ghosting/overlap.
/// @param[in] max_facet_to_cell_links Bound on the number of cells a
/// facet can be connected to.
/// @return A mesh distributed on the communicator `comm`.
template <typename U>
Mesh<typename std::remove_reference_t<typename U::value_type>>
create_mesh(MPI_Comm comm, std::span<const std::int64_t> cells,
            const fem::CoordinateElement<
                std::remove_reference_t<typename U::value_type>>& elements,
            const U& x, std::array<std::size_t, 2> xshape, GhostMode ghost_mode,
            std::optional<std::int32_t> max_facet_to_cell_links = 2)
{
  // A single rank has nothing to partition, so skip the default
  // partitioner and just assign global indices.
  graph::Partitioner partitioner
      = dolfinx::MPI::size(comm) == 1
            ? graph::Partitioner{.fn = graph::partition_fn(nullptr)}
            : graph::Partitioner{};
  return create_mesh(comm, comm, std::vector{cells}, std::vector{elements},
                     comm, x, xshape, partitioner, ghost_mode,
                     max_facet_to_cell_links, 1);
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
  const fem::ElementDofLayout layout
      = geometry.cmaps().front().create_dof_layout();

  const std::vector<std::int32_t> x_indices
      = entities_to_geometry(mesh, dim, subentity_to_entity, true).first;

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

  // Sub-geometry coordinate element
  CellType sub_xcell
      = cell_entity_type(geometry.cmaps().front().cell_shape(), dim, 0);

  // Special handling of point meshes, as they only support constant
  // basis functions
  int degree
      = (sub_xcell == CellType::point) ? 0 : geometry.cmaps().front().degree();
  fem::CoordinateElement<T> sub_cmap(sub_xcell, degree,
                                     geometry.cmaps().front().variant());

  // Sub-geometry input_global_indices
  const std::vector<std::int64_t>& igi = geometry.input_global_indices();
  std::vector<std::int64_t> sub_igi;
  sub_igi.reserve(subx_to_x_dofmap.size());
  std::ranges::transform(subx_to_x_dofmap, std::back_inserter(sub_igi),
                         [&igi](auto sub_x_dof) { return igi[sub_x_dof]; });

  // Create geometry
  return {Geometry(
              sub_x_dof_index_map,
              std::vector<std::vector<std::int32_t>>{std::move(sub_x_dofmap)},
              {sub_cmap}, std::move(sub_x), geometry.dim(), std::move(sub_igi)),
          std::move(subx_to_x_dofmap)};
}

/// @brief Create a new mesh consisting of a subset of entities in a
/// mesh.
/// @param[in] mesh The mesh.
/// @param[in] dim Dimension entities in `mesh` that will be cells in
/// the sub-mesh.
/// @param[in] entities Indices of entities in `mesh` to include in the
/// sub-mesh.
/// @return A new mesh, and maps from the new mesh entities, vertices,
/// and geometry to the input mesh entities, vertices, and geometry.
template <std::floating_point T>
std::tuple<Mesh<T>, EntityMap, EntityMap, std::vector<std::int32_t>>
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

  Mesh<T> submesh
      = Mesh(mesh.comm(), std::make_shared<Topology>(std::move(topology)),
             std::move(geometry));
  EntityMap entity_map(mesh.topology(), submesh.topology(), dim,
                       subentity_to_entity);
  EntityMap vertex_map(mesh.topology(), submesh.topology(), 0,
                       subvertex_to_vertex);
  return {std::move(submesh), std::move(entity_map), std::move(vertex_map),
          std::move(subx_to_x_dofmap)};
}

/// @brief Transfer a meshtags object from a parent to a submesh.
///
/// @param[in] tags The meshtags object on the parent mesh.
/// @param[in] submesh_topology The topology of the submesh.
/// @param[in] vertex_map Map from submesh vertex to parent mesh vertex.
/// @param[in] cell_map Map from submesh cell to parent mesh entity.
/// @return A meshtags object on the submesh.
template <typename T>
MeshTags<T> transfer_meshtags_to_submesh(
    const MeshTags<T>& tags,
    std::shared_ptr<const dolfinx::mesh::Topology> submesh_topology,
    const EntityMap& vertex_map, const EntityMap& cell_map)
{
  int tag_dim = tags.dim();
  int submesh_tdim = submesh_topology->dim();
  auto topology = tags.topology();
  if (tag_dim > submesh_tdim)
  {
    throw std::runtime_error("Tag dimension must be less than or equal to "
                             "submesh dimension");
  }
  std::shared_ptr<const dolfinx::common::IndexMap> sub_cell_imap
      = submesh_topology->index_map(submesh_tdim);
  if (!sub_cell_imap)
  {
    throw std::runtime_error(
        std::format("Entities of dimension {} does not exist in mesh topology.",
                    submesh_tdim));
  }

  // Create a map from parent entity to submesh cell
  std::int32_t submesh_num_cells
      = sub_cell_imap->size_local() + sub_cell_imap->num_ghosts();
  auto sub_cells = std::ranges::views::iota(0, submesh_num_cells);
  std::vector<std::int32_t> sub_cell_to_parent_entity
      = cell_map.sub_topology_to_topology(sub_cells, false);

  // Create a full lookup for all cells on the parent mesh, as the tag can have
  // entities that are not in the submesh
  auto parent_entity_imap = topology->index_map(submesh_tdim);
  if (!parent_entity_imap)
  {
    throw std::runtime_error(std::format(
        "Entities of dimension {} does not exist in parent mesh topology.",
        submesh_tdim));
  }
  std::size_t num_parent_entities
      = parent_entity_imap->size_local() + parent_entity_imap->num_ghosts();
  std::vector<std::int32_t> parent_entity_to_sub_cell(num_parent_entities, -1);
  for (std::size_t i = 0; i < sub_cell_to_parent_entity.size(); ++i)
    parent_entity_to_sub_cell[sub_cell_to_parent_entity[i]]
        = static_cast<std::int32_t>(i);

  // Get map from submesh vertex to parent vertex
  std::vector<std::int32_t> sub_to_parent_vertex;
  {
    auto sub_vertex_map = submesh_topology->index_map(0);
    std::int32_t num_sub_vertices
        = sub_vertex_map->size_local() + sub_vertex_map->num_ghosts();
    auto sub_vertices = std::ranges::views::iota(0, num_sub_vertices);

    sub_to_parent_vertex
        = vertex_map.sub_topology_to_topology(sub_vertices, false);
  }
  // Access various connectivity maps
  auto sub_e_to_v = submesh_topology->connectivity(tag_dim, 0);
  auto sub_c_to_e = submesh_topology->connectivity(submesh_tdim, tag_dim);
  auto sub_entity_imap = submesh_topology->index_map(tag_dim);
  auto e_to_v = topology->connectivity(tag_dim, 0);
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      e_to_sub_cell = nullptr;
  if (tag_dim != submesh_tdim)
  {
    e_to_sub_cell = topology->connectivity(tag_dim, submesh_tdim);
    if (!e_to_sub_cell)
    {
      throw std::runtime_error(
          std::format("Missing connectivity between {} and {} in parent mesh",
                      tag_dim, submesh_tdim));
    }
  }

  if (!sub_e_to_v)
  {
    throw std::runtime_error(std::format(
        "Missing connectivity between {} and {} in submesh", tag_dim, 0));
  }
  if (!sub_c_to_e)
  {
    throw std::runtime_error(
        std::format("Missing connectivity between {} and {} in submesh",
                    submesh_tdim, tag_dim));
  }
  if (!sub_entity_imap)
  {
    throw std::runtime_error(std::format(
        "Entities of dimension {} does not exist in submesh topology.",
        tag_dim));
  }
  if (!e_to_v)
  {
    throw std::runtime_error(
        std::format("Missing connectivity between {} and 0", tag_dim));
  }

  // Prepare sub entity to parent map
  std::size_t num_sub_entities
      = sub_entity_imap->size_local() + sub_entity_imap->num_ghosts();
  constexpr T max_val = std::numeric_limits<T>::max();
  std::vector<T> submesh_values(num_sub_entities, max_val);
  std::vector<std::int32_t> submesh_indices(num_sub_entities);
  std::iota(submesh_indices.begin(), submesh_indices.end(), 0);

  std::span<const std::int32_t> tagged_entities = tags.indices();
  std::span<const T> tagged_values = tags.values();
  // For each entity in the tag, find all cells of the submesh connected to this
  // entity
  for (std::size_t i = 0; i < tagged_entities.size(); ++i)
  {
    auto find_and_map_sub_entity
        = [tag_dim, submesh_tdim, &e_to_v, &parent_entity_to_sub_cell,
           &sub_to_parent_vertex, &sub_e_to_v, &sub_c_to_e,
           &e_to_sub_cell](std::int32_t entity)
    {
      // Fast exit if the tag dimension is the same as the submesh dimension,
      // as we can directly map the parent entity to the submesh cell
      if (tag_dim == submesh_tdim)
        return parent_entity_to_sub_cell[entity];

      // Given an entity in the parent meshtag, find all submesh-cells that are
      // entities in parent mesh that contain this entity.
      auto entity_vertices = e_to_v->links(entity);
      auto parent_sub_cells = e_to_sub_cell->links(entity);
      auto submesh_cells
          = parent_sub_cells
            | std::views::transform([&parent_entity_to_sub_cell](auto c)
                                    { return parent_entity_to_sub_cell[c]; })
            | std::views::filter([](auto sub_cell) { return sub_cell != -1; });
      for (auto sub_cell : submesh_cells)
      {
        for (auto sub_entity : sub_c_to_e->links(sub_cell))
        {
          // Convert submesh entity vertices to parent vertices
          auto parent_vertices
              = sub_e_to_v->links(sub_entity)
                | std::views::transform([&sub_to_parent_vertex](auto v)
                                        { return sub_to_parent_vertex[v]; });

          // Check if all parent vertices of the submesh entity are in the
          // parent entity
          bool entity_matches = std::ranges::all_of(
              parent_vertices,
              [&entity_vertices](auto p_v)
              {
                // With C++23 this can use std::ranges::contains
                return std::ranges::find(entity_vertices, p_v)
                       != std::ranges::end(entity_vertices);
              });

          // If a match is found, apply values and exit the lambda immediately
          if (entity_matches)
            return sub_entity;
        }
      }
      return -1;
    };

    // Execute the search for the current entity
    std::int32_t sub_entity = find_and_map_sub_entity(tagged_entities[i]);
    if (sub_entity != -1)
      submesh_values[sub_entity] = tagged_values[i];
  }

  // Filter out the entities that were never mapped (values still equal max)
  std::vector<std::int32_t> filtered_indices;
  std::vector<T> filtered_values;
  filtered_indices.reserve(num_sub_entities);
  filtered_values.reserve(num_sub_entities);
  for (std::size_t i = 0; i < submesh_values.size(); ++i)
  {
    if (submesh_values[i] != max_val)
    {
      filtered_indices.push_back(submesh_indices[i]);
      filtered_values.push_back(submesh_values[i]);
    }
  }
  filtered_indices.shrink_to_fit();
  filtered_values.shrink_to_fit();
  MeshTags<T> new_meshtag(submesh_topology, tag_dim, filtered_indices,
                          filtered_values, tags.name());
  return new_meshtag;
}

} // namespace dolfinx::mesh
