// Copyright (C) 2019-2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "Topology.h"
#include "graphbuild.h"
#include <basix/mdspan.hpp>
#include <concepts>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <functional>
#include <mpi.h>
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

namespace
{
/// Re-order an adjacency list
template <typename T>
graph::AdjacencyList<T> reorder_list(const graph::AdjacencyList<T>& list,
                                     std::span<const std::int32_t> nodemap)
{
  // Copy existing data to keep ghost values (not reordered)
  std::vector<T> data(list.array());
  std::vector<std::int32_t> offsets(list.offsets().size());

  // Compute new offsets (owned and ghost)
  offsets[0] = 0;
  for (std::size_t n = 0; n < nodemap.size(); ++n)
    offsets[nodemap[n] + 1] = list.num_links(n);
  for (std::size_t n = nodemap.size(); n < (std::size_t)list.num_nodes(); ++n)
    offsets[n + 1] = list.num_links(n);
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  graph::AdjacencyList<T> list_new(std::move(data), std::move(offsets));

  for (std::size_t n = 0; n < nodemap.size(); ++n)
  {
    auto links_old = list.links(n);
    auto links_new = list_new.links(nodemap[n]);
    assert(links_old.size() == links_new.size());
    std::copy(links_old.begin(), links_old.end(), links_new.begin());
  }

  return list_new;
}
} // namespace

// See https://github.com/doxygen/doxygen/issues/9552
/// Signature for the cell partitioning function. The function should
/// compute the destination rank for cells currently on this rank.
using CellPartitionFunction = std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm comm, int nparts, int tdim,
    const graph::AdjacencyList<std::int64_t>& cells)>;

namespace impl
{
/// The coordinates of 'vertices' for for entities of a give dimension
/// that are attached to specified facets.
///
/// @pre The provided facets must be on the boundary of the mesh.
///
/// @param[in] mesh Mesh to compute the vertex coordinates for
/// @param[in] dim Topological dimension of the entities
/// @param[in] facets List of facets on the meh boundary
/// @return (0) Entities attached to the boundary facets, (1) vertex
/// coordinates (shape is `(3, num_vertices)`) and (2) map from vertex
/// in the full mesh to the position (column) in the vertex coordinates
/// array (set to -1 if vertex in full mesh is not in the coordinate
/// array).
template <typename T>
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
    std::sort(vertices.begin(), vertices.end());
    vertices.erase(std::unique(vertices.begin(), vertices.end()),
                   vertices.end());
    std::sort(entities.begin(), entities.end());
    entities.erase(std::unique(entities.begin(), entities.end()),
                   entities.end());
  }

  // Get geometry data
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
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
    const int c = v_to_c->links(v).front();
    auto cell_vertices = c_to_v->links(c);
    auto it = std::find(cell_vertices.begin(), cell_vertices.end(), v);
    assert(it != cell_vertices.end());
    const int local_pos = std::distance(cell_vertices.begin(), it);

    auto dofs = x_dofmap.links(c);
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
/// @note Collective
///
/// @param[in] topology Mesh topology
/// @return Sorted list of owned facet indices that are exterior facets
/// of the mesh.
std::vector<std::int32_t> exterior_facet_indices(const Topology& topology);

// See https://github.com/doxygen/doxygen/issues/9552
/// Signature for the cell partitioning function. The function should
/// compute the destination rank for cells currently on this rank.
/*
///
/// @param[in] comm MPI Communicator
/// @param[in] nparts Number of partitions
/// @param[in] tdim Topological dimension
/// @param[in] cells Cells on this process. The ith entry in list
/// contains the global indices for the cell vertices. Each cell can
/// appear only once across all processes. The cell vertex indices are
/// not necessarily contiguous globally, i.e. the maximum index across
/// all processes can be greater than the number of vertices. High-order
/// 'nodes', e.g. mid-side points, should not be included.
/// @param[in] ghost_mode How to overlap the cell partitioning: none,
/// shared_facet or shared_vertex
/// @return Destination ranks for each cell on this process
*/
using CellPartitionFunction = std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm comm, int nparts, int tdim,
    const graph::AdjacencyList<std::int64_t>& cells)>;

/// Extract topology from cell data, i.e. extract cell vertices
/// @param[in] cell_type The cell shape
/// @param[in] layout The layout of geometry 'degrees-of-freedom' on the
/// reference cell
/// @param[in] cells List of 'nodes' for each cell using global indices.
/// The layout must be consistent with `layout`.
/// @return Cell topology. The global indices will, in general, have
/// 'gaps' due to mid-side and other higher-order nodes being removed
/// from the input `cell`.
graph::AdjacencyList<std::int64_t>
extract_topology(const CellType& cell_type, const fem::ElementDofLayout& layout,
                 const graph::AdjacencyList<std::int64_t>& cells);

/// @brief Compute greatest distance between any two vertices of the
/// mesh entities (`h`).
/// @param[in] mesh Mesh that the entities belong to.
/// @param[in] entities Indices (local to process) of entities to
/// compute `h` for.
/// @param[in] dim Topological dimension of the entities.
/// @returns Greatest distance between any two vertices, `h[i]`
/// corresponds to the entity `entities[i]`.
template <typename T>
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
  auto delta_norm = [](const auto& p0, const auto& p1)
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
template <typename T>
std::vector<T> cell_normals(const Mesh<T>& mesh, int dim,
                            std::span<const std::int32_t> entities)
{
  auto topology = mesh.topology();
  assert(topology);

  if (entities.empty())
    return std::vector<T>();

  if (topology->cell_types().size() > 1)
    throw std::runtime_error("Mixed topology not supported");
  if (topology->cell_types()[0] == CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cell");

  const int gdim = mesh.geometry().dim();
  const CellType type = cell_entity_type(topology->cell_types()[0], dim, 0);

  // Find geometry nodes for topology entities
  std::span<const T> x = mesh.geometry().x();

  // Orient cells if they are tetrahedron
  bool orient = false;
  if (topology->cell_types()[0] == CellType::tetrahedron)
    orient = true;

  std::vector<std::int32_t> geometry_entities
      = entities_to_geometry(mesh, dim, entities, orient);

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
      std::transform(p[1].begin(), p[1].end(), p[0].begin(), t.begin(),
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
      std::transform(p[1].begin(), p[1].end(), p[0].begin(), dp1.begin(),
                     [](auto x, auto y) { return x - y; });
      std::transform(p[2].begin(), p[2].end(), p[0].begin(), dp2.begin(),
                     [](auto x, auto y) { return x - y; });

      // Define cell normal via cross product of first two edges
      std::array<T, 3> ni = math::cross(dp1, dp2);
      T norm = std::sqrt(ni[0] * ni[0] + ni[1] * ni[1] + ni[2] * ni[2]);
      std::transform(ni.begin(), ni.end(), std::next(n.begin(), 3 * i),
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
      std::transform(p[1].begin(), p[1].end(), p[0].begin(), dp1.begin(),
                     [](auto x, auto y) { return x - y; });
      std::transform(p[2].begin(), p[2].end(), p[0].begin(), dp2.begin(),
                     [](auto x, auto y) { return x - y; });

      // Define cell normal via cross product of first two edges
      std::array<T, 3> ni = math::cross(dp1, dp2);
      T norm = std::sqrt(ni[0] * ni[0] + ni[1] * ni[1] + ni[2] * ni[2]);
      std::transform(ni.begin(), ni.end(), std::next(n.begin(), 3 * i),
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
template <typename T>
std::vector<T> compute_midpoints(const Mesh<T>& mesh, int dim,
                                 std::span<const std::int32_t> entities)
{
  if (entities.empty())
    return std::vector<T>();

  std::span<const T> x = mesh.geometry().x();

  // Build map from entity -> geometry dof
  // FIXME: This assumes a linear geometry.
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
      std::transform(p.begin(), p.end(), xg.begin(), p.begin(),
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
template <typename T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
compute_vertex_coords(const mesh::Mesh<T>& mesh)
{
  auto topology = mesh.topology();
  assert(topology);
  const int tdim = topology->dim();

  // Create entities and connectivities
  mesh.topology_mutable()->create_connectivity(tdim, 0);

  // Get all vertex 'node' indices
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const std::int32_t num_vertices = topology->index_map(0)->size_local()
                                    + topology->index_map(0)->num_ghosts();
  auto c_to_v = topology->connectivity(tdim, 0);
  assert(c_to_v);
  std::vector<std::int32_t> vertex_to_node(num_vertices);
  for (int c = 0; c < c_to_v->num_nodes(); ++c)
  {
    auto x_dofs = x_dofmap.links(c);
    auto vertices = c_to_v->links(c);
    for (std::size_t i = 0; i < vertices.size(); ++i)
      vertex_to_node[vertices[i]] = x_dofs[i];
  }

  // Pack coordinates of vertices
  std::span<const T> x_nodes = mesh.geometry().x();
  std::vector<T> x_vertices(3 * vertex_to_node.size(), 0.0);
  for (std::size_t i = 0; i < vertex_to_node.size(); ++i)
  {
    const int pos = 3 * vertex_to_node[i];
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
    std::experimental::mdspan<
        const T,
        std::experimental::extents<std::size_t, 3,
                                   std::experimental::dynamic_extent>>>::value;

/// @brief Compute indices of all mesh entities that evaluate to true
/// for the provided geometric marking function.
///
/// An entity is considered marked if the marker function evaluates true
/// for all of its vertices.
///
/// @param[in] mesh Mesh to mark entities on.
/// @param[in] dim Topological dimension of the entities to be
/// considered.
/// @param[in] marker Marking function, returns `true` for a point that
/// is 'marked', and `false` otherwise.
/// @returns List of marked entity indices, including any ghost indices
/// (indices local to the process)
template <std::floating_point T, MarkerFn<T> U>
std::vector<std::int32_t> locate_entities(const Mesh<T>& mesh, int dim,
                                          U marker)
{
  namespace stdex = std::experimental;
  using cmdspan3x_t
      = stdex::mdspan<const T,
                      stdex::extents<std::size_t, 3, stdex::dynamic_extent>>;

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
/// An entity is considered marked if the marker function evaluates true
/// for all of its vertices.
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
/// process)
template <std::floating_point T, MarkerFn<T> U>
std::vector<std::int32_t> locate_entities_boundary(const Mesh<T>& mesh, int dim,
                                                   U marker)
{
  auto topology = mesh.topology();
  assert(topology);
  const int tdim = topology->dim();
  if (dim == tdim)
  {
    throw std::runtime_error(
        "Cannot use mesh::locate_entities_boundary (boundary) for cells.");
  }

  // Compute list of boundary facets
  mesh.topology_mutable()->create_entities(tdim - 1);
  mesh.topology_mutable()->create_connectivity(tdim - 1, tdim);
  const std::vector<std::int32_t> boundary_facets
      = exterior_facet_indices(*topology);

  namespace stdex = std::experimental;
  using cmdspan3x_t
      = stdex::mdspan<const T,
                      stdex::extents<std::size_t, 3, stdex::dynamic_extent>>;

  // Run marker function on the vertex coordinates
  const auto [facet_entities, xdata, vertex_to_pos]
      = impl::compute_vertex_coords_boundary(mesh, dim, boundary_facets);
  cmdspan3x_t x(xdata.data(), 3, xdata.size() / 3);
  const std::vector<std::int8_t> marked = marker(x);
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

/// @brief Determine the indices in the geometry data for each vertex of
/// the given mesh entities.
///
/// @warning This function should not be used unless there is no
/// alternative. It may be removed in the future.
///
/// @param[in] mesh The mesh
/// @param[in] dim Topological dimension of the entities of interest
/// @param[in] entities Entity indices (local) to compute the vertex
/// geometry indices for
/// @param[in] orient If true, in 3D, reorients facets to have
/// consistent normal direction
/// @return Indices in the geometry array for the entity vertices. The
/// shape is `(num_entities, num_vertices_per_entity)` and the storage
/// is row-major. The index `indices[i, j]` is the position in the
/// geometry array of the `j`-th vertex of the `entity[i]`.
template <typename T>
std::vector<std::int32_t>
entities_to_geometry(const Mesh<T>& mesh, int dim,
                     std::span<const std::int32_t> entities, bool orient)
{
  auto topology = mesh.topology();
  assert(topology);

  if (topology->cell_types().size() > 1)
    throw std::runtime_error("Mixed topology not supported");
  CellType cell_type = topology->cell_types()[0];
  if (cell_type == CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cells");
  if (orient and (cell_type != CellType::tetrahedron or dim != 2))
    throw std::runtime_error("Can only orient facets of a tetrahedral mesh");

  const Geometry<T>& geometry = mesh.geometry();
  auto x = geometry.x();

  const int tdim = topology->dim();
  mesh.topology_mutable()->create_entities(dim);
  mesh.topology_mutable()->create_connectivity(dim, tdim);
  mesh.topology_mutable()->create_connectivity(dim, 0);
  mesh.topology_mutable()->create_connectivity(tdim, 0);

  const graph::AdjacencyList<std::int32_t>& xdofs = geometry.dofmap();
  auto e_to_c = topology->connectivity(dim, tdim);
  if (!e_to_c)
  {
    throw std::runtime_error(
        "Entity-to-cell connectivity has not been computed.");
  }

  auto e_to_v = topology->connectivity(dim, 0);
  if (!e_to_v)
  {
    throw std::runtime_error(
        "Entity-to-vertex connectivity has not been computed.");
  }

  const auto c_to_v = topology->connectivity(tdim, 0);
  if (!e_to_v)
  {
    throw std::runtime_error(
        "Cell-to-vertex connectivity has not been computed.");
  }

  const std::size_t num_vertices
      = num_cell_vertices(cell_entity_type(cell_type, dim, 0));
  std::vector<std::int32_t> geometry_idx(entities.size() * num_vertices);
  for (std::size_t i = 0; i < entities.size(); ++i)
  {
    const std::int32_t idx = entities[i];
    // Always pick the second cell to be consistent with the e_to_v connectivity
    const std::int32_t cell = e_to_c->links(idx).back();
    auto ev = e_to_v->links(idx);
    assert(ev.size() == num_vertices);
    const auto cv = c_to_v->links(cell);
    const auto xc = xdofs.links(cell);
    for (std::size_t j = 0; j < num_vertices; ++j)
    {
      int k = std::distance(cv.begin(), std::find(cv.begin(), cv.end(), ev[j]));
      assert(k < (int)cv.size());
      geometry_idx[i * num_vertices + j] = xc[k];
    }

    if (orient)
    {
      // Compute cell midpoint
      std::array<T, 3> midpoint = {0, 0, 0};
      for (std::int32_t j : xc)
        for (int k = 0; k < 3; ++k)
          midpoint[k] += x[3 * j + k];
      std::transform(midpoint.begin(), midpoint.end(), midpoint.begin(),
                     [size = xc.size()](auto x) { return x / size; });

      // Compute vector triple product of two edges and vector to midpoint
      std::array<T, 3> p0, p1, p2;
      std::copy_n(std::next(x.begin(), 3 * geometry_idx[i * num_vertices + 0]),
                  3, p0.begin());
      std::copy_n(std::next(x.begin(), 3 * geometry_idx[i * num_vertices + 1]),
                  3, p1.begin());
      std::copy_n(std::next(x.begin(), 3 * geometry_idx[i * num_vertices + 2]),
                  3, p2.begin());

      std::array<T, 9> a;
      std::transform(midpoint.begin(), midpoint.end(), p0.begin(), a.begin(),
                     [](auto x, auto y) { return x - y; });
      std::transform(p1.begin(), p1.end(), p0.begin(), std::next(a.begin(), 3),
                     [](auto x, auto y) { return x - y; });
      std::transform(p2.begin(), p2.end(), p0.begin(), std::next(a.begin(), 6),
                     [](auto x, auto y) { return x - y; });

      // Midpoint direction should be opposite to normal, hence this
      // should be negative. Switch points if not.
      if (math::det(a.data(), {3, 3}) > 0.0)
      {
        std::swap(geometry_idx[i * num_vertices + 1],
                  geometry_idx[i * num_vertices + 2]);
      }
    }
  }

  return geometry_idx;
}

/// Create a function that computes destination rank for mesh cells in
/// this rank by applying the default graph partitioner to the dual
/// graph of the mesh
/// @return Function that computes the destination ranks for each cell
CellPartitionFunction create_cell_partitioner(mesh::GhostMode ghost_mode
                                              = mesh::GhostMode::none,
                                              const graph::partition_fn& partfn
                                              = &graph::partition_graph);

/// @brief Compute incident indices
/// @param[in] topology The topology
/// @param[in] entities List of indices of topological dimension `d0`
/// @param[in] d0 Topological dimension
/// @param[in] d1 Topological dimension
/// @return List of entities of topological dimension `d1` that are
/// incident to entities in `entities` (topological dimension `d0`)
std::vector<std::int32_t>
compute_incident_entities(const Topology& topology,
                          std::span<const std::int32_t> entities, int d0,
                          int d1);

/// Create a mesh using a provided mesh partitioning function
template <typename U>
Mesh<typename std::remove_reference_t<typename U::value_type>> create_mesh(
    MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
    const std::vector<fem::CoordinateElement<
        typename std::remove_reference_t<typename U::value_type>>>& elements,
    const U& x, std::array<std::size_t, 2> xshape,
    const CellPartitionFunction& cell_partitioner)
{
  const fem::ElementDofLayout dof_layout = elements[0].create_dof_layout();

  // Function top build geometry. Used to scope memory operations.
  auto build_topology = [](auto comm, auto& elements, auto& dof_layout,
                           auto& cells, auto& cell_partitioner)
  {
    // -- Partition topology

    // Note: the function extract_topology (returns an
    // AdjacencyList<std::int64_t>) extract topology data, e.g. just the
    // vertices. For P1 geometry this should just be the identity
    // operator. For other elements the filtered lists may have 'gaps',
    // i.e. the indices might not be contiguous. We don't create an
    // object before calling cell_partitioner to ensure that memory is
    // freed immediately.
    //
    // Note: extract_topology could be skipped for 'P1' elements since
    // it is just the identity

    // Compute the destination rank for cells on this process via graph
    // partitioning.
    const int size = dolfinx::MPI::size(comm);
    const int tdim = cell_dim(elements[0].cell_shape());
    const graph::AdjacencyList<std::int32_t> dest = cell_partitioner(
        comm, size, tdim,
        extract_topology(elements[0].cell_shape(), dof_layout, cells));

    // -- Distribute cells (topology, includes higher-order 'nodes')

    // Distribute cells to destination rank
    auto [cell_nodes, src, original_cell_index0, ghost_owners]
        = graph::build::distribute(comm, cells, dest);

    // Release memory (src is not used)
    decltype(src)().swap(src);

    // -- Extra cell topology

    // Extract cell 'topology', i.e. extract the vertices for each cell
    // and discard any 'higher-order' nodes

    graph::AdjacencyList<std::int64_t> cells_extracted
        = extract_topology(elements[0].cell_shape(), dof_layout, cell_nodes);

    // -- Re-order cells

    // Build local dual graph for owned cells to apply re-ordering to
    const std::int32_t num_owned_cells
        = cells_extracted.num_nodes() - ghost_owners.size();

    auto [graph, unmatched_facets, max_v, facet_attached_cells]
        = build_local_dual_graph(
            std::span<const std::int64_t>(
                cells_extracted.array().data(),
                cells_extracted.offsets()[num_owned_cells]),
            std::span<const std::int32_t>(cells_extracted.offsets().data(),
                                          num_owned_cells + 1),
            tdim);

    const std::vector<int> remap = graph::reorder_gps(graph);

    // Create re-ordered cell lists (leaves ghosts unchanged)
    std::vector<std::int64_t> original_cell_index(original_cell_index0.size());
    for (std::size_t i = 0; i < remap.size(); ++i)
      original_cell_index[remap[i]] = original_cell_index0[i];
    std::copy_n(std::next(original_cell_index0.cbegin(), num_owned_cells),
                ghost_owners.size(),
                std::next(original_cell_index.begin(), num_owned_cells));
    cells_extracted = reorder_list(cells_extracted, remap);
    cell_nodes = reorder_list(cell_nodes, remap);

    // -- Create Topology

    // Boundary vertices are marked as unknown
    std::vector<std::int64_t> boundary_vertices(unmatched_facets);
    std::sort(boundary_vertices.begin(), boundary_vertices.end());
    boundary_vertices.erase(
        std::unique(boundary_vertices.begin(), boundary_vertices.end()),
        boundary_vertices.end());

    // Remove -1 if it occurs in boundary vertices (may occur in mixed topology)
    if (boundary_vertices.size() > 0 and boundary_vertices[0] == -1)
      boundary_vertices.erase(boundary_vertices.begin());

    // Create cells and vertices with the ghosting requested. Input
    // topology includes cells shared via facet, but ghosts will be
    // removed later if not required by ghost_mode.

    std::vector<std::int32_t> cell_group_offsets
        = {0, std::int32_t(cells_extracted.num_nodes() - ghost_owners.size()),
           cells_extracted.num_nodes()};
    std::vector<mesh::CellType> cell_type = {elements[0].cell_shape()};
    return std::pair{create_topology(comm, cells_extracted, original_cell_index,
                                     ghost_owners, cell_type,
                                     cell_group_offsets, boundary_vertices),
                     std::move(cell_nodes)};
  };

  auto [topology, cell_nodes]
      = build_topology(comm, elements, dof_layout, cells, cell_partitioner);

  // Create connectivity required to compute the Geometry (extra
  // connectivities for higher-order geometries)
  int tdim = topology.dim();
  for (int e = 1; e < tdim; ++e)
  {
    if (dof_layout.num_entity_dofs(e) > 0)
      topology.create_entities(e);
  }

  if (elements[0].needs_dof_permutations())
    topology.create_entity_permutations();

  Geometry geometry
      = create_geometry(comm, topology, elements, cell_nodes, x, xshape[1]);

  return Mesh<typename U::value_type>(
      comm, std::make_shared<Topology>(std::move(topology)),
      std::move(geometry));
}

/// @brief Create a mesh using the default partitioner.
///
/// This function takes mesh input data that is distributed across
/// processes and creates a mesh::Mesh, with the mesh cell distribution
/// determined by the default cell partitioner. The default partitioner
/// is based a graph partitioning.
///
/// @param[in] comm The MPI communicator to build the mesh on
/// @param[in] cells The cells on the this MPI rank. Each cell (node in
/// the `AdjacencyList`) is defined by its 'nodes' (using global
/// indices). For lowest order cells this will be just the cell
/// vertices. For higher-order cells, other cells 'nodes' will be
/// included.
/// @param[in] elements The coordinate elements that describe the
/// geometric mapping for cells
/// @param[in] x The coordinates of mesh nodes
/// @param[in] xshape The shape of `x`. It should be `(num_points, gdim)`.
/// @param[in] ghost_mode The requested type of cell ghosting/overlap
/// @return A distributed Mesh.
template <typename U>
Mesh<typename std::remove_reference_t<typename U::value_type>>
create_mesh(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
            const std::vector<fem::CoordinateElement<double>>& elements,
            const U& x, std::array<std::size_t, 2> xshape, GhostMode ghost_mode)
{
  return create_mesh(comm, cells, elements, x, xshape,
                     create_cell_partitioner(ghost_mode));
}

/// Create a new mesh consisting of a subset of entities in a mesh.
/// @param[in] mesh The mesh
/// @param[in] dim Entity dimension
/// @param[in] entities List of entity indices in `mesh` to include in
/// the new mesh
/// @return The new mesh, and maps from the new mesh entities, vertices,
/// and geometry to the input mesh entities, vertices, and geometry.
template <typename T>
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
  auto [geometry, subx_to_x_dofmap] = mesh::create_subgeometry(
      *mesh.topology(), mesh.geometry(), dim, subentity_to_entity);

  return {Mesh<T>(mesh.comm(), std::make_shared<Topology>(std::move(topology)),
                  std::move(geometry)),
          std::move(subentity_to_entity), std::move(subvertex_to_vertex),
          std::move(subx_to_x_dofmap)};
}

} // namespace dolfinx::mesh
