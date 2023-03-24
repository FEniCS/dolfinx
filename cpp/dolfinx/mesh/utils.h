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
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <functional>
#include <mpi.h>
#include <span>

namespace dolfinx::fem
{
class ElementDofLayout;
}

namespace dolfinx::mesh
{
enum class CellType;
template <typename T>
class Mesh;
class Topology;

/// Enum for different partitioning ghost modes
enum class GhostMode : int
{
  none,
  shared_facet,
  shared_vertex
};

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
/// @param[in] mesh The mesh that the entities belong to.
/// @param[in] entities Indices (local to process) of entities to
/// compute `h` for.
/// @param[in] dim Topological dimension of the entities.
/// @returns The greatest distance between any two vertices, `h[i]`
/// corresponds to the entity `entities[i]`.
std::vector<double> h(const Mesh<double>& mesh,
                      std::span<const std::int32_t> entities, int dim);

/// @brief Compute normal to given cell (viewed as embedded in 3D)
/// @returns The entity normals. The shape is `(entities.size(), 3)` and
/// the storage is row-major.
std::vector<double> cell_normals(const Mesh<double>& mesh, int dim,
                                 std::span<const std::int32_t> entities);

/// @brief Compute the midpoints for mesh entities of a given dimension.
/// @returns The entity midpoints. The shape is `(entities.size(), 3)`
/// and the storage is row-major.
std::vector<double> compute_midpoints(const Mesh<double>& mesh, int dim,
                                      std::span<const std::int32_t> entities);

namespace impl
{
/// The coordinates for all 'vertices' in the mesh
/// @param[in] mesh The mesh to compute the vertex coordinates for
/// @return The vertex coordinates. The shape is `(3, num_vertices)` and
/// the jth column hold the coordinates of vertex j.
template <typename T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
compute_vertex_coords(const mesh::Mesh<T>& mesh)
{
  const mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();

  // Create entities and connectivities
  mesh.topology_mutable().create_connectivity(tdim, 0);

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

/// Compute indices of all mesh entities that evaluate to true for the
/// provided geometric marking function. An entity is considered marked
/// if the marker function evaluates true for all of its vertices.
///
/// @param[in] mesh The mesh
/// @param[in] dim The topological dimension of the entities to be
/// considered
/// @param[in] marker The marking function
/// @returns List of marked entity indices, including any ghost indices
/// (indices local to the process)
// template <typename T>
// std::vector<std::int32_t> locate_entities(
//     const Mesh<T>& mesh, int dim,
//     const std::function<std::vector<std::int8_t>(
//         std::experimental::mdspan<
//             const T, std::experimental::extents<
//                          std::size_t, 3,
//                          std::experimental::dynamic_extent>>)>&
//         marker)
template <typename T, typename U>
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

  const mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();

  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(tdim, 0);
  if (dim < tdim)
    mesh.topology_mutable().create_connectivity(dim, 0);

  // Iterate over entities of dimension 'dim' to build vector of marked
  // entities
  auto e_to_v = topology.connectivity(dim, 0);
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

/// Compute indices of all mesh entities that are attached to an owned
/// boundary facet and evaluate to true for the provided geometric
/// marking function. An entity is considered marked if the marker
/// function evaluates true for all of its vertices.
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
/// @param[in] mesh The mesh
/// @param[in] dim The topological dimension of the entities to be
/// considered. Must be less than the topological dimension of the mesh.
/// @param[in] marker The marking function
/// @returns List of marked entity indices (indices local to the
/// process)
std::vector<std::int32_t> locate_entities_boundary(
    const Mesh<double>& mesh, int dim,
    const std::function<std::vector<std::int8_t>(
        std::experimental::mdspan<
            const double,
            std::experimental::extents<
                std::size_t, 3, std::experimental::dynamic_extent>>)>& marker);

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
std::vector<std::int32_t>
entities_to_geometry(const Mesh<double>& mesh, int dim,
                     std::span<const std::int32_t> entities, bool orient);

/// @brief Compute the indices of all exterior facets that are owned by
/// the caller.
///
/// An exterior facet (co-dimension 1) is one that is connected globally
/// to only one cell of co-dimension 0).
///
/// @note Collective
///
/// @param[in] topology The mesh topology
/// @return Sorted list of owned facet indices that are exterior facets
/// of the mesh.
std::vector<std::int32_t> exterior_facet_indices(const Topology& topology);

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
Mesh<typename std::remove_reference_t<typename U::value_type>>
create_mesh(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
            const fem::CoordinateElement& element, const U& x,
            std::array<std::size_t, 2> xshape,
            const CellPartitionFunction& cell_partitioner)
{
  const fem::ElementDofLayout dof_layout = element.create_dof_layout();

  // Function top build geometry. Used to scope memory operations.
  auto build_topology = [](auto comm, auto& element, auto& dof_layout,
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
    const int tdim = cell_dim(element.cell_shape());
    const graph::AdjacencyList<std::int32_t> dest = cell_partitioner(
        comm, size, tdim,
        extract_topology(element.cell_shape(), dof_layout, cells));

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
        = extract_topology(element.cell_shape(), dof_layout, cell_nodes);

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

    /// Re-order an adjacency list
    auto reorder_list
        = [](const auto& list, std::span<const std::int32_t> nodemap)
    {
      using X =
          typename std::remove_reference_t<decltype(list.array())>::value_type;

      // Copy existing data to keep ghost values (not reordered)
      std::vector<X> data(list.array());
      std::vector<std::int32_t> offsets(list.offsets().size());

      // Compute new offsets (owned and ghost)
      offsets[0] = 0;
      for (std::size_t n = 0; n < nodemap.size(); ++n)
        offsets[nodemap[n] + 1] = list.num_links(n);
      for (std::size_t n = nodemap.size(); n < (std::size_t)list.num_nodes();
           ++n)
        offsets[n + 1] = list.num_links(n);
      std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
      graph::AdjacencyList<X> list_new(std::move(data), std::move(offsets));

      for (std::size_t n = 0; n < nodemap.size(); ++n)
      {
        auto links_old = list.links(n);
        auto links_new = list_new.links(nodemap[n]);
        assert(links_old.size() == links_new.size());
        std::copy(links_old.begin(), links_old.end(), links_new.begin());
      }

      return list_new;
    };

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
    return std::pair{create_topology(comm, cells_extracted, original_cell_index,
                                     ghost_owners, element.cell_shape(),
                                     boundary_vertices),
                     std::move(cell_nodes)};
  };

  auto [topology, cell_nodes]
      = build_topology(comm, element, dof_layout, cells, cell_partitioner);

  // Create connectivity required to compute the Geometry (extra
  // connectivities for higher-order geometries)
  int tdim = topology.dim();
  for (int e = 1; e < tdim; ++e)
  {
    if (dof_layout.num_entity_dofs(e) > 0)
      topology.create_entities(e);
  }

  if (element.needs_dof_permutations())
    topology.create_entity_permutations();

  Geometry geometry
      = create_geometry(comm, topology, element, cell_nodes, x, xshape[1]);
  return Mesh<typename U::value_type>(comm, std::move(topology),
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
/// @param[in] element The coordinate element that describes the
/// geometric mapping for cells
/// @param[in] x The coordinates of mesh nodes
/// @param[in] xshape The shape of `x`. It should be `(num_points, gdim)`.
/// @param[in] ghost_mode The requested type of cell ghosting/overlap
/// @return A distributed Mesh.
// template <typename T>
// Mesh<T> create_mesh(MPI_Comm comm,
//                     const graph::AdjacencyList<std::int64_t>& cells,
//                     const fem::CoordinateElement& element, std::span<const T>
//                     x, std::array<std::size_t, 2> xshape, GhostMode
//                     ghost_mode)
template <typename U>
Mesh<typename std::remove_reference_t<typename U::value_type>>
create_mesh(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
            const fem::CoordinateElement& element, const U& x,
            std::array<std::size_t, 2> xshape, GhostMode ghost_mode)
{
  return create_mesh(comm, cells, element, x, xshape,
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
  mesh.topology_mutable().create_connectivity(dim, 0);
  auto [topology, subentity_to_entity, subvertex_to_vertex]
      = mesh::create_subtopology(mesh.topology(), dim, entities);

  // Create sub-geometry
  const int tdim = mesh.topology().dim();
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(dim, tdim);
  mesh.topology_mutable().create_connectivity(tdim, dim);
  auto [geometry, subx_to_x_dofmap] = mesh::create_subgeometry(
      mesh.topology(), mesh.geometry(), dim, subentity_to_entity);

  return {Mesh<T>(mesh.comm(), std::move(topology), std::move(geometry)),
          std::move(subentity_to_entity), std::move(subvertex_to_vertex),
          std::move(subx_to_x_dofmap)};
}

} // namespace dolfinx::mesh
