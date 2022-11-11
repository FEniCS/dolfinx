// Copyright (C) 2019-2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

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
std::vector<double> h(const Mesh& mesh, std::span<const std::int32_t> entities,
                      int dim);

/// @brief Compute normal to given cell (viewed as embedded in 3D)
/// @returns The entity normals. The shape is `(entities.size(), 3)` and
/// the storage is row-major.
std::vector<double> cell_normals(const Mesh& mesh, int dim,
                                 std::span<const std::int32_t> entities);

/// @brief Compute the midpoints for mesh entities of a given dimension.
/// @returns The entity midpoints. The shape is `(entities.size(), 3)`
/// and the storage is row-major.
std::vector<double> compute_midpoints(const Mesh& mesh, int dim,
                                      std::span<const std::int32_t> entities);

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
std::vector<std::int32_t> locate_entities(
    const Mesh& mesh, int dim,
    const std::function<std::vector<std::int8_t>(
        std::experimental::mdspan<
            const double,
            std::experimental::extents<
                std::size_t, 3, std::experimental::dynamic_extent>>)>& marker);

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
    const Mesh& mesh, int dim,
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
entities_to_geometry(const Mesh& mesh, int dim,
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

/// Compute incident indices
/// @param[in] mesh The mesh
/// @param[in] entities List of indices of topological dimension `d0`
/// @param[in] d0 Topological dimension
/// @param[in] d1 Topological dimension
/// @return List of entities of topological dimension `d1` that are
/// incident to entities in `entities` (topological dimension `d0`)
std::vector<std::int32_t> compute_incident_entities(
    const Mesh& mesh, std::span<const std::int32_t> entities, int d0, int d1);

} // namespace dolfinx::mesh
