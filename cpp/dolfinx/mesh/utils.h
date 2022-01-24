// Copyright (C) 2019-2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <functional>
#include <mpi.h>
#include <xtl/xspan.hpp>

namespace dolfinx::fem
{
class ElementDofLayout;
}

namespace dolfinx::mesh
{
enum class CellType;
class Mesh;

/// Enum for different partitioning ghost modes
enum class GhostMode : int
{
  none,
  shared_facet,
  shared_vertex
};

/// Signature for the cell partitioning function. The function should
/// compute the destination rank for cells currently on this rank.
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
using CellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        MPI_Comm comm, int nparts, int tdim,
        const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
        dolfinx::mesh::GhostMode ghost_mode)>;

/// Extract topology from cell data, i.e. extract cell vertices
/// @param[in] cell_type The cell shape
/// @param[in] layout The layout of geometry 'degrees-of-freedom' on the
/// reference cell
/// @param[in] cells List of 'nodes' for each cell using global indices.
/// The layout must be consistent with \p layout.
/// @return Cell topology. The global indices will, in general, have
/// 'gaps' due to mid-side and other higher-order nodes being removed
/// from the input @p cell.
graph::AdjacencyList<std::int64_t>
extract_topology(const CellType& cell_type, const fem::ElementDofLayout& layout,
                 const graph::AdjacencyList<std::int64_t>& cells);

/// Compute greatest distance between any two vertices
std::vector<double> h(const Mesh& mesh,
                      const xtl::span<const std::int32_t>& entities, int dim);

/// Compute normal to given cell (viewed as embedded in 3D)
xt::xtensor<double, 2>
cell_normals(const Mesh& mesh, int dim,
             const xtl::span<const std::int32_t>& entities);

/// Compute the midpoints for mesh entities of a given dimension
xt::xtensor<double, 2>
compute_midpoints(const Mesh& mesh, int dim,
                  const xtl::span<const std::int32_t>& entities);

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
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        marker);

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
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        marker);

/// Compute the indices the geometry data for the vertices of the given
/// mesh entities
///
/// @param[in] mesh Mesh
/// @param[in] dim Topological dimension of the entities of interest
/// @param[in] entity_list List of entity indices (local)
/// @param[in] orient If true, in 3D, reorients facets to have
/// consistent normal direction
/// @return Indices in the geometry array for the mesh entity vertices, i.e.
/// indices(i, j) is the position in the geometry array of the j-th vertex of
/// the entity entity_list[i].
xt::xtensor<std::int32_t, 2>
entities_to_geometry(const Mesh& mesh, int dim,
                     const xtl::span<const std::int32_t>& entity_list,
                     bool orient);

/// Compute the indices (local) of all exterior facets. An exterior facet
/// (co-dimension 1) is one that is connected globally to only one cell of
/// co-dimension 0).
/// @param[in] mesh Mesh
/// @return List of facet indices of exterior facets of the mesh
std::vector<std::int32_t> exterior_facet_indices(const Mesh& mesh);

/// Create a function that computes destination rank for mesh cells in
/// this rank by applying the default graph partitioner to the dual
/// graph of the mesh
/// @return Function that computes the destination ranks for each cell
CellPartitionFunction create_cell_partitioner(const graph::partition_fn& partfn
                                              = &graph::partition_graph);

/// Compute incident indices
/// @param[in] mesh The mesh
/// @param[in] entities List of indices of topological dimension `d0`
/// @param[in] d0 Topological dimension
/// @param[in] d1 Topological dimension
/// @return List of entities of topological dimension `d1` that are
/// incident to entities in `entities` (topological dimension `d0`)
std::vector<std::int32_t>
compute_incident_entities(const Mesh& mesh,
                          const xtl::span<const std::int32_t>& entities, int d0,
                          int d1);

} // namespace dolfinx::mesh
