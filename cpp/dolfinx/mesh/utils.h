// Copyright (C) 2019-2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <dolfinx/common/array2d.h>
#include <dolfinx/common/span.hpp>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <functional>

namespace dolfinx
{
namespace fem
{
class ElementDofLayout;
}

namespace mesh
{
enum class CellType;
enum class GhostMode : int;
class Mesh;

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
                      const tcb::span<const std::int32_t>& entities, int dim);

/// Compute normal to given cell (viewed as embedded in 3D)
common::array2d<double>
cell_normals(const Mesh& mesh, int dim,
             const tcb::span<const std::int32_t>& entities);

/// Compute midpoints or mesh entities of a given dimension
common::array2d<double>
midpoints(const mesh::Mesh& mesh, int dim,
          const tcb::span<const std::int32_t>& entities);

/// Compute indicies of all mesh entities that evaluate to true for the
/// provided geometric marking function. An entity is considered marked
/// if the marker function evaluates true for all of its vertices.
///
/// @param[in] mesh The mesh
/// @param[in] dim The topological dimension of the entities to be
///   considered
/// @param[in] marker The marking function
/// @returns List of marked entity indices, including any ghost indices
///   (indices local to the process)
std::vector<std::int32_t> locate_entities(
    const mesh::Mesh& mesh, int dim,
    const std::function<std::vector<bool>(const common::array2d<double>&)>&
        marker);

/// Compute indicies of all mesh entities that are attached to an owned
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
    const mesh::Mesh& mesh, int dim,
    const std::function<std::vector<bool>(const common::array2d<double>&)>&
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
common::array2d<std::int32_t>
entities_to_geometry(const mesh::Mesh& mesh, int dim,
                     const tcb::span<const std::int32_t>& entity_list,
                     bool orient);

/// Compute the indices (local) of all exterior facets. An exterior facet
/// (co-dimension 1) is one that is connected globally to only one cell of
/// co-dimension 0).
/// @param[in] mesh Mesh
/// @return List of facet indices of exterior facets of the mesh
std::vector<std::int32_t> exterior_facet_indices(const Mesh& mesh);

/// Compute destination rank for mesh cells in this rank by applying the
/// default graph partitioner to the dual graph of the mesh
///
/// @param[in] comm MPI Communicator
/// @param[in] n Number of partitions
/// @param[in] cell_type Cell type
/// @param[in] cells Cells on this process. The ith entry in list
/// contains the global indices for the cell vertices. Each cell can
/// appear only once across all processes. The cell vertex indices are
/// not necessarily contiguous globally, i.e. the maximum index across
/// all processes can be greater than the number of vertices. High-order
/// 'nodes', e.g. mid-side points, should not be included.
/// @param[in] ghost_mode How to overlap the cell partitioning: none,
/// shared_facet or shared_vertex
/// @return Destination rank for each cell on this process
graph::AdjacencyList<std::int32_t>
partition_cells_graph(MPI_Comm comm, int n, const mesh::CellType cell_type,
                      const graph::AdjacencyList<std::int64_t>& cells,
                      mesh::GhostMode ghost_mode);

/// Compute destination rank for mesh cells on this rank by applying the
/// a provided graph partitioner to the dual graph of the mesh
graph::AdjacencyList<std::int32_t>
partition_cells_graph(MPI_Comm comm, int n, const mesh::CellType cell_type,
                      const graph::AdjacencyList<std::int64_t>& cells,
                      mesh::GhostMode ghost_mode,
                      const graph::partition_fn& partfn);

} // namespace mesh
} // namespace dolfinx
