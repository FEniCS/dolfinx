// Copyright (C) 2012-2020 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <tuple>
#include <vector>

namespace dolfinx::mesh
{
template <typename T>
class MeshTags;
template <typename T>
class Mesh;
enum class GhostMode;
} // namespace dolfinx::mesh

namespace dolfinx::common
{
class IndexMap;
} // namespace dolfinx::common

namespace dolfinx::refinement
{

/// @brief Communicate edge markers between processes that share edges.
///
/// @param[in] neighbor_comm MPI Communicator for neighborhood
/// @param[in] marked_for_update Lists of edges to be updated on each
/// neighbor. `marked_for_update[r]` is the list of edge indices that
/// are marked by the caller and are shared with local MPI rank `r`.
/// @param[in, out] marked_edges Marker for each edge on the calling
/// process
/// @param[in] map Index map for the mesh edges
void update_logical_edgefunction(
    MPI_Comm neighbor_comm,
    const std::vector<std::vector<std::int32_t>>& marked_for_update,
    std::vector<std::int8_t>& marked_edges, const common::IndexMap& map);

/// @brief Add new vertex for each marked edge, and create
/// new_vertex_coordinates and global_edge->new_vertex map.
///
/// Communicate new vertices with MPI to all affected processes.
///
/// @param[in] neighbor_comm MPI Communicator for neighborhood
/// @param[in] shared_edges
/// @param[in] mesh Existing mesh
/// @param[in] marked_edges
/// @return (0) map from local edge index to new vertex global index,
/// (1) the coordinates of the new vertices (row-major storage) and (2)
/// the shape of the new coordinates.
std::tuple<std::map<std::int32_t, std::int64_t>, std::vector<double>,
           std::array<std::size_t, 2>>
create_new_vertices(MPI_Comm neighbor_comm,
                    const graph::AdjacencyList<int>& shared_edges,
                    const mesh::Mesh<double>& mesh,
                    std::span<const std::int8_t> marked_edges);

/// Use vertex and topology data to partition new mesh across
/// processes
/// @param[in] old_mesh
/// @param[in] cell_topology Topology of cells, (vertex indices)
/// @param[in] new_coords New coordinates, row-major storage
/// @param[in] xshape The shape of `new_coords`
/// @param[in] redistribute Call graph partitioner if true
/// @param[in] ghost_mode None or shared_facet
/// @return New mesh
mesh::Mesh<double>
partition(const mesh::Mesh<double>& old_mesh,
          const graph::AdjacencyList<std::int64_t>& cell_topology,
          std::span<const double> new_coords, std::array<std::size_t, 2> xshape,
          bool redistribute, mesh::GhostMode ghost_mode);

/// @todo Fix docstring. It is unclear.
///
/// @brief Add indices to account for extra n values on this process.
///
/// This is a utility to help add new topological vertices on each
/// process into the space of the index map.
///
/// @param[in] map Index map for the current mesh vertices
/// @param[in] n Number of new entries to be accommodated on this
/// process
/// @return Global indices as if "n" extra values are appended on each
/// process
std::vector<std::int64_t> adjust_indices(const common::IndexMap& map,
                                         std::int32_t n);

/// @brief Transfer facet MeshTags from coarse mesh to refined mesh
/// @note The refined mesh must not have been redistributed during
/// refinement
/// @note GhostMode must be GhostMode.none
/// @param[in] meshtag Facet tags on parent mesh
/// @param[in] refined_mesh Refined mesh based on parent mesh
/// @param[in] cell Parent cell of each cell in refined mesh
/// @param[in] facet Local facets of parent in each cell in refined mesh
/// @return MeshTags on refined mesh
mesh::MeshTags<std::int32_t>
transfer_facet_meshtag(const mesh::MeshTags<std::int32_t>& meshtag,
                       std::shared_ptr<const mesh::Mesh<double>> refined_mesh,
                       std::span<const std::int32_t> cell,
                       std::span<const std::int8_t> facet);

/// @brief Transfer cell MeshTags from coarse mesh to refined mesh.
///
/// @note The refined mesh must not have been redistributed during
/// refinement.
/// @note GhostMode must be GhostMode.none
///
/// @param[in] parent_meshtag Cell MeshTags on parent mesh
/// @param[in] refined_mesh Refined mesh based on parent mesh
/// @param[in] parent_cell Parent cell of each cell in refined mesh
/// @return MeshTags on refined mesh, values copied over from coarse
/// mesh
mesh::MeshTags<std::int32_t>
transfer_cell_meshtag(const mesh::MeshTags<std::int32_t>& parent_meshtag,
                      std::shared_ptr<const mesh::Mesh<double>> refined_mesh,
                      std::span<const std::int32_t> parent_cell);
} // namespace dolfinx::refinement
