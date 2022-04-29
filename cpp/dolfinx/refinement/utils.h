// Copyright (C) 2012-2020 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace dolfinx::mesh
{
template <typename T>
class MeshTags;
class Mesh;
enum class GhostMode;
} // namespace dolfinx::mesh

namespace dolfinx::common
{
class IndexMap;
class IndexMapNew;
} // namespace dolfinx::common

namespace dolfinx::refinement
{

/// @brief Compute the sharing of edges between processes.
///
/// The resulting MPI_Comm is over the neighborhood of shared edges,
/// allowing direct communication between peers. The resulting map is
/// from local edge index to the set of neighbors (within the comm) that
/// share that edge.
/// @param[in] mesh Mesh
/// @return pair of comm and map
std::pair<MPI_Comm, std::map<std::int32_t, std::vector<int>>>
compute_edge_sharing(const mesh::Mesh& mesh);

/// Transfer marked edges between processes.
/// @param[in] neighbor_comm MPI Communicator for neighborhood
/// @param[in] marked_for_update Lists of edges to be updates on each
/// neighbor
/// @param[in, out] marked_edges Marked edges to be updated
/// @param[in] map_e IndexMap for edges
void update_logical_edgefunction(
    MPI_Comm neighbor_comm,
    const std::vector<std::vector<std::int32_t>>& marked_for_update,
    std::vector<std::int8_t>& marked_edges, const common::IndexMapNew& map_e);

/// Add new vertex for each marked edge, and create
/// new_vertex_coordinates and global_edge->new_vertex map.
/// Communicate new vertices with MPI to all affected processes.
/// @param[in] neighbor_comm MPI Communicator for neighborhood
/// @param[in] shared_edges
/// @param[in] mesh Existing mesh
/// @param[in] marked_edges
/// @return edge_to_new_vertex map and geometry array
std::pair<std::map<std::int32_t, std::int64_t>, xt::xtensor<double, 2>>
create_new_vertices(
    MPI_Comm neighbor_comm,
    const std::map<std::int32_t, std::vector<std::int32_t>>& shared_edges,
    const mesh::Mesh& mesh, const std::vector<std::int8_t>& marked_edges);

/// Use vertex and topology data to partition new mesh across
/// processes
/// @param[in] old_mesh
/// @param[in] cell_topology Topology of cells, (vertex indices)
/// @param[in] new_vertex_coordinates
/// @param[in] redistribute Call graph partitioner if true
/// @param[in] ghost_mode None or shared_facet
/// @return New mesh
mesh::Mesh partition(const mesh::Mesh& old_mesh,
                     const graph::AdjacencyList<std::int64_t>& cell_topology,
                     const xt::xtensor<double, 2>& new_vertex_coordinates,
                     bool redistribute, mesh::GhostMode ghost_mode);

/// @brief brief description indices to account for extra n values on each
/// process.
///
/// This is a utility to help add new topological vertices on each
/// process into the space of the index map.
///
/// @param[in] index_map Index map for the current mesh vertices
/// @param[in] n Number of new entries to be accommodated on this process
/// @return Global indices as if "n" extra values are appended on each
/// process
std::vector<std::int64_t> adjust_indices(const common::IndexMapNew& index_map,
                                         std::int32_t n);

/// Transfer facet MeshTags from coarse mesh to refined mesh
/// @note The refined mesh must not have been redistributed during refinement
/// @note GhostMode must be GhostMode.none
/// @param[in] parent_meshtag Facet MeshTags on parent mesh
/// @param[in] refined_mesh Refined mesh based on parent mesh
/// @param[in] parent_cell Parent cell of each cell in refined mesh
/// @param[in] parent_facet Local facets of parent in each cell in refined mesh
/// @return MeshTags on refined mesh, values copied over from coarse mesh
mesh::MeshTags<std::int32_t>
transfer_facet_meshtag(const mesh::MeshTags<std::int32_t>& parent_meshtag,
                       const mesh::Mesh& refined_mesh,
                       const std::vector<std::int32_t>& parent_cell,
                       const std::vector<std::int8_t>& parent_facet);

/// Transfer cell MeshTags from coarse mesh to refined mesh
/// @note The refined mesh must not have been redistributed during refinement
/// @note GhostMode must be GhostMode.none
/// @param[in] parent_meshtag Cell MeshTags on parent mesh
/// @param[in] refined_mesh Refined mesh based on parent mesh
/// @param[in] parent_cell Parent cell of each cell in refined mesh
/// @return MeshTags on refined mesh, values copied over from coarse mesh
mesh::MeshTags<std::int32_t>
transfer_cell_meshtag(const mesh::MeshTags<std::int32_t>& parent_meshtag,
                      const mesh::Mesh& refined_mesh,
                      const std::vector<std::int32_t>& parent_cell);
} // namespace dolfinx::refinement
