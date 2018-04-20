// Copyright (C) 2008-2015 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

struct ufc_dofmap;

namespace dolfin
{

namespace common
{
class IndexMap;
}

namespace mesh
{
class Cell;
class Mesh;
class SubDomain;
} // namespace mesh

namespace fem
{
class DofMap;
class UFC;

/// Builds a DofMap on a mesh::Mesh

class DofMapBuilder
{

public:
  /// Build dofmap. The constrained domain may be a null pointer, in
  /// which case it is ignored.
  ///
  /// @param[out] dofmap
  /// @param[in] dolfin_mesh
  static void build(fem::DofMap& dofmap, const mesh::Mesh& dolfin_mesh);

  /// Build sub-dofmap. This is a view into the parent dofmap.
  ///
  /// @param[out] sub_dofmap
  /// @param[in] parent_dofmap
  /// @param[in] component
  /// @param[in] mesh
  static void build_sub_map_view(fem::DofMap& sub_dofmap,
                                 const fem::DofMap& parent_dofmap,
                                 const std::vector<std::size_t>& component,
                                 const mesh::Mesh& mesh);

private:
  // Build simple local UFC-based dofmap data structure
  static std::vector<std::vector<dolfin::la_index_t>>
  build_local_ufc_dofmap(const ufc_dofmap& ufc_dofmap, const mesh::Mesh& mesh);

  // Compute which process 'owns' each node (point at which dofs live)
  //   - node_ownership = -1 -> dof shared but not 'owned' by this
  //     process
  //   - node_ownership = 0  -> dof owned by this process and shared
  //     with other processes
  //   - node_ownership = 1  -> dof owned by this process and not
  //     shared
  //
  // Also computes map from shared node to sharing processes and a
  // set of process that share dofs on this process.
  // Returns: number of locally owned nodes
  static int compute_node_ownership(
      std::vector<short int>& node_ownership,
      std::unordered_map<int, std::vector<int>>& shared_node_to_processes,
      std::set<int>& neighbours,
      const std::vector<std::vector<la_index_t>>& node_dofmap,
      const std::vector<int>& boundary_nodes,
      const std::set<std::size_t>& global_nodes,
      const std::vector<std::size_t>& node_local_to_global,
      const mesh::Mesh& mesh, const std::size_t global_dim);

  // Build dofmap based on re-ordered nodes
  static std::vector<std::vector<la_index_t>>
  build_dofmap(const std::vector<std::vector<la_index_t>>& node_dofmap,
               const std::vector<int>& old_to_new_node_local,
               const std::size_t block_size);

  // Compute set of global dofs (e.g. Reals associated with global
  // Lagrange multipliers) based on UFC numbering. Global dofs are
  // not associated with any mesh entity. The returned indices are
  // local to the process.
  static std::set<std::size_t>
  compute_global_dofs(std::shared_ptr<const ufc_dofmap> ufc_dofmap,
                      const std::vector<int64_t>& num_mesh_entities_local);

  // FIXME: Try to simplify this function to make pure. Needs some care
  //        because it's called recursively.
  // Iterate recursively over all sub-dof maps to find global
  // degrees of freedom
  static void
  _compute_global_dofs(std::set<std::size_t>& global_dofs,
                       std::size_t& offset_local,
                       std::shared_ptr<const ufc_dofmap> ufc_dofmap,
                       const std::vector<int64_t>& num_mesh_entities_local);

  // Recursively extract UFC sub-dofmap and compute offset
  static std::shared_ptr<ufc_dofmap>
  extract_ufc_sub_dofmap(const ufc_dofmap& ufc_dofmap, std::size_t& offset,
                         const std::vector<std::size_t>& component,
                         const std::vector<int64_t>& num_global_mesh_entities);

  // Compute block size, e.g. in 3D elasticity block_size = 3
  static std::size_t compute_blocksize(const ufc_dofmap& ufc_dofmap,
                                       std::size_t tdim);

  // Build graph from UFC 'node' dofmap. Returns (ufc_dofmap,
  // node_dofmap, node_local_to_global, num_mesh_entities_global)
  static std::tuple<std::shared_ptr<const ufc_dofmap>,
                    std::vector<std::vector<la_index_t>>,
                    std::vector<std::size_t>, std::vector<int64_t>>
  build_ufc_node_graph(std::shared_ptr<const ufc_dofmap> ufc_dofmap,
                       const mesh::Mesh& mesh, const std::size_t block_size);

  // Mark shared nodes. Boundary nodes are assigned a random
  // positive integer, interior nodes are marked as -1, interior
  // nodes in ghost layer of other processes are marked -2, and
  // ghost nodes are marked as -3
  static std::vector<int>
  compute_shared_nodes(const std::vector<std::vector<la_index_t>>& node_dofmap,
                       const std::size_t num_nodes_local,
                       const ufc_dofmap& ufc_dofmap, const mesh::Mesh& mesh);

  // FIXME: document better
  // Return (old-to-new_local, local_to_global_unowned) maps
  static std::pair<std::vector<int>, std::vector<std::size_t>>
  compute_node_reordering(
      const std::unordered_map<int, std::vector<int>>&
          node_to_sharing_processes,
      const std::vector<std::size_t>& old_local_to_global,
      const std::vector<std::vector<la_index_t>>& node_dofmap,
      const std::vector<short int>& node_ownership,
      const std::set<std::size_t>& global_nodes, const MPI_Comm mpi_comm);

  static void
  get_cell_entities_local(const mesh::Cell& cell,
                          std::vector<std::vector<int64_t>>& entity_indices,
                          const std::vector<bool>& needs_mesh_entities);

  static void
  get_cell_entities_global(const mesh::Cell& cell,
                           std::vector<std::vector<int64_t>>& entity_indices,
                           const std::vector<bool>& needs_mesh_entities);

  // Compute number of mesh entities for dimensions required by
  // dofmap
  static std::vector<int64_t>
  compute_num_mesh_entities_local(const mesh::Mesh& mesh,
                                  const std::vector<bool>& needs_mesh_entities);
};
} // namespace fem
} // namespace dolfin