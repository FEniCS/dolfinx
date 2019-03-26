// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <memory>
#include <set>
#include <tuple>
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
class ElementDofMap;
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
  static std::tuple<std::size_t, std::unique_ptr<common::IndexMap>,
                    std::unordered_map<int, std::vector<int>>, std::set<int>,
                    std::vector<PetscInt>>
  build(const ufc_dofmap& ufc_map, const mesh::Mesh& dolfin_mesh);

  /// Build sub-dofmap. This is a view into the parent dofmap.
  ///
  /// @param[out] sub_dofmap
  /// @param[in] parent_dofmap
  /// @param[in] component
  /// @param[in] mesh
  static std::tuple<ufc_dofmap*, std::int64_t, std::int64_t,
                    std::vector<PetscInt>>
  build_sub_map_view(const DofMap& parent_dofmap,
                     const ufc_dofmap& parent_ufc_dofmap,
                     const int parent_block_size,
                     const std::int64_t parent_offset,
                     const std::vector<std::size_t>& component,
                     const mesh::Mesh& mesh);

private:
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
  // Returns: (number of locally owned nodes, node_ownership,
  // shared_node_to_processes, neighbours)
  static std::tuple<int, std::vector<short int>,
                    std::unordered_map<int, std::vector<int>>, std::set<int>>
  compute_node_ownership(const std::vector<std::vector<PetscInt>>& node_dofmap,
                         const std::vector<int>& boundary_nodes,
                         const std::vector<std::size_t>& node_local_to_global,
                         const mesh::Mesh& mesh, const std::size_t global_dim);

  // Build dofmap based on re-ordered nodes
  static std::vector<std::vector<PetscInt>>
  build_dofmap(const std::vector<std::vector<PetscInt>>& node_dofmap,
               const std::vector<int>& old_to_new_node_local,
               const std::size_t block_size);

  // Compute set of global dofs (e.g. Reals associated with global
  // Lagrange multipliers) based on UFC numbering. Global dofs are
  // not associated with any mesh entity. The returned indices are
  // local to the process.
  static std::pair<std::set<std::size_t>, std::size_t>
  extract_global_dofs(const ufc_dofmap& ufc_map,
                      const std::vector<int64_t>& num_mesh_entities_local,
                      std::set<std::size_t> global_dofs = {},
                      std::size_t offset_local = 0);

  // Recursively extract UFC sub-dofmap and cell-wise components for sub-map
  static std::pair<ufc_dofmap*, int> extract_ufc_sub_dofmap(
      const ufc_dofmap& ufc_dofmap, const std::vector<std::size_t>& component,
      const std::vector<int>& num_cell_entities, int offset = 0);

  // Build graph from ElementDofmap. Returns (node_dofmap,
  // node_local_to_global)
  static std::tuple<std::vector<std::vector<PetscInt>>,
                    std::vector<std::size_t>>
  build_ufc_node_graph(const ElementDofMap& el_dm_blocked,
                       const mesh::Mesh& mesh);

  // Mark shared nodes. Boundary nodes are assigned a random
  // positive integer, interior nodes are marked as -1, interior
  // nodes in ghost layer of other processes are marked -2, and
  // ghost nodes are marked as -3
  static std::vector<int>
  compute_shared_nodes(const std::vector<std::vector<PetscInt>>& node_dofmap,
                       const std::size_t num_nodes_local,
                       const ElementDofMap& el_dm, const mesh::Mesh& mesh);

  // FIXME: document better
  // Return (old-to-new_local, local_to_global_unowned) maps
  static std::pair<std::vector<int>, std::vector<std::size_t>>
  compute_node_reordering(const std::unordered_map<int, std::vector<int>>&
                              node_to_sharing_processes,
                          const std::vector<std::size_t>& old_local_to_global,
                          const std::vector<std::vector<PetscInt>>& node_dofmap,
                          const std::vector<short int>& node_ownership,
                          const MPI_Comm mpi_comm);

  static void
  get_cell_entities_local(std::vector<std::vector<int64_t>>& entity_indices,
                          const mesh::Cell& cell,
                          const std::vector<bool>& needs_mesh_entities);

  static void
  get_cell_entities_global(std::vector<std::vector<int64_t>>& entity_indices,
                           const mesh::Cell& cell,
                           const std::vector<bool>& needs_mesh_entities);

  // Compute number of mesh entities for dimensions required by
  // dofmap
  static std::vector<int64_t>
  compute_num_mesh_entities_local(const mesh::Mesh& mesh,
                                  const std::vector<bool>& needs_mesh_entities);
};
} // namespace fem
} // namespace dolfin
