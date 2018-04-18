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
}

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
  /// @param[in] constrained_domain
  static void build(fem::DofMap& dofmap, const mesh::Mesh& dolfin_mesh,
                    std::shared_ptr<const mesh::SubDomain> constrained_domain);

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
  // Build modified global entity indices that account for periodic
  // bcs
  static std::size_t build_constrained_vertex_indices(
      const mesh::Mesh& mesh,
      const std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>>&
          slave_to_master_vertices,
      std::vector<std::int64_t>& modified_vertex_indices_global);

  // Build simple local UFC-based dofmap data structure (does not
  // account for master/slave constraints)
  static void
  build_local_ufc_dofmap(std::vector<std::vector<dolfin::la_index_t>>& dofmap,
                         const ufc_dofmap& ufc_dofmap, const mesh::Mesh& mesh);

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
  static void
  build_dofmap(std::vector<std::vector<la_index_t>>& dofmap,
               const std::vector<std::vector<la_index_t>>& node_dofmap,
               const std::vector<int>& old_to_new_node_local,
               const std::size_t block_size);

  // Compute set of global dofs (e.g. Reals associated with global
  // Lagrange multipliers) based on UFC numbering. Global dofs are
  // not associated with any mesh entity. The returned indices are
  // local to the process.
  static std::set<std::size_t>
  compute_global_dofs(std::shared_ptr<const ufc_dofmap> ufc_dofmap,
                      const std::vector<int64_t>& num_mesh_entities_local);

  // Iterate recursively over all sub-dof maps to find global
  // degrees of freedom
  static void
  compute_global_dofs(std::set<std::size_t>& global_dofs,
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

  static void compute_constrained_mesh_indices(
      std::vector<std::vector<std::int64_t>>& global_entity_indices,
      std::vector<int64_t>& num_mesh_entities_global,
      const std::vector<bool>& needs_mesh_entities, const mesh::Mesh& mesh,
      const mesh::SubDomain& constrained_domain);

  static std::shared_ptr<const ufc_dofmap> build_ufc_node_graph(
      std::vector<std::vector<la_index_t>>& node_dofmap,
      std::vector<std::size_t>& node_local_to_global,
      std::vector<int64_t>& num_mesh_entities_global,
      std::shared_ptr<const ufc_dofmap> ufc_dofmap, const mesh::Mesh& mesh,
      std::shared_ptr<const mesh::SubDomain> constrained_domain,
      const std::size_t block_size);

  static std::shared_ptr<const ufc_dofmap> build_ufc_node_graph_constrained(
      std::vector<std::vector<la_index_t>>& node_dofmap,
      std::vector<std::size_t>& node_local_to_global,
      std::vector<int>& node_ufc_local_to_local,
      std::vector<int64_t>& num_mesh_entities_global,
      std::shared_ptr<const ufc_dofmap> ufc_dofmap, const mesh::Mesh& mesh,
      std::shared_ptr<const mesh::SubDomain> constrained_domain,
      const std::size_t block_size);

  // Mark shared nodes. Boundary nodes are assigned a random
  // positive integer, interior nodes are marked as -1, interior
  // nodes in ghost layer of other processes are marked -2, and
  // ghost nodes are marked as -3
  static void
  compute_shared_nodes(std::vector<int>& boundary_nodes,
                       const std::vector<std::vector<la_index_t>>& node_dofmap,
                       const std::size_t num_nodes_local,
                       const ufc_dofmap& ufc_dofmap, const mesh::Mesh& mesh);

  static void compute_node_reordering(
      common::IndexMap& index_map, std::vector<int>& old_to_new_local,
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

  static void get_cell_entities_global_constrained(
      const mesh::Cell& cell, std::vector<std::vector<int64_t>>& entity_indices,
      const std::vector<std::vector<std::int64_t>>& global_entity_indices,
      const std::vector<bool>& needs_mesh_entities);

  // Compute number of mesh entities for dimensions required by
  // dofmap
  static std::vector<int64_t>
  compute_num_mesh_entities_local(const mesh::Mesh& mesh,
                                  const std::vector<bool>& needs_mesh_entities);
};
}
}