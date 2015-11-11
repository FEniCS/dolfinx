// Copyright (C) 2008-2015 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Niclas Jansson 2009.
// Modified by Garth Wells 2009-2012
// Modified by Mikael Mortensen 2012.
// Modified by Martin Alnaes, 2015

#ifndef __DOF_MAP_BUILDER_H
#define __DOF_MAP_BUILDER_H

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ufc
{
  class dofmap;
}

namespace dolfin
{

  class DofMap;
  class Mesh;
  class IndexMap;
  class SubDomain;
  class UFC;

  /// Documentation of class

  class DofMapBuilder
  {

  public:

    /// Build dofmap. The constrained domain may be a null pointer, in
    /// which case it is ignored.
    static void build(DofMap& dofmap, const Mesh& dolfin_mesh,
                      std::shared_ptr<const SubDomain> constrained_domain);

    /// Build sub-dofmap. This is a view into the parent dofmap.
    static void build_sub_map_view(DofMap& sub_dofmap,
                                   const DofMap& parent_dofmap,
                                   const std::vector<std::size_t>& component,
                                   const Mesh& mesh);

  private:

    // Build modified global entity indices that account for periodic
    // bcs
    static std::size_t build_constrained_vertex_indices(
      const Mesh& mesh,
      const std::map<unsigned int, std::pair<unsigned int,
      unsigned int>>& slave_to_master_vertices,
      std::vector<std::size_t>& modified_vertex_indices_global);

    // Build simple local UFC-based dofmap data structure (does not
    // account for master/slave constraints)
    static void
      build_local_ufc_dofmap(std::vector<std::vector<dolfin::la_index>>& dofmap,
                             const ufc::dofmap& ufc_dofmap,
                             const Mesh& mesh);

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
      const std::vector<std::vector<la_index>>& node_dofmap,
      const std::vector<int>& boundary_nodes,
      const std::set<std::size_t>& global_nodes,
      const std::vector<std::size_t>& node_local_to_global,
      const Mesh& mesh,
      const std::size_t global_dim);

    // Build dofmap based on re-ordered nodes
    static void
      build_dofmap(std::vector<std::vector<la_index>>& dofmap,
                   const std::vector<std::vector<la_index>>& node_dofmap,
                   const std::vector<int>& old_to_new_node_local,
                   const std::size_t block_size);

    // Compute set of global dofs (e.g. Reals associated with global
    // Lagrange multipliers) based on UFC numbering. Global dofs are
    // not associated with any mesh entity. The returned indices are
    // local to the process.
    static std::set<std::size_t>
      compute_global_dofs(std::shared_ptr<const ufc::dofmap> ufc_dofmap,
                     const std::vector<std::size_t>& num_mesh_entities_local);

    // Iterate recursively over all sub-dof maps to find global
    // degrees of freedom
    static void
      compute_global_dofs(std::set<std::size_t>& global_dofs,
                       std::size_t& offset_local,
                       std::shared_ptr<const ufc::dofmap> ufc_dofmap,
                       const std::vector<std::size_t>& num_mesh_entities_local);

    // Recursively extract UFC sub-dofmap and compute offset
    static std::shared_ptr<ufc::dofmap> extract_ufc_sub_dofmap(
      const ufc::dofmap& ufc_dofmap,
      std::size_t& offset,
      const std::vector<std::size_t>& component,
      const std::vector<std::size_t>& num_global_mesh_entities);

    // Compute block size, e.g. in 3D elasticity block_size = 3
    static std::size_t compute_blocksize(const ufc::dofmap& ufc_dofmap);

    static void compute_constrained_mesh_indices(
      std::vector<std::vector<std::size_t>>& global_entity_indices,
      std::vector<std::size_t>& num_mesh_entities_global,
      const std::vector<bool>& needs_mesh_entities,
      const Mesh& mesh,
      const SubDomain& constrained_domain);

    static std::shared_ptr<const ufc::dofmap>
      build_ufc_node_graph(
        std::vector<std::vector<la_index>>& node_dofmap,
        std::vector<std::size_t>& node_local_to_global,
        std::vector<std::size_t>& num_mesh_entities_global,
        std::shared_ptr<const ufc::dofmap> ufc_dofmap,
        const Mesh& mesh,
        std::shared_ptr<const SubDomain> constrained_domain,
        const std::size_t block_size);

    static std::shared_ptr<const ufc::dofmap>
      build_ufc_node_graph_constrained(
        std::vector<std::vector<la_index>>& node_dofmap,
        std::vector<std::size_t>& node_local_to_global,
        std::vector<int>& node_ufc_local_to_local,
        std::vector<std::size_t>& num_mesh_entities_global,
        std::shared_ptr<const ufc::dofmap> ufc_dofmap,
        const Mesh& mesh,
        std::shared_ptr<const SubDomain> constrained_domain,
        const std::size_t block_size);


    // Mark shared nodes. Boundary nodes are assigned a random
    // positive integer, interior nodes are marked as -1, interior
    // nodes in ghost layer of other processes are marked -2, and
    // ghost nodes are marked as -3
    static void compute_shared_nodes(
      std::vector<int>& boundary_nodes,
      const std::vector<std::vector<la_index>>& node_dofmap,
      const std::size_t num_nodes_local,
      const ufc::dofmap& ufc_dofmap,
      const Mesh& mesh);

    static void compute_node_reordering(
      IndexMap& index_map,
      std::vector<int>& old_to_new_local,
      const std::unordered_map<int, std::vector<int>>& node_to_sharing_processes,
      const std::vector<std::size_t>& old_local_to_global,
      const std::vector<std::vector<la_index>>& node_dofmap,
      const std::vector<short int>& node_ownership,
      const std::set<std::size_t>& global_nodes,
      const MPI_Comm mpi_comm);

    static void get_cell_entities_local(const Cell& cell,
      std::vector<std::vector<std::size_t>>& entity_indices,
      const std::vector<bool>& needs_mesh_entities);

    static void get_cell_entities_global(const Cell& cell,
      std::vector<std::vector<std::size_t>>& entity_indices,
      const std::vector<bool>& needs_mesh_entities);

    static void get_cell_entities_global_constrained(const Cell& cell,
      std::vector<std::vector<std::size_t>>& entity_indices,
      const std::vector<std::vector<std::size_t>>& global_entity_indices,
      const std::vector<bool>& needs_mesh_entities);

    // Compute number of mesh entities for dimensions required by
    // dofmap
    static std::vector<std::size_t> compute_num_mesh_entities_local(
      const Mesh& mesh, const std::vector<bool>& needs_mesh_entities);

  };
}

#endif
