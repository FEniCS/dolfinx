// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMapBuilder.h"
#include "DofMap.h"
#include "ElementDofLayout.h"
#include <cstdlib>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/BoostGraphOrdering.h>
#include <dolfinx/graph/GraphBuilder.h>
#include <dolfinx/graph/SCOTCH.h>
#include <dolfinx/mesh/DistributedMeshTools.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <memory>
#include <numeric>
#include <random>
#include <utility>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
//-----------------------------------------------------------------------------
struct DofMapStructure
{
  std::vector<PetscInt> data;
  std::vector<std::int32_t> cell_ptr;

  std::int32_t num_cells() const { return cell_ptr.size() - 1; }
  std::int32_t num_dofs(std::int32_t cell) const
  {
    return cell_ptr[cell + 1] - cell_ptr[cell];
  }
  const std::int32_t& dof(int cell, int i) const
  {
    return data[cell_ptr[cell] + i];
  }
  std::int32_t& dof(int cell, int i) { return data[cell_ptr[cell] + i]; }

  const std::int32_t* dofs(int cell) const { return &data[cell_ptr[cell]]; }
  std::int32_t* dofs(int cell) { return &data[cell_ptr[cell]]; }
};
//-----------------------------------------------------------------------------
// Build a simple dofmap from ElementDofmap based on mesh entity indices
std::tuple<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>,
           std::vector<std::pair<std::int8_t, std::int32_t>>>
build_basic_dofmap(const mesh::Mesh& mesh,
                   const ElementDofLayout& element_dof_layout)
{
  // Start timer for dofmap initialization
  common::Timer t0("Init dofmap from element dofmap");

  // Topological dimension
  const int D = mesh.topology().dim();

  // Generate and number required mesh entities
  std::vector<bool> needs_entities(D + 1, false);
  std::vector<std::int32_t> num_mesh_entities_local(D + 1, 0),
      num_mesh_entities_global(D + 1, 0);
  for (int d = 0; d <= D; ++d)
  {
    if (element_dof_layout.num_entity_dofs(d) > 0)
    {
      needs_entities[d] = true;
      mesh.create_entities(d);
      num_mesh_entities_local[d] = mesh.num_entities(d);
      num_mesh_entities_global[d] = mesh.num_entities_global(d);
    }
  }

  // Collect cell -> entity connectivities
  std::vector<std::shared_ptr<const graph::AdjacencyList<std::int32_t>>>
      connectivity;
  for (int d = 0; d <= D; ++d)
    connectivity.push_back(mesh.topology().connectivity(D, d));

  // Number of dofs on this process
  std::int32_t local_size(0), d(0);
  for (std::int32_t n : num_mesh_entities_local)
    local_size += n * element_dof_layout.num_entity_dofs(d++);

  // Number of dofs per cell
  const int local_dim = element_dof_layout.num_dofs();

  // Allocate dofmap memory
  DofMapStructure dofmap;
  dofmap.data.resize(mesh.num_entities(D) * local_dim);
  dofmap.cell_ptr.resize(mesh.num_entities(D) + 1, local_dim);
  dofmap.cell_ptr[0] = 0;
  std::partial_sum(dofmap.cell_ptr.begin() + 1, dofmap.cell_ptr.end(),
                   dofmap.cell_ptr.begin() + 1);

  // Allocate entity indices array
  std::vector<std::vector<int32_t>> entity_indices_local(D + 1);
  std::vector<std::vector<int64_t>> entity_indices_global(D + 1);
  for (int d = 0; d <= D; ++d)
  {
    const int num_entities = mesh::cell_num_entities(mesh.cell_type(), d);
    entity_indices_local[d].resize(num_entities);
    entity_indices_global[d].resize(num_entities);
  }

  // Entity dofs on cell (dof = entity_dofs[dim][entity][index])
  const std::vector<std::vector<std::set<int>>>& entity_dofs
      = element_dof_layout.entity_dofs_all();

  // Compute cell dof permutations
  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      permutations = fem::compute_dof_permutations(mesh, element_dof_layout);

  // Storage for local-to-global map
  std::vector<std::int64_t> local_to_global(local_size);

  // Dof (dim, entity index) marker
  std::vector<std::pair<std::int8_t, std::int32_t>> dof_entity(local_size);

  // Build dofmaps from ElementDofmap
  for (int c = 0; c < connectivity[0]->num_nodes(); ++c)
  {
    // Get local (process) and global cell entity indices
    for (int d = 0; d < D; ++d)
    {
      if (needs_entities[d])
      {
        const std::vector<std::int64_t>& global_indices
            = mesh.topology().global_indices(d);
        assert(global_indices.size() > 0);
        auto entities = connectivity[d]->edges(c);
        for (int i = 0; i < entities.rows(); ++i)
        {
          entity_indices_local[d][i] = entities[i];
          entity_indices_global[d][i] = global_indices[entities[i]];
        }
      }
    }

    // Handle cell index separately because cell.entities(D) doesn't work.
    if (needs_entities[D])
    {
      const std::vector<std::int64_t>& global_indices
          = mesh.topology().global_indices(D);
      entity_indices_global[D][0] = global_indices[c];
      entity_indices_local[D][0] = c;
    }

    // Iterate over each topological dimension of cell
    std::int32_t offset_local = 0;
    std::int64_t offset_global = 0;
    for (auto e_dofs_d = entity_dofs.begin(); e_dofs_d != entity_dofs.end();
         ++e_dofs_d)
    {
      const std::int8_t d = std::distance(entity_dofs.begin(), e_dofs_d);

      // Iterate over each entity of current dimension d
      for (auto e_dofs = e_dofs_d->begin(); e_dofs != e_dofs_d->end(); ++e_dofs)
      {
        // Get entity indices (local to cell, local to process, and
        // global)
        const std::int32_t e = std::distance(e_dofs_d->begin(), e_dofs);
        const std::int32_t e_index_local = entity_indices_local[d][e];
        const std::int64_t e_index_global = entity_indices_global[d][e];

        // Loop over dofs belong to entity e of dimension d (d, e)
        // d: topological dimension
        // e: local entity index
        // dof_local: local index of dof at (d, e)
        const std::int32_t num_entity_dofs = e_dofs->size();
        for (auto dof_local = e_dofs->begin(); dof_local != e_dofs->end();
             ++dof_local)
        {
          const std::int32_t count = std::distance(e_dofs->begin(), dof_local);
          const std::int32_t dof
              = offset_local + num_entity_dofs * e_index_local + count;
          dofmap.dof(c, permutations(c, *dof_local)) = dof;
          local_to_global[dof]
              = offset_global + num_entity_dofs * e_index_global + count;
          dof_entity[dof] = {d, e_index_local};
        }
      }
      offset_local += entity_dofs[d][0].size() * num_mesh_entities_local[d];
      offset_global += entity_dofs[d][0].size() * num_mesh_entities_global[d];
    }
  }

  return {graph::AdjacencyList<std::int32_t>(dofmap.data, dofmap.cell_ptr),
          local_to_global, dof_entity};
}
//-----------------------------------------------------------------------------

/// Compute re-ordering map from old local index to new local index. The
/// M dofs owned by this process are reordered for locality and fill the
/// positions [0, ..., M). Dof owned by another process are placed at
/// the end, i.e. in the positions [M, ..., N), where N is the total
/// number of dofs on this process.
///
/// @param [in] dofmap The basic dofmap data
/// @param [in] topology The mesh topology
/// @return The pair (old-to-new local index map, M), where M is the
///   number of dofs owned by this process
std::pair<std::vector<std::int32_t>, std::int32_t> compute_reordering_map(
    const graph::AdjacencyList<std::int32_t>& dofmap,
    const std::vector<std::pair<std::int8_t, std::int32_t>>& dof_entity,
    const mesh::Topology& topology)
{
  // Get ownership offset for each dimension
  const int D = topology.dim();
  std::vector<std::int32_t> offset(D + 1, -1);
  for (std::size_t d = 0; d < offset.size(); ++d)
  {
    auto map = topology.index_map(d);
    if (map)
      offset[d] = map->size_local();
  }

  // Count locally owned dofs
  std::vector<bool> owned(dof_entity.size(), false);
  for (auto e = dof_entity.begin(); e != dof_entity.end(); ++e)
  {
    if (e->second < offset[e->first])
    {
      const std::size_t i = std::distance(dof_entity.begin(), e);
      owned[i] = true;
    }
  }

  // Create map from old index to new contiguous numbering for locally
  // owned dofs. Set to -1 for unowned dofs.
  std::vector<int> original_to_contiguous(dof_entity.size(), -1);
  std::int32_t owned_size = 0;
  for (std::size_t i = 0; i < original_to_contiguous.size(); ++i)
  {
    if (owned[i])
      original_to_contiguous[i] = owned_size++;
  }

  // Build local graph, based on dof map with contiguous numbering
  // (unowned dofs excluded)
  dolfinx::graph::Graph graph(owned_size);
  std::vector<int> local_old;
  for (std::int32_t cell = 0; cell < dofmap.num_nodes(); ++cell)
  {
    // Loop over nodes collecting valid local nodes
    local_old.clear();
    auto nodes = dofmap.edges(cell);
    for (std::int32_t i = 0; i < nodes.rows(); ++i)
    {
      // Add to graph if node is owned
      assert(nodes[i] < (int)original_to_contiguous.size());
      const int n = original_to_contiguous[nodes[i]];
      if (n != -1)
      {
        assert(n < (int)graph.size());
        local_old.push_back(n);
      }
    }

    for (std::size_t i = 0; i < local_old.size(); ++i)
      for (std::size_t j = 0; j < local_old.size(); ++j)
        if (i != j)
          graph[local_old[i]].insert(local_old[j]);
  }

  // Reorder owned nodes
  const std::string ordering_library = "SCOTCH";
  std::vector<int> node_remap;
  if (ordering_library == "Boost")
    node_remap = graph::BoostGraphOrdering::compute_cuthill_mckee(graph, true);
  else if (ordering_library == "SCOTCH")
    std::tie(node_remap, std::ignore) = graph::SCOTCH::compute_gps(graph);
  else if (ordering_library == "random")
  {
    // NOTE: Randomised dof ordering should only be used for
    // testing/benchmarking
    node_remap.resize(graph.size());
    std::iota(node_remap.begin(), node_remap.end(), 0);
    std::random_device rd;
    std::default_random_engine g(rd());
    std::shuffle(node_remap.begin(), node_remap.end(), g);
  }
  else
  {
    throw std::runtime_error("Requested library '" + ordering_library
                             + "' is unknown");
  }

  // Reconstruct remaped nodes, and place un-owned nodes at the end
  std::vector<int> old_to_new(dof_entity.size(), -1);
  std::int32_t unowned_pos = owned_size;
  assert(old_to_new.size() == original_to_contiguous.size());
  for (std::size_t i = 0; i < original_to_contiguous.size(); ++i)
  {
    // Put nodes that are not owned at the end, otherwise re-number
    const std::int32_t index = original_to_contiguous[i];
    if (index >= 0)
      old_to_new[i] = node_remap[index];
    else
      old_to_new[i] = unowned_pos++;
  }

  return {old_to_new, owned_size};
}
//-----------------------------------------------------------------------------

/// Get global indices for unowned dofs
/// @param [in] topology The mesh topology
/// @param [in] num_owned The number of nodes owned by this process
/// @param [in] process_offset The node offset for this process, i.e.
///   the global index of owned node i is i + process_offset
/// @param [in] global_indices_old The old global index of the old local
///   node i
/// @param [in] old_to_new The old local index to new local index map
/// @param [in] dof_entity The ith entry gives (topological dim, local
///   index) of the mesh entity to which node i (old local index) is
///   associated
std::vector<std::int64_t> get_global_indices(
    const mesh::Topology& topology, const std::int32_t num_owned,
    const std::int64_t process_offset,
    const std::vector<std::int64_t>& global_indices_old,
    const std::vector<std::int32_t>& old_to_new,
    const std::vector<std::pair<std::int8_t, std::int32_t>>& dof_entity)
{
  const int D = topology.dim();

  // Get ownership offset for each dimension
  // Build list flag for owned mesh entities that are shared, i.e. are a
  // ghost on a neighbour
  std::vector<std::int32_t> offset(D + 1, -1);
  std::vector<std::vector<bool>> shared_entity(D + 1);
  for (std::size_t d = 0; d < shared_entity.size(); ++d)
  {
    auto map = topology.index_map(d);
    if (map)
    {
      offset[d] = map->size_local();
      shared_entity[d] = std::vector<bool>(offset[d], false);
      const std::vector<std::int32_t>& forward_indices = map->forward_indices();
      for (auto entity : forward_indices)
        shared_entity[d][entity] = true;
    }
  }

  // Build  [local_new - num_owned] -> global old array
  std::vector<std::int64_t> local_new_to_global_old(old_to_new.size()
                                                    - num_owned);
  for (std::size_t i = 0; i < global_indices_old.size(); ++i)
  {
    std::int32_t local_new = old_to_new[i] - num_owned;
    if (local_new >= 0)
      local_new_to_global_old[local_new] = global_indices_old[i];
  }

  // Build list of (global old, global new) index pairs for dofs that
  // are ghosted on other processes
  std::vector<std::vector<std::int64_t>> global(D + 1);

  // Loop over all dofs
  for (std::size_t i = 0; i < dof_entity.size(); ++i)
  {
    // Topological dimension of mesh entity that dof is associated with
    const int d = dof_entity[i].first;

    // Index of mesh entity that dof is associated with
    assert(offset[d] != -1);
    const int entity = dof_entity[i].second;

    if (shared_entity[d][entity])
    {
      global[d].push_back(global_indices_old[i]);
      global[d].push_back(old_to_new[i] + process_offset);
    }
  }

  std::map<std::int64_t, std::int64_t> global_old_new;
  for (int d = 0; d <= D; ++d)
  {
    auto map = topology.index_map(d);
    if (map)
    {
      // Get number of processes in neighbourhood
      MPI_Comm comm = map->mpi_comm_neighborhood();
      int num_neighbours(-1), outdegree(-2), weighted(-1);
      MPI_Dist_graph_neighbors_count(comm, &num_neighbours, &outdegree,
                                     &weighted);
      assert(num_neighbours == outdegree);

      // Number and values to send and receive
      const int num_indices = global[d].size();
      std::vector<int> num_indices_recv(num_neighbours);
      MPI_Neighbor_allgather(&num_indices, 1, MPI_INT, num_indices_recv.data(),
                             1, MPI_INT, comm);

      // Compute displacements for data to receive. Last entry has total
      // number of received items.
      std::vector<int> disp(num_neighbours + 1, 0);
      std::partial_sum(num_indices_recv.begin(), num_indices_recv.end(),
                       disp.begin() + 1);

      // Send/receive global index of dofs with bcs to all neighbours
      std::vector<std::int64_t> dofs_received(disp.back());
      MPI_Neighbor_allgatherv(global[d].data(), global[d].size(), MPI_INT64_T,
                              dofs_received.data(), num_indices_recv.data(),
                              disp.data(), MPI_INT64_T, comm);

      // Build (global old, global new) map
      for (std::size_t i = 0; i < dofs_received.size(); i += 2)
        global_old_new.insert({dofs_received[i], dofs_received[i + 1]});
    }
  }

  std::vector<std::int64_t> local_to_global_new(old_to_new.size() - num_owned);
  for (std::size_t i = 0; i < local_new_to_global_old.size(); ++i)
  {
    auto it = global_old_new.find(local_new_to_global_old[i]);
    assert(it != global_old_new.end());
    local_to_global_new[i] = it->second;
  }

  return local_to_global_new;
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
fem::DofMap
DofMapBuilder::build(const mesh::Mesh& mesh,
                     std::shared_ptr<const ElementDofLayout> element_dof_layout)
{
  assert(element_dof_layout);
  const int bs = element_dof_layout->block_size();
  std::shared_ptr<common::IndexMap> index_map;
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofmap;
  if (bs == 1)
  {
    std::tie(index_map, dofmap)
        = DofMapBuilder::build(mesh, *element_dof_layout, 1);
  }
  else
  {
    std::tie(index_map, dofmap)
        = DofMapBuilder::build(mesh, *element_dof_layout->sub_dofmap({0}), bs);
  }

  return fem::DofMap(element_dof_layout, index_map, dofmap);
}
//-----------------------------------------------------------------------------
fem::DofMap DofMapBuilder::build_submap(const DofMap& dofmap_parent,
                                        const std::vector<int>& component,
                                        const mesh::Mesh& mesh)
{
  assert(!component.empty());
  const int D = mesh.topology().dim();

  // Set element dof layout and cell dimension
  std::shared_ptr<const ElementDofLayout> element_dof_layout
      = dofmap_parent.element_dof_layout->sub_dofmap(component);

  // Get components in parent map that correspond to sub-dofs
  assert(dofmap_parent.element_dof_layout);
  const std::vector<int> element_map_view
      = dofmap_parent.element_dof_layout->sub_view(component);

  // Build dofmap by extracting from parent
  const std::int32_t dofs_per_cell = element_map_view.size();
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofmap(dofs_per_cell
                                                   * mesh.num_entities(D));
  for (auto& cell : mesh::MeshRange(mesh, D))
  {
    const int c = cell.index();
    auto cell_dmap_parent = dofmap_parent.cell_dofs(c);
    for (std::int32_t i = 0; i < dofs_per_cell; ++i)
      dofmap[c * dofs_per_cell + i] = cell_dmap_parent[element_map_view[i]];
  }

  return DofMap(element_dof_layout, dofmap_parent.index_map, dofmap);
}
//-----------------------------------------------------------------------------
std::tuple<std::unique_ptr<common::IndexMap>,
           Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
DofMapBuilder::build(const mesh::Mesh& mesh,
                     const ElementDofLayout& element_dof_layout,
                     const std::int32_t block_size)
{
  common::Timer t0("Init dofmap");

  if (element_dof_layout.block_size() != 1)
    throw std::runtime_error("Block size of 1 expected when building dofmap.");

  const int D = mesh.topology().dim();

  // Build a simple dofmap based on mesh entity numbering. Returns:
  //  - dofmap (local indices)
  //  - local-to-global dof index map
  const auto [node_graph0, local_to_global0, dof_entity0]
      = build_basic_dofmap(mesh, element_dof_layout);

  // Compute global dofmap dimension
  std::int64_t global_dimension = 0;
  for (int d = 0; d < D + 1; ++d)
  {
    if (element_dof_layout.num_entity_dofs(d) > 0)
    {
      mesh.create_entities(d);
      const std::int64_t n = mesh.num_entities_global(d);
      global_dimension += n * element_dof_layout.num_entity_dofs(d);
    }
  }

  // Build re-ordering map for data locality and get number of owned
  // nodes
  const auto [old_to_new, num_owned]
      = compute_reordering_map(node_graph0, dof_entity0, mesh.topology());

  // Compute process offset for owned nodes
  const std::int64_t process_offset
      = dolfinx::MPI::global_offset(mesh.mpi_comm(), num_owned, true);

  // Get global indices for unowned dofs
  const std::vector<std::int64_t> local_to_global_unowned
      = get_global_indices(mesh.topology(), num_owned, process_offset,
                           local_to_global0, old_to_new, dof_entity0);

  // Create IndexMap for dofs range on this process
  auto index_map = std::make_unique<common::IndexMap>(
      mesh.mpi_comm(), num_owned, local_to_global_unowned, block_size);
  assert(index_map);
  assert(
      dolfinx::MPI::sum(mesh.mpi_comm(), (std::int64_t)index_map->size_local())
      == global_dimension);

  // FIXME: There is an assumption here on the dof order for an element.
  //        It should come from the ElementDofLayout.
  // Build re-ordered dofmap, accounting for block size
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofmap(node_graph0.array().rows()
                                                   * block_size);
  for (std::int32_t cell = 0; cell < node_graph0.num_nodes(); ++cell)
  {
    const std::int32_t local_dim0 = node_graph0.num_edges(cell);
    auto old_nodes = node_graph0.edges(cell);
    for (std::int32_t j = 0; j < local_dim0; ++j)
    {
      const std::int32_t old_node = old_nodes[j];
      const std::int32_t new_node = old_to_new[old_node];
      for (std::int32_t block = 0; block < block_size; ++block)
      {
        dofmap[cell * block_size * local_dim0 + block * local_dim0 + j]
            = block_size * new_node + block;
      }
    }
  }

  return std::make_tuple(std::move(index_map), std::move(dofmap));
}
//-----------------------------------------------------------------------------
