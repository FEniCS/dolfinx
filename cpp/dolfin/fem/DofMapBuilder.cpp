// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMapBuilder.h"
#include "DofMap.h"
#include "ElementDofLayout.h"
#include <cstdlib>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
#include <dolfin/graph/BoostGraphOrdering.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/graph/SCOTCH.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <memory>
#include <numeric>
#include <random>
#include <stdlib.h>
#include <utility>

using namespace dolfin;
using namespace dolfin::fem;

namespace
{
//-----------------------------------------------------------------------------
struct DofMapStructure
{
  std::vector<PetscInt> data;
  std::vector<std::int32_t> cell_ptr;
  std::vector<std::int64_t> global_indices;

  std::int32_t num_cells() const { return cell_ptr.size() - 1; }
  std::int32_t num_dofs(std::int32_t cell) const
  {
    return cell_ptr[cell + 1] - cell_ptr[cell];
  }
  const PetscInt& dof(int cell, int i) const
  {
    return data[cell_ptr[cell] + i];
  }
  PetscInt& dof(int cell, int i) { return data[cell_ptr[cell] + i]; }

  const PetscInt* dofs(int cell) const { return &data[cell_ptr[cell]]; }
  PetscInt* dofs(int cell) { return &data[cell_ptr[cell]]; }
};
//-----------------------------------------------------------------------------
void get_cell_entities(
    std::vector<std::vector<std::int32_t>>& entity_indices_local,
    std::vector<std::vector<std::int64_t>>& entity_indices_global,
    const mesh::Cell& cell, const std::vector<bool>& needs_mesh_entities)
{
  const mesh::MeshTopology& topology = cell.mesh().topology();
  const int D = topology.dim();
  for (int d = 0; d < D; ++d)
  {
    if (needs_mesh_entities[d])
    {
      assert(topology.have_global_indices(d));
      const std::vector<std::int64_t>& global_indices
          = topology.global_indices(d);
      const std::int32_t* entities = cell.entities(d);
      for (std::size_t i = 0; i < cell.num_entities(d); ++i)
      {
        entity_indices_local[d][i] = entities[i];
        entity_indices_global[d][i] = global_indices[entities[i]];
      }
    }
  }
  // Handle cell index separately because cell.entities(D) doesn't work.
  if (needs_mesh_entities[D])
  {
    entity_indices_global[D][0] = cell.global_index();
    entity_indices_local[D][0] = cell.index();
  }
}
//-----------------------------------------------------------------------------
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
std::tuple<int, std::vector<std::int8_t>,
           std::unordered_map<int, std::vector<int>>, std::set<int>>
compute_node_ownership(const DofMapStructure& dofmap,
                       const std::vector<std::int8_t>& shared_nodes,
                       const mesh::Mesh& mesh, const std::size_t global_dim)
{
  // Get number of nodes
  const std::size_t num_nodes_local = dofmap.global_indices.size();

  // Global-to-local node map for nodes on boundary
  std::map<std::size_t, int> global_to_local;

  // Initialise node ownership array, provisionally all owned
  std::vector<std::int8_t> node_ownership(num_nodes_local, 1);

  // Communication buffers
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const std::size_t num_processes = dolfin::MPI::size(mpi_comm);
  const std::size_t process_number = dolfin::MPI::rank(mpi_comm);
  std::vector<std::vector<std::size_t>> send_buffer(num_processes);
  std::vector<std::vector<std::size_t>> recv_buffer(num_processes);

  // Add a counter to the start of each send buffer
  for (std::uint32_t i = 0; i != num_processes; ++i)
    send_buffer[i].push_back(0);

  // FIXME: could get rid of global_to_local map since response will
  // come back in same order

  // Loop over nodes and buffer nodes on process boundaries
  for (std::size_t i = 0; i < num_nodes_local; ++i)
  {
    if (shared_nodes[i] == 0)
    {
      // Shared node - send out to matching process to determine
      // ownership and other sharing processes

      // Send global index
      const std::size_t global_index = dofmap.global_indices[i];
      const std::size_t dest
          = dolfin::MPI::index_owner(mpi_comm, global_index, global_dim);
      send_buffer[dest].push_back(global_index);
      global_to_local.insert(std::make_pair(global_index, i));
    }
  }

  // Make note of current size of each send buffer i.e. the number of
  // boundary nodes, labelled '0'
  for (std::uint32_t i = 0; i != num_processes; ++i)
    send_buffer[i][0] = send_buffer[i].size() - 1;

  // Additionally send any ghost or ghost-shared nodes to determine
  // sharing (but not ownership)
  for (std::size_t i = 0; i < num_nodes_local; ++i)
  {
    if (shared_nodes[i] == -3 or shared_nodes[i] == -2)
    {
      // Send global index
      const std::size_t global_index = dofmap.global_indices[i];
      const std::size_t dest
          = dolfin::MPI::index_owner(mpi_comm, global_index, global_dim);
      send_buffer[dest].push_back(global_index);
      global_to_local.insert(std::make_pair(global_index, i));
    }
  }

  // Send to sorting process
  dolfin::MPI::all_to_all(mpi_comm, send_buffer, recv_buffer);

  // Map from global index to sharing processes
  std::map<std::size_t, std::vector<std::uint32_t>> global_to_procs;
  for (std::uint32_t i = 0; i != num_processes; ++i)
  {
    const std::vector<std::size_t>& recv_i = recv_buffer[i];
    const std::size_t num_boundary_nodes = recv_i[0];

    for (std::uint32_t j = 1; j != num_boundary_nodes + 1; ++j)
    {
      auto map_it = global_to_procs.find(recv_i[j]);
      if (map_it == global_to_procs.end())
        global_to_procs.insert(
            std::make_pair(recv_i[j], std::vector<std::uint32_t>(1, i)));
      else
        map_it->second.push_back(i);
    }
  }

  // Randomise process order. First process will be owner.
  const std::size_t seed = process_number;
  std::default_random_engine random_engine(seed);
  for (auto p = global_to_procs.begin(); p != global_to_procs.end(); ++p)
    std::shuffle(p->second.begin(), p->second.end(), random_engine);

  // Add other sharing processes (ghosts etc) which cannot be owners
  for (std::uint32_t i = 0; i != num_processes; ++i)
  {
    const std::vector<std::size_t>& recv_i = recv_buffer[i];
    const std::size_t num_boundary_nodes = recv_i[0];

    for (std::uint32_t j = num_boundary_nodes + 1; j != recv_i.size(); ++j)
    {
      auto map_it = global_to_procs.find(recv_i[j]);
      if (map_it == global_to_procs.end())
        global_to_procs.insert(
            std::make_pair(recv_i[j], std::vector<std::uint32_t>(1, i)));
      else
        map_it->second.push_back(i);
    }
  }

  // Send response back to originators in same order
  std::vector<std::vector<std::size_t>> send_response(num_processes);
  for (std::uint32_t i = 0; i != num_processes; ++i)
    for (auto q = recv_buffer[i].begin() + 1; q != recv_buffer[i].end(); ++q)
    {
      std::vector<std::uint32_t>& gprocs = global_to_procs[*q];
      send_response[i].push_back(gprocs.size());
      send_response[i].insert(send_response[i].end(), gprocs.begin(),
                              gprocs.end());
    }

  dolfin::MPI::all_to_all(mpi_comm, send_response, recv_buffer);
  // [n_sharing, owner, others]
  std::unordered_map<int, std::vector<int>> shared_node_to_processes;
  for (std::uint32_t i = 0; i != num_processes; ++i)
  {
    auto q = recv_buffer[i].begin();
    for (auto p = send_buffer[i].begin() + 1; p != send_buffer[i].end(); ++p)
    {
      const std::uint32_t num_sharing = *q;
      if (num_sharing > 1)
      {
        const std::size_t global_index = *p;
        const std::size_t owner = *(q + 1);
        std::set<std::size_t> sharing_procs(q + 1, q + 1 + num_sharing);
        sharing_procs.erase(process_number);

        auto it = global_to_local.find(global_index);
        assert(it != global_to_local.end());
        const int received_node_local = it->second;
        const int node_status = shared_nodes[received_node_local];
        assert(node_status != -1);

        // First check to see if this is a ghost/ghost-shared node, and
        // set ownership accordingly. Otherwise use the ownership from
        // the sorting process
        if (node_status == -2)
          node_ownership[received_node_local] = 0;
        else if (node_status == -3)
          node_ownership[received_node_local] = -1;
        else if (owner == process_number)
          node_ownership[received_node_local] = 0;
        else
          node_ownership[received_node_local] = -1;

        shared_node_to_processes[received_node_local]
            = std::vector<int>(sharing_procs.begin(), sharing_procs.end());
      }

      q += num_sharing + 1;
    }
  }

  // Build set of neighbouring processes
  std::set<int> neighbouring_procs;
  for (auto it = shared_node_to_processes.begin();
       it != shared_node_to_processes.end(); ++it)
  {
    neighbouring_procs.insert(it->second.begin(), it->second.end());
  }

  // Count number of owned nodes
  int num_owned_nodes = 0;
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    if (node_ownership[i] >= 0)
      ++num_owned_nodes;
  }

  return std::make_tuple(std::move(num_owned_nodes), std::move(node_ownership),
                         std::move(shared_node_to_processes),
                         std::move(neighbouring_procs));
}
//-----------------------------------------------------------------------------
// TODO: Make clear what is being assumed on the dof order for an
// element
// Build dofmap based on re-ordered nodes
std::vector<std::vector<PetscInt>>
build_dofmap(const DofMapStructure& node_dofmap,
             const std::vector<int>& old_to_new_node_local,
             const std::size_t block_size)
{
  std::vector<std::vector<PetscInt>> dofmap(node_dofmap.num_cells());
  for (std::size_t i = 0; i < dofmap.size(); ++i)
  {
    const std::size_t local_dim0 = node_dofmap.num_dofs(i);
    dofmap[i].resize(block_size * local_dim0);
    for (std::size_t j = 0; j < local_dim0; ++j)
    {
      const int old_node = node_dofmap.dof(i, j);
      assert(old_node < (int)old_to_new_node_local.size());
      const int new_node = old_to_new_node_local[old_node];
      for (std::size_t block = 0; block < block_size; ++block)
      {
        assert((block * local_dim0 + j) < dofmap[i].size());
        dofmap[i][block * local_dim0 + j] = block_size * new_node + block;
      }
    }
  }
  return dofmap;
}
//-----------------------------------------------------------------------------
// Build a simple dofmap from ElementDofmap based on mesh entity indices
DofMapStructure build_basic_dofmap(const mesh::Mesh& mesh,
                                   const ElementDofLayout& element_dof_layout)
{
  // Start timer for dofmap initialization
  common::Timer t0("Init dofmap from element dofmap");

  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Generate and number required mesh entities
  std::vector<bool> needs_entities(D + 1, false);
  std::vector<std::int32_t> num_mesh_entities_local(D + 1, 0),
      num_mesh_entities_global(D + 1, 0);
  for (std::size_t d = 0; d <= D; ++d)
  {
    if (element_dof_layout.num_entity_dofs(d) > 0)
    {
      needs_entities[d] = true;
      mesh.init(d);
      mesh::DistributedMeshTools::number_entities(mesh, d);
      num_mesh_entities_local[d] = mesh.num_entities(d);
      num_mesh_entities_global[d] = mesh.num_entities_global(d);
    }
  }

  // Number of dofs on this process
  std::int32_t local_size(0), d(0);
  for (std::int32_t n : num_mesh_entities_local)
    local_size += n * element_dof_layout.num_entity_dofs(d++);

  // Number of dofs per cell
  const int local_dim = element_dof_layout.num_dofs();

  // Allocate dofmap memory
  DofMapStructure dofmap;
  dofmap.global_indices.resize(local_size);
  dofmap.data.resize(mesh.num_entities(D) * local_dim);
  dofmap.cell_ptr.resize(mesh.num_entities(D) + 1, local_dim);
  dofmap.cell_ptr[0] = 0;
  std::partial_sum(dofmap.cell_ptr.begin() + 1, dofmap.cell_ptr.end(),
                   dofmap.cell_ptr.begin() + 1);

  // Allocate entity indices array
  std::vector<std::vector<int32_t>> entity_indices_local(D + 1);
  std::vector<std::vector<int64_t>> entity_indices_global(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
  {
    entity_indices_local[d].resize(mesh.type().num_entities(d));
    entity_indices_global[d].resize(mesh.type().num_entities(d));
  }

  // Build dofmaps from ElementDofmap
  std::vector<std::size_t> ufc_nodes_global(local_dim);
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    // Get local (process) and global cell entity indices
    get_cell_entities(entity_indices_local, entity_indices_global, cell,
                      needs_entities);

    // Entity dofs on cell (dof = entity_dofs[dim][entity][index])
    const std::vector<std::vector<std::set<int>>>& entity_dofs
        = element_dof_layout.entity_dofs();

    // Iterate over topological dimensions
    std::int32_t offset_local = 0;
    std::int64_t offset_global = 0;
    for (auto e_dofs_d = entity_dofs.begin(); e_dofs_d != entity_dofs.end();
         ++e_dofs_d)
    {
      const std::int32_t d = std::distance(entity_dofs.begin(), e_dofs_d);

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
          dofmap.dof(cell.index(), *dof_local) = dof;
          dofmap.global_indices[dof]
              = offset_global + num_entity_dofs * e_index_global + count;
        }
      }
      offset_local += entity_dofs[d][0].size() * num_mesh_entities_local[d];
      offset_global += entity_dofs[d][0].size() * num_mesh_entities_global[d];
    }
  }

  return dofmap;
}
//-----------------------------------------------------------------------------
// Compute sharing marker for each node. Boundary nodes are assigned a
// positive integer, interior nodes are marked as -1, interior nodes in
// ghost layer of other processes are marked -2, and ghost nodes are
// marked as -3
std::vector<std::int8_t>
compute_sharing_markers(const DofMapStructure& node_dofmap,
                        const ElementDofLayout& element_dof_layout,
                        const mesh::Mesh& mesh)
{
  // Initialise mesh
  const int D = mesh.topology().dim();

  // Allocate data and initialise all nodes to -1 (provisionally, owned
  // and not shared)
  std::vector<std::int8_t> shared_nodes(node_dofmap.global_indices.size(), -1);

  // Get facet closure dofs
  const std::vector<std::set<int>>& facet_table
      = element_dof_layout.entity_closure_dofs()[D - 1];

  // Mark dofs associated ghost cells as ghost dofs (-3), provisionally
  bool has_ghost_cells = false;
  for (auto& c : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    const PetscInt* cell_nodes = node_dofmap.dofs(c.index());
    if (c.is_shared())
    {
      const int status = (c.is_ghost()) ? -3 : -2;
      for (std::int32_t i = 0; i < node_dofmap.num_dofs(c.index()); ++i)
      {
        // Ensure not already set (for R space)
        if (shared_nodes[cell_nodes[i]] == -1)
          shared_nodes[cell_nodes[i]] = status;
      }
    }

    // Change all non-ghost facet dofs of ghost cells to '0'
    if (c.is_ghost())
    {
      has_ghost_cells = true;
      for (auto& f : mesh::EntityRange<mesh::Facet>(c))
      {
        if (!f.is_ghost())
        {
          const std::set<int>& facet_nodes = facet_table[c.index(f)];
          for (auto facet_node : facet_nodes)
          {
            const int facet_node_local = cell_nodes[facet_node];
            shared_nodes[facet_node_local] = 0;
          }
        }
      }
    }
  }

  if (has_ghost_cells)
    return shared_nodes;

  // Mark nodes on inter-process boundary
  for (auto& f : mesh::MeshRange<mesh::Facet>(mesh, mesh::MeshRangeType::ALL))
  {
    // Skip if facet is not shared
    // NOTE: second test is for periodic problems
    if (!f.is_shared() and f.num_entities(D) == 2)
      continue;

    // Get cell to which facet belongs (pick first)
    const mesh::Cell cell0(mesh, f.entities(D)[0]);

    // Get dofs (process-wise indices) on cell
    const PetscInt* cell_nodes = node_dofmap.dofs(cell0.index());

    // Get dofs which are on the facet
    const std::set<int>& facet_nodes = facet_table[cell0.index(f)];

    // Mark boundary nodes and insert into map
    for (auto facet_node : facet_nodes)
    {
      // Get facet node local index and assign "0" - shared, owner
      // unassigned
      PetscInt facet_node_local = cell_nodes[facet_node];
      if (shared_nodes[facet_node_local] < 0)
        shared_nodes[facet_node_local] = 0;
    }
  }

  return shared_nodes;
}
//-----------------------------------------------------------------------------
// Compute re-ordering map of indices.
std::vector<std::int32_t>
compute_reordering_map(const DofMapStructure& node_dofmap,
                       const std::vector<std::int8_t>& node_ownership)
{
  // Create map from old index to new contiguous numbering for locally
  // owned dofs. Set to -1 for unowned dofs.
  std::int32_t owned_size = 0;
  std::vector<int> original_to_contiguous(node_ownership.size(), -1);
  for (std::size_t i = 0; i < original_to_contiguous.size(); ++i)
  {
    if (node_ownership[i] >= 0)
      original_to_contiguous[i] = owned_size++;
  }

  // Build local graph, based on dof map with contiguous numbering
  // (unowned dofs excluded)
  dolfin::graph::Graph graph(owned_size);
  std::vector<int> local_old;
  for (std::int32_t cell = 0; cell < node_dofmap.num_cells(); ++cell)
  {
    // Loop over nodes collecting valid local nodes
    local_old.clear();
    const PetscInt* nodes = node_dofmap.dofs(cell);
    for (std::int32_t i = 0; i < node_dofmap.num_dofs(cell); ++i)
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
    for (std::size_t i = 0; i < node_remap.size(); ++i)
      node_remap[i] = i;
    std::random_shuffle(node_remap.begin(), node_remap.end());
  }
  else
  {
    throw std::runtime_error("Requested library '" + ordering_library
                             + "' is unknown");
  }

  // Reconstruct remaped nodes, with -1 for unowned
  std::vector<int> old_to_new(node_ownership.size(), -1);
  std::int32_t unowned_pos = owned_size;
  for (std::size_t old_index = 0; old_index < node_ownership.size();
       ++old_index)
  {
    std::int32_t index = original_to_contiguous[old_index];

    // Skip nodes that are not owned
    if (index < 0)
    {
      assert(old_index < old_to_new.size());
      old_to_new[old_index] = unowned_pos++;
    }
    else
    {
      // Set new node number
      assert(old_index < old_to_new.size());
      old_to_new[old_index] = node_remap[index];
    }
  }

  return old_to_new;
}
//-----------------------------------------------------------------------------
// Compute global indices for unowned dofs
std::vector<std::size_t> compute_global_indices(
    const std::size_t process_offset,
    const std::vector<std::int32_t>& old_to_new,
    const std::unordered_map<int, std::vector<int>>& node_to_sharing_processes,
    const DofMapStructure& node_dofmap,
    const std::vector<std::int8_t>& node_ownership, MPI_Comm mpi_comm)
{
  // Count number of locally owned and unowned nodes
  std::int32_t owned_local_size(0), unowned_local_size(0);
  for (std::int8_t node : node_ownership)
  {
    if (node >= 0)
      ++owned_local_size;
    else if (node == -1)
      ++unowned_local_size;
    else
    {
      throw std::runtime_error(
          "Compute node reordering - invalid node ownership index.");
    }
  }
  // assert((unowned_local_size + owned_local_size) == node_ownership.size());
  // assert((unowned_local_size + owned_local_size)
  //        == node_dofmap.global_indices.size());

  // Create global-to-local index map for local un-owned nodes
  std::vector<std::pair<std::size_t, int>> node_pairs;
  node_pairs.reserve(unowned_local_size);
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    if (node_ownership[i] == -1)
      node_pairs.push_back(std::make_pair(node_dofmap.global_indices[i], i));
  }
  std::map<std::size_t, int> global_to_local_nodes_unowned(node_pairs.begin(),
                                                           node_pairs.end());

  // Buffer nodes that are owned and shared with another process
  const std::size_t mpi_size = dolfin::MPI::size(mpi_comm);
  std::vector<std::vector<std::size_t>> send_buffer(mpi_size);
  for (std::size_t old_index = 0; old_index < node_ownership.size();
       ++old_index)
  {
    // If this node is shared and owned, buffer old and new (global)
    // node index for sending
    if (node_ownership[old_index] == 0)
    {
      auto it = node_to_sharing_processes.find(old_index);
      if (it != node_to_sharing_processes.end())
      {
        for (auto p = it->second.begin(); p != it->second.end(); ++p)
        {
          // Buffer old and new global indices to send
          send_buffer[*p].push_back(node_dofmap.global_indices[old_index]);
          send_buffer[*p].push_back(process_offset + old_to_new[old_index]);
        }
      }
    }
  }

  std::vector<std::vector<std::size_t>> recv_buffer(mpi_size);
  dolfin::MPI::all_to_all(mpi_comm, send_buffer, recv_buffer);

  std::vector<std::size_t> local_to_global_unowned(unowned_local_size);
  for (std::size_t src = 0; src < mpi_size; ++src)
  {
    for (auto q = recv_buffer[src].begin(); q != recv_buffer[src].end(); q += 2)
    {
      const std::size_t received_old_index_global = *q;
      const std::size_t received_new_index_global = *(q + 1);
      auto it = global_to_local_nodes_unowned.find(received_old_index_global);
      assert(it != global_to_local_nodes_unowned.end());

      const int received_old_index_local = it->second;
      const int pos = old_to_new[received_old_index_local] - owned_local_size;
      assert(pos >= 0);
      assert(pos < owned_local_size);
      local_to_global_unowned[pos] = received_new_index_global;
    }
  }

  return local_to_global_unowned;
}
//-----------------------------------------------------------------------------
// FIXME: document better
// Return (old-to-new_local, local_to_global_unowned) maps
std::pair<std::vector<int>, std::vector<std::size_t>> compute_node_reordering(
    const std::unordered_map<int, std::vector<int>>& node_to_sharing_processes,
    const DofMapStructure& node_dofmap,
    const std::vector<std::int8_t>& node_ownership, MPI_Comm mpi_comm)
{
  // Count number of locally owned nodes
  std::size_t owned_local_size(0), unowned_local_size(0);
  for (std::int8_t node : node_ownership)
  {
    if (node >= 0)
      ++owned_local_size;
    else if (node == -1)
      ++unowned_local_size;
    else
    {
      throw std::runtime_error(
          "Compute node reordering - invalid node ownership index.");
    }
  }
  assert((unowned_local_size + owned_local_size) == node_ownership.size());
  assert((unowned_local_size + owned_local_size)
         == node_dofmap.global_indices.size());

  // Create global-to-local index map for local un-owned nodes
  std::vector<std::pair<std::size_t, int>> node_pairs;
  node_pairs.reserve(unowned_local_size);
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    if (node_ownership[i] == -1)
      node_pairs.push_back(std::make_pair(node_dofmap.global_indices[i], i));
  }
  std::map<std::size_t, int> global_to_local_nodes_unowned(node_pairs.begin(),
                                                           node_pairs.end());

  // Build graph for re-ordering. Below block is scoped to clear working
  // data structures once graph is constructed.
  dolfin::graph::Graph graph(owned_local_size);

  // Create contiguous local numbering for locally owned dofs
  std::size_t my_counter = 0;
  std::vector<int> old_to_contiguous_node_index(node_ownership.size(), -1);
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    if (node_ownership[i] >= 0)
      old_to_contiguous_node_index[i] = my_counter++;
  }

  // Build local graph, based on old dof map, with contiguous numbering
  for (std::int32_t cell = 0; cell < node_dofmap.num_cells(); ++cell)
  {
    // Cell dofmaps with old local indices
    const PetscInt* nodes = node_dofmap.dofs(cell);
    std::vector<int> local_old;

    // Loop over nodes collecting valid local nodes
    for (std::int32_t i = 0; i < node_dofmap.num_dofs(cell); ++i)
    {
      // Old node index (0)
      const int n0_old = nodes[i];

      // New node index (0)
      assert(n0_old < (int)old_to_contiguous_node_index.size());
      const int n0_local = old_to_contiguous_node_index[n0_old];

      // Add to graph if node n0_local is owned
      if (n0_local != -1)
      {
        assert(n0_local < (int)graph.size());
        local_old.push_back(n0_local);
      }
    }

    for (std::size_t i = 0; i < local_old.size(); ++i)
      for (std::size_t j = 0; j < local_old.size(); ++j)
        if (i != j)
          graph[local_old[i]].insert(local_old[j]);
  }

  // Reorder nodes
  // const std::string ordering_library
  //     = dolfin::parameter::parameters["dof_ordering_library"];
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
    for (std::size_t i = 0; i < node_remap.size(); ++i)
      node_remap[i] = i;
    std::random_shuffle(node_remap.begin(), node_remap.end());
  }
  else
  {
    throw std::runtime_error("Requested library '" + ordering_library
                             + "' is unknown");
  }

  // Compute offset for owned nodes
  const std::size_t process_offset
      = dolfin::MPI::global_offset(mpi_comm, owned_local_size, true);

  // Allocate space
  std::vector<int> old_to_new_local(node_ownership.size(), -1);

  // Renumber owned nodes, and buffer nodes that are owned and shared
  // with another process
  const std::size_t mpi_size = dolfin::MPI::size(mpi_comm);
  std::vector<std::vector<std::size_t>> send_buffer(mpi_size);
  std::vector<std::vector<std::size_t>> recv_buffer(mpi_size);
  std::size_t counter = 0;
  for (std::size_t old_node_index_local = 0;
       old_node_index_local < node_ownership.size(); ++old_node_index_local)
  {
    // Skip nodes that are not owned (will receive global index later)
    if (node_ownership[old_node_index_local] < 0)
      continue;

    // Set new node number
    assert(counter < node_remap.size());
    assert(old_node_index_local < old_to_new_local.size());
    old_to_new_local[old_node_index_local] = node_remap[counter];

    // If this node is shared and owned, buffer old and new (global)
    // node index for sending
    if (node_ownership[old_node_index_local] == 0)
    {
      auto it = node_to_sharing_processes.find(old_node_index_local);
      if (it != node_to_sharing_processes.end())
      {
        for (auto p = it->second.begin(); p != it->second.end(); ++p)
        {
          // Buffer old and new global indices to send
          send_buffer[*p].push_back(
              node_dofmap.global_indices[old_node_index_local]);
          send_buffer[*p].push_back(process_offset + node_remap[counter]);
        }
      }
    }
    ++counter;
  }

  dolfin::MPI::all_to_all(mpi_comm, send_buffer, recv_buffer);

  std::vector<std::size_t> local_to_global_unowned(unowned_local_size);
  std::size_t off_process_node_counter = 0;
  for (std::size_t src = 0; src != mpi_size; ++src)
  {
    for (auto q = recv_buffer[src].begin(); q != recv_buffer[src].end(); q += 2)
    {
      const std::size_t received_old_node_index_global = *q;
      const std::size_t received_new_node_index_global = *(q + 1);
      auto it
          = global_to_local_nodes_unowned.find(received_old_node_index_global);
      assert(it != global_to_local_nodes_unowned.end());

      const int received_old_node_index_local = it->second;
      local_to_global_unowned[off_process_node_counter]
          = received_new_node_index_global;
      // off_process_owner[off_process_node_counter] = src;

      const int new_index_local = owned_local_size + off_process_node_counter;
      assert(old_to_new_local[received_old_node_index_local] < 0);
      old_to_new_local[received_old_node_index_local] = new_index_local;
      off_process_node_counter++;
    }
  }

  // Sanity check
  for (int it : old_to_new_local)
  {
    assert(it != -1);
  }

  return std::make_pair(std::move(old_to_new_local),
                        std::move(local_to_global_unowned));
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
std::tuple<std::size_t, std::unique_ptr<common::IndexMap>,
           std::unordered_map<int, std::vector<int>>, std::set<int>,
           std::vector<PetscInt>>
DofMapBuilder::build(const mesh::Mesh& mesh,
                     const ElementDofLayout& element_dof_layout, int block_size)
{
  common::Timer t0("Init dofmap");

  const int D = mesh.topology().dim();

  // Build a simple dofmap based on mesh entity numbering.  Returns:
  //  - dofmap (local indices)
  //  - local-to-global dof index map)
  DofMapStructure node_graph0 = build_basic_dofmap(mesh, element_dof_layout);

  // Compute global dofmap dimension
  std::size_t global_dimension = 0;
  for (int d = 0; d < D + 1; ++d)
  {
    const std::int64_t n = mesh.num_entities_global(d);
    global_dimension += n * element_dof_layout.num_entity_dofs(d);
  }

  // Re-order and switch to local indexing in dofmap when distributed
  // for process locality and set local_range

  // Mark shared and non-shared nodes. Boundary nodes are assigned a
  // random positive integer, interior nodes are marked as -1, interior
  // nodes in ghost layer of other processes are marked -2, and ghost
  // nodes are marked as -3,
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  const std::vector<std::int8_t> shared_nodes
      = compute_sharing_markers(node_graph0, element_dof_layout, mesh);

  // Compute node ownership:
  // (a) Number of owned nodes;
  // (b) owned and shared nodes (and owned and un-owned):
  //    -1: unowned, 0: owned and shared, 1: owned and not shared;
  // (c) map from shared node to sharing processes; and
  // (d) set of all processes that share dofs with this process
  std::vector<std::int8_t> node_ownership0;
  std::unordered_map<int, std::vector<int>> shared_node_to_processes0;
  int num_owned_nodes;
  std::set<int> neighbouring_procs;
  std::tie(num_owned_nodes, node_ownership0, shared_node_to_processes0,
           neighbouring_procs)
      = compute_node_ownership(node_graph0, shared_nodes, mesh,
                               global_dimension);

  // TEST: local re-ordering
  const std::vector<std::int32_t> old_to_new
      = compute_reordering_map(node_graph0, node_ownership0);

  // Compute process offset for owned nodes
  const std::size_t process_offset
      = dolfin::MPI::global_offset(mesh.mpi_comm(), num_owned_nodes, true);

  // TEST: local-to-global for unowned dofs
  const std::vector<std::size_t> local_to_global_unowned_test
      = compute_global_indices(process_offset, old_to_new,
                               shared_node_to_processes0, node_graph0,
                               node_ownership0, mesh.mpi_comm());

  // Compute node re-ordering for process index locality, and spatial
  // locality within a process, including
  // (a) Old-to-new node indices (local)
  // (b) Owning process for nodes that are not owned by this process
  // (c) New local node index to new global node index
  // (d) Old local node index to new local node index
  std::vector<int> node_old_to_new_local;
  std::vector<std::size_t> local_to_global_unowned;
  std::tie(node_old_to_new_local, local_to_global_unowned)
      = compute_node_reordering(shared_node_to_processes0, node_graph0,
                                node_ownership0, mesh.mpi_comm());

  // Create IndexMap for dofs range on this process
  auto index_map = std::make_unique<common::IndexMap>(
      mesh.mpi_comm(), num_owned_nodes, local_to_global_unowned, block_size);
  assert(index_map);
  assert(dolfin::MPI::sum(mesh.mpi_comm(), (std::size_t)index_map->size_local())
         == global_dimension);

  // Update shared_nodes following the reordering
  std::unordered_map<int, std::vector<int>> shared_nodes_foo;
  for (auto it = shared_node_to_processes0.begin();
       it != shared_node_to_processes0.end(); ++it)
  {
    const int new_node = node_old_to_new_local[it->first];
    shared_nodes_foo[new_node] = it->second;
  }

  // Build dofmap from original node 'dof' map, and applying the
  // 'old_to_new_local' map for the re-ordered node indices
  const std::vector<std::vector<PetscInt>> dofmap_graph
      = build_dofmap(node_graph0, node_old_to_new_local, block_size);

  // Build flattened dofmap graph
  std::vector<PetscInt> cell_dofmap;
  for (auto const& cell_dofs : dofmap_graph)
    cell_dofmap.insert(cell_dofmap.end(), cell_dofs.begin(), cell_dofs.end());

  return std::make_tuple(std::move(block_size * global_dimension),
                         std::move(index_map), std::move(shared_nodes_foo),
                         std::move(neighbouring_procs), std::move(cell_dofmap));
}
//-----------------------------------------------------------------------------
std::tuple<std::int64_t, std::vector<PetscInt>>
DofMapBuilder::build_sub_map_view(const DofMap& parent_dofmap,
                                  const ElementDofLayout& parent_element_dofmap,
                                  const std::vector<std::size_t>& component,
                                  const mesh::Mesh& mesh)
{
  assert(!component.empty());
  const int D = mesh.topology().dim();

  std::vector<int> num_cell_entities(D + 1);
  const mesh::CellType& cell_type = mesh.type();
  for (int d = 0; d <= D; ++d)
    num_cell_entities[d] = cell_type.num_entities(d);

  // Extract mesh entities that require initialisation
  std::vector<bool> needs_entities(D + 1);
  for (int d = 0; d <= D; ++d)
    needs_entities[d] = parent_element_dofmap.num_entity_dofs(d) > 0;

  // Alternative with ElementDofLayout
  std::shared_ptr<const ElementDofLayout> sub_el_dm
      = parent_element_dofmap.sub_dofmap(component);
  const std::vector<int> sub_el_map
      = parent_element_dofmap.sub_dofmap_mapping(component);

  std::vector<std::vector<PetscInt>> sub_dofmap_graph(mesh.num_entities(D));
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    const int c = cell.index();
    // FIXME: count global dofs
    const int num_sub_dofs = sub_el_dm->num_dofs();
    auto dmap_parent = parent_dofmap.cell_dofs(c);
    sub_dofmap_graph[c].resize(num_sub_dofs);
    for (int i = 0; i < num_sub_dofs; ++i)
      sub_dofmap_graph[c][i] = dmap_parent[sub_el_map[i]];
  }

  // Store number of global mesh entities and set global dimension
  std::size_t global_dimension = 0;
  for (int d = 0; d < D + 1; ++d)
  {
    const std::int64_t n = mesh.num_entities_global(d);
    global_dimension += n * sub_el_dm->num_entity_dofs(d);
  }

  // Flatten new dofmap
  std::vector<PetscInt> cell_dofmap;
  for (auto const& cell_dofs : sub_dofmap_graph)
    cell_dofmap.insert(cell_dofmap.end(), cell_dofs.begin(), cell_dofs.end());

  return std::make_tuple(std::move(global_dimension), std::move(cell_dofmap));
}
//-----------------------------------------------------------------------------
