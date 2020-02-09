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
// Sharing marker for a node
enum class sharing_marker : std::int8_t
{
  boundary,
  interior,
  interior_ghost_layer,
  ghost
};

// Ownership marker for a node
enum class ownership : std::int8_t
{
  not_owned,      // Shared but not 'owned'
  owned_shared,   // Owned and shared with other processes
  owned_exclusive // Owned and not shared with other processes
};

//-----------------------------------------------------------------------------
struct DofMapStructure
{
  std::vector<PetscInt> data;
  std::vector<std::int32_t> cell_ptr;
  std::vector<std::int64_t> global_indices;

  // entity dim, local mesh entity index
  std::vector<std::pair<std::int8_t, std::int32_t>> dof_entity;

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
// Compute which process 'owns' each node (point at which dofs live).
// Also computes map from shared node to sharing processes and a set of
// process that share dofs on this process.
std::tuple<std::int32_t, std::vector<ownership>,
           std::unordered_map<std::int32_t, std::vector<std::int32_t>>>
compute_ownership(const DofMapStructure& dofmap,
                  const std::vector<sharing_marker>& shared_nodes,
                  const mesh::Mesh& mesh, const std::int64_t global_dim)
{
  // Get number of nodes
  const std::int32_t num_nodes_local = dofmap.global_indices.size();

  // Global-to-local node map for nodes on boundary
  std::map<std::int64_t, std::int32_t> global_to_local;

  // Communication buffers
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const std::int32_t num_processes = dolfinx::MPI::size(mpi_comm);
  const std::int32_t process_number = dolfinx::MPI::rank(mpi_comm);
  std::vector<std::vector<std::int64_t>> send_buffer(num_processes);

  // Add a counter to the start of each send buffer
  for (std::int32_t i = 0; i < num_processes; ++i)
    send_buffer[i].push_back(0);

  // FIXME: could get rid of global_to_local map since response will
  // come back in same order

  // Loop over nodes and buffer global indices of nodes on process
  // boundaries
  for (std::int32_t i = 0; i < num_nodes_local; ++i)
  {
    if (shared_nodes[i] == sharing_marker::boundary)
    {
      // Shared node - send out to matching process to determine
      // ownership and other sharing processes

      // Send global index
      const std::int64_t global_index = dofmap.global_indices[i];
      const std::int32_t dest
          = dolfinx::MPI::index_owner(mpi_comm, global_index, global_dim);
      send_buffer[dest].push_back(global_index);
      global_to_local.insert(std::pair(global_index, i));
    }
  }

  // Make note of current size of each send buffer i.e. the number of
  // boundary nodes
  for (std::int32_t i = 0; i < num_processes; ++i)
    send_buffer[i][0] = send_buffer[i].size() - 1;

  // Additionally send any ghost or ghost-shared nodes to determine
  // sharing (but not ownership)
  for (std::int32_t i = 0; i < num_nodes_local; ++i)
  {
    if (shared_nodes[i] == sharing_marker::ghost
        or shared_nodes[i] == sharing_marker::interior_ghost_layer)
    {
      // Send global index
      const std::int64_t global_index = dofmap.global_indices[i];
      const std::int32_t dest
          = dolfinx::MPI::index_owner(mpi_comm, global_index, global_dim);
      send_buffer[dest].push_back(global_index);
      global_to_local.insert(std::pair(global_index, i));
    }
  }

  // Send to sorting process
  std::vector<std::vector<std::int64_t>> recv_buffer(num_processes);
  dolfinx::MPI::all_to_all(mpi_comm, send_buffer, recv_buffer);

  // Map from global index to sharing processes
  std::map<std::int64_t, std::vector<std::int32_t>> global_to_procs;
  for (std::int32_t i = 0; i < num_processes; ++i)
  {
    const std::vector<std::int64_t>& recv_i = recv_buffer[i];
    const std::int32_t num_boundary_nodes = recv_i[0];

    for (std::int32_t j = 1; j < num_boundary_nodes + 1; ++j)
    {
      auto map_it = global_to_procs.find(recv_i[j]);
      if (map_it == global_to_procs.end())
        global_to_procs.insert({recv_i[j], std::vector<std::int32_t>(1, i)});
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
  for (std::int32_t i = 0; i < num_processes; ++i)
  {
    const std::vector<std::int64_t>& recv_i = recv_buffer[i];
    const std::int32_t num_boundary_nodes = recv_i[0];

    for (std::size_t j = num_boundary_nodes + 1; j < recv_i.size(); ++j)
    {
      auto map_it = global_to_procs.find(recv_i[j]);
      if (map_it == global_to_procs.end())
        global_to_procs.insert({recv_i[j], std::vector<std::int32_t>(1, i)});
      else
        map_it->second.push_back(i);
    }
  }

  // Send response back to originators in same order
  std::vector<std::vector<std::int64_t>> send_response(num_processes);
  for (std::int32_t i = 0; i < num_processes; ++i)
  {
    for (auto q = recv_buffer[i].begin() + 1; q != recv_buffer[i].end(); ++q)
    {
      std::vector<std::int32_t>& gprocs = global_to_procs[*q];
      send_response[i].push_back(gprocs.size());
      send_response[i].insert(send_response[i].end(), gprocs.begin(),
                              gprocs.end());
    }
  }

  // Initialise node ownership array, provisionally all owned exclusively
  std::vector<ownership> node_ownership(num_nodes_local,
                                        ownership::owned_exclusive);

  dolfinx::MPI::all_to_all(mpi_comm, send_response, recv_buffer);
  // [n_sharing, owner, others]
  std::unordered_map<int, std::vector<int>> shared_node_to_processes;
  for (std::int32_t i = 0; i < num_processes; ++i)
  {
    auto q = recv_buffer[i].begin();
    for (auto p = send_buffer[i].begin() + 1; p != send_buffer[i].end(); ++p)
    {
      const std::int32_t num_sharing = *q;
      if (num_sharing > 1)
      {
        const std::int64_t global_index = *p;
        const std::int32_t owner = *(q + 1);
        std::set<std::int32_t> sharing_procs(q + 1, q + 1 + num_sharing);
        sharing_procs.erase(process_number);

        auto it = global_to_local.find(global_index);
        assert(it != global_to_local.end());
        const std::int32_t received_node_local = it->second;
        const sharing_marker node_status = shared_nodes[received_node_local];
        assert(node_status != sharing_marker::interior);

        // First check to see if this is a ghost/ghost-shared node, and
        // set ownership accordingly. Otherwise use the ownership from
        // the sorting process
        if (node_status == sharing_marker::interior_ghost_layer)
          node_ownership[received_node_local] = ownership::owned_shared;
        else if (node_status == sharing_marker::ghost)
          node_ownership[received_node_local] = ownership::not_owned;
        else if (owner == process_number)
          node_ownership[received_node_local] = ownership::owned_shared;
        else
          node_ownership[received_node_local] = ownership::not_owned;

        shared_node_to_processes[received_node_local]
            = std::vector<int>(sharing_procs.begin(), sharing_procs.end());
      }

      q += num_sharing + 1;
    }
  }

  // Count number of owned nodes
  int num_owned_nodes = 0;
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    if (node_ownership[i] != ownership::not_owned)
      ++num_owned_nodes;
  }

  return std::tuple(std::move(num_owned_nodes), std::move(node_ownership),
                    std::move(shared_node_to_processes));
}
//-----------------------------------------------------------------------------
// Build a simple dofmap from ElementDofmap based on mesh entity indices
DofMapStructure build_basic_dofmap(const mesh::Mesh& mesh,
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
  dofmap.global_indices.resize(local_size);
  dofmap.data.resize(mesh.num_entities(D) * local_dim);
  dofmap.cell_ptr.resize(mesh.num_entities(D) + 1, local_dim);
  dofmap.cell_ptr[0] = 0;
  std::partial_sum(dofmap.cell_ptr.begin() + 1, dofmap.cell_ptr.end(),
                   dofmap.cell_ptr.begin() + 1);

  // Dof (dim, entity index) marker
  dofmap.dof_entity.resize(local_size);

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

  // Build dofmaps from ElementDofmap
  // for (auto& cell : mesh::MeshRange(mesh, D, mesh::MeshRangeType::ALL))
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
          dofmap.global_indices[dof]
              = offset_global + num_entity_dofs * e_index_global + count;
          dofmap.dof_entity[dof] = {d, e_index_local};
        }
      }
      offset_local += entity_dofs[d][0].size() * num_mesh_entities_local[d];
      offset_global += entity_dofs[d][0].size() * num_mesh_entities_global[d];
    }
  }

  return dofmap;
}
//-----------------------------------------------------------------------------
// Compute sharing marker for each node
std::vector<sharing_marker>
compute_sharing_markers(const DofMapStructure& dofmap,
                        const ElementDofLayout& element_dof_layout,
                        const mesh::Mesh& mesh)
{
  // Initialise mesh
  const int D = mesh.topology().dim();

  // Allocate data and initialise all nodes to 'interior'
  // (provisionally, owned and not shared)
  std::vector<sharing_marker> shared_nodes(dofmap.global_indices.size(),
                                           sharing_marker::interior);

  // Get facet closure dofs
  const std::vector<std::set<int>>& facet_table
      = element_dof_layout.entity_closure_dofs_all()[D - 1];

  // Mark dofs associated ghost cells as ghost dofs, provisionally
  bool has_ghost_cells = false;
  const std::int32_t ghost_offset_c
      = mesh.topology().index_map(D)->size_local();
  const std::int32_t ghost_offset_f
      = mesh.topology().index_map(D - 1)->size_local();

  const std::map<std::int32_t, std::set<std::int32_t>>& sharing_map_c
      = mesh.topology().shared_entities(D);
  for (auto& c : mesh::MeshRange(mesh, D, mesh::MeshRangeType::ALL))
  {
    const bool ghost_cell = c.index() >= ghost_offset_c;
    const PetscInt* cell_nodes = dofmap.dofs(c.index());
    if (sharing_map_c.find(c.index()) != sharing_map_c.end())
    {
      // Cell is shared
      const sharing_marker status = ghost_cell
                                        ? sharing_marker::ghost
                                        : sharing_marker::interior_ghost_layer;
      for (std::int32_t i = 0; i < dofmap.num_dofs(c.index()); ++i)
      {
        // Ensure not already set (for R space)
        if (shared_nodes[cell_nodes[i]] == sharing_marker::interior)
          shared_nodes[cell_nodes[i]] = status;
      }
    }

    // Change all non-ghost facet dofs of ghost cells to boundary dofs
    if (ghost_cell)
    {
      // Is a ghost cell
      has_ghost_cells = true;
      for (auto& f : mesh::EntityRange(c, D - 1))
      {
        if (!(f.index() >= ghost_offset_f))
        {
          // Not a ghost facet
          const std::set<int>& facet_nodes = facet_table[c.index(f)];
          for (auto facet_node : facet_nodes)
          {
            const int facet_node_local = cell_nodes[facet_node];
            shared_nodes[facet_node_local] = sharing_marker::boundary;
          }
        }
      }
    }
  }

  if (has_ghost_cells)
    return shared_nodes;

  // Mark nodes on inter-process boundary
  const std::map<std::int32_t, std::set<std::int32_t>>& sharing_map_f
      = mesh.topology().shared_entities(D - 1);
  for (auto& f : mesh::MeshRange(mesh, D - 1, mesh::MeshRangeType::ALL))
  {
    // Skip if facet is not shared
    // NOTE: second test is for periodic problems
    if (sharing_map_f.find(f.index()) == sharing_map_f.end())
      continue;

    // Get cell to which facet belongs (pick first)
    const mesh::MeshEntity cell0(mesh, D, f.entities(D)[0]);

    // Get dofs (process-wise indices) on cell
    const PetscInt* cell_nodes = dofmap.dofs(cell0.index());

    // Get dofs which are on the facet
    const std::set<int>& facet_nodes = facet_table[cell0.index(f)];

    // Mark boundary nodes and insert into map
    for (auto facet_node : facet_nodes)
    {
      // Get facet node local index and assign "boundary"  - shared,
      // owner unassigned
      PetscInt facet_node_local = cell_nodes[facet_node];
      shared_nodes[facet_node_local] = sharing_marker::boundary;
    }
  }

  return shared_nodes;
}
//-----------------------------------------------------------------------------
// Compute re-ordering map of indices.
std::vector<std::int32_t>
compute_reordering_map(const DofMapStructure& dofmap,
                       const std::vector<ownership>& node_ownership)
{
  // Create map from old index to new contiguous numbering for locally
  // owned dofs. Set to -1 for unowned dofs.
  std::int32_t owned_size = 0;
  std::vector<int> original_to_contiguous(node_ownership.size(), -1);
  for (std::size_t i = 0; i < original_to_contiguous.size(); ++i)
  {
    if (node_ownership[i] != ownership::not_owned)
      original_to_contiguous[i] = owned_size++;
  }

  // Build local graph, based on dof map with contiguous numbering
  // (unowned dofs excluded)
  dolfinx::graph::Graph graph(owned_size);
  std::vector<int> local_old;
  for (std::int32_t cell = 0; cell < dofmap.num_cells(); ++cell)
  {
    // Loop over nodes collecting valid local nodes
    local_old.clear();
    const PetscInt* nodes = dofmap.dofs(cell);
    for (std::int32_t i = 0; i < dofmap.num_dofs(cell); ++i)
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

  // Reconstruct remaped nodes, with -1 for unowned
  std::vector<int> old_to_new(node_ownership.size(), -1);
  std::int32_t unowned_pos = owned_size;
  for (std::size_t old_index = 0; old_index < node_ownership.size();
       ++old_index)
  {
    const std::int32_t index = original_to_contiguous[old_index];

    // Put nodes that are not owned at the end, otherwise re-number
    if (index < 0)
    {
      assert(old_index < old_to_new.size());
      old_to_new[old_index] = unowned_pos++;
    }
    else
    {
      assert(old_index < old_to_new.size());
      old_to_new[old_index] = node_remap[index];
    }
  }

  return old_to_new;
}
//-----------------------------------------------------------------------------
// Compute global indices for unowned dofs
Eigen::Array<std::int64_t, Eigen::Dynamic, 1> compute_global_indices(
    const std::int64_t process_offset,
    const std::vector<std::int32_t>& old_to_new,
    const std::unordered_map<int, std::vector<int>>& node_to_sharing_processes,
    const DofMapStructure& dofmap, const std::vector<ownership>& node_ownership,
    MPI_Comm mpi_comm)
{
  // Count number of locally owned and unowned nodes
  std::int32_t owned_local_size(0), unowned_local_size(0);
  for (auto node : node_ownership)
  {
    switch (node)
    {
    case ownership::not_owned:
      ++unowned_local_size;
      break;
    default:
      ++owned_local_size;
    }
  }
  assert((unowned_local_size + owned_local_size)
         == (std::int32_t)dofmap.global_indices.size());

  // Create global-to-local index map for local un-owned nodes
  std::vector<std::pair<std::int64_t, int>> node_pairs;
  node_pairs.reserve(unowned_local_size);
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    // if (node_ownership[i] == -1)
    if (node_ownership[i] == ownership::not_owned)
      node_pairs.push_back(std::pair(dofmap.global_indices[i], i));
  }
  std::map<std::int64_t, int> global_to_local_nodes_unowned(node_pairs.begin(),
                                                            node_pairs.end());

  // Buffer nodes that are owned and shared with another process
  const std::int32_t mpi_size = dolfinx::MPI::size(mpi_comm);
  std::vector<std::vector<std::int64_t>> send_buffer(mpi_size);
  for (std::size_t old_index = 0; old_index < node_ownership.size();
       ++old_index)
  {
    // If this node is shared and owned, buffer old and new (global)
    // node index for sending
    // if (node_ownership[old_index] == 0)
    if (node_ownership[old_index] == ownership::owned_shared)
    {
      auto it = node_to_sharing_processes.find(old_index);
      if (it != node_to_sharing_processes.end())
      {
        for (auto p = it->second.begin(); p != it->second.end(); ++p)
        {
          // Buffer old and new global indices to send
          send_buffer[*p].push_back(dofmap.global_indices[old_index]);
          send_buffer[*p].push_back(process_offset + old_to_new[old_index]);
        }
      }
    }
  }

  std::vector<std::vector<std::int64_t>> recv_buffer(mpi_size);
  dolfinx::MPI::all_to_all(mpi_comm, send_buffer, recv_buffer);

  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> local_to_global_unowned(
      unowned_local_size);
  for (std::int32_t src = 0; src < mpi_size; ++src)
  {
    for (auto q = recv_buffer[src].begin(); q != recv_buffer[src].end(); q += 2)
    {
      const std::int64_t received_old_index_global = *q;
      const std::int64_t received_new_index_global = *(q + 1);
      auto it = global_to_local_nodes_unowned.find(received_old_index_global);
      assert(it != global_to_local_nodes_unowned.end());

      const int received_old_index_local = it->second;
      const int pos = old_to_new[received_old_index_local] - owned_local_size;
      assert(pos >= 0);
      assert(pos < unowned_local_size);
      local_to_global_unowned[pos] = received_new_index_global;
    }
  }

  return local_to_global_unowned;
}
//-----------------------------------------------------------------------------
// Compute re-ordering map of indices
std::pair<std::vector<std::int32_t>, std::int32_t>
compute_reordering_map_new(const DofMapStructure& dofmap,
                           const mesh::Topology& topology)
{
  // Get ownership offset for each dimension
  const int D = topology.dim();
  std::vector<std::int32_t> offset(D + 1, -1);
  for (std::size_t d = 0; d < offset.size(); ++d)
  {
    auto map = topology.index_map(d);
    if (map)
    {
      // std::cout << "Getting offset for dim: " << d << ", " <<
      // map->size_local()
      //           << std::endl;
      offset[d] = map->size_local();
    }
  }

  // Count locally owned dofs
  std::vector<bool> owned(dofmap.dof_entity.size(), false);
  for (auto e = dofmap.dof_entity.begin(); e != dofmap.dof_entity.end(); ++e)
  {
    if (e->second < offset[e->first])
    {
      const std::size_t i = std::distance(dofmap.dof_entity.begin(), e);
      owned[i] = true;
    }
  }

  // Create map from old index to new contiguous numbering for locally
  // owned dofs. Set to -1 for unowned dofs.
  std::vector<int> original_to_contiguous(dofmap.dof_entity.size(), -1);
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
  for (std::int32_t cell = 0; cell < dofmap.num_cells(); ++cell)
  {
    // Loop over nodes collecting valid local nodes
    local_old.clear();
    const std::int32_t* nodes = dofmap.dofs(cell);
    for (std::int32_t i = 0; i < dofmap.num_dofs(cell); ++i)
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
  std::vector<int> old_to_new(dofmap.dof_entity.size(), -1);
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
// Get global indices for unowned dofs
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
      // std::cout << "NNNNNNNNNNN" << std::endl;
      // std::map<std::int64_t, std::int64_t> global_old_new;
      for (std::size_t i = 0; i < dofs_received.size(); i += 2)
      {
        // std::cout << "Global-Global map" << std::endl;
        global_old_new.insert({dofs_received[i], dofs_received[i + 1]});
      }
    }
  }

  // std::vector<std::pair<std::int64_t, std::int32_t>> global_to_local;
  std::vector<std::int64_t> local_to_global_new(old_to_new.size() - num_owned);
  for (std::size_t i = 0; i < local_new_to_global_old.size(); ++i)
  {
    auto it = global_old_new.find(local_new_to_global_old[i]);
    assert(it != global_old_new.end());
    local_to_global_new[i] = it->second;
    // std::cout << "Local, global new: " << i << ", " << local_to_global_new[i]
    //           << std::endl;
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
  const DofMapStructure node_graph0
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

  // NEW (no communication)
  // Build re-ordering map for data locality. Owned dofs are re-ordred
  // via an ordering algorithm and placed at start, [0, ...,
  // num_owned_nodes -1]. Unowned dofs are placed at end of the
  // re-ordered list. [num_owned_nodes, ..., num_nodes -1].
  // std::cout << "Build re-ordering map" << std::endl;
  const auto [old_to_new, num_owned]
      = compute_reordering_map_new(node_graph0, mesh.topology());

  // NEW
  // Compute process offset for owned nodes
  const std::int64_t process_offset
      = dolfinx::MPI::global_offset(mesh.mpi_comm(), num_owned, true);
  // std::cout << "Offset: " << process_offset << std::endl;

  // NEW (TODO)
  // Get global indices for unowned dofs
  // std::cout << "Get ghost indices" << std::endl;
  const std::vector<std::int64_t> local_to_global_unowned =
  get_global_indices(
      mesh.topology(), num_owned, process_offset, node_graph0.global_indices,
      old_to_new, node_graph0.dof_entity);

  // const int rank = MPI::rank(MPI_COMM_WORLD);
  // // std::cout << "Owned: " << rank << ", " << num_owned << std::endl;
  // // std::cout << "Offset: " << rank << ", " << num_owned << std::endl;
  // if (rank == 2)
  // {
  //   std::cout << "Owned: " << rank << ", " << num_owned << std::endl;
  //   std::cout << "Offset: " << rank << ", " << process_offset <<
  // std::endl;
  //   std::cout << "Num ghosts: " << rank << ", "
  //             << local_to_global_unowned.size() << std::endl;
  //   for (std::size_t i = 0; i < local_to_global_unowned.size(); ++i)
  //   {
  //     std::cout << "  Ghosts: " << rank << ", " <<
  // local_to_global_unowned[i]
  //               << std::endl;
  //   }
  // }

  // OLD

  // Mark shared and non-shared nodes. Boundary nodes are assigned a
  // random positive integer, interior nodes are marked as -1, interior
  // nodes in ghost layer of other processes are marked -2, and ghost
  // nodes are marked as -3,
  // mesh.create_entities(D - 1);
  // mesh.create_connectivity(D - 1, D);
  // const std::vector<sharing_marker> shared_nodes
  //     = compute_sharing_markers(node_graph0, element_dof_layout, mesh);

  // // Compute node ownership:
  // // (a) Number of owned nodes;
  // // (b) owned and shared nodes (and owned and un-owned):
  // //    -1: unowned, 0: owned and shared, 1: owned and not shared; and
  // // (c) map from shared node to sharing processes.
  // const auto [num_owned, node_ownership0, shared_node_to_processes0]
  //     = compute_ownership(node_graph0, shared_nodes, mesh, global_dimension);

  // // Build re-ordering map for data locality. Owned dofs are re-ordred
  // // via an ordering algorithm and placed at start, [0, ...,
  // // num_owned_nodes -1]. Unowned dofs are placed at end of the
  // // re-ordered list. [num_owned_nodes, ..., num_nodes -1].
  // const std::vector<std::int32_t> old_to_new
  //     = compute_reordering_map(node_graph0, node_ownership0);

  // // Compute process offset for owned nodes. Global indices for owned
  // // dofs are (index_local + process_offset)
  // const std::int64_t process_offset
  //     = dolfinx::MPI::global_offset(mesh.mpi_comm(), num_owned, true);

  // // Get global indices for unowned unowned dofs
  // const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> local_to_global_unowned
  //     = compute_global_indices(process_offset, old_to_new,
  //                              shared_node_to_processes0, node_graph0,
  //                              node_ownership0, mesh.mpi_comm());

  // std::cout << "Create index map" << std::endl;

  // Create IndexMap for dofs range on this process
  auto index_map = std::make_unique<common::IndexMap>(
      mesh.mpi_comm(), num_owned, local_to_global_unowned, block_size);
  assert(index_map);
  assert(
      dolfinx::MPI::sum(mesh.mpi_comm(), (std::int64_t)index_map->size_local())
      == global_dimension);

  // std::cout << "Build dof map" << std::endl;

  // FIXME: There is an assumption here on the dof order for an element.
  //        It should come from the ElementDofLayout.
  // Build re-ordered dofmap, accounting for block size
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofmap(node_graph0.data.size()
                                                   * block_size);
  for (std::int32_t cell = 0; cell < node_graph0.num_cells(); ++cell)
  {
    const std::int32_t local_dim0 = node_graph0.num_dofs(cell);
    for (std::int32_t j = 0; j < local_dim0; ++j)
    {
      const std::int32_t old_node = node_graph0.dof(cell, j);
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
