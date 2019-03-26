// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMapBuilder.h"
#include "DofMap.h"
#include "ElementDofMap.h"
#include <cstdlib>
#include <dolfin/common/IndexMap.h>
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
#include <random>
// #include <spdlog/spdlog.h>
#include <stdlib.h>
#include <ufc.h>
#include <utility>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
std::tuple<std::size_t, std::unique_ptr<common::IndexMap>,
           std::unordered_map<int, std::vector<int>>, std::set<int>,
           std::vector<PetscInt>>
DofMapBuilder::build(const ufc_dofmap& ufc_map, const mesh::Mesh& mesh)
{
  common::Timer t0("Init dofmap");

  // Create element dofmap from ufc_dofmap and cell type
  ElementDofMap el_dm(ufc_map, mesh.type());

  // Extract needs_entities as vector of bool
  const std::size_t D = mesh.topology().dim();
  std::vector<bool> needs_entities(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
    needs_entities[d] = el_dm.num_entity_dofs(d) > 0;

  // For mesh entities required by UFC dofmap, compute number of
  // mesh entities on this process
  const std::vector<int64_t> num_mesh_entities_local
      = compute_num_mesh_entities_local(mesh, needs_entities);

  // Compute a 'node' dofmap based on a UFC dofmap (node is a point with a fixed
  // number of dofs). Returns:
  //  - node dofmap (node_dofmap)
  //  - local-to-global node indices (node_local_to_global)
  std::vector<std::size_t> node_local_to_global0;
  std::vector<std::vector<PetscInt>> node_graph0;
  ufc_dofmap* ufc_node_dofmap;

  const std::size_t bs = el_dm.block_size();

  std::tie(ufc_node_dofmap, node_graph0, node_local_to_global0)
      = build_ufc_node_graph(ufc_map, el_dm, mesh, bs);
  assert(ufc_node_dofmap);

  // Set global dofmap dimension
  std::size_t global_dimension = 0;
  for (std::size_t d = 0; d < D + 1; ++d)
  {
    const std::int64_t n = mesh.num_entities_global(d);
    global_dimension += n * el_dm.num_entity_dofs(d);
  }
  assert(global_dimension % bs == 0);

  // Re-order and switch to local indexing in dofmap when distributed
  // for process locality and set local_range

  // Mark shared nodes. Boundary nodes are assigned a random
  // positive integer, interior nodes are marked as -1, interior
  // nodes in ghost layer of other processes are marked -2, and
  // ghost nodes are marked as -3
  std::vector<int> shared_nodes = compute_shared_nodes(
      node_graph0, node_local_to_global0.size(), *ufc_node_dofmap, mesh);

  free(ufc_node_dofmap);

  // Compute:
  // (a) owned and shared nodes (and owned and un-owned):
  //    -1: unowned, 0: owned and shared, 1: owned and not shared;
  // (b) map from shared node to sharing processes; and
  // (c) set of all processes that share dofs with this process
  std::vector<short int> node_ownership0;
  std::unordered_map<int, std::vector<int>> shared_node_to_processes0;
  int num_owned_nodes;
  std::set<int> neighbours;
  std::tie(num_owned_nodes, node_ownership0, shared_node_to_processes0,
           neighbours)
      = compute_node_ownership(node_graph0, shared_nodes, node_local_to_global0,
                               mesh, global_dimension / bs);

  // Compute node re-ordering for process index locality, and spatial
  // locality within a process, including
  // (a) Old-to-new node indices (local)
  // (b) Owning process for nodes that are not owned by this process
  // (c) New local node index to new global node index
  // (d) Old local node index to new local node index
  std::vector<int> node_old_to_new_local;
  std::vector<std::size_t> local_to_global_unowned;
  std::tie(node_old_to_new_local, local_to_global_unowned)
      = compute_node_reordering(shared_node_to_processes0,
                                node_local_to_global0, node_graph0,
                                node_ownership0, mesh.mpi_comm());

  auto index_map = std::make_unique<common::IndexMap>(
      mesh.mpi_comm(), num_owned_nodes, local_to_global_unowned, bs);
  assert(index_map);
  assert(MPI::sum(mesh.mpi_comm(), bs * index_map->size_local())
         == global_dimension);

  // Update shared_nodes for node reordering
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
      = build_dofmap(node_graph0, node_old_to_new_local, bs);

  // Build flattened dofmap graph
  std::vector<PetscInt> cell_dofmap;
  for (auto const& cell_dofs : dofmap_graph)
    cell_dofmap.insert(cell_dofmap.end(), cell_dofs.begin(), cell_dofs.end());

  return std::make_tuple(std::move(global_dimension), std::move(index_map),
                         std::move(shared_nodes_foo), std::move(neighbours),
                         std::move(cell_dofmap));
}
//-----------------------------------------------------------------------------
std::tuple<ufc_dofmap*, std::int64_t, std::int64_t, std::vector<PetscInt>>
DofMapBuilder::build_sub_map_view(const DofMap& parent_dofmap,
                                  const ufc_dofmap& parent_ufc_dofmap,
                                  const int parent_block_size,
                                  const std::int64_t parent_cell_offset,
                                  const std::vector<std::size_t>& component,
                                  const mesh::Mesh& mesh)
{
  assert(!component.empty());
  const std::size_t D = mesh.topology().dim();

  std::vector<int> num_cell_entities(D + 1);
  const mesh::CellType& cell_type = mesh.type();
  for (std::size_t d = 0; d <= D; ++d)
    num_cell_entities[d] = cell_type.num_entities(d);

  // Extract mesh entities that require initialisation
  std::vector<bool> needs_entities(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
    needs_entities[d] = parent_ufc_dofmap.num_entity_dofs[d] > 0;

  // Extract local UFC sub-dofmap from parent and update offset
  ufc_dofmap* ufc_sub_dofmap = nullptr;
  std::int64_t ufc_cell_offset;
  std::tie(ufc_sub_dofmap, ufc_cell_offset) = extract_ufc_sub_dofmap_new(
      parent_ufc_dofmap, component, num_cell_entities, parent_cell_offset);
  assert(ufc_sub_dofmap);

  std::vector<std::vector<PetscInt>> sub_dofmap_graph(mesh.num_entities(D));
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    const int c = cell.index();
    // FIXME: count global dofs
    const int num_sub_dofs = ufc_sub_dofmap->num_element_support_dofs;
    auto dmap_parent = parent_dofmap.cell_dofs(c);
    sub_dofmap_graph[c].resize(num_sub_dofs);
    for (int i = 0; i < num_sub_dofs; ++i)
      sub_dofmap_graph[c][i] = dmap_parent[ufc_cell_offset + i];
  }

  // Store number of global mesh entities and set global dimension
  std::size_t global_dimension = 0;
  for (std::size_t d = 0; d < D + 1; ++d)
  {
    const std::int64_t n = mesh.num_entities_global(d);
    global_dimension += n * ufc_sub_dofmap->num_entity_dofs[d];
  }

  // Flatten new dofmap
  std::vector<PetscInt> cell_dofmap;
  for (auto const& cell_dofs : sub_dofmap_graph)
    cell_dofmap.insert(cell_dofmap.end(), cell_dofs.begin(), cell_dofs.end());

  return std::make_tuple(ufc_sub_dofmap, std::move(ufc_cell_offset),
                         std::move(global_dimension), std::move(cell_dofmap));
}
//-----------------------------------------------------------------------------
std::vector<std::vector<PetscInt>>
DofMapBuilder::build_local_ufc_dofmap(const ufc_dofmap& ufc_dofmap,
                                      const mesh::Mesh& mesh)
{
  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Extract needs_entities as vector
  std::vector<bool> needs_entities(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
    needs_entities[d] = (ufc_dofmap.num_entity_dofs[d] > 0);

  // Generate and number required mesh entities (locally)
  std::vector<int64_t> num_mesh_entities(D + 1, 0);
  for (std::size_t d = 0; d <= D; ++d)
  {
    if (needs_entities[d])
    {
      mesh.init(d);
      num_mesh_entities[d] = mesh.num_entities(d);
    }
  }

  // Allocate entity indices array
  std::vector<std::vector<int64_t>> entity_indices(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
    entity_indices[d].resize(mesh.type().num_entities(d));

  // Build dofmap from ufc_dofmap
  const int num_element_dofs = ufc_dofmap.num_element_support_dofs
                               + ufc_dofmap.num_global_support_dofs;
  std::vector<std::vector<PetscInt>> dofmap(
      mesh.num_entities(D), std::vector<PetscInt>(num_element_dofs));
  std::vector<int64_t> dof_holder(num_element_dofs);
  std::vector<const int64_t*> _entity_indices(entity_indices.size());

  const mesh::CellType& cell_type = mesh.type();
  std::vector<std::vector<std::vector<int>>> entity_dofs(D + 1);
  for (std::size_t d = 0; d < entity_dofs.size(); ++d)
  {
    entity_dofs[d].resize(cell_type.num_entities(d));
    for (std::size_t i = 0; i < entity_dofs[d].size(); ++i)
    {
      entity_dofs[d][i].resize(ufc_dofmap.num_entity_dofs[d]);
      ufc_dofmap.tabulate_entity_dofs(entity_dofs[d][i].data(), d, i);
    }
  }

  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    // Fill entity indices array
    get_cell_entities_local(entity_indices, cell, needs_entities);

    // Tabulate dofs for cell
    for (std::size_t i = 0; i < entity_indices.size(); ++i)
      _entity_indices[i] = entity_indices[i].data();
    GenericDofMap::ufc_tabulate_dofs(dof_holder.data(), entity_dofs,
                                     num_mesh_entities.data(),
                                     _entity_indices.data());
    std::copy(dof_holder.begin(), dof_holder.end(),
              dofmap[cell.index()].begin());
  }

  return dofmap;
}
//-----------------------------------------------------------------------------
std::tuple<int, std::vector<short int>,
           std::unordered_map<int, std::vector<int>>, std::set<int>>
DofMapBuilder::compute_node_ownership(
    const std::vector<std::vector<PetscInt>>& dofmap,
    const std::vector<int>& shared_nodes,
    const std::vector<std::size_t>& local_to_global, const mesh::Mesh& mesh,
    const std::size_t global_dim)
{
  // spdlog::debug("Determining node ownership for parallel dof map");

  // Get number of nodes
  const std::size_t num_nodes_local = local_to_global.size();

  // Global-to-local node map for nodes on boundary
  std::map<std::size_t, int> global_to_local;

  // Initialise node ownership array, provisionally all owned
  std::vector<short int> node_ownership(num_nodes_local, 1);

  // Communication buffers
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const std::size_t num_processes = MPI::size(mpi_comm);
  const std::size_t process_number = MPI::rank(mpi_comm);
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
      const std::size_t global_index = local_to_global[i];
      const std::size_t dest
          = MPI::index_owner(mpi_comm, global_index, global_dim);
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
      const std::size_t global_index = local_to_global[i];
      const std::size_t dest
          = MPI::index_owner(mpi_comm, global_index, global_dim);
      send_buffer[dest].push_back(global_index);
      global_to_local.insert(std::make_pair(global_index, i));
    }
  }

  // Send to sorting process
  MPI::all_to_all(mpi_comm, send_buffer, recv_buffer);

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

  MPI::all_to_all(mpi_comm, send_response, recv_buffer);
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
  std::set<int> neighbours;
  for (auto it = shared_node_to_processes.begin();
       it != shared_node_to_processes.end(); ++it)
  {
    neighbours.insert(it->second.begin(), it->second.end());
  }

  // Count number of owned nodes
  int num_owned_nodes = 0;
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    if (node_ownership[i] >= 0)
      ++num_owned_nodes;
  }

  // spdlog::debug("Finished determining dof ownership for parallel dof map");

  return std::make_tuple(std::move(num_owned_nodes), std::move(node_ownership),
                         std::move(shared_node_to_processes),
                         std::move(neighbours));
}
//-----------------------------------------------------------------------------
std::pair<ufc_dofmap*, int> DofMapBuilder::extract_ufc_sub_dofmap_new(
    const ufc_dofmap& ufc_map, const std::vector<std::size_t>& component,
    const std::vector<int>& num_cell_entities, int offset)
{
  // Check that a sub system has been specified
  if (component.empty())
  {
    throw std::runtime_error("Extracting subsystem of degree of freedom "
                             "mapping - no system was specified");
  }

  // Check if there are any sub systems, and for available sub systems
  const std::size_t num_sub_dofmaps = ufc_map.num_sub_dofmaps;
  if (num_sub_dofmaps == 0)
  {
    throw std::runtime_error("Extracting subsystem of degree of freedom "
                             "mapping - there are no subsystems");
  }
  else if (component[0] >= num_sub_dofmaps)
  {
    throw std::runtime_error("Requested subsystem ("
                             + std::to_string(component[0])
                             + ") out of range [0, "
                             + std::to_string(ufc_map.num_sub_dofmaps) + ")");
  }

  // Add to offset if necessary
  for (std::size_t i = 0; i < component[0]; i++)
  {
    // Extract sub dofmap
    ufc_dofmap* ufc_tmp_dofmap = ufc_map.create_sub_dofmap(i);
    assert(ufc_tmp_dofmap);

    // Get offset
    unsigned int d = 0;
    for (auto& n : num_cell_entities)
    {
      offset += n * ufc_tmp_dofmap->num_entity_dofs[d];
      ++d;
    }

    free(ufc_tmp_dofmap);
  }

  // Create UFC sub-system
  ufc_dofmap* sub_dofmap = ufc_map.create_sub_dofmap(component[0]);
  assert(sub_dofmap);

  // Return sub-system if sub-sub-system should not be extracted,
  // otherwise recursively extract the sub sub system
  if (component.size() == 1)
    return std::make_pair(sub_dofmap, std::move(offset));
  else
  {
    std::vector<std::size_t> sub_component;
    for (std::size_t i = 1; i < component.size(); ++i)
      sub_component.push_back(component[i]);

    std::pair<ufc_dofmap*, int> p = extract_ufc_sub_dofmap_new(
        *sub_dofmap, sub_component, num_cell_entities, offset);
    free(sub_dofmap);
    return p;
  }
}
//-----------------------------------------------------------------------------
std::tuple<ufc_dofmap*, std::vector<std::vector<PetscInt>>,
           std::vector<std::size_t>>
DofMapBuilder::build_ufc_node_graph(const ufc_dofmap& ufc_map,
                                    const ElementDofMap& el_dm,
                                    const mesh::Mesh& mesh,
                                    const std::size_t block_size)
{
  // Start timer for dofmap initialization
  common::Timer t0("Init dofmap from UFC dofmap");

  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Extract needs_entities as vector
  std::vector<bool> needs_entities(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
    needs_entities[d] = (el_dm.num_entity_dofs(d) > 0);

  // Generate and number required mesh entities (local & global, and
  // constrained global)
  std::vector<int64_t> num_mesh_entities_local(D + 1, 0);
  std::vector<int64_t> num_mesh_entities_global(D + 1, 0);
  for (std::size_t d = 0; d <= D; ++d)
  {
    if (needs_entities[d])
    {
      mesh.init(d);
      mesh::DistributedMeshTools::number_entities(mesh, d);
      num_mesh_entities_local[d] = mesh.num_entities(d);
      num_mesh_entities_global[d] = mesh.num_entities_global(d);
    }
  }

  // remove
  ufc_dofmap* dofmap = nullptr;
  if (block_size > 1)
    dofmap = ufc_map.create_sub_dofmap(0);
  else
    dofmap = ufc_map.create();
  // remove

  const ElementDofMap& el_dm2 = (block_size > 1) ? el_dm.sub_dofmap(0) : el_dm;

  unsigned int d = 0;
  std::size_t local_size = 0;
  for (auto& n : num_mesh_entities_local)
  {
    local_size += n * el_dm2.num_entity_dofs(d);
    ++d;
  }

  // Allocate space for dof map
  std::vector<std::vector<PetscInt>> node_dofmap(mesh.num_entities(D));

  const std::size_t local_dim = el_dm2.num_dofs();

  // Holder for UFC 64-bit dofmap integers
  std::vector<int64_t> ufc_nodes_global(local_dim);
  std::vector<int64_t> ufc_nodes_local(local_dim);

  // Allocate entity indices array
  std::vector<std::vector<int64_t>> entity_indices(D + 1);
  std::vector<const int64_t*> entity_indices_ptr(entity_indices.size());
  for (std::size_t d = 0; d <= D; ++d)
  {
    entity_indices[d].resize(mesh.type().num_entities(d));
    entity_indices_ptr[d] = entity_indices[d].data();
  }

  // Resize local-to-global map
  std::vector<std::size_t> node_local_to_global(local_size);

  // Vector for dof permutation
  //  std::vector<int> permutation(local_dim);

  // Build dofmaps from ufc_dofmap
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    // Get reference to container for cell dofs
    std::vector<PetscInt>& cell_nodes = node_dofmap[cell.index()];
    cell_nodes.resize(local_dim);

    // Tabulate standard UFC dof map for first space (local)
    get_cell_entities_local(entity_indices, cell, needs_entities);
    GenericDofMap::ufc_tabulate_dofs(
        ufc_nodes_local.data(), el_dm2.entity_dofs(),
        num_mesh_entities_local.data(), entity_indices_ptr.data());

    // Tabulate standard UFC dof map for first space (global)
    get_cell_entities_global(entity_indices, cell, needs_entities);
    GenericDofMap::ufc_tabulate_dofs(
        ufc_nodes_global.data(), el_dm2.entity_dofs(),
        num_mesh_entities_global.data(), entity_indices_ptr.data());

    // Get the edge and facet permutations of the dofs for this cell,
    // based on global vertex indices.
    //    dofmap->tabulate_dof_permutations(permutation.data(),
    //                                      entity_indices_ptr[0]);

    // Copy to cell dofs, with permutation
    for (unsigned int i = 0; i < local_dim; ++i)
      cell_nodes[i] = ufc_nodes_local[i];

    // Build local-to-global map for nodes
    for (std::size_t i = 0; i < local_dim; ++i)
    {
      assert(ufc_nodes_local[i] < (int)node_local_to_global.size());
      node_local_to_global[ufc_nodes_local[i]] = ufc_nodes_global[i];
    }
  }

  return std::make_tuple(dofmap, std::move(node_dofmap),
                         std::move(node_local_to_global));
}
//-----------------------------------------------------------------------------
std::vector<int> DofMapBuilder::compute_shared_nodes(
    const std::vector<std::vector<PetscInt>>& node_dofmap,
    const std::size_t num_nodes_local, const ufc_dofmap& ufc_dofmap,
    const mesh::Mesh& mesh)
{
  // Initialise mesh
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);

  // Allocate data and initialise all facets to -1 (provisionally,
  // owned and not shared)
  std::vector<int> shared_nodes(num_nodes_local, -1);

  std::vector<int> facet_nodes(ufc_dofmap.num_entity_closure_dofs[D - 1]);

  // Mark dofs associated ghost cells as ghost dofs (provisionally)
  bool has_ghost_cells = false;
  for (auto& c : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    const std::vector<PetscInt>& cell_nodes = node_dofmap[c.index()];
    if (c.is_shared())
    {
      const int status = (c.is_ghost()) ? -3 : -2;
      for (std::size_t i = 0; i < cell_nodes.size(); ++i)
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
          ufc_dofmap.tabulate_entity_closure_dofs(facet_nodes.data(), D - 1,
                                                  c.index(f));
          for (std::size_t i = 0; i < facet_nodes.size(); ++i)
          {
            std::size_t facet_node_local = cell_nodes[facet_nodes[i]];
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

    // Tabulate dofs (local) on cell
    const std::vector<PetscInt>& cell_nodes = node_dofmap[cell0.index()];

    // Tabulate which dofs are on the facet
    ufc_dofmap.tabulate_entity_closure_dofs(facet_nodes.data(), D - 1,
                                            cell0.index(f));

    // Mark boundary nodes and insert into map
    for (std::size_t i = 0; i < facet_nodes.size(); ++i)
    {
      // Get facet node local index and assign "0" - shared, owner
      // unassigned
      size_t facet_node_local = cell_nodes[facet_nodes[i]];
      if (shared_nodes[facet_node_local] < 0)
        shared_nodes[facet_node_local] = 0;
    }
  }

  return shared_nodes;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<std::size_t>>
DofMapBuilder::compute_node_reordering(
    const std::unordered_map<int, std::vector<int>>& node_to_sharing_processes,
    const std::vector<std::size_t>& old_local_to_global,
    const std::vector<std::vector<PetscInt>>& node_dofmap,
    const std::vector<short int>& node_ownership, MPI_Comm mpi_comm)
{
  // Count number of locally owned nodes
  std::size_t owned_local_size = 0;
  std::size_t unowned_local_size = 0;
  for (auto node = node_ownership.begin(); node != node_ownership.end(); ++node)
  {
    if (*node >= 0)
      ++owned_local_size;
    else if (*node == -1)
      ++unowned_local_size;
    else
    {
      throw std::runtime_error(
          "Compute node reordering - invalid node ownership index.");
    }
  }
  assert((unowned_local_size + owned_local_size) == node_ownership.size());
  assert((unowned_local_size + owned_local_size) == old_local_to_global.size());

  // Create global-to-local index map for local un-owned nodes
  std::vector<std::pair<std::size_t, int>> node_pairs;
  node_pairs.reserve(unowned_local_size);
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    if (node_ownership[i] == -1)
      node_pairs.push_back(std::make_pair(old_local_to_global[i], i));
  }
  std::map<std::size_t, int> global_to_local_nodes_unowned(node_pairs.begin(),
                                                           node_pairs.end());
  std::vector<std::pair<std::size_t, int>>().swap(node_pairs);

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

  // Build local graph, based on old dof map, with contiguous
  // numbering
  for (std::size_t cell = 0; cell < node_dofmap.size(); ++cell)
  {
    // Cell dofmaps with old local indices
    const std::vector<PetscInt>& nodes = node_dofmap[cell];
    std::vector<int> local_old;

    // Loop over nodes collecting valid local nodes
    for (std::size_t i = 0; i < nodes.size(); ++i)
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
      = MPI::global_offset(mpi_comm, owned_local_size, true);

  // Allocate space
  std::vector<int> old_to_new_local(node_ownership.size(), -1);

  // Renumber owned nodes, and buffer nodes that are owned but shared
  // with another process
  const std::size_t mpi_size = MPI::size(mpi_comm);
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
          send_buffer[*p].push_back(old_local_to_global[old_node_index_local]);
          send_buffer[*p].push_back(process_offset + node_remap[counter]);
        }
      }
    }
    ++counter;
  }

  MPI::all_to_all(mpi_comm, send_buffer, recv_buffer);

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
  for (auto it : old_to_new_local)
  {
    assert(it != -1);
  }

  return std::make_pair(std::move(old_to_new_local),
                        std::move(local_to_global_unowned));
}
//-----------------------------------------------------------------------------
std::vector<std::vector<PetscInt>> DofMapBuilder::build_dofmap(
    const std::vector<std::vector<PetscInt>>& node_dofmap,
    const std::vector<int>& old_to_new_node_local, const std::size_t block_size)
{
  // Build dofmap looping over nodes
  std::vector<std::vector<PetscInt>> dofmap(node_dofmap.size());
  for (std::size_t i = 0; i < node_dofmap.size(); ++i)
  {
    const std::size_t local_dim0 = node_dofmap[i].size();
    dofmap[i].resize(block_size * local_dim0);
    for (std::size_t j = 0; j < local_dim0; ++j)
    {
      const int old_node = node_dofmap[i][j];
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
void DofMapBuilder::get_cell_entities_local(
    std::vector<std::vector<int64_t>>& entity_indices, const mesh::Cell& cell,
    const std::vector<bool>& needs_mesh_entities)
{
  const std::size_t D = cell.mesh().topology().dim();
  {
    for (std::size_t d = 0; d < D; ++d)
    {
      if (needs_mesh_entities[d])
      {
        for (std::size_t i = 0; i < cell.num_entities(d); ++i)
          entity_indices[d][i] = cell.entities(d)[i];
      }
    }
  }

  // Handle cell index separately because cell.entities(D) doesn't work.
  if (needs_mesh_entities[D])
    entity_indices[D][0] = cell.index();
}
//-----------------------------------------------------------------------------
void DofMapBuilder::get_cell_entities_global(
    std::vector<std::vector<int64_t>>& entity_indices, const mesh::Cell& cell,
    const std::vector<bool>& needs_mesh_entities)
{
  const mesh::MeshTopology& topology = cell.mesh().topology();
  const std::size_t D = topology.dim();
  for (std::size_t d = 0; d < D; ++d)
  {
    if (needs_mesh_entities[d])
    {
      // TODO: Check if this ever will be false in here
      if (topology.have_global_indices(d))
      {
        const auto& global_indices = topology.global_indices(d);
        for (std::size_t i = 0; i < cell.num_entities(d); ++i)
          entity_indices[d][i] = global_indices[cell.entities(d)[i]];
      }
      else
      {
        for (std::size_t i = 0; i < cell.num_entities(d); ++i)
          entity_indices[d][i] = cell.entities(d)[i];
      }
    }
  }
  // Handle cell index separately because cell.entities(D) doesn't work.
  if (needs_mesh_entities[D])
  {
    if (topology.have_global_indices(D))
      entity_indices[D][0] = cell.global_index();
    else
      entity_indices[D][0] = cell.index();
  }
}
// TODO: The above and below functions are _very_ similar, can they be
// combined?
//-----------------------------------------------------------------------------
std::vector<int64_t> DofMapBuilder::compute_num_mesh_entities_local(
    const mesh::Mesh& mesh, const std::vector<bool>& needs_mesh_entities)
{
  const std::size_t D = mesh.topology().dim();
  std::vector<int64_t> num_mesh_entities_local(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
  {
    if (needs_mesh_entities[d])
    {
      mesh.init(d);
      num_mesh_entities_local[d] = mesh.num_entities(d);
    }
  }
  return num_mesh_entities_local;
}
//-----------------------------------------------------------------------------
