// Copyright (C) 2008-2015 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMapBuilder.h"
#include "DofMap.h"
#include <cstdlib>
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
#include <dolfin/parameter/GlobalParameters.h>
#include <memory>
#include <random>
#include <ufc.h>
#include <utility>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
void DofMapBuilder::build(fem::DofMap& dofmap, const mesh::Mesh& mesh)
{
  assert(dofmap._ufc_dofmap);

  // Start timer for dofmap initialization
  common::Timer t0("Init dofmap");

  // Check if dofmap is distributed (based on mesh MPI communicator)
  const bool distributed = dolfin::MPI::size(mesh.mpi_comm()) > 1;

  // Check if UFC dofmap should not be re-ordered (only applicable in
  // serial)
  const bool reorder_ufc = dolfin::parameter::parameters["reorder_dofs_serial"];
  const bool reorder = (distributed or reorder_ufc) ? true : false;

  // Sanity checks on UFC dofmap
  const std::size_t D = mesh.topology().dim();
  assert(dofmap._ufc_dofmap);

  // Extract needs_entities as vector
  std::vector<bool> needs_entities(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
    needs_entities[d] = (dofmap._ufc_dofmap->num_entity_dofs(d) > 0);

  // For mesh entities required by UFC dofmap, compute number of
  // entities on this process
  const std::vector<int64_t> num_mesh_entities_local
      = compute_num_mesh_entities_local(mesh, needs_entities);

  // NOTE: We test for global dofs here because the the function
  // DofMapBuilder::compute_block size cannot distinguish between
  // global and discontinuous Lagrange elements because the UFC dofmap
  // associates global dofs with cells, which is not correct. See
  // https://bitbucket.org/fenics-project/ffc/issue/61. If global dofs
  // are present, we set the block size to 1.

  // Compute local UFC indices of any 'global' dofs
  const std::set<std::size_t> global_dofs
      = compute_global_dofs(dofmap._ufc_dofmap, num_mesh_entities_local);

  // Determine and set dof block size (block size must be 1 if UFC map
  // is not re-ordered or if global dofs are present)
  const std::size_t bs = (global_dofs.empty() and reorder)
                             ? compute_blocksize(*dofmap._ufc_dofmap, D)
                             : 1;

  // Compute a 'node' dofmap based on a UFC dofmap. Returns:
  // - node dofmap (node_dofmap)
  // - local-to-global node indices (node_local_to_global)
  // - UFC local indices to new local indices
  //   (node_ufc_local_to_local, constrained maps only)
  // - Number of mesh entities (global), which may differ from that
  //   from the mesh if the dofmap is constrained
  std::vector<std::size_t> node_local_to_global0;
  std::vector<std::vector<la_index_t>> node_graph0;
  std::vector<int> node_ufc_local_to_local0;
  std::shared_ptr<const ufc_dofmap> ufc_node_dofmap;
  std::tie(ufc_node_dofmap, node_graph0, node_local_to_global0,
           dofmap._num_mesh_entities_global)
      = build_ufc_node_graph(dofmap._ufc_dofmap, mesh, bs);

  assert(ufc_node_dofmap);

  // Set local (cell) dimension
  dofmap._cell_dimension = dofmap._ufc_dofmap->num_element_dofs;

  // Set global dimension
  dofmap._global_dimension = 0;
  unsigned int d = 0;
  for (auto& n : dofmap._num_mesh_entities_global)
  {
    dofmap._global_dimension += n * dofmap._ufc_dofmap->num_entity_dofs(d);
    ++d;
  }

  // Compute local UFC indices of any 'global' dofs, and re-map if
  // required, e.g., in case that dofmap is periodic
  std::set<std::size_t> global_nodes0
      = compute_global_dofs(ufc_node_dofmap, num_mesh_entities_local);
  if (!node_ufc_local_to_local0.empty())
  {
    std::set<std::size_t> remapped_global_nodes;
    for (auto node : global_nodes0)
    {
      assert(node < node_ufc_local_to_local0.size());
      remapped_global_nodes.insert(node_ufc_local_to_local0[node]);
    }
    global_nodes0 = remapped_global_nodes;
  }

  // Dynamic data structure to build dofmap graph
  std::vector<std::vector<la_index_t>> dofmap_graph;

  // Re-order and switch to local indexing in dofmap when distributed
  // for process locality and set local_range
  if (reorder)
  {
    // Mark shared nodes. Boundary nodes are assigned a random
    // positive integer, interior nodes are marked as -1, interior
    // nodes in ghost layer of other processes are marked -2, and
    // ghost nodes are marked as -3
    std::vector<int> shared_nodes = compute_shared_nodes(
        node_graph0, node_local_to_global0.size(), *ufc_node_dofmap, mesh);

    // Compute:
    // (a) owned and shared nodes (and owned and un-owned):
    //    -1: unowned, 0: owned and shared, 1: owned and not shared;
    // (b) map from shared node to sharing processes; and
    // (c) set of all processes that share dofs with this process
    std::vector<short int> node_ownership0;
    std::unordered_map<int, std::vector<int>> shared_node_to_processes0;
    int num_owned_nodes;
    std::tie(num_owned_nodes, node_ownership0, shared_node_to_processes0,
             dofmap._neighbours)
        = compute_node_ownership(node_graph0, shared_nodes, global_nodes0,
                                 node_local_to_global0, mesh,
                                 dofmap._global_dimension / bs);

    dofmap._index_map = std::make_shared<common::IndexMap>(mesh.mpi_comm(),
                                                           num_owned_nodes, bs);

    // Sanity check
    assert(
        MPI::sum(mesh.mpi_comm(),
                 bs * dofmap._index_map->size(common::IndexMap::MapSize::OWNED))
        == (std::size_t)dofmap._global_dimension);

    // Compute node re-ordering for process index locality and spatial
    // locality within a process, including
    // (a) Old-to-new node indices (local)
    // (b) Owning process for nodes that are not owned by this process
    // (c) New local node index to new global node index
    // (d) Old local node index to new local node index
    assert(dofmap._index_map);
    std::vector<int> node_old_to_new_local;
    std::vector<std::size_t> local_to_global_unowned;
    std::tie(node_old_to_new_local, local_to_global_unowned)
        = compute_node_reordering(
            shared_node_to_processes0, node_local_to_global0, node_graph0,
            node_ownership0, global_nodes0, mesh.mpi_comm());

    dofmap._index_map->set_block_local_to_global(local_to_global_unowned);

    // FIXME: Simplify after constrained domain removal
    // Update UFC-local-to-local map to account for re-ordering
    // UFC dofmap was not altered, old_to_new is same as UFC-to-new
    dofmap._ufc_local_to_local = node_old_to_new_local;

    // Update shared_nodes for node reordering
    dofmap._shared_nodes.clear();
    for (auto it = shared_node_to_processes0.begin();
         it != shared_node_to_processes0.end(); ++it)
    {
      const int new_node = node_old_to_new_local[it->first];
      dofmap._shared_nodes[new_node] = it->second;
    }

    // Update global_nodes for node reordering
    dofmap._global_nodes.clear();
    for (auto it = global_nodes0.begin(); it != global_nodes0.end(); ++it)
      dofmap._global_nodes.insert(node_old_to_new_local[*it]);

    // Build dofmap from original node 'dof' map and applying the
    // 'old_to_new_local' map for the re-ordered node indices
    dofmap_graph = build_dofmap(node_graph0, node_old_to_new_local, bs);
  }
  else
  {
    // UFC dofmap has not been re-ordered
    assert(!distributed);
    dofmap_graph = node_graph0;
    dofmap._ufc_local_to_local = node_ufc_local_to_local0;
    if (dofmap._ufc_local_to_local.empty()
        && dofmap._ufc_dofmap->num_sub_dofmaps > 0)
    {
      dofmap._ufc_local_to_local.resize(dofmap._global_dimension);
      for (std::size_t i = 0; i < dofmap._ufc_local_to_local.size(); ++i)
        dofmap._ufc_local_to_local[i] = i;
    }

    dofmap._index_map = std::make_shared<common::IndexMap>(
        mesh.mpi_comm(), dofmap._global_dimension, bs);

    dofmap._shared_nodes.clear();

    // Store global nodes
    dofmap._global_nodes = global_nodes0;
  }

  // Clear ufc_local-to-local map if dofmap has no sub-maps
  if (dofmap._ufc_dofmap->num_sub_dofmaps == 0)
    std::vector<int>().swap(dofmap._ufc_local_to_local);

  // Build flattened dofmap graph
  dofmap._dofmap.clear();
  for (auto const& cell_dofs : dofmap_graph)
  {
    dofmap._dofmap.insert(dofmap._dofmap.end(), cell_dofs.begin(),
                          cell_dofs.end());
  }
}
//-----------------------------------------------------------------------------
void DofMapBuilder::build_sub_map_view(
    fem::DofMap& sub_dofmap, const fem::DofMap& parent_dofmap,
    const std::vector<std::size_t>& component, const mesh::Mesh& mesh)
{
  // Note: Ownership range is set to zero since dofmap is a view
  assert(!component.empty());

  // Convenience reference to parent UFC dofmap
  assert(parent_dofmap._ufc_dofmap);
  const ufc_dofmap& parent_ufc_dofmap = *parent_dofmap._ufc_dofmap;

  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Extract needs_entities as vector
  std::vector<bool> needs_entities(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
    needs_entities[d] = (parent_ufc_dofmap.num_entity_dofs(d) > 0);

  // Generate and number required mesh entities for local UFC map
  const std::vector<int64_t> num_mesh_entities_local
      = compute_num_mesh_entities_local(mesh, needs_entities);

  // Initialise UFC offset from parent
  std::size_t ufc_offset = parent_dofmap._ufc_offset;

  // Extract local UFC sub-dofmap from parent and update offset
  sub_dofmap._ufc_dofmap = extract_ufc_sub_dofmap(
      parent_ufc_dofmap, ufc_offset, component, num_mesh_entities_local);
  assert(sub_dofmap._ufc_dofmap);

  // Set UFC sub-dofmap offset
  sub_dofmap._ufc_offset = ufc_offset;

  // Build local UFC-based dof map for sub-dofmap
  // Dynamic data structure to build dofmap graph
  std::vector<std::vector<la_index_t>> sub_dofmap_graph
      = build_local_ufc_dofmap(*sub_dofmap._ufc_dofmap, mesh);

  // Add offset to local UFC dofmap
  for (std::size_t i = 0; i < sub_dofmap_graph.size(); ++i)
  {
    for (std::size_t j = 0; j < sub_dofmap_graph[i].size(); ++j)
    {
      sub_dofmap_graph[i][j] += ufc_offset;
    }
  }

  // Store number of global mesh entities and set global dimension
  sub_dofmap._num_mesh_entities_global
      = parent_dofmap._num_mesh_entities_global;
  assert(!sub_dofmap._num_mesh_entities_global.empty());
  sub_dofmap._global_dimension = 0;
  unsigned int d = 0;
  for (auto& n : sub_dofmap._num_mesh_entities_global)
  {
    sub_dofmap._global_dimension
        += n * sub_dofmap._ufc_dofmap->num_entity_dofs(d);
    ++d;
  }

  // Copy data from parent
  // FIXME: Do we touch sub_dofmap.index_map() in this routine? If yes, then
  //        this routine has incorrect constness!
  sub_dofmap.index_map() = parent_dofmap.index_map();
  sub_dofmap._shared_nodes = parent_dofmap._shared_nodes;
  sub_dofmap._neighbours = parent_dofmap._neighbours;

  // Store UFC local to re-ordered local if submap has any submaps
  if (sub_dofmap._ufc_dofmap->num_sub_dofmaps > 0)
    sub_dofmap._ufc_local_to_local = parent_dofmap._ufc_local_to_local;
  else
    sub_dofmap._ufc_local_to_local.clear();

  if (parent_dofmap._ufc_local_to_local.empty())
  {
    throw std::runtime_error("Building  sub-dofmap view - re-ordering map not "
                             "available. It may be been cleared by the user");
  }

  // Map to re-ordered dofs
  const std::vector<int>& local_to_local = parent_dofmap._ufc_local_to_local;
  const std::size_t bs = parent_dofmap.block_size();
  for (auto cell_map = sub_dofmap_graph.begin();
       cell_map != sub_dofmap_graph.end(); ++cell_map)
  {
    for (auto dof = cell_map->begin(); dof != cell_map->end(); ++dof)
    {
      const std::div_t div = std::div((int)*dof, (int)local_to_local.size());
      const std::size_t node = div.rem;
      const std::size_t component = div.quot;

      // Get dof from UFC local-to-local map
      assert(node < local_to_local.size());
      std::size_t current_dof = bs * local_to_local[node] + component;

      // Set dof index in transformed dofmap
      *dof = current_dof;
    }
  }

  // Set local (cell) dimension
  sub_dofmap._cell_dimension = sub_dofmap._ufc_dofmap->num_element_dofs;

  // Construct flattened dofmap
  sub_dofmap._dofmap.clear();
  for (auto const& cell_dofs : sub_dofmap_graph)
  {
    sub_dofmap._dofmap.insert(sub_dofmap._dofmap.end(), cell_dofs.begin(),
                              cell_dofs.end());
  }
}
//-----------------------------------------------------------------------------
std::vector<std::vector<dolfin::la_index_t>>
DofMapBuilder::build_local_ufc_dofmap(const ufc_dofmap& ufc_dofmap,
                                      const mesh::Mesh& mesh)
{
  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Extract needs_entities as vector
  std::vector<bool> needs_entities(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
    needs_entities[d] = (ufc_dofmap.num_entity_dofs(d) > 0);

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
  std::vector<std::vector<dolfin::la_index_t>> dofmap(
      mesh.num_cells(), std::vector<la_index_t>(ufc_dofmap.num_element_dofs));
  std::vector<int64_t> dof_holder(ufc_dofmap.num_element_dofs);
  std::vector<const int64_t*> _entity_indices(entity_indices.size());
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    // Fill entity indices array
    get_cell_entities_local(entity_indices, cell, needs_entities);

    // Tabulate dofs for cell
    for (std::size_t i = 0; i < entity_indices.size(); ++i)
      _entity_indices[i] = entity_indices[i].data();
    ufc_dofmap.tabulate_dofs(dof_holder.data(), num_mesh_entities.data(),
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
    const std::vector<std::vector<la_index_t>>& dofmap,
    const std::vector<int>& shared_nodes,
    const std::set<std::size_t>& global_nodes,
    const std::vector<std::size_t>& local_to_global, const mesh::Mesh& mesh,
    const std::size_t global_dim)
{
  log::log(TRACE, "Determining node ownership for parallel dof map");

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

        // First check to see if this is a ghost/ghost-shared node,
        // and set ownership accordingly. Otherwise use the ownership
        // from the sorting process
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

  // Shared ownership for global dofs (after neighbour calculation)
  std::vector<int> all_procs;
  for (std::uint32_t i = 0; i != num_processes; ++i)
    if (i != process_number)
      all_procs.push_back((int)i);

  // Add/remove global dofs to/from relevant sets (last process owns
  // global nodes)
  for (auto node = global_nodes.begin(); node != global_nodes.end(); ++node)
  {
    assert(*node < node_ownership.size());
    if (process_number == num_processes - 1)
    {
      node_ownership[*node] = 0;
    }
    else
    {
      node_ownership[*node] = -1;
      --num_owned_nodes;
    }
    shared_node_to_processes.insert(std::make_pair(*node, all_procs));
  }

  log::log(TRACE, "Finished determining dof ownership for parallel dof map");

  return std::make_tuple(num_owned_nodes, std::move(node_ownership),
                         std::move(shared_node_to_processes),
                         std::move(neighbours));
}
//-----------------------------------------------------------------------------
std::set<std::size_t> DofMapBuilder::compute_global_dofs(
    std::shared_ptr<const ufc_dofmap> ufc_dofmap,
    const std::vector<int64_t>& num_mesh_entities_local)
{
  // Compute global dof indices
  std::size_t offset_local = 0;
  std::set<std::size_t> global_dof_indices;
  _compute_global_dofs(global_dof_indices, offset_local, ufc_dofmap,
                       num_mesh_entities_local);

  return global_dof_indices;
}
//-----------------------------------------------------------------------------
void DofMapBuilder::_compute_global_dofs(
    std::set<std::size_t>& global_dofs, std::size_t& offset_local,
    const std::shared_ptr<const ufc_dofmap> ufc_dofmap,
    const std::vector<int64_t>& num_mesh_entities_local)
{
  assert(ufc_dofmap);

  if (ufc_dofmap->num_sub_dofmaps == 0)
  {
    // Check if dofmap is for global dofs
    bool global_dof = true;
    for (std::size_t d = 0; d < num_mesh_entities_local.size(); ++d)
    {
      if (ufc_dofmap->num_entity_dofs(d) > 0)
      {
        global_dof = false;
        break;
      }
    }

    if (global_dof)
    {
      unsigned int d = 0;
      std::size_t ndofs = 0;
      for (auto& n : num_mesh_entities_local)
      {
        ndofs += n * ufc_dofmap->num_entity_dofs(d);
        ++d;
      }

      // Check that we have just one dof
      if (ndofs != 1)
      {
        throw std::runtime_error("Computing global degrees of freedom - global "
                                 "degree of freedom has dimension != 1");
      }

      // Create dummy entity_indices argument to tabulate single
      // global dof
      const int64_t** dummy_entity_indices = nullptr;
      int64_t dof_local = 0;
      ufc_dofmap->tabulate_dofs(&dof_local, num_mesh_entities_local.data(),
                                dummy_entity_indices);

      // Insert global dof index
      std::pair<std::set<std::size_t>::iterator, bool> ret
          = global_dofs.insert(dof_local + offset_local);
      if (!ret.second)
      {
        std::runtime_error("Computing global degrees of freedom - global "
                           "degree of freedom already exists");
      }
    }
  }
  else
  {
    // Loop through sub-dofmap looking for global dofs
    for (int i = 0; i < ufc_dofmap->num_sub_dofmaps; ++i)
    {
      // Extract sub-dofmap and initialise
      std::shared_ptr<struct ufc_dofmap> sub_dofmap(
          ufc_dofmap->create_sub_dofmap(i));
      _compute_global_dofs(global_dofs, offset_local, sub_dofmap,
                           num_mesh_entities_local);

      // Get offset
      if (sub_dofmap->num_sub_dofmaps == 0)
      {
        unsigned int d = 0;
        for (auto& n : num_mesh_entities_local)
        {
          offset_local += n * sub_dofmap->num_entity_dofs(d);
          ++d;
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------
std::shared_ptr<ufc_dofmap> DofMapBuilder::extract_ufc_sub_dofmap(
    const ufc_dofmap& ufc_dofmap, std::size_t& offset,
    const std::vector<std::size_t>& component,
    const std::vector<int64_t>& num_mesh_entities)
{
  // Check if there are any sub systems
  if (ufc_dofmap.num_sub_dofmaps == 0)
  {
    throw std::runtime_error("Extracting subsystem of degree of freedom "
                             "mapping - there are no subsystems");
  }

  // Check that a sub system has been specified
  if (component.empty())
  {
    throw std::runtime_error("Extracting subsystem of degree of freedom "
                             "mapping - no system was specified");
  }

  // Check the number of available sub systems
  if ((int)component[0] >= ufc_dofmap.num_sub_dofmaps)
  {
    throw std::runtime_error(
        "Requested subsystem (" + std::to_string(component[0])
        + ") out of range [0, " + std::to_string(ufc_dofmap.num_sub_dofmaps)
        + ")");
  }

  // Add to offset if necessary
  for (std::size_t i = 0; i < component[0]; i++)
  {
    // Extract sub dofmap
    std::unique_ptr<struct ufc_dofmap> ufc_tmp_dofmap(
        ufc_dofmap.create_sub_dofmap(i));
    assert(ufc_tmp_dofmap);

    // Get offset
    unsigned int d = 0;
    for (auto& n : num_mesh_entities)
    {
      offset += n * ufc_tmp_dofmap->num_entity_dofs(d);
      ++d;
    }
  }

  // Create UFC sub-system
  std::shared_ptr<struct ufc_dofmap> sub_dofmap(
      ufc_dofmap.create_sub_dofmap(component[0]));
  assert(sub_dofmap);

  // Return sub-system if sub-sub-system should not be extracted,
  // otherwise recursively extract the sub sub system
  if (component.size() == 1)
    return sub_dofmap;
  else
  {
    std::vector<std::size_t> sub_component;
    for (std::size_t i = 1; i < component.size(); ++i)
      sub_component.push_back(component[i]);

    std::shared_ptr<struct ufc_dofmap> sub_sub_dofmap = extract_ufc_sub_dofmap(
        *sub_dofmap, offset, sub_component, num_mesh_entities);

    return sub_sub_dofmap;
  }
}
//-----------------------------------------------------------------------------
std::size_t DofMapBuilder::compute_blocksize(const ufc_dofmap& ufc_dofmap,
                                             std::size_t tdim)
{
  bool has_block_structure = false;
  if (ufc_dofmap.num_sub_dofmaps > 1)
  {
    // Create UFC first sub-dofmap
    std::unique_ptr<struct ufc_dofmap> ufc_sub_dofmap0(
        ufc_dofmap.create_sub_dofmap(0));
    assert(ufc_sub_dofmap0);

    // Create UFC sub-dofmaps and check if all sub dofmaps have the
    // same number of dofs per entity
    if (ufc_sub_dofmap0->num_sub_dofmaps != 0)
      has_block_structure = false;
    else
    {
      // Assume dof map has block structure, then check
      has_block_structure = true;

      // Create UFC sub-dofmaps and check that all sub dofmaps have
      // the same number of dofs per entity
      for (int i = 1; i < ufc_dofmap.num_sub_dofmaps; ++i)
      {
        std::unique_ptr<struct ufc_dofmap> ufc_sub_dofmap(
            ufc_dofmap.create_sub_dofmap(i));
        assert(ufc_sub_dofmap);
        for (std::size_t d = 0; d <= tdim; ++d)
        {
          if (ufc_sub_dofmap->num_entity_dofs(d)
              != ufc_sub_dofmap0->num_entity_dofs(d))
          {
            has_block_structure = false;
            break;
          }
        }
      }
    }
  }

  if (has_block_structure)
    return ufc_dofmap.num_sub_dofmaps;
  else
    return 1;
}
//-----------------------------------------------------------------------------
std::tuple<std::shared_ptr<const ufc_dofmap>,
           std::vector<std::vector<la_index_t>>, std::vector<std::size_t>,
           std::vector<int64_t>>
DofMapBuilder::build_ufc_node_graph(
    std::shared_ptr<const ufc_dofmap> ufc_dofmap, const mesh::Mesh& mesh,
    const std::size_t block_size)
{
  assert(ufc_dofmap);

  // Start timer for dofmap initialization
  common::Timer t0("Init dofmap from UFC dofmap");

  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Extract needs_entities as vector
  std::vector<bool> needs_entities(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
    needs_entities[d] = (ufc_dofmap->num_entity_dofs(d) > 0);

  // Generate and number required mesh entities (local & global, and
  // constrained global)
  std::vector<int64_t> num_mesh_entities_local(D + 1, 0);
  std::vector<int64_t> num_mesh_entities_global_unconstrained(D + 1, 0);
  for (std::size_t d = 0; d <= D; ++d)
  {
    if (needs_entities[d])
    {
      mesh.init(d);
      mesh::DistributedMeshTools::number_entities(mesh, d);
      num_mesh_entities_local[d] = mesh.num_entities(d);
      num_mesh_entities_global_unconstrained[d] = mesh.num_entities_global(d);
    }
  }

  // Extract sub-dofmaps
  std::vector<std::shared_ptr<const struct ufc_dofmap>> dofmaps(block_size);
  std::vector<std::size_t> offset_local(block_size + 1, 0);
  if (block_size > 1)
  {
    std::vector<std::size_t> component(1);
    std::size_t _offset_local = 0;
    for (std::size_t i = 0; i < block_size; ++i)
    {
      component[0] = i;
      dofmaps[i] = extract_ufc_sub_dofmap(*ufc_dofmap, _offset_local, component,
                                          num_mesh_entities_local);
      offset_local[i] = _offset_local;
    }
  }
  else
    dofmaps[0] = ufc_dofmap;

  offset_local[block_size] = 0;
  unsigned int d = 0;
  for (auto& n : num_mesh_entities_local)
  {
    offset_local[block_size] += n * ufc_dofmap->num_entity_dofs(d);
    ++d;
  }

  std::vector<int64_t> num_mesh_entities_global
      = num_mesh_entities_global_unconstrained;

  // Allocate space for dof map
  std::vector<std::vector<la_index_t>> node_dofmap(mesh.num_cells());

  // Get standard local elem2ent dimension
  const std::size_t local_dim = dofmaps[0]->num_element_dofs;

  // Holder for UFC 64-bit dofmap integers
  std::vector<int64_t> ufc_nodes_global(local_dim);
  std::vector<int64_t> ufc_nodes_local(local_dim);

  // Allocate entity indices array
  std::vector<std::vector<int64_t>> entity_indices(D + 1);
  std::vector<const int64_t*> entity_indices_ptr(entity_indices.size());
  for (std::size_t d = 0; d <= D; ++d)
    entity_indices[d].resize(mesh.type().num_entities(d));

  // Resize local-to-global map
  std::vector<std::size_t> node_local_to_global(offset_local[1]);

  // Build dofmaps from ufc_dofmap
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    // Get reference to container for cell dofs
    std::vector<la_index_t>& cell_nodes = node_dofmap[cell.index()];
    cell_nodes.resize(local_dim);

    // Tabulate standard UFC dof map for first space (local)
    get_cell_entities_local(entity_indices, cell, needs_entities);
    // FIXME: Can the pointers be copied outside of this loop?
    for (std::size_t i = 0; i < entity_indices.size(); ++i)
      entity_indices_ptr[i] = entity_indices[i].data();
    dofmaps[0]->tabulate_dofs(ufc_nodes_local.data(),
                              num_mesh_entities_local.data(),
                              entity_indices_ptr.data());
    std::copy(ufc_nodes_local.begin(), ufc_nodes_local.end(),
              cell_nodes.begin());

    // Tabulate standard UFC dof map for first space (global)
    get_cell_entities_global(entity_indices, cell, needs_entities);
    // FIXME: Do the pointers need to be copied again?
    for (std::size_t i = 0; i < entity_indices.size(); ++i)
      entity_indices_ptr[i] = entity_indices[i].data();
    dofmaps[0]->tabulate_dofs(ufc_nodes_global.data(),
                              num_mesh_entities_global_unconstrained.data(),
                              entity_indices_ptr.data());

    // Build local-to-global map for nodes
    for (std::size_t i = 0; i < local_dim; ++i)
    {
      assert(ufc_nodes_local[i] < (int)node_local_to_global.size());
      node_local_to_global[ufc_nodes_local[i]] = ufc_nodes_global[i];
    }
  }

  return std::make_tuple(dofmaps[0], std::move(node_dofmap),
                         std::move(node_local_to_global),
                         std::move(num_mesh_entities_global));
}
//-----------------------------------------------------------------------------
std::vector<int> DofMapBuilder::compute_shared_nodes(
    const std::vector<std::vector<la_index_t>>& node_dofmap,
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

  std::vector<int> facet_nodes(ufc_dofmap.num_facet_dofs);

  // Mark dofs associated ghost cells as ghost dofs (provisionally)
  bool has_ghost_cells = false;
  for (auto& c : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    const std::vector<la_index_t>& cell_nodes = node_dofmap[c.index()];
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
          ufc_dofmap.tabulate_facet_dofs(facet_nodes.data(), c.index(f));
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
    const std::vector<la_index_t>& cell_nodes = node_dofmap[cell0.index()];

    // Tabulate which dofs are on the facet
    ufc_dofmap.tabulate_facet_dofs(facet_nodes.data(), cell0.index(f));

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
    const std::vector<std::vector<la_index_t>>& node_dofmap,
    const std::vector<short int>& node_ownership,
    const std::set<std::size_t>& global_nodes, MPI_Comm mpi_comm)
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

  // Build graph for re-ordering. Below block is scoped to clear
  // working data structures once graph is constructed.
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
    const std::vector<la_index_t>& nodes = node_dofmap[cell];
    std::vector<int> local_old;

    // Loop over nodes collecting valid local nodes
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
      if (global_nodes.find(nodes[i]) != global_nodes.end())
        continue;

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
  const std::string ordering_library
      = dolfin::parameter::parameters["dof_ordering_library"];
  std::vector<int> node_remap;
  if (ordering_library == "Boost")
  {
    node_remap
        = dolfin::graph::BoostGraphOrdering::compute_cuthill_mckee(graph, true);
  }
  else if (ordering_library == "SCOTCH")
    node_remap = dolfin::graph::SCOTCH::compute_gps(graph);
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
  //  off_process_owner.resize(unowned_local_size);
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
std::vector<std::vector<la_index_t>> DofMapBuilder::build_dofmap(
    const std::vector<std::vector<la_index_t>>& node_dofmap,
    const std::vector<int>& old_to_new_node_local, const std::size_t block_size)
{
  // Build dofmap looping over nodes
  std::vector<std::vector<la_index_t>> dofmap(node_dofmap.size());
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
