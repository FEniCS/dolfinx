// Copyright (C) 2011-2014 Garth N. Wells and Chris Richardson
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
// Modified by Anders Logg 2011
//
// First added:  2011-09-17
// Last changed: 2014-07-02

#include <boost/multi_array.hpp>

#include "dolfin/common/MPI.h"
#include "dolfin/common/Timer.h"
#include "dolfin/graph/Graph.h"
#include "dolfin/graph/SCOTCH.h"
#include "dolfin/log/log.h"
#include "BoundaryMesh.h"
#include "Facet.h"
#include "Mesh.h"
#include "MeshEntityIterator.h"
#include "MeshFunction.h"
#include "Vertex.h"

#include "DistributedMeshTools.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void DistributedMeshTools::number_entities(const Mesh& mesh, std::size_t d)
{
  Timer timer("Number distributed mesh entities");

  // Return if global entity indices have already been calculated
  if (mesh.topology().have_global_indices(d))
    return;

  // Const-cast to allow data to be attached
  Mesh& _mesh = const_cast<Mesh&>(mesh);

  if (MPI::size(mesh.mpi_comm()) == 1)
  {
    mesh.init(d);

    // Set global entity numbers in mesh
    _mesh.topology().init(d, mesh.num_entities(d), mesh.num_entities(d));
    _mesh.topology().init_global_indices(d, mesh.num_entities(d));
    for (MeshEntityIterator e(mesh, d); !e.end(); ++e)
      _mesh.topology().set_global_index(d, e->index(), e->index());

    return;
  }

  // Get shared entities map
  std::map<unsigned int, std::set<unsigned int>>&
    shared_entities = _mesh.topology().shared_entities(d);

  // Number entities
  std::vector<std::size_t> global_entity_indices;
  const std::map<unsigned int, std::pair<unsigned int, unsigned int>>
    slave_entities;
  const std::size_t num_global_entities = number_entities(mesh, slave_entities,
                                                          global_entity_indices,
                                                          shared_entities, d);

  // Set global entity numbers in mesh
  _mesh.topology().init(d, mesh.num_entities(d), num_global_entities);
  _mesh.topology().init_global_indices(d, global_entity_indices.size());
  for (std::size_t i = 0; i < global_entity_indices.size(); ++i)
    _mesh.topology().set_global_index(d, i, global_entity_indices[i]);
}
//-----------------------------------------------------------------------------
std::size_t DistributedMeshTools::number_entities(
  const Mesh& mesh,
  const std::map<unsigned int, std::pair<unsigned int,
  unsigned int>>& slave_entities,
  std::vector<std::size_t>& global_entity_indices,
  std::map<unsigned int, std::set<unsigned int>>& shared_entities,
  std::size_t d)
{
  // Developer note: This function should use global_vertex_indices
  // for the global mesh indices and *not* access these through the
  // mesh. In some cases special numbering is passed in which differs
  // from mesh global numbering, e.g. when computing mesh entity
  // numbering for problems with periodic boundary conditions.

  Timer timer("Number mesh entities for distributed mesh (for specified vertex ids)");

  // Check that we're not re-numbering vertices (these are fixed at
  // mesh construction)
  if (d == 0)
  {
    dolfin_error("MeshPartitioning.cpp",
                 "number mesh entities",
                 "Global vertex indices exist at input. Cannot be renumbered");
  }

  // Check that we're not re-numbering cells (these are fixed at mesh
  // construction)
  if (d == mesh.topology().dim())
  {
    shared_entities.clear();
    global_entity_indices = mesh.topology().global_indices(d);
    return mesh.size_global(d);

    /*
    dolfin_error("MeshPartitioning.cpp",
                 "number mesh entities",
                 "Global cell indices exist at input. Cannot be renumbered");
    */
  }

  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  // Get number of processes and process number
  const std::size_t num_processes = MPI::size(mpi_comm);
  const std::size_t process_number = MPI::rank(mpi_comm);

  // Initialize entities of dimension d locally
  mesh.init(d);

  // Build list of slave entities to exclude from ownership computation
  std::vector<bool> exclude(mesh.num_entities(d), false);
  std::map<unsigned int, std::pair<unsigned int,
                                   unsigned int>>::const_iterator s;
  for (s = slave_entities.begin(); s != slave_entities.end(); ++s)
    exclude[s->first] = true;

  // Build entity global [vertex list]-to-[local entity index]
  // map. Exclude any slave entities.
  std::map<std::vector<std::size_t>, unsigned int> entities;
  std::pair<std::vector<std::size_t>, unsigned int> entity;
  for (MeshEntityIterator e(mesh, d, "all"); !e.end(); ++e)
  {
    const std::size_t local_index = e->index();
    if (!exclude[local_index])
    {
      entity.second = local_index;
      entity.first = std::vector<std::size_t>();
      for (VertexIterator vertex(*e); !vertex.end(); ++vertex)
        entity.first.push_back(vertex->global_index());
      std::sort(entity.first.begin(), entity.first.end());
      entities.insert(entity);
    }
  }

  // Get vertex global indices
  const std::vector<std::size_t>& global_vertex_indices
    = mesh.topology().global_indices(0);

  // Get shared vertices (local index, [sharing processes])
  const std::map<unsigned int, std::set<unsigned int>>& shared_vertices_local
    = mesh.topology().shared_entities(0);

  // Compute ownership of entities of dimension d ([entity vertices], data):
  //  [0]: owned and shared (will be numbered by this process, and number
  //       communicated to other processes)
  //  [1]: not owned but shared (will be numbered by another process, and number
  //       communicated to this processes)
  std::array<std::map<Entity, EntityData>, 2> entity_ownership;
  std::vector<std::size_t> owned_entities;
  compute_entity_ownership(mpi_comm, entities, shared_vertices_local,
                           global_vertex_indices, d, owned_entities,
                           entity_ownership);

  // Split shared entities for convenience
  const std::map<Entity, EntityData>& owned_shared_entities
    = entity_ownership[0];
  std::map<Entity, EntityData>& unowned_shared_entities = entity_ownership[1];

  // Number of entities 'owned' by this process
  const std::size_t num_local_entities = owned_entities.size()
    + owned_shared_entities.size();

  // Compute global number of entities and local process offset
  const std::pair<std::size_t, std::size_t> num_global_entities
    = compute_num_global_entities(mpi_comm, num_local_entities, num_processes,
                                  process_number);

  // Extract offset
  std::size_t offset = num_global_entities.second;

  // Prepare list of global entity numbers. Check later that nothing
  // is equal to std::numeric_limits<std::size_t>::max()
  global_entity_indices
    = std::vector<std::size_t>(mesh.size(d),
                               std::numeric_limits<std::size_t>::max());

  std::map<Entity, EntityData>::const_iterator it;

  // Number exclusively owned entities
  for (std::size_t i = 0; i < owned_entities.size(); ++i)
    global_entity_indices[owned_entities[i]] = offset++;

  // Number shared entities that this process is responsible for
  // numbering
  std::map<Entity, EntityData>::const_iterator it1;
  for (it1 = owned_shared_entities.begin();
       it1 != owned_shared_entities.end(); ++it1)
  {
    global_entity_indices[it1->second.local_index] = offset++;
  }

  // Communicate indices for shared entities (owned by this process)
  // and get indices for shared but not owned entities
  std::vector<std::vector<std::size_t>> send_values(num_processes);
  std::vector<std::size_t> destinations;
  for (it1 = owned_shared_entities.begin();
       it1 != owned_shared_entities.end(); ++it1)
  {
    // Get entity index
    const unsigned int local_entity_index = it1->second.local_index;
    const std::size_t global_entity_index
      = global_entity_indices[local_entity_index];
    dolfin_assert(global_entity_index
                  != std::numeric_limits<std::size_t>::max());

    // Get entity processes (processes sharing the entity)
    const std::vector<unsigned int>& entity_processes = it1->second.processes;

    // Get entity vertices (global vertex indices)
    const Entity& e = it1->first;

    // Prepare data for sending
    for (std::size_t j = 0; j < entity_processes.size(); ++j)
    {
      // Store interleaved: entity index, number of vertices, global
      // vertex indices
      std::size_t p = entity_processes[j];
      send_values[p].push_back(global_entity_index);
      send_values[p].push_back(e.size());
      send_values[p].insert(send_values[p].end(), e.begin(), e.end());
    }
  }

  // Send data
  std::vector<std::vector<std::size_t>> received_values;
  MPI::all_to_all(mpi_comm, send_values, received_values);

  // Fill in global entity indices received from lower ranked
  // processes
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    for (std::size_t i = 0; i < received_values[p].size();)
    {
      const std::size_t global_index = received_values[p][i++];
      const std::size_t entity_size = received_values[p][i++];
      Entity e;
      for (std::size_t j = 0; j < entity_size; ++j)
        e.push_back(received_values[p][i++]);

      // Access unowned entity data
      std::map<Entity, EntityData>::const_iterator recv_entity;
      recv_entity = unowned_shared_entities.find(e);

      // Sanity check, should not receive an entity we don't need
      if (recv_entity == unowned_shared_entities.end())
      {
        std::stringstream msg;
        msg << "Process " << MPI::rank(mpi_comm)
            << " received illegal entity given by ";
        msg << " with global index " << global_index;
        msg << " from process " << p;
        dolfin_error("MeshPartitioning.cpp",
                     "number mesh entities",
                     msg.str());
      }

      const std::size_t local_entity_index = recv_entity->second.local_index;
      dolfin_assert(global_entity_indices[local_entity_index]
                    == std::numeric_limits<std::size_t>::max());
      global_entity_indices[local_entity_index] = global_index;
    }
  }

  // Get slave indices from master
  {
    std::vector<std::vector<std::size_t>>
      slave_send_buffer(MPI::size(mpi_comm));
    std::vector<std::vector<std::size_t>>
      local_slave_index(MPI::size(mpi_comm));
    for (s = slave_entities.begin(); s != slave_entities.end(); ++s)
    {
      // Local index on remote process
      slave_send_buffer[s->second.first].push_back(s->second.second);

      // Local index on this
      local_slave_index[s->second.first].push_back(s->first);
    }
    std::vector<std::vector<std::size_t>> slave_receive_buffer;
    MPI::all_to_all(mpi_comm, slave_send_buffer, slave_receive_buffer);

    // Send back master indices
    for (std::size_t p = 0; p < slave_receive_buffer.size(); ++p)
    {
      slave_send_buffer[p].clear();
      for (std::size_t i = 0; i < slave_receive_buffer[p].size(); ++i)
      {
        const std::size_t local_master = slave_receive_buffer[p][i];
        slave_send_buffer[p].push_back(global_entity_indices[local_master]);
      }
    }
    MPI::all_to_all(mpi_comm, slave_send_buffer, slave_receive_buffer);

    // Set slave indices to received master indices
    for (std::size_t p = 0; p < slave_receive_buffer.size(); ++p)
    {
      for (std::size_t i = 0; i < slave_receive_buffer[p].size(); ++i)
      {
        const std::size_t slave_index = local_slave_index[p][i];
        global_entity_indices[slave_index] = slave_receive_buffer[p][i];
      }
    }
  }

  // Sanity check
  for (std::size_t i = 0; i < global_entity_indices.size(); ++i)
  {
    dolfin_assert(global_entity_indices[i]
                  != std::numeric_limits<std::size_t>::max());
  }

  // Build shared_entities (global index, [sharing processes])
  shared_entities.clear();
  std::map<Entity, EntityData>::const_iterator e;
  for (e = owned_shared_entities.begin(); e != owned_shared_entities.end(); ++e)
  {
    const EntityData& ed = e->second;
    shared_entities[ed.local_index]
      = std::set<unsigned int>(ed.processes.begin(),
                               ed.processes.end());
  }
  for (e = unowned_shared_entities.begin();
       e != unowned_shared_entities.end(); ++e)
  {
    const EntityData& ed = e->second;
    shared_entities[ed.local_index]
      = std::set<unsigned int>(ed.processes.begin(),
                               ed.processes.end());
  }

  // Return number of global entities
  return num_global_entities.first;
}
//-----------------------------------------------------------------------------
std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t>>>
DistributedMeshTools::locate_off_process_entities(const std::vector<std::size_t>& entity_indices,
                                                  std::size_t dim,
                                                  const Mesh& mesh)
{
  Timer timer("Locate off-process entities");

  if (dim == 0)
  {
    warning("DistributedMeshTools::host_processes has not been tested for vertices.");
  }

  // Mesh topology dim
  const std::size_t D = mesh.topology().dim();

  // Check that entity is a vertex or a cell
  if (dim != 0 && dim != D)
  {
    dolfin_error("DistributedMeshTools.cpp",
                 "compute off-process indices",
                 "This version of DistributedMeshTools::host_processes is only for vertices or cells");
  }

  // Check that global numbers have been computed.
  if (!mesh.topology().have_global_indices(dim))
  {
    dolfin_error("DistributedMeshTools.cpp",
                 "compute off-process indices",
                 "Global mesh entity numbers have not been computed");
  }

  // Check that global numbers have been computed.
  if (!mesh.topology().have_global_indices(D))
  {
    dolfin_error("DistributedMeshTools.cpp",
                 "compute off-process indices",
                 "Global mesh entity numbers have not been computed");
  }

  // Get global cell entity indices on this process
  const std::vector<std::size_t> global_entity_indices
      = mesh.topology().global_indices(dim);

  dolfin_assert(global_entity_indices.size() == mesh.num_cells());

  // Prepare map to hold process numbers
  std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t>>>
    processes;

  // FIXME: work on optimising below code

  // List of indices to send
  std::vector<std::size_t> my_entities;

  // Remove local cells from my_entities to reduce communication
  if (dim == D)
  {
    // In order to fill vector my_entities...
    // build and populate a local set for non-local cells
    std::set<std::size_t> set_of_my_entities(entity_indices.begin(),
                                             entity_indices.end());

    const std::map<unsigned int, std::set<unsigned int>>& sharing_map
      = mesh.topology().shared_entities(D);

    // FIXME: This can be made more efficient by exploiting fact that
    //        set is sorted
    // Remove local cells from set_of_my_entities to reduce communication
    for (std::size_t j = 0; j < global_entity_indices.size(); ++j)
    {
      if (sharing_map.find(j) != sharing_map.end())
        set_of_my_entities.erase(global_entity_indices[j]);
    }
    // Copy entries from set_of_my_entities to my_entities
    my_entities = std::vector<std::size_t>(set_of_my_entities.begin(),
                                           set_of_my_entities.end());
  }
  else
    my_entities = entity_indices;

  // FIXME: handle case when my_entities.empty()
  //dolfin_assert(!my_entities.empty());

  // Prepare data structures for send/receive
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const std::size_t num_proc = MPI::size(mpi_comm);
  const std::size_t proc_num = MPI::rank(mpi_comm);
  const std::size_t max_recv = MPI::max(mpi_comm, my_entities.size());
  std::vector<std::size_t> off_process_entities(max_recv);

  // Send and receive data
  for (std::size_t k = 1; k < num_proc; ++k)
  {
    const std::size_t src  = (proc_num - k + num_proc) % num_proc;
    const std::size_t dest = (proc_num + k) % num_proc;
    MPI::send_recv(mpi_comm, my_entities, dest, off_process_entities, src);

    const std::size_t recv_entity_count = off_process_entities.size();

    // Check if this process owns received entities, and if so
    // store local index
    std::vector<std::size_t> my_hosted_entities;
    {
      // Build a temporary map hosting global_entity_indices
      std::map<std::size_t, std::size_t> map_of_global_entity_indices;
      for (std::size_t j = 0; j < global_entity_indices.size(); j++)
        map_of_global_entity_indices[global_entity_indices[j]] = j;

      for (std::size_t j = 0; j < recv_entity_count; j++)
      {
        // Check if this process hosts 'received_entity'
        const std::size_t received_entity = off_process_entities[j];
        std::map<std::size_t, std::size_t>::const_iterator it;
        it = map_of_global_entity_indices.find(received_entity);
        if (it != map_of_global_entity_indices.end())
        {
          const std::size_t local_index = it->second;
          my_hosted_entities.push_back(received_entity);
          my_hosted_entities.push_back(local_index);
        }
      }
    }

    // Send/receive hosted cells
    const std::size_t max_recv_host_proc
      = MPI::max(mpi_comm, my_hosted_entities.size());
    std::vector<std::size_t> host_processes(max_recv_host_proc);
    MPI::send_recv(mpi_comm, my_hosted_entities, src, host_processes, dest);

    const std::size_t recv_hostproc_count = host_processes.size();
    for (std::size_t j = 0; j < recv_hostproc_count; j += 2)
    {
      const std::size_t global_index = host_processes[j];
      const std::size_t local_index  = host_processes[j + 1];
      processes[global_index].insert({dest, local_index});
    }

    // FIXME: Do later for efficiency
    // Remove entries from entities (from my_entities) that cannot
    // reside on more processes (i.e., cells)
  }

  // Sanity check
  const std::set<std::size_t> test_set(my_entities.begin(), my_entities.end());
  const std::size_t number_expected = test_set.size();
  if (number_expected != processes.size())
  {
    dolfin_error("DistributedMeshTools.cpp",
                 "compute off-process indices",
                 "Sanity check failed");
  }

  return processes;
}
//-----------------------------------------------------------------------------
std::unordered_map<unsigned int,
                   std::vector<std::pair<unsigned int, unsigned int>>>
  DistributedMeshTools::compute_shared_entities(const Mesh& mesh, std::size_t d)
{
  Timer timer("Computed shared mesh entities");

  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const unsigned int comm_size = MPI::size(mpi_comm);

  // Return empty set if running in serial
  if (MPI::size(mpi_comm) == 1)
  {
    return std::unordered_map<unsigned int, std::vector<std::pair<unsigned int, unsigned int>>>();
  }

  // Initialize entities of dimension d
  mesh.init(d);

  // Number entities (globally)
  number_entities(mesh, d);

  // Get shared entities to processes map
  const std::map<unsigned int, std::set<unsigned int>>&
    shared_entities = mesh.topology().shared_entities(d);

  // Get local-to-global indices map
  const std::vector<std::size_t>& global_indices_map
    = mesh.topology().global_indices(d);

  // Global-to-local map for each process
  std::unordered_map<std::size_t, std::unordered_map<std::size_t, std::size_t>>
    global_to_local;

  // Pack global indices for sending to sharing processes
  std::vector<std::vector<std::size_t>> send_indices(comm_size);
  std::vector<std::vector<std::size_t>> local_sent_indices(comm_size);
  std::map<unsigned int, std::set<unsigned int>>::const_iterator shared_entity;
  for (shared_entity = shared_entities.begin();
       shared_entity != shared_entities.end(); ++shared_entity)
  {
    // Local index
    const unsigned int local_index = shared_entity->first;

    // Global index
    dolfin_assert(local_index < global_indices_map.size());
    std::size_t global_index = global_indices_map[local_index];

    // Destination process
    const std::set<unsigned int>& sharing_processes = shared_entity->second;

    // Pack data for sending and build global-to-local map
    std::set<unsigned int>::const_iterator dest;
    for (dest = sharing_processes.begin(); dest != sharing_processes.end();
         ++dest)
    {
      send_indices[*dest].push_back(global_index);
      local_sent_indices[*dest].push_back(local_index);
      global_to_local[*dest].insert({global_index, local_index});
    }
  }

  std::vector<std::vector<std::size_t>> recv_entities;
  MPI::all_to_all(mpi_comm, send_indices, recv_entities);

  // Clear send data
  send_indices.clear();
  send_indices.resize(comm_size);

  // Determine local entities indices for received global entity indices
  std::unordered_map<std::size_t, std::vector<std::size_t>>::const_iterator
    received_global_indices;
  for (std::size_t p = 0; p < recv_entities.size(); ++p)
  {
    // Get process number of neighbour
    const std::size_t sending_proc = p;

    if (recv_entities[p].size() > 0)
    {
      // Get global-to-local map for neighbour process
      std::unordered_map<std::size_t,
                         std::unordered_map<std::size_t,
                                            std::size_t>>::const_iterator
        it = global_to_local.find(sending_proc);
      dolfin_assert(it != global_to_local.end());
      const std::unordered_map<std::size_t, std::size_t>&
        neighbour_global_to_local = it->second;

      // Build vector of local indices
      const std::vector<std::size_t>& global_indices_recv
        = recv_entities[p];
      for (std::size_t i = 0; i < global_indices_recv.size(); ++i)
      {
        // Global index
        const std::size_t global_index = global_indices_recv[i];

        // Find local index corresponding to global index
        std::unordered_map<std::size_t, std::size_t>::const_iterator
          n_global_to_local = neighbour_global_to_local.find(global_index);

        dolfin_assert(n_global_to_local != neighbour_global_to_local.end());
        const std::size_t my_local_index = n_global_to_local->second;
        send_indices[sending_proc].push_back(my_local_index);
      }
    }
  }

  MPI::all_to_all(mpi_comm, send_indices, recv_entities);

  // Build map
  std::unordered_map<unsigned int,
                     std::vector<std::pair<unsigned int, unsigned int>>>
    shared_local_indices_map;

  // Loop over data received from each process
  std::unordered_map<std::size_t, std::vector<std::size_t>>::const_iterator
    received_local_indices;
  for (std::size_t p = 0; p < recv_entities.size(); ++p)
  {
    if (recv_entities[p].size() > 0)
    {
      // Process that shares entities
      const std::size_t proc = p;

      // Local indices on sharing process
      const std::vector<std::size_t>& neighbour_local_indices
        = recv_entities[p];

      // Local indices on this process
      const std::vector<std::size_t>& my_local_indices = local_sent_indices[p];

      // Check that sizes match
      dolfin_assert(my_local_indices.size() == neighbour_local_indices.size());

      for (std::size_t i = 0; i < neighbour_local_indices.size(); ++i)
      {
        shared_local_indices_map[my_local_indices[i]].push_back({proc, neighbour_local_indices[i]});
      }
    }
  }

  return shared_local_indices_map;
}
//-----------------------------------------------------------------------------
void DistributedMeshTools::compute_entity_ownership(
  const MPI_Comm mpi_comm,
  const std::map<std::vector<std::size_t>, unsigned int>& entities,
  const std::map<unsigned int, std::set<unsigned int>>& shared_vertices_local,
  const std::vector<std::size_t>& global_vertex_indices,
  std::size_t d,
  std::vector<std::size_t>& owned_entities,
  std::array<std::map<Entity, EntityData>, 2>& shared_entities)
{
  Timer timer("Compute mesh entity ownership");

  // Build global-to-local indices map for shared vertices
  std::map<std::size_t, std::set<unsigned int>> shared_vertices;
  std::map<unsigned int, std::set<unsigned int>>::const_iterator v;
  for (v = shared_vertices_local.begin(); v != shared_vertices_local.end(); ++v)
  {
    dolfin_assert(v->first < global_vertex_indices.size());
    shared_vertices.insert({global_vertex_indices[v->first], v->second});
  }

  // Entity ownership list ([entity vertices], data):
  //  [0]: owned and shared (will be numbered by this process, and number
  //       communicated to other processes)
  //  [1]: not owned but shared (will be numbered by another process, and number
  //       communicated to this processes)

  // Compute preliminary ownership lists (shared_entities) without
  // communication
  compute_preliminary_entity_ownership(mpi_comm, shared_vertices, entities,
                                       owned_entities, shared_entities);

  // Qualify boundary entities. We need to find out if the shared
  // (shared with lower ranked process) entities are entities of a
  // lower ranked process.  If not, this process becomes the lower
  // ranked process for the entity in question, and is therefore
  // responsible for communicating values to the higher ranked
  // processes (if any).
  compute_final_entity_ownership(mpi_comm, owned_entities, shared_entities);
}
//-----------------------------------------------------------------------------
void DistributedMeshTools::compute_preliminary_entity_ownership(
  const MPI_Comm mpi_comm,
  const std::map<std::size_t, std::set<unsigned int>>& shared_vertices,
  const std::map<Entity, unsigned int>& entities,
  std::vector<std::size_t>& owned_entities,
  std::array<std::map<Entity, EntityData>, 2>& shared_entities)
{
  // Entities
  std::map<Entity, EntityData>& owned_shared_entities = shared_entities[0];
  std::map<Entity, EntityData>& unowned_shared_entities = shared_entities[1];

  // Clear maps
  owned_entities.clear();
  owned_shared_entities.clear();
  unowned_shared_entities.clear();

  // Get my process number
  const std::size_t process_number = MPI::rank(mpi_comm);

  // Iterate over all local entities
  std::map<std::vector<std::size_t>,  unsigned int>::const_iterator it;
  for (it = entities.begin(); it != entities.end(); ++it)
  {
    const Entity& entity = it->first;
    const std::size_t local_entity_index = it->second;

    // Compute which processes entity is shared with
    std::vector< unsigned int> entity_processes;
    if (is_shared(entity, shared_vertices))
    {
      // Processes sharing first vertex of entity
      std::vector<std::size_t>
        intersection(shared_vertices.find(entity[0])->second.begin(),
                     shared_vertices.find(entity[0])->second.end());
      std::vector<std::size_t>::iterator intersection_end = intersection.end();

      // Loop over entity vertices
      for (std::size_t i = 1; i < entity.size(); ++i)
      {
        // Global vertex index
        const std::size_t v = entity[i];

        // Sharing processes
        const std::set< unsigned int>& shared_vertices_v
          = shared_vertices.find(v)->second;

        intersection_end
          = std::set_intersection(intersection.begin(), intersection_end,
                                  shared_vertices_v.begin(),
                                  shared_vertices_v.end(),
                                  intersection.begin());
      }
      entity_processes = std::vector< unsigned int>(intersection.begin(),
                                                    intersection_end);
    }

    // Check if entity is master, slave or shared but not owned (shared
    // with lower ranked process)
    bool shared_but_not_owned = false;
    for (std::size_t i = 0; i < entity_processes.size(); ++i)
    {
      if (entity_processes[i] < process_number)
      {
        shared_but_not_owned = true;
        break;
      }
    }

    if (entity_processes.empty())
    {
      owned_entities.push_back(local_entity_index);
    }
    else if (shared_but_not_owned)
    {
      unowned_shared_entities[entity] = EntityData(local_entity_index,
                                                   entity_processes);
    }
    else
    {
      owned_shared_entities[entity] = EntityData(local_entity_index,
                                                 entity_processes);
    }
  }
}
//-----------------------------------------------------------------------------
void DistributedMeshTools::compute_final_entity_ownership(
  const MPI_Comm mpi_comm,
  std::vector<std::size_t>& owned_entities,
  std::array<std::map<Entity, EntityData>, 2>& shared_entities)
{
  // Entities ([entity vertices], index) to be numbered
  std::map<Entity, EntityData>& owned_shared_entities = shared_entities[0];
  std::map<Entity, EntityData>& unowned_shared_entities = shared_entities[1];

  // Get MPI number of processes and process number
  const std::size_t num_processes = MPI::size(mpi_comm);
  const std::size_t process_number = MPI::rank(mpi_comm);

  // Communicate common entities, starting with the entities we think
  // are shared but not owned
  std::vector<std::vector<std::size_t>>
    send_common_entity_values(num_processes);
  for (std::map<Entity, EntityData>::const_iterator it
         = unowned_shared_entities.begin(); it != unowned_shared_entities.end();
       ++it)
  {
    // Get entity vertices (global vertex indices)
    const Entity& entity = it->first;

    // Get entity processes (processes that might share the entity)
    const std::vector<unsigned int>& entity_processes = it->second.processes;

    // Prepare data for sending
    for (std::size_t j = 0; j < entity_processes.size(); ++j)
    {
      const std::size_t p = entity_processes[j];
      send_common_entity_values[p].push_back(entity.size());
      send_common_entity_values[p].insert(send_common_entity_values[p].end(),
                                          entity.begin(), entity.end());
    }
  }

  // Communicate common entities, add the entities we think are owned
  // and shared
  for (std::map<Entity, EntityData>::const_iterator it
         = owned_shared_entities.begin();
       it != owned_shared_entities.end(); ++it)
  {
    // Get entity vertices (global vertex indices)
    const Entity& entity = it->first;

    // Get entity processes (processes that might share the entity)
    const std::vector<unsigned int>& entity_processes = it->second.processes;

    // Prepare data for sending
    for (std::size_t j = 0; j < entity_processes.size(); ++j)
    {
      const unsigned int p = entity_processes[j];
      dolfin_assert(process_number < p);
      send_common_entity_values[p].push_back(entity.size());
      send_common_entity_values[p].insert(send_common_entity_values[p].end(),
                                          entity.begin(), entity.end());
    }
  }

  // Communicate common entities
  std::vector<std::vector<std::size_t>> received_common_entity_values;
  MPI::all_to_all(mpi_comm, send_common_entity_values,
                  received_common_entity_values);

  // Check if entities received are really entities
  std::vector<std::vector<std::size_t>> send_is_entity_values(num_processes);
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    for (std::size_t i = 0; i < received_common_entity_values[p].size();)
    {
      // Get entity
      const std::size_t entity_size = received_common_entity_values[p][i++];
      Entity entity;
      for (std::size_t j = 0; j < entity_size; ++j)
        entity.push_back(received_common_entity_values[p][i++]);

      // Check if received really is an entity on this process (in
      // which case it will be in owned or unowned entities)
      bool is_entity = false;
      if (unowned_shared_entities.find(entity) != unowned_shared_entities.end()
          || owned_shared_entities.find(entity) != owned_shared_entities.end())
      {
        is_entity = true;
      }

      // Add information about entity (whether it's actually an
      // entity) to send to other processes
      send_is_entity_values[p].push_back(entity_size);
      for (std::size_t j = 0; j < entity_size; ++j)
        send_is_entity_values[p].push_back(entity[j]);
      send_is_entity_values[p].push_back(is_entity);
    }
  }

  // Send data back (list of requested entities that are really
  // entities)
  std::vector<std::vector<std::size_t>> received_is_entity_values;
  MPI::all_to_all(mpi_comm, send_is_entity_values, received_is_entity_values);

  // Create map from entities to processes where it is an entity
  std::map<Entity, std::vector<unsigned int>> entity_processes;
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    for (std::size_t i = 0; i < received_is_entity_values[p].size();)
    {
      const std::size_t entity_size = received_is_entity_values[p][i++];
      Entity entity;
      for (std::size_t j = 0; j < entity_size; ++j)
        entity.push_back(received_is_entity_values[p][i++]);
      const std::size_t is_entity = received_is_entity_values[p][i++];
      if (is_entity == 1)
      {
        // Add entity since it is actually an entity for process p
        entity_processes[entity].push_back(p);
      }
    }
  }

  // Fix the list of entities we do not own (numbered by lower ranked
  // process)
  std::vector<std::vector<std::size_t>> unignore_entities;
  std::map<Entity, EntityData>::iterator entity;
  for (entity = unowned_shared_entities.begin();
       entity != unowned_shared_entities.end(); ++entity)
  {
    const Entity& entity_vertices = entity->first;
    EntityData& entity_data = entity->second;
    const unsigned int local_entity_index = entity_data.local_index;
    if (entity_processes.find(entity_vertices) != entity_processes.end())
    {
      const std::vector<unsigned int>& common_processes
        = entity_processes[entity_vertices];
      dolfin_assert(!common_processes.empty());
      const std::size_t min_proc = *(std::min_element(common_processes.begin(),
                                                      common_processes.end()));

      if (process_number < min_proc)
      {
        // Move from unowned to owned
        owned_shared_entities[entity_vertices] = EntityData(local_entity_index,
                                                            common_processes);

        // Add entity to list of entities that should be removed from
        // the unowned entity list.
        unignore_entities.push_back(entity_vertices);
      }
      else
        entity_data.processes = common_processes;
    }
    else
    {
      // Move from unowned to owned exclusively
      owned_entities.push_back(local_entity_index);

      // Add entity to list of entities that should be removed from the
      // shared but not owned entity list
      unignore_entities.push_back(entity_vertices);
    }
  }

  // Remove unowned shared entities that should not be shared
  for (std::size_t i = 0; i < unignore_entities.size(); ++i)
    unowned_shared_entities.erase(unignore_entities[i]);

  // Fix the list of entities we share
  std::vector<std::vector<std::size_t>> unshare_entities;
  for (std::map<Entity, EntityData>::iterator it
         = owned_shared_entities.begin();
       it != owned_shared_entities.end(); ++it)
  {
    const Entity& e = it->first;
    const unsigned int local_entity_index = it->second.local_index;
    if (entity_processes.find(e) == entity_processes.end())
    {
      // Move from shared to owned elusively
      owned_entities.push_back(local_entity_index);
      unshare_entities.push_back(e);
    }
    else
    {
      // Update processor list of shared entities
      it->second.processes = entity_processes[e];
    }
  }

  // Remove shared entities that should not be shared
  for (std::size_t i = 0; i < unshare_entities.size(); ++i)
    owned_shared_entities.erase(unshare_entities[i]);
}
//-----------------------------------------------------------------------------
bool DistributedMeshTools::is_shared(const Entity& entity,
         const std::map<std::size_t, std::set<unsigned int>>& shared_vertices)
{
  // Iterate over entity vertices
  Entity::const_iterator e;
  for (e = entity.begin(); e != entity.end(); ++e)
  {
    // Return false if an entity vertex is not in the list (map) of
    // shared entities
    if (shared_vertices.find(*e) == shared_vertices.end())
      return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
DistributedMeshTools::compute_num_global_entities(const MPI_Comm mpi_comm,
                                                  std::size_t num_local_entities,
                                                  std::size_t num_processes,
                                                  std::size_t process_number)
{
  // Communicate number of local entities
  std::vector<std::size_t> num_entities_to_number;
  MPI::all_gather(mpi_comm, num_local_entities, num_entities_to_number);

  // Compute offset
  const std::size_t offset
    = std::accumulate(num_entities_to_number.begin(),
                      num_entities_to_number.begin() + process_number,
                      (std::size_t)0);

  // Compute number of global entities
  const std::size_t num_global = std::accumulate(num_entities_to_number.begin(),
                                                 num_entities_to_number.end(),
                                                 (std::size_t)0);

  return {num_global, offset};
}
//-----------------------------------------------------------------------------
void DistributedMeshTools::init_facet_cell_connections(Mesh& mesh)
{
  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Initialize entities of dimension d
  mesh.init(D - 1);

  // Initialise local facet-cell connections.
  mesh.init(D - 1, D);

  // Global numbering
  number_entities(mesh, D - 1);

  // Calculate the number of global cells attached to each facet
  // essentially defining the exterior surface
  // FIXME: should this be done earlier, e.g. at partitioning stage
  // when dual graph is built?

  // Create vector to hold number of cells connected to each
  // facet. Initially copy over from local values.
  std::vector<unsigned int> num_global_neighbors(mesh.num_facets());

  std::map<unsigned int, std::set<unsigned int>>& shared_facets
    = mesh.topology().shared_entities(D - 1);

  // Check if no ghost cells
  if (mesh.topology().ghost_offset(D) == mesh.topology().size(D))
  {
    // Copy local values
    for (FacetIterator f(mesh); !f.end(); ++f)
      num_global_neighbors[f->index()] = f->num_entities(D);

    // All shared facets must have two cells, if no ghost cells
    for (auto f_it = shared_facets.begin();
             f_it != shared_facets.end(); ++f_it)
      num_global_neighbors[f_it->first] = 2;
  }
  else
  {
    // With ghost cells, shared facets may be on an external edge,
    // so need to check connectivity with the cell owner.

    const std::size_t mpi_size = MPI::size(mesh.mpi_comm());
    std::vector<std::vector<std::size_t>> send_facet(mpi_size);
    std::vector<std::vector<std::size_t>> recv_facet(mpi_size);

    // Map shared facets
    std::map<std::size_t, std::size_t> global_to_local_facet;

    for (MeshEntityIterator f(mesh, D - 1, "all"); !f.end(); ++f)
    {
      // Insert shared facets into mapping
      if (f->is_shared())
        global_to_local_facet.insert({f->global_index(), f->index()});
      // Copy local values
      const std::size_t n_cells = f->num_entities(D);
      num_global_neighbors[f->index()] = n_cells;

      if (f->is_ghost() && n_cells == 1)
      {
        // Singly attached ghost facet - check with owner of attached
        // cell
        const Cell c(mesh, f->entities(D)[0]);
        dolfin_assert(c.is_ghost());
        send_facet[c.owner()].push_back(f->global_index());
      }
    }

    MPI::all_to_all(mesh.mpi_comm(), send_facet, recv_facet);

    // Convert received global facet index into number of attached
    // cells and return to sender
    std::vector<std::vector<std::size_t>> send_response(mpi_size);
    for (unsigned int p = 0; p != mpi_size; ++p)
    {
      for (auto r = recv_facet[p].begin(); r != recv_facet[p].end(); ++r)
      {
        auto map_it = global_to_local_facet.find(*r);
        dolfin_assert(map_it != global_to_local_facet.end());
        const Facet local_facet(mesh, map_it->second);
        const std::size_t n_cells = local_facet.num_entities(D);
        send_response[p].push_back(n_cells);
      }
    }

    MPI::all_to_all(mesh.mpi_comm(), send_response, recv_facet);

    // Insert received result into same facet that it came from
    for (unsigned int p = 0; p != mpi_size; ++p)
    {
      for (unsigned int i = 0; i != recv_facet[p].size(); ++i)
      {
        auto f_it = global_to_local_facet.find(send_facet[p][i]);
        dolfin_assert(f_it != global_to_local_facet.end());
        num_global_neighbors[f_it->second] = recv_facet[p][i];
      }
    }
  }

  mesh.topology()(D - 1, D).set_global_size(num_global_neighbors);
}
//-----------------------------------------------------------------------------
std::vector<double>
DistributedMeshTools::reorder_vertices_by_global_indices(const Mesh& mesh)
{
  std::vector<double> ordered_coordinates(mesh.coordinates());
  reorder_values_by_global_indices(mesh, ordered_coordinates,
                                   mesh.geometry().dim());
  return ordered_coordinates;
}
//-----------------------------------------------------------------------------
void DistributedMeshTools::reorder_values_by_global_indices(const Mesh& mesh,
                                      std::vector<double>& data,
                                      const std::size_t width)
{
  Timer t("DistributedMeshTools: reorder vertex values");

  dolfin_assert(mesh.num_vertices()*width == data.size());

  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  // Get shared vertices
  const std::map<unsigned int, std::set<unsigned int>>& shared_vertices
    = mesh.topology().shared_entities(0);

  // My process rank
  const unsigned int mpi_rank = MPI::rank(mpi_comm);

  const std::size_t tdim = mesh.topology().dim();
  std::set<unsigned int> non_local_vertices;
  if (mesh.topology().size(tdim) == mesh.topology().ghost_offset(tdim))
  {
    // No ghost cells - exclude shared entities which are on lower
    // rank processes
    for (auto sh = shared_vertices.begin(); sh != shared_vertices.end(); ++sh)
    {
      const unsigned int lowest_proc = *(sh->second.begin());
      if (lowest_proc < mpi_rank)
        non_local_vertices.insert(sh->first);
    }
  }
  else
  {
    // Iterate through ghost cells, adding non-ghost vertices which
    // are in lower rank process cells to a set for exclusion from
    // output
    for (CellIterator c(mesh, "ghost"); !c.end(); ++c)
    {
      const unsigned int cell_owner = c->owner();
      for (VertexIterator v(*c); !v.end(); ++v)
        if (!v->is_ghost() && cell_owner < mpi_rank)
          non_local_vertices.insert(v->index());
    }
  }

  // Reference to data to send, reorganised as a 2D boost::multi_array
  boost::multi_array_ref<double, 2>
    data_array(data.data(), boost::extents[mesh.num_vertices()][width]);

  std::vector<std::size_t> global_indices;
  std::vector<double> reduced_data;

  // Remove clashing data with multiple copies on different processes
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    const std::size_t vidx = v->index();
    if (non_local_vertices.find(vidx) == non_local_vertices.end())
    {
      global_indices.push_back(v->global_index());
      reduced_data.insert(reduced_data.end(),
                 data_array[vidx].begin(), data_array[vidx].end());
    }
  }

  data = reduced_data;
  reorder_values_by_global_indices(mesh.mpi_comm(), data,
                                   width, global_indices);
}
//-----------------------------------------------------------------------------
void DistributedMeshTools::reorder_values_by_global_indices(MPI_Comm mpi_comm,
                           std::vector<double>& values,
                           const std::size_t width,
                           const std::vector<std::size_t>& global_indices)
{

  // Number of items to redistribute
  const std::size_t num_local_indices = global_indices.size();
  dolfin_assert(width*num_local_indices == values.size());

  boost::multi_array_ref<double, 2> vertex_array(values.data(),
                      boost::extents[num_local_indices][width]);

  // Calculate size of overall global vector by finding max index value
  // anywhere
  const std::size_t global_vector_size
    = MPI::max(mpi_comm, *std::max_element(global_indices.begin(),
                                           global_indices.end())) + 1;

  // Send unwanted values off process
  const std::size_t mpi_size = MPI::size(mpi_comm);
  std::vector<std::vector<std::size_t>> values_to_send0(mpi_size);
  std::vector<std::vector<double>> values_to_send1(mpi_size);

  // Go through local vector and append value to the appropriate list
  // to send to correct process
  for (std::size_t i = 0; i != num_local_indices; ++i)
  {
    const std::size_t global_i = global_indices[i];
    const std::size_t process_i
      = MPI::index_owner(mpi_comm, global_i, global_vector_size);
    values_to_send0[process_i].push_back(global_i);
    values_to_send1[process_i].insert(values_to_send1[process_i].end(),
                                      vertex_array[i].begin(),
                                      vertex_array[i].end());
  }

  // Redistribute the values to the appropriate process - including
  // self All values are "in the air" at this point, so local vector
  // can be cleared
  std::vector<std::vector<std::size_t>> received_values0;
  std::vector<std::vector<double>> received_values1;
  MPI::all_to_all(mpi_comm, values_to_send0, received_values0);
  MPI::all_to_all(mpi_comm, values_to_send1, received_values1);

  // When receiving, just go through all received values and place
  // them in the local partition of the global vector.
  const std::pair<std::size_t, std::size_t> range
    = MPI::local_range(mpi_comm, global_vector_size);
  values.resize((range.second - range.first)*width);
  boost::multi_array_ref<double, 2>
    new_vertex_array(values.data(),
                     boost::extents[range.second - range.first][width]);

  for (std::size_t p = 0; p != received_values0.size(); ++p)
  {
    const std::vector<std::size_t>& received_global_data0
      = received_values0[p];
    const std::vector<double>& received_global_data1 = received_values1[p];
    for (std::size_t j = 0; j != received_global_data0.size(); ++j)
    {
      const std::size_t global_i = received_global_data0[j];
      dolfin_assert(global_i >= range.first && global_i < range.second);
      std::copy(received_global_data1.begin() + j*width,
                received_global_data1.begin() + (j + 1)*width,
                new_vertex_array[global_i - range.first].begin());
    }
  }
}
//-----------------------------------------------------------------------------
