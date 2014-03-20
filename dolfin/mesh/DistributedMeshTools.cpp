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
// Last changed: 2014-02-14

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
std::vector<std::size_t> DistributedMeshTools::entity_key(const MeshEntity& e)
{
  std::vector<std::size_t> e_key;
  for (VertexIterator v(e); !v.end(); ++v)
    e_key.push_back(v->global_index());
  std::sort(e_key.begin(), e_key.end());
  return e_key;
}
//-----------------------------------------------------------------------------
void DistributedMeshTools::ghost_number_entities(const Mesh& mesh,
                                                 std::size_t d)
{
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const unsigned int tdim = mesh.topology().dim();

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

  // Cells which are also on other processes
  const std::map<unsigned int, std::set<unsigned int> >& shared_cells 
    = mesh.topology().shared_entities(tdim);
  dolfin_assert(!shared_cells.empty());

  // Ownership of cells
  const std::vector<std::size_t>& ghost_owner
    = mesh.data().array("ghost_owner", tdim);
  dolfin_assert(!ghost_owner.empty());

  const unsigned int mpi_rank = MPI::rank(mesh.mpi_comm());
  const unsigned int mpi_size = MPI::size(mesh.mpi_comm());

  // Communicate shared entities
  std::vector<std::vector<std::size_t> > send_numbered_entities(mpi_size);
  std::vector<std::vector<std::size_t> > recv_numbered_entities(mpi_size);
  std::vector<std::vector<std::size_t> > send_unnumbered_entities(mpi_size);
  std::vector<std::vector<std::size_t> > recv_unnumbered_entities(mpi_size);

  // Map from entity global vertices to local entity index 
  // which will be used later to look up received entities
  std::map<std::vector<std::size_t>, unsigned int> non_local_entity_map;

  // Number all local entities with global indices 
  // (leaving other entries blank)
  const std::size_t num_global_vertices = mesh.topology().size_global(0);
  const std::size_t num_local_entities = mesh.topology().size(d);
  std::vector<std::size_t> global_entity_indices(num_local_entities);

  // Initially index from zero, add offset later
  std::size_t ecount = 0;
  for (MeshEntityIterator e(mesh, d); !e.end(); ++e)
  {
    // Get cell ownership around entity
    dolfin::Set<unsigned int> cell_owner;
    for (CellIterator c(*e); !c.end(); ++c)
      cell_owner.insert(ghost_owner[c->index()]);

    if (cell_owner.size() == 1 && cell_owner[0] == mpi_rank)
      global_entity_indices[e->index()] = ecount++;
    else
    {
      // Non-local entity - save to map,
      // ready to receive from another process
      non_local_entity_map.insert(std::make_pair
                                  (entity_key(*e), e->index()));
    }
  }

  // Add local offset to all numbered indices
  const unsigned int local_offset = MPI::global_offset(mpi_comm, ecount, true);
  const unsigned int sum_local_numbered = MPI::sum(mpi_comm, ecount);

  // FIXME: use std::transform?
  for (auto it = global_entity_indices.begin(); 
       it != global_entity_indices.end(); ++it)
    *it += local_offset;

  // Send shared entities to matching process based on 
  // MPI::index_owner of first vertex
  for (MeshEntityIterator e(mesh, d); !e.end(); ++e)
  {
    // Get set of cell sharing processes around entity
    dolfin::Set<std::size_t> cell_share;
    for (CellIterator c(*e); !c.end(); ++c)
    {
      auto map_it = shared_cells.find(c->index());
      if (map_it != shared_cells.end())
        cell_share.insert(map_it->second.begin(), map_it->second.end());
    }
    if (cell_share.size() > 0)
    {
      const std::vector<std::size_t> e_key = entity_key(*e);
      unsigned int dest = MPI::index_owner(mpi_comm,
                                           e_key[0],
                                           num_global_vertices);
      auto map_it = non_local_entity_map.find(e_key);
      if (map_it == non_local_entity_map.end())
      {
        // Already numbered locally, but needed remotely
        std::vector<std::size_t>& send_dest = send_numbered_entities[dest];
        send_dest.push_back(global_entity_indices[e->index()]);
        send_dest.insert(send_dest.end(), e_key.begin(), e_key.end());
      }
      else
      {
        // Unknown numbering, may be numbered elsewhere
        // or globally unnumbered
        std::vector<std::size_t>& send_dest = send_unnumbered_entities[dest];   
        send_dest.insert(send_dest.end(), e_key.begin(), e_key.end());
      }    
    }
  }

  MPI::all_to_all(mpi_comm, 
                  send_numbered_entities, recv_numbered_entities);
  MPI::all_to_all(mpi_comm, 
                  send_unnumbered_entities, recv_unnumbered_entities);

  const CellType& cell_type = mesh.type();
  const std::size_t num_entity_vertices = cell_type.num_vertices(d);  

  // Collect incoming numbered entities into a map 
  std::map<std::vector<std::size_t>, std::size_t> shared_entity_numbering;
  for (auto p = recv_numbered_entities.begin(); 
       p != recv_numbered_entities.end(); ++p)
  {
    for (auto q = p->begin(); q != p->end(); q += num_entity_vertices + 1)
    {
      const std::vector<std::size_t> qvec(q + 1, q + 1 + num_entity_vertices);
      dolfin_assert(shared_entity_numbering.find(qvec) 
                    == shared_entity_numbering.end());
      shared_entity_numbering.insert(std::make_pair(qvec, *q));
    }
  }

  // Reset counter and number remaining unclaimed entities in a new
  // map, which will be merged with the main map after adding an
  // offset
  ecount = 0;
  std::map <std::vector<std::size_t>, std::size_t> new_shared_numbering;
  for (auto p = recv_unnumbered_entities.begin(); 
       p != recv_unnumbered_entities.end(); ++p)
  {
    for (auto q = p->begin(); q != p->end(); q += num_entity_vertices)
    {
      const std::vector<std::size_t> qvec(q, q + num_entity_vertices);
      auto map_it = shared_entity_numbering.find(qvec);
      if (map_it == shared_entity_numbering.end())
      {
        // If not already in map, insert and increment counter
        if(new_shared_numbering.insert(std::make_pair(qvec, ecount)).second)
          ecount++;
      }
    }
  }

  const unsigned int map_offset = sum_local_numbered 
    + MPI::global_offset(mpi_comm, ecount, true);
  const std::size_t num_global_entities = sum_local_numbered 
    + MPI::sum(mpi_comm, ecount);
  
  // Add offset to new entity indices and merge into main map
  for (auto map_it = new_shared_numbering.begin(); 
       map_it != new_shared_numbering.end(); ++map_it)
  {
    map_it->second += map_offset;
    if(shared_entity_numbering.insert(*map_it).second == false)
    {
      dolfin_error("DistributedMeshTools.cpp",
                   "number entity",
                   "Index clash");
    }
  }
  
  // Get ready to send entity numbering back to requesting processes
  std::vector<std::vector<std::size_t> > send_global_idx(mpi_size);
  std::vector<std::vector<std::size_t> > recv_global_idx(mpi_size);

  // For each received unnumbered entity, reflect back the global index
  for (unsigned int i = 0; i != mpi_size; ++i)
  {
    for (auto q = recv_unnumbered_entities[i].begin(); 
         q != recv_unnumbered_entities[i].end(); q += num_entity_vertices)
    {
      std::vector<std::size_t> qvec(q, q + num_entity_vertices);
      auto map_it = shared_entity_numbering.find(qvec);
      dolfin_assert(map_it != shared_entity_numbering.end());
      send_global_idx[i].push_back(map_it->second);
    }
  }

  MPI::all_to_all(mpi_comm, send_global_idx, recv_global_idx);

  // Match up incoming global indices with entries in the
  // non_local_entity_map to place in correct local entry
  for (unsigned int i = 0; i != mpi_size; ++i)
  {
    unsigned int j = 0;
    dolfin_assert(send_unnumbered_entities[i].size() 
                  == recv_global_idx[i].size() * num_entity_vertices);
    
    for (auto q = send_unnumbered_entities[i].begin(); 
         q != send_unnumbered_entities[i].end(); q += num_entity_vertices)
    {
      std::vector<std::size_t> qvec(q, q + num_entity_vertices);
      auto map_it = non_local_entity_map.find(qvec);
      dolfin_assert(map_it != non_local_entity_map.end());
      global_entity_indices[map_it->second] = recv_global_idx[i][j++];
      non_local_entity_map.erase(map_it);
    }
  }

  // Check all entities have been numbered
  dolfin_assert(non_local_entity_map.size() == 0);

  // Set global entity numbers in mesh
  _mesh.topology().init(d, mesh.num_entities(d), num_global_entities);
  _mesh.topology().init_global_indices(d, global_entity_indices.size());
  for (std::size_t i = 0; i < global_entity_indices.size(); ++i)
    _mesh.topology().set_global_index(d, i, global_entity_indices[i]);

}
//-----------------------------------------------------------------------------
void DistributedMeshTools::number_entities(const Mesh& mesh, std::size_t d)
{
  Timer timer("Build mesh number mesh entities");

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
  std::map<unsigned int, std::set<unsigned int> >&
    shared_entities = _mesh.topology().shared_entities(d);

  // Number entities
  std::vector<std::size_t> global_entity_indices;
  const std::map<unsigned int, std::pair<unsigned int, unsigned int> >
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
  unsigned int> >& slave_entities,
  std::vector<std::size_t>& global_entity_indices,
  std::map<unsigned int, std::set<unsigned int> >& shared_entities,
  std::size_t d)
{
  // Developer note: This function should use global_vertex_indices for
  // the global mesh indices and *not* access these through the mesh. In
  // some cases special numbering is passed in which differs from mesh
  // global numbering, e.g. when computing mesh entity numbering for
  // problems with periodic boundary conditions.

  Timer timer("PARALLEL x: Number mesh entities");

  // Check that we're not re-numbering vertices (these are fixed at
  // mesh construction)
  if (d == 0)
  {
    dolfin_error("MeshPartitioning.cpp",
                 "number mesh entities",
                 "Gloval vertex indices exist at input. Cannot be renumbered");
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
                 "Global cells indices exist at input. Cannot be renumbered");
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
                                   unsigned int> >::const_iterator s;
  for (s = slave_entities.begin(); s != slave_entities.end(); ++s)
    exclude[s->first] = true;

  // Build entity global [vertex list]-to-[local entity index]
  // map. Exclude any slave entities.
  std::map<std::vector<std::size_t>, unsigned int> entities;
  std::pair<std::vector<std::size_t>, unsigned int> entity;
  for (MeshEntityIterator e(mesh, d); !e.end(); ++e)
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
  const std::map<unsigned int, std::set<unsigned int> >& shared_vertices_local
                            = mesh.topology().shared_entities(0);

  // Compute ownership of entities of dimension d ([entity vertices], data):
  //  [0]: owned and shared (will be numbered by this process, and number
  //       communicated to other processes)
  //  [1]: not owned but shared (will be numbered by another process, and number
  //       communicated to this processes)
  boost::array<std::map<Entity, EntityData>, 2> entity_ownership;
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
  global_entity_indices = std::vector<std::size_t>(mesh.size(d),
                               std::numeric_limits<std::size_t>::max());

  std::map<Entity, EntityData>::const_iterator it;

  // Number exlusively owned entities
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
  std::vector<std::vector<std::size_t> > send_values(num_processes);
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
  std::vector<std::vector<std::size_t> > received_values;
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
    std::vector<std::vector<std::size_t> >
      slave_send_buffer(MPI::size(mpi_comm));
    std::vector<std::vector<std::size_t> >
      local_slave_index(MPI::size(mpi_comm));
    for (s = slave_entities.begin(); s != slave_entities.end(); ++s)
    {
      // Local index on remote process
      slave_send_buffer[s->second.first].push_back(s->second.second);

      // Local index on this
      local_slave_index[s->second.first].push_back(s->first);
    }
    std::vector<std::vector<std::size_t> > slave_receive_buffer;
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
std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t> > >
DistributedMeshTools::locate_off_process_entities(const std::vector<std::size_t>& entity_indices,
                                                  std::size_t dim,
                                                  const Mesh& mesh)
{
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
  std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t> > >
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

    // FIXME: This can be made more efficient by exploiting fact that
    //        set is sorted
    // Remove local cells from set_of_my_entities to reduce communication
    for (std::size_t j = 0; j < global_entity_indices.size(); ++j)
      set_of_my_entities.erase(global_entity_indices[j]);

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
      processes[global_index].insert(std::make_pair(dest, local_index));
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
boost::unordered_map<unsigned int, std::vector<std::pair<unsigned int, unsigned int> > >
  DistributedMeshTools::compute_shared_entities(const Mesh& mesh, std::size_t d)
{
  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const unsigned int comm_size = MPI::size(mpi_comm);

  // Return empty set if running in serial
  if (MPI::size(mpi_comm) == 1)
  {
    return boost::unordered_map<unsigned int, std::vector<std::pair<unsigned int, unsigned int> > >();
  }

  // Initialize entities of dimension d
  mesh.init(d);

  // Number entities (globally)
  number_entities(mesh, d);

  // Get shared entities to processes map
  const std::map<unsigned int, std::set<unsigned int> >&
    shared_entities = mesh.topology().shared_entities(d);

  // Get local-to-global indices map
  const std::vector<std::size_t>& global_indices_map
    = mesh.topology().global_indices(d);

  // Global-to-local map for each process
  boost::unordered_map<std::size_t, boost::unordered_map<std::size_t, std::size_t> > global_to_local;

  // Pack global indices for sending to sharing processes
  std::vector<std::vector<std::size_t> > send_indices(comm_size);
  std::vector<std::vector<std::size_t> > local_sent_indices(comm_size);
  std::map<unsigned int, std::set<unsigned int> >::const_iterator shared_entity;
  for (shared_entity = shared_entities.begin();
       shared_entity != shared_entities.end(); ++shared_entity)
  {
    // Local index
    const unsigned int local_index = shared_entity->first;

    // Global index
    dolfin_assert(local_index < global_indices_map.size());
    std::size_t global_index = global_indices_map[local_index];

    // Destinarion process
    const std::set<unsigned int>& sharing_processes = shared_entity->second;

    // Pack data for sending and build global-to-local map
    std::set<unsigned int>::const_iterator dest;
    for (dest = sharing_processes.begin(); dest != sharing_processes.end();
         ++dest)
    {
      send_indices[*dest].push_back(global_index);
      local_sent_indices[*dest].push_back(local_index);
      global_to_local[*dest].insert(std::make_pair(global_index, local_index));
    }
  }

  std::vector<std::vector<std::size_t> > recv_entities;
  MPI::all_to_all(mpi_comm, send_indices, recv_entities);

  // Clear send data
  send_indices.clear();
  send_indices.resize(comm_size);

  // Determine local entities indices for received global entity indices
  boost::unordered_map<std::size_t, std::vector<std::size_t> >::const_iterator
    received_global_indices;
  for (std::size_t p = 0; p < recv_entities.size(); ++p)
  {
    // Get process number of neighbour
    const std::size_t sending_proc = p;

    if (recv_entities[p].size() > 0)
    {
      // Get global-to-local map for neighbour process
      boost::unordered_map<std::size_t, boost::unordered_map<std::size_t, std::size_t> >::const_iterator
        it = global_to_local.find(sending_proc);
      dolfin_assert(it != global_to_local.end());
      const boost::unordered_map<std::size_t, std::size_t>&
        neighbour_global_to_local = it->second;

      // Build vector of local indices
      const std::vector<std::size_t>& global_indices_recv
        = recv_entities[p];
      for (std::size_t i = 0; i < global_indices_recv.size(); ++i)
      {
        // Global index
        const std::size_t global_index = global_indices_recv[i];

        // Find local index corresponding to global index
        boost::unordered_map<std::size_t, std::size_t>::const_iterator
          n_global_to_local = neighbour_global_to_local.find(global_index);

        dolfin_assert(n_global_to_local != neighbour_global_to_local.end());
        const std::size_t my_local_index = n_global_to_local->second;
        send_indices[sending_proc].push_back(my_local_index);
      }
    }
  }

  MPI::all_to_all(mpi_comm, send_indices, recv_entities);

  // Build map
  boost::unordered_map<unsigned int, std::vector<std::pair<unsigned int, unsigned int> > >
    shared_local_indices_map;

  // Loop over data received from each process
  boost::unordered_map<std::size_t, std::vector<std::size_t> >::const_iterator
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
        shared_local_indices_map[my_local_indices[i]].push_back(std::make_pair(proc,
                                                                               neighbour_local_indices[i]));
      }
    }
  }

  return shared_local_indices_map;
}
//-----------------------------------------------------------------------------
void DistributedMeshTools::compute_entity_ownership(
  const MPI_Comm mpi_comm,
  const std::map<std::vector<std::size_t>, unsigned int>& entities,
  const std::map<unsigned int, std::set<unsigned int> >& shared_vertices_local,
  const std::vector<std::size_t>& global_vertex_indices,
  std::size_t d,
  std::vector<std::size_t>& owned_entities,
  boost::array<std::map<Entity, EntityData>, 2>& shared_entities)
{
  // Build global-to-local indices map for shared vertices
  std::map<std::size_t, std::set<unsigned int> > shared_vertices;
  std::map<unsigned int, std::set<unsigned int> >::const_iterator v;
  for (v = shared_vertices_local.begin(); v != shared_vertices_local.end(); ++v)
  {
    dolfin_assert(v->first < global_vertex_indices.size());
    shared_vertices.insert(std::make_pair(global_vertex_indices[v->first],
                                          v->second));
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
  const std::map<std::size_t, std::set<unsigned int> >& shared_vertices,
  const std::map<Entity, unsigned int>& entities,
  std::vector<std::size_t>& owned_entities,
  boost::array<std::map<Entity, EntityData>, 2>& shared_entities)
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
  boost::array<std::map<Entity, EntityData>, 2>& shared_entities)
{
  // Entities ([entity vertices], index) to be numbered
  std::map<Entity, EntityData>& owned_shared_entities = shared_entities[0];
  std::map<Entity, EntityData>& unowned_shared_entities = shared_entities[1];

  // Get MPI number of processes and process number
  const std::size_t num_processes = MPI::size(mpi_comm);
  const std::size_t process_number = MPI::rank(mpi_comm);

  // Communicate common entities, starting with the entities we think
  // are shared but not owned
  std::vector<std::vector<std::size_t> >
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
  std::vector<std::vector<std::size_t> > received_common_entity_values;
  MPI::all_to_all(mpi_comm, send_common_entity_values,
                  received_common_entity_values);

  // Check if entities received are really entities
  std::vector<std::vector<std::size_t> > send_is_entity_values(num_processes);
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    for (std::size_t i = 0; i < received_common_entity_values[p].size();)
    {
      // Get entity
      const std::size_t entity_size = received_common_entity_values[p][i++];
      Entity entity;
      for (std::size_t j = 0; j < entity_size; ++j)
        entity.push_back(received_common_entity_values[p][i++]);

      // Check if received really is an entity on this process (in which
      // case it will be in owned or unowned entities)
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

  // Send data back (list of requested entities that are really entities)
  std::vector<std::vector<std::size_t> > received_is_entity_values;
  MPI::all_to_all(mpi_comm, send_is_entity_values, received_is_entity_values);

  // Create map from entities to processes where it is an entity
  std::map<Entity, std::vector<unsigned int> > entity_processes;
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
  std::vector<std::vector<std::size_t> > unignore_entities;
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
  std::vector<std::vector<std::size_t> > unshare_entities;
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
         const std::map<std::size_t, std::set<unsigned int> >& shared_vertices)
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
                      num_entities_to_number.begin() + process_number, 0);

  // Compute number of global entities
  const std::size_t num_global = std::accumulate(num_entities_to_number.begin(),
                                                 num_entities_to_number.end(),
                                                 0);

  return std::make_pair(num_global, offset);
}
//-----------------------------------------------------------------------------
void DistributedMeshTools::init_facet_cell_connections_by_ghost(Mesh& mesh)
{
  Timer timer("Connect and number facets (parallel)");

  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Get mesh topology and connectivity
  // cell-facet, facet-cell and facet-vertex
  MeshTopology& topology = mesh.topology();
  MeshConnectivity& cf = topology(D, D - 1);
  MeshConnectivity& fc = topology(D - 1, D);
  MeshConnectivity& fv = topology(D - 1, 0);
  
  // Check if entities have already been computed
  dolfin_assert(topology.size(D - 1) == 0);

  // Connectivity should not already exist  
  dolfin_assert(cf.empty() && fv.empty() && fc.empty());

  const CellType& cell_type = mesh.type();

  // Initialize local array of entities
  const std::size_t m = cell_type.num_entities(D - 1);
  const std::size_t num_facet_vertices = cell_type.num_vertices(D - 1);
  std::vector<std::vector<unsigned int> >
    f_vertices(m, std::vector<unsigned int>(num_facet_vertices, 0));

  // List of facet indices connected to cell
  std::vector<std::vector<unsigned int> > connectivity_cf(mesh.num_cells());

  // List of vertex indices connected to facet
  std::vector<std::vector<unsigned int> > connectivity_fv;

  // List of cell indices connected to facet
  std::vector<std::vector<unsigned int> > connectivity_fc;

  std::size_t current_facet = 0;
  // At present, there is a maximum of four facets per cell
  std::size_t max_cf_connections = 4;

  // Local indexing of facets
  std::map<std::vector<unsigned int>, unsigned int> fvertices_to_index;

  // Additionally calculate cell-cell connectivity via facet
  // for local cell reordering
  Graph g_dual(mesh.num_cells());

  // Loop over cells
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    // Cell index
    const std::size_t cell_index = c->index();
    connectivity_cf[cell_index].reserve(max_cf_connections);

    // Get vertices from cell
    const unsigned int* vertices = c->entities(0);
    dolfin_assert(vertices);

    // Create facets
    cell_type.create_entities(f_vertices, D - 1, vertices);

    // Iterate over the given list of facets
    std::vector<std::vector<unsigned int> >::iterator facet;
    for (facet = f_vertices.begin(); facet != f_vertices.end(); ++facet)
    {
      // Sort (to use as map key)
      std::sort(facet->begin(), facet->end());

      // Insert into map
      std::pair<std::map<std::vector<unsigned int>, 
                         unsigned int>::iterator, bool>
        it = fvertices_to_index.insert(std::make_pair(*facet, current_facet));

      // Facet index
      std::size_t f_index = it.first->second;

      // Add facet index to cell-facet connectivity
      connectivity_cf[cell_index].push_back(f_index);

      // If new key was inserted, increment counter
      if (it.second)
      {
        // Add list of new entity vertices
        connectivity_fv.push_back(*facet);
        connectivity_fc.push_back(std::vector<unsigned int>(1, cell_index));

        // Increase counter
        current_facet++;
      }
      else
      {
        // Second cell connected to this facet - erase map entry
        fvertices_to_index.erase(it.first);
        std::vector<unsigned int>& cfc = connectivity_fc[f_index];
        const unsigned int other_cell = cfc[0];
        cfc.push_back(cell_index);
        g_dual[cell_index].insert(other_cell);
        g_dual[other_cell].insert(cell_index);
      }
    }
  }

  // Initialise connectivity data structure
  topology.init(D - 1, connectivity_fv.size(), connectivity_fv.size());

  // Copy connectivity data into MeshTopology data structures
  cf.set(connectivity_cf);
  fc.set(connectivity_fc);
  fv.set(connectivity_fv);

  ghost_number_entities(mesh, D - 1);  
}
//-----------------------------------------------------------------------------
bool DistributedMeshTools::is_local_facet(
      unsigned int mpi_rank,
      const std::vector<std::size_t>& vertices,
      const std::vector<std::size_t>& cell_owner)
{
  if (cell_owner.size() == 1)
  { 
    if (cell_owner[0] == mpi_rank)
      // Local external facet
      return true;
    else
      // Non-local external facet
      return false;
  }
  
  dolfin_assert(cell_owner.size() == 2);

  if (cell_owner[0] == mpi_rank
      && cell_owner[1] == mpi_rank)
    // Local internal facet
    return true;

  if (cell_owner[0] != mpi_rank
      && cell_owner[1] != mpi_rank)
    // Non-local internal facet
    return false;
  
  // Make decision here about facets shared with other processes.
  // Must be consistent across processes.

  const unsigned int other_proc 
    = (cell_owner[0] == mpi_rank) ? cell_owner[1] : cell_owner[0];
  
  // FIXME: example switch based on first vertex even or odd.
  bool vswitch = (vertices[0]%2 == 0);

  if (other_proc > mpi_rank) 
    return vswitch;
  else
    return not vswitch;
  
  return false;
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

  // Build entity(vertex list)-to-local-vertex-index map
  std::map<std::vector<std::size_t>, unsigned int> entities;
  for (MeshEntityIterator e(mesh, D - 1); !e.end(); ++e)
  {
    std::vector<std::size_t> entity;
    for (VertexIterator vertex(*e); !vertex.end(); ++vertex)
      entity.push_back(vertex->global_index());
    std::sort(entity.begin(), entity.end());
    entities[entity] = e->index();
  }

  // Get shared vertices (local index, [sharing processes])
  const std::map<unsigned int, std::set<unsigned int> >& shared_vertices_local
                            = mesh.topology().shared_entities(0);
  const std::vector<std::size_t>& global_vertex_indices
      = mesh.topology().global_indices(0);

  // Compute ownership of entities ([entity vertices], data):
  //  [0]: owned and shared (will be numbered by this process, and number
  //       communicated to other processes)
  //  [1]: not owned but shared (will be numbered by another process, and number
  //       communicated to this processes)
  std::vector<std::size_t> owned_entities;
  boost::array<std::map<Entity, EntityData>, 2> entity_ownership;
  compute_entity_ownership(mesh.mpi_comm(), entities, shared_vertices_local,
                           global_vertex_indices, D - 1, owned_entities,
                           entity_ownership);

  // Split ownership for convenience
  const std::map<Entity, EntityData>& owned_shared_entities
    = entity_ownership[0];
  const std::map<Entity, EntityData>& unowned_shared_entities
    = entity_ownership[1];

  // Create vector to hold number of cells connected to each
  // facet. Assume facet is internal, then modify for external facets.
  std::vector<unsigned int> num_global_neighbors(mesh.num_facets(), 2);

  // Add facets that are locally connected to one cell only
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    if (facet->num_entities(D) == 1)
      num_global_neighbors[facet->index()] = 1;
  }

  // Handle facets on internal partition boundaries
  std::map<Entity, EntityData>::const_iterator it;

  for (it = owned_shared_entities.begin();
       it != owned_shared_entities.end(); ++it)
  {
    num_global_neighbors[entities.find(it->first)->second] = 2;
  }

  for (it = unowned_shared_entities.begin();
       it != unowned_shared_entities.end(); ++it)
  {
    num_global_neighbors[entities.find(it->first)->second] = 2;
  }
  mesh.topology()(D - 1, mesh.topology().dim()).set_global_size(num_global_neighbors);
}
//-----------------------------------------------------------------------------
