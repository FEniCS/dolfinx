// Copyright (C) 2008-2009 Niclas Jansson, Ola Skavhaug and Anders Logg
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
// Modified by Kent-Andre Mardal 2011
// Modified by Anders Logg 2011
//
// First added:  2008-12-01
// Last changed: 2011-11-14

#include <algorithm>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <boost/multi_array.hpp>

#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/graph/ParMETIS.h>
#include <dolfin/graph/SCOTCH.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "BoundaryMesh.h"
#include "Facet.h"
#include "LocalMeshData.h"
#include "Mesh.h"
#include "MeshDistributed.h"
#include "MeshEditor.h"
#include "MeshEntityIterator.h"
#include "MeshFunction.h"
#include "MeshTopology.h"
#include "MeshValueCollection.h"
#include "Point.h"
#include "Vertex.h"
#include "MeshPartitioning.h"


using namespace dolfin;

// Explicitly instantiate some templated functions to help the Python
// wrappers
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<std::size_t, unsigned int>, std::size_t> >& local_value_data,
   MeshValueCollection<std::size_t>& mesh_values);
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<std::size_t, unsigned int>, int> >& local_value_data,
   MeshValueCollection<int>& mesh_values);
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<std::size_t, unsigned int>, double> >& local_value_data,
   MeshValueCollection<double>& mesh_values);
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<std::size_t, unsigned int>, bool> >& local_value_data,
   MeshValueCollection<bool>& mesh_values);

//-----------------------------------------------------------------------------
void MeshPartitioning::build_distributed_mesh(Mesh& mesh)
{
  if (MPI::num_processes() > 1)
  {
    // Create and distribute local mesh data
    LocalMeshData local_mesh_data(mesh);

    // Build distributed mesh
    build_distributed_mesh(mesh, local_mesh_data);
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_distributed_mesh(Mesh& mesh,
                                              const LocalMeshData& local_data)
{
  // Partition mesh
  partition(mesh, local_data);

  // Create MeshDomains from local_data
  build_mesh_domains(mesh, local_data);

  // Number facets (see https://bugs.launchpad.net/dolfin/+bug/733834)
  number_entities(mesh, mesh.topology().dim() - 1);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::number_entities(const Mesh& _mesh, uint d)
{
  // FIXME: Break up this function

  Timer timer("PARALLEL x: Number mesh entities");
  Mesh& mesh = const_cast<Mesh&>(_mesh);

  // Check for vertices
  if (d == 0 && mesh.topology().dim() > 1)
  {
    dolfin_error("MeshPartitioning.cpp",
                 "number mesh entities",
                 "Vertex indices do not exist; need vertices to number entities of dimension 0");
  }

  // Return if global entity indices are already calculated (proceed if
  // d is a facet because facets will be marked)
  if (d != (mesh.topology().dim() - 1) && mesh.topology().have_global_indices(d))
    return;

  // Get number of processes and process number
  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Initialize entities of dimension d
  mesh.init(d);

  // Get shared vertices (global index, [sharing processes])
  std::map<std::size_t, std::set<uint> >& shared_vertices
                            = mesh.topology().shared_entities(0);

  // Build entity(vertex list)-to-global-vertex-index map
  std::map<std::vector<std::size_t>, std::size_t> entities;
  for (MeshEntityIterator e(mesh, d); !e.end(); ++e)
  {
    std::vector<std::size_t> entity;
    for (VertexIterator vertex(*e); !vertex.end(); ++vertex)
      entity.push_back(vertex->global_index());
    std::sort(entity.begin(), entity.end());
    entities[entity] = e->index();
  }

  // Entities ([entity vertices], index) to be numbered
  std::map<Entity, EntityData> owned_exclusive_entities;

  // Candidate entities ([entity vertices], index) to number and send
  // number to other, higher rank processes
  std::map<Entity, EntityData> owned_shared_entities;

  // Candidates entities ([entity vertices], index) to being numbered
  // by another, lower ranked process. We need to check that the entity
  // is really an entity at the other process. If not, we must number
  // it ourself
  std::map<Entity, EntityData> unowned_shared_entities;

  // Compute ownership of entities
  compute_entity_ownership(entities, shared_vertices,
                           owned_exclusive_entities,
                           owned_shared_entities,
                           unowned_shared_entities);

  // ---- break here

  /// --- Mark exterior facets

  // Create mesh markers for exterior facets
  if (d == (mesh.topology().dim() - 1))
  {
    std::vector<unsigned int> _num_connected_cells
      = num_connected_cells(mesh, entities, owned_shared_entities,
                            unowned_shared_entities);
    mesh.topology()(d, mesh.topology().dim()).set_global_size(_num_connected_cells);
  }

  // Compute global number of entities and process offset
  const std::size_t num_local_entities = owned_exclusive_entities.size()
                                        + owned_shared_entities.size();
  const std::pair<std::size_t, std::size_t> num_global_entities
      = compute_num_global_entities(num_local_entities, num_processes,
                                    process_number);

  // Store number of global entities
  mesh.topology().init_global(d, num_global_entities.first);

  // Extract offset
  std::size_t offset = num_global_entities.second;

  // Return if global entity indices are already calculated
  if (mesh.topology().have_global_indices(d))
    return;

  /// ---- Numbering

  // Prepare list of entity numbers. Check later that nothing is -1
  std::vector<int> entity_indices(mesh.size(d), -1);

  std::map<Entity, EntityData>::const_iterator it;

  // Number exlusively owned entities
  for (it = owned_exclusive_entities.begin(); it != owned_exclusive_entities.end(); ++it)
    entity_indices[it->second.index] = offset++;

  // Number shared entities
  std::map<Entity, EntityData>::const_iterator it1;
  for (it1 = owned_shared_entities.begin(); it1 != owned_shared_entities.end(); ++it1)
    entity_indices[it1->second.index] = offset++;

  // Communicate indices for shared entities and get indices for ignored
  std::vector<std::size_t> send_values;
  std::vector<uint> destinations;
  for (it1 = owned_shared_entities.begin(); it1 != owned_shared_entities.end(); ++it1)
  {
    // Get entity index
    const std::size_t local_entity_index = it1->second.index;
    const int entity_index = entity_indices[local_entity_index];
    dolfin_assert(entity_index != -1);

    // Get entity vertices (global vertex indices)
    const Entity& entity = it1->first;

    // Get entity processes (processes sharing the entity)
    const std::vector<unsigned int>& entity_processes = it1->second.processes;

    // Prepare data for sending
    for (std::size_t j = 0; j < entity_processes.size(); ++j)
    {
      // Store interleaved: entity index, number of vertices, global
      // vertex indices
      send_values.push_back(entity_index);
      send_values.push_back(entity.size());
      send_values.insert(send_values.end(), entity.begin(), entity.end());
      destinations.insert(destinations.end(), entity.size() + 2, entity_processes[j]);
    }
  }

  // Send data
  std::vector<std::size_t> received_values;
  std::vector<uint> sources;
  MPI::distribute(send_values, destinations, received_values, sources);

  // Fill in global entity indices recieved from lower ranked processes
  for (std::size_t i = 0; i < received_values.size();)
  {
    const uint p = sources[i];
    const std::size_t global_index = received_values[i++];
    const std::size_t entity_size = received_values[i++];
    Entity entity;
    for (std::size_t j = 0; j < entity_size; ++j)
      entity.push_back(received_values[i++]);

    // Sanity check, should not receive an entity we don't need
    if (unowned_shared_entities.find(entity) == unowned_shared_entities.end())
    {
      std::stringstream msg;
      msg << "Process " << MPI::process_number() << " received illegal entity given by ";
      msg << " with global index " << global_index;
      msg << " from process " << p;
      dolfin_error("MeshPartitioning.cpp",
                   "number mesh entities",
                   msg.str());
    }

    const std::size_t local_entity_index
      = unowned_shared_entities.find(entity)->second.index;
    dolfin_assert(entity_indices[local_entity_index] == -1);
    entity_indices[local_entity_index] = global_index;
  }

  // Create mesh data
  mesh.topology().init_global_indices(d, entity_indices.size());
  for (std::size_t i = 0; i < entity_indices.size(); ++i)
  {
    if (entity_indices[i] < 0)
      log(WARNING, "Missing global number for local entity (%d, %d).", d, i);

    dolfin_assert(entity_indices[i] >= 0);
    mesh.topology().set_global_index(d, i, entity_indices[i]);
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::partition(Mesh& mesh, const LocalMeshData& mesh_data)
{
  // Compute cell partition
  std::vector<uint> cell_partition;
  const std::string partitioner = parameters["mesh_partitioner"];
  if (partitioner == "SCOTCH")
    SCOTCH::compute_partition(cell_partition, mesh_data);
  else if (partitioner == "ParMETIS")
    ParMETIS::compute_partition(cell_partition, mesh_data);
  else
  {
    dolfin_error("MeshPartitioning.cpp",
                 "partition mesh",
                 "Mesh partitioner '%s' is not known. Known partitioners are 'SCOTCH' or 'ParMETIS'", partitioner.c_str());
  }

  // Distribute cells
  Timer timer("PARALLEL 2: Distribute mesh (cells and vertices)");
  std::vector<std::size_t> global_cell_indices;
  boost::multi_array<std::size_t, 2> cell_vertices;
  distribute_cells(global_cell_indices, cell_vertices, mesh_data,
                   cell_partition);

  // Distribute vertices
  std::vector<std::size_t> vertex_indices;
  boost::multi_array<double, 2> vertex_coordinates;
  std::map<std::size_t, std::size_t> vertex_global_to_local;
  distribute_vertices(vertex_indices, vertex_coordinates,
                      vertex_global_to_local, cell_vertices, mesh_data);
  timer.stop();

  // Build mesh
  build_mesh(mesh, global_cell_indices, cell_vertices, vertex_indices,
             vertex_coordinates, vertex_global_to_local,
             mesh_data.tdim, mesh_data.gdim, mesh_data.num_global_cells,
             mesh_data.num_global_vertices);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_mesh_domains(Mesh& mesh,
                                          const LocalMeshData& local_data)
{
  // Local domain data
  const std::map<uint, std::vector< std::pair<std::pair<std::size_t, uint>, uint> > >
    domain_data = local_data.domain_data;
  if (domain_data.empty())
    return;

  // Initialse mesh domains
  const uint D = mesh.topology().dim();
  mesh.domains().init(D);

  std::map<uint, std::vector< std::pair<std::pair<std::size_t, uint>, uint> > >::const_iterator dim_data;
  for (dim_data = domain_data.begin(); dim_data != domain_data.end(); ++dim_data)
  {
    // Get mesh value collection used for marking
    const std::size_t dim = dim_data->first;
    dolfin_assert(mesh.domains().markers(dim));
    MeshValueCollection<uint>& markers = *(mesh.domains().markers(dim));

    const std::vector< std::pair<std::pair<std::size_t, uint>, uint> >&
        local_value_data = dim_data->second;
    build_mesh_value_collection(mesh, local_value_data, markers);
  }
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
  MeshPartitioning::compute_num_global_entities(std::size_t num_local_entities,
                                                uint num_processes,
                                                uint process_number)
{
  // Communicate number of local entities
  std::vector<std::size_t> num_entities_to_number;
  MPI::all_gather(num_local_entities, num_entities_to_number);

  // Compute offset
  const std::size_t offset = std::accumulate(num_entities_to_number.begin(),
                           num_entities_to_number.begin() + process_number, 0);

  // Compute number of global entities
  const std::size_t num_global = std::accumulate(num_entities_to_number.begin(),
                                          num_entities_to_number.end(), 0);

  return std::make_pair(num_global, offset);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::compute_entity_ownership(
          const std::map<Entity, std::size_t>& entities,
          const std::map<std::size_t, std::set<uint> >& shared_vertices,
          std::map<Entity, EntityData>& owned_exclusive_entities,
          std::map<Entity, EntityData>& owned_shared_entities,
          std::map<Entity, EntityData>& unowned_shared_entities)
{
  // Compute preliminat ownership
  compute_preliminary_entity_ownership(entities, shared_vertices,
                                       owned_exclusive_entities,
                                       owned_shared_entities,
                                       unowned_shared_entities);

  // Qualify boundary entities. We need to find out if the ignored
  // (shared with lower ranked process) entities are entities of a
  // lower ranked process.  If not, this process becomes the lower
  // ranked process for the entity in question, and is therefore
  // responsible for communicating values to the higher ranked
  // processes (if any).

  compute_final_entity_ownership(owned_exclusive_entities,
                                 owned_shared_entities,
                                 unowned_shared_entities);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::compute_preliminary_entity_ownership(
  const std::map<Entity, std::size_t>& entities,
  const std::map<std::size_t, std::set<unsigned int> >& shared_vertices,
  std::map<Entity, EntityData>& owned_exclusive_entities,
  std::map<Entity, EntityData>& owned_shared_entities,
  std::map<Entity, EntityData>& unowned_shared_entities)
{
  // Clear maps
  owned_exclusive_entities.clear();
  owned_shared_entities.clear();
  unowned_shared_entities.clear();

  // Get process number
  const uint process_number = MPI::process_number();

  // Iterate over all entities
  std::map<std::vector<std::size_t>, std::size_t>::const_iterator it;
  for (it = entities.begin(); it != entities.end(); ++it)
  {
    const Entity& entity = it->first;
    const std::size_t local_entity_index = it->second;

    // Compute which processes entity is shared with
    std::vector<unsigned int> entity_processes;
    if (in_overlap(entity, shared_vertices))
    {
      std::vector<std::size_t> intersection(shared_vertices.find(entity[0])->second.begin(),
                                     shared_vertices.find(entity[0])->second.end());
      std::vector<std::size_t>::iterator intersection_end = intersection.end();

      for (std::size_t i = 1; i < entity.size(); ++i)
      {
        const std::size_t v = entity[i];
        const std::set<unsigned int>& shared_vertices_v
          = shared_vertices.find(v)->second;

        intersection_end
          = std::set_intersection(intersection.begin(), intersection_end,
                                  shared_vertices_v.begin(), shared_vertices_v.end(),
                                  intersection.begin());
      }
      entity_processes = std::vector<unsigned int>(intersection.begin(), intersection_end);
    }

    // Check if entity is ignored (shared with lower ranked process)
    bool ignore = false;
    for (std::size_t i = 0; i < entity_processes.size(); ++i)
    {
      if (entity_processes[i] < process_number)
      {
        ignore = true;
        break;
      }
    }

    // Check cases
    if (entity_processes.empty())
      owned_exclusive_entities[entity] = EntityData(local_entity_index);
    else if (ignore)
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
void MeshPartitioning::compute_final_entity_ownership(
  std::map<Entity, EntityData>& owned_exclusive_entities,
  std::map<Entity, EntityData>& owned_shared_entities,
  std::map<Entity, EntityData>& unowned_shared_entities)
{
  const std::size_t process_number = MPI::process_number();

  // Communicate common entities, starting with the entities we think
  // should be ignored
  std::vector<std::size_t> send_common_entity_values;
  std::vector<unsigned int> destinations_common_entity;
  std::map<Entity, EntityData>::const_iterator it;
  for (it = unowned_shared_entities.begin(); it != unowned_shared_entities.end(); ++it)
  {
    // Get entity vertices (global vertex indices)
    const Entity& entity = it->first;

    // Get entity processes (processes might sharing the entity)
    const std::vector<unsigned int>& entity_processes = it->second.processes;

    // Prepare data for sending
    for (std::size_t j = 0; j < entity_processes.size(); ++j)
    {
      const unsigned int p = entity_processes[j];
      send_common_entity_values.push_back(entity.size());
      send_common_entity_values.insert(send_common_entity_values.end(), entity.begin(), entity.end());
      destinations_common_entity.insert(destinations_common_entity.end(), entity.size() + 1, p);
    }
  }

  // Communicate common entities, add the entities we think should be
  // shared as well
  std::map<Entity, EntityData>::const_iterator it1;
  for (it1 = owned_shared_entities.begin(); it1 != owned_shared_entities.end(); ++it1)
  {
    // Get entity vertices (global vertex indices)
    const Entity& entity = it1->first;

    // Get entity processes (processes might sharing the entity)
    const std::vector<unsigned int>& entity_processes = it1->second.processes;

    // Prepare data for sending
    for (std::size_t j = 0; j < entity_processes.size(); ++j)
    {
      const std::size_t p = entity_processes[j];
      dolfin_assert(process_number < p);
      send_common_entity_values.push_back(entity.size());
      send_common_entity_values.insert(send_common_entity_values.end(), entity.begin(), entity.end());
      destinations_common_entity.insert(destinations_common_entity.end(), entity.size() + 1, p);
    }
  }

  // Communicate common entities
  std::vector<std::size_t> received_common_entity_values;
  std::vector<unsigned int> sources_common_entity;
  MPI::distribute(send_common_entity_values, destinations_common_entity,
                  received_common_entity_values, sources_common_entity);

  // Check if entities received are really entities
  std::vector<std::size_t> send_is_entity_values;
  std::vector<unsigned int> destinations_is_entity;
  for (std::size_t i = 0; i < received_common_entity_values.size();)
  {
    // Get entity
    const std::size_t p =  sources_common_entity[i];
    const std::size_t entity_size = received_common_entity_values[i++];
    Entity entity;
    for (std::size_t j = 0; j < entity_size; ++j)
      entity.push_back(received_common_entity_values[i++]);

    // Check if it is an entity (in which case it will be in ignored or
    // shared entities)
    std::size_t is_entity = 0;
    if (unowned_shared_entities.find(entity) != unowned_shared_entities.end()
          || owned_shared_entities.find(entity) != owned_shared_entities.end())
    {
      is_entity = 1;
    }

    // Add information about entity (whether it's actually an entity) to send
    // to other processes
    send_is_entity_values.push_back(entity_size);
    destinations_is_entity.push_back(p);
    for (std::size_t j = 0; j < entity_size; ++j)
    {
      send_is_entity_values.push_back(entity[j]);
      destinations_is_entity.push_back(p);
    }
    send_is_entity_values.push_back(is_entity);
    destinations_is_entity.push_back(p);
  }

  // Send data back (list of requested entities that are really entities)
  std::vector<std::size_t> received_is_entity_values;
  std::vector<unsigned int> sources_is_entity;
  MPI::distribute(send_is_entity_values, destinations_is_entity,
                  received_is_entity_values, sources_is_entity);

  // Create map from entities to processes where it is an entity
  std::map<Entity, std::vector<unsigned int> > entity_processes;
  for (std::size_t i = 0; i < received_is_entity_values.size();)
  {
    const std::size_t p = sources_is_entity[i];
    const std::size_t entity_size = received_is_entity_values[i++];
    Entity entity;
    for (std::size_t j = 0; j < entity_size; ++j)
      entity.push_back(received_is_entity_values[i++]);
    const std::size_t is_entity = received_is_entity_values[i++];
    if (is_entity == 1)
    {
      // Add entity since it is actually an entity for process p
      entity_processes[entity].push_back(p);
    }
  }

  // Fix the list of entities we ignore (numbered by lower ranked process)
  std::vector<std::vector<std::size_t> > unignore_entities;
  for (it = unowned_shared_entities.begin(); it != unowned_shared_entities.end(); ++it)
  {
    const Entity& entity = it->first;
    const std::size_t local_entity_index = it->second.index;
    if (entity_processes.find(entity) != entity_processes.end())
    {
      std::vector<unsigned int> common_processes = entity_processes[entity];
      dolfin_assert(!common_processes.empty());
      const std::size_t min_proc = *(std::min_element(common_processes.begin(), common_processes.end()));

      if (process_number < min_proc)
      {
        // Move from ignored to shared
        owned_shared_entities[entity] = EntityData(local_entity_index,
                                                   common_processes);

        // Add entity to list of entities that should be removed from
        // the ignored entity list.
        unignore_entities.push_back(entity);
      }
    }
    else
    {
      // Move from ignored to owned
      owned_exclusive_entities[entity] = EntityData(local_entity_index);

      // Add entity to list of entities that should be removed from the
      // ignored entity list
      unignore_entities.push_back(entity);
    }
  }

  // Remove ignored entities that should not be ignored
  for (std::size_t i = 0; i < unignore_entities.size(); ++i)
    unowned_shared_entities.erase(unignore_entities[i]);

  // Fix the list of entities we share
  std::vector<std::vector<std::size_t> > unshare_entities;
  for (std::map<Entity, EntityData>::iterator it = owned_shared_entities.begin();
         it != owned_shared_entities.end(); ++it)
  {
    const Entity& entity = it->first;
    const std::size_t local_entity_index = it->second.index;
    if (entity_processes.find(entity) == entity_processes.end())
    {
      // Move from shared to owned
      owned_exclusive_entities[entity] = EntityData(local_entity_index);
      unshare_entities.push_back(entity);
    }
    else
    {
      // Update processor list of shared entities
      it->second.processes = entity_processes[entity];
    }
  }

  // Remove shared entities that should not be shared
  for (std::size_t i = 0; i < unshare_entities.size(); ++i)
    owned_shared_entities.erase(unshare_entities[i]);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_cells(std::vector<std::size_t>& global_cell_indices,
                                    boost::multi_array<std::size_t, 2>& cell_vertices,
                                    const LocalMeshData& mesh_data,
                                    const std::vector<uint>& cell_partition)
{
  // This function takes the partition computed by the partitioner
  // (which tells us to which process each of the local cells stored in
  // LocalMeshData on this process belongs. We use MPI::distribute to
  // redistribute all cells (the global vertex indices of all cells).

  // Get dimensions of local mesh_data
  const std::size_t num_local_cells = mesh_data.cell_vertices.size();
  dolfin_assert(mesh_data.global_cell_indices.size() == num_local_cells);
  const std::size_t num_cell_vertices = mesh_data.num_vertices_per_cell;
  if (!mesh_data.cell_vertices.empty())
  {
    if (mesh_data.cell_vertices[0].size() != num_cell_vertices)
    {
      dolfin_error("MeshPartitioning.cpp",
                   "distribute cells",
                   "Mismatch in number of cell vertices (%d != %d) on process %d",
                   mesh_data.cell_vertices[0].size(), num_cell_vertices, MPI::process_number());
    }
  }

  // Build array of cell-vertex connectivity and partition vector
  // Distribute the global cell number as well
  std::vector<std::size_t> send_cell_vertices;
  std::vector<unsigned int> destinations_cell_vertices;
  send_cell_vertices.reserve(num_local_cells*(num_cell_vertices + 1));
  destinations_cell_vertices.reserve(num_local_cells*(num_cell_vertices + 1));
  for (std::size_t i = 0; i < num_local_cells; i++)
  {
    send_cell_vertices.push_back(mesh_data.global_cell_indices[i]);
    destinations_cell_vertices.push_back(cell_partition[i]);
    for (std::size_t j = 0; j < num_cell_vertices; j++)
    {
      send_cell_vertices.push_back(mesh_data.cell_vertices[i][j]);
      destinations_cell_vertices.push_back(cell_partition[i]);
    }
  }

  // Distribute cell-vertex connectivity
  std::vector<std::size_t> received_cell_vertices;
  MPI::distribute(send_cell_vertices, destinations_cell_vertices,
                  received_cell_vertices);
  dolfin_assert(received_cell_vertices.size() % (num_cell_vertices + 1) == 0);
  destinations_cell_vertices.clear();

  // Put mesh_data back into mesh_data.cell_vertices
  const std::size_t num_new_local_cells = received_cell_vertices.size()/(num_cell_vertices + 1);
  cell_vertices.resize(boost::extents[num_new_local_cells][num_cell_vertices]);
  global_cell_indices.resize(num_new_local_cells);

  // Loop over new cells
  for (std::size_t i = 0; i < num_new_local_cells; ++i)
  {
    global_cell_indices[i] = received_cell_vertices[i*(num_cell_vertices + 1)];
    for (std::size_t j = 0; j < num_cell_vertices; ++j)
      cell_vertices[i][j] = received_cell_vertices[i*(num_cell_vertices + 1) + j + 1];
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_vertices(std::vector<std::size_t>& vertex_indices,
  boost::multi_array<double, 2>& vertex_coordinates,
  std::map<std::size_t, std::size_t>& glob2loc,
  const boost::multi_array<std::size_t, 2>& cell_vertices,
  const LocalMeshData& mesh_data)
{
  // This function distributes all vertices (coordinates and local-to-global
  // mapping) according to the cells that are stored on each process. This
  // happens in several stages: First each process figures out which vertices
  // it needs (by looking at its cells) and where those vertices are located.
  // That information is then distributed so that each process learns where
  // it needs to send its vertices.

  // Get number of processes
  const std::size_t num_processes = MPI::num_processes();

  // Get geometric dimension
  const uint gdim = mesh_data.gdim;

  // Compute which vertices we need
  std::set<std::size_t> needed_vertex_indices;
  boost::multi_array<std::size_t, 2>::const_iterator vertices;
  for (vertices = cell_vertices.begin(); vertices != cell_vertices.end(); ++vertices)
    needed_vertex_indices.insert(vertices->begin(), vertices->end());

  // Compute where (process number) the vertices we need are located
  std::vector<std::size_t> send_vertex_indices;
  std::vector<unsigned int> destinations_vertex;
  std::vector<std::vector<std::size_t> > vertex_location(num_processes);
  std::set<std::size_t>::const_iterator required_vertex;
  for (required_vertex = needed_vertex_indices.begin();
        required_vertex != needed_vertex_indices.end(); ++required_vertex)
  {
    // Get process that has required vertex
    const uint location = MPI::index_owner(*required_vertex, mesh_data.num_global_vertices);
    destinations_vertex.push_back(location);
    send_vertex_indices.push_back(*required_vertex);
    vertex_location[location].push_back(*required_vertex);
  }

  // Send required vertices to other proceses, and receive back vertices
  // required by othe processes.
  std::vector<std::size_t> received_vertex_indices;
  std::vector<unsigned int> sources_vertex;
  MPI::distribute(send_vertex_indices, destinations_vertex,
                  received_vertex_indices, sources_vertex);
  dolfin_assert(received_vertex_indices.size() == sources_vertex.size());

  // Distribute vertex coordinates
  std::vector<double> send_vertex_coordinates;
  std::vector<unsigned int> destinations_vertex_coordinates;
  const std::pair<std::size_t, std::size_t> local_vertex_range = MPI::local_range(mesh_data.num_global_vertices);
  for (std::size_t i = 0; i < sources_vertex.size(); ++i)
  {
    dolfin_assert(received_vertex_indices[i] >= local_vertex_range.first
                      && received_vertex_indices[i] < local_vertex_range.second);
    const std::size_t location = received_vertex_indices[i] - local_vertex_range.first;
    const std::vector<double>& x = mesh_data.vertex_coordinates[location];
    dolfin_assert(x.size() == gdim);
    for (std::size_t j = 0; j < gdim; ++j)
    {
      send_vertex_coordinates.push_back(x[j]);
      destinations_vertex_coordinates.push_back(sources_vertex[i]);
    }
  }
  std::vector<double> received_vertex_coordinates;
  std::vector<unsigned int> sources_vertex_coordinates;
  MPI::distribute(send_vertex_coordinates, destinations_vertex_coordinates,
                  received_vertex_coordinates, sources_vertex_coordinates);

  // Set index counters to first position in recieve buffers
  std::vector<std::size_t> index_counters(num_processes, 0);

  // Clear data
  vertex_indices.clear();
  glob2loc.clear();

  // Store coordinates and construct global to local mapping
  const std::size_t num_local_vertices = received_vertex_coordinates.size()/gdim;
  vertex_coordinates.resize(boost::extents[num_local_vertices][gdim]);
  vertex_indices.resize(num_local_vertices);
  for (std::size_t i = 0; i < num_local_vertices; ++i)
  {
    for (std::size_t j = 0; j < gdim; ++j)
      vertex_coordinates[i][j] = received_vertex_coordinates[i*gdim + j];

    const std::size_t sender_process = sources_vertex_coordinates[i*gdim];
    const std::size_t global_vertex_index
      = vertex_location[sender_process][index_counters[sender_process]++];
    glob2loc[global_vertex_index] = i;
    vertex_indices[i] = global_vertex_index;
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_mesh(Mesh& mesh,
              const std::vector<std::size_t>& global_cell_indices,
              const boost::multi_array<std::size_t, 2>& cell_vertices,
              const std::vector<std::size_t>& vertex_indices,
              const boost::multi_array<double, 2>& vertex_coordinates,
              const std::map<std::size_t, std::size_t>& vertex_global_to_local,
              uint tdim, uint gdim, std::size_t num_global_cells,
              std::size_t num_global_vertices)
{
  Timer timer("PARALLEL 3: Build mesh (from local mesh data)");

  // Get number of processes and process number
  const std::size_t num_processes = MPI::num_processes();
  const std::size_t process_number = MPI::process_number();

  // Open mesh for editing
  mesh.clear();
  MeshEditor editor;
  editor.open(mesh, gdim, tdim);

  // Add vertices
  editor.init_vertices(vertex_coordinates.size());
  Point p(gdim);
  dolfin_assert(vertex_indices.size() == vertex_coordinates.size());
  for (std::size_t i = 0; i < vertex_coordinates.size(); ++i)
  {
    for (uint j = 0; j < gdim; ++j)
      p[j] = vertex_coordinates[i][j];
    editor.add_vertex_global(i, vertex_indices[i], p);
  }

  // Add cells
  editor.init_cells(cell_vertices.size());
  const uint num_cell_vertices = tdim + 1;
  std::vector<std::size_t> cell(num_cell_vertices);
  for (std::size_t i = 0; i < cell_vertices.size(); ++i)
  {
    for (std::size_t j = 0; j < num_cell_vertices; ++j)
    {
      std::map<std::size_t, std::size_t>::const_iterator iter
          = vertex_global_to_local.find(cell_vertices[i][j]);
      dolfin_assert(iter != vertex_global_to_local.end());
      cell[j] = iter->second;
    }
    editor.add_cell(i, global_cell_indices[i], cell);
  }

  // Close mesh: Note that this must be done after creating the global
  // vertex map or otherwise the ordering in mesh.close() will be wrong
  // (based on local numbers).
  editor.close();

  // Set global number of cells and vertices
  mesh.topology().init_global(0, num_global_vertices);
  mesh.topology().init_global(tdim,  num_global_cells);

  /// Communicate global number of boundary vertices to all processes

  // Construct boundary mesh
  BoundaryMesh bmesh(mesh);

  const MeshFunction<unsigned int>& boundary_vertex_map = bmesh.vertex_map();
  const std::size_t boundary_size = boundary_vertex_map.size();

  // Build sorted array of global boundary vertex indices (global
  // numbering)
  std::vector<std::size_t> global_vertex_send(boundary_size);
  for (std::size_t i = 0; i < boundary_size; ++i)
    global_vertex_send[i] = vertex_indices[boundary_vertex_map[i]];
  std::sort(global_vertex_send.begin(), global_vertex_send.end());

  // Distribute boundaries' sizes
  std::vector<std::size_t> boundary_sizes;
  MPI::all_gather(boundary_size, boundary_sizes);

  // Receive buffer
  std::vector<std::size_t> global_vertex_recv;

  // Create shared_vertices data structure: mapping from shared vertices
  // to list of neighboring processes
  std::map<std::size_t, std::set<unsigned int> >& shared_vertices
        = mesh.topology().shared_entities(0);
  shared_vertices.clear();

  // Distribute boundaries and build mappings
  for (std::size_t i = 1; i < num_processes; ++i)
  {
    // We send data to process p - i (i steps to the left)
    const int p = (process_number - i + num_processes) % num_processes;

    // We receive data from process p + i (i steps to the right)
    const int q = (process_number + i) % num_processes;

    // Send and receive
    MPI::send_recv(global_vertex_send, p, global_vertex_recv, q);

    // Compute intersection of global indices
    std::vector<std::size_t> intersection(std::min(global_vertex_send.size(), global_vertex_recv.size()));
    std::vector<std::size_t>::iterator intersection_end
      = std::set_intersection(global_vertex_send.begin(), global_vertex_send.end(),
                              global_vertex_recv.begin(), global_vertex_recv.end(),
                              intersection.begin());

    // Fill shared vertices information
    std::vector<std::size_t>::const_iterator index;
    for (index = intersection.begin(); index != intersection_end; ++index)
      shared_vertices[*index].insert(q);
  }
}
//-----------------------------------------------------------------------------
bool MeshPartitioning::in_overlap(const Entity& entity,
                const std::map<std::size_t, std::set<unsigned int> >& shared)
{
  // Iterate over entity vertices
  Entity::const_iterator e;
  for (e = entity.begin(); e != entity.end(); ++e)
  {
    // Return false if an entity vertex is not in the list (map) of
    // shared entities
    if (shared.find(*e) == shared.end())
      return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
std::vector<unsigned int> MeshPartitioning::num_connected_cells(const Mesh& mesh,
             const std::map<Entity, std::size_t>& entities,
             const std::map<Entity, EntityData>& owned_shared_entities,
             const std::map<Entity, EntityData>& unowned_shared_entities)
{
  // Topological dimension
  const uint D = mesh.topology().dim();

  // Create vector to hold number of cells connected to each facet. Assume
  // facet is internal, then modify for external facets.
  std::vector<unsigned int> num_global_neighbors(mesh.num_facets(), 2);

  // FIXME: Check that everything is correctly initalised

  // Add facets that are locally connected to one cell only
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    if (facet->num_entities(D) == 1)
      num_global_neighbors[facet->index()] = 1;
  }

  // Handle facets on internal partition boundaries
  std::map<Entity, EntityData>::const_iterator it;
  for (it = owned_shared_entities.begin(); it != owned_shared_entities.end(); ++it)
    num_global_neighbors[entities.find(it->first)->second] = 2;

  for (it = unowned_shared_entities.begin(); it != unowned_shared_entities.end(); ++it)
    num_global_neighbors[entities.find(it->first)->second] = 2;

  return num_global_neighbors;
}
//-----------------------------------------------------------------------------
