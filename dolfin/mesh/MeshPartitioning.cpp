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
#include "MeshValueCollection.h"
#include "ParallelData.h"
#include "Point.h"
#include "Vertex.h"
#include "MeshPartitioning.h"


using namespace dolfin;

// Utility functions for debugging/printing
template<typename InputIterator>
void print_container(std::ostream& ostr, InputIterator itbegin, InputIterator itend, const char* delimiter=", ")
{
  std::copy(itbegin, itend, std::ostream_iterator<typename InputIterator::value_type>(ostr, delimiter));
}

// Explicitly instantiate some templated functions to help the Python wrappers
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<uint, uint>, uint> >& local_value_data,
   MeshValueCollection<uint>& mesh_values);
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<uint, uint>, int> >& local_value_data,
   MeshValueCollection<int>& mesh_values);
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<uint, uint>, double> >& local_value_data,
   MeshValueCollection<double>& mesh_values);
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<uint, uint>, bool> >& local_value_data,
   MeshValueCollection<bool>& mesh_values);

//-----------------------------------------------------------------------------
void MeshPartitioning::build_distributed_mesh(Mesh& mesh)
{
  // Create and distribute local mesh data
  dolfin_debug("creating local mesh data");
  LocalMeshData local_mesh_data(mesh);
  dolfin_debug("created local mesh data");

  // Partition mesh based on local mesh data
  partition(mesh, local_mesh_data);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_distributed_mesh(Mesh& mesh, LocalMeshData& local_data)
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
  if (d == 0)
  {
    dolfin_error("MeshPartitioning.cpp",
                 "number mesh entities",
                 "Vertex indices do not exist; need vertices to number entities of dimension 0");
  }

  // Return if global entity indices are already calculated
  if (mesh.parallel_data().have_global_entity_indices(d))
    return;

  // Initialize entities of dimension d
  mesh.init(d);

  // Get number of processes and process number
  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Get global vertex indices
  MeshFunction<unsigned int>& global_vertex_indices = mesh.parallel_data().global_entity_indices(0);

  // Get shared vertices
  std::map<uint, std::vector<uint> >& shared_vertices = mesh.parallel_data().shared_vertices();

  // Sort shared vertices
  for (std::map<uint, std::vector<uint> >::iterator it = shared_vertices.begin(); it != shared_vertices.end(); ++it)
    std::sort(it->second.begin(), it->second.end());

  // Build entity-to-global-vertex-number information
  std::map<std::vector<uint>, uint> entities;
  for (MeshEntityIterator e(mesh, d); !e.end(); ++e)
  {
    std::vector<uint> entity;
    for (VertexIterator vertex(*e); !vertex.end(); ++vertex)
      entity.push_back(global_vertex_indices[vertex->index()]);
    std::sort(entity.begin(), entity.end());
    entities[entity] = e->index();
  }

  /// Find out which entities to ignore, which to number and which to
  /// number and send to other processes. Entities shared by two or
  /// more processes are numbered by the lower ranked process.

  // Entities to number
  std::map<std::vector<uint>, uint> owned_entity_indices;

  // Candidates to number and send to other, higher rank processes
  std::map<std::vector<uint>, uint> shared_entity_indices;
  std::map<std::vector<uint>, std::vector<uint> > shared_entity_processes;

  // Candidates for being numbered by another, lower ranked process. We need
  // to check that the entity is really an entity at the other process. If not,
  // we must number it ourself
  std::map<std::vector<uint>, uint> ignored_entity_indices;
  std::map<std::vector<uint>, std::vector<uint> > ignored_entity_processes;

  compute_preliminary_entity_ownership(entities, shared_vertices,
                           owned_entity_indices, shared_entity_indices,
                           shared_entity_processes, ignored_entity_indices,
                           ignored_entity_processes);

  // Qualify boundary entities. We need to find out if the ignored
  // (shared with lower ranked process) entities are entities of a
  // lower ranked process.  If not, this process becomes the lower
  // ranked process for the entity in question, and is therefore
  // responsible for communicating values to the higher ranked
  // processes (if any).

  compute_final_entity_ownership(owned_entity_indices, shared_entity_indices,
                           shared_entity_processes, ignored_entity_indices,
                           ignored_entity_processes);

  /// --- Mark exterior facets

  // Create mesh markers for exterior facets
  if (d == (mesh.topology().dim() - 1))
  {
    MeshFunction<bool>& exterior_facets = mesh.parallel_data().exterior_facet();
    exterior_facets.init(d);
    mark_nonshared(entities, shared_entity_indices, ignored_entity_indices,
                   exterior_facets);
  }

  // Compute global number of entities and process offset
  const uint num_local_entities = owned_entity_indices.size() + shared_entity_indices.size();
  const std::pair<uint, uint> num_global_entities = compute_num_global_entities(num_local_entities,
                                                     num_processes,
                                                     process_number);
  // Extract offset
  uint offset = num_global_entities.second;

  // Store number of global entities
  mesh.parallel_data().num_global_entities()[d] = num_global_entities.first;


  /// ---- Numbering

  // Prepare list of entity numbers. Check later that nothing is -1
  std::vector<int> entity_indices(mesh.size(d), -1);

  std::map<std::vector<uint>, uint>::const_iterator it;

  // Number owned entities
  for (it = owned_entity_indices.begin();  it != owned_entity_indices.end(); ++it)
    entity_indices[it->second] = offset++;

  // Number shared entities
  for (it = shared_entity_indices.begin(); it != shared_entity_indices.end(); ++it)
    entity_indices[it->second] = offset++;

  // Communicate indices for shared entities and get indices for ignored
  std::vector<uint> send_values, destinations;
  for (it = shared_entity_indices.begin(); it != shared_entity_indices.end(); ++it)
  {
    // Get entity index
    const uint local_entity_index = it->second;
    const int entity_index = entity_indices[local_entity_index];
    assert(entity_index != -1);

    // Get entity vertices (global vertex indices)
    const std::vector<uint>& entity = it->first;

    // Get entity processes (processes sharing the entity)
    const std::vector<uint>& entity_processes = shared_entity_processes[entity];

    // Prepare data for sending
    for (uint j = 0; j < entity_processes.size(); ++j)
    {
      // Store interleaved: entity index, number of vertices, global vertex indices
      send_values.push_back(entity_index);
      send_values.push_back(entity.size());
      send_values.insert(send_values.end(), entity.begin(), entity.end());
      destinations.insert(destinations.end(), entity.size() + 2, entity_processes[j]);
    }
  }

  // Send data
  std::vector<uint> received_values, sources;
  MPI::distribute(send_values, destinations, received_values, sources);

  // Fill in global entity indices recieved from lower ranked processes
  for (uint i = 0; i < received_values.size();)
  {
    const uint p = sources[i];
    const uint global_index = received_values[i++];
    const uint entity_size = received_values[i++];
    std::vector<uint> entity;
    for (uint j = 0; j < entity_size; ++j)
      entity.push_back(received_values[i++]);

    // Sanity check, should not receive an entity we don't need
    if (ignored_entity_indices.find(entity) == ignored_entity_indices.end())
    {
      std::stringstream msg;
      msg << "Process " << MPI::process_number() << " received illegal entity given by ";
      print_container(msg, entity.begin(), entity.end());
      msg << " with global index " << global_index;
      msg << " from process " << p;
      dolfin_error("MeshPartitioning.cpp",
                   "number mesh entities",
                   msg.str());
    }

    const uint local_entity_index = ignored_entity_indices.find(entity)->second;
    assert(entity_indices[local_entity_index] == -1);
    entity_indices[local_entity_index] = global_index;
  }

  // Create mesh data
  MeshFunction<unsigned int>& global_entity_indices = mesh.parallel_data().global_entity_indices(d);
  for (uint i = 0; i < entity_indices.size(); ++i)
  {
    if (entity_indices[i] < 0)
      log(WARNING, "Missing global number for local entity (%d, %d).", d, i);

    assert(entity_indices[i] >= 0);
    assert(i < global_entity_indices.size());

    global_entity_indices[i] = static_cast<uint>(entity_indices[i]);
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::partition(Mesh& mesh, LocalMeshData& mesh_data)
{
  // Compute cell partition
  std::vector<uint> cell_partition;
  const std::string partitioner = parameters["mesh_partitioner"];
  if (partitioner == "SCOTCH")
    SCOTCH::compute_partition(cell_partition, mesh_data);
  else if (partitioner == "ParMETIS")
    ParMETIS::compute_partition(cell_partition, mesh_data);
  else
    dolfin_error("MeshPartitioning.cpp",
                 "partition mesh",
                 "Mesh partitioner '%s' is not known. Known partitioners are 'SCOTCH' or 'ParMETIS'", partitioner.c_str());

  // Distribute cells
  Timer timer("PARALLEL 2: Distribute mesh (cells and vertices)");
  distribute_cells(mesh_data, cell_partition);

  // Distribute vertices
  std::map<uint, uint> vertex_global_to_local;
  distribute_vertices(mesh_data, vertex_global_to_local);
  timer.stop();

  // Build mesh
  build_mesh(mesh, mesh_data, vertex_global_to_local);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_mesh_domains(Mesh& mesh,
                                          const LocalMeshData& local_data)
{
  // Local domain data
  const std::map<uint, std::vector< std::pair<std::pair<dolfin::uint, dolfin::uint>, dolfin::uint> > > domain_data
      = local_data.domain_data;
  if (domain_data.size() == 0)
    return;

  // Initialse mesh domains
  const uint D = mesh.topology().dim();
  mesh.domains().init(D);

  std::map<uint, std::vector< std::pair<std::pair<dolfin::uint, dolfin::uint>, dolfin::uint> > >::const_iterator dim_data;
  for (dim_data = domain_data.begin(); dim_data != domain_data.end(); ++dim_data)
  {
    // Get mesh value collection used for marking
    const uint dim = dim_data->first;
    MeshValueCollection<uint>& markers = mesh.domains().markers(dim);

    const std::vector< std::pair<std::pair<dolfin::uint, dolfin::uint>, dolfin::uint> >& local_value_data = dim_data->second;
    build_mesh_value_collection(mesh, local_value_data, markers);
  }
}
//-----------------------------------------------------------------------------
std::pair<unsigned int, unsigned int>
  MeshPartitioning::compute_num_global_entities(uint num_local_entities,
                                                uint num_processes,
                                                uint process_number)
{
  // Communicate number of local entities
  std::vector<uint> num_entities_to_number;
  MPI::all_gather(num_local_entities, num_entities_to_number);

  // Compute offset
  const uint offset = std::accumulate(num_entities_to_number.begin(),
                           num_entities_to_number.begin() + process_number, 0);

  // Compute number of global entities
  const uint num_global = std::accumulate(num_entities_to_number.begin(),
                                          num_entities_to_number.end(), 0);

  return std::make_pair(num_global, offset);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::compute_preliminary_entity_ownership(const std::map<std::vector<uint>, uint>& entities,
  const std::map<uint, std::vector<uint> >& shared_vertices,
  std::map<std::vector<uint>, uint>& owned_entity_indices,
  std::map<std::vector<uint>, uint>& shared_entity_indices,
  std::map<std::vector<uint>, std::vector<uint> >& shared_entity_processes,
  std::map<std::vector<uint>, uint>& ignored_entity_indices,
  std::map<std::vector<uint>, std::vector<uint> >& ignored_entity_processes)
{
  owned_entity_indices.clear();
  shared_entity_indices.clear();
  shared_entity_processes.clear();
  ignored_entity_indices.clear();
  ignored_entity_processes.clear();

  // Get process number
  const uint process_number = MPI::process_number();

  // Iterate over all entities
  for (std::map<std::vector<uint>, uint>::const_iterator it = entities.begin(); it != entities.end(); ++it)
  {
    const std::vector<uint>& entity = it->first;
    const uint local_entity_index = it->second;

    // Compute which processes entity is shared with
    std::vector<uint> entity_processes;
    if (in_overlap(entity, shared_vertices))
    {
      std::vector<uint> intersection = shared_vertices.find(entity[0])->second;
      std::vector<uint>::iterator intersection_end = intersection.end();

      for (uint i = 1; i < entity.size(); ++i)
      {
        const uint v = entity[i];
        const std::vector<uint>& shared_vertices_v = shared_vertices.find(v)->second;
        intersection_end = std::set_intersection(intersection.begin(),
                                   intersection_end, shared_vertices_v.begin(),
                                   shared_vertices_v.end(), intersection.begin());
      }
      entity_processes = std::vector<uint>(intersection.begin(), intersection_end);
    }

    // Check if entity is ignored (shared with lower ranked process)
    bool ignore = false;
    for (uint i = 0; i < entity_processes.size(); ++i)
    {
      if (entity_processes[i] < process_number)
      {
        ignore = true;
        break;
      }
    }

    // Check cases
    if (entity_processes.size() == 0)
      owned_entity_indices[entity] = local_entity_index;
    else if (ignore)
    {
      ignored_entity_indices[entity] = local_entity_index;
      ignored_entity_processes[entity] = entity_processes;
    }
    else
    {
      shared_entity_indices[entity] = local_entity_index;
      shared_entity_processes[entity] = entity_processes;
    }
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::compute_final_entity_ownership(std::map<std::vector<uint>, uint>& owned_entity_indices,
          std::map<std::vector<uint>, uint>& shared_entity_indices,
          std::map<std::vector<uint>, std::vector<uint> >& shared_entity_processes,
          std::map<std::vector<uint>, uint>& ignored_entity_indices,
          std::map<std::vector<uint>, std::vector<uint> >& ignored_entity_processes)
{
  const uint process_number = MPI::process_number();

  std::map<std::vector<uint>, std::vector<uint> >::const_iterator it;

  // Communicate common entities, starting with the entities we think should be ignored
  std::vector<uint> send_common_entity_values, destinations_common_entity;
  for (it = ignored_entity_processes.begin(); it != ignored_entity_processes.end(); ++it)
  {
    // Get entity vertices (global vertex indices)
    const std::vector<uint>& entity = it->first;

    // Get entity processes (processes might sharing the entity)
    const std::vector<uint>& entity_processes = it->second;

    // Prepare data for sending
    for (uint j = 0; j < entity_processes.size(); ++j)
    {
      const uint p = entity_processes[j];
      send_common_entity_values.push_back(entity.size());
      send_common_entity_values.insert(send_common_entity_values.end(), entity.begin(), entity.end());
      destinations_common_entity.insert(destinations_common_entity.end(), entity.size() + 1, p);
    }
  }

  // Communicate common entities, add the entities we think should be shared as well
  for (it = shared_entity_processes.begin(); it != shared_entity_processes.end(); ++it)
  {
    // Get entity vertices (global vertex indices)
    const std::vector<uint>& entity = it->first;

    // Get entity processes (processes might sharing the entity)
    const std::vector<uint>& entity_processes = it->second;

    // Prepare data for sending
    for (uint j = 0; j < entity_processes.size(); ++j)
    {
      const uint p = entity_processes[j];
      assert(process_number < p);
      send_common_entity_values.push_back(entity.size());
      send_common_entity_values.insert(send_common_entity_values.end(), entity.begin(), entity.end());
      destinations_common_entity.insert(destinations_common_entity.end(), entity.size() + 1, p);
    }
  }

  // Communicate common entities
  std::vector<uint> received_common_entity_values, sources_common_entity;
  MPI::distribute(send_common_entity_values, destinations_common_entity,
                  received_common_entity_values, sources_common_entity);

  // Check if entities received are really entities
  std::vector<uint> send_is_entity_values, destinations_is_entity;
  for (uint i = 0; i < received_common_entity_values.size();)
  {
    // Get entity
    const uint p =  sources_common_entity[i];
    const uint entity_size = received_common_entity_values[i++];
    std::vector<uint> entity;
    for (uint j = 0; j < entity_size; ++j)
      entity.push_back(received_common_entity_values[i++]);

    // Check if it is an entity (in which case it will be in ignored or
    // shared entities)
    uint is_entity = 0;
    if (ignored_entity_indices.find(entity) != ignored_entity_indices.end()
          || shared_entity_indices.find(entity) != shared_entity_indices.end())
    {
      is_entity = 1;
    }

    // Add information about entity (whether it's actually an entity) to send
    // to other processes
    send_is_entity_values.push_back(entity_size);
    destinations_is_entity.push_back(p);
    for (uint j = 0; j < entity_size; ++j)
    {
      send_is_entity_values.push_back(entity[j]);
      destinations_is_entity.push_back(p);
    }
    send_is_entity_values.push_back(is_entity);
    destinations_is_entity.push_back(p);
  }

  // Send data back (list of requested entities that are really entities)
  std::vector<uint> received_is_entity_values, sources_is_entity;
  MPI::distribute(send_is_entity_values, destinations_is_entity,
                  received_is_entity_values, sources_is_entity);

  // Create map from entities to processes where it is an entity
  std::map<std::vector<uint>, std::vector<uint> > entity_processes;
  for (uint i = 0; i < received_is_entity_values.size();)
  {
    const uint p = sources_is_entity[i];
    const uint entity_size = received_is_entity_values[i++];
    std::vector<uint> entity;
    for (uint j = 0; j < entity_size; ++j)
      entity.push_back(received_is_entity_values[i++]);
    const uint is_entity = received_is_entity_values[i++];
    if (is_entity == 1)
    {
      // Add entity since it is actually an entity for process p
      entity_processes[entity].push_back(p);
    }
  }

  // Fix the list of entities we ignore (numbered by lower ranked process)
  std::vector<std::vector<uint> > unignore_entities;
  for (std::map<std::vector<uint>, uint>::const_iterator it = ignored_entity_indices.begin(); it != ignored_entity_indices.end(); ++it)
  {
    const std::vector<uint> entity = it->first;
    const uint local_entity_index = it->second;
    if (entity_processes.find(entity) != entity_processes.end())
    {
      std::vector<uint> common_processes = entity_processes[entity];
      assert(common_processes.size() > 0);
      const uint min_proc = *(std::min_element(common_processes.begin(), common_processes.end()));

      if (process_number < min_proc)
      {
        // Move from ignored to shared
        shared_entity_indices[entity] = local_entity_index;
        shared_entity_processes[entity] = common_processes;

        // Add entity to list of entities that should be removed from the ignored entity list.
        unignore_entities.push_back(entity);
      }
    }
    else
    {
      // Move from ignored to owned
      owned_entity_indices[entity] = local_entity_index;

      // Add entity to list of entities that should be removed from the ignored entity list
      unignore_entities.push_back(entity);
    }
  }

  // Remove ignored entities that should not be ignored
  for (uint i = 0; i < unignore_entities.size(); ++i)
  {
    ignored_entity_indices.erase(unignore_entities[i]);
    ignored_entity_processes.erase(unignore_entities[i]);
  }

  // Fix the list of entities we share
  std::vector<std::vector<uint> > unshare_entities;
  for (std::map<std::vector<uint>, uint>::const_iterator it = shared_entity_indices.begin(); it != shared_entity_indices.end(); ++it)
  {
    const std::vector<uint>& entity = it->first;
    const uint local_entity_index = it->second;
    if (entity_processes.find(entity) == entity_processes.end())
    {
      // Move from shared to owned
      owned_entity_indices[entity] = local_entity_index;
      unshare_entities.push_back(entity);
    }
    else
    {
      // Update processor list of shared entities
      shared_entity_processes[entity] = entity_processes[entity];
    }
  }

  // Remove shared entities that should not be shared
  for (uint i = 0; i < unshare_entities.size(); ++i)
  {
    shared_entity_indices.erase(unshare_entities[i]);
    shared_entity_processes.erase(unshare_entities[i]);
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_cells(LocalMeshData& mesh_data,
                                        const std::vector<uint>& cell_partition)
{
  // This function takes the partition computed by the partitioner
  // (which tells us to which process each of the local cells stored in
  // LocalMeshData on this process belongs. We use MPI::distribute to
  // redistribute all cells (the global vertex indices of all cells).

  // Get global cell indices
  const std::vector<uint>& global_cell_indices = mesh_data.global_cell_indices;

  // Get dimensions of local mesh_data
  const uint num_local_cells = mesh_data.cell_vertices.size();
  assert(global_cell_indices.size() == num_local_cells);
  const uint num_cell_vertices = mesh_data.num_vertices_per_cell;
  if (mesh_data.cell_vertices.size() > 0)
  {
    if (mesh_data.cell_vertices[0].size() != num_cell_vertices)
      dolfin_error("MeshPartitioning.cpp",
                   "distribute cells",
                   "Mismatch in number of cell vertices (%d != %d) on process %d",
                   mesh_data.cell_vertices[0].size(), num_cell_vertices, MPI::process_number());
  }

  // Build array of cell-vertex connectivity and partition vector
  // Distribute the global cell number as well
  std::vector<uint> send_cell_vertices, destinations_cell_vertices;
  send_cell_vertices.reserve(num_local_cells*(num_cell_vertices + 1));
  destinations_cell_vertices.reserve(num_local_cells*(num_cell_vertices + 1));
  for (uint i = 0; i < num_local_cells; i++)
  {
    send_cell_vertices.push_back(global_cell_indices[i]);
    destinations_cell_vertices.push_back(cell_partition[i]);
    for (uint j = 0; j < num_cell_vertices; j++)
    {
      send_cell_vertices.push_back(mesh_data.cell_vertices[i][j]);
      destinations_cell_vertices.push_back(cell_partition[i]);
    }
  }

  // Distribute cell-vertex connectivity
  std::vector<uint> received_cell_vertices;
  MPI::distribute(send_cell_vertices, destinations_cell_vertices,
                  received_cell_vertices);
  assert(received_cell_vertices.size() % (num_cell_vertices + 1) == 0);
  destinations_cell_vertices.clear();

  // Clear mesh data
  mesh_data.cell_vertices.clear();
  mesh_data.global_cell_indices.clear();

  // Put mesh_data back into mesh_data.cell_vertices
  const uint num_new_local_cells = received_cell_vertices.size()/(num_cell_vertices + 1);
  mesh_data.cell_vertices.reserve(num_new_local_cells);
  mesh_data.global_cell_indices.reserve(num_new_local_cells);

  // Loop over new cells
  for (uint i = 0; i < num_new_local_cells; ++i)
  {
    mesh_data.global_cell_indices.push_back(received_cell_vertices[i*(num_cell_vertices + 1)]);
    std::vector<uint> cell(num_cell_vertices);
    for (uint j = 0; j < num_cell_vertices; ++j)
      cell[j] = received_cell_vertices[i*(num_cell_vertices + 1) + j + 1];
    mesh_data.cell_vertices.push_back(cell);
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_vertices(LocalMeshData& mesh_data,
                                           std::map<uint, uint>& glob2loc)
{
  // This function distributes all vertices (coordinates and local-to-global
  // mapping) according to the cells that are stored on each process. This
  // happens in several stages: First each process figures out which vertices
  // it needs (by looking at its cells) and where those vertices are located.
  // That information is then distributed so that each process learns where
  // it needs to send its vertices.

  // Get number of processes
  const uint num_processes = MPI::num_processes();

  // Compute which vertices we need
  std::set<uint> needed_vertex_indices;
  std::vector<std::vector<uint> >::const_iterator vertices;
  for (vertices = mesh_data.cell_vertices.begin(); vertices != mesh_data.cell_vertices.end(); ++vertices)
    needed_vertex_indices.insert(vertices->begin(), vertices->end());

  // Compute where (process number) the vertices we need are located
  std::vector<uint> send_vertex_indices, destinations_vertex;
  std::vector<std::vector<uint> > vertex_location(num_processes);
  std::set<uint>::const_iterator required_vertex;
  for (required_vertex = needed_vertex_indices.begin(); required_vertex != needed_vertex_indices.end(); ++required_vertex)
  {
    // Get process that has required vertex
    const uint location = MPI::index_owner(*required_vertex, mesh_data.num_global_vertices);
    destinations_vertex.push_back(location);
    send_vertex_indices.push_back(*required_vertex);
    vertex_location[location].push_back(*required_vertex);
  }

  // Send required vertices to other proceses, and receive back vertices
  // required by othe processes.
  std::vector<uint> received_vertex_indices, sources_vertex;
  MPI::distribute(send_vertex_indices, destinations_vertex,
                  received_vertex_indices, sources_vertex);
  assert(received_vertex_indices.size() == sources_vertex.size());

  // Distribute vertex coordinates
  std::vector<double> send_vertex_coordinates;
  std::vector<uint> destinations_vertex_coordinates;
  const uint vertex_size =  mesh_data.gdim;
  const std::pair<uint, uint> local_vertex_range = MPI::local_range(mesh_data.num_global_vertices);
  for (uint i = 0; i < sources_vertex.size(); ++i)
  {
    assert(received_vertex_indices[i] >= local_vertex_range.first && received_vertex_indices[i] < local_vertex_range.second);
    const uint location = received_vertex_indices[i] - local_vertex_range.first;
    const std::vector<double>& x = mesh_data.vertex_coordinates[location];
    assert(x.size() == vertex_size);
    for (uint j = 0; j < vertex_size; ++j)
    {
      send_vertex_coordinates.push_back(x[j]);
      destinations_vertex_coordinates.push_back(sources_vertex[i]);
    }
  }
  std::vector<double> received_vertex_coordinates;
  std::vector<uint> sources_vertex_coordinates;
  MPI::distribute(send_vertex_coordinates, destinations_vertex_coordinates,
                  received_vertex_coordinates, sources_vertex_coordinates);

  // Set index counters to first position in recieve buffers
  std::vector<uint> index_counters(num_processes, 0);

  // Clear data
  mesh_data.vertex_coordinates.clear();
  mesh_data.vertex_indices.clear();
  glob2loc.clear();

  // Store coordinates and construct global to local mapping
  const uint num_local_vertices = received_vertex_coordinates.size()/vertex_size;
  for (uint i = 0; i < num_local_vertices; ++i)
  {
    std::vector<double> vertex(vertex_size);
    for (uint j = 0; j < vertex_size; ++j)
      vertex[j] = received_vertex_coordinates[i*vertex_size+j];
    mesh_data.vertex_coordinates.push_back(vertex);

    const uint sender_process = sources_vertex_coordinates[i*vertex_size];
    const uint global_vertex_index = vertex_location[sender_process][index_counters[sender_process]++];
    glob2loc[global_vertex_index] = i;
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_mesh(Mesh& mesh,
                                  const LocalMeshData& mesh_data,
                                  std::map<uint, uint>& glob2loc)
{
  Timer timer("PARALLEL 3: Build mesh (from local mesh data)");

  // Get number of processes and process number
  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Open mesh for editing
  mesh.clear();
  MeshEditor editor;
  editor.open(mesh, mesh_data.gdim, mesh_data.tdim);

  // Add vertices
  editor.init_vertices(mesh_data.vertex_coordinates.size());
  Point p(mesh_data.gdim);
  for (uint i = 0; i < mesh_data.vertex_coordinates.size(); ++i)
  {
    for (uint j = 0; j < mesh_data.gdim; ++j)
      p[j] = mesh_data.vertex_coordinates[i][j];
    editor.add_vertex(i, p);
  }

  // Add cells
  editor.init_cells(mesh_data.cell_vertices.size());
  const uint num_cell_vertices = mesh_data.tdim + 1;
  std::vector<uint> cell(num_cell_vertices);
  for (uint i = 0; i < mesh_data.cell_vertices.size(); ++i)
  {
    for (uint j = 0; j < num_cell_vertices; ++j)
      cell[j] = glob2loc[mesh_data.cell_vertices[i][j]];
    editor.add_cell(i, cell);
  }

  // Construct local to global mapping based on the global to local mapping
  MeshFunction<unsigned int>& global_vertex_indices = mesh.parallel_data().global_entity_indices(0);
  for (std::map<uint, uint>::const_iterator iter = glob2loc.begin(); iter != glob2loc.end(); ++iter)
    global_vertex_indices[iter->second] = iter->first;

  // Construct local to global mapping for cells
  MeshFunction<unsigned int>& global_cell_indices = mesh.parallel_data().global_entity_indices(mesh_data.tdim);
  const std::vector<uint>& gci = mesh_data.global_cell_indices;
  assert(global_cell_indices.size() > 0);
  assert(global_cell_indices.size() == gci.size());
  for(uint i = 0; i < gci.size(); ++i)
    global_cell_indices[i] = gci[i];

  // Close mesh: Note that this must be done after creating the global vertex map or
  // otherwise the ordering in mesh.close() will be wrong (based on local numbers).
  editor.close();

  // Construct array of length topology().dim() that holds the number of global mesh entities
  std::vector<uint>& num_global_entities = mesh.parallel_data().num_global_entities();
  num_global_entities.resize(mesh_data.tdim + 1);
  std::fill(num_global_entities.begin(), num_global_entities.end(), 0);

  num_global_entities[0] = mesh_data.num_global_vertices;
  num_global_entities[mesh_data.tdim] = mesh_data.num_global_cells;

  /// Communicate global number of boundary vertices to all processes

  // Construct boundary mesh
  BoundaryMesh bmesh(mesh);
  const MeshFunction<unsigned int>& boundary_vertex_map = bmesh.vertex_map();
  const uint boundary_size = boundary_vertex_map.size();

  // Build sorted array of global boundary vertex indices (global numbering)
  std::vector<uint> global_vertex_send(boundary_size);
  for (uint i = 0; i < boundary_size; ++i)
    global_vertex_send[i] = global_vertex_indices[boundary_vertex_map[i]];
  std::sort(global_vertex_send.begin(), global_vertex_send.end());

  // Distribute boundaries' sizes
  std::vector<uint> boundary_sizes;
  MPI::all_gather(boundary_size, boundary_sizes);

  // Find largest boundary size (for recv buffer)
  //const uint max_boundary_size = *std::max_element(boundary_sizes.begin(), boundary_sizes.end());

  // Recieve buffer
  std::vector<uint> global_vertex_recv;

  // Create shared_vertices data structure: mapping from shared vertices to list of neighboring processes
  std::map<uint, std::vector<uint> >& shared_vertices = mesh.parallel_data().shared_vertices();
  shared_vertices.clear();

  // Distribute boundaries and build mappings
  for (uint i = 1; i < num_processes; ++i)
  {
    // We send data to process p - i (i steps to the left)
    const int p = (process_number - i + num_processes) % num_processes;

    // We receive data from process p + i (i steps to the right)
    const int q = (process_number + i) % num_processes;

    // Send and receive
    //MPI::send_recv(&global_vertex_send[0], boundary_size, p, &global_vertex_recv[0], boundary_sizes[q], q);
    MPI::send_recv(global_vertex_send, p, global_vertex_recv, q);

    // Compute intersection of global indices
    //std::vector<uint> intersection(std::min(boundary_size, boundary_sizes[q]));
    std::vector<uint> intersection(std::min(global_vertex_send.size(), global_vertex_recv.size()));
    std::vector<uint>::iterator intersection_end = std::set_intersection(
         global_vertex_send.begin(), global_vertex_send.end(),
         global_vertex_recv.begin(), global_vertex_recv.end(),
         intersection.begin());

    //std::vector<uint>::iterator intersection_end = std::set_intersection(
    //     global_vertex_send.begin(), global_vertex_send.end(),
    //     &global_vertex_recv[0], &global_vertex_recv[0] + boundary_sizes[q],
    //     intersection.begin());

    // Fill shared vertices information
    std::vector<uint>::const_iterator index;
    for (index = intersection.begin(); index != intersection_end; ++index)
      shared_vertices[*index].push_back(q);
  }
}
//-----------------------------------------------------------------------------
bool MeshPartitioning::in_overlap(const std::vector<uint>& entity,
                             const std::map<uint, std::vector<uint> >& shared)
{
  std::vector<uint>::const_iterator e;
  for (e = entity.begin(); e != entity.end(); ++e)
  {
    if (shared.find(*e) == shared.end())
      return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
void MeshPartitioning::mark_nonshared(const std::map<std::vector<uint>, uint>& entities,
               const std::map<std::vector<uint>, uint>& shared_entity_indices,
               const std::map<std::vector<uint>, uint>& ignored_entity_indices,
               MeshFunction<bool>& exterior)
{
  // Set all to false (not exterior)
  exterior.set_all(false);

  const Mesh& mesh = exterior.mesh();
  const uint D = mesh.topology().dim();

  assert(exterior.dim() == D - 1);

  // FIXME: Check that everything is correctly initalised

  // Add facets that are connected to one cell only
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    if (facet->num_entities(D) == 1)
      exterior[*facet] = true;
  }

  // Remove all entities on internal partition boundaries
  std::map<std::vector<uint>, uint>::const_iterator it;
  for (it = shared_entity_indices.begin(); it != shared_entity_indices.end(); ++it)
    exterior[entities.find(it->first)->second] = false;
  for (it = ignored_entity_indices.begin(); it != ignored_entity_indices.end(); ++it)
    exterior[entities.find(it->first)->second] = false;
}
//-----------------------------------------------------------------------------
