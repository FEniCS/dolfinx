// Copyright (C) 2008-2009 Niclas Jansson, Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-12-01
// Last changed: 2011-02-03

#include <algorithm>
#include <iterator>

#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/graph/ParMETIS.h>
#include <dolfin/graph/SCOTCH.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "BoundaryMesh.h"
#include "LocalMeshData.h"
#include "Mesh.h"
#include "MeshEditor.h"
#include "MeshEntityIterator.h"
#include "MeshFunction.h"
#include "ParallelData.h"
#include "Point.h"
#include "Vertex.h"
#include "MeshPartitioning.h"


using namespace dolfin;

// Utility functions for debugging/printing
template<typename InputIterator> void print_container(std::ostream& ostr, InputIterator itbegin, InputIterator itend, const char* delimiter=", ")
{
  std::copy(itbegin, itend, std::ostream_iterator<typename InputIterator::value_type>(ostr, delimiter));
}

template<typename InputIterator> void print_container(std::string msg, InputIterator itbegin, InputIterator itend, const char* delimiter=", ")
{
  std::stringstream msg_stream;
  msg_stream << msg;
  print_container(msg_stream, itbegin, itend);
  info(msg_stream.str());
}

template<typename Map> void print_vec_map(std::ostream& ostr, Map map, const char* delimiter=", ")
{
  for (typename Map::iterator it = map.begin(); it !=map.end(); ++it)
  {
    print_container(ostr, (*it).first.begin(), (*it).first.end(), " ");
    ostr << ": " << (*it).second << delimiter;
  }
}

template<typename Map> void print_vec_map(std::string msg, Map map, const char* delimiter=", ")
{
  std::stringstream msg_stream;
  msg_stream << msg << " ";
  print_vec_map(msg_stream, map, delimiter);
  info(msg_stream.str());
}

//-----------------------------------------------------------------------------
void MeshPartitioning::partition(Mesh& mesh)
{
  // Create and distribute local mesh data
  dolfin_debug("creating local mesh data");
  LocalMeshData mesh_data(mesh);
  dolfin_debug("created local mesh data");

  // Partition mesh based on local mesh data
  partition(mesh, mesh_data);
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
    error("Unknown mesh partition '%s'.", partitioner.c_str());

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
void MeshPartitioning::number_entities(const Mesh& _mesh, uint d)
{
  // FIXME: Break up this function

  Timer timer("PARALLEL x: Number mesh entities");
  Mesh& mesh = const_cast<Mesh&>(_mesh);

  // Check for vertices
  if (d == 0)
    error("Unable to number entities of dimension 0. Vertex indices must already exist.");

  // Return if global entity indices are already calculated
  if (mesh.parallel_data().have_global_entity_indices(d))
    return;

  // Get number of processes and process number
  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Get global vertex indices and overlap
  MeshFunction<unsigned int>& global_vertex_indices = mesh.parallel_data().global_entity_indices(0);
  std::map<uint, std::vector<uint> >* overlap = mesh.data().vector_mapping("overlap");
  assert(overlap);

  // Sort overlap
  for (std::map<uint, std::vector<uint> >::iterator it = (*overlap).begin(); it != (*overlap).end(); ++it)
    std::sort((*it).second.begin(), (*it).second.end());

  // Initialize entities of dimension d
  mesh.init(d);

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

  // Iterate over all entities
  for (std::map<std::vector<uint>, uint>::const_iterator it = entities.begin(); it != entities.end(); ++it)
  {
    const std::vector<uint>& entity = (*it).first;
    const uint local_entity_index = (*it).second;

    // Compute which processes entity is shared with
    std::vector<uint> entity_processes;
    if (in_overlap(entity, *overlap))
    {
      std::vector<uint> intersection = (*overlap)[entity[0]];
      std::vector<uint>::iterator intersection_end = intersection.end();

      for (uint i = 1; i < entity.size(); ++i)
      {
        const uint v = entity[i];
        intersection_end = std::set_intersection(intersection.begin(), intersection_end,
                                                 (*overlap)[v].begin(), (*overlap)[v].end(), intersection.begin());
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

  // Qualify boundary entities. We need to find out if the ignored
  // (shared with lower ranked process) entities are entities of a
  // lower ranked process.  If not, this process becomes the lower
  // ranked process for the entity in question, and is therefore
  // responsible for communicating values to the higher ranked
  // processes (if any).

  // Communicate common entities, starting with the entities we think should be ignored
  std::vector<uint> common_entity_values;
  std::vector<uint> common_entity_partition;
  for (std::map<std::vector<uint>, std::vector<uint> >::const_iterator it = ignored_entity_processes.begin();
       it != ignored_entity_processes.end(); ++it)
  {
    // Get entity vertices (global vertex indices)
    const std::vector<uint>& entity = (*it).first;

    // Get entity processes (processes might sharing the entity)
    const std::vector<uint>& entity_processes = (*it).second;

    // Prepare data for sending
    for (uint j = 0; j < entity_processes.size(); ++j)
    {
      const uint p = entity_processes[j];
      common_entity_values.push_back(entity.size());
      common_entity_values.insert(common_entity_values.end(), entity.begin(), entity.end());
      common_entity_partition.insert(common_entity_partition.end(), entity.size() + 1, p);
    }
  }

  // Communicate common entities, add the entities we think should be shared as well
  for (std::map<std::vector<uint>, std::vector<uint> >::const_iterator it = shared_entity_processes.begin();
       it != shared_entity_processes.end(); ++it)
  {
    // Get entity vertices (global vertex indices)
    const std::vector<uint>& entity = (*it).first;

    // Get entity processes (processes might sharing the entity)
    const std::vector<uint>& entity_processes = (*it).second;

    // Prepare data for sending
    for (uint j = 0; j < entity_processes.size(); ++j)
    {
      const uint p = entity_processes[j];
      assert(process_number < p);
      common_entity_values.push_back(entity.size());
      common_entity_values.insert(common_entity_values.end(), entity.begin(), entity.end());
      common_entity_partition.insert(common_entity_partition.end(), entity.size() + 1, p);
    }
  }

  // Communicate common entities
  MPI::distribute(common_entity_values, common_entity_partition);

  // Check if entities received are really entities
  std::vector<uint> is_entity_values;
  std::vector<uint> is_entity_partition;
  for (uint i = 0; i < common_entity_values.size();)
  {
    // Get entity
    const uint p = common_entity_partition[i];
    const uint entity_size = common_entity_values[i++];
    std::vector<uint> entity;
    for (uint j = 0; j < entity_size; ++j)
      entity.push_back(common_entity_values[i++]);

    // Check if it is an entity (in which case it will be in ignored or shared entities)
    uint is_entity = 0;
    if ((ignored_entity_indices.count(entity) > 0 ) || (shared_entity_indices.count(entity) > 0 ))
    {
      is_entity = 1;
    }

    // Add information about entity (whether it's actually an entity) to send to other processes
    is_entity_values.push_back(entity_size);
    is_entity_partition.push_back(p);
    for (uint j = 0; j < entity_size; ++j)
    {
      is_entity_values.push_back(entity[j]);
      is_entity_partition.push_back(p);
    }
    is_entity_values.push_back(is_entity);
    is_entity_partition.push_back(p);
  }

  // Clean up no longer needed data structures
  common_entity_values.clear();
  common_entity_partition.clear();

  // Send data back (list of requested entities that are really entities)
  MPI::distribute(is_entity_values, is_entity_partition);

  // Create map from entities to processes where it is an entity
  std::map<std::vector<uint>, std::vector<uint> > entity_processes;
  for (uint i = 0; i < is_entity_values.size();)
  {
    const uint p = is_entity_partition[i];
    const uint entity_size = is_entity_values[i++];
    std::vector<uint> entity;
    for (uint j = 0; j < entity_size; ++j)
      entity.push_back(is_entity_values[i++]);
    const uint is_entity = is_entity_values[i++];
    if (is_entity == 1)
    {
      // Add entity since it is actually an entity for process p
      entity_processes[entity].push_back(p);
    }
  }

  // Clean up no longer needed data structures
  is_entity_values.clear();
  is_entity_partition.clear();

  // Fix the list of entities we ignore (numbered by lower ranked process)
  std::vector<std::vector<uint> > unignore_entities;
  for (std::map<std::vector<uint>, uint>::const_iterator it = ignored_entity_indices.begin(); it != ignored_entity_indices.end(); ++it)
  {
    const std::vector<uint> entity = (*it).first;
    const uint local_entity_index = (*it).second;
    if (entity_processes.count(entity) > 0)
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
  for (uint i=0; i < unignore_entities.size(); ++i)
  {
    ignored_entity_indices.erase(unignore_entities[i]);
    ignored_entity_processes.erase(unignore_entities[i]);
  }
  unignore_entities.clear();

  // Fix the list of entities we share
  std::vector<std::vector<uint> > unshare_entities;
  for (std::map<std::vector<uint>, uint>::const_iterator it = shared_entity_indices.begin(); it != shared_entity_indices.end(); ++it)
  {
    const std::vector<uint>& entity = (*it).first;
    const uint local_entity_index = (*it).second;
    if (entity_processes.count(entity) == 0)
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
  for (uint i=0; i < unshare_entities.size(); ++i)
  {
    shared_entity_indices.erase(unshare_entities[i]);
    shared_entity_processes.erase(unshare_entities[i]);
  }
  unshare_entities.clear();

  // Create mesh markers for exterior facets
  if (d == (mesh.topology().dim() - 1))
  {
    mark_nonshared(entities, shared_entity_indices, ignored_entity_indices,
                   mesh, d, "exterior facets");
  }

  // Communicate number of entities to number
  std::vector<uint> num_entities_to_number(num_processes);
  std::fill(num_entities_to_number.begin(), num_entities_to_number.end(), 0);
  num_entities_to_number[process_number] = owned_entity_indices.size() + shared_entity_indices.size();
  MPI::gather(num_entities_to_number);

  // Compute offset
  uint offset = 0;
  for (uint i = 0; i < process_number; ++i)
    offset += num_entities_to_number[i];

  // Compute number of global entities
  uint num_global = offset;
  for (uint i = process_number; i < num_processes; ++i)
    num_global += num_entities_to_number[i];

  // Store number of global entities
  std::vector<uint>* num_global_entities = mesh.data().array("num global entities");
  assert(num_global_entities);
  (*num_global_entities)[d] = num_global;

  // Prepare list of entity numbers. Check later that nothing is -1
  std::vector<int> entity_indices(mesh.size(d));
  std::fill(entity_indices.begin(), entity_indices.end(), -1);

  // Number owned entities
  for (std::map<std::vector<uint>, uint>::const_iterator it = owned_entity_indices.begin();
      it != owned_entity_indices.end(); ++it)
  {
    entity_indices[(*it).second] = offset++;
  }

  // Number shared entities
  for (std::map<std::vector<uint>, uint>::const_iterator it = shared_entity_indices.begin();
        it != shared_entity_indices.end(); ++it)
  {
    entity_indices[(*it).second] = offset++;
  }

  // Communicate indices for shared entities and get indices for ignored
  std::vector<uint> values;
  std::vector<uint> partition;
  for (std::map<std::vector<uint>, uint>::const_iterator it = shared_entity_indices.begin();
        it != shared_entity_indices.end(); ++it)
  {
    // Get entity index
    const uint local_entity_index = (*it).second;
    const int entity_index = entity_indices[local_entity_index];
    assert(entity_index != -1);

    // Get entity vertices (global vertex indices)
    const std::vector<uint>& entity = (*it).first;

    // Get entity processes (processes sharing the entity)
    const std::vector<uint>& entity_processes = shared_entity_processes[entity];

    // Prepare data for sending
    for (uint j = 0; j < entity_processes.size(); ++j)
    {
      // Store interleaved: entity index, number of vertices, global vertex indices
      values.push_back(entity_index);
      values.push_back(entity.size());
      values.insert(values.end(), entity.begin(), entity.end());
      partition.insert(partition.end(), entity.size() + 2, entity_processes[j]);
    }
  }

  // Clean up no longer needed data structures
  entity_processes.clear();
  shared_entity_indices.clear();
  shared_entity_processes.clear();

  // Send data
  MPI::distribute(values, partition);

  // Fill in global entity indices recieved from lower ranked processes
  for (uint i = 0; i < values.size();)
  {
    const uint p = partition[i];
    const uint global_index = values[i++];
    const uint entity_size = values[i++];
    std::vector<uint> entity;
    for (uint j = 0; j < entity_size; ++j)
      entity.push_back(values[i++]);

    // Sanity check, should not receive an entity we don't need
    if (ignored_entity_indices.count(entity) == 0)
    {
      std::stringstream msg;
      msg << "Process " << MPI::process_number() << " received illegal entity given by ";
      print_container(msg, entity.begin(), entity.end());
      msg << " with global index " << global_index;
      msg << " from process " << p;
      error(msg.str());
    }

    const uint local_entity_index = ignored_entity_indices[entity];
    assert(entity_indices[local_entity_index] == -1);
    entity_indices[local_entity_index] = global_index;
  }

  // Clean up no longer needed data structures
  values.clear();
  partition.clear();
  ignored_entity_indices.clear();
  ignored_entity_processes.clear();

  // Create mesh data
  MeshFunction<unsigned int>& global_entity_indices = mesh.parallel_data().global_entity_indices(d);
  for (uint i = 0; i < entity_indices.size(); ++i)
  {
    if (entity_indices[i] < 0)
      info(WARNING, "Missing global number for local entity (%d, %d).", d, i);
    assert(entity_indices[i] >= 0);
    assert(i < global_entity_indices.size());
    global_entity_indices[i] = static_cast<uint>(entity_indices[i]);
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_cells(LocalMeshData& mesh_data,
                                        const std::vector<uint>& cell_partition)
{
  // This function takes the partition computed by ParMETIS (which tells us
  // to which process each of the local cells stored in LocalMeshData on this
  // process belongs. We use MPI::distribute to redistribute all cells (the
  // global vertex indices of all cells).

  // Get dimensions of local mesh_data
  uint num_local_cells = mesh_data.cell_vertices.size();
  assert(mesh_data.global_cell_indices.size() == num_local_cells);
  const uint num_cell_vertices = mesh_data.cell_vertices[0].size();

  // Build array of cell-vertex connectivity and partition vector
  // Distribute the global cell number as well
  std::vector<uint> cell_vertices;
  std::vector<uint> cell_vertices_partition;
  cell_vertices.reserve(num_local_cells*(num_cell_vertices+1));
  cell_vertices_partition.reserve(num_local_cells*(num_cell_vertices+1));
  for (uint i = 0; i < num_local_cells; i++)
  {
    cell_vertices.push_back(mesh_data.global_cell_indices[i]);
    cell_vertices_partition.push_back(cell_partition[i]);
    for (uint j = 0; j < num_cell_vertices; j++)
    {
      cell_vertices.push_back(mesh_data.cell_vertices[i][j]);
      cell_vertices_partition.push_back(cell_partition[i]);
    }
  }

  // Distribute cell-vertex connectivity
  MPI::distribute(cell_vertices, cell_vertices_partition);
  cell_vertices_partition.clear();

  // Put mesh_data back into mesh_data.cell_vertices
  mesh_data.cell_vertices.clear();
  mesh_data.global_cell_indices.clear();
  assert(cell_vertices.size() % (num_cell_vertices + 1) == 0);
  num_local_cells = cell_vertices.size() / (num_cell_vertices + 1);
  mesh_data.cell_vertices.reserve(num_local_cells);
  mesh_data.global_cell_indices.reserve(num_local_cells);
  for (uint i = 0; i < num_local_cells; ++i)
  {
    mesh_data.global_cell_indices.push_back(cell_vertices[i*(num_cell_vertices + 1)]);
    std::vector<uint> cell(num_cell_vertices);
    for (uint j = 0; j < num_cell_vertices; ++j)
      cell[j] = cell_vertices[i*(num_cell_vertices + 1) + j + 1];
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
  std::vector<std::vector<uint> >& cell_vertices = mesh_data.cell_vertices;
  for (uint i = 0; i < cell_vertices.size(); ++i)
    for (uint j = 0; j < cell_vertices[i].size(); ++j)
      needed_vertex_indices.insert(cell_vertices[i][j]);

  // Compute where (process number) the vertices we need are located
  std::vector<uint> vertex_partition;
  std::vector<std::vector<uint> > vertex_location(num_processes);
  std::vector<uint> vertex_indices;
  std::set<uint>::const_iterator it;
  for (it = needed_vertex_indices.begin(); it != needed_vertex_indices.end(); ++it)
  {
    const uint location = MPI::index_owner(*it, mesh_data.num_global_vertices);
    vertex_partition.push_back(location);
    vertex_indices.push_back(*it);
    vertex_location[location].push_back(*it);
  }
  needed_vertex_indices.clear();
  MPI::distribute(vertex_indices, vertex_partition);
  assert(vertex_indices.size() == vertex_partition.size());

  // Distribute vertex coordinates
  std::vector<double> vertex_coordinates;
  std::vector<uint> vertex_coordinates_partition;
  const uint vertex_size =  mesh_data.vertex_coordinates[0].size();
  std::pair<uint, uint> local_vertex_range = MPI::local_range(mesh_data.num_global_vertices);
  for (uint i = 0; i < vertex_partition.size(); ++i)
  {
    assert(vertex_indices[i] >= local_vertex_range.first && vertex_indices[i] < local_vertex_range.second);
    const uint location = vertex_indices[i] - local_vertex_range.first;
    const std::vector<double>& x = mesh_data.vertex_coordinates[location];
    assert(x.size() == vertex_size);
    for (uint j = 0; j < vertex_size; ++j)
    {
      vertex_coordinates.push_back(x[j]);
      vertex_coordinates_partition.push_back(vertex_partition[i]);
    }
  }
  MPI::distribute(vertex_coordinates, vertex_coordinates_partition);

  // Set index counters to first position in recieve buffers
  std::vector<uint> index_counters(num_processes);
  std::fill(index_counters.begin(), index_counters.end(), 0);

  // Store coordinates and construct global to local mapping
  mesh_data.vertex_coordinates.clear();
  mesh_data.vertex_indices.clear();
  glob2loc.clear();
  const uint num_local_vertices = vertex_coordinates.size()/vertex_size;
  for (uint i = 0; i < num_local_vertices; ++i)
  {
    std::vector<double> vertex(vertex_size);
    for (uint j = 0; j < vertex_size; ++j)
      vertex[j] = vertex_coordinates[i*vertex_size+j];
    mesh_data.vertex_coordinates.push_back(vertex);
    const uint sender_process = vertex_coordinates_partition[i*vertex_size];
    const uint global_vertex_index
        = vertex_location[sender_process][index_counters[sender_process]++];
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
    global_vertex_indices[(*iter).second] = (*iter).first;

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
  std::vector<uint>* num_global_entities = mesh.data().create_array("num global entities", mesh_data.tdim + 1);
  for (uint d = 0; d <= mesh_data.tdim; ++d)
    (*num_global_entities)[d] = 0;

  (*num_global_entities)[0] = mesh_data.num_global_vertices;
  (*num_global_entities)[mesh_data.tdim] = mesh_data.num_global_cells;

  /// Communicate global number of boundary vertices to all processes

  // Construct boundary mesh
  BoundaryMesh bmesh(mesh);
  boost::shared_ptr<MeshFunction<unsigned int> > boundary_vertex_map = bmesh.data().mesh_function("vertex map");
  assert(boundary_vertex_map);
  const uint boundary_size = boundary_vertex_map->size();

  // Build sorted array of global boundary vertex indices (global numbering)
  std::vector<uint> global_vertex_send(boundary_size);
  for (uint i = 0; i < boundary_size; ++i)
    global_vertex_send[i] = global_vertex_indices[(*boundary_vertex_map)[i]];
  std::sort(global_vertex_send.begin(), global_vertex_send.end());

  // Distribute boundaries' sizes
  std::vector<uint> boundary_sizes(num_processes);
  boundary_sizes[process_number] = boundary_size;
  MPI::gather(boundary_sizes);

  // Find largest boundary size (for recv buffer)
  const uint max_boundary_size = *(std::max_element(boundary_sizes.begin(), boundary_sizes.end()));

  // Recieve buffer
  std::vector<uint> global_vertex_recv(max_boundary_size);

  // Create overlap: mapping from shared vertices to list of neighboring processes
  std::map<uint, std::vector<uint> >* overlap = mesh.data().create_vector_mapping("overlap");

  // Distribute boundaries and build mappings
  for (uint i = 1; i < num_processes; ++i)
  {
    // We send data to process p - i (i steps to the left)
    const int p = (process_number - i + num_processes) % num_processes;

    // We receive data from process p + i (i steps to the right)
    const int q = (process_number + i) % num_processes;

    // Send and receive
    MPI::send_recv(&global_vertex_send[0], boundary_size, p, &global_vertex_recv[0], boundary_sizes[q], q);

    // Compute intersection of global indices
    std::vector<uint> intersection(std::min(boundary_size, boundary_sizes[q]));
    std::vector<uint>::iterator intersection_end = std::set_intersection(
         global_vertex_send.begin(), global_vertex_send.end(),
         &global_vertex_recv[0], &global_vertex_recv[0] + boundary_sizes[q],
         intersection.begin());

    // Fill overlap information
    for (std::vector<uint>::const_iterator index = intersection.begin();
        index != intersection_end; ++index)
    {
      (*overlap)[*index].push_back(q);
    }
  }
}
//-----------------------------------------------------------------------------
bool MeshPartitioning::in_overlap(const std::vector<uint>& entity,
                                  std::map<uint, std::vector<uint> >& overlap)
{
  for (uint i = 0; i < entity.size(); ++i)
    if (overlap.count(entity[i]) == 0)
      return false;
  return true;
}
//-----------------------------------------------------------------------------
void MeshPartitioning::mark_nonshared(const std::map<std::vector<uint>, uint>& entities,
               const std::map<std::vector<uint>, uint>& shared_entity_indices,
               const std::map<std::vector<uint>, uint>& ignored_entity_indices,
               Mesh& mesh, uint d, std::string name)
{
  // Create mesh markers and mark all
  boost::shared_ptr<MeshFunction<unsigned int> > exterior = mesh.data().create_mesh_function(name, d);
  exterior->set_all(1);

  // Remove all entities in the overlap
  std::map<std::vector<uint>, uint>::const_iterator it;
  for (it = shared_entity_indices.begin(); it != shared_entity_indices.end(); ++it)
    (*exterior)[entities.find(it->first)->second] = 0;
  for (it = ignored_entity_indices.begin(); it != ignored_entity_indices.end(); ++it)
    (*exterior)[entities.find(it->first)->second] = 0;
}
//-----------------------------------------------------------------------------
