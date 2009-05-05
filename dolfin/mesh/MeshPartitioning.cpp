// Copyright (C) 2008 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-12-01
// Last changed: 2009-05-05

#include <vector>
#include <algorithm>
#include <dolfin/log/log.h>
#include <dolfin/main/MPI.h>
#include "LocalMeshData.h"
#include "Point.h"
#include "Vertex.h"
#include "BoundaryMesh.h"
#include "MeshEntityIterator.h"
#include "MeshEditor.h"
#include "MeshFunction.h"
#include "MeshPartitioning.h"

#if defined HAS_PARMETIS

#include <parmetis.h>

using namespace dolfin;



//-----------------------------------------------------------------------------
void MeshPartitioning::partition(Mesh& mesh, LocalMeshData& mesh_data)
{
  dolfin_debug("Partitioning mesh...");

  // Compute cell partition
  std::vector<uint> cell_partition;
  compute_partition(cell_partition, mesh_data);

  // Distribute cells
  distribute_cells(mesh_data, cell_partition);

  // Distribute vertices
  std::map<uint, uint> glob2loc;
  distribute_vertices(mesh_data, glob2loc);

  // Build mesh
  build_mesh(mesh, mesh_data, glob2loc);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::number_entities(Mesh& mesh, uint d)
{
  // Check for vertices
  if (d == 0)
    error("Unable to number entities of dimension 0. Vertex indices must already exist.");

  info("Computing global numbers for mesh entities of dimension %d", d);

  // Get number of processes and process number
  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Get global vertex indices and overlap
  MeshFunction<uint>* global_vertex_indices = mesh.data().mesh_function("global entity indices 0");
  std::map<uint, std::vector<uint> >* overlap = mesh.data().vector_mapping("overlap");
  dolfin_assert(global_vertex_indices);
  dolfin_assert(overlap);

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
      entity.push_back(global_vertex_indices->get(vertex->index()));
    std::sort(entity.begin(), entity.end());
    entities[entity] = e->index();
  }
  
  /// Find out which entities to ignore, which to number and which to number
  /// and send to other processes. Entities shared by two or more processes
  /// are numbered by the lower ranked process.

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
    const uint e = (*it).second;


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
      owned_entity_indices[entity] = e;
    else if (ignore)
    {
      ignored_entity_indices[entity] = e;
      ignored_entity_processes[entity] = entity_processes;
    }
    else
    {
      shared_entity_indices[entity] = e;
      shared_entity_processes[entity] = entity_processes;
    }
  }

  // Qualify boundary entities
  // We need to find out if the ignored (shared with lower ranked process) entities are entities of a lower ranked process.
  // If not, this process becomes the lower ranked process for the entity in question, and is therefore responsible for 
  // communicating values to the higher ranked processes (if any).

  
  // Communicate common entitis, starting with the entities we think should be ignored 
  std::vector<uint> common_entity_values;
  std::vector<uint> common_entity_partition;
  for (std::map<std::vector<uint>, std::vector<uint> >::const_iterator it = ignored_entity_processes.begin(); it != ignored_entity_processes.end(); ++it)
  {
    // Get entity vertices (global vertex indices)
    const std::vector<uint>& entity = (*it).first;

    // Get entity processes (processes might sharing the entity)
    const std::vector<uint>& entity_processes = (*it).second;

    // Prepare data for sending
    for (uint j = 0; j < entity_processes.size(); ++j)
    {   
      const uint p = entity_processes[j];
      if ( p < process_number)
      {
        common_entity_values.push_back(entity.size());
        common_entity_values.insert(common_entity_values.end(), entity.begin(), entity.end());
        common_entity_partition.insert(common_entity_partition.end(), entity.size() + 1, p);
      }
    }
  }

  // Communicate common entitis, add the entities we think should be shared as well 
  for (std::map<std::vector<uint>, std::vector<uint> >::const_iterator it = shared_entity_processes.begin(); it != shared_entity_processes.end(); ++it)
  {
    // Get entity vertices (global vertex indices)
    const std::vector<uint>& entity = (*it).first;

    // Get entity processes (processes might sharing the entity)
    const std::vector<uint>& entity_processes = (*it).second;

    // Prepare data for sending
    for (uint j = 0; j < entity_processes.size(); ++j)
    {   
      const uint p = entity_processes[j];
      if ( p < process_number)
      {
        common_entity_values.push_back(entity.size());
        common_entity_values.insert(common_entity_values.end(), entity.begin(), entity.end());
        common_entity_partition.insert(common_entity_partition.end(), entity.size() + 1, p);
      }
    }

  }

  // Send data
  MPI::distribute(common_entity_values, common_entity_partition);

  std::vector<uint> is_entity_values;
  std::vector<uint> is_entity_partition;
  for (uint i = 0; i < common_entity_values.size();)
  {
    const uint p = common_entity_partition[i];
    const uint entity_size = common_entity_values[i++];
    std::vector<uint> entity;
    uint is_entity = 0;
    for (uint j=0; j < entity_size; ++j)
      entity.push_back(common_entity_values[i++]);

    if ( (ignored_entity_indices.count(entity) > 0 ) or ( shared_entity_indices.count(entity) > 0 ) )
    {
        is_entity = 1;
    }
    
    is_entity_values.push_back(entity_size);
    is_entity_partition.push_back(p);

    for (uint k=0; k < entity_size; ++k)
    {
      is_entity_values.push_back(entity[k]);
      is_entity_partition.push_back(p);
    }
    is_entity_values.push_back(is_entity);
    is_entity_partition.push_back(p);
  }

  common_entity_values.clear();
  common_entity_partition.clear();

  // Send data back (list of entities that should not be ignored by the process)
  MPI::distribute(is_entity_values, is_entity_partition);

  // entity_processes is the list of edges in the overlap that are actually edges
  std::map<std::vector<uint>, std::vector<uint> > entity_processes;
  for (uint i = 0; i < is_entity_values.size();)
  {
    const uint p = is_entity_partition[i];
    const uint entity_size = is_entity_values[i++];
    std::vector<uint> entity;
    for (uint j=0; j < entity_size; ++j)
      entity.push_back(is_entity_values[i++]);
    const uint is_entity = is_entity_values[i++];
    if (is_entity == 1)
    {
      // The local entity is an entity of a lower ranked process, so we can safely ignore it.
      entity_processes[entity].push_back(p);
    }
  }

  is_entity_values.clear();
  is_entity_partition.clear();

  // Fix the list of entities we ignore (numbered by lower ranked process)
  for (std::map<std::vector<uint>, uint>::const_iterator it = ignored_entity_indices.begin(); it != ignored_entity_indices.end(); ++it)
  {
    const std::vector<uint> entity = (*it).first;
    const uint e = (*it).second;
    if (entity_processes.count(entity) > 0 )
    {
      std::vector<uint> common_processes = entity_processes[entity];
      dolfin_assert(common_processes.size() > 0);
      const uint max_proc = *(std::max_element(common_processes.begin(), common_processes.end()));

      if ( max_proc > process_number )
      {
        // Move from ignored to shared
        shared_entity_indices[entity] = e;
        shared_entity_processes[entity] = common_processes;

        // Add entity to list of entities that should be removed from the ignored entity list.
        ignored_entity_indices.erase(entity);
        ignored_entity_processes.erase(entity);
      }

    } else
    {
      // Move from ignored to owned
      owned_entity_indices[entity] = e;

      // Add entity to list of entities that should be removed from the ignored entity list.
      ignored_entity_indices.erase(entity);
      ignored_entity_processes.erase(entity);
    }
  }

  // Fix the list of entities we share 
  for (std::map<std::vector<uint>, uint>::const_iterator it = shared_entity_indices.begin(); it != shared_entity_indices.end(); ++it)
  {
    const std::vector<uint> entity = (*it).first;
    const uint e = (*it).second;
    if (entity_processes.count(entity) == 0 )
    {
      // Move from ignored to owned
      owned_entity_indices[entity] = e;
      shared_entity_indices.erase(entity);
      shared_entity_processes.erase(entity);
    }
  }

  // Communicate all offsets
  std::vector<uint> offsets(num_processes);
  std::fill(offsets.begin(), offsets.end(), 0);
  offsets[process_number] = owned_entity_indices.size() + shared_entity_indices.size();
  MPI::gather(offsets);

  // Compute offset
  uint offset = 0;
  for (uint i = 0; i < process_number; ++i)
    offset += offsets[i];

  // Number owned entities
  std::vector<int> entity_indices(mesh.size(d));
  std::fill(entity_indices.begin(), entity_indices.end(), -1);
  for (std::map<std::vector<uint>, uint>::const_iterator it = owned_entity_indices.begin(); it != owned_entity_indices.end(); ++it)
    entity_indices[(*it).second] = offset++;

  for (std::map<std::vector<uint>, uint>::const_iterator it = shared_entity_indices.begin(); it != shared_entity_indices.end(); ++it)
    entity_indices[(*it).second] = offset++;

  // Communicate indices for shared entities
  std::vector<uint> values;
  std::vector<uint> partition;
  for (std::map<std::vector<uint>, uint>::const_iterator it = shared_entity_indices.begin(); it != shared_entity_indices.end(); ++it)
  {
    // Get entity index
    const uint e = (*it).second;
    const int entity_index = entity_indices[e];
    dolfin_assert(entity_index != -1);

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

  entity_processes.clear();
  shared_entity_indices.clear();
  shared_entity_processes.clear();

  // Send data
  MPI::distribute(values, partition);

  // Fill in global entity indices recieved from lower ranked processes
  for (uint i = 0; i < values.size();)
  {
    const uint global_index = values[i++];
    const uint entity_size = values[i++];
    std::vector<uint> entity;
    for (uint j=0; j < entity_size; ++j)
      entity.push_back(values[i++]);

    if (ignored_entity_indices.count(entity) == 0) 
    {
      std::stringstream msg;
      msg << "Erroneously got enity given by ";
      print_container(msg, entity.begin(), entity.end());
      msg << " with global index " << global_index;
      warning(msg.str());
    }

    entity_indices[ignored_entity_indices[entity]] = global_index;
  }

  values.clear();
  partition.clear();
  ignored_entity_indices.clear();
  ignored_entity_processes.clear();
  
  // Create mesh data
  std::stringstream mesh_data_name;
  mesh_data_name << "global entity indices " << d;
  MeshFunction<uint>* global_entity_indices = mesh.data().create_mesh_function(mesh_data_name.str(), d);
  for (uint i = 0; i < entity_indices.size(); ++i)
    global_entity_indices->set(i, entity_indices[i]);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::compute_partition(std::vector<uint>& cell_partition,
                                         const LocalMeshData& mesh_data)
{
  // This function prepares data for ParMETIS (which is a pain
  // since ParMETIS has the worst possible interface), calls
  // ParMETIS, and then collects the results from ParMETIS.

  dolfin_debug("Computing cell partition using ParMETIS...");

  // Get number of processes and process number
  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Get dimensions of local mesh_data
  const uint num_local_cells = mesh_data.cell_vertices.size();
  const uint num_cell_vertices = mesh_data.cell_vertices[0].size();
  dolfin_debug1("num_local_cells = %d", num_local_cells);

  // Communicate number of cells between all processors
  std::vector<uint> num_cells(num_processes);
  num_cells[process_number] = num_local_cells;
  MPI::gather(num_cells);

  // Build elmdist array with cell offsets for all processors
  int* elmdist = new int[num_processes + 1];
  elmdist[0] = 0;
  for (uint i = 1; i < num_processes + 1; ++i)
    elmdist[i] = elmdist[i - 1] + num_cells[i - 1];

  // Build eptr and eind arrays storing cell-vertex connectivity
  int* eptr = new int[num_local_cells + 1];
  int* eind = new int[num_local_cells * num_cell_vertices];
  for (uint i = 0; i < num_local_cells; i++)
  {
    dolfin_assert(mesh_data.cell_vertices[i].size() == num_cell_vertices);
    eptr[i] = i * num_cell_vertices;
    for (uint j = 0; j < num_cell_vertices; j++)
      eind[eptr[i] + j] = mesh_data.cell_vertices[i][j];
  }
  eptr[num_local_cells] = num_local_cells * num_cell_vertices;

  // Number of nodes shared for dual graph (partition along facets)
  int ncommonnodes = num_cell_vertices - 1;

  // Number of partitions (one for each process)
  int nparts = num_processes;

  // Strange weight arrays needed by ParMETIS
  int ncon = 1;
  float* tpwgts = new float[ncon*nparts];
  for (int i = 0; i < ncon*nparts; i++)
    tpwgts[i] = 1.0 / static_cast<float>(nparts);
  float* ubvec = new float[ncon];
  for (int i = 0; i < ncon; i++)
    ubvec[i] = 1.05;

  // Options for ParMETIS, use default
  int options[3];
  options[0] = 0;
  options[1] = 0;
  options[2] = 0;
  
  // Partitioning array to be computed by ParMETIS (note bug in manual: vertices, not cells!)
  int* part = new int[num_local_cells];

  // Prepare remaining arguments for ParMETIS
  int* elmwgt = 0;
  int wgtflag = 0;
  int numflag = 0;
  int edgecut = 0;

  // Construct communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;
  
  // Call ParMETIS to partition mesh
  ParMETIS_V3_PartMeshKway(elmdist, eptr, eind,
                           elmwgt, &wgtflag, &numflag, &ncon,
                           &ncommonnodes, &nparts,
                           tpwgts, ubvec, options,
                           &edgecut, part, &(*comm)); 
  info("Partitioned mesh, edge cut is %d.", edgecut);

  // Copy mesh_data
  cell_partition.clear();
  cell_partition.reserve(num_local_cells);
  for (uint i = 0; i < num_local_cells; i++)
    cell_partition.push_back(static_cast<uint>(part[i]));
  
  // Cleanup
  delete [] elmdist;
  delete [] eptr;
  delete [] eind;
  delete [] tpwgts;
  delete [] ubvec;
  delete [] part;
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_cells(LocalMeshData& mesh_data,
                                        const std::vector<uint>& cell_partition)
{
  // This function takes the partition computed by ParMETIS (which tells us
  // to which process each of the local cells stored in LocalMeshData on this
  // process belongs. We use MPI::distribute to redistribute all cells (the
  // global vertex indices of all cells).

  dolfin_debug("Distributing cells...");

  // Get dimensions of local mesh_data
  uint num_local_cells = mesh_data.cell_vertices.size();
  const uint num_cell_vertices = mesh_data.cell_vertices[0].size();

  // Build array of cell-vertex connectivity and partition vector
  std::vector<uint> cell_vertices;
  std::vector<uint> cell_vertices_partition;
  cell_vertices.reserve(num_local_cells*num_cell_vertices);
  cell_vertices_partition.reserve(num_local_cells*num_cell_vertices);
  for (uint i = 0; i < num_local_cells; i++)
  {
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
  num_local_cells = cell_vertices.size()/num_cell_vertices;
  for (uint i = 0; i < num_local_cells; ++i)
  {
    std::vector<uint> cell(num_cell_vertices);
    for (uint j = 0; j < num_cell_vertices; ++j)
      cell[j] = cell_vertices[i*num_cell_vertices+j];
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

  dolfin_debug("Distributing vertices...");
  
  // Compute which vertices we need
  std::set<uint> needed_vertex_indices;
  std::vector<std::vector<uint> >& cell_vertices = mesh_data.cell_vertices;
  for (uint i = 0; i < cell_vertices.size(); ++i)
    for (uint j = 0; j < cell_vertices[i].size(); ++j)
      needed_vertex_indices.insert(cell_vertices[i][j]);

  // Compute where (process number) the vertices we need are located
  std::vector<uint> vertex_partition;
  std::vector<std::vector<uint> > vertex_location(mesh_data.num_processes);
  std::vector<uint> vertex_indices;
  std::set<uint>::const_iterator it;
  for (it = needed_vertex_indices.begin(); it != needed_vertex_indices.end(); ++it)
  {
    const uint location = mesh_data.initial_vertex_location(*it);
    vertex_partition.push_back(location);
    vertex_indices.push_back(*it);
    vertex_location[location].push_back(*it);
  }
  needed_vertex_indices.clear(); 
  MPI::distribute(vertex_indices, vertex_partition);
  dolfin_assert(vertex_indices.size() == vertex_partition.size());

  // Distribute vertex coordinates
  std::vector<double> vertex_coordinates;
  std::vector<uint> vertex_coordinates_partition;
  const uint vertex_size =  mesh_data.vertex_coordinates[0].size();
  for (uint i = 0; i < vertex_partition.size(); ++i)
  {
    const std::vector<double>& x = mesh_data.vertex_coordinates[mesh_data.local_vertex_number(vertex_indices[i])];
    dolfin_assert(x.size() == vertex_size);
    for (uint j = 0; j < vertex_size; ++j)
    {
      vertex_coordinates.push_back(x[j]);
      vertex_coordinates_partition.push_back(vertex_partition[i]);
    }
  }
  MPI::distribute(vertex_coordinates, vertex_coordinates_partition);

  // Set index counters to first position in recieve buffers
  std::vector<uint> index_counters(mesh_data.num_processes);
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
    uint sender_process = vertex_coordinates_partition[i*vertex_size];
    uint global_vertex_index = vertex_location[sender_process][index_counters[sender_process]++];
    glob2loc[global_vertex_index] = i;
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_mesh(Mesh& mesh,
                                  const LocalMeshData& mesh_data,
                                  std::map<uint, uint>& glob2loc)
{
  // Open mesh for editing
  MeshEditor editor;
  editor.open(mesh, mesh_data.cell_type->cell_type(), mesh_data.gdim, mesh_data.tdim);

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
  const uint num_cell_vertices = mesh_data.cell_type->num_entities(0);
  std::vector<uint> cell(num_cell_vertices);
  for (uint i = 0; i < mesh_data.cell_vertices.size(); ++i)
  {
    for (uint j = 0; j < num_cell_vertices; ++j)
      cell[j] = glob2loc[mesh_data.cell_vertices[i][j]];
    editor.add_cell(i, cell);
  }
  editor.close();

  // Construct local to global mapping based on the global to local mapping
  MeshFunction<uint>* global_vertex_indices = mesh.data().create_mesh_function("global entity indices 0", 0);
  dolfin_assert(global_vertex_indices);
  for (std::map<uint, uint>::const_iterator iter = glob2loc.begin(); iter != glob2loc.end(); ++iter)
    global_vertex_indices->set((*iter).second, (*iter).first);

  /// Communicate global number of boundary vertices to all processes

  // Construct boundary mesh 
  BoundaryMesh bmesh(mesh);
  MeshFunction<uint>* boundary_vertex_map = bmesh.data().mesh_function("vertex map");
  dolfin_assert(boundary_vertex_map);
  const uint boundary_size = boundary_vertex_map->size();

  // Build sorted array of global boundary vertex indices (global numbering)
  uint global_vertex_send[boundary_size];
  for (uint i = 0; i < boundary_size; ++i)
    global_vertex_send[i] = global_vertex_indices->get(boundary_vertex_map->get(i));
  std::sort(global_vertex_send, global_vertex_send + boundary_size);

  // Get number of processes and process number
  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Distribute boundaries' sizes
  std::vector<uint> boundary_sizes(num_processes);
  boundary_sizes[process_number] = boundary_size;
  MPI::gather(boundary_sizes);

  // Find largest boundary size (for recv buffer)
  const uint max_boundary_size = *(std::max_element(boundary_sizes.begin(), boundary_sizes.end()));

  // Recieve buffer
  uint global_vertex_recv[max_boundary_size];

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
    MPI::send_recv(global_vertex_send, boundary_size, p, global_vertex_recv, boundary_sizes[q], q);

    // Compute intersection of global indices

    std::vector<uint> intersection(std::min(boundary_size, boundary_sizes[q]));
    std::vector<uint>::iterator intersection_end = std::set_intersection(
         global_vertex_send, global_vertex_send + boundary_size, 
         global_vertex_recv, global_vertex_recv + boundary_sizes[q], intersection.begin()); 


    // Fill overlap information
    for (std::vector<uint>::const_iterator index = intersection.begin(); index != intersection_end; ++index)
      (*overlap)[*index].push_back(q);
  }
}
//-----------------------------------------------------------------------------

#else

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshPartitioning::partition(Mesh& mesh, LocalMeshData& data)
{
  error("Mesh partitioning requires MPI and ParMETIS.");
}
//-----------------------------------------------------------------------------
void MeshPartitioning::number_entities(Mesh& mesh, uint d)
{
  error("Mesh partitioning requires MPI and ParMETIS.");
}
//-----------------------------------------------------------------------------
void MeshPartitioning::compute_partition(std::vector<uint>& cell_partition,
                                         const LocalMeshData& data)
{
  error("Mesh partitioning requires MPI and ParMETIS.");
}

//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_cells(LocalMeshData& data,
                                        const std::vector<uint>& cell_partition)
{
  error("Mesh partitioning requires MPI and ParMETIS.");
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_vertices(LocalMeshData& data,
                                           std::map<uint, uint>& glob2loc)
{
  error("Mesh partitioning requires MPI and ParMETIS.");
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_mesh(Mesh& mesh, const LocalMeshData& data,
                                  std::map<uint, uint>& glob2loc)
{
  error("Mesh partitioning requires MPI and ParMETIS.");
}
//-----------------------------------------------------------------------------

#endif

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
