// Copyright (C) 2008 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-12-01
// Last changed: 2009-04-22

#include <sstream>
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
  std::vector<std::vector<uint> > entities(mesh.size(d));
  for (MeshEntityIterator e(mesh, d); !e.end(); ++e)
  {
    std::vector<uint> entity_vertices;
    for (VertexIterator vertex(*e); !vertex.end(); ++vertex)
      entity_vertices.push_back(global_vertex_indices->get(vertex->index()));
    std::sort(entity_vertices.begin(), entity_vertices.end());
    entities[e->index()] = entity_vertices;
  }
  
  /// Find out which entities to ignore, which to number and which to number
  /// and send to other processes. Entities shared by two or more processes
  /// are numbered by the lower ranked process.

  // Entities to ignore (shared with lower rank process)
  std::map<std::vector<uint>, uint> ignored_entities;
  ignored_entities.clear();

  // Entities to number
  std::vector<uint> owned_entities;
    
  // Entites to number and send (shared with higher rank processes)
  std::vector<uint> shared_entities;
  std::vector<std::vector<uint> > shared_entity_vertices;
  std::vector<std::vector<uint> > shared_entity_processes;

  // Iterate over all entities
  for (uint e = 0; e < entities.size(); ++e)
  {
    const std::vector<uint>& entity_vertices = entities[e];

    // Compute which processes entity is shared with
    std::vector<uint> entity_processes;
    if (in_overlap(entity_vertices, *overlap))
    {
      std::vector<uint> intersection = (*overlap)[entity_vertices[0]];
      std::vector<uint>::iterator intersection_end = intersection.end();
      std::vector<uint>::iterator iter;

      for (uint i = 1; i < entity_vertices.size(); ++i)
      {
        const uint v = entity_vertices[i];

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
      owned_entities.push_back(e);
    else if (ignore)
      ignored_entities[entity_vertices] = e;
    else
    {
      owned_entities.push_back(e);
      shared_entities.push_back(e);
      shared_entity_vertices.push_back(entity_vertices);
      shared_entity_processes.push_back(entity_processes);
    }
  }
     
  // Communicate all offsets
  std::vector<uint> offsets(num_processes);
  std::fill(offsets.begin(), offsets.end(), 0);
  offsets[process_number] = owned_entities.size();
  MPI::gather(offsets);

  // Compute offset
  uint global_offset = 0;
  for (uint i = 0; i < process_number; ++i)
    global_offset += offsets[i];

  // Number owned entities
  std::vector<int> entity_indices(mesh.size(d));
  std::fill(entity_indices.begin(), entity_indices.end(), -1);
  for (uint i = 0; i < owned_entities.size(); ++i)
  {
    const uint e = owned_entities[i];
    entity_indices[e] = global_offset + i;
  }

  // Communicate indices for shared entities
  std::vector<uint> values;
  std::vector<uint> partition;
  for (uint i = 0; i < shared_entities.size(); ++i)
  {
    // Get entity index
    const uint e = shared_entities[i];
    const int entity_index = entity_indices[e];
    dolfin_assert(entity_index != -1);

    // Get entity vertices (global vertex indices)
    const std::vector<uint>& entity_vertices = shared_entity_vertices[i];

    // Get entity processes (processes sharing the entity)
    const std::vector<uint>& entity_processes = shared_entity_processes[i];

    // Prepare data for sending
    for (uint j = 0; j < entity_processes.size(); ++j)
    {   
      // Store interleaved: entity index, number of vertices, global vertex indices
      values.push_back(entity_index);
      values.push_back(entity_vertices.size());
      for (uint k = 0; k < entity_vertices.size(); ++k)
        values.push_back(entity_vertices[k]);

      // Processes to communicate values to
      const uint p = entity_processes[j];
      for (uint k = 0; k < 2 + entity_vertices.size(); ++k)
        partition.push_back(p);
    }
  }

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

    if (ignored_entities.count(entity) == 0) 
      error("Recieved global value for non-ignored entity.");

    entity_indices[ignored_entities[entity]] = global_index;
  }
  
  // Create mesh data
  std::stringstream name;
  name << "global entity indices " << d;
  MeshFunction<uint>* global_entity_indices = mesh.data().create_mesh_function(name.str(), d);
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
  message("Partitioned mesh, edge cut is %d.", edgecut);

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
  std::map<uint, uint>::iterator iter; 
  for (iter = glob2loc.begin(); iter != glob2loc.end(); ++iter)
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
    for (std::vector<uint>::iterator index = intersection.begin(); index != intersection_end; ++index)
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
bool MeshPartitioning::in_overlap(const std::vector<uint>& entity_vertices,
                                  std::map<uint, std::vector<uint> >& overlap)
{
  for (uint i = 0; i < entity_vertices.size(); ++i)
    if (overlap.count(entity_vertices[i]) == 0)
      return false;
  return true;
}
//-----------------------------------------------------------------------------
