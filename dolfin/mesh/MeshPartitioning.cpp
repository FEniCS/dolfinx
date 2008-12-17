// Copyright (C) 2008 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-12-01
// Last changed: 2008-12-15

#include <dolfin/log/log.h>
#include <dolfin/main/MPI.h>
#include "LocalMeshData.h"
#include "Point.h"
#include "MeshEditor.h"
#include "MeshPartitioning.h"

#if defined HAS_PARMETIS && HAS_MPI

#include <parmetis.h>
// FIXME: Should not need mpi.h here, just MPI class instead
#include <mpi.h>

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

  // FIXME: Move this part to MPI wrapper
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  
  // Call ParMETIS to partition mesh
  ParMETIS_V3_PartMeshKway(elmdist, eptr, eind,
                           elmwgt, &wgtflag, &numflag, &ncon,
                           &ncommonnodes, &nparts,
                           tpwgts, ubvec, options,
                           &edgecut, part, &comm);
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
  std::vector<uint> index_counters(mesh_data.num_processes);

  // Store coordinates
  mesh_data.vertex_coordinates.clear();
  mesh_data.vertex_indices.clear();
  const uint num_local_vertices = vertex_coordinates.size()/vertex_size;
  for (uint i = 0; i < num_local_vertices; ++i)
  {
    std::vector<double> vertex(vertex_size);
    for (uint j = 0; j < vertex_size; ++j)
      vertex[j] = vertex_coordinates[i*vertex_size+j];
    mesh_data.vertex_coordinates.push_back(vertex);
    uint sender_process = vertex_coordinates_partition[i*vertex_size];
    uint global_vertex_index = vertex_location[sender_process][index_counters[sender_process]++];
    //mesh_data.vertex_indices.push_back(global_vertex_index);
    glob2loc[global_vertex_index] = i;
  }
  for (uint i = 0; i< mesh_data.vertex_indices.size(); ++i)
    glob2loc[mesh_data.vertex_indices[i]] = i;

  // FIXME: Need to store vertex_indices
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_mesh(Mesh& mesh,
                                  const LocalMeshData& mesh_data,
                                  std::map<uint, uint>& glob2loc)
{
  // Open mesh for editing
  MeshEditor editor;
  editor.open(mesh, mesh_data.cell_type->cellType(), mesh_data.gdim, mesh_data.tdim);

  // Add vertices
  editor.initVertices(mesh_data.vertex_coordinates.size());
  Point p(mesh_data.gdim);
  for (uint i = 0; i < mesh_data.vertex_coordinates.size(); ++i)
  {
    for (uint j = 0; j < mesh_data.gdim; ++j)
      p[j] = mesh_data.vertex_coordinates[i][j];
    editor.addVertex(i, p);
  }
  
  // Add cells
  editor.initCells(mesh_data.cell_vertices.size());
  const uint num_vertices = mesh_data.cell_type->numEntities(0);
  Array<uint> a(num_vertices);
  for (uint i = 0; i < mesh_data.cell_vertices.size(); ++i)
  {
    for (uint j = 0; j < num_vertices; ++j)
    {
      const uint idx = mesh_data.cell_vertices[i][j];
      const uint gidx = glob2loc[idx];
      a[j] = gidx;
    }
    editor.addCell(i, a);
  }

  editor.close();
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
