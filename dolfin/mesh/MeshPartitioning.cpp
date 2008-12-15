// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Garth N. Wells, 2008.
// Modified by Niclas Jansson, 2008.
//
// First added:  2007-04-03
// Last changed: 2008-12-09

#include <dolfin/log/log.h>
#include <dolfin/main/MPI.h>
#include <dolfin/graph/Graph.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/graph/GraphPartition.h>
#include <dolfin/parameter/parameters.h>
#include <dolfin/mesh/LocalMeshData.h>
#include "Cell.h"
#include "Facet.h"
#include "Vertex.h"
#include "MeshFunction.h"
#include "MeshPartitioning.h"

#if defined HAS_PARMETIS && HAS_MPI

#include <parmetis.h>
#include <mpi.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshPartitioning::partition(Mesh& mesh, LocalMeshData& data)
{
  dolfin_debug("Partitioning mesh...");

  // Compute cell partition
  std::vector<uint> cell_partition;
  compute_partition(cell_partition, data);

  // Distribute cells
  distribute_cells(cell_partition, data);

  // Distribute vertices
  distribute_vertices(data);

  // Build mesh
  build_mesh(mesh, data);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::compute_partition(std::vector<uint>& cell_partition,
                                         const LocalMeshData& data)
{
  dolfin_debug("Computing cell partition using ParMETIS...");

  // Get number of processes and process number
  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Get dimensions of local data
  const uint num_local_cells = data.cell_vertices().size();
  const uint num_cell_vertices = data.cell_vertices()[0].size();
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
    dolfin_assert(data.cell_vertices()[i].size() == num_cell_vertices);
    eptr[i] = i * num_cell_vertices;
    for (uint j = 0; j < num_cell_vertices; j++)
      eind[eptr[i] + j] = data.cell_vertices()[i][j];
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

  // Copy data
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
void MeshPartitioning::distribute_cells(LocalMeshData& data,
                                        const std::vector<uint>& cell_partition)
{
  dolfin_debug("Distributing mesh data according to cell partition...");

  // Get dimensions of local data
  const uint num_local_cells = data.cell_vertices().size();
  const uint num_cell_vertices = data.cell_vertices()[0].size();

  // Build array of cell-vertex connectivity and partition vector
  std::vector<uint> cell_vertices;
  std::vector<uint> cell_vertices_partition;
  cell_vertices.reserve(num_local_cells*num_cell_vertices);
  cell_vertices_partition.reserve(num_local_cells*num_cell_vertices);
  for (uint i = 0; i < num_local_cells; i++)
  {
    for (uint j = 0; j < num_cell_vertices; j++)
    {
      cell_vertices.push_back(data.cell_vertices()[i][j]);
      cell_vertices_partition.push_back(cell_partition[i]);
    }
  }
  
  // Distribute cell-vertex connectivity
  MPI::distribute(cell_vertices, cell_vertices_partition);
  cell_vertices_partition.clear();

  // FIXME: Put data back into data.cell_vertices
  data.cell_vertices.clear();
  
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_vertices(LocalMeshData& data)
{

  // Compute needs: std::set<uint> needed_vertex_indices
  // Compute where they are located (simple formula): foo_partition
  // MPI::distribute(needed_vertex_indices, vertex_partition);
  // Build arrays: vertex_coordinates, vertex_coordinates_partition

  // MPI::distribute(vertex_coordinates, vertex_coordinates_partition);

  // FIXME: Put data back into data.vertex_coordinates
  data.vertex_coordinates.clear();

}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_mesh(Mesh& mesh, const LocalMeshData& data)
{
  // FIXME: Use MeshEditor to build Mesh from LocalMeshData

  MeshEditor editor;
  editor.open(mesh);
  
  editor.initVertices(data.vertex_coordinates.size());
  editor.initCells(data.cell_vertices.size());

  for (...)
    editor.addVertex(...);

  for (...)
    editor.addCell(...);

  editor.close();
}
//-----------------------------------------------------------------------------
#else

//-----------------------------------------------------------------------------
void MeshPartitioning::partition(Mesh& mesh, LocalMeshData& data)
{
  error("Mesh partitioning requires MPI and ParMETIS.");
}
//-----------------------------------------------------------------------------
void MeshPartitioning::partition_vertices(const LocalMeshData& data,
                                          int* part)
{
  error("Mesh partitioning requires MPI and ParMETIS.");
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_vertices(LocalMeshData& data,
                                           const int* part)
{
  error("Mesh partitioning requires MPI and ParMETIS.");
}
//-----------------------------------------------------------------------------
void MeshPartitioning::partition_cells()
{
  error("Mesh partitioning requires MPI and ParMETIS.");
}
//-----------------------------------------------------------------------------

#endif
