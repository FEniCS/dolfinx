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
  
  // Compute geometric partitioning of vertices
  int* part = new int[data.vertex_coordinates().size()];
  partition_vertices(data, part);

  // Redistribute local mesh data according to partition
  distribute_vertices(data, part);

  // Compute topological partitioning of cells
  partition_cells();

  // Cleanup
  delete [] part;
}
//-----------------------------------------------------------------------------
void MeshPartitioning::partition_vertices(const LocalMeshData& data,
                                          int* part)
{
  dolfin_debug("Computing geometric partitioning of vertices...");

  // Get number of processes and process number
  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Get dimensions of local data
  const uint num_local_vertices = data.vertex_coordinates().size();
  dolfin_assert(num_local_vertices > 0);
  const uint gdim = data.vertex_coordinates()[0].size();

  // FIXME: Why is this necessary?
  // Duplicate MPI communicator
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  // Communicate number of vertices between all processors
  int* vtxdist = new int[num_processes + 1];
  vtxdist[process_number] = static_cast<int>(num_local_vertices);
  dolfin_debug("Communicating vertex distribution across processors...");
  MPI_Allgather(&vtxdist[process_number], 1, MPI_INT, 
                 vtxdist,                 1, MPI_INT, MPI_COMM_WORLD);

  // Build vtxdist array with vertex offsets for all processor
  int sum = vtxdist[0];
  vtxdist[0] = 0;
  for (uint i = 1; i < num_processes + 1; ++i)
  {
    const int tmp = vtxdist[i];
    vtxdist[i] = sum;
    sum += tmp;
  }

  // Prepare arguments for ParMetis
  int ndims = static_cast<int>(gdim);
  float* xyz = new float[gdim*num_local_vertices];
  for (uint i = 0; i < num_local_vertices; ++i)
    for (uint j = 0; j < gdim; ++j)
      xyz[i*gdim + j] = data.vertex_coordinates()[i][j];

  // Call ParMETIS to partition vertex distribution array
  dolfin_debug("Calling ParMETIS to distribute vertices");
  ParMETIS_V3_PartGeom(vtxdist, &ndims, xyz, part, &comm);
  dolfin_debug("Done calling ParMETIS to distribute vertices");

  // Cleanup
  delete [] vtxdist;
  delete [] xyz;
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_vertices(LocalMeshData& data,
                                           const int* part)
{
  dolfin_debug("Distributing local mesh data according to vertex partition...");

  
}
//-----------------------------------------------------------------------------
void MeshPartitioning::partition_cells()
{
  dolfin_debug("Computing topological partitioning of cells...");  
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
