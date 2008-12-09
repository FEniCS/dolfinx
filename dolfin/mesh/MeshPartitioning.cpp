// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Garth N. Wells, 2008.
// Modified by Niclas Jansson, 2008.
//
// First added:  2007-04-03
// Last changed: 2008-10-29

#include <dolfin/log/log.h>
#include <dolfin/main/MPI.h>
#include <mpi.h>
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
void MeshPartitioning::partition(Mesh& mesh, const LocalMeshData& data)
{
  dolfin_debug("Partitioning mesh...");
  
  // Get number of processes and process number
  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Get dimensions of local data
  const uint num_local_vertices = data.vertex_coordinates().size();
  const uint num_local_cells = data.cell_vertices().size();
  dolfin_assert(num_local_vertices > 0);
  dolfin_assert(num_local_cells > 0);
  const uint gdim = data.vertex_coordinates()[0].size();

  // Duplicate MPI communicator
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  // Prepare arguments for ParMetis

  // Communicate vertex distribution to all processes
  int* vtxdist = new int[num_processes+1];
  vtxdist[process_number] = static_cast<int>(num_local_vertices);
  dolfin_debug("Communicating vertex distribution across processors...");
  MPI_Allgather(&vtxdist[process_number], 1, MPI_INT, 
                 vtxdist,                 1, MPI_INT, MPI_COMM_WORLD);

  int sum = vtxdist[0];
  vtxdist[0] = 0;
  for (uint i = 1; i < num_processes + 1; ++i)
  {
    const int tmp = vtxdist[i];
    vtxdist[i] = sum;
    sum += tmp;
  }  

  int ndims = static_cast<int>(gdim);
  int* part = new int[num_local_vertices];
  float* xyz = new float[gdim*num_local_vertices];
  for (uint i = 0; i < num_local_vertices; ++i)
    for (uint j = 0; j < gdim; ++j)
      xyz[i*gdim + j] = data.vertex_coordinates()[i][j];

  // Call ParMETIS to partition vertex distribution array
  dolfin_debug("Calling ParMETIS to distribute vertices");
  ParMETIS_V3_PartGeom(vtxdist, &ndims, xyz, part, &comm);
  dolfin_debug("Done calling ParMETIS to distribute vertices");
}
//-----------------------------------------------------------------------------
void MeshPartitioning::partition(Mesh& mesh, MeshFunction<uint>& partitions,
                                 uint num_partitions)
{
  // FIXME: This is an old implementation that keeps a single global mesh
  // FIXME: on all processors. Should this be removed?

  // Initialise MeshFunction partitions
  partitions.init(mesh, mesh.topology().dim());

  // Create graph
  Graph graph(mesh, Graph::dual);

  // Partition graph
  GraphPartition::partition(graph, num_partitions, partitions.values());

  bool report_edge_cut = dolfin_get("report edge cut");
  if (report_edge_cut)
    GraphPartition::edgecut(graph, num_partitions, partitions.values());
}
//-----------------------------------------------------------------------------
void MeshPartitioning::partition_vertices()
{


}
//-----------------------------------------------------------------------------
void MeshPartitioning::partition_cells()
{

}
//-----------------------------------------------------------------------------

#endif
