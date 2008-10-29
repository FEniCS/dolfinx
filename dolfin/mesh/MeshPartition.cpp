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
#include <dolfin/graph/Graph.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/graph/GraphPartition.h>
#include <dolfin/parameter/parameters.h>
#include "Cell.h"
#include "Facet.h"
#include "Vertex.h"
#include "MeshFunction.h"
#include "MeshPartition.h"

#ifdef HAS_PARMETIS
#include <parmetis.h>
#endif

#ifdef HAS_MPI
#include <mpi.h>
#endif

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshPartition::partition(Mesh& mesh, MeshFunction<uint>& partitions,
                              uint num_partitions)
{
  // Initialise MeshFunction partitions
  partitions.init(mesh, mesh.topology().dim());

  // Create graph
  Graph graph(mesh, Graph::dual);

  // Partition graph
  GraphPartition::partition(graph, num_partitions, partitions.values());

  bool report_edge_cut = dolfin_get("report edge cut");
  if(report_edge_cut)
    GraphPartition::edgecut(graph, num_partitions, partitions.values());
}
//-----------------------------------------------------------------------------
void MeshPartition::partition(std::string meshfile, uint num_partitions)
{
  File infile(meshfile);
  Mesh mesh;
  infile >> mesh;
  MeshFunction<uint> partitions(mesh, mesh.topology().dim());
  partition(mesh, partitions, num_partitions);

  error("MeshPartition::partition(std::string meshfile, uint num_partitions) not implemented");
  for (FacetIterator f(mesh); !f.end(); ++f) 
  {
    for (CellIterator c(*f); !c.end(); ++c) 
    {
      // Do the dirty work here.
    }
  }
}
//-----------------------------------------------------------------------------
void MeshPartition::partition(Mesh& mesh, MeshFunction<uint>& partitions)
{
  partitionCommonMetis(mesh, partitions, 0);
}
//-----------------------------------------------------------------------------
void MeshPartition::partition(Mesh& mesh, MeshFunction<uint>& partitions,
			      MeshFunction<uint>& weight)
{
  partitionCommonMetis(mesh, partitions, &weight);
}
//-----------------------------------------------------------------------------
void MeshPartition::partitionCommonMetis(Mesh& mesh, 
					 MeshFunction<uint>& partitions,
					 MeshFunction<uint>* weight)
{
  // FIXME add code for dual graph based partitioning,
}
//-----------------------------------------------------------------------------

