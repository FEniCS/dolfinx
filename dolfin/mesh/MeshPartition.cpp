// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2007-04-03
// Last changed: 2008-02-11

#include <dolfin/graph/Graph.h>
#include <dolfin/graph/GraphPartition.h>
#include "MeshPartition.h"
#include "MeshFunction.h"
#include <dolfin/parameter/parameters.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshPartition::partition(Mesh& mesh,
                              MeshFunction<uint>& partitions,
                              uint num_partitions)
{
  partitions.init(mesh, mesh.topology().dim());
  Graph graph(mesh);
  GraphPartition::partition(graph, num_partitions, partitions.values());

  bool report_edge_cut = dolfin_get("report edge cut");
  if(report_edge_cut)
    GraphPartition::edgecut(graph, num_partitions, partitions.values());
}
//-----------------------------------------------------------------------------
