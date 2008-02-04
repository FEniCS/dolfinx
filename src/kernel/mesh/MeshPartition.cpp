// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2007-04-03
// Last changed: 2008-02-04

#include <dolfin/Graph.h>
#include <dolfin/GraphPartition.h>
#include <dolfin/MeshPartition.h>
#include <dolfin/MeshFunction.h>
#include <dolfin/parameters.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshPartition::partition(Mesh& mesh,
                              MeshFunction<uint>& partitions,
                              uint num_partitions)
{
  dolfin_debug1("partitions.init(mesh, %d)", mesh.topology().dim());
  partitions.init(mesh, mesh.topology().dim());
  Graph graph(mesh);
  GraphPartition::partition(graph, num_partitions, partitions.values());

  bool report_edge_cut = get("report edge cut");
  if(report_edge_cut)
    GraphPartition::edgecut(graph, num_partitions, partitions.values());
}
//-----------------------------------------------------------------------------
