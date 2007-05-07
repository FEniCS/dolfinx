// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-03
// Last changed: 2007-04-24

#include <dolfin/Graph.h>
#include <dolfin/GraphPartition.h>
#include <dolfin/MeshPartition.h>

using namespace dolfin;

void MeshPartition::partition(Mesh& mesh, uint num_part, uint* vtx_part)
{
  Graph graph(mesh);
  GraphPartition::partition(graph, num_part, vtx_part);
}
