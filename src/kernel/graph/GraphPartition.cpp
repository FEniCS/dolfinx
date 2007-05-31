// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-03
// Last changed: 2007-05-31

#include <dolfin/GraphPartition.h>
#include <iostream>

#ifdef HAVE_SCOTCH_H
extern "C"
{
  #include <scotch.h>
}
#endif

using namespace dolfin;

//-----------------------------------------------------------------------------
void GraphPartition::partition(Graph& graph, uint num_part, uint* vtx_part)
{
  #ifdef HAVE_SCOTCH_H

  SCOTCH_Graph grafdat;
  SCOTCH_Strat strat;

  if (SCOTCH_graphInit (&grafdat) != 0) {
  }
  if (SCOTCH_graphBuild (&grafdat, 0, static_cast<int>(graph.numVertices()), reinterpret_cast<int*>(graph.offsets()), NULL, NULL, NULL, static_cast<int>(graph.numArches()), reinterpret_cast<int*>(graph.connectivity()), NULL) != 0) {
  }

  SCOTCH_stratInit(&strat);

  // Only some graphs successfully partitioned, why?
  if (SCOTCH_graphPart (&grafdat, num_part, &strat, reinterpret_cast<int*>(vtx_part)) != 0) {
  }

  SCOTCH_stratExit (&strat);
  SCOTCH_graphExit (&grafdat);

  #else
    error("GraphPartition requires Scotch");
  #endif
}
//-----------------------------------------------------------------------------
void GraphPartition::check(Graph& graph, uint num_part, uint* vtx_part)
{
  std::cout << "Checking that all vertices are partitioned" << std::endl;

  // Check that all vertices are partitioned
  for(uint i=0; i<graph.numVertices(); ++i)
  {
    if(vtx_part[i] == num_part)
      error("Vertex %d not partitioned", i);
  }

  // Check that partitions are continuous
  // One way to do this is by checking (for all partitions) that there is a 
  // path from every vertex in a partition to all other vertices of the 
  // partition.
  /*
  // This does not work
  for(uint i=0; i<graph.numVertices(); ++i)
  {
	 // For all other vertices
	 for(uint j=0; j<i; ++j)
	 {
		// If vertices shares partition check that they are neighbors
		if(vtx_part[i] == vtx_part[j] && !graph.adjacent(i, j))
		{
		  dolfin_error2("Vertex %d not adjacent to vertex %d, but in the same partition", i, j);
		}
	 }
	 for(uint j=i+1; j<graph.numVertices(); ++j)
	 {
		// If vertices shares partition check that they are neighbors
		if(vtx_part[i] == vtx_part[j] && !graph.adjacent(i, j))
		{
		  dolfin_error2("Vertex %d not adjacent to vertex %d, but in the same partition", i, j);
		}
	 }
  }
  */
}
//-----------------------------------------------------------------------------
void GraphPartition::eval(Graph& graph, uint num_part, uint* vtx_part)
{
  std::cout << "Evaluating partition quality" << std::endl;

  // Number of vertices per partition
  uint* part_sizes = new uint[num_part];

  // Initialize part_sizes array to 0
  for(uint i=0; i<num_part; ++i)
    part_sizes[i] = 0;

  // Count number of vertices per partition
  for(uint i=0; i<graph.numVertices(); ++i)
  {
    part_sizes[vtx_part[i]]++;
  }

  // Print number of vertices per partition
  std::cout << "partition\tnum_vtx" << std::endl;
  for(uint i=0; i<num_part; ++i)
    std::cout << i << "\t\t" << part_sizes[i] << std::endl;

  std::cout << "edge-cut: " << edgecut(graph, num_part, vtx_part) << std::endl;
}
//-----------------------------------------------------------------------------
dolfin::uint GraphPartition::edgecut(Graph& graph, uint num_part, uint* vtx_part)
{
  // Calculate edge-cut
  uint edge_cut = 0;
  for(uint i=0; i<graph.numVertices(); ++i)
  {
    for(uint j=0; j<graph.numEdges(i); ++j)
    {
      int edge_index = (int) (graph.offsets()[(int) i] + j);
      uint nvtx = graph.connectivity()[edge_index];
      // If neighbor not in same partition
      if(vtx_part[i] != vtx_part[nvtx])
      {
        //dolfin_debug2("Vertex %d not in same partition as vertex %d", i, nvtx);
        edge_cut++;
      }
    }
  }
  // Edges visited twice
  edge_cut /= 2;

  return edge_cut;
}
//-----------------------------------------------------------------------------
void GraphPartition::disp(Graph& graph, uint num_part, uint* vtx_part)
{
  std::cout << "Number of partitions: " << num_part << std::endl;
  std::cout << "Partition vector" << std::endl;

  for(uint i = 0; i < graph.numVertices(); ++i)
  {
    std::cout << vtx_part[i] << " ";
  }
  std::cout << std::endl;
}
//-----------------------------------------------------------------------------
