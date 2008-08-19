// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-04-03
// Last changed: 2008-08-17

#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include "GraphPartition.h"

#ifdef HAS_SCOTCH
extern "C"
{
  #include <scotch.h>
}
#endif

using namespace dolfin;

//-----------------------------------------------------------------------------
void GraphPartition::partition(Graph& graph, uint num_part, uint* vtx_part)
{
  if(num_part == 0)
    error("Minimum number of partitions is 1");

#ifdef HAS_SCOTCH

  SCOTCH_Graph grafdat;
  SCOTCH_Strat strat;

  if (SCOTCH_graphInit(&grafdat) != 0) 
  {
    // FIXME: Why do we have a control statement here?
  }
  if (SCOTCH_graphBuild(&grafdat, 0, static_cast<int>(graph.numVertices()), 
                        reinterpret_cast<int*>(graph.offsets()), NULL, NULL, 
                        NULL, static_cast<int>(graph.numEdges()), 
                        reinterpret_cast<int*>(graph.connectivity()), NULL) != 0) 
  {
    // FIXME: Why do we have a control statement here?
  }

  SCOTCH_stratInit(&strat);

  // Only some graphs successfully partitioned, why?
  if (SCOTCH_graphPart (&grafdat, num_part, &strat, reinterpret_cast<int*>(vtx_part)) != 0) 
  {
    // FIXME: Why do we have a control statement here?
  }

  SCOTCH_stratExit (&strat);
  SCOTCH_graphExit (&grafdat);

#else
  error("GraphPartition requires SCOTCH");
#endif
}
//-----------------------------------------------------------------------------
void GraphPartition::check(Graph& graph, uint num_part, uint* vtx_part)
{
  cout << "Checking that all vertices are partitioned" << endl;

  // Check that all vertices are partitioned
  for(uint i=0; i < graph.numVertices(); ++i)
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
  cout << "Evaluating partition quality" << endl;

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
  cout << "partition\tnum_vtx" << endl;
  for(uint i=0; i<num_part; ++i)
    cout << i << "\t\t" << part_sizes[i] << endl;

  cout << "edge-cut: " << edgecut(graph, num_part, vtx_part) << endl;
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
  cout << "Number of partitions: " << num_part << endl;
  cout << "Partition vector" << endl;

  for(uint i = 0; i < graph.numVertices(); ++i)
    cout << vtx_part[i] << " ";
  cout << endl;
}
//-----------------------------------------------------------------------------
