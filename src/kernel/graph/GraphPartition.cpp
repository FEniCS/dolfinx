// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-03
// Last changed: 2007-04-24

#include <dolfin/GraphPartition.h>
#include <iostream>
#include <deque>

using namespace dolfin;

void GraphPartition::partition(Graph& graph, uint num_part, uint* vtx_part)
{
  // Basic breadth first algorithm
  dolfin_assert(vtx_part);
  dolfin_assert(num_part <= graph.numVertices());

  uint nn = graph.numVertices();
  uint size = nn/num_part;
  dolfin_debug2("nn=%d, size=%d", nn, size);

  // Initialize vtx_part to num_part
  for(uint i=0; i<nn; ++i)
    vtx_part[i] = num_part;

  // Put visited vertices in a queue
  std::deque<uint> start_queue, partition_queue;
  start_queue.push_back(0);
  uint part_size = 0;

  // For each partition
  for(uint i=0; i<num_part; ++i)
  {
    dolfin_debug1("\nPartion no: %d", i);
	 partition_queue.push_back(start_queue.front());
	 start_queue.pop_front();

    // Insert vertices in partition
    while(part_size < size && !partition_queue.empty())
    {
      uint vertex = partition_queue.front();
      dolfin_debug1("Current vertex: %d", vertex);
      partition_queue.pop_front();
      vtx_part[vertex] = i;
      dolfin_debug2("Assigning vertex %d to partition %d", vertex, i);
      part_size++;

      dolfin_debug1("Checking %d neigbors", graph.numEdges(vertex));
      uint found = 0;
      // Look for unvisited neigbors of current vertex
      for(uint j=0; j<graph.numEdges(vertex); ++j)
      {
        int edge_index = (int) (graph.offsets()[(int)vertex] + j);
        uint nvtx = graph.connectivity()[edge_index];
        if(vtx_part[nvtx] == num_part)
        {
			 dolfin_debug1("Found unvisited vertex %d", nvtx);
          found++;
          partition_queue.push_back(nvtx);
			 vtx_part[nvtx] = 0; // Mark vertex as visited
        }
      }
      dolfin_debug1("Found %d unvisited vertices", found);
    }
	 while(!partition_queue.empty())
	 {
      dolfin_debug1("partition_queue.front(): %d", partition_queue.front());
		start_queue.push_back(partition_queue.front());
		partition_queue.pop_front();
	 }
    part_size = 0;
  }

  // Assign remaining vertices to partitions
  while(!start_queue.empty())
  {
    partition_queue.push_back(start_queue.front());
    start_queue.pop_front();

    while(!partition_queue.empty())
	 {
		uint vertex = partition_queue.front();
		partition_queue.pop_front();

	   dolfin_debug1("Found unpartitioned vertex: %d", vertex);
		// Insert vertex into same partition as first neigbor
		int edge_index = (int) (graph.offsets()[(int) vertex ]);
		uint nvtx = graph.connectivity()[edge_index];
		vtx_part[vertex] = vtx_part[nvtx];
		dolfin_debug2("Assigning vertex %d to partition %d", vertex, vtx_part[nvtx]);
		
		// If remaining vertice has unvisited neigbors add to partition queue
      // Look for unvisited neigbors of current vertex
		uint found = 0;
      for(uint j=0; j<graph.numEdges(vertex); ++j)
      {
        int edge_index = (int) (graph.offsets()[(int)vertex] + j);
        uint nvtx = graph.connectivity()[edge_index];
        if(vtx_part[nvtx] == num_part)
        {
			 dolfin_debug1("Found unvisited vertex %d", nvtx);
          found++;
          partition_queue.push_back(nvtx);
			 vtx_part[nvtx] = 0; // Mark vertex as visited
        }
      }
      dolfin_debug1("Found %d unvisited vertices", found);
	 }
	 
  }

}
//-----------------------------------------------------------------------------
void GraphPartition::check(Graph& graph, uint num_part, uint* vtx_part)
{
  std::cout << "Checking that all vertices are partitioned" << std::endl;

  // Check that all vertices are partitioned
  for(uint i=0; i<graph.numVertices(); ++i)
  {
    if(vtx_part[i] == num_part)
      dolfin_error1("Vertex %d not partitioned", i);
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
real GraphPartition::edgecut(Graph& graph, uint num_part, uint* vtx_part)
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
		  dolfin_debug2("Vertex %d not in same partition as vertex %d", i, nvtx);
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

  for(uint i=0; i<graph.numVertices(); ++i)
  {
    std::cout << vtx_part[i];
  }
  std::cout << std::endl;
}
