// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-03
// Last changed: 2007-04-18

#include <dolfin/GraphPartition.h>
#include <iostream>
#include <deque>

using namespace dolfin;

void GraphPartition::partition(Graph& graph, uint num_part, uint* vtx_part)
{
  // Basic breadth first algorithm
  dolfin_assert(vtx_part);
  dolfin_assert(num_part < graph.numVertices());

  uint nn = graph.numVertices();
  uint size = nn/num_part;
  dolfin_debug2("nn=%d, size=%d", nn, size);

  // Initialize vtx_part to num_part
  for(uint i=0; i<nn; ++i)
    vtx_part[i] = num_part;

  // Put visited vertices in a queue
  std::deque<uint> q;
  q.push_back(0);
  vtx_part[0] = 0;
  uint sum_partitioned = 0;
  uint part_size = 1;

  // For each partition
  for(uint i=0; i<num_part; ++i)
  {
    dolfin_debug1("\nPartion no: %d", i);

    // Insert vertices in partition
    while(part_size < size)
    {
      uint vertex = q.front();
      dolfin_debug1("Current vertex: %d", vertex);
      q.pop_front();
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
          found++;
          q.push_back(nvtx);
          vtx_part[nvtx] = i;
        }
      }
      dolfin_debug1("Found %d unvisited vertices", found);
    }
    sum_partitioned += part_size;
    part_size = 0;
  }

  // Assign remaining vertices to partitions
  while(!q.empty())
  {
    uint vertex = q.front();
    q.pop_front();
    dolfin_debug1("Found unpartitioned vertex: %d", vertex);

    // Insert vertex into same partition as first neigbor
    int edge_index = (int) (graph.offsets()[(int) vertex ]);
    uint nvtx = graph.connectivity()[edge_index];
    vtx_part[vertex] = vtx_part[nvtx];
    dolfin_debug2("Assigning vertex %d to partition %d", vertex, vtx_part[nvtx]);
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

  // Calculate edge-cut
  uint edge_cut = 0;
  for(uint i=0; i<graph.numVertices(); ++i)
  {
    for(uint j=0; j<graph.numEdges(i); ++j)
    {
      int edge_index = (int) (graph.offsets()[(int) i] + j);
      uint nvtx = graph.connectivity()[edge_index];
      // If neighbor not in same partition
      if(vtx_part[i] == vtx_part[nvtx])
      {
        edge_cut++;
      }
    }
  }
  // Edges visited twice
  edge_cut /= 2;

  std::cout << "edge-cut: " << edge_cut << std::endl;
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
