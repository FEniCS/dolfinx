// Copyright (C) 2007 Magnus Vikstrom
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-03
// Last changed: 2007-04-22

#ifndef __GRAPH_PARTITIONING_H
#define __GRAPH_PARTITIONING_H

#include <dolfin/constants.h>
#include <dolfin/Graph.h>

namespace dolfin
{
  /// This class provides a set of functions to partition a Graph

  class GraphPartition
  {
  public:
    
    /// Partition a graph into num_part partitions
    static void partition(Graph& graph, uint num_part, uint* vtx_part);

    /// Check partition correctness
    static void check(Graph& graph, uint num_part, uint* vtx_part);

    /// Evaluate partition quality
    static void eval(Graph& graph, uint num_part, uint* vtx_part);

    /// Display partitioning
    static void disp(Graph& graph, uint num_part, uint* vtx_part);
	 
    /// Calculate edge_cut
    static real edgecut(Graph& graph, uint num_part, uint* vtx_part);

    /// Return partition vector (add typemap to dolfin.i instead)
    static uint* create(uint size) { return new uint[size]; }
  };

}

#endif
