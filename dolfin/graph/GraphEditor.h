// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-02-12
// Last changed: 2008-08-19


#ifndef __GRAPH_EDITOR_H
#define __GRAPH_EDITOR_H

#include <dolfin/common/types.h>
#include "Graph.h"

namespace dolfin
{
  
  /// A simple graph editor for creating graphs

  class GraphEditor
  {
  public:
    
    /// Constructor
    GraphEditor();
    
    /// Destructor
    ~GraphEditor();

    /// Open graph of given type
    void open(Graph& graph, Graph::Type type);

    /// Open graph of given type
    void open(Graph& graph, std::string type);

    /// Specify number of vertices
    void initVertices(uint num_vertices);
    
    /// Specify number of edges
    void initEdges(uint num_edges);

    /// Add vertex u with num_edges = number of outgoing edges. For undirected 
    /// graphs, edge must "belong" to a vertex and not be counted twice.
    void addVertex(uint u, uint num_edges);

    /// Add edge from vertex u to vertex v
    void addEdge(uint u, uint v);

    /// Close graph, finish editing
    void close();

  private:

    // Clear all data
    void clear();

    // Next available vertex
    uint next_vertex;

    // Count number of edges from addtion of vertices
    uint edge_count;

    // The mesh
    Graph* graph;

  };

}

#endif
