// Copyright (C) 2006 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-02-12
// Last changed: 2007-03-09

#ifndef __GRAPH_EDITOR_H
#define __GRAPH_EDITOR_H

#include <dolfin/constants.h>
#include <dolfin/Graph.h>

namespace dolfin
{
  
  class Graph;
  
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

    /// Add vertex v
    void addVertex(uint u, uint num_edges);

    /// Add edge from vertex u to vertex v
    void addEdge(uint u, uint v);

    /// Close graph, finish editing
    void close();

  private:

    /// Add arch from vertex u to vertex v
    void addArch(uint u, uint v);

    // Clear all data
    void clear();

    // Next available vertex
    uint next_vertex;

    // Next available arch
    uint next_arch;

    // The mesh
    Graph* graph;

  };

}

#endif
