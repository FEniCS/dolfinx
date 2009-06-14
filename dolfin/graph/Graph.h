// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
// Modified by Anders Logg, 2008-2009.
//
// First added:  2007-02-12
// Last changed: 2009-04-27

#ifndef __GRAPH_H
#define __GRAPH_H

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{
  class XMLGraph;
  class LogStream;
  class Mesh;

  /// A Graph consists of a set of vertices and edges.
  ///
  /// The graph is stored in Compressed Sparse Row (CSR) format. This format
  /// stores edges and vertices separately in two arrays, with the indices
  /// into these arrays corresponding to the identifier for the vertex or
  /// edge, respectively. The edge array stores the edge destination
  /// vertices while the vertice array stores the offset into the edge array.
  /// E.g. the edges connected to vertex i are:
  /// edges[vertices[i]], edges[vertices[i]+1], ..., edges[vertices[i]-1].
  ///
  /// In a graph with n vertices the vertex array will be of size n+1. The
  /// edge array will be of size m in a directed graph and size 2m in a
  /// undirected graph (an edge between vertices u and v is stored as
  /// (v,u) as well as (u,v)).
  ///
  /// Example graph:
  ///      0 -- 1
  ///      | \  |
  ///      |  \ |
  ///      2 -- 3
  ///
  /// Stored as:
  ///
  /// edges    = [1 2 3 0 3 0 3 0 1 2]
  /// vertices = [0 3 5 7 10]
  ///
  /// Note that the last integer of vertices does not represent a vertex, but
  /// is there to support edge iteration as described above.
  ///
  /// CSR format minimizes memory usage and is suitable for large graphs
  /// that do not change.

  class Graph : public Variable
  {
    friend class GraphEditor;
    friend class GraphBuilder;

  public:

    /// Enum for different graph types
    enum Type { directed, undirected };

    /// Enum for different mesh - graph representations
    // Put this in class MeshPartitioning ?
    enum Representation { nodal, dual };

    /// Create empty graph
    Graph();

    /// Create graph of mesh
    Graph(Mesh& mesh, Graph::Representation rep = dual);

    /// Copy constructor
    Graph(const Graph& graph);

    /// Create graph from given file
    Graph(std::string filename);

    /// Destructor
    ~Graph();

    /// Initialise graph data structures
    void init(uint num_vertices, uint num_edges);

    /// Return number of vertices
    inline uint num_vertices() const { return _num_vertices; }

    /// Return number of edges
    inline uint num_edges() const { return _num_edges; }

    /// Return number of edges incident to vertex u
    inline uint num_edges(uint u) const { return _vertices[u+1] - _vertices[u]; }

    /// Check if vertex u is adjacent to vertex v
    bool adjacent(uint u, uint v);

    /// Return edge weights
    inline uint* edge_weights() const { return _edge_weights; }

    /// Return vertex weights
    inline uint* vertex_weights() const { return _vertex_weights; }

    /// Return array of edges for all vertices
    inline uint* connectivity() const { return _edges; }

    /// Return array of offsets for edges of all vertices
    inline uint* offsets() const { return _vertices; }

    /// Return graph type
    inline Type type() const { return _type; }

    /// Partition a graph into num_part partitions
    void partition(uint num_part, uint* vtx_part);

    /// Return graph type as a string
    std::string typestr() const;

    /// Display graph data
    void disp();

    /// Clear graph data
    void clear();

    /// Output
    friend LogStream& operator<< (LogStream& stream, const Graph& graph);

    typedef XMLGraph XMLHandler;

  private:

    uint _num_edges;
    uint _num_vertices;

    uint* _edges;
    uint* _vertices;

    uint* _edge_weights;
    uint* _vertex_weights;

    Type _type;

    Representation _representation;

  };

}

#endif
