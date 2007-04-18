// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-02-12
// Last changed: 2007-03-21

#ifndef __GRAPH_H
#define __GRAPH_H

#include <dolfin/constants.h>
#include <dolfin/Variable.h>
#include <dolfin/Mesh.h>

namespace dolfin
{
  /// A Graph consists of a set of vertices and edges.
  ///
  /// The graph is stored in Compressed Sparse Row (CSR) format. This format
  /// stores edges and vertices separately in two arrays, with the indices 
  /// into these arrays corresponding to the identifier for the vertex or 
  /// edge, respectively. The edge array stores the edge destination 
  /// vertices while the vertice array stores the offset into the edge array.
  /// E.g. the edges connected to vertex 0 are:
  /// edges[vertices[0]], edges[vertices[0]+1], ..., edges[vertices[1]-1].
  ///
  /// Example graph:
  ///      0 -- 1
  ///      | \  |
  ///      |  \ |
  ///      2 -- 3
  ///
  /// Stored as:
  /// 
  /// edges = [123030312]
  /// vertices = [0357]
  ///
  /// CSR format minimizes memory usage and is suitable for large graphs
  /// that do not change.
  
  class Graph : public Variable
  {
    friend class GraphEditor;
    
  public:
    
    /// Enum for different graph types
    enum Type { directed, undirected };
    
    /// Enum for different mesh - graph representations
    // Put this in class MeshPartitioning ?
    enum Representation { nodal, dual };
    
    /// Create empty graph
    Graph();
    
    /// Copy constructor
    Graph(const Graph& graph);
    
    /// Create graph from given file
    Graph(std::string filename);
    
    /// Create graph from mesh
    Graph(Mesh& mesh, Representation type);
    
    /// Create graph from mesh
    Graph(Mesh& mesh, std::string type);

    /// Destructor
    ~Graph();
    
    /// Assignment
    //const Graph& operator=(const Graph& graph);
    
    // Return number of vertices
    inline uint numVertices() const { return num_vertices; }
    
    // Return number of edges
    inline uint numEdges() const { return num_edges; }
    
    // Return number of arches (outgoing edges)
    inline uint numArches() const { return num_arches; }
    
    // Return number of edges incident to vertex u
    unsigned int numEdges(uint u);
    
    // Return edge weights
    inline uint* edgeWeights() const { return edge_weights; }
    
    // Return vertex weights
    inline uint* vertexWeights() const { return vertex_weights; }
    
    // Return array of edges for all vertices
    inline uint* connectivity() const { return edges; }
    
    // Return array of offsets for edges of all vertices
    inline uint* offsets() const { return vertices; }
    
    // Return graph type
    inline Type type() const { return _type; }
    
    // Return graph type as a string
    std::string typestr();
    
    // Display graph data
    void disp();
    
    // Clear graph data
    void clear();
    
    /// Output
    friend LogStream& operator<< (LogStream& stream, const Graph& graph);
    
  private:
    
    uint num_edges;
    uint num_arches;
    uint num_vertices;
    
    uint* edges;
    uint* vertices;
    
    uint* edge_weights;
    uint* vertex_weights;
    
    Type _type;

    Representation _representation;

    void createNodal(Mesh& mesh);
    void createDual(Mesh& mesh);
  };
  
}

#endif
