// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BOUNDARY_DATA_H
#define __BOUNDARY_DATA_H

#include <dolfin/NodeIterator.h>
#include <dolfin/List.h>

namespace dolfin {

  class Node;
  class Edge;
  class Face;
  class Mesh;

  class BoundaryData {
  public:
    
    // Create an empty set of boundary data
    BoundaryData(Mesh& mesh);

    /// Destructor
    ~BoundaryData();

    /// Clear all data
    void clear();

    /// Add node to the boundary
    void add(Node& node);

    /// Add edge to the boundary
    void add(Edge& edge);

    /// Add face to the boundary
    void add(Face& face);

    /// Check if the boundary is empty
    bool empty();

    /// Return number of nodes on the boundary
    int noNodes() const;

    /// Return number of edges on the boundary
    int noEdges() const;

    /// Return number of faces on the boundary
    int noFaces() const;

    /// Friends
    friend class Mesh;
    friend class BoundaryInit;
    friend class NodeIterator::BoundaryNodeIterator;
    friend class EdgeIterator::BoundaryEdgeIterator;
    friend class FaceIterator::BoundaryFaceIterator;

  private:

    // Change the mesh pointer
    void setMesh(Mesh& mesh);

    // The mesh
    Mesh* mesh;
    
    // A list of all nodes on the boundary
    List<Node*> nodes;

    // A list of all edges on the boundary
    List<Edge*> edges;

    // A list of all faces on the boundary
    List<Face*> faces;

  };

}

#endif
