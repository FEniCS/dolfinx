// Copyright (C) 2003-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2006-03-10

#ifndef __BOUNDARY_DATA_H
#define __BOUNDARY_DATA_H

#include <dolfin/VertexIterator.h>
#include <dolfin/PList.h>

namespace dolfin
{

  class Vertex;
  class Edge;
  class Face;
  class Mesh;

  class BoundaryData
  {
  public:
    
    // Create an empty set of boundary data
    BoundaryData(Mesh& mesh);

    /// Destructor
    ~BoundaryData();

    /// Clear all data
    void clear();

    /// Add vertex to the boundary
    void add(Vertex& vertex);

    /// Add edge to the boundary
    void add(Edge& edge);

    /// Add face to the boundary
    void add(Face& face);

    /// Check if the boundary is empty
    bool empty();

    /// Return number of vertices on the boundary
    int numVertices() const;

    /// Return number of edges on the boundary
    int numEdges() const;

    /// Return number of faces on the boundary
    int numFaces() const;

    /// Friends
    friend class Mesh;
    friend class BoundaryInit;
    friend class VertexIterator::BoundaryVertexIterator;
    friend class EdgeIterator::BoundaryEdgeIterator;
    friend class FaceIterator::BoundaryFaceIterator;

  private:

    // Change the mesh pointer
    void setMesh(Mesh& mesh);

    // The mesh
    Mesh* mesh;
    
    // A list of all vertices on the boundary
    PList<Vertex*> vertices;

    // A list of all edges on the boundary
    PList<Edge*> edges;

    // A list of all faces on the boundary
    PList<Face*> faces;

  };

}

#endif
