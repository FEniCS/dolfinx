// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FACE_H
#define __FACE_H

#include <dolfin/Array.h>

namespace dolfin {
  
  class Edge;

  /// A Face consists of a list of edges.
  ///
  /// A triangle has no faces.
  /// A face has three edges for a tetrahedron.

  class Face {
  public:

    /// Create an empty face
    Face();

    /// Destructor
    ~Face();

    /// Clear face data
    void clear();

    /// Return id of face
    int id() const;

    /// Return the number of edges
    //int size() const;

    /// Return edge number i
    Edge& edge(int i) const;

    /// Return cell neighbor number i
    Cell& cell(int i) const;

    /// Check if face consists of the given edges
    bool equals(const Edge& e0, const Edge& e1, const  Edge& e2) const;
    
    /// Check if face consists of the given edges
    bool equals(const Edge& e0, const Edge& e1) const;

    ///--- Output ---
   
    /// Display condensed face data
    friend LogStream& operator<<(LogStream& stream, const Face& face);

    /// Friends
    friend class MeshData;
    friend class MeshInit;
    friend class EdgeIterator::FaceEdgeIterator;

  private:
    
    // Specify global face number
    int setID(int id, Mesh& mesh);

    // Set the mesh pointer
    void setMesh(Mesh& mesh);

    /// Specify three edges
    void set(Edge& e0, Edge& e1, Edge& e2);
    
    // The mesh containing this face
    Mesh* mesh;

    // Global face number
    int _id;

    // Connectivity
    Array<Cell*> fc;

    // The list of edges
    Array<Edge*> fe;

  };

}

#endif
