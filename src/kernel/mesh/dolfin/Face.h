// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __FACE_H
#define __FACE_H

#include <dolfin/PArray.h>
#include <set>

namespace dolfin
{
  
  class Edge;

  /// A Face consists of a list of edges.
  ///
  /// A triangle has no faces.
  /// A face has three edges for a tetrahedron.

  class Face
  {
  public:

    /// Create an empty face
    Face();

    /// Destructor
    ~Face();

    /// Clear face data
    void clear();
   
    /// Return id of face
    int id() const;

    /// Return number of edges
    unsigned int noEdges() const;

    /// Return number of cell neighbors
    unsigned int noCellNeighbors() const;

    /// Return edge number i
    Edge& edge(int i) const;

    /// Return cell neighbor number i
    Cell& cell(int i) const;

    /// Return the mesh containing the face
    Mesh& mesh();
    
    /// Return the mesh containing the face (const version)
    const Mesh& mesh() const;

    /// Check if face consists of the given edges
    bool equals(const Edge& e0, const Edge& e1, const  Edge& e2) const;
    
    /// Check if face consists of the given edges
    bool equals(const Edge& e0, const Edge& e1) const;

    /// Check if edge contains the node
    bool contains(const Node& n) const;

    ///--- Output ---
   
    /// Display condensed face data
    friend LogStream& operator<<(LogStream& stream, const Face& face);

    /// Friends
    friend class MeshData;
    friend class MeshInit;
    friend class EdgeIterator::FaceEdgeIterator;
    friend class CellIterator::FaceCellIterator;
    friend class Tetrahedron;

    // Boundary information
    std::set<int> fbids;

  private:

    // Specify global face number
    int setID(int id, Mesh& mesh);

    // Set the mesh pointer
    void setMesh(Mesh& mesh);

    /// Specify three edges
    void set(Edge& e0, Edge& e1, Edge& e2);
    
    // The mesh containing this face
    Mesh* _mesh;

    // Global face number
    int _id;

    // The list of edges
    PArray<Edge*> fe;
    
    // Connectivity
    PArray<Cell*> fc;
    
  };

}

#endif
