// Copyright (C) 2003-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2006-03-10

#ifndef __BOUNDARY_H
#define __BOUNDARY_H

#include <dolfin/BoundaryData.h>

namespace dolfin
{

  class Mesh;

  class Boundary
  {
  public:
    
    /// Create an empty boundary
    Boundary();

    /// Create a boundary for given mesh
    Boundary(Mesh& mesh);

    /// Destructor
    ~Boundary();

    /// Return number of vertices on the boundary
    int numVertices() const;

    /// Return number of edges on the boundary
    int numEdges() const;

    /// Return number of faces on the boundary
    int numFaces() const;

    /// Return number of facets on the boundary
    int numFacets() const;

    /// Friends
    friend class VertexIterator::BoundaryVertexIterator;
    friend class EdgeIterator::BoundaryEdgeIterator;
    friend class FaceIterator::BoundaryFaceIterator;

  private:

    // Compute boundary (and clear old data)
    void init();

    // Clear boundary
    void clear();
    
    // The mesh
    Mesh* mesh;

  };

}

#endif
