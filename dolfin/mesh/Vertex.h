// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-01
// Last changed: 2008-10-23

#ifndef __VERTEX_H
#define __VERTEX_H

#include "Point.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"

namespace dolfin
{

  /// A Vertex is a MeshEntity of topological dimension 0.

  class Vertex : public MeshEntity
  {
  public:

    /// Create vertex on given mesh
    Vertex(const Mesh& mesh, uint index) : MeshEntity(mesh, 0, index) {}

    /// Create vertex from mesh entity
    Vertex(MeshEntity& entity) : MeshEntity(entity.mesh(), 0, entity.index()) {}

    /// Destructor
    ~Vertex() {}

    /// Return value of vertex coordinate i
    inline double x(uint i) const { return _mesh.geometry().x(_index, i); }

    /// Return vertex coordinates as a 3D point value
    inline Point point() const { return _mesh.geometry().point(_index); }

    /// Return array of vertex coordinates
    inline const double* x() const { return _mesh.geometry().x(_index); }
    
  };

  /// A VertexIterator is a MeshEntityIterator of topological dimension 0.
  
  class VertexIterator : public MeshEntityIterator
  {
  public:
    
    VertexIterator(const Mesh& mesh) : MeshEntityIterator(mesh, 0) {}
    VertexIterator(const MeshEntity& entity) : MeshEntityIterator(entity, 0) {}

    inline const Vertex& operator*() { return *operator->(); }
    inline const Vertex* operator->() { return static_cast<Vertex*>(MeshEntityIterator::operator->()); }

  };

}

#endif
