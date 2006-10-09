// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-01
// Last changed: 2006-06-22

#ifndef __NEW_VERTEX_H
#define __NEW_VERTEX_H

#include <dolfin/Point.h>
#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

namespace dolfin
{

  /// A Vertex is a MeshEntity of topological dimension 0.

  class Vertex : public MeshEntity
  {
  public:

    /// Create vertex on given mesh
    Vertex(Mesh& mesh, uint index) : MeshEntity(mesh, 0, index) {}
    
    /// Create vertex from mesh entity
    Vertex(MeshEntity& entity) : MeshEntity(entity.mesh(), 0, entity.index()) {}

    /// Destructor
    ~Vertex() {}

    /// Return x-coordinate of vertex
    inline real x() const { return _mesh.geometry().x(_index, 0); }

    /// Return y-coordinate of vertex
    inline real y() const { return (_dim >= 1 ? _mesh.geometry().x(_index, 1) : 0.0); }

    /// Return z-coordinate of vertex
    inline real z() const { return (_dim >= 2 ? _mesh.geometry().x(_index, 2) : 0.0); }

    /// Return value of coordinate in given direction
    inline real x(uint i) const { return (_dim >= i ? _mesh.geometry().x(_index, i) : 0.0); }

    /// Return coordinates of the vertex
    inline Point point() const { return Point(x(), y(), z()); }

  };

  /// A VertexIterator is a MeshEntityIterator of topological dimension 0.
  
  class VertexIterator : public MeshEntityIterator
  {
  public:
    
    VertexIterator(Mesh& mesh) : MeshEntityIterator(mesh, 0) {}
    VertexIterator(MeshEntity& entity) : MeshEntityIterator(entity, 0) {}
    VertexIterator(MeshEntityIterator& it) : MeshEntityIterator(it, 0) {}

    inline Vertex& operator*()
    { return static_cast<Vertex&>(*static_cast<MeshEntityIterator>(*this)); }

    inline Vertex* operator->()
    { return &static_cast<Vertex&>(*static_cast<MeshEntityIterator>(*this)); }

  };

}

#endif
