// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-01
// Last changed: 2006-10-19

#ifndef __VERTEX_H
#define __VERTEX_H

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

    /// Return value of vertex coordinate i
    inline real x(uint i) const { return _mesh.geometry().x(_index, i); }

    /// Return vertex coordinates as a 3D point value
    inline Point point() const { return _mesh.geometry().point(_index); }

    /// Return array of vertex coordinates
    inline real* coordinates() { return _mesh.geometry().x(_index); }

    /// Return array of vertex coordinates
    inline const real* coordinates() const { return _mesh.geometry().x(_index); }
    
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
