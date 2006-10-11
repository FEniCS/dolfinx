// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-01
// Last changed: 2006-10-11

#ifndef __NEW_VERTEX_H
#define __NEW_VERTEX_H

#include <dolfin/NewPoint.h>
#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

namespace dolfin
{

  /// A Vertex is a MeshEntity of topological dimension 0.

  class NewVertex : public MeshEntity
  {
  public:

    /// Create vertex on given mesh
    NewVertex(NewMesh& mesh, uint index) : MeshEntity(mesh, 0, index) {}
    
    /// Create vertex from mesh entity
    NewVertex(MeshEntity& entity) : MeshEntity(entity.mesh(), 0, entity.index()) {}

    /// Destructor
    ~NewVertex() {}

    /// Return value of vertex coordinate in direction i
    inline real x(uint i) const { return _mesh.geometry().x(_index, i); }

    /// Return vertex coordinates as a 3D point value
    inline NewPoint point() const { return _mesh.geometry().point(_index); }
    
  };

  /// A VertexIterator is a MeshEntityIterator of topological dimension 0.
  
  class NewVertexIterator : public MeshEntityIterator
  {
  public:
    
    NewVertexIterator(NewMesh& mesh) : MeshEntityIterator(mesh, 0) {}
    NewVertexIterator(MeshEntity& entity) : MeshEntityIterator(entity, 0) {}
    NewVertexIterator(MeshEntityIterator& it) : MeshEntityIterator(it, 0) {}

    inline NewVertex& operator*()
    { return static_cast<NewVertex&>(*static_cast<MeshEntityIterator>(*this)); }

    inline NewVertex* operator->()
    { return &static_cast<NewVertex&>(*static_cast<MeshEntityIterator>(*this)); }

  };

}

#endif
