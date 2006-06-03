// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-01
// Last changed: 2006-06-01

#ifndef __NEW_VERTEX_H
#define __NEW_VERTEX_H

#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

namespace dolfin
{

  /// A Vertex is a MeshEntity of topological dimension 0.

  class NewVertex : public MeshEntity
  {
  public:

    /// Constructor
    NewVertex(NewMesh& mesh, uint index) : MeshEntity(mesh, 0, index) {}

    /// Destructor
    ~NewVertex() {}

  };

  /// A VertexIterator is a MeshEntityIterator of topological dimension 0.
  
  class NewVertexIterator : public MeshEntityIterator
  {
  public:
    
    NewVertexIterator(NewMesh& mesh) : MeshEntityIterator(mesh, 0) {}
    NewVertexIterator(MeshEntity& entity) : MeshEntityIterator(entity, 0) {}
    NewVertexIterator(MeshEntityIterator& it) : MeshEntityIterator(it, 0) {}

  };    

}

#endif
