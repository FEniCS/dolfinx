// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-06-12

#ifndef __NEW_EDGE_H
#define __NEW_EDGE_H

#include <dolfin/NewPoint.h>
#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

namespace dolfin
{

  /// An Edge is a MeshEntity of topological dimension 1.

  class NewEdge : public MeshEntity
  {
  public:

    /// Create edge on given mesh
    NewEdge(NewMesh& mesh, uint index) : MeshEntity(mesh, 1, index) {}

    /// Create edge from mesh entity
    NewEdge(MeshEntity& entity) : MeshEntity(entity.mesh(), 1, entity.index()) {}

    /// Destructor
    ~NewEdge() {}

    /// Return coordinates of edge midpoint
    NewPoint midpoint();

  };

  /// An EdgeIterator is a MeshEntityIterator of topological dimension 1.
  
  class NewEdgeIterator : public MeshEntityIterator
  {
  public:
    
    NewEdgeIterator(NewMesh& mesh) : MeshEntityIterator(mesh, 1) {}
    NewEdgeIterator(MeshEntity& entity) : MeshEntityIterator(entity, 1) {}
    NewEdgeIterator(MeshEntityIterator& it) : MeshEntityIterator(it, 1) {}

    inline NewEdge& operator*()
    { return static_cast<NewEdge&>(*static_cast<MeshEntityIterator>(*this)); }

    inline NewEdge* operator->()
    { return &static_cast<NewEdge&>(*static_cast<MeshEntityIterator>(*this)); }

  };    

}

#endif
