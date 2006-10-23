// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-10-23

#ifndef __EDGE_H
#define __EDGE_H

#include <dolfin/Point.h>
#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

namespace dolfin
{

  /// An Edge is a MeshEntity of topological dimension 1.

  class Edge : public MeshEntity
  {
  public:

    /// Create edge on given mesh
    Edge(Mesh& mesh, uint index) : MeshEntity(mesh, 1, index) {}

    /// Create edge from mesh entity
    Edge(MeshEntity& entity) : MeshEntity(entity.mesh(), 1, entity.index()) {}

    /// Destructor
    ~Edge() {}

    /// Compute coordinates of edge midpoint as a 3D point value
    Point midpoint();

  };

  /// An EdgeIterator is a MeshEntityIterator of topological dimension 1.
  
  class EdgeIterator : public MeshEntityIterator
  {
  public:
    
    EdgeIterator(Mesh& mesh) : MeshEntityIterator(mesh, 1) {}
    EdgeIterator(MeshEntity& entity) : MeshEntityIterator(entity, 1) {}
    EdgeIterator(MeshEntityIterator& it) : MeshEntityIterator(it, 1) {}

    inline Edge& operator*()
    { return static_cast<Edge&>(*static_cast<MeshEntityIterator>(*this)); }

    inline Edge* operator->()
    { return &static_cast<Edge&>(*static_cast<MeshEntityIterator>(*this)); }

  };    

}

#endif
