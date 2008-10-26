// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman 2006.
//
// First added:  2006-06-02
// Last changed: 2007-05-02

#ifndef __EDGE_H
#define __EDGE_H

#include <dolfin/common/types.h>
#include "Point.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"

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

    /// Compute Euclidian length of edge 
    double length();

    /// Compute coordinates of edge midpoint as a 3D point value
    Point midpoint();

  };

  /// An EdgeIterator is a MeshEntityIterator of topological dimension 1.
  
  class EdgeIterator : public MeshEntityIterator
  {
  public:
    
    EdgeIterator(const Mesh& mesh) : MeshEntityIterator(mesh, 1) {}
    EdgeIterator(const MeshEntity& entity) : MeshEntityIterator(entity, 1) {}

    inline Edge& operator*() { return *operator->(); }
    inline Edge* operator->() { return static_cast<Edge*>(MeshEntityIterator::operator->()); }

  };    

}

#endif
