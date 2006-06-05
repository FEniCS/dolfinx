// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-06-02

#ifndef __NEW_EDGE_H
#define __NEW_EDGE_H

#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

namespace dolfin
{

  /// An Edge is a MeshEntity of topological dimension 1.

  class NewEdge : public MeshEntity
  {
  public:

    /// Constructor
    NewEdge(NewMesh& mesh, uint index) : MeshEntity(mesh, 1, index) {}

    /// Destructor
    ~NewEdge() {}

  };

  /// An EdgeIterator is a MeshEntityIterator of topological dimension 1.
  
  class NewEdgeIterator : public MeshEntityIterator
  {
  public:
    
    NewEdgeIterator(NewMesh& mesh) : MeshEntityIterator(mesh, 1) {}
    NewEdgeIterator(MeshEntity& entity) : MeshEntityIterator(entity, 1) {}
    NewEdgeIterator(MeshEntityIterator& it) : MeshEntityIterator(it, 1) {}

  };    

}

#endif
