// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-02
// Last changed: 2007-10-23

#ifndef __FACE_H
#define __FACE_H

#include "MeshEntity.h"
#include "MeshEntityIterator.h"

namespace dolfin
{

  /// A Face is a MeshEntity of topological dimension 2.

  class Face : public MeshEntity
  {
  public:

    /// Constructor
    Face(Mesh& mesh, uint index) : MeshEntity(mesh, 2, index) {}

    /// Destructor
    ~Face() {}

  };

  /// A FaceIterator is a MeshEntityIterator of topological dimension 2.
  
  class FaceIterator : public MeshEntityIterator
  {
  public:
    
    FaceIterator(Mesh& mesh) : MeshEntityIterator(mesh, 2) {}
    FaceIterator(MeshEntity& entity) : MeshEntityIterator(entity, 2) {}

    inline Face& operator*() { return *operator->(); }
    inline Face* operator->() { return static_cast<Face*>(MeshEntityIterator::operator->()); }

  };    

}

#endif
