// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-06-02

#ifndef __NEW_FACE_H
#define __NEW_FACE_H

#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

namespace dolfin
{

  /// A Face is a MeshEntity of topological dimension 2.

  class NewFace : public MeshEntity
  {
  public:

    /// Constructor
    NewFace(NewMesh& mesh, uint index) : MeshEntity(mesh, 2, index) {}

    /// Destructor
    ~NewFace() {}

  };

  /// A FaceIterator is a MeshEntityIterator of topological dimension 2.
  
  class NewFaceIterator : public MeshEntityIterator
  {
  public:
    
    NewFaceIterator(NewMesh& mesh) : MeshEntityIterator(mesh, 2) {}
    NewFaceIterator(MeshEntity& entity) : MeshEntityIterator(entity, 2) {}
    NewFaceIterator(MeshEntityIterator& it) : MeshEntityIterator(it, 2) {}

  };    

}

#endif
