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
    FaceIterator(MeshEntityIterator& it) : MeshEntityIterator(it, 2) {}

    inline Face& operator*()
    { return static_cast<Face&>(*static_cast<MeshEntityIterator>(*this)); }

    inline Face* operator->()
    { return &static_cast<Face&>(*static_cast<MeshEntityIterator>(*this)); }

  };    

}

#endif
