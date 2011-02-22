// Copyright (C) 2006-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-02
// Last changed: 2010-09-15

#ifndef __FACE_H
#define __FACE_H

#include "dolfin/common/types.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"
#include "MeshFunction.h"

namespace dolfin
{

  class Point;

  /// A Face is a MeshEntity of topological dimension 2.

  class Face : public MeshEntity
  {
  public:

    /// Constructor
    Face(const Mesh& mesh, uint index) : MeshEntity(mesh, 2, index) {}

    /// Destructor
    ~Face() {}

    /// Calculate the area of the face (triangle)
    double area() const;

    /// Compute component i of normal of given face with respect to the cell
    double normal(uint i) const;

    /// Compute normal of given face with respect to the cell
    Point normal() const;

  };

  /// A FaceIterator is a MeshEntityIterator of topological dimension 2.

  class FaceIterator : public MeshEntityIterator
  {
  public:

    FaceIterator(const Mesh& mesh) : MeshEntityIterator(mesh, 2) {}
    FaceIterator(const MeshEntity& entity) : MeshEntityIterator(entity, 2) {}

    inline Face& operator*() { return *operator->(); }
    inline Face* operator->() { return static_cast<Face*>(MeshEntityIterator::operator->()); }

  };

  /// A FaceFunction is a MeshFunction of topological dimension 2.

  template <class T> class FaceFunction : public MeshFunction<T>
  {
  public:

    FaceFunction(const Mesh& mesh) : MeshFunction<T>(mesh, 2) {}

    FaceFunction(const Mesh& mesh, const T& value)
      : MeshFunction<T>(mesh, 2, value) {}

  };

}

#endif
