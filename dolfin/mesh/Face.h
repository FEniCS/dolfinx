// Copyright (C) 2006-2010 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2011
//
// First added:  2006-06-02
// Last changed: 2011-02-22

#ifndef __FACE_H
#define __FACE_H

#include <memory>

#include "MeshEntity.h"
#include "MeshEntityIteratorBase.h"
#include "MeshFunction.h"

namespace dolfin
{

  class Mesh;
  class Point;

  /// A Face is a MeshEntity of topological dimension 2.

  class Face : public MeshEntity
  {
  public:

    /// Constructor
    Face(const Mesh& mesh, std::size_t index) : MeshEntity(mesh, 2, index) {}

    /// Destructor
    ~Face() {}

    /// Calculate the area of the face (triangle)
    double area() const;

    /// Compute component i of the normal to the face
    double normal(std::size_t i) const;

    /// Compute normal to the face
    Point normal() const;

  };

  /// A FaceIterator is a MeshEntityIterator of topological dimension 2.
  typedef MeshEntityIteratorBase<Face> FaceIterator;

  /// A FaceFunction is a MeshFunction of topological dimension 2.
  template <typename T> class FaceFunction : public MeshFunction<T>
  {
  public:

    FaceFunction(std::shared_ptr<const Mesh> mesh)
      : MeshFunction<T>(mesh, 2) {}

    FaceFunction(std::shared_ptr<const Mesh> mesh, const T& value)
      : MeshFunction<T>(mesh, 2, value) {}

  };

}

#endif
