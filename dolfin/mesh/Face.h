// Copyright (C) 2006-2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MeshEntity.h"
#include "MeshEntityIteratorBase.h"

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

  /// Compute normal to the face
  Point normal() const;
};

/// A FaceIterator is a MeshEntityIterator of topological dimension 2.
typedef MeshEntityIteratorBase<Face> FaceIterator;
}


