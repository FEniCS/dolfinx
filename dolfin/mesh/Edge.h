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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Johan Hoffman 2006.
// Modified by Kristoffer Selim 2009.
//
// First added:  2006-06-02
// Last changed: 2011-02-08

#ifndef __EDGE_H
#define __EDGE_H

#include <dolfin/common/types.h>
#include "Point.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"
#include "MeshFunction.h"

namespace dolfin
{

  /// An Edge is a MeshEntity of topological dimension 1.

  class Edge : public MeshEntity
  {
  public:

    /// Create edge on given mesh
    Edge(const Mesh& mesh, uint index) : MeshEntity(mesh, 1, index) {}

    /// Create edge from mesh entity
    Edge(MeshEntity& entity) : MeshEntity(entity.mesh(), 1, entity.index()) {}

    /// Destructor
    ~Edge() {}

    /// Compute Euclidean length of edge
    double length() const;

    /// Compute dot product between edge and other edge
    double dot(const Edge& edge) const;

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

  /// An EdgeFunction is a MeshFunction of topological dimension 1.

  template <class T> class EdgeFunction : public MeshFunction<T>
  {
  public:

    EdgeFunction(const Mesh& mesh) : MeshFunction<T>(mesh, 1) {}

    EdgeFunction(const Mesh& mesh, const T& value)
      : MeshFunction<T>(mesh, 1, value) {}

  };

}

#endif
