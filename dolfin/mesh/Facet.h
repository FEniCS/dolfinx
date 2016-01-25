// Copyright (C) 2006-2015 Anders Logg
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
// Modified by Garth N. Wells, 2009-2011.
// Modified by Martin Alnaes, 2015

#ifndef __FACET_H
#define __FACET_H

#include <memory>

#include <utility>
#include <vector>
#include "Cell.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshEntityIteratorBase.h"
#include "MeshFunction.h"

namespace dolfin
{

  /// A Facet is a MeshEntity of topological codimension 1.

  class Facet : public MeshEntity
  {
  public:

    /// Constructor
    Facet(const Mesh& mesh, std::size_t index)
      : MeshEntity(mesh, mesh.topology().dim() - 1, index) {}

    /// Destructor
    ~Facet() {}

    /// Compute component i of the normal to the facet
    double normal(std::size_t i) const;

    /// Compute normal to the facet
    Point normal() const;

    /// Compute squared distance to given point.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point.
    /// *Returns*
    ///     double
    ///         The squared distance to the point.
    double squared_distance(const Point& point) const;

    /// Compute distance to given point.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point.
    /// *Returns*
    ///     double
    ///         The distance to the point.
    double distance(const Point& point) const
    {
      return sqrt(squared_distance(point));
    }

    /// Return true if facet is an exterior facet (relative to global mesh,
    /// so this function will return false for facets on partition
    /// boundaries). Facet connectivity must be initialized before
    /// calling this function.
    bool exterior() const;

  };

  /// A FacetIterator is a MeshEntityIterator of topological
  /// codimension 1.
  typedef MeshEntityIteratorBase<Facet> FacetIterator;

  /// A FacetFunction is a MeshFunction of topological codimension 1.
  template <typename T> class FacetFunction : public MeshFunction<T>
  {
  public:

    FacetFunction(std::shared_ptr<const Mesh> mesh)
      : MeshFunction<T>(mesh, mesh->topology().dim() - 1) {}

    FacetFunction(std::shared_ptr<const Mesh> mesh, const T& value)
      : MeshFunction<T>(mesh, mesh->topology().dim() - 1, value) {}

  };

}

#endif
