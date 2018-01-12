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

#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshEntityIteratorBase.h"
#include "MeshTopology.h"
#include <utility>
#include <vector>

namespace dolfin
{

/// A Facet is a MeshEntity of topological codimension 1.

class Facet : public MeshEntity
{
public:
  /// Constructor
  Facet(const Mesh& mesh, std::size_t index)
      : MeshEntity(mesh, mesh.topology().dim() - 1, index)
  {
  }

  /// Destructor
  ~Facet() {}

  /// Compute normal to the facet
  Point normal() const;

  /// Compute squared distance to given point.
  ///
  /// @param     point (_Point_)
  ///         The point.
  /// @return     double
  ///         The squared distance to the point.
  double squared_distance(const Point& point) const;

  /// Return true if facet is an exterior facet (relative to global mesh,
  /// so this function will return false for facets on partition
  /// boundaries). Facet connectivity must be initialized before
  /// calling this function.
  bool exterior() const;
};

/// A FacetIterator is a MeshEntityIterator of topological
/// codimension 1.
typedef MeshEntityIteratorBase<Facet> FacetIterator;
}

#endif
