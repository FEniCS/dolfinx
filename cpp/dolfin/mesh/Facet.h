// Copyright (C) 2006-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "MeshEntity.h"
#include "Topology.h"
#include <dolfin/geometry/Point.h>
#include <utility>

namespace dolfin
{

namespace mesh
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
  geometry::Point normal() const;

  /// Compute squared distance to given point.
  ///
  /// @param     point (_geometry::Point_)
  ///         The point.
  /// @return     double
  ///         The squared distance to the point.
  double squared_distance(const geometry::Point& point) const;

  /// Return true if facet is an exterior facet (relative to global
  /// mesh, so this function will return false for facets on partition
  /// boundaries). Facet connectivity must be initialized before calling
  /// this function.
  bool exterior() const;
};
} // namespace mesh
} // namespace dolfin
