// Copyright (C) 2018 Chris N Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/types.h>
#include <vector>

#include "MeshConnectivity.h"

namespace dolfin
{
namespace mesh
{

/// CoordinateDofs contains the connectivity from MeshEntities to the geometric
/// points which make up the mesh.

class CoordinateDofs
{
public:
  /// Constructor
  /// @param tdim
  ///   Topological Dimension
  /// @param cell_dofs
  ///   Connections from cells to points in MeshGeometry (cell_dofs)
  CoordinateDofs(std::uint32_t tdim);

  /// Copy constructor
  CoordinateDofs(const CoordinateDofs& topology) = default;

  /// Move constructor
  CoordinateDofs(CoordinateDofs&& topology) = default;

  /// Destructor
  ~CoordinateDofs() = default;

  /// Copy assignment
  CoordinateDofs& operator=(const CoordinateDofs& topology) = default;

  /// Move assignment
  CoordinateDofs& operator=(CoordinateDofs&& topology) = default;

  /// Initialise entity->point dofs for dimension dim
  /// @param dim
  ///   Dimension of entity
  /// @param point_dofs
  ///   Array containing point dofs for each entity
  void init(std::size_t dim, Eigen::Ref<const EigenRowArrayXXi32> point_dofs);

  /// Get the entity points associated with entities of dimension i
  ///
  /// @param dim
  ///   Entity dimension
  /// @return MeshConnectivity
  ///   Connections from entities of given dimension to points
  const MeshConnectivity& entity_points(std::uint32_t dim) const
  {
    assert(dim < _coord_dofs.size());
    return _coord_dofs[dim];
  }

private:
  // Connectivity from entities to points.
  // Initially only defined for cells
  std::vector<MeshConnectivity> _coord_dofs;
};
}
}
