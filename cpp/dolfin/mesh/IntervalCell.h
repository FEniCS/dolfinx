// Copyright (C) 2006-2017 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Cell;
class MeshEntity;
template <typename T>
class MeshFunction;

/// This class implements functionality for interval cell meshes.

class IntervalCell : public mesh::CellTypeOld
{
public:
  /// Specify cell type and facet type
  IntervalCell() : mesh::CellTypeOld(CellType::interval) {}

  /// Compute squared distance to given point (3D enabled)
  double squared_distance(const mesh::Cell& cell,
                          const Eigen::Vector3d& point) const;

  /// Compute squared distance to given point. This version takes
  /// the two vertex coordinates as 3D points. This makes it
  /// possible to reuse this function for computing the (squared)
  /// distance to a triangle.
  static double squared_distance(const Eigen::Vector3d& point,
                                 const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b);
};
} // namespace mesh
} // namespace dolfin
