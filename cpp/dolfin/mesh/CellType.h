// Copyright (C) 2006-2017 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.h"
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Cell;
class MeshEntity;

/// This class provides a common interface for different cell types.
/// Each cell type implements mesh functionality that is specific to
/// a certain type of cell.

class CellTypeOld
{
public:
  /// Constructor
  CellTypeOld(CellType cell_type);

  /// Destructor
  virtual ~CellTypeOld() = default;

  /// Create cell type from type (factory function)
  static CellTypeOld* create(CellType type);

  /// Compute squared distance to given point
  virtual double squared_distance(const Cell& cell,
                                  const Eigen::Vector3d& point) const = 0;

  /// Compute of given facet with respect to the cell
  virtual Eigen::Vector3d normal(const Cell& cell, std::size_t facet) const = 0;

  /// Compute normal to given cell (viewed as embedded in 3D)
  virtual Eigen::Vector3d cell_normal(const Cell& cell) const = 0;

  const CellType type;
};
} // namespace mesh
} // namespace dolfin
