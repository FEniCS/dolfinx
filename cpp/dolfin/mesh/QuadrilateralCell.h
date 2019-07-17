// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include <vector>

namespace dolfin
{
namespace mesh
{
class Cell;

/// This class implements functionality for quadrilateral cells.

class QuadrilateralCell : public CellTypeOld
{
public:
  /// Specify cell type and facet type
  QuadrilateralCell() : mesh::CellTypeOld(CellType::quadrilateral) {}

  /// Compute squared distance to given point (3D enabled)
  double squared_distance(const mesh::Cell& cell,
                          const Eigen::Vector3d& point) const;

  /// Compute of given facet with respect to the cell
  Eigen::Vector3d normal(const mesh::Cell& cell, std::size_t facet) const;

  /// Compute normal to given cell (viewed as embedded in 3D)
  Eigen::Vector3d cell_normal(const mesh::Cell& cell) const;
};
} // namespace mesh
} // namespace dolfin
