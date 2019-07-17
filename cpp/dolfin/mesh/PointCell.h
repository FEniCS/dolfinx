// Copyright (C) 2007-2007 Kristian B. Oelgaard
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
/// This class implements functionality for point cell meshes.

class PointCell : public CellTypeOld
{
public:
  /// Specify cell type and facet type
  PointCell() : CellTypeOld(CellType::point) {}

  /// Compute squared distance to given point
  double squared_distance(const mesh::Cell& cell,
                          const Eigen::Vector3d& point) const;
};
} // namespace mesh
} // namespace dolfin
