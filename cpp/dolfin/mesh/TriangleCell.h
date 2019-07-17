// Copyright (C) 2006-2017 Anders Logg
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

/// This class implements functionality for triangular meshes.

class TriangleCell : public CellTypeOld
{
public:
  /// Specify cell type and facet type
  TriangleCell() : CellTypeOld(CellType::triangle) {}

  /// Compute squared distance to given point (3D enabled)
  double squared_distance(const mesh::Cell& cell,
                          const Eigen::Vector3d& point) const;

};
} // namespace mesh
} // namespace dolfin
