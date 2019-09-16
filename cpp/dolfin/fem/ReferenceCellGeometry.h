// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/mesh/cell_types.h>

namespace dolfin
{

namespace fem
{

  /// Tabulates the vertex positions for the reference cell
class ReferenceCellGeometry
{
public:
  /// Get geometric points for all vertices
  static Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  get_vertices(mesh::CellType cell_type);
};
} // namespace fem
} // namespace dolfin
