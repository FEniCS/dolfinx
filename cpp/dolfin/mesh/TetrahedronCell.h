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
class Cell;

/// This class implements functionality for tetrahedral cell meshes.

class TetrahedronCell : public CellTypeOld
{
public:
  /// Specify cell type and facet type
  TetrahedronCell() : mesh::CellTypeOld(CellType::tetrahedron) {}

  /// Compute squared distance to given point
  double squared_distance(const mesh::Cell& cell,
                          const Eigen::Vector3d& point) const;

private:
  // Find local index of edge i according to ordering convention
  std::size_t find_edge(std::size_t i, const mesh::Cell& cell) const;

  // Check whether point is outside region defined by facet ABC.
  // The fourth vertex is needed to define the orientation.
  bool point_outside_of_plane(const Eigen::Vector3d& point,
                              const Eigen::Vector3d& A,
                              const Eigen::Vector3d& B,
                              const Eigen::Vector3d& C,
                              const Eigen::Vector3d& D) const;
};
} // namespace mesh
} // namespace dolfin
