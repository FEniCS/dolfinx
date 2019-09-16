// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MeshFunction.h"
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dolfin
{
namespace mesh
{
class Mesh;
class MeshEntity;

/// The class provides functions to quantify mesh quality

class MeshQuality
{
public:
  /// Compute the minimum and maximum radius ratio of cells (across all
  /// processes)
  /// @param[in] mesh The mesh
  /// @return The [minimum, maximum] cell radii ratio
  ///         (geometric_dimension * * inradius / circumradius,
  ///         geometric_dimension is normalization factor). It has range
  ///         zero to one. Zero indicates a degenerate element.
  static std::array<double, 2> radius_ratio_min_max(const Mesh& mesh);

  /// Get internal dihedral angles of a tetrahedral cell
  static std::array<double, 6> dihedral_angles(const mesh::MeshEntity& cell);

  /// Get internal minimum and maximum dihedral angles of a 3D mesh
  static std::array<double, 2> dihedral_angles_min_max(const Mesh& mesh);

  /// Create (dihedral angles, number of cells) data for creating a
  /// histogram of dihedral
  static std::pair<std::vector<double>, std::vector<std::size_t>>
  dihedral_angle_histogram_data(const Mesh& mesh, std::size_t num_bins);
};
} // namespace mesh
} // namespace dolfin
