// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Cell.h"
#include <boost/multi_array.hpp>
#include <dolfin/common/constants.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dolfin
{

class Mesh;

/// The class provides functions to quantify mesh quality

class MeshQuality
{
public:
  /// Compute the radius ratio for all cells.
  /// @param mesh (std::shared_ptr<const Mesh>)
  /// @return     MeshFunction<double>
  ///         The cell radius ratio radius ratio geometric_dimension *
  ///         * inradius / circumradius (geometric_dimension
  ///         is normalization factor). It has range zero to one.
  ///         Zero indicates a degenerate element.
  ///
  static MeshFunction<double> radius_ratios(std::shared_ptr<const Mesh> mesh);

  /// Compute the minimum and maximum radius ratio of cells
  /// (across all processes)
  /// @param mesh (const Mesh&)
  /// @return    std::pair<double, double>
  ///         The [minimum, maximum] cell radii ratio (geometric_dimension *
  ///         * inradius / circumradius, geometric_dimension
  ///         is normalization factor). It has range zero to one.
  ///         Zero indicates a degenerate element.
  ///
  static std::pair<double, double> radius_ratio_min_max(const Mesh& mesh);

  /// Create (ratio, number of cells) data for creating a histogram
  /// of cell quality
  /// @param mesh (const Mesh&)
  /// @param num_bins (std::size_t)
  /// @return std::pair<std::vector<double>, std::vector<double>>
  static std::pair<std::vector<double>, std::vector<double>>
  radius_ratio_histogram_data(const Mesh& mesh, std::size_t num_bins = 50);

  /// Create Matplotlib string to plot cell quality histogram
  /// @param mesh (const Mesh&)
  /// @param num_intervals (std::size_t)
  /// @return std::string
  static std::string radius_ratio_matplotlib_histogram(const Mesh& mesh,
                                                       std::size_t num_intervals
                                                       = 50);

  /// Get internal dihedral angles of a tetrahedral cell
  static void dihedral_angles(const Cell& cell, std::vector<double>& dh_angle);

  /// Get internal minimum and maximum dihedral angles of a 3D mesh
  static std::pair<double, double> dihedral_angles_min_max(const Mesh& mesh);

  /// Create (dihedral angles, number of cells) data for creating a histogram
  /// of dihedral
  static std::pair<std::vector<double>, std::vector<double>>
  dihedral_angles_histogram_data(const Mesh& mesh, std::size_t num_bins = 100);

  /// Create Matplotlib string to plot dihedral angles quality histogram
  static std::string
  dihedral_angles_matplotlib_histogram(const Mesh& mesh,
                                       std::size_t num_intervals = 100);
};
}
