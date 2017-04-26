// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2013-10-07
// Last changed:

#ifndef __MESH_QUALITY_H
#define __MESH_QUALITY_H

#include <string>
#include <utility>
#include <vector>
#include <boost/multi_array.hpp>
#include <memory>
#include "Cell.h"

namespace dolfin
{

    class Mesh;

  /// The class provides functions to quantify mesh quality

  class MeshQuality
  {
  public:

    /// Compute the radius ratio for all cells.
    /// @param mesh (std::shared_ptr<const Mesh>)
    /// @return     CellFunction<double>
    ///         The cell radius ratio radius ratio geometric_dimension *
    ///         * inradius / circumradius (geometric_dimension
    ///         is normalization factor). It has range zero to one.
    ///         Zero indicates a degenerate element.
    ///
    static CellFunction<double>
      radius_ratios(std::shared_ptr<const Mesh> mesh);


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
    static std::pair<std::vector<double>, std::vector<double> >
      radius_ratio_histogram_data(const Mesh& mesh,
                                  std::size_t num_bins = 50);

    /// Create Matplotlib string to plot cell quality histogram
    /// @param mesh (const Mesh&)
    /// @param num_intervals (std::size_t)
    /// @return std::string
    static std::string
      radius_ratio_matplotlib_histogram(const Mesh& mesh,
					std::size_t num_intervals = 50);

    /// Get internal dihedral angles of a tetrahedral cell
    static void dihedral_angles(const Cell& cell, std::vector<double>& dh_angle);

    /// Get internal minimum and maximum dihedral angles of a 3D mesh
    static std::pair<double, double>
        dihedral_angles_min_max(const Mesh& mesh);

    /// Create (dihedral angles, number of cells) data for creating a histogram
    /// of dihedral
    static std::pair<std::vector<double>, std::vector<double> >
      dihedral_angles_histogram_data(const Mesh& mesh,
                                  std::size_t num_bins = 100);

    /// Create Matplotlib string to plot dihedral angles quality histogram
    static std::string
      dihedral_angles_matplotlib_histogram(const Mesh& mesh,
                                std::size_t num_intervals = 100);
  };

}

#endif
