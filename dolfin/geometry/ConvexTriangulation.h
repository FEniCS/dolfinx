// Copyright (C) 2016 Anders Logg, August Johansson and Benjamin Kehlet
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
// First added:  2016-06-01
// Last changed: 2017-09-30

#ifndef __CONVEX_TRIANGULATION
#define __CONVEX_TRIANGULATION

#include <vector>
#include "Point.h"

namespace dolfin
{

  /// This class implements algorithms for triangulating convex
  /// domains represented as a set of points.

  class ConvexTriangulation
  {
  public:

    /// Tdim independent wrapper
    static std::vector<std::vector<Point>>
    triangulate(const std::vector<Point>& p,
                std::size_t gdim,
                std::size_t tdim);

    /// Triangulate 1D
    static std::vector<std::vector<Point>>
    triangulate_1d(const std::vector<Point>& pm,
		   std::size_t gdim)
    {
      return _triangulate_1d(pm, gdim);
    }

    /// Triangulate using the Graham scan 2D
    static std::vector<std::vector<Point>>
    triangulate_graham_scan_2d(const std::vector<Point>& pm)
    {
      return _triangulate_graham_scan_2d(pm);
    }

    /// Triangulate using the Graham scan 3D
    static std::vector<std::vector<Point>>
    triangulate_graham_scan_3d(const std::vector<Point>& pm);

    /// Determine if there are self-intersecting tetrahedra
    static bool selfintersects(const std::vector<std::vector<Point>>& p);

  private:

    // Implementation declarations

    /// Implementation of 1D triangulation
    static std::vector<std::vector<Point>>
    _triangulate_1d(const std::vector<Point>& pm,
		    std::size_t gdim);

    /// Implementation of Graham scan 2D
    static std::vector<std::vector<Point>>
    _triangulate_graham_scan_2d(const std::vector<Point>& pm);

    /// Implementation of Graham scan 3D
    static std::vector<std::vector<Point>>
    _triangulate_graham_scan_3d(const std::vector<Point>& pm);
  };

} // end namespace dolfin
#endif
