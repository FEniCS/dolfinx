// Copyright (C) 2016 Anders Logg, August Johansson and Benjamin Kehlet
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Point.h"
#include <vector>

namespace dolfin
{

/// This class implements algorithms for triangulating convex
/// domains represented as a set of points.

class ConvexTriangulation
{
public:
  /// Tdim independent wrapper
  static std::vector<std::vector<Point>>
  triangulate(const std::vector<Point>& p, std::size_t gdim, std::size_t tdim);

  /// Triangulate 1D
  static std::vector<std::vector<Point>>
  triangulate_1d(const std::vector<Point>& pm, std::size_t gdim)
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
  _triangulate_1d(const std::vector<Point>& pm, std::size_t gdim);

  /// Implementation of Graham scan 2D
  static std::vector<std::vector<Point>>
  _triangulate_graham_scan_2d(const std::vector<Point>& pm);

  /// Implementation of Graham scan 3D
  static std::vector<std::vector<Point>>
  _triangulate_graham_scan_3d(const std::vector<Point>& pm);
};

} // end namespace dolfin

