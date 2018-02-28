// Copyright (C) 2016-2017 Anders Logg, August Johansson and Benjamin Kehlet
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Point.h"
#include <dolfin/log/LogStream.h>
#include <vector>

namespace dolfin
{

/// This class implements geometric predicates, i.e. function that
/// return either true or false.

class GeometryPredicates
{
public:
  /// Check whether simplex is degenerate
  static bool is_degenerate(const std::vector<Point>& simplex,
                            std::size_t gdim);

  /// Check whether simplex is degenerate (2D version)
  static bool is_degenerate_2d(const std::vector<Point>& simplex);

  /// Check whether simplex is degenerate (3D version)
  static bool is_degenerate_3d(const std::vector<Point>& simplex);

  /// Check whether simplex is finite (not Inf or NaN)
  static bool is_finite(const std::vector<Point>& simplex);

  /// Check whether simplex is finite (not Inf or NaN)
  static bool is_finite(const std::vector<double>& simplex);

  /// Check whether the convex hull is degenerate
  static bool convex_hull_is_degenerate(const std::vector<Point>& p,
                                        std::size_t gdim);

};
}
