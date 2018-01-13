// Copyright (C) 2016-2017 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Point.h"
#include <string>
#include <vector>

namespace dolfin
{

/// This class provides useful functionality for debugging algorithms
/// dealing with geometry such as collision detection and intersection
/// triangulation.

class GeometryDebugging
{
public:
  /// Print coordinates of a point.
  /// Example usage: print(p0)
  static void print(const Point& point);

  /// Print coordinates of a simplex.
  /// Example usage: print({p0, p1, p2})
  static void print(const std::vector<Point>& simplex);

  /// Print coordinates of a pair of simplices.
  /// Example usage: print({p0, p1, p2}, {q0, q1})
  static void print(const std::vector<Point>& simplex_0,
                    const std::vector<Point>& simplex_1);

  /// Plot a point (print matplotlib code).
  /// Example usage: plot(p0)
  static void plot(const Point& point);

  /// Plot a simplex (print matplotlib code).
  /// Example usage: plot({p0, p1, p2})
  static void plot(const std::vector<Point>& simplex);

  /// Plot a pair of simplices (print matplotlib code).
  /// Example usage: plot({p0, p1, p2}, {q0, q1})
  static void plot(const std::vector<Point>& simplex_0,
                   const std::vector<Point>& simplex_1);

  /// Initialize plotting (print matplotlib code).
  static void init_plot();

  /// Compact point to string conversion
  static std::string point2string(const Point& p);

  /// Compact simplex to string conversion
  static std::string simplex2string(const std::vector<Point>& simplex);

private:
  // Check whether plotting has been initialized
  static bool _initialized;
};
}


