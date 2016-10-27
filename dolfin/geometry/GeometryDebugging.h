// Copyright (C) 2016 Anders Logg
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
// First added:  2016-05-05
// Last changed: 2016-05-05

#ifndef __GEOMETRY_DEBUGGING_H
#define __GEOMETRY_DEBUGGING_H

#include <vector>
#include <string>
#include "Point.h"

namespace dolfin
{

  /// This class provides useful functionality for debugging algorithms
  /// dealing with geometry such as collision detection and intersection
  /// triangulation.

  class GeometryDebugging
  {
  public:

    /// Print coordinates of a simplex.
    /// Example usage: print({p0, p1, p2}).
    static void print(std::vector<Point> simplex);

    /// Print coordinates of a pair of simplices.
    /// Example usage: print({p0, p1, p2}, {q0, q1}).
    static void print(std::vector<Point> simplex_0,
                      std::vector<Point> simplex_1);

    /// Plot a simplex (print matplotlib code).
    /// Example usage: plot({p0, p1, p2}).
    static void plot(std::vector<Point> simplex);

    /// Plot a pair of simplices (print matplotlib code).
    /// Example usage: plot({p0, p1, p2}, {q0, q1}).
    static void plot(std::vector<Point> simplex_0,
                     std::vector<Point> simplex_1);

    /// Initialize plotting (print matplotlib code).
    static void init_plot();

    /// Compact point to string conversion
    static std::string point2string(Point p);

    /// Compact simplex to string conversion
    static std::string simplex2string(std::vector<Point> simplex);

  };

}

#endif
