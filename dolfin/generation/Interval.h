// Copyright (C) 2012 Benjamin Kehlet
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
// First added:  2012-11-10
// Last changed: 2012-11-10

#ifndef __INTERVAL_H
#define __INTERVAL_H

#include "dolfin/generation/IntervalMesh.h"

namespace dolfin
{

  /// Interval mesh of the 1D line [a,b].  Given the number of cells
  /// (nx) in the axial direction, the total number of intervals will
  /// be nx and the total number of vertices will be (nx + 1).
  ///
  /// This class is deprecated. Use _IntervalMesh_.
  class Interval : public IntervalMesh
  {
  public:

    /// Constructor
    ///
    /// *Arguments*
    ///     nx (std::size_t)
    ///         The number of cells.
    ///     a (double)
    ///         The minimum point (inclusive).
    ///     b (double)
    ///         The maximum point (inclusive).
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         // Create a mesh of 25 cells in the interval [-1,1]
    ///         Interval mesh(25, -1.0, 1.0);
    ///
    Interval(std::size_t nx, double a, double b)
      : IntervalMesh(nx, a, b)
    {
	warning("Interval is deprecated. Use IntervalMesh.");
    }

  };

}

#endif
