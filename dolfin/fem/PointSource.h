// Copyright (C) 2011 Anders Logg
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

#ifndef __POINT_SOURCE_H
#define __POINT_SOURCE_H

#include <memory>
#include <dolfin/geometry/Point.h>

namespace dolfin
{

  // Forward declarations
  class FunctionSpace;
  class GenericVector;

  /// This class provides an easy mechanism for adding a point source
  /// (Dirac delta function) to the right-hand side vector in a
  /// variational problem. The associated function space must be
  /// scalar in order for the inner product with the (scalar) Dirac
  /// delta function to be well defined.

  class PointSource
  {
  public:

    /// Create point source at given point of given magnitude
    PointSource(std::shared_ptr<const FunctionSpace> V, const Point& p,
                double magnitude=1.0);

    /// Destructor
    ~PointSource();

    /// Apply (add) point source to right-hand side vector
    void apply(GenericVector& b);

  private:

    // Check that function space is scalar
    void check_is_scalar(const FunctionSpace& V);

    // The function space
    std::shared_ptr<const FunctionSpace> _function_space;

    // The point
    Point _p;

    // Magnitude
    double _magnitude;

  };

}

#endif
