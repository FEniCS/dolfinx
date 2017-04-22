// Copyright (C) 2011 Garth N. Wells
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
// First added:  2011-06-01
// Last changed:

#ifndef __LEGENDRE_H
#define __LEGENDRE_H

#include <cstddef>

namespace dolfin
{

  /// Interface for computing Legendre polynomials via Boost.

  class Legendre
  {
  public:

    /// Evaluate polynomial of order n at point x
    /// @param n (std::size_t)
    ///   Order
    /// @param x (double)
    ///   Point
    /// @return double
    ///   Legendre polynomial value at x
    static double eval(std::size_t n, double x);

    /// Evaluate first derivative of polynomial of order n at point x
    /// @param n (std::size_t)
    ///   Order
    /// @param x (double)
    ///   Point
    /// @return double
    ///   Legendre polynomial derivative value at x
    static double ddx(std::size_t n, double x);

    /// Evaluate second derivative of polynomial of order n at point x
    /// @param n (std::size_t)
    ///   Order
    /// @param x (double)
    ///   Point
    /// @return double
    ///   Legendre polynomial 2nd derivative value at x
    static double d2dx(std::size_t n, double x);

  };

}

#endif
