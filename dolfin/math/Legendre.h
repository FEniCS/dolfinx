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

#include <dolfin/common/types.h>

namespace dolfin
{

  /// Interface for computing Legendre polynomials via Boost.

  class Legendre
  {
  public:

    /// Evaluate polynomial of order n at point x
    static double eval(uint n, double x);

    /// Evaluate first derivative of polynomial of order n at point x
    static double ddx(uint n, double x);

    /// Evaluate second derivative of polynomial of order n at point x
    static double d2dx(uint n, double x);

  };

}

#endif
