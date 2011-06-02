// Copyright (C) 2003-2009 Anders Logg
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
// First added:  2003-06-03
// Last changed: 2009-02-17

#ifndef __LEGENDRE_H
#define __LEGENDRE_H

#include <vector>
#include <dolfin/common/types.h>
#include <dolfin/common/real.h>

namespace dolfin
{

  /// Legendre polynomial of given degree n on the interval [-1,1].
  ///
  ///   P0(x) = 1
  ///   P1(x) = x
  ///   P2(x) = (3x^2 - 1) / 2
  ///   ...
  ///
  /// The function values and derivatives are computed using
  /// three-term recurrence formulas.

  class Legendre
  {
  public:

    Legendre(uint n);

    /// Evaluation at given point
    real operator() (real x);

    /// Evaluation of derivative at given point
    real ddx(real x);

    /// Evaluation of second derivative at given point
    real d2dx(real x);

    /// Evaluation of arbitrary order, nn <= n (useful ie in RadauQuadrature)
    real eval(uint nn, real x);

    real ddx(uint n, real x);
    real d2dx(uint n, real x);


  private:

    const uint n;
    real cache_x;
    std::vector<real> cache;

  };

}

#endif
