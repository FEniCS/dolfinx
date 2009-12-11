// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-06-03
// Last changed: 2009-02-17

#ifndef __LEGENDRE_H
#define __LEGENDRE_H

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
    ~Legendre();

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
    

    uint n;

    real cache_x;
    real* cache;

  };

}

#endif
