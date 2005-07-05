// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-06-03
// Last changed: 2005

#ifndef __LEGENDRE_H
#define __LEGENDRE_H

#include <dolfin/constants.h>

namespace dolfin {

  /// Legendre polynomial of given degree n on the interval [-1,1].
  ///
  ///   P0(x) = 1
  ///   P1(x) = x
  ///   P2(x) = (3x^2 - 1) / 2
  ///   ...
  ///
  /// The function values and derivatives are computed using
  /// three-term recurrence formulas.

  class Legendre {
  public:

    Legendre(int n);

    /// Evaluation at given point
    real operator() (real x);
    
    /// Evaluation of derivative at given point
    real ddx(real x);

    /// Evaluation of second derivative at given point
    real d2dx(real x);

  private:
    
    real eval (int n, real x);
    real ddx  (int n, real x);
    real d2dx (int n, real x);

    int n;

  };

}

#endif
