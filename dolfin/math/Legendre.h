// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-06-03
// Last changed: 2005

#ifndef __LEGENDRE_H
#define __LEGENDRE_H

#include <dolfin/common/types.h>

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
    double operator() (double x);
    
    /// Evaluation of derivative at given point
    double ddx(double x);

    /// Evaluation of second derivative at given point
    double d2dx(double x);

  private:
    
    double eval (int n, double x);
    double ddx  (int n, double x);
    double d2dx (int n, double x);

    int n;

  };

}

#endif
