// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __RADAU_QUADRATURE_H
#define __RADAU_QUADRATURE_H

#include <dolfin/GaussianQuadrature.h>

namespace dolfin {

  /// Radau (Gauss-Radau) quadrature on the interval [-1,1].
  /// The n quadrature points are given by the zeros of
  ///
  ///     ( Pn-1(x) + Pn(x) ) / (1+x)
  ///
  /// where Pn is the n:th Legendre polynomial.
  ///
  /// The quadrature points are computed using Newton's method, and
  /// the quadrature weights are computed by solving a linear system
  /// determined by the condition that Radau quadrature with n points
  /// should be exact for polynomials of degree 2n-2.

  class RadauQuadrature : public GaussianQuadrature {
  public:
    
    RadauQuadrature(int n);
    
  private:

    void computePoints();

  };
  
}

#endif
