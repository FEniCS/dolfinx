// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LOBATTO_QUADRATURE_H
#define __LOBATTO_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin {

  /// Lobatto (Gauss-Lobatto) quadrature on the interval [-1,1].
  /// The n quadrature points are given by the end-points -1 and 1,
  /// and the zeros of Pn'(x), where Pn(x) is the n:th Legendre polynomial.
  ///
  /// The quadrature points are computed using Newton's method, and
  /// the quadrature weights are computed by solving a linear system
  /// determined by the condition that Lobatto quadrature with n points
  /// should be exact for polynomials of degree 2n-3.

  class LobattoQuadrature : public Quadrature {
  public:
    
    LobattoQuadrature(int n);

  private:

    void computePoints();
    void computeWeights();
    
  };
  
}

#endif
