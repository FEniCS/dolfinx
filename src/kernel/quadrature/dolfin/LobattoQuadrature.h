// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LOBATTO_QUADRATURE_H
#define __LOBATTO_QUADRATURE_H

#include <dolfin/GaussianQuadrature.h>

namespace dolfin {

  /// Lobatto (Gauss-Lobatto) quadrature on the interval [-1,1].
  /// The n quadrature points are given by the end-points -1 and 1,
  /// and the zeros of P{n-1}'(x), where P{n-1}(x) is the (n-1):th
  /// Legendre polynomial.
  ///
  /// The quadrature points are computed using Newton's method, and
  /// the quadrature weights are computed by solving a linear system
  /// determined by the condition that Lobatto quadrature with n points
  /// should be exact for polynomials of degree 2n-3.

  class LobattoQuadrature : public GaussianQuadrature {
  public:
    
    LobattoQuadrature(int n);

  private:

    void computePoints();
    
  };

  LogStream& operator<<(LogStream& stream, const LobattoQuadrature& gauss);
  
}

#endif
