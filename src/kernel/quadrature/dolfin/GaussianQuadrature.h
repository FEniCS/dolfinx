// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GAUSSIAN_QUADRATURE_H
#define __GAUSSIAN_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin {
  
  /// Gaussian-type quadrature rule on the real line,
  /// including Gauss, Radau, and Lobatto quadrature.
  ///
  /// Points and weights are computed to be exact within a tolerance
  /// of DOLFIN_EPS. Comparing with known exact values for n <= 3 shows
  /// that we obtain full precision (16 digits, error less than 2e-16).

  class GaussianQuadrature : public Quadrature {
  public:
    
    GaussianQuadrature(int n);
    
  protected:
    
    void computeWeights();
    bool check(int q);
    
  };
  
}

#endif
