// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GAUSSIAN_QUADRATURE_H
#define __GAUSSIAN_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin {
  
  /// Gaussian-type quadrature rule on the real line,
  /// including Gauss, Radau, and Lobatto quadrature

  class GaussianQuadrature : public Quadrature {
  public:
    
    GaussianQuadrature(int n) : Quadrature(n) {};
    
  protected:
    
    void computeWeights();
    bool check(int q);
    
  };
  
}

#endif
