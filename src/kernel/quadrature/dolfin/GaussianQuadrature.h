// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GAUSSIAN_RULES_H
#define __GAUSSIAN_RULES_H

#include <dolfin/Quadrature.h>

namespace dolfin {
  
  /// Collection of gaussian-type quadrature rules on the real line,
  /// including Gauss, Radau, and Lobatto Quadrature

  class GaussianRules : public Quadrature {
  public:
    
    GaussianRules(int n);
    
  protected:

    void computeWeights();
    
  };
  
}

#endif
