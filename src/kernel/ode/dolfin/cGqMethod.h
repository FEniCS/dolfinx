// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CGQ_METHOD_H
#define __CGQ_METHOD_H

#include <dolfin/Method.h>

namespace dolfin {

  /// Contains all numeric constants, such as nodal points and nodal weights,
  /// needed for the cG(q) method. The order q must be at least 1. Note that
  /// q refers to the polynomial order and not the order of convergence for
  /// the method, which is 2q.

  class cGqMethod : public Method {
  public:
    
    cGqMethod(unsigned int q);

    void show() const;
    
  protected:

    void computeQuadrature();
    void computeBasis();
    void computeWeights();

  };

}

#endif
