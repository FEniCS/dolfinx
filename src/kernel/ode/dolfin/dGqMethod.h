// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DGQ_METHOD_H
#define __DGQ_METHOD_H

#include <dolfin/Method.h>

namespace dolfin {

  /// Contains all numeric constants, such as nodal points and nodal weights,
  /// needed for the dG(q) method. The order q must be at least 0. Note that
  /// q refers to the polynomial order and not the order of convergence for
  /// the method, which is 2q + 1.

  class dGqMethod : public Method {
  public:
    
    dGqMethod(int q);

    void show() const;
    
  protected:

    void computeQuadrature();
    void computeBasis();
    void computeWeights();

  };

}

#endif
