// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_DGQ_METHOD_H
#define __NEW_DGQ_METHOD_H

#include <dolfin/NewMethod.h>

namespace dolfin
{

  /// Contains all numeric constants, such as nodal points and nodal weights,
  /// needed for the dG(q) method. The order q must be at least 0. Note that
  /// q refers to the polynomial order and not the order of convergence for
  /// the method, which is 2q + 1.

  class NewdGqMethod : public NewMethod
  {
  public:
    
    NewdGqMethod(unsigned int q);

    real ueval(real x0, real values[], real tau) const;
    real ueval(real x0, real values[], uint i) const;
    real residual(real x0, real values[], real f, real k) const;
    real timestep(real r, real tol, real kmax) const;

    void disp() const;
    
  protected:

    void computeQuadrature();
    void computeBasis();
    void computeWeights();

  };

}

#endif
