// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_CGQ_METHOD_H
#define __NEW_CGQ_METHOD_H

#include <dolfin/NewMethod.h>

namespace dolfin
{

  /// Contains all numeric constants, such as nodal points and nodal weights,
  /// needed for the cG(q) method. The order q must be at least 1. Note that
  /// q refers to the polynomial order and not the order of convergence for
  /// the method, which is 2q.

  class NewcGqMethod : public NewMethod
  {
  public:
    
    NewcGqMethod(unsigned int q);

    /// Evaluate solution at given point
    real ueval(real x0, real values[], real tau) const;

    /// Evaluate solution at given node (inline optimized)
    inline real ueval(real x0, real values[], uint i) const
    { return ( i == 0 ? x0 : values[i - 1] ); }

    /// Compute residual at right end-point    
    real residual(real x0, real values[], real f, real k) const;

    /// Compute new time step based on the given residual
    real timestep(real r, real tol, real kmax) const;

    /// Compute error estimate (modulo stability factor)
    real error(real k, real r) const;

    /// Display method data
    void disp() const;
    
  protected:

    void computeQuadrature();
    void computeBasis();
    void computeWeights();

  };

}

#endif
